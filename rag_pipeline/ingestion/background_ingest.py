"""
Background Ingestion Manager for Centurion Capital LLC RAG Pipeline.

Runs PDF ingestion in background threads so the Streamlit UI remains
responsive — users can submit queries for already-ingested documents
while new documents are being ingested concurrently.

Architecture
------------
- **Module-level singleton** (``BackgroundIngestionManager``) — persists
  across Streamlit reruns within the same server process.
- Each background task creates **isolated** RAGConfig / VectorStoreManager /
  EmbeddingService / PDFIngestionService instances so there are zero
  thread-safety concerns with the main-thread services.
- ChromaDB PersistentClient handles concurrent reads & writes to the
  same on-disk store transparently.
- Task status is tracked in a thread-safe dict and polled by the UI on
  every Streamlit rerun.

Usage (from ``ui_components.py``)::

    from rag_pipeline.ingestion.background_ingest import get_ingestion_manager

    mgr = get_ingestion_manager()
    mgr.submit(file_name, file_bytes)    # non-blocking
    tasks = mgr.get_active_tasks()       # poll on each rerun
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task status model
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IngestionTask:
    """Tracks the lifecycle of a single background ingestion job."""

    task_id: str
    file_name: str
    status: TaskStatus = TaskStatus.PENDING
    stage: str = ""
    stage_pct: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Background Ingestion Manager (singleton)
# ---------------------------------------------------------------------------

class BackgroundIngestionManager:
    """Manages concurrent PDF ingestion in background threads.

    Thread-safe.  A single instance is shared across all Streamlit
    sessions in the same server process.
    """

    _instance: Optional["BackgroundIngestionManager"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "BackgroundIngestionManager":
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        # Two workers: one can ingest while queries still flow on the
        # main thread; a second worker handles parallel multi-file uploads.
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="rag-ingest"
        )
        self._tasks: Dict[str, IngestionTask] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._initialized = True
        logger.info("BackgroundIngestionManager initialised (max_workers=2)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        file_name: str,
        file_bytes: bytes,
        *,
        on_complete_invalidate: bool = True,
    ) -> IngestionTask:
        """Submit a PDF for background ingestion.

        Returns immediately with an ``IngestionTask`` that can be polled
        for progress via ``get_task()`` or ``get_all_tasks()``.
        """
        task_id = f"{file_name}_{uuid.uuid4().hex[:8]}"
        task = IngestionTask(task_id=task_id, file_name=file_name)

        with self._lock:
            self._tasks[task_id] = task

        future = self._executor.submit(
            self._run_ingestion, task_id, file_name, file_bytes
        )
        with self._lock:
            self._futures[task_id] = future

        logger.info("Submitted background ingestion: %s (task=%s)", file_name, task_id)
        return task

    def get_task(self, task_id: str) -> Optional[IngestionTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[IngestionTask]:
        with self._lock:
            return list(self._tasks.values())

    def get_active_tasks(self) -> List[IngestionTask]:
        with self._lock:
            return [
                t
                for t in self._tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            ]

    def get_recently_completed(self, max_age_s: float = 300) -> List[IngestionTask]:
        """Return tasks that completed within *max_age_s* seconds."""
        cutoff = time.time() - max_age_s
        with self._lock:
            return [
                t
                for t in self._tasks.values()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                and t.completed_at is not None
                and t.completed_at >= cutoff
            ]

    def has_active_tasks(self) -> bool:
        with self._lock:
            return any(
                t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                for t in self._tasks.values()
            )

    def pop_completed(self) -> List[IngestionTask]:
        """Return and remove all completed / failed tasks.

        Call this from the main thread to consume results and clear
        old entries.
        """
        with self._lock:
            done_ids = [
                tid
                for tid, t in self._tasks.items()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]
            done = [self._tasks.pop(tid) for tid in done_ids]
            for tid in done_ids:
                self._futures.pop(tid, None)
            return done

    def clear_all(self) -> None:
        """Cancel pending futures and clear task list."""
        with self._lock:
            for fut in self._futures.values():
                fut.cancel()
            self._tasks.clear()
            self._futures.clear()

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    def _run_ingestion(
        self,
        task_id: str,
        file_name: str,
        file_bytes: bytes,
    ) -> Dict[str, Any]:
        """Execute ingestion in a background thread.

        Creates **isolated** service instances so there is no contention
        with the main-thread query engine.
        """
        with self._lock:
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.stage = "Initialising…"

        try:
            # ---- Create isolated services for this thread ----
            from rag_pipeline.config import RAGConfig
            from rag_pipeline.storage.vector_store import VectorStoreManager
            from rag_pipeline.storage.embeddings import EmbeddingService
            from rag_pipeline.ingestion.pdf_ingestion import PDFIngestionService

            config = RAGConfig()
            vs = VectorStoreManager(config)
            embedder = EmbeddingService(config)
            svc = PDFIngestionService(
                vs, config, embedder,
                on_change_callback=None,  # cache invalidation in main thread
            )

            # ---- Progress callback (thread-safe) ----
            def _progress(stage: str, pct: float) -> None:
                with self._lock:
                    task.stage = stage
                    task.stage_pct = pct

            # ---- Run ingestion ----
            result = svc.ingest_uploaded_bytes(
                file_name=file_name,
                file_bytes=file_bytes,
                progress_callback=_progress,
            )

            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = time.time()
                task.stage = "Complete"
                task.stage_pct = 1.0

            logger.info(
                "Background ingestion complete: %s %s", file_name, result.get("status")
            )
            return result

        except Exception as exc:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                task.completed_at = time.time()
                task.stage = "Failed"
            logger.error(
                "Background ingestion failed for %s: %s", file_name, exc, exc_info=True
            )
            raise


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

def get_ingestion_manager() -> BackgroundIngestionManager:
    """Return the process-wide ``BackgroundIngestionManager`` singleton."""
    return BackgroundIngestionManager()
