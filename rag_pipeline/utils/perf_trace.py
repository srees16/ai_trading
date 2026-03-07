"""
Performance Tracing for Centurion Capital LLC RAG Pipeline.

Provides lightweight latency instrumentation for every pipeline stage.
Each ``PipelineTrace`` instance captures wall-clock timings so that
bottlenecks are visible in logs and can be surfaced in the UI.

Usage:
    from rag_pipeline.utils.perf_trace import PipelineTrace

    trace = PipelineTrace()
    with trace.span("embedding"):
        embedding = model.encode(query)
    print(trace.summary())

Design goals:
    - Zero external dependencies (stdlib only).
    - Thread-safe for use with concurrent.futures.
    - Minimal overhead (< 0.05 ms per span).
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Span — a single timed operation
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """Represents a single timed operation within the pipeline."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock duration in milliseconds."""
        if self.end_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0

    @property
    def elapsed_s(self) -> float:
        """Wall-clock duration in seconds."""
        return self.elapsed_ms / 1000.0


# ---------------------------------------------------------------------------
# Pipeline trace — collects all spans for one query
# ---------------------------------------------------------------------------

@dataclass
class PipelineTrace:
    """
    Accumulates timing spans for one end-to-end RAG query.

    Thread-safe: spans can be added concurrently from a
    ``ThreadPoolExecutor``.
    """

    spans: List[Span] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _pipeline_start: float = field(default=0.0, repr=False)
    _pipeline_end: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------ #
    # Context-manager API
    # ------------------------------------------------------------------ #

    @contextmanager
    def span(
        self, name: str, **metadata: Any
    ) -> Generator[Span, None, None]:
        """
        Time a block of code and record it as a named span.

        Example::

            with trace.span("rerank", n_chunks=42):
                reranked = reranker.rerank(...)
        """
        s = Span(name=name, start_time=time.perf_counter(), metadata=metadata)
        try:
            yield s
        finally:
            s.end_time = time.perf_counter()
            with self._lock:
                self.spans.append(s)

    # ------------------------------------------------------------------ #
    # Pipeline-level start / stop
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Mark the pipeline start time."""
        self._pipeline_start = time.perf_counter()

    def stop(self) -> None:
        """Mark the pipeline end time."""
        self._pipeline_end = time.perf_counter()

    @property
    def total_ms(self) -> float:
        """Total pipeline wall-clock time in milliseconds."""
        if self._pipeline_end <= 0 or self._pipeline_start <= 0:
            return sum(s.elapsed_ms for s in self.spans)
        return (self._pipeline_end - self._pipeline_start) * 1000.0

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict of all span timings."""
        return {
            "total_ms": round(self.total_ms, 2),
            "spans": {
                s.name: {
                    "elapsed_ms": round(s.elapsed_ms, 2),
                    "metadata": s.metadata,
                }
                for s in self.spans
            },
        }

    def summary(self, log: bool = True) -> str:
        """
        Human-readable latency summary.

        Logs at INFO level by default.  Returns the string for embedding
        in the RAGResponse or UI.
        """
        lines: List[str] = ["--- Pipeline Latency ---"]
        for s in self.spans:
            meta_str = (
                f"  ({', '.join(f'{k}={v}' for k, v in s.metadata.items())})"
                if s.metadata
                else ""
            )
            lines.append(f"  {s.name:.<30s} {s.elapsed_ms:>8.1f} ms{meta_str}")
        lines.append(f"  {'TOTAL':.<30s} {self.total_ms:>8.1f} ms")
        text = "\n".join(lines)
        if log:
            logger.info(text)
        return text

    def get_span(self, name: str) -> Optional[Span]:
        """Look up a span by name (returns the first match)."""
        for s in self.spans:
            if s.name == name:
                return s
        return None
