"""
Deterministic Replay Engine.

Records every decision-relevant event to a log and can replay them
in exact order to reproduce any historical trading session.

Integrates with the EventBus:
  - In live/paper mode: records events to a persistent JSONL log
  - In backtest mode: reads the log and replays through the bus

Usage::

    from infrastructure.replay_engine import replay_engine

    # Start recording
    replay_engine.start_recording("session_2025_03_16")

    # ... trading happens, events flow through event_bus ...

    # Stop and get log path
    path = replay_engine.stop_recording()

    # Later: deterministic replay
    count = replay_engine.replay_session("session_2025_03_16")
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path("data/event_logs")


class ReplayEngine:
    """Records and replays event streams for deterministic reproduction."""

    def __init__(self, log_dir: Optional[Path] = None):
        self._log_dir = log_dir or _DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._recording = False
        self._session_name: Optional[str] = None
        self._log_file = None
        self._lock = threading.Lock()
        self._event_count = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start_recording(self, session_name: str) -> Path:
        """Begin recording events to a JSONL file."""
        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording")
            self._session_name = session_name
            path = self._log_dir / f"{session_name}.jsonl"
            self._log_file = open(path, "a", encoding="utf-8")
            self._recording = True
            self._event_count = 0
            logger.info("Replay recording started: %s", path)
            return path

    def record_event(self, event_dict: dict) -> None:
        """Append a single event to the active log."""
        if not self._recording:
            return
        with self._lock:
            if self._log_file:
                self._log_file.write(json.dumps(event_dict, default=str) + "\n")
                self._event_count += 1

    def stop_recording(self) -> Optional[str]:
        """Stop recording and close the log file."""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            if self._log_file:
                self._log_file.close()
                path = self._log_file.name
                self._log_file = None
                logger.info("Replay recording stopped: %d events → %s", self._event_count, path)
                return path
            return None

    def replay_session(
        self,
        session_name: str,
        *,
        topic_filter: Optional[str] = None,
    ) -> int:
        """
        Replay a recorded session through the event bus.

        Returns the number of events replayed.
        """
        from infrastructure.event_bus import event_bus, Event

        path = self._log_dir / f"{session_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Session log not found: {path}")

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                topic = data.get("topic", "")
                if topic_filter and topic != topic_filter:
                    continue

                event_bus.emit(
                    topic=topic,
                    payload=data.get("payload"),
                    source=data.get("source", "replay"),
                    correlation_id=data.get("correlation_id"),
                )
                count += 1

        logger.info("Replayed %d events from session '%s'", count, session_name)
        return count

    def list_sessions(self) -> List[dict]:
        """List available recorded sessions."""
        sessions = []
        for f in sorted(self._log_dir.glob("*.jsonl")):
            stat = f.stat()
            sessions.append({
                "name": f.stem,
                "path": str(f),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })
        return sessions


replay_engine = ReplayEngine()
