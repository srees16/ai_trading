"""
Fault Isolation — Process / thread boundary management.

Each subsystem runs in its own supervised context so that a crash in
one component (e.g. a scraper timeout) doesn't cascade to the order
execution engine.

Provides:
  - SupervisedWorker: wraps a callable in a fault-isolated thread
  - ProcessSupervisor: manages child processes (for heavy compute)
  - CircuitBreaker: stops calling a failing dependency after N failures

Usage::

    from infrastructure.fault_isolation import SupervisedWorker, CircuitBreaker

    worker = SupervisedWorker("scraper", target=scrape_news, restart_on_fail=True)
    worker.start()

    cb = CircuitBreaker("yfinance", failure_threshold=5, reset_timeout=60)
    with cb:
        data = yf.download(ticker)
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Supervised Worker ────────────────────────────────────────────────

class SupervisedWorker:
    """
    Runs a target function in a daemon thread with crash recovery.

    If the target raises, the worker logs the error and optionally
    restarts after a cooldown.
    """

    def __init__(
        self,
        name: str,
        target: Callable,
        *,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        restart_on_fail: bool = True,
        max_restarts: int = 5,
        cooldown_seconds: float = 5.0,
    ):
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.restart_on_fail = restart_on_fail
        self.max_restarts = max_restarts
        self.cooldown_seconds = cooldown_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._restart_count = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._restart_count = 0
        self._spawn()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _spawn(self) -> None:
        self._thread = threading.Thread(
            target=self._run_loop, name=f"worker-{self.name}", daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.target(*self.args, **self.kwargs)
                break  # Clean exit
            except Exception:
                self._restart_count += 1
                logger.exception(
                    "Worker '%s' crashed (restart %d/%d)",
                    self.name, self._restart_count, self.max_restarts,
                )
                if not self.restart_on_fail or self._restart_count >= self.max_restarts:
                    logger.error("Worker '%s' exceeded max restarts — stopping", self.name)
                    break
                time.sleep(self.cooldown_seconds)


# ── Circuit Breaker ──────────────────────────────────────────────────

class CircuitBreaker:
    """
    Prevents cascading failures by short-circuiting calls to a
    dependency that is consistently failing.

    States:
      - CLOSED: normal operation
      - OPEN: calls are rejected immediately
      - HALF_OPEN: one test call allowed to check recovery
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._last_failure_time: float = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "OPEN":
                if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                    self._state = "HALF_OPEN"
            return self._state

    @contextmanager
    def __call__(self):
        """Use as context manager: ``with cb(): ...``"""
        current = self.state
        if current == "OPEN":
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN — call rejected")

        try:
            yield
            self._on_success()
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"

    def _on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= self.failure_threshold:
                self._state = "OPEN"
                logger.warning(
                    "Circuit '%s' tripped OPEN after %d failures",
                    self.name, self._failures,
                )

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected by an open circuit breaker."""
    pass
