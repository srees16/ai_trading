"""
Latency Tracker — Microsecond-precision measurement for the hot path.

Tracks latency from market data arrival to order generation and alerts
when the 100-200ms SLA is breached.

Usage::

    from infrastructure.latency_tracker import latency_tracker

    with latency_tracker.measure("market_data_to_signal"):
        process_tick(tick)

    # Check SLA
    stats = latency_tracker.get_stats("market_data_to_signal")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Deque, Dict, Optional

logger = logging.getLogger(__name__)

# SLA threshold in milliseconds
_DEFAULT_SLA_MS = 200.0


@dataclass
class LatencyStats:
    label: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    sla_breaches: int = 0


class LatencyTracker:
    """Collects per-label latency measurements."""

    def __init__(self, *, sla_ms: float = _DEFAULT_SLA_MS, window: int = 10_000):
        self._sla_ms = sla_ms
        self._window = window
        self._samples: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._breaches: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, label: str):
        """Context manager to time a code block."""
        t0 = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._record(label, elapsed_ms)

    def record(self, label: str, elapsed_ms: float) -> None:
        """Manually record a measurement."""
        self._record(label, elapsed_ms)

    def _record(self, label: str, elapsed_ms: float) -> None:
        with self._lock:
            self._samples[label].append(elapsed_ms)
            if elapsed_ms > self._sla_ms:
                self._breaches[label] += 1
                logger.warning(
                    "SLA breach: %s took %.2fms (limit: %.0fms)",
                    label, elapsed_ms, self._sla_ms,
                )

    def get_stats(self, label: str) -> Optional[LatencyStats]:
        with self._lock:
            samples = list(self._samples.get(label, []))
            breaches = self._breaches.get(label, 0)

        if not samples:
            return None

        samples_sorted = sorted(samples)
        n = len(samples_sorted)
        return LatencyStats(
            label=label,
            count=n,
            total_ms=sum(samples_sorted),
            min_ms=samples_sorted[0],
            max_ms=samples_sorted[-1],
            p50_ms=samples_sorted[n // 2],
            p95_ms=samples_sorted[int(n * 0.95)],
            p99_ms=samples_sorted[int(n * 0.99)],
            sla_breaches=breaches,
        )

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        with self._lock:
            labels = list(self._samples.keys())
        return {label: self.get_stats(label) for label in labels if self.get_stats(label)}

    def reset(self, label: Optional[str] = None) -> None:
        with self._lock:
            if label:
                self._samples.pop(label, None)
                self._breaches.pop(label, None)
            else:
                self._samples.clear()
                self._breaches.clear()


latency_tracker = LatencyTracker()
