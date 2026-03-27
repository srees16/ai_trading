"""
Monitoring & Ops Layer — Health checks, latency dashboards, audit trail.

Subscribes to all event bus topics for comprehensive observability:
  - Latency tracking per stage
  - Error rate monitoring
  - Event throughput
  - Health status for each subsystem

Emits:
  - ``monitoring.health_check``
  - ``monitoring.alert``
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Centralized monitoring that listens to event bus topics.
    """

    def __init__(self):
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._last_events: Dict[str, dict] = {}
        self._start_time = time.monotonic()
        self._registered = False

    def register_listeners(self) -> None:
        """Subscribe to key event bus topics for monitoring."""
        if self._registered:
            return
        from infrastructure.event_bus import event_bus, EventPriority

        @event_bus.on("*", priority=EventPriority.MONITOR)
        def _on_any_event(event):
            self._event_counts[event.topic] += 1
            self._last_events[event.topic] = {
                "timestamp": event.wall_clock.isoformat(),
                "source": event.source,
            }

        # Track failures
        for topic in ("risk.check_failed", "execution.order_rejected"):
            @event_bus.on(topic, priority=EventPriority.MONITOR)
            def _on_error(event, _topic=topic):
                self._error_counts[_topic] += 1

        self._registered = True
        logger.info("Monitoring listeners registered")

    def get_health(self) -> dict:
        """Return overall system health status."""
        from infrastructure.latency_tracker import latency_tracker

        uptime = time.monotonic() - self._start_time
        total_events = sum(self._event_counts.values())
        total_errors = sum(self._error_counts.values())

        latency_stats = latency_tracker.get_all_stats()
        latency_summary = {}
        for label, stats in latency_stats.items():
            latency_summary[label] = {
                "p50_ms": stats.p50_ms,
                "p95_ms": stats.p95_ms,
                "p99_ms": stats.p99_ms,
                "sla_breaches": stats.sla_breaches,
            }

        return {
            "status": "healthy" if total_errors == 0 else "degraded",
            "uptime_seconds": round(uptime, 1),
            "total_events": total_events,
            "total_errors": total_errors,
            "event_counts": dict(self._event_counts),
            "error_counts": dict(self._error_counts),
            "latency": latency_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_event_throughput(self) -> dict:
        """Events per second by topic."""
        uptime = max(time.monotonic() - self._start_time, 1)
        return {
            topic: round(count / uptime, 2)
            for topic, count in self._event_counts.items()
        }


# Singleton
monitoring_service = MonitoringService()
