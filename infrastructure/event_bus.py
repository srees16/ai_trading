"""
Event Bus — In-process pub/sub with deterministic replay support.

Every component in Centurion reacts to events (ticks, fills, signals)
rather than polling.  The EventBus provides:

* **Publish / Subscribe** with topic-based routing
* **Async & sync** listener support
* **Event log** for deterministic replay (backtest uses historical replay)
* **Correlation IDs** for distributed tracing
* **Priority ordering** — higher-priority listeners fire first

Usage::

    from infrastructure.event_bus import event_bus, Event

    # Subscribe
    @event_bus.on("market_data.tick")
    def on_tick(event: Event):
        print(event.payload)

    # Publish
    event_bus.emit("market_data.tick", {"symbol": "RELIANCE.NS", "ltp": 2450.0})

    # Replay from log
    event_bus.replay(from_ts=start, to_ts=end)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ── Event model ──────────────────────────────────────────────────────

class EventPriority(IntEnum):
    """Listener priority — lower number = fires first."""
    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 90
    MONITOR = 100  # logging / audit listeners


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event envelope."""
    topic: str
    payload: Any
    timestamp: float = field(default_factory=time.monotonic)
    wall_clock: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "wall_clock": self.wall_clock.isoformat(),
            "correlation_id": self.correlation_id,
            "source": self.source,
        }


@dataclass
class _Subscription:
    callback: Callable
    priority: EventPriority = EventPriority.NORMAL
    is_async: bool = False


# ── Event Bus ────────────────────────────────────────────────────────

class EventBus:
    """Thread-safe in-process event bus with deterministic replay."""

    def __init__(self, *, enable_log: bool = True, max_log_size: int = 100_000):
        self._subs: Dict[str, List[_Subscription]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_log: List[Event] = []
        self._log_enabled = enable_log
        self._max_log_size = max_log_size

    # ── Subscribe ────────────────────────────────────────────────

    def on(
        self,
        topic: str,
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Callable:
        """Decorator to register a sync or async listener."""
        def decorator(fn: Callable) -> Callable:
            is_async = asyncio.iscoroutinefunction(fn)
            sub = _Subscription(callback=fn, priority=priority, is_async=is_async)
            with self._lock:
                self._subs[topic].append(sub)
                self._subs[topic].sort(key=lambda s: s.priority)
            return fn
        return decorator

    def subscribe(
        self,
        topic: str,
        callback: Callable,
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """Imperative subscription (non-decorator)."""
        is_async = asyncio.iscoroutinefunction(callback)
        sub = _Subscription(callback=callback, priority=priority, is_async=is_async)
        with self._lock:
            self._subs[topic].append(sub)
            self._subs[topic].sort(key=lambda s: s.priority)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        with self._lock:
            self._subs[topic] = [
                s for s in self._subs[topic] if s.callback is not callback
            ]

    # ── Publish ──────────────────────────────────────────────────

    def emit(
        self,
        topic: str,
        payload: Any = None,
        *,
        source: str = "",
        correlation_id: Optional[str] = None,
    ) -> Event:
        """
        Emit an event synchronously.  All listeners for *topic* are
        invoked in priority order on the calling thread.
        """
        evt = Event(
            topic=topic,
            payload=payload,
            source=source,
            correlation_id=correlation_id or uuid.uuid4().hex[:12],
        )
        self._append_log(evt)
        self._dispatch(evt)
        return evt

    async def emit_async(
        self,
        topic: str,
        payload: Any = None,
        *,
        source: str = "",
        correlation_id: Optional[str] = None,
    ) -> Event:
        """Emit and await async listeners."""
        evt = Event(
            topic=topic,
            payload=payload,
            source=source,
            correlation_id=correlation_id or uuid.uuid4().hex[:12],
        )
        self._append_log(evt)
        await self._dispatch_async(evt)
        return evt

    # ── Dispatch ─────────────────────────────────────────────────

    def _dispatch(self, evt: Event) -> None:
        with self._lock:
            subs = list(self._subs.get(evt.topic, []))
            # Also dispatch to wildcard listeners
            subs.extend(self._subs.get("*", []))
        subs.sort(key=lambda s: s.priority)
        for sub in subs:
            try:
                if sub.is_async:
                    # Schedule on running loop or skip
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(sub.callback(evt))
                    except RuntimeError:
                        pass  # No running loop — skip async listener
                else:
                    sub.callback(evt)
            except Exception:
                logger.exception("Listener %s failed for event %s", sub.callback, evt.topic)

    async def _dispatch_async(self, evt: Event) -> None:
        with self._lock:
            subs = list(self._subs.get(evt.topic, []))
            subs.extend(self._subs.get("*", []))
        subs.sort(key=lambda s: s.priority)
        for sub in subs:
            try:
                if sub.is_async:
                    await sub.callback(evt)
                else:
                    sub.callback(evt)
            except Exception:
                logger.exception("Listener %s failed for event %s", sub.callback, evt.topic)

    # ── Event log & replay ───────────────────────────────────────

    def _append_log(self, evt: Event) -> None:
        if not self._log_enabled:
            return
        self._event_log.append(evt)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]

    def get_log(
        self,
        *,
        topic: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Event]:
        """Return filtered event log entries."""
        events = self._event_log
        if topic:
            events = [e for e in events if e.topic == topic]
        if since is not None:
            events = [e for e in events if e.timestamp >= since]
        return list(events)

    def replay(
        self,
        events: Optional[Sequence[Event]] = None,
        *,
        topic: Optional[str] = None,
    ) -> int:
        """
        Replay events through the bus (deterministic replay).

        If *events* is ``None``, replays from the internal log.
        Returns number of events replayed.
        """
        source = events if events is not None else self.get_log(topic=topic)
        count = 0
        for evt in source:
            self._dispatch(evt)
            count += 1
        return count

    def clear_log(self) -> None:
        self._event_log.clear()

    # ── Introspection ────────────────────────────────────────────

    @property
    def topics(self) -> List[str]:
        with self._lock:
            return list(self._subs.keys())

    def listener_count(self, topic: str) -> int:
        with self._lock:
            return len(self._subs.get(topic, []))


# ── Module-level singleton ───────────────────────────────────────────
event_bus = EventBus()
