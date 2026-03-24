"""
Internal webhook dispatcher for Centurion Core.

Routes real-time market events from the Kite WebSocket ticker to
registered subscribers (DB updater, Streamlit UI cache, alert manager, etc.)
without any HTTP overhead — all callbacks are in-process.

Thread-safe: the WebSocket runs on a background thread, so all subscriber
callbacks are invoked from that thread via a ThreadPoolExecutor.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Set

from .events import EventType, WebhookEvent, WebhookSubscription

logger = logging.getLogger(__name__)

# Type alias for subscriber callbacks
EventCallback = Callable[[WebhookEvent], None]


class WebhookDispatcher:
    """
    In-process event dispatcher.

    Replaces the polling loop — instead of every Streamlit fragment
    calling ``kite.quote()`` on a timer, the WebSocket pushes ticks
    and the dispatcher fans them out to all registered handlers.

    Usage
    -----
    >>> dispatcher = WebhookDispatcher()
    >>> dispatcher.subscribe("db_updater", [EventType.TICK_BATCH], my_db_callback)
    >>> dispatcher.dispatch(WebhookEvent(event_type=EventType.TICK_BATCH, payload={...}))
    """

    _instance: Optional["WebhookDispatcher"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton — one dispatcher per process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_workers: int = 4):
        if self._initialized:
            return
        self._subscriptions: Dict[str, WebhookSubscription] = {}
        self._callbacks: Dict[str, EventCallback] = {}
        self._event_index: Dict[EventType, Set[str]] = defaultdict(set)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="webhook"
        )
        self._processed: Set[str] = set()
        self._processed_max = 10_000          # ring-buffer dedup
        self._stats = {
            "dispatched": 0,
            "delivered": 0,
            "errors": 0,
            "duplicates": 0,
        }
        self._initialized = True
        logger.info("WebhookDispatcher initialised (workers=%d)", max_workers)

    # ── Subscription management ────────────────────────────────

    def subscribe(
        self,
        subscriber_id: str,
        events: List[EventType],
        callback: EventCallback,
        description: str = "",
    ) -> None:
        """Register *callback* to receive events of the given types."""
        sub = WebhookSubscription(
            subscriber_id=subscriber_id,
            events=events,
            description=description,
        )
        self._subscriptions[subscriber_id] = sub
        self._callbacks[subscriber_id] = callback
        for evt in events:
            self._event_index[evt].add(subscriber_id)
        logger.info("Subscribed: %s %s", subscriber_id, [e.value for e in events])

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber."""
        sub = self._subscriptions.pop(subscriber_id, None)
        self._callbacks.pop(subscriber_id, None)
        if sub:
            for evt in sub.events:
                self._event_index[evt].discard(subscriber_id)
        logger.info("Unsubscribed: %s", subscriber_id)

    # ── Event dispatch ─────────────────────────────────────────

    def dispatch(self, event: WebhookEvent) -> None:
        """Fan-out *event* to all matching subscribers (non-blocking)."""
        # Deduplication
        if event.event_id in self._processed:
            self._stats["duplicates"] += 1
            return
        self._processed.add(event.event_id)
        if len(self._processed) > self._processed_max:
            # Trim oldest half (set ordering is insertion-order in CPython 3.7+)
            trim = len(self._processed) // 2
            self._processed = set(list(self._processed)[trim:])

        self._stats["dispatched"] += 1

        subscriber_ids = self._event_index.get(event.event_type, set())
        for sid in subscriber_ids:
            sub = self._subscriptions.get(sid)
            cb = self._callbacks.get(sid)
            if sub and sub.is_active and cb:
                self._executor.submit(self._deliver, sid, cb, event)

    def _deliver(
        self, subscriber_id: str, callback: EventCallback, event: WebhookEvent
    ) -> None:
        """Invoke a single subscriber callback with error handling."""
        try:
            callback(event)
            self._stats["delivered"] += 1
        except Exception:
            self._stats["errors"] += 1
            logger.exception(
                "Webhook delivery failed: subscriber=%s, event=%s",
                subscriber_id,
                event.event_type.value,
            )

    # ── Introspection ──────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)

    def list_subscribers(self) -> List[Dict]:
        return [
            {
                "id": sid,
                "events": [e.value for e in sub.events],
                "active": sub.is_active,
                "description": sub.description,
            }
            for sid, sub in self._subscriptions.items()
        ]

    def reset(self) -> None:
        """Tear down all subscriptions (used in tests / shutdown)."""
        self._subscriptions.clear()
        self._callbacks.clear()
        self._event_index.clear()
        self._processed.clear()
        for k in self._stats:
            self._stats[k] = 0
        logger.info("WebhookDispatcher reset")
