"""
Webhook subscriber handlers for Centurion Core.

These are the callback functions that get registered with the
WebhookDispatcher to react to real-time events pushed by the
Kite WebSocket ticker.

Each handler is a standalone function that can be individually
subscribed/unsubscribed without affecting others.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Database Updater — writes ticks to `stocks` table
# ═══════════════════════════════════════════════════════════════

class DBTickHandler:
    """
    Receives TICK_BATCH events and bulk-updates the PostgreSQL stocks
    table — replacing the old ``update_stocks_in_db()`` polling call.

    Uses a single connection with autocommit for throughput. Throttles
    writes to at most once every ``min_interval`` seconds.
    """

    def __init__(self, min_interval: float = 1.0):
        self._min_interval = min_interval
        self._last_write: float = 0.0
        self._lock = threading.Lock()
        self._pending: Dict[str, Dict] = {} # symbol latest tick data

    def __call__(self, event) -> None:
        """WebhookDispatcher callback entry point."""
        from .events import EventType
        if event.event_type != EventType.TICK_BATCH:
            return

        ticks = event.payload.get("ticks", [])
        if not ticks:
            return

        # Merge into pending buffer (latest tick wins)
        with self._lock:
            for t in ticks:
                self._pending[t["name"]] = t

        # Throttle writes
        now = time.time()
        if now - self._last_write < self._min_interval:
            return

        self._flush_to_db()

    def _flush_to_db(self) -> None:
        """Write all pending tick data to the database."""
        with self._lock:
            if not self._pending:
                return
            batch = dict(self._pending)
            self._pending.clear()

        self._last_write = time.time()

        try:
            import os, sys
            _kite_root = os.path.dirname(os.path.dirname(__file__))
            if _kite_root not in sys.path:
                sys.path.append(_kite_root)
            from core.db_service import get_connection

            conn = get_connection()
            conn.autocommit = True
            cur = conn.cursor()

            for symbol, data in batch.items():
                cur.execute(
                    """
                    INSERT INTO stocks (name, high, low, volume, ltp, change, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (name) DO UPDATE
                    SET high = EXCLUDED.high,
                        low  = EXCLUDED.low,
                        volume = EXCLUDED.volume,
                        ltp  = EXCLUDED.ltp,
                        change = EXCLUDED.change,
                        updated_at = NOW();
                    """,
                    (
                        symbol,
                        data.get("high"),
                        data.get("low"),
                        data.get("volume"),
                        data.get("ltp"),
                        data.get("change_pct"),
                    ),
                )

            cur.close()
            conn.close()
            logger.debug("DB tick update: %d symbols written", len(batch))

        except Exception:
            logger.exception("DB tick handler write failed")


# ═══════════════════════════════════════════════════════════════
# 2. NSE Market Status Monitor — push-based session tracking
# ═══════════════════════════════════════════════════════════════

class NSEMarketStatusMonitor:
    """
    Monitors NSE market status and dispatches MARKET_* events.

    Instead of checking NSE status on every Streamlit page render,
    this runs a background poller (every 60 s) and only fires an
    event when the status *changes*.

    This is the one place where a small amount of polling is
    acceptable — NSE doesn't offer webhooks, and a 60-second
    check for market open/close is minimal.
    """

    def __init__(self, check_interval: int = 60):
        self._check_interval = check_interval
        self._current_status: Optional[str] = None
        self._current_pill: str = "pill-closed"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background status monitor."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="nse-status-monitor",
        )
        self._thread.start()
        logger.info("NSE market status monitor started (interval=%ds)", self._check_interval)

    def stop(self) -> None:
        self._running = False

    @property
    def status(self) -> str:
        """Current market status label (e.g. 'Live', 'Closed', 'Pre-Open')."""
        with self._lock:
            return self._current_status or "Unknown"

    @property
    def pill_class(self) -> str:
        """CSS pill class for Streamlit UI."""
        with self._lock:
            return self._current_pill

    def _monitor_loop(self) -> None:
        """Background thread: check NSE status periodically."""
        from .dispatcher import WebhookDispatcher
        from .events import EventType, WebhookEvent

        dispatcher = WebhookDispatcher()

        while self._running:
            try:
                pill, label = self._fetch_nse_status()
                with self._lock:
                    old_status = self._current_status
                    self._current_status = label
                    self._current_pill = pill

                # Only dispatch if status changed
                if old_status is not None and label != old_status:
                    event_map = {
                        "Live": EventType.MARKET_OPEN,
                        "Closed": EventType.MARKET_CLOSE,
                        "Pre-Open": EventType.MARKET_PRE_OPEN,
                    }
                    evt_type = event_map.get(label, EventType.MARKET_STATUS_CHANGE)
                    dispatcher.dispatch(WebhookEvent(
                        event_type=evt_type,
                        payload={
                            "status": label,
                            "previous": old_status,
                            "pill_class": pill,
                        },
                        source="nse_api",
                    ))
                    logger.info("NSE market status changed: %s %s", old_status, label)

            except Exception:
                logger.debug("NSE status check failed", exc_info=True)

            time.sleep(self._check_interval)

    @staticmethod
    def _fetch_nse_status():
        """Fetch Capital Market status from NSE API. Returns (pill_class, label)."""
        import requests as req

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        sess = req.Session()
        sess.get("https://www.nseindia.com", headers=headers, timeout=4)
        resp = sess.get(
            "https://www.nseindia.com/api/marketStatus",
            headers=headers, timeout=4,
        )
        resp.raise_for_status()
        for mkt in resp.json().get("marketState", []):
            if mkt.get("market") == "Capital Market":
                status = (mkt.get("marketStatus") or "").lower()
                if status in ("open", "live"):
                    return "pill-open", "Live"
                elif "pre" in status:
                    return "pill-pre", "Pre-Open"
                elif "close" in status:
                    return "pill-closed", "Closed"
                else:
                    return "pill-pre", status.title()

        return "pill-closed", "Closed"


# ═══════════════════════════════════════════════════════════════
# 3. UI Tick Cache — thread-safe cache for Streamlit reads
# ═══════════════════════════════════════════════════════════════

class UITickCache:
    """
    A lightweight cache that the Streamlit UI reads from instead
    of calling kite.quote().

    Updated by TICK_BATCH events from the dispatcher.
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._last_update: float = 0.0
        self._update_count: int = 0

    def __call__(self, event) -> None:
        """WebhookDispatcher callback entry point."""
        from .events import EventType
        if event.event_type != EventType.TICK_BATCH:
            return

        ticks = event.payload.get("ticks", [])
        with self._lock:
            for t in ticks:
                self._cache[t["name"]] = t
            self._last_update = time.time()
            self._update_count += 1

    def get_all(self) -> Dict[str, Dict]:
        """Return all cached quotes (thread-safe copy)."""
        with self._lock:
            return dict(self._cache)

    def get(self, symbol: str) -> Optional[Dict]:
        with self._lock:
            return self._cache.get(symbol)

    @property
    def last_update(self) -> float:
        return self._last_update

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def update_count(self) -> int:
        return self._update_count


# ═══════════════════════════════════════════════════════════════
# 4. Connection / Session Watchdog
# ═══════════════════════════════════════════════════════════════

class SessionWatchdog:
    """
    Listens for SESSION_EXPIRED / WS_DISCONNECTED events and
    triggers re-authentication (2FA flow) when needed.
    """

    def __init__(self, on_session_expired=None):
        self._on_session_expired = on_session_expired
        self._last_alert: float = 0.0

    def __call__(self, event) -> None:
        from .events import EventType

        if event.event_type == EventType.SESSION_EXPIRED:
            logger.warning("Session expired event received")
            if self._on_session_expired and (time.time() - self._last_alert > 60):
                self._last_alert = time.time()
                self._on_session_expired(event)

        elif event.event_type == EventType.WS_DISCONNECTED:
            reason = event.payload.get("reason", "")
            if reason == "reconnect_exhausted":
                logger.error("WebSocket reconnection exhausted — session may need refresh")
                if self._on_session_expired:
                    self._on_session_expired(event)
