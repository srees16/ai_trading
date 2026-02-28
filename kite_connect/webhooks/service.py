"""
Webhook service orchestrator for Centurion Core — Indian Stocks.

This module wires together all webhook components after 2FA auth:

  1. KiteWebSocketService  — receives real-time ticks via WebSocket
  2. WebhookDispatcher     — fans out events to subscribers
  3. DBTickHandler         — writes ticks to PostgreSQL
  4. UITickCache           — thread-safe cache for Streamlit reads
  5. NSEMarketStatusMonitor — background market session tracker
  6. SessionWatchdog       — monitors connection health

Usage (from zerodha_live.py after Kite login):

    from kite_connect.webhooks.service import WebhookService
    svc = WebhookService.get_instance()
    svc.start(kite, stock_symbols)
    quotes = svc.get_quotes()         # replaces fetch_realtime_quotes()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from .dispatcher import WebhookDispatcher
from .events import EventType
from .handlers import (
    DBTickHandler,
    NSEMarketStatusMonitor,
    SessionWatchdog,
    UITickCache,
)
from .ticker import KiteWebSocketService
from .alert_engine import PriceAlertEngine
from .timescale_handler import TimescaleTickHandler

logger = logging.getLogger(__name__)


class WebhookService:
    """
    Top-level orchestrator — single entry point for the webhook system.

    Call ``start()`` once after 2FA authentication, then read quotes
    from ``get_quotes()`` instead of ``kite.quote()`` polling.
    """

    _instance: Optional["WebhookService"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def get_instance(cls) -> "WebhookService":
        """Return the singleton. Creates one if needed."""
        return cls()

    def __init__(self):
        if self._initialized:
            return

        # Components
        self._ws_service: Optional[KiteWebSocketService] = None
        self._dispatcher = WebhookDispatcher()
        self._db_handler = DBTickHandler(min_interval=1.0)
        self._ui_cache = UITickCache()
        self._nse_monitor = NSEMarketStatusMonitor(check_interval=60)
        self._session_watchdog = SessionWatchdog()
        self._alert_engine = PriceAlertEngine.get_instance()
        self._timescale_handler = TimescaleTickHandler(min_interval=0.5)

        # State
        self._started = False
        self._kite = None
        self._stock_symbols: List[str] = []

        self._initialized = True

    # ── Lifecycle ──────────────────────────────────────────────

    def start(
        self,
        kite,
        stock_symbols: List[str],
        tick_mode: str = "quote",
        enable_nse_monitor: bool = True,
        on_session_expired=None,
    ) -> None:
        """
        Boot the entire webhook pipeline.

        Parameters
        ----------
        kite : KiteConnect
            Authenticated Kite instance (after 2FA login).
        stock_symbols : list[str]
            Stock symbols to subscribe (e.g. ["RELIANCE", "TCS"]).
        tick_mode : str
            ``"full"`` (with depth), ``"quote"`` (OHLC), ``"ltp"`` (price only).
        enable_nse_monitor : bool
            Whether to start the NSE market status background monitor.
        on_session_expired : callable, optional
            Called when the session needs re-authentication.
        """
        if self._started:
            logger.info("WebhookService already started. Updating subscriptions.")
            self._update_subscriptions(stock_symbols)
            return

        self._kite = kite
        self._stock_symbols = list(stock_symbols)

        # 1. Register subscribers with the dispatcher
        self._dispatcher.subscribe(
            "db_tick_handler",
            [EventType.TICK_BATCH],
            self._db_handler,
            description="Writes real-time ticks to PostgreSQL stocks table",
        )
        self._dispatcher.subscribe(
            "ui_tick_cache",
            [EventType.TICK_BATCH],
            self._ui_cache,
            description="Thread-safe tick cache for Streamlit UI reads",
        )

        if on_session_expired:
            self._session_watchdog = SessionWatchdog(on_session_expired)
        self._dispatcher.subscribe(
            "session_watchdog",
            [EventType.SESSION_EXPIRED, EventType.WS_DISCONNECTED],
            self._session_watchdog,
            description="Monitors session health and triggers re-auth",
        )

        # -- New: Alert engine (evaluates price conditions on every tick)
        self._dispatcher.subscribe(
            "alert_engine",
            [EventType.TICK_BATCH],
            self._alert_engine,
            description="Price alert engine — evaluates user-defined conditions",
        )

        # -- New: TimescaleDB tick persistence
        self._timescale_handler.ensure_schema()
        self._dispatcher.subscribe(
            "timescale_ticks",
            [EventType.TICK_BATCH],
            self._timescale_handler,
            description="Writes raw ticks to TimescaleDB hypertable for OHLC aggregates",
        )

        # 2. Start the Kite WebSocket ticker
        self._ws_service = KiteWebSocketService(kite, mode=tick_mode)
        self._ws_service.subscribe_symbols(stock_symbols)
        self._ws_service.start()

        # 3. Start NSE market status monitor
        if enable_nse_monitor:
            self._nse_monitor.start()

        self._started = True
        logger.info(
            "WebhookService started: %d symbols, mode=%s, nse_monitor=%s",
            len(stock_symbols), tick_mode, enable_nse_monitor,
        )

    def stop(self) -> None:
        """Shut down all components."""
        if self._ws_service:
            self._ws_service.stop()
        self._nse_monitor.stop()
        self._dispatcher.reset()
        self._started = False
        logger.info("WebhookService stopped")

    # ── Public API (replaces polling functions) ────────────────

    def get_quotes(self) -> Dict[str, Dict]:
        """
        Return latest quotes — **zero API calls**.

        Drop-in replacement for ``fetch_realtime_quotes()``.
        Returns ``{symbol: {high, low, volume, ltp, change_pct}}``.
        """
        return self._ui_cache.get_all()

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for a single symbol."""
        return self._ui_cache.get(symbol)

    def get_market_status(self):
        """Return (pill_class, label) for the NSE market status."""
        return self._nse_monitor.pill_class, self._nse_monitor.status

    def get_status(self) -> Dict:
        """Full system status for the UI status panel."""
        ws_status = self._ws_service.get_status() if self._ws_service else {}
        return {
            "started": self._started,
            "websocket": ws_status,
            "ui_cache_count": self._ui_cache.count,
            "ui_cache_updates": self._ui_cache.update_count,
            "ui_cache_last_update": self._ui_cache.last_update,
            "nse_status": self._nse_monitor.status,
            "dispatcher_stats": self._dispatcher.stats,
            "subscribers": self._dispatcher.list_subscribers(),
            "alert_engine": {
                "active_alerts": self._alert_engine.active_count,
                "total_alerts": self._alert_engine.total_count,
                **self._alert_engine.stats,
            },
            "timescale": {
                "schema_ready": self._timescale_handler._schema_ready,
                **self._timescale_handler.stats,
            },
        }

    @property
    def is_streaming(self) -> bool:
        """Whether the WebSocket is actively receiving data."""
        return bool(self._ws_service and self._ws_service.is_connected)

    @property
    def market_is_open(self) -> bool:
        """True when the NSE Capital Market session is open/live."""
        status = self._nse_monitor.status
        return status in ("Live", "Pre-Open")

    @property
    def quotes_count(self) -> int:
        return self._ui_cache.count

    @property
    def last_update_time(self) -> float:
        return self._ui_cache.last_update

    # ── Internal ───────────────────────────────────────────────

    def _update_subscriptions(self, new_symbols: List[str]) -> None:
        """Add/remove symbols from the live WebSocket subscription."""
        if not self._ws_service:
            return

        current = set(self._stock_symbols)
        new = set(new_symbols)

        to_add = new - current
        to_remove = current - new

        if to_add:
            self._ws_service.subscribe_symbols(list(to_add))
        if to_remove:
            self._ws_service.unsubscribe_symbols(list(to_remove))

        self._stock_symbols = list(new)

    def refresh_session(self, kite) -> None:
        """Re-bind to a fresh Kite session after 2FA re-auth."""
        self._kite = kite
        if self._ws_service:
            self._ws_service.update_session(kite)
        logger.info("WebhookService session refreshed")

    def add_event_listener(
        self,
        listener_id: str,
        events: List[EventType],
        callback,
        description: str = "",
    ) -> None:
        """Register an additional event listener (e.g. strategy signals)."""
        self._dispatcher.subscribe(listener_id, events, callback, description)

    def remove_event_listener(self, listener_id: str) -> None:
        self._dispatcher.unsubscribe(listener_id)
