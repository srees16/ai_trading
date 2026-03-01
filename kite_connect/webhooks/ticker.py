"""
Kite Connect WebSocket ticker — push-based real-time price streaming.

Replaces the polling approach (``kite.quote()`` every N seconds) with
a persistent WebSocket connection that receives ticks from Zerodha's
servers *only when prices change*.

Architecture
────────────
  KiteTicker (WebSocket)
       │  on_ticks(ticks)
       ▼
  KiteWebSocketService
       │  _on_ticks()          ← parses raw ticks → TickData
       │  _batch_and_dispatch() ← batches & sends via dispatcher
       ▼
  WebhookDispatcher
       │  dispatch(TICK_BATCH)
       ▼
  ┌─────────────┬───────────────┬──────────────┐
  │ DB Updater  │ UI Cache      │ Alert System │
  └─────────────┴───────────────┴──────────────┘

Benefits over polling:
  • Zero API calls for price data (no kite.quote() needed)
  • Sub-second latency (prices arrive as they change)
  • No unnecessary calls when market is idle
  • Connection-based rate limit (1 WebSocket) vs per-call rate limit
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

# Append kite_connect to path
_kite_root = os.path.dirname(os.path.dirname(__file__))
if _kite_root not in sys.path:
    sys.path.append(_kite_root)

from .dispatcher import WebhookDispatcher
from .events import EventType, TickData, WebhookEvent

logger = logging.getLogger(__name__)


class KiteWebSocketService:
    """
    Manages the Kite Connect WebSocket connection for real-time tick data.

    After 2FA authentication creates a valid ``KiteConnect`` session, this
    service opens a persistent WebSocket to receive live price updates for
    all subscribed instruments (ind stocks).

    Usage
    -----
    >>> from kite_connect.auth.kite_session import create_kite_session
    >>> kite = create_kite_session()
    >>> ws_service = KiteWebSocketService(kite)
    >>> ws_service.subscribe_symbols(["RELIANCE", "TCS", "INFY"])
    >>> ws_service.start()  # non-blocking — runs on background thread
    """

    _instance: Optional["KiteWebSocketService"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton — one WebSocket connection per process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, kite=None, mode: str = "full"):
        """
        Parameters
        ----------
        kite : KiteConnect
            Authenticated Kite instance (after 2FA login).
        mode : str
            Tick mode — ``"full"`` (OHLC + depth), ``"quote"`` (OHLC),
            or ``"ltp"`` (last price only).
        """
        if self._initialized:
            # Allow re-binding to a fresh kite session after token refresh
            if kite is not None:
                self._kite = kite
                self._api_key = kite.api_key
                self._access_token = kite.access_token
            return

        self._kite = kite
        self._api_key = kite.api_key if kite else ""
        self._access_token = kite.access_token if kite else ""
        self._mode = mode

        # Instrument tracking
        self._instrument_tokens: Set[int] = set()
        self._symbol_to_token: Dict[str, int] = {}
        self._token_to_symbol: Dict[int, str] = {}

        # Latest tick cache (symbol → TickData)
        self._tick_cache: Dict[str, TickData] = {}
        self._tick_cache_lock = threading.Lock()

        # WebSocket state
        self._ticker = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._last_tick_time: float = 0.0

        # Dispatcher
        self._dispatcher = WebhookDispatcher()

        # Batching config — collect ticks for N ms then dispatch
        self._batch_interval = 0.5            # seconds
        self._batch_buffer: List[TickData] = []
        self._batch_lock = threading.Lock()
        self._batch_thread: Optional[threading.Thread] = None

        self._initialized = True
        logger.info("KiteWebSocketService initialised (mode=%s)", mode)

    # ── Instrument subscription ────────────────────────────────

    def subscribe_symbols(
        self,
        symbols: List[str],
        exchange: str = "NSE",
    ) -> None:
        """
        Resolve trading symbols to instrument tokens and subscribe.

        Calls ``kite.instruments(exchange)`` once to build a lookup,
        then subscribes the matching tokens on the WebSocket.
        """
        if not self._kite:
            raise RuntimeError("Kite session not set. Pass kite to constructor.")

        # Build symbol → token map from instruments dump
        if not self._symbol_to_token:
            logger.info("Fetching instrument list from Kite for %s…", exchange)
            instruments = self._kite.instruments(exchange)
            for inst in instruments:
                name = inst.get("tradingsymbol", "")
                token = inst.get("instrument_token", 0)
                # Only map EQ segment (skip futures, options, indices)
                segment = inst.get("segment", "")
                if segment in (f"{exchange}", f"{exchange}-EQ", "INDICES"):
                    self._symbol_to_token[name] = token
                    self._token_to_symbol[token] = name
            logger.info(
                "Instrument map built: %d symbols for %s",
                len(self._symbol_to_token), exchange,
            )

        # Resolve requested symbols
        new_tokens = set()
        not_found = []
        for sym in symbols:
            token = self._symbol_to_token.get(sym)
            if token:
                new_tokens.add(token)
            else:
                not_found.append(sym)

        if not_found:
            logger.warning(
                "%d symbols not found in instrument list: %s",
                len(not_found), not_found[:20],
            )

        added = new_tokens - self._instrument_tokens
        self._instrument_tokens.update(new_tokens)

        logger.info(
            "Subscribed %d new tokens (%d total). Not found: %d",
            len(added), len(self._instrument_tokens), len(not_found),
        )

        # If already connected, hot-subscribe
        if self._ticker and self._connected and added:
            token_list = list(added)
            self._ticker.subscribe(token_list)
            self._set_mode(token_list)

    def unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Remove symbols from the WebSocket subscription."""
        tokens_to_remove = set()
        for sym in symbols:
            token = self._symbol_to_token.get(sym)
            if token and token in self._instrument_tokens:
                tokens_to_remove.add(token)
                self._instrument_tokens.discard(token)
                self._tick_cache.pop(sym, None)

        if self._ticker and self._connected and tokens_to_remove:
            self._ticker.unsubscribe(list(tokens_to_remove))

    # ── WebSocket lifecycle ────────────────────────────────────

    def start(self) -> None:
        """Start the WebSocket connection on a background thread."""
        if self._running:
            logger.info("WebSocket already running")
            return

        from kiteconnect import KiteTicker

        self._ticker = KiteTicker(
            api_key=self._api_key,
            access_token=self._access_token,
            reconnect=True,
            reconnect_max_tries=50,
            reconnect_max_delay=60,
        )

        # Wire up callbacks
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_reconnect = self._on_reconnect
        self._ticker.on_noreconnect = self._on_noreconnect
        self._ticker.on_order_update = self._on_order_update

        self._running = True

        # Start batch flusher thread
        self._batch_thread = threading.Thread(
            target=self._batch_flush_loop,
            daemon=True,
            name="ws-batch-flusher",
        )
        self._batch_thread.start()

        # KiteTicker.connect(threaded=True) spawns its own daemon thread
        logger.info("Starting Kite WebSocket (tokens=%d)…", len(self._instrument_tokens))
        self._ticker.connect(threaded=True)

        # Dispatch connection event
        self._dispatcher.dispatch(WebhookEvent(
            event_type=EventType.WS_CONNECTED,
            payload={"tokens": len(self._instrument_tokens)},
        ))

    def stop(self) -> None:
        """Gracefully shut down the WebSocket."""
        self._running = False
        if self._ticker:
            try:
                self._ticker.close()
            except Exception:
                pass
        self._connected = False
        self._dispatcher.dispatch(WebhookEvent(
            event_type=EventType.WS_DISCONNECTED,
            payload={"reason": "manual_stop"},
        ))
        logger.info("WebSocket stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Tick cache (for UI reads) ──────────────────────────────

    def get_latest_ticks(self) -> Dict[str, TickData]:
        """Return the latest tick for every subscribed symbol (thread-safe)."""
        with self._tick_cache_lock:
            return dict(self._tick_cache)

    def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get the latest tick for a single symbol."""
        with self._tick_cache_lock:
            return self._tick_cache.get(symbol)

    def get_quotes_dict(self) -> Dict[str, Dict]:
        """
        Return quotes in the same format as ``fetch_realtime_quotes()``
        so the existing DB/UI code works without changes.

        Returns ``{symbol: {high, low, volume, ltp, change_pct}}``
        """
        with self._tick_cache_lock:
            return {
                sym: tick.to_db_dict()
                for sym, tick in self._tick_cache.items()
            }

    @property
    def last_tick_time(self) -> float:
        return self._last_tick_time

    @property
    def tick_count(self) -> int:
        with self._tick_cache_lock:
            return len(self._tick_cache)

    # ── KiteTicker callbacks ───────────────────────────────────

    def _on_connect(self, ws, response):
        """Called when WebSocket connection is established."""
        self._connected = True
        self._reconnect_count = 0
        logger.info("WebSocket connected. Subscribing %d tokens…", len(self._instrument_tokens))

        if self._instrument_tokens:
            token_list = list(self._instrument_tokens)
            ws.subscribe(token_list)
            self._set_mode(token_list)

        self._dispatcher.dispatch(WebhookEvent(
            event_type=EventType.WS_CONNECTED,
            payload={"tokens_subscribed": len(self._instrument_tokens)},
        ))

    def _on_ticks(self, ws, ticks: list):
        """
        Called for every tick batch from Kite WebSocket.

        Parses raw ticks into TickData, updates the cache, and
        buffers them for batched dispatch.
        """
        if not ticks:
            return

        parsed: List[TickData] = []
        for raw_tick in ticks:
            try:
                tick = TickData.from_kite_tick(raw_tick, self._token_to_symbol)
                parsed.append(tick)

                # Update cache
                with self._tick_cache_lock:
                    self._tick_cache[tick.tradingsymbol] = tick
            except Exception:
                logger.debug("Failed to parse tick: %s", raw_tick, exc_info=True)

        self._last_tick_time = time.time()

        # Buffer for batched dispatch
        with self._batch_lock:
            self._batch_buffer.extend(parsed)

    def _on_close(self, ws, code, reason):
        """Called when WebSocket is closed."""
        self._connected = False
        logger.warning("WebSocket closed: code=%s, reason=%s", code, reason)
        self._dispatcher.dispatch(WebhookEvent(
            event_type=EventType.WS_DISCONNECTED,
            payload={"code": code, "reason": str(reason)},
        ))

    def _on_error(self, ws, code, reason):
        """Called on WebSocket error."""
        logger.error("WebSocket error: code=%s, reason=%s", code, reason)
        self._dispatcher.dispatch(WebhookEvent(
            event_type=EventType.WS_ERROR,
            payload={"code": code, "reason": str(reason)},
        ))

    def _on_reconnect(self, ws, attempts_count):
        """Called when WebSocket is reconnecting."""
        self._connected = False
        self._reconnect_count = attempts_count
        logger.info("WebSocket reconnecting… attempt %d", attempts_count)
        self._dispatcher.dispatch(WebhookEvent(
            event_type=EventType.WS_RECONNECTING,
            payload={"attempt": attempts_count},
        ))

    def _on_noreconnect(self, ws):
        """Called when all reconnection attempts are exhausted."""
        self._connected = False
        self._running = False
        logger.error("WebSocket: all reconnect attempts exhausted.")
        self._dispatcher.dispatch(WebhookEvent(
            event_type=EventType.WS_DISCONNECTED,
            payload={"reason": "reconnect_exhausted"},
        ))

    def _on_order_update(self, ws, data):
        """Called when an order update is received via WebSocket."""
        status = (data.get("status") or "").upper()
        event_map = {
            "COMPLETE": EventType.TRADE_EXECUTED,
            "REJECTED": EventType.ORDER_REJECTED,
            "CANCELLED": EventType.ORDER_CANCELLED,
        }
        event_type = event_map.get(status, EventType.ORDER_PLACED)
        self._dispatcher.dispatch(WebhookEvent(
            event_type=event_type,
            payload=data,
            source="kite_order_ws",
        ))

    # ── Batch flusher ──────────────────────────────────────────

    def _batch_flush_loop(self):
        """
        Background thread that flushes accumulated ticks to the
        dispatcher at a fixed interval.

        This avoids dispatching per-tick (which would be thousands
        of events/sec) and instead sends one TICK_BATCH event every
        ``_batch_interval`` seconds.
        """
        while self._running:
            time.sleep(self._batch_interval)

            with self._batch_lock:
                if not self._batch_buffer:
                    continue
                batch = self._batch_buffer[:]
                self._batch_buffer.clear()

            # Build payload
            payload = {
                "ticks": [t.to_db_dict() for t in batch],
                "count": len(batch),
                "symbols": list({t.tradingsymbol for t in batch}),
                "timestamp": time.time(),
            }

            self._dispatcher.dispatch(WebhookEvent(
                event_type=EventType.TICK_BATCH,
                payload=payload,
            ))

    # ── Helpers ────────────────────────────────────────────────

    def _set_mode(self, tokens: list):
        """Set the subscription mode for a list of tokens."""
        mode_map = {"full": "MODE_FULL", "quote": "MODE_QUOTE", "ltp": "MODE_LTP"}
        from kiteconnect import KiteTicker
        mode_const = getattr(KiteTicker, mode_map.get(self._mode, "MODE_FULL"))
        self._ticker.set_mode(mode_const, tokens)

    def update_session(self, kite) -> None:
        """
        Re-bind to a fresh Kite session (after 2FA re-auth).

        Stops the current WebSocket and restarts with the new access token.
        """
        was_running = self._running
        self.stop()
        time.sleep(1)

        self._kite = kite
        self._api_key = kite.api_key
        self._access_token = kite.access_token

        if was_running:
            self.start()

    def get_status(self) -> Dict:
        """Return a status snapshot for the UI."""
        return {
            "connected": self._connected,
            "running": self._running,
            "subscribed_tokens": len(self._instrument_tokens),
            "cached_ticks": self.tick_count,
            "last_tick_time": self._last_tick_time,
            "reconnect_count": self._reconnect_count,
            "dispatcher_stats": self._dispatcher.stats,
        }
