"""
Webhook event models for Centurion Core — Indian Stocks.

Defines the event types and data structures used by the webhook
dispatcher to propagate real-time market events.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    """Market events that can trigger webhook notifications."""

    # ── Price & Tick Events ────────────────────────────────────
    TICK_UPDATE = "tick.update"               # Real-time tick from WebSocket
    TICK_BATCH = "tick.batch"                 # Batch of ticks (bulk update)

    # ── Market Session Events ─────────────────────────────────
    MARKET_OPEN = "market.open"               # Market session started
    MARKET_CLOSE = "market.close"             # Market session ended
    MARKET_PRE_OPEN = "market.pre_open"       # Pre-open session
    MARKET_STATUS_CHANGE = "market.status"    # Any NSE status change

    # ── Trade Events ──────────────────────────────────────────
    TRADE_EXECUTED = "trade.executed"          # Order filled
    ORDER_PLACED = "order.placed"             # Order submitted
    ORDER_CANCELLED = "order.cancelled"       # Order cancelled
    ORDER_REJECTED = "order.rejected"         # Order rejected by exchange

    # ── Strategy / Signal Events ──────────────────────────────
    RSI_SIGNAL = "strategy.rsi_signal"        # RSI signal detected
    STRATEGY_ALERT = "strategy.alert"         # Generic strategy alert

    # ── Connection Events ─────────────────────────────────────
    WS_CONNECTED = "ws.connected"             # WebSocket connected
    WS_DISCONNECTED = "ws.disconnected"       # WebSocket disconnected
    WS_RECONNECTING = "ws.reconnecting"       # WebSocket reconnecting
    WS_ERROR = "ws.error"                     # WebSocket error

    # ── System Events ─────────────────────────────────────────
    SESSION_EXPIRED = "session.expired"       # Kite session token expired
    DB_UPDATED = "db.updated"                 # DB write completed


@dataclass
class TickData:
    """Parsed real-time tick data from Kite WebSocket."""

    instrument_token: int
    tradingsymbol: str
    last_price: float
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    volume: int = 0
    change_pct: float = 0.0
    last_trade_time: Optional[str] = None
    oi: int = 0                               # Open interest (F&O)
    depth: Optional[Dict] = None              # Market depth (buy/sell)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_kite_tick(cls, tick: Dict, symbol_map: Dict[int, str]) -> "TickData":
        """
        Parse a raw Kite WebSocket tick dict into a TickData instance.

        Parameters
        ----------
        tick : dict
            Raw tick from KiteTicker on_ticks callback.
        symbol_map : dict
            Mapping of instrument_token tradingsymbol.
        """
        token = tick.get("instrument_token", 0)
        ohlc = tick.get("ohlc", {})
        last_price = tick.get("last_price", 0.0)
        close = ohlc.get("close", 0.0)
        change_pct = (
            round(((last_price - close) / close) * 100, 2)
            if close
            else 0.0
        )

        ltt = tick.get("last_trade_time")
        ltt_str = ltt.isoformat() if hasattr(ltt, "isoformat") else str(ltt) if ltt else None

        return cls(
            instrument_token=token,
            tradingsymbol=symbol_map.get(token, f"TOKEN:{token}"),
            last_price=last_price,
            high=ohlc.get("high", last_price),
            low=ohlc.get("low", last_price),
            open=ohlc.get("open", last_price),
            close=close,
            volume=tick.get("volume_traded", tick.get("volume", 0)),
            change_pct=change_pct,
            last_trade_time=ltt_str,
            oi=tick.get("oi", 0),
            depth=tick.get("depth"),
        )

    def to_db_dict(self) -> Dict[str, Any]:
        """Return a dict matching the DB update schema (stocks table)."""
        return {
            "name": self.tradingsymbol,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "ltp": self.last_price,
            "change_pct": self.change_pct,
        }


@dataclass
class WebhookEvent:
    """A single event dispatched through the webhook system."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.TICK_UPDATE
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "kite_websocket"            # Origin identifier

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON transport / logging."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class WebhookSubscription:
    """
    Registration record for an internal event subscriber.

    Subscribers are Python callables (not external HTTP endpoints)
    because the push happens within the same process.
    """

    subscriber_id: str
    events: List[EventType]
    is_active: bool = True
    description: str = ""
