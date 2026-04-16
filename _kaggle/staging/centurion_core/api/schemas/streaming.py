"""
Pydantic schemas for real-time streaming endpoints.

Covers: SSE tick streaming, WebSocket messages, Kite Postback,
TimescaleDB OHLC aggregates, and price alerts.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SSE / WebSocket Tick Messages
# ---------------------------------------------------------------------------

class StreamTickData(BaseModel):
    """Single tick emitted via SSE or WebSocket."""
    symbol: str
    last_price: float
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    volume: int = 0
    change_pct: float = 0.0
    oi: int = 0
    timestamp: float


class StreamMessage(BaseModel):
    """Envelope for SSE / WebSocket messages."""
    event: str = Field(
        ...,
        description="Event type: tick_batch | alert | status | error",
    )
    data: Any = None
    ts: float = Field(default_factory=lambda: __import__("time").time())


# ---------------------------------------------------------------------------
# WebSocket Subscribe / Unsubscribe
# ---------------------------------------------------------------------------

class WSSubscribeRequest(BaseModel):
    """Client server: subscribe to symbols via WebSocket."""
    action: str = Field(
        "subscribe",
        description="subscribe | unsubscribe | ping",
    )
    symbols: List[str] = Field(
        default_factory=list,
        description="Trading symbols to subscribe/unsubscribe",
    )


# ---------------------------------------------------------------------------
# Kite Postback (order update callback)
# ---------------------------------------------------------------------------

class KitePostbackPayload(BaseModel):
    """
    Zerodha Kite order postback payload.

    Zerodha sends a POST to the configured postback URL whenever an
    order transitions.  This schema matches their payload format.
    See: https://kite.trade/docs/connect/v3/postbacks/
    """
    user_id: Optional[str] = None
    order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    placed_by: Optional[str] = None
    status: Optional[str] = Field(
        None,
        description="COMPLETE | CANCELLED | REJECTED | OPEN | TRIGGER PENDING",
    )
    status_message: Optional[str] = None
    status_message_raw: Optional[str] = None
    tradingsymbol: Optional[str] = None
    exchange: Optional[str] = None
    order_type: Optional[str] = None
    transaction_type: Optional[str] = None
    validity: Optional[str] = None
    product: Optional[str] = None
    quantity: Optional[int] = None
    filled_quantity: Optional[int] = None
    pending_quantity: Optional[int] = None
    price: Optional[float] = None
    average_price: Optional[float] = None
    trigger_price: Optional[float] = None
    disclosed_quantity: Optional[int] = None
    variety: Optional[str] = None
    tag: Optional[str] = None
    order_timestamp: Optional[str] = None
    exchange_timestamp: Optional[str] = None
    exchange_update_timestamp: Optional[str] = None
    unfilled_quantity: Optional[int] = None
    checksum: Optional[str] = Field(
        None,
        description="SHA-256 checksum for payload verification",
    )
    meta: Optional[Dict[str, Any]] = None


class PostbackResponse(BaseModel):
    """Acknowledgement for a received postback."""
    success: bool = True
    message: str = "Postback received"
    event_type: Optional[str] = None
    order_id: Optional[str] = None


# ---------------------------------------------------------------------------
# TimescaleDB OHLC Aggregates
# ---------------------------------------------------------------------------

class OHLCBar(BaseModel):
    """Single OHLC bar from continuous aggregate."""
    bucket: datetime = Field(..., description="Beginning of the time bucket")
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    vwap: Optional[float] = None
    trade_count: int = 0


class OHLCRequest(BaseModel):
    """Request OHLC bars for a symbol."""
    symbol: str
    interval: str = Field(
        "1m",
        description="Aggregate interval: 1m | 5m | 15m | 1h",
    )
    limit: int = Field(100, ge=1, le=1000)


class OHLCResponse(BaseModel):
    """Response containing OHLC bars."""
    success: bool = True
    symbol: str
    interval: str
    bars: List[OHLCBar] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Price Alerts
# ---------------------------------------------------------------------------

class AlertCondition(str, Enum):
    """Supported alert conditions."""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    CHANGE_PCT_ABOVE = "change_pct_above"
    CHANGE_PCT_BELOW = "change_pct_below"
    VOLUME_ABOVE = "volume_above"


class CreateAlertRequest(BaseModel):
    """Create a new price alert."""
    symbol: str = Field(..., examples=["RELIANCE"])
    condition: AlertCondition
    threshold: float = Field(..., description="Threshold value for the condition")
    message: Optional[str] = Field(
        None,
        description="Custom alert message (auto-generated if omitted)",
    )
    one_shot: bool = Field(
        True,
        description="If True, alert fires once then deactivates",
    )


class AlertInfo(BaseModel):
    """Details of an active alert."""
    alert_id: str
    symbol: str
    condition: AlertCondition
    threshold: float
    message: str = ""
    one_shot: bool = True
    active: bool = True
    created_at: float
    triggered_at: Optional[float] = None
    trigger_count: int = 0


class AlertListResponse(BaseModel):
    """List of configured alerts."""
    success: bool = True
    alerts: List[AlertInfo] = Field(default_factory=list)
    active_count: int = 0
    total_count: int = 0


class AlertCreateResponse(BaseModel):
    """Response from creating an alert."""
    success: bool = True
    alert_id: str = ""
    message: str = ""
