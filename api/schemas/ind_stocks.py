"""
Pydantic schemas for Indian Stocks (Kite Connect / Zerodha) API endpoints.

Covers: authentication, quotes, orders, positions, holdings, option chains,
NSE data, and webhook configuration.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class KiteLoginRequest(BaseModel):
    """Request to initiate Kite login flow."""
    api_key: Optional[str] = Field(None, description="Override API key (uses env default if omitted)")
    request_token: str = Field(..., description="Request token from Kite login redirect")


class KiteLoginResponse(BaseModel):
    """Response from Kite login."""
    success: bool = True
    user_id: str = ""
    login_time: Optional[datetime] = None
    message: str = "Authenticated successfully"


class KiteSessionStatus(BaseModel):
    """Current Kite session status."""
    authenticated: bool = False
    user_id: Optional[str] = None
    api_key_set: bool = False
    market_open: bool = False


# ---------------------------------------------------------------------------
# Quotes & Market Data
# ---------------------------------------------------------------------------

class QuoteRequest(BaseModel):
    """Request for stock quotes."""
    instruments: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=[["NSE:RELIANCE", "NSE:TCS", "NSE:INFY"]],
        description="List of instrument keys in EXCHANGE:SYMBOL format",
    )


class QuoteData(BaseModel):
    """Single instrument quote."""
    instrument: str
    last_price: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    volume: Optional[int] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    timestamp: Optional[datetime] = None


class QuoteResponse(BaseModel):
    """Response containing instrument quotes."""
    success: bool = True
    quotes: List[QuoteData]


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

class PlaceOrderRequest(BaseModel):
    """Request to place an order on Zerodha."""
    symbol: str = Field(..., examples=["RELIANCE"])
    exchange: str = Field("NSE", examples=["NSE", "BSE", "NFO"])
    transaction_type: str = Field(..., examples=["BUY", "SELL"])
    quantity: int = Field(..., gt=0)
    order_type: str = Field("MARKET", examples=["MARKET", "LIMIT", "SL", "SL-M"])
    product: str = Field("CNC", examples=["CNC", "MIS", "NRML"])
    price: Optional[float] = Field(None, description="Required for LIMIT orders")
    trigger_price: Optional[float] = Field(None, description="Required for SL/SL-M orders")
    validity: str = Field("DAY", examples=["DAY", "IOC"])
    tag: Optional[str] = Field(None, max_length=20, description="Optional order tag")


class OrderResponse(BaseModel):
    """Response after placing/modifying an order."""
    success: bool = True
    order_id: str = ""
    message: str = ""


class OrderInfo(BaseModel):
    """Details of an existing order."""
    order_id: str
    symbol: str
    exchange: str
    transaction_type: str
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    order_type: str
    product: str
    status: str
    filled_quantity: int = 0
    average_price: float = 0.0
    placed_at: Optional[datetime] = None


class OrderBookResponse(BaseModel):
    """Response listing all orders."""
    success: bool = True
    orders: List[OrderInfo]


class CancelOrderRequest(BaseModel):
    """Request to cancel an order."""
    order_id: str
    variety: str = Field("regular", examples=["regular", "amo", "co", "iceberg"])


# ---------------------------------------------------------------------------
# Positions & Holdings
# ---------------------------------------------------------------------------

class PositionData(BaseModel):
    """Single position record."""
    symbol: str
    exchange: str
    product: str
    quantity: int
    average_price: float
    last_price: Optional[float] = None
    pnl: Optional[float] = None
    day_change: Optional[float] = None


class PositionsResponse(BaseModel):
    """Response listing positions."""
    success: bool = True
    net: List[PositionData] = Field(default_factory=list)
    day: List[PositionData] = Field(default_factory=list)


class HoldingData(BaseModel):
    """Single holding record."""
    symbol: str
    exchange: str
    isin: str = ""
    quantity: int
    average_price: float
    last_price: Optional[float] = None
    pnl: Optional[float] = None
    day_change: Optional[float] = None
    day_change_pct: Optional[float] = None


class HoldingsResponse(BaseModel):
    """Response listing holdings."""
    success: bool = True
    holdings: List[HoldingData]
    total_investment: float = 0.0
    total_current_value: float = 0.0
    total_pnl: float = 0.0
    day_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Option Chain
# ---------------------------------------------------------------------------

class OptionChainRequest(BaseModel):
    """Request for option chain data."""
    index: str = Field("NIFTY", examples=["NIFTY", "BANKNIFTY"])
    expiry: Optional[str] = Field(
        None,
        description="Specific expiry date (YYYY-MM-DD). Uses nearest if omitted.",
    )
    strike_range: int = Field(10, ge=1, le=30, description="Number of strikes above/below ATM")


class OptionStrikeData(BaseModel):
    """Data for a single option strike."""
    strike: float
    ce_ltp: Optional[float] = None
    ce_oi: Optional[int] = None
    ce_volume: Optional[int] = None
    ce_iv: Optional[float] = None
    pe_ltp: Optional[float] = None
    pe_oi: Optional[int] = None
    pe_volume: Optional[int] = None
    pe_iv: Optional[float] = None


class OptionChainResponse(BaseModel):
    """Response containing option chain data."""
    success: bool = True
    index: str
    spot_price: Optional[float] = None
    atm_strike: Optional[float] = None
    expiry: Optional[str] = None
    expiries: List[str] = Field(default_factory=list)
    chain: List[OptionStrikeData] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# NSE Market Data
# ---------------------------------------------------------------------------

class NSEStockData(BaseModel):
    """Single NSE stock row from live equity CSV data."""
    symbol: str
    name: str = ""
    ltp: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[int] = None


class NSEMarketResponse(BaseModel):
    """Response containing NSE market data."""
    success: bool = True
    count: int = 0
    stocks: List[NSEStockData] = Field(default_factory=list)
    index_group: Optional[str] = None


# ---------------------------------------------------------------------------
# Webhooks / WebSocket
# ---------------------------------------------------------------------------

class WebhookConfigRequest(BaseModel):
    """Request to configure webhook subscriptions."""
    instruments: List[str] = Field(
        ...,
        min_length=1,
        description="Instrument tokens or EXCHANGE:SYMBOL keys to subscribe",
    )
    mode: str = Field("ltp", examples=["ltp", "quote", "full"])


class WebhookStatusResponse(BaseModel):
    """Response showing webhook/WebSocket status."""
    success: bool = True
    connected: bool = False
    subscribed_count: int = 0
    market_open: bool = False
    last_tick_time: Optional[datetime] = None


class TickData(BaseModel):
    """Single tick data point."""
    instrument_token: int
    last_price: Optional[float] = None
    change: Optional[float] = None
    volume: Optional[int] = None
    timestamp: Optional[datetime] = None
    mode: str = "ltp"
