"""
Indian Stocks (Kite Connect / Zerodha) API router.

Endpoints for authentication, quotes, orders, positions, holdings,
option chains, NSE data, and webhook status.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_kite_session, set_kite_session
from api.schemas.common import ErrorResponse, SuccessResponse
from api.schemas.ind_stocks import (
    CancelOrderRequest,
    HoldingData,
    HoldingsResponse,
    KiteLoginRequest,
    KiteLoginResponse,
    KiteSessionStatus,
    NSEMarketResponse,
    NSEStockData,
    OptionChainRequest,
    OptionChainResponse,
    OptionStrikeData,
    OrderBookResponse,
    OrderInfo,
    OrderResponse,
    PlaceOrderRequest,
    PositionData,
    PositionsResponse,
    QuoteData,
    QuoteRequest,
    QuoteResponse,
    WebhookConfigRequest,
    WebhookStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ind-stocks", tags=["Indian Stocks"])


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _require_kite():
    """Return kite instance or raise 401."""
    kite = get_kite_session()
    if kite is None:
        raise HTTPException(
            status_code=401,
            detail="Kite session not authenticated. POST /ind-stocks/auth first.",
        )
    return kite


# -----------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------

@router.post(
    "/auth",
    response_model=KiteLoginResponse,
    summary="Authenticate with Kite Connect",
)
async def kite_login(request: KiteLoginRequest):
    """
    Exchange a request_token for an access_token and store the session.
    """
    try:
        from kiteconnect import KiteConnect
        from kite_connect.core.config import ZERODHA_API_KEY, ZERODHA_API_SECRET

        api_key = request.api_key or ZERODHA_API_KEY
        # create with larger pool to support concurrent requests
        kite = KiteConnect(
            api_key=api_key,
            pool={"pool_maxsize": int(os.getenv("KITE_POOL_MAXSIZE", "20"))},
        )
        data = kite.generate_session(request.request_token, api_secret=ZERODHA_API_SECRET)
        kite.set_access_token(data["access_token"])

        set_kite_session(kite)

        return KiteLoginResponse(
            success=True,
            user_id=data.get("user_id", ""),
            login_time=datetime.utcnow(),
            message="Authenticated successfully",
        )
    except Exception as exc:
        logger.exception("Kite login failed")
        raise HTTPException(status_code=401, detail=str(exc))


@router.get(
    "/auth/status",
    response_model=KiteSessionStatus,
    summary="Check Kite session status",
)
async def kite_status():
    """Return current Kite session authentication status."""
    kite = get_kite_session()
    authenticated = kite is not None

    market_open = False
    user_id = None
    if authenticated:
        try:
            profile = kite.profile()
            user_id = profile.get("user_id")
        except Exception:
            pass

        try:
            from kite_connect.webhooks.service import WebhookService
            ws = WebhookService()
            market_open = ws.market_is_open
        except Exception:
            pass

    return KiteSessionStatus(
        authenticated=authenticated,
        user_id=user_id,
        api_key_set=authenticated,
        market_open=market_open,
    )


# -----------------------------------------------------------------------
# Quotes & Market Data
# -----------------------------------------------------------------------

@router.post(
    "/quotes",
    response_model=QuoteResponse,
    summary="Get live quotes for instruments",
)
async def get_quotes(request: QuoteRequest):
    """Fetch live quotes (LTP, OHLC, volume) for the given instruments."""
    kite = _require_kite()
    try:
        raw = kite.quote(request.instruments)
        quotes = []
        for inst_key, q in raw.items():
            ohlc = q.get("ohlc", {})
            quotes.append(
                QuoteData(
                    instrument=inst_key,
                    last_price=q.get("last_price"),
                    change=q.get("net_change"),
                    change_pct=q.get("change"),  # Kite provides this directly in some responses
                    volume=q.get("volume"),
                    open=ohlc.get("open"),
                    high=ohlc.get("high"),
                    low=ohlc.get("low"),
                    close=ohlc.get("close"),
                    timestamp=q.get("timestamp"),
                )
            )
        return QuoteResponse(success=True, quotes=quotes)
    except Exception as exc:
        logger.exception("Quote fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/quotes/{exchange}/{symbol}",
    response_model=QuoteResponse,
    summary="Get quote for a single instrument",
)
async def get_single_quote(exchange: str, symbol: str):
    """Get live quote for EXCHANGE:SYMBOL (e.g. NSE/RELIANCE)."""
    kite = _require_kite()
    inst_key = f"{exchange.upper()}:{symbol.upper()}"
    try:
        raw = kite.quote([inst_key])
        q = raw.get(inst_key, {})
        ohlc = q.get("ohlc", {})
        return QuoteResponse(
            success=True,
            quotes=[
                QuoteData(
                    instrument=inst_key,
                    last_price=q.get("last_price"),
                    change=q.get("net_change"),
                    volume=q.get("volume"),
                    open=ohlc.get("open"),
                    high=ohlc.get("high"),
                    low=ohlc.get("low"),
                    close=ohlc.get("close"),
                    timestamp=q.get("timestamp"),
                )
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Orders
# -----------------------------------------------------------------------

@router.post(
    "/orders",
    response_model=OrderResponse,
    summary="Place a new order",
)
async def place_order(request: PlaceOrderRequest):
    """Place an order through Zerodha Kite Connect."""
    kite = _require_kite()
    try:
        from kite_connect.trading.order_service import place_order as _place_order

        result = _place_order(
            kite,
            symbol=request.symbol,
            exchange=request.exchange,
            transaction_type=request.transaction_type,
            quantity=request.quantity,
            order_type=request.order_type,
            product=request.product,
            price=request.price,
            trigger_price=request.trigger_price,
            validity=request.validity,
        )
        if result.get("success"):
            return OrderResponse(
                success=True,
                order_id=str(result.get("order_id", "")),
                message="Order placed successfully",
            )
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Order placement failed"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Order placement failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/orders",
    response_model=OrderBookResponse,
    summary="Get order book",
)
async def get_order_book():
    """Retrieve all orders for the current trading day."""
    kite = _require_kite()
    try:
        from kite_connect.trading.order_service import get_order_book as _get_order_book

        orders_raw = _get_order_book(kite)
        orders = [
            OrderInfo(
                order_id=str(o.get("order_id", "")),
                symbol=o.get("tradingsymbol", ""),
                exchange=o.get("exchange", ""),
                transaction_type=o.get("transaction_type", ""),
                quantity=o.get("quantity", 0),
                price=o.get("price"),
                trigger_price=o.get("trigger_price"),
                order_type=o.get("order_type", ""),
                product=o.get("product", ""),
                status=o.get("status", ""),
                filled_quantity=o.get("filled_quantity", 0),
                average_price=o.get("average_price", 0.0),
                placed_at=o.get("order_timestamp"),
            )
            for o in orders_raw
        ]
        return OrderBookResponse(success=True, orders=orders)
    except Exception as exc:
        logger.exception("Order book fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Cancel an order",
)
async def cancel_order(order_id: str, variety: str = "regular"):
    """Cancel a pending order by order_id."""
    kite = _require_kite()
    try:
        from kite_connect.trading.order_service import cancel_order as _cancel_order

        result = _cancel_order(kite, order_id, variety=variety)
        if result.get("success"):
            return OrderResponse(success=True, order_id=order_id, message="Order cancelled")
        raise HTTPException(status_code=400, detail=result.get("error", "Cancel failed"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Positions & Holdings
# -----------------------------------------------------------------------

@router.get(
    "/positions",
    response_model=PositionsResponse,
    summary="Get current positions",
)
async def get_positions():
    """Get net and day positions."""
    kite = _require_kite()
    try:
        from kite_connect.trading.order_service import get_positions as _get_positions

        raw = _get_positions(kite)
        net = [
            PositionData(
                symbol=p.get("tradingsymbol", ""),
                exchange=p.get("exchange", ""),
                product=p.get("product", ""),
                quantity=p.get("quantity", 0),
                average_price=p.get("average_price", 0.0),
                last_price=p.get("last_price"),
                pnl=p.get("pnl"),
                day_change=p.get("day_m2m"),
            )
            for p in (raw.get("net") or [])
        ]
        day = [
            PositionData(
                symbol=p.get("tradingsymbol", ""),
                exchange=p.get("exchange", ""),
                product=p.get("product", ""),
                quantity=p.get("quantity", 0),
                average_price=p.get("average_price", 0.0),
                last_price=p.get("last_price"),
                pnl=p.get("pnl"),
                day_change=p.get("day_m2m"),
            )
            for p in (raw.get("day") or [])
        ]
        return PositionsResponse(success=True, net=net, day=day)
    except Exception as exc:
        logger.exception("Positions fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/holdings",
    response_model=HoldingsResponse,
    summary="Get portfolio holdings",
)
async def get_holdings():
    """Get current portfolio holdings."""
    kite = _require_kite()
    try:
        from kite_connect.trading.order_service import get_holdings as _get_holdings

        raw = _get_holdings(kite)
        holdings = []
        total_investment = 0.0
        total_current = 0.0
        total_pnl = 0.0
        day_pnl = 0.0

        for h in raw:
            avg = h.get("average_price", 0)
            qty = h.get("quantity", 0)
            ltp = h.get("last_price") or 0
            inv = avg * qty
            cur = ltp * qty
            pnl = cur - inv
            d_change = h.get("day_change", 0) or 0
            d_change_pct = h.get("day_change_percentage", 0) or 0

            total_investment += inv
            total_current += cur
            total_pnl += pnl
            day_pnl += d_change * qty

            holdings.append(
                HoldingData(
                    symbol=h.get("tradingsymbol", ""),
                    exchange=h.get("exchange", ""),
                    isin=h.get("isin", ""),
                    quantity=qty,
                    average_price=avg,
                    last_price=ltp,
                    pnl=pnl,
                    day_change=d_change,
                    day_change_pct=d_change_pct,
                )
            )

        return HoldingsResponse(
            success=True,
            holdings=holdings,
            total_investment=total_investment,
            total_current_value=total_current,
            total_pnl=total_pnl,
            day_pnl=day_pnl,
        )
    except Exception as exc:
        logger.exception("Holdings fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Option Chain
# -----------------------------------------------------------------------

@router.post(
    "/option-chain",
    response_model=OptionChainResponse,
    summary="Fetch live option chain",
)
async def get_option_chain(request: OptionChainRequest):
    """Fetch option chain for NIFTY or BANKNIFTY centred on ATM."""
    kite = _require_kite()
    try:
        from kite_connect.options.option_chain import (
            discover_expiries,
            fetch_option_chain,
        )

        # Discover expiries
        expiries = discover_expiries(kite, index=request.index)

        # Use requested expiry or first available
        expiry_code = ""
        if request.expiry:
            # Try to match user-supplied date to a known expiry code
            expiry_code = request.expiry
        elif expiries:
            expiry_code = expiries[0]

        chain_data = fetch_option_chain(
            kite,
            index=request.index,
            expiry_code=expiry_code,
            num_strikes=request.strike_range * 2,
        )

        chain = [
            OptionStrikeData(
                strike=row["strike"],
                ce_ltp=row.get("ce_ltp"),
                ce_oi=row.get("ce_oi"),
                ce_volume=row.get("ce_volume"),
                pe_ltp=row.get("pe_ltp"),
                pe_oi=row.get("pe_oi"),
                pe_volume=row.get("pe_volume"),
            )
            for row in chain_data.get("strikes", [])
        ]

        return OptionChainResponse(
            success=True,
            index=request.index,
            spot_price=chain_data.get("spot"),
            atm_strike=chain_data.get("atm_strike"),
            expiry=expiry_code,
            expiries=expiries,
            chain=chain,
        )
    except Exception as exc:
        logger.exception("Option chain fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# NSE Market Data
# -----------------------------------------------------------------------

@router.get(
    "/nse/stocks",
    response_model=NSEMarketResponse,
    summary="Get NSE equity data from database",
)
async def get_nse_stocks(
    index_group: Optional[str] = Query(
        None,
        description="Filter by index group (NIFTY50, NIFTYBANK, NIFTYIT, NIFTYENERGY)",
    ),
    limit: int = Query(50, ge=1, le=500),
):
    """Return NSE stock data from the livestocks_ind database."""
    try:
        from kite_connect.core.db_service import get_connection
        from kite_connect.core.config import INDEX_GROUPS

        conn = get_connection(dbname="livestocks_ind")
        cur = conn.cursor()

        if index_group and index_group.upper() in INDEX_GROUPS:
            symbols = INDEX_GROUPS[index_group.upper()]
            placeholders = ",".join(["%s"] * len(symbols))
            cur.execute(
                f"SELECT symbol, name, ltp, change, open, high, low, volume "
                f"FROM live_stocks WHERE symbol IN ({placeholders}) "
                f"ORDER BY symbol LIMIT %s",
                symbols + [limit],
            )
        else:
            cur.execute(
                "SELECT symbol, name, ltp, change, open, high, low, volume "
                "FROM live_stocks ORDER BY symbol LIMIT %s",
                (limit,),
            )

        rows = cur.fetchall()
        cur.close()
        conn.close()

        stocks = [
            NSEStockData(
                symbol=r[0],
                name=r[1] or "",
                ltp=r[2],
                change=r[3],
                open=r[4],
                high=r[5],
                low=r[6],
                volume=r[7],
            )
            for r in rows
        ]

        return NSEMarketResponse(
            success=True,
            count=len(stocks),
            stocks=stocks,
            index_group=index_group,
        )
    except Exception as exc:
        logger.exception("NSE data fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Webhook / WebSocket Status
# -----------------------------------------------------------------------

@router.get(
    "/webhooks/status",
    response_model=WebhookStatusResponse,
    summary="Get WebSocket/webhook connection status",
)
async def webhook_status():
    """Return the status of the Kite WebSocket ticker."""
    try:
        from kite_connect.webhooks.service import WebhookService

        ws = WebhookService()
        return WebhookStatusResponse(
            success=True,
            connected=ws.is_running,
            subscribed_count=ws.subscribed_count,
            market_open=ws.market_is_open,
            last_tick_time=ws.last_tick_time,
        )
    except Exception as exc:
        logger.warning("Webhook status unavailable: %s", exc)
        return WebhookStatusResponse(success=True, connected=False)
