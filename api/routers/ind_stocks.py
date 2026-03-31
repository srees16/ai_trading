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


# -----------------------------------------------------------------------
# Vince Metrics (Phase 5)
# -----------------------------------------------------------------------

@router.get("/vince/metrics", response_model=SuccessResponse)
async def vince_metrics():
    """Return Ralph Vince risk metrics: optimal-f, geometric mean, kelly,
    equalized weights, fundamental equation, and per-symbol snapshots."""
    try:
        from services.vince_metrics import get_vince_tracker
        tracker = get_vince_tracker()
        data = tracker.to_dict()
        return SuccessResponse(success=True, data=data)
    except Exception as exc:
        logger.error("Vince metrics error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Penfold Trend Analysis (IND Stocks)
# -----------------------------------------------------------------------

@router.post(
    "/penfold/analysis",
    summary="Penfold trend analysis for IND stocks",
    description=(
        "Returns per-symbol Dow Theory trend, Turtle breakout, ATR band, "
        "retracement signals, and combined Penfold forecast for NSE stocks. "
        "Automated trades are placed via Kite; use this endpoint for signal "
        "transparency and manual override decisions."
    ),
)
async def ind_penfold_analysis(tickers: Optional[List[str]] = None):
    """Run Penfold trend analysis on IND stocks.

    Returns actionable signals:
    - Dow Theory daily/weekly trend direction + confidence
    - Turtle 4W channel breakout levels + stop
    - ATR band expansion forecast
    - Retracement pullback entry signal
    - Combined Penfold forecast (-20 to +20)
    - Equity curve R² and UPI when trade history is available
    """
    import asyncio

    try:
        from utils import download_ind_ohlcv
        from strategies.penfold_trend import (
            compute_penfold_trend_analysis,
            compute_weekly_dow_filter,
            compute_turtle_breakout,
            compute_atr_band_breakout,
            compute_retracement_entry,
            compute_dow_trend,
            equity_curve_r_squared,
            ulcer_performance_index,
        )

        # Default: use pipeline universe or top NSE stocks
        if not tickers:
            try:
                from config import Config
                from kite_connect.nse.screener import get_nse_universe
                tickers = get_nse_universe(tier=getattr(Config, "NSE_UNIVERSE_TIER", "NIFTY200"))
            except Exception:
                tickers = [
                    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
                    "ASIANPAINT", "SUNPHARMA", "HCLTECH", "WIPRO", "TATAMOTORS",
                ]

        results = []
        for sym in tickers:
            try:
                df = download_ind_ohlcv(sym, period="6mo")
                if df is None or len(df) < 64:
                    continue

                # Combined Penfold analysis
                penfold = compute_penfold_trend_analysis(df, sym)

                # Weekly Dow trend
                weekly_dow = compute_weekly_dow_filter(df)

                # Individual signals
                turtle = compute_turtle_breakout(df, sym)
                atr_band_fc = compute_atr_band_breakout(df, sym)
                retracement = compute_retracement_entry(df, sym)

                price = float(df["Close"].iloc[-1])

                entry = {
                    "symbol": sym,
                    "current_price": round(price, 2),
                    "combined_forecast": round(penfold.combined_forecast, 2),
                    "dow_trend_daily": penfold.dow_trend_daily,
                    "dow_confidence": round(penfold.dow_confidence, 2),
                    "dow_trend_weekly": weekly_dow,
                    "weekly_aligned": penfold.weekly_aligned,
                    "turtle": {
                        "entry_side": turtle.entry_side if turtle else "",
                        "forecast": round(turtle.forecast, 2) if turtle else 0.0,
                        "channel_high": round(turtle.channel_high, 2) if turtle else 0.0,
                        "channel_low": round(turtle.channel_low, 2) if turtle else 0.0,
                        "stop_level": round(turtle.stop_level, 2) if turtle else 0.0,
                    },
                    "atr_band": {
                        "forecast": round(atr_band_fc, 2) if atr_band_fc is not None else 0.0,
                        "action": (
                            "BUY" if atr_band_fc is not None and atr_band_fc > 10.0
                            else "SELL" if atr_band_fc is not None and atr_band_fc < -10.0
                            else "NEUTRAL"
                        ),
                    },
                    "retracement": {
                        "trend_direction": retracement.trend_direction if retracement else "",
                        "forecast": round(retracement.forecast, 2) if retracement else 0.0,
                        "entry_level": round(retracement.entry_level, 2) if retracement else 0.0,
                        "stop_level": round(retracement.stop_level, 2) if retracement else 0.0,
                    },
                    "action": (
                        "BUY" if penfold.combined_forecast > 5.0
                        else "SELL" if penfold.combined_forecast < -5.0
                        else "HOLD"
                    ),
                }
                results.append(entry)
            except Exception as exc:
                logger.debug("Penfold analysis skipped for %s: %s", sym, exc)

        # Sort by absolute forecast strength
        results.sort(key=lambda x: abs(x["combined_forecast"]), reverse=True)

        # Attach equity curve quality metrics if trade history exists
        risk_metrics = {}
        try:
            from services.vince_metrics import get_vince_tracker
            vt = get_vince_tracker()
            snap = vt.get_snapshot("__portfolio__")
            if snap and snap.n_trades >= 10:
                import pandas as pd
                import numpy as np
                # Build equity curve from cumulative PnL
                equity = pd.Series(getattr(snap, "equity_curve", []))
                if len(equity) >= 20:
                    risk_metrics["r_squared"] = round(equity_curve_r_squared(equity), 4)
                    risk_metrics["upi"] = round(ulcer_performance_index(equity), 3)
        except Exception:
            pass

        return {
            "success": True,
            "count": len(results),
            "analysis": results,
            "risk_metrics": risk_metrics,
            "note": "combined_forecast > 5 = BUY, < -5 = SELL. "
                    "Weekly Dow alignment increases conviction. "
                    "Trades automated via Kite for IND stocks.",
        }
    except Exception as exc:
        logger.exception("IND Penfold analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/penfold/calibrate",
    summary="Run Penfold-enhanced calibration backtest",
    description="Runs an expanding-window backtest including Penfold trend forecasts "
                "and returns CAGR, Sharpe, max DD, and per-strategy contribution.",
)
async def ind_penfold_calibrate(
    tickers: Optional[List[str]] = None,
    lookback_months: int = Query(default=12, ge=3, le=36),
):
    """Calibration backtest with Penfold trend integration."""
    import asyncio

    try:
        from services.penfold_backtest import run_penfold_enhanced_backtest

        if not tickers:
            try:
                from config import Config
                from kite_connect.nse.screener import get_nse_universe
                tickers = get_nse_universe(
                    tier=getattr(Config, "NSE_UNIVERSE_TIER", "NIFTY200")
                )[:30]  # Top 30 for calibration speed
            except Exception:
                tickers = [
                    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
                ]

        result = await asyncio.to_thread(
            run_penfold_enhanced_backtest, tickers, lookback_months
        )
        return result
    except Exception as exc:
        logger.exception("IND Penfold calibration failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────── Ehlers DSP Analysis ──────────────────────

@router.post(
    "/ehlers/analysis",
    response_model=SuccessResponse,
    summary="Ehlers DSP analysis for IND stocks",
    description="Computes Ehlers DSP indicators (Fisher, MAMA/FAMA, SuperSmoother, "
                "Sinewave, SNR, Adaptive RSI, Dominant Cycle) for each symbol.",
)
async def ind_ehlers_analysis(tickers: Optional[List[str]] = None):
    """Ehlers DSP analysis for IND stocks — integrated to auto-order pipeline."""
    import asyncio

    try:
        from strategies.ehlers_dsp import compute_ehlers_analysis_batch
        from utils import download_ind_ohlcv

        if not tickers:
            try:
                from config import Config
                from kite_connect.nse.screener import get_nse_universe
                tickers = get_nse_universe(tier=getattr(Config, "NSE_UNIVERSE_TIER", "NIFTY200"))[:20]
            except Exception:
                tickers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                           "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]

        ohlcv_data = await asyncio.to_thread(
            lambda: {s: download_ind_ohlcv(s, period="6mo") for s in tickers}
        )
        ohlcv_data = {s: d for s, d in ohlcv_data.items() if d is not None and len(d) >= 50}
        analysis = await asyncio.to_thread(compute_ehlers_analysis_batch, ohlcv_data)

        results = []
        for sym, ea in analysis.items():
            if ea is None:
                continue
            results.append({
                "symbol": sym,
                "fisher_transform": round(ea.fisher_transform, 4),
                "fisher_trigger": round(ea.fisher_trigger, 4),
                "mama": round(ea.mama, 4),
                "fama": round(ea.fama, 4),
                "mama_fama_trend": "BULL" if ea.mama > ea.fama else "BEAR",
                "sinewave": round(ea.sinewave, 4),
                "leadsine": round(ea.leadsine, 4),
                "snr_db": round(ea.snr, 2),
                "adaptive_rsi": round(ea.adaptive_rsi, 2),
                "dominant_cycle": round(ea.dominant_cycle, 1),
                "composite_forecast": round(ea.composite_forecast, 2),
                "action": "BUY" if ea.composite_forecast > 5 else "SELL" if ea.composite_forecast < -5 else "HOLD",
            })

        results.sort(key=lambda x: abs(x["composite_forecast"]), reverse=True)
        return {"success": True, "count": len(results), "analysis": results}
    except Exception as exc:
        logger.exception("IND Ehlers analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────── Ruggiero Cybernetic Analysis ─────────────

@router.post(
    "/cybernetic/analysis",
    response_model=SuccessResponse,
    summary="Ruggiero cybernetic analysis for IND stocks",
    description="Computes intermarket signals, seasonal bias, trend classification, "
                "multi-timeframe alignment for each symbol.",
)
async def ind_cybernetic_analysis(tickers: Optional[List[str]] = None):
    """Ruggiero cybernetic analysis — integrated to auto-order pipeline."""
    import asyncio

    try:
        from strategies.ruggiero_cybernetic import (
            compute_cybernetic_analysis_batch, IND_INTERMARKET_DRIVERS,
        )
        from utils import download_ind_ohlcv
        import yfinance as yf

        if not tickers:
            try:
                from config import Config
                from kite_connect.nse.screener import get_nse_universe
                tickers = get_nse_universe(tier=getattr(Config, "NSE_UNIVERSE_TIER", "NIFTY200"))[:20]
            except Exception:
                tickers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                           "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]

        ohlcv_data = await asyncio.to_thread(
            lambda: {s: download_ind_ohlcv(s, period="6mo") for s in tickers}
        )
        ohlcv_data = {s: d for s, d in ohlcv_data.items() if d is not None and len(d) >= 50}

        # Download intermarket drivers
        driver_syms = list(IND_INTERMARKET_DRIVERS.keys())
        driver_dfs = await asyncio.to_thread(
            lambda: {s: yf.download(s, period="6mo", progress=False) for s in driver_syms}
        )
        driver_dfs = {s: d for s, d in driver_dfs.items() if d is not None and len(d) > 20}

        analysis = await asyncio.to_thread(
            compute_cybernetic_analysis_batch, ohlcv_data, driver_dfs, IND_INTERMARKET_DRIVERS
        )

        results = []
        for sym, ca in analysis.items():
            if ca is None:
                continue
            results.append({
                "symbol": sym,
                "intermarket_forecast": round(ca.intermarket_forecast, 2),
                "seasonal_bias": round(ca.seasonal_bias.combined_bias, 4),
                "trend_strength": ca.trend_class.trend_strength,
                "adx": round(ca.trend_class.adx, 2),
                "multi_tf_alignment": round(ca.multi_tf_alignment, 4),
                "composite_forecast": round(ca.composite_forecast, 2),
                "action": "BUY" if ca.composite_forecast > 5 else "SELL" if ca.composite_forecast < -5 else "HOLD",
            })

        results.sort(key=lambda x: abs(x["composite_forecast"]), reverse=True)
        return {"success": True, "count": len(results), "analysis": results}
    except Exception as exc:
        logger.exception("IND Cybernetic analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────── Vince Leverage Space ─────────────────────

@router.post(
    "/vince/leverage-space",
    response_model=SuccessResponse,
    summary="Vince Leverage Space analysis for IND stocks",
    description="Computes optimal-f, secure-f, active equity ratio, "
                "and leverage recommendation per symbol.",
)
async def ind_vince_leverage_space(tickers: Optional[List[str]] = None):
    """Vince Leverage Space — caps auto-order sizing via secure_f."""
    import asyncio

    try:
        from strategies.vince_leverage import compute_vince_leverage_batch
        from utils import download_ind_ohlcv
        from config import Config

        if not tickers:
            try:
                from kite_connect.nse.screener import get_nse_universe
                tickers = get_nse_universe(tier=getattr(Config, "NSE_UNIVERSE_TIER", "NIFTY200"))[:20]
            except Exception:
                tickers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                           "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]

        ohlcv_data = await asyncio.to_thread(
            lambda: {s: download_ind_ohlcv(s, period="6mo") for s in tickers}
        )
        ohlcv_data = {s: d for s, d in ohlcv_data.items() if d is not None and len(d) >= 20}

        capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500000)
        hwm = getattr(Config, "_HWM", capital)
        insurance = getattr(Config, "VINCE_INSURANCE_PCT_IND", 0.15)
        max_lev = getattr(Config, "CARVER_MAX_LEVERAGE", 4.0)

        analysis = await asyncio.to_thread(
            compute_vince_leverage_batch, ohlcv_data, capital, hwm,
            0.20, insurance, max_lev, ""
        )

        results = []
        for sym, va in analysis.items():
            results.append({
                "symbol": sym,
                "optimal_f": va.optimal_f,
                "secure_f": va.secure_f,
                "kelly_fraction": va.kelly_fraction,
                "win_rate": round(va.win_rate, 4),
                "avg_win_loss_ratio": va.avg_win_loss_ratio,
                "leverage_recommendation": va.leverage_recommendation,
                "active_equity_ratio": va.active_equity_ratio,
                "max_drawdown_at_f": va.max_drawdown_at_f,
            })

        results.sort(key=lambda x: x["leverage_recommendation"], reverse=True)
        return {"success": True, "count": len(results), "analysis": results}
    except Exception as exc:
        logger.exception("IND Vince leverage space failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────── Masters Prediction Quality ───────────────

@router.post(
    "/prediction-quality",
    response_model=SuccessResponse,
    summary="Masters prediction quality assessment",
    description="Assesses forecast quality via directional accuracy, IC, "
                "Monte Carlo significance, equity R², and quality gate.",
)
async def ind_prediction_quality(tickers: Optional[List[str]] = None):
    """Masters prediction quality — gates auto-order pipeline forecasts."""
    import asyncio
    import numpy as np

    try:
        from strategies.masters_prediction import compute_prediction_quality
        from utils import download_ind_ohlcv

        if not tickers:
            try:
                from config import Config
                from kite_connect.nse.screener import get_nse_universe
                tickers = get_nse_universe(tier=getattr(Config, "NSE_UNIVERSE_TIER", "NIFTY200"))[:20]
            except Exception:
                tickers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                           "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]

        ohlcv_data = await asyncio.to_thread(
            lambda: {s: download_ind_ohlcv(s, period="6mo") for s in tickers}
        )

        results = []
        for sym, df in ohlcv_data.items():
            if df is None or len(df) < 60:
                continue
            close = df["Close"].values.astype(float)
            returns = np.diff(close[-60:]) / np.maximum(np.abs(close[-61:-1]), 1e-10)
            # Use SMA crossover as proxy forecast
            sma10 = np.convolve(close, np.ones(10)/10, mode='valid')
            sma20 = np.convolve(close, np.ones(20)/20, mode='valid')
            min_len = min(len(sma10), len(sma20), len(returns))
            forecasts = (sma10[-min_len:] - sma20[-min_len:])
            acts = returns[-min_len:]

            pq = compute_prediction_quality(sym, forecasts, acts, mc_permutations=500)
            results.append({
                "symbol": sym,
                "directional_accuracy": pq.directional_accuracy,
                "information_coefficient": pq.information_coefficient,
                "r_squared": pq.r_squared,
                "brier_score": pq.brier_score,
                "monte_carlo_p_value": pq.monte_carlo_p_value,
                "is_significant": pq.is_significant,
                "quality_score": pq.quality_score,
                "confidence_multiplier": pq.confidence_multiplier,
                "action": "TRUST" if pq.quality_score > 0.5 else "CAUTION" if pq.quality_score > 0.3 else "SUPPRESS",
            })

        results.sort(key=lambda x: x["quality_score"], reverse=True)
        return {"success": True, "count": len(results), "analysis": results}
    except Exception as exc:
        logger.exception("IND prediction quality failed")
        raise HTTPException(status_code=500, detail=str(exc))
