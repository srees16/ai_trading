"""
API v1 Gateway Router — maps Next.js frontend paths to existing service logic.

This router provides the /api/v1/* endpoints expected by the Next.js frontend,
delegating to existing internal modules. Missing functionality (DriveWealth,
FML, TTS chapters) is stubbed with minimal implementations.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_db_service, get_kite_session, get_rag_engine, get_trading_system

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["API v1"])


# ─── Request / Response Models ──────────────────────────────────────────

class AnalysisRunRequest(BaseModel):
    tickers: List[str]
    market: str = "US"
    period: str = "1y"


class BacktestRunRequest(BaseModel):
    strategy_id: str
    tickers: List[str]
    params: Dict[str, Any] = {}
    initial_capital: float = 100000
    period: str = "1y"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    market: str = "US"


class VerdictRunRequest(BaseModel):
    tickers: List[str]
    market: str = "US"
    date_range: List[str] = ["", ""]
    skip_layers: List[str] = []
    weights: Dict[str, float] = {"core": 0.3, "strategy": 0.3, "ml_features": 0.2, "robustness": 0.2}
    batch_size: int = 5


class ScreenerRunRequest(BaseModel):
    screener: Dict[str, Any]
    risk: Dict[str, Any]
    tickers: List[str] = []


class ChapterRunRequest(BaseModel):
    chapters: List[str]


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class DWLoginRequest(BaseModel):
    client_id: str
    client_secret: str
    app_key: str
    user_id: str
    account_id: str


class OrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: int
    limit_price: Optional[float] = None


# ─── Auth ────────────────────────────────────────────────────────────────

@router.post("/auth/login")
async def api_login(req: LoginRequest):
    """JWT login for the frontend."""
    from api.auth import authenticate_user, create_session_token
    ok, display_name, role = authenticate_user(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session_token(req.username, role)
    return {
        "access_token": token,
        "refresh_token": token,
        "user": {"username": req.username, "name": display_name, "role": role},
    }


@router.get("/auth/me")
async def api_auth_me(request: Request):
    """Return the current user from the session token."""
    from api.auth import verify_session_token

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = auth_header[7:]
    payload = verify_session_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"username": payload["u"], "name": payload["u"], "role": payload["r"]}


@router.post("/auth/logout")
async def api_logout():
    """Logout — client clears tokens; server acknowledges."""
    return {"ok": True}


@router.post("/auth/change-password")
async def api_change_password(req: ChangePasswordRequest, request: Request):
    """Change the authenticated user's password."""
    from api.auth import verify_session_token, _verify_password, CREDENTIALS_YAML

    # Verify current session
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    payload = verify_session_token(auth_header[7:])
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    username = payload["u"]

    # Load credentials
    import yaml
    if not CREDENTIALS_YAML.exists():
        raise HTTPException(status_code=500, detail="Credentials file not found")
    with open(CREDENTIALS_YAML, "r") as fh:
        creds = yaml.safe_load(fh) or {}
    users = creds.get("users", {})
    user = users.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Verify current password
    if not _verify_password(req.current_password, user.get("password", "")):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Validate new password
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

    # Hash and save
    import bcrypt
    hashed = bcrypt.hashpw(req.new_password.encode(), bcrypt.gensalt()).decode()
    users[username]["password"] = hashed
    creds["users"] = users
    with open(CREDENTIALS_YAML, "w") as fh:
        yaml.dump(creds, fh, default_flow_style=False)

    return {"ok": True}


# ─── Analysis ───────────────────────────────────────────────────────────

@router.post("/analysis/run")
async def analysis_run(req: AnalysisRunRequest):
    """Run the full analysis pipeline for given tickers."""
    try:
        if req.market == "US":
            from scrapers.us_aggregator import USNewsAggregator
        else:
            from scrapers.ind_aggregator import IndianNewsAggregator as USNewsAggregator

        from sentiment import SentimentAnalyzer
        from metrics import MetricsCalculator
        from decision_engine import DecisionEngine

        aggregator = USNewsAggregator()
        analyzer = SentimentAnalyzer()
        calculator = MetricsCalculator()
        engine = DecisionEngine()

        news_items = await asyncio.to_thread(aggregator.get_news, req.tickers)
        analyzed = analyzer.analyze_batch(news_items)

        metrics_map = {}
        for ticker in req.tickers:
            try:
                m = await asyncio.to_thread(calculator.calculate, ticker, req.period)
                metrics_map[ticker] = m
            except Exception:
                metrics_map[ticker] = None

        signals = []
        for item in analyzed:
            ticker = item.get("ticker", "")
            m = metrics_map.get(ticker)
            decision = engine.decide(item, m)
            signals.append({
                "news_item": item,
                "metrics": m,
                "decision": decision.get("decision", "HOLD"),
                "decision_score": decision.get("score", 0),
                "reasoning": decision.get("reasoning", ""),
                "timestamp": item.get("timestamp", ""),
            })

        summary = {"total": len(signals), "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0}
        for s in signals:
            key = s["decision"].lower()
            if key in summary:
                summary[key] += 1

        # Persist to DB if available
        db = get_db_service()
        run_id = None
        if db:
            try:
                run_id = str(db.save_analysis_run(req.market, req.tickers, signals))
            except Exception as e:
                logger.warning("Failed to save analysis run: %s", e)

        return {"run_id": run_id, "signals": signals, "summary": summary}
    except Exception as e:
        logger.error("Analysis error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/latest")
async def analysis_latest(market: str = "US"):
    """Get the most recent analysis run."""
    db = get_db_service()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        result = db.get_latest_analysis(market)
        return result or {"run_id": None, "signals": [], "summary": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/metrics")
async def analysis_metrics(tickers: str, market: str = "US"):
    """Get stock metrics for given tickers."""
    from metrics import MetricsCalculator
    calc = MetricsCalculator()
    results = []
    for ticker in tickers.split(","):
        ticker = ticker.strip()
        if not ticker:
            continue
        try:
            m = await asyncio.to_thread(calc.calculate, ticker)
            if m:
                results.append(m)
        except Exception:
            pass
    return results


# ─── Macro Indicators ────────────────────────────────────────────────────

@router.get("/macro/snapshot")
async def macro_snapshot(market: str = "IND"):
    """Get macro-economic indicators (VIX, yields, commodities)."""
    try:
        from scrapers.macro.macro_indicators import MacroIndicators
        mi = MacroIndicators()
        snap = await asyncio.to_thread(mi.fetch, market=market)
        return {
            "vix": snap.vix if market == "US" else snap.india_vix,
            "vix_label": "CBOE VIX" if market == "US" else "India VIX",
            "index_name": "S&P 500" if market == "US" else "Nifty 50",
            "index_price": snap.sp500_price if market == "US" else snap.nifty50_price,
            "index_change_pct": snap.sp500_change_pct if market == "US" else snap.nifty50_change_pct,
            "us_10y_yield": snap.us_10y_yield,
            "gold_price": snap.gold_price,
            "crude_oil_price": snap.crude_oil_price,
            "macro_sentiment_label": snap.macro_sentiment_label,
            "macro_sentiment_score": snap.macro_sentiment_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/macro/fear-greed")
async def macro_fear_greed():
    """Get India Fear & Greed index."""
    try:
        from scrapers.macro.macro_indicators import MacroIndicators
        from scrapers.macro.india_fear_greed import IndiaFearGreedIndex
        from scrapers.ind_news.fii_dii_flows import FIIDIIFlows

        mi = MacroIndicators()
        snap = await asyncio.to_thread(mi.fetch, market="IND")
        flows = await FIIDIIFlows().fetch()
        fg = IndiaFearGreedIndex()
        result = await fg.compute(
            india_vix=snap.india_vix,
            fii_net_crore=flows.fii_net,
            nifty_change_pct=snap.nifty50_change_pct,
        )
        return {"score": result.score, "label": result.label}
    except Exception as e:
        return {"score": None, "label": "N/A"}


# ─── Backtest ────────────────────────────────────────────────────────────

@router.get("/backtest/strategies")
async def backtest_strategies(market: str = "US"):
    """List available trading strategies."""
    try:
        from strategies import get_available_strategies
        return get_available_strategies(market)
    except ImportError:
        # Fallback: import from the us_stocks or ind_stocks modules
        try:
            ts = get_trading_system()
            return ts.list_strategies() if ts else []
        except Exception:
            return []


@router.post("/backtest/run")
async def backtest_run(req: BacktestRunRequest):
    """Run a strategy backtest."""
    try:
        ts = get_trading_system(tickers=req.tickers)
        if not ts:
            raise HTTPException(status_code=503, detail="Trading system unavailable")

        result = await asyncio.to_thread(
            ts.run_backtest,
            strategy_id=req.strategy_id,
            tickers=req.tickers,
            params=req.params,
            initial_capital=req.initial_capital,
            period=req.period,
            start_date=req.start_date,
            end_date=req.end_date,
        )

        # Persist
        db = get_db_service()
        if db:
            try:
                db.save_backtest_result(req.market, result)
            except Exception as e:
                logger.warning("Failed to save backtest: %s", e)

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Backtest error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{backtest_id}")
async def backtest_get(backtest_id: str):
    """Get a specific backtest result."""
    db = get_db_service()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    result = db.get_backtest_result(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return result


# ─── Verdict ─────────────────────────────────────────────────────────────

@router.post("/verdict/run")
async def verdict_run(req: VerdictRunRequest):
    """Run the multi-layer verdict engine."""
    try:
        from decision_engine import DecisionEngine
        engine = DecisionEngine()

        results = []
        for ticker in req.tickers:
            try:
                result = await asyncio.to_thread(
                    engine.verdict,
                    ticker=ticker,
                    market=req.market,
                    weights=req.weights,
                    skip_layers=req.skip_layers,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Verdict error for %s: %s", ticker, e)

        return results
    except Exception as e:
        logger.error("Verdict error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── History ─────────────────────────────────────────────────────────────

@router.get("/history/signals")
async def history_signals(market: str = "US", page: int = 1, limit: int = 50):
    """Get signal history from the database."""
    db = get_db_service()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        data = db.get_signal_history(market=market, page=page, limit=limit)
        total = db.count_signals(market=market)
        return {"data": data, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/backtests")
async def history_backtests(market: str = "US", page: int = 1, limit: int = 50):
    """Get backtest history from the database."""
    db = get_db_service()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        data = db.get_backtest_history(market=market, page=page, limit=limit)
        total = db.count_backtests(market=market)
        return {"data": data, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Screener ────────────────────────────────────────────────────────────

@router.post("/screener/run")
async def screener_run(req: ScreenerRunRequest):
    """Run the stock screener pipeline."""
    try:
        from kite_connect.trading.screener import NSEScreener
        from kite_connect.trading.risk_manager import RiskManager

        screener = NSEScreener(**req.screener)
        risk_mgr = RiskManager(**req.risk)

        stocks = await asyncio.to_thread(screener.screen, req.tickers or None)
        passed = [s for s in stocks if s.get("passed", False)]
        plans = risk_mgr.generate_trade_plans(passed) if passed else []

        return {"stocks": stocks, "trade_plans": plans, "summary": {"screened": len(stocks), "passed": len(passed)}}
    except Exception as e:
        logger.error("Screener error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screener/execute")
async def screener_execute(req: Dict[str, Any]):
    """Execute trade plans via Kite."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite session not active")
    try:
        from kite_connect.trading.order_manager import OrderManager
        om = OrderManager(kite)
        results = await asyncio.to_thread(om.execute_plans, req.get("plans", []))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/monitor")
async def screener_monitor():
    """Get trade monitor summary."""
    try:
        from kite_connect.trading.trade_monitor import TradeMonitor
        monitor = TradeMonitor()
        return monitor.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Kite Connect ───────────────────────────────────────────────────────

@router.get("/kite/session/status")
async def kite_session_status():
    """Check if a Kite session is currently active."""
    kite = get_kite_session()
    if not kite:
        return {"active": False, "profile": None}
    try:
        profile = await asyncio.to_thread(kite.profile)
        return {"active": True, "profile": profile}
    except Exception:
        return {"active": False, "profile": None}


@router.post("/kite/session/start")
async def kite_session_start():
    """Start a new Kite Connect session.

    First tries the stored request_token.  If expired, attempts auto-login
    via Selenium + TOTP (with a timeout).  If that also fails, returns a
    structured ``needs_login`` response so the frontend can prompt the user
    to complete the OAuth flow manually.
    """
    import concurrent.futures
    from api.dependencies import set_kite_session

    # If already active, return immediately
    existing = get_kite_session()
    if existing:
        try:
            profile = await asyncio.to_thread(existing.profile)
            return {"success": True, "profile": profile, "message": "Session already active"}
        except Exception:
            pass  # session expired, proceed with re-login

    # Step 1: Try stored request_token (fast, no browser)
    try:
        from kite_connect.auth.kite_session import try_stored_token
        kite = await asyncio.to_thread(try_stored_token)
        if kite:
            set_kite_session(kite)
            profile = await asyncio.to_thread(kite.profile)
            return {"success": True, "profile": profile}
    except Exception:
        pass  # token invalid/expired, continue

    # Step 2: Try auto-login with Selenium + TOTP (with timeout)
    try:
        from kite_connect.auth.kite_session import create_kite_session
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            kite = await asyncio.wait_for(
                loop.run_in_executor(pool, create_kite_session),
                timeout=90,
            )
        set_kite_session(kite)
        profile = await asyncio.to_thread(kite.profile)
        return {"success": True, "profile": profile}
    except asyncio.TimeoutError:
        logger.warning("Kite auto-login timed out after 90s")
    except Exception as e:
        logger.warning("Kite auto-login failed: %s", e)

    # Step 3: Return structured response so frontend can prompt manual login
    from kite_connect.core.config import LOGIN_URL
    return {
        "success": False,
        "needs_login": True,
        "login_url": LOGIN_URL,
        "message": "Token expired. Please complete the Kite login and paste the request_token.",
    }


class KiteTokenRequest(BaseModel):
    request_token: str


@router.post("/kite/session/complete")
async def kite_session_complete(body: KiteTokenRequest):
    """Complete a Kite session using a manually-provided request_token.

    Called after the user completes OAuth login and obtains a request_token
    from the Kite redirect URL.
    """
    import os
    from api.dependencies import set_kite_session
    from kiteconnect import KiteConnect

    try:
        from kite_connect.core.config import API_KEY, API_SECRET
        pool_cfg = {"pool_maxsize": int(os.getenv("KITE_POOL_MAXSIZE", "20"))}
        kite = KiteConnect(api_key=API_KEY, pool=pool_cfg)
        data = await asyncio.to_thread(
            kite.generate_session, body.request_token, API_SECRET,
        )
        kite.set_access_token(data["access_token"])
        set_kite_session(kite)

        # Persist the new token so next restart can reuse it
        try:
            from kite_connect.auth.kite_auth import update_kite_app
            update_kite_app(body.request_token)
        except Exception:
            pass

        profile = await asyncio.to_thread(kite.profile)
        return {"success": True, "profile": profile}
    except Exception as e:
        logger.error("Kite session complete failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/kite/session/stop")
async def kite_session_stop():
    """Disconnect the active Kite session."""
    from api.dependencies import set_kite_session

    kite = get_kite_session()
    if kite:
        try:
            await asyncio.to_thread(kite.invalidate_access_token)
        except Exception:
            pass
    set_kite_session(None)
    return {"success": True}


@router.get("/kite/quotes")
async def kite_quotes(symbols: str):
    """Get live quotes for comma-separated symbols."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite session not active")
    try:
        from kite_connect.core.quotes import get_batch_quotes
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        quotes = await asyncio.to_thread(get_batch_quotes, kite, syms)
        return quotes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kite/holdings")
async def kite_holdings():
    """Get portfolio holdings."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite session not active")
    try:
        holdings = await asyncio.to_thread(kite.holdings)
        return holdings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kite/positions")
async def kite_positions():
    """Get current positions."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite session not active")
    try:
        positions = await asyncio.to_thread(kite.positions)
        return positions.get("net", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kite/orders")
async def kite_orders():
    """Get order book."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite session not active")
    try:
        orders = await asyncio.to_thread(kite.orders)
        return orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kite/orders")
async def kite_place_order(order: Dict[str, Any]):
    """Place an order via Kite."""
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite session not active")
    try:
        variety = order.pop("variety", "regular")
        order_id = await asyncio.to_thread(
            kite.place_order,
            variety=variety,
            **order,
        )
        return {"order_id": str(order_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Options ─────────────────────────────────────────────────────────────

@router.get("/options/indices")
async def options_indices():
    """Get index quotes."""
    kite = get_kite_session()
    if not kite:
        # Return static data
        return [
            {"index": "NIFTY 50", "ltp": 0, "change": 0, "change_pct": 0},
            {"index": "BANK NIFTY", "ltp": 0, "change": 0, "change_pct": 0},
        ]
    try:
        from kite_connect.nse.index_data import get_index_quotes
        return await asyncio.to_thread(get_index_quotes, kite)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options/expiries")
async def options_expiries(symbol: str):
    """Get available expiry dates for an option symbol."""
    try:
        from kite_connect.options.chain import get_expiry_dates
        return await asyncio.to_thread(get_expiry_dates, symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options/chain")
async def options_chain(symbol: str, expiry: str):
    """Get option chain for a symbol and expiry."""
    try:
        from kite_connect.options.chain import get_option_chain
        return await asyncio.to_thread(get_option_chain, symbol, expiry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── DriveWealth ─────────────────────────────────────────────────────────

_dw_session: Dict[str, Any] = {}


@router.post("/drivewealth/login")
async def dw_login(req: DWLoginRequest):
    """Login to DriveWealth API."""
    try:
        from services.drivewealth import DriveWealthClient
        client = DriveWealthClient(
            client_id=req.client_id,
            client_secret=req.client_secret,
            app_key=req.app_key,
        )
        token = await asyncio.to_thread(client.authenticate, req.user_id)
        account = await asyncio.to_thread(client.get_account, req.account_id)
        _dw_session["client"] = client
        _dw_session["account_id"] = req.account_id
        return {"token": token, "account": account}
    except ImportError:
        raise HTTPException(status_code=501, detail="DriveWealth module not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drivewealth/account")
async def dw_account():
    """Get DriveWealth account info."""
    client = _dw_session.get("client")
    if not client:
        raise HTTPException(status_code=401, detail="Not connected to DriveWealth")
    try:
        account = await asyncio.to_thread(client.get_account, _dw_session["account_id"])
        return account
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drivewealth/positions")
async def dw_positions():
    """Get DriveWealth positions."""
    client = _dw_session.get("client")
    if not client:
        raise HTTPException(status_code=401, detail="Not connected to DriveWealth")
    try:
        positions = await asyncio.to_thread(client.get_positions, _dw_session["account_id"])
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drivewealth/orders")
async def dw_place_order(req: OrderRequest):
    """Place a DriveWealth order."""
    client = _dw_session.get("client")
    if not client:
        raise HTTPException(status_code=401, detail="Not connected to DriveWealth")
    try:
        result = await asyncio.to_thread(
            client.place_order,
            account_id=_dw_session["account_id"],
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            quantity=req.quantity,
            limit_price=req.limit_price,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Financial ML ────────────────────────────────────────────────────────

@router.get("/fml/chapters")
async def fml_chapters():
    """List available Financial ML chapters."""
    try:
        from financial_ML.applied import get_chapters
        return get_chapters()
    except ImportError:
        # Fallback: scan the readings directory for chapter files
        try:
            from pathlib import Path
            fml_dir = Path(__file__).resolve().parent.parent / "financial_ML" / "readings"
            chapters = []
            if fml_dir.exists():
                for f in sorted(fml_dir.iterdir()):
                    if f.suffix == ".py" and not f.name.startswith("_"):
                        chapters.append({
                            "key": f.stem,
                            "title": f.stem.replace("_", " ").title(),
                            "category": "Readings",
                        })
            return chapters
        except Exception:
            return []


@router.post("/fml/run")
async def fml_run(req: ChapterRunRequest):
    """Run selected Financial ML chapters. Returns a batch_id for progress tracking."""
    import uuid
    batch_id = str(uuid.uuid4())

    # Start async execution
    try:
        from financial_ML.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(batch_id, req.chapters))
    except ImportError:
        logger.warning("FML module not found — run will be a no-op")

    return {"batch_id": batch_id}


@router.get("/fml/progress/{batch_id}")
async def fml_progress(batch_id: str):
    """SSE stream for FML batch progress."""
    async def event_stream():
        try:
            from financial_ML.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                await asyncio.sleep(1)
        except ImportError:
            import json
            yield f"data: {json.dumps({'batch_id': batch_id, 'total': 0, 'completed': 0, 'chapters': {}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/fml/history")
async def fml_history(page: int = 1, limit: int = 50):
    """Get Financial ML batch run history."""
    db = get_db_service()
    if not db:
        return {"data": [], "total": 0}
    try:
        data = db.get_fml_history(page=page, limit=limit)
        total = db.count_fml_runs()
        return {"data": data, "total": total}
    except Exception:
        return {"data": [], "total": 0}


# ─── Test & Tune Trading Systems ─────────────────────────────────────────

@router.get("/tts/chapters")
async def tts_chapters():
    """List available Test & Tune chapters."""
    try:
        from testune_trade_sys.applied import get_chapters
        return get_chapters()
    except ImportError:
        try:
            from pathlib import Path
            tts_dir = Path(__file__).resolve().parent.parent / "testune_trade_sys"
            chapters = []
            if tts_dir.exists():
                for f in sorted(tts_dir.iterdir()):
                    if f.suffix == ".py" and not f.name.startswith("_"):
                        chapters.append({
                            "key": f.stem,
                            "title": f.stem.replace("_", " ").title(),
                            "category": "Trading",
                        })
            return chapters
        except Exception:
            return []


@router.post("/tts/run")
async def tts_run(req: ChapterRunRequest):
    """Run selected Test & Tune chapters."""
    import uuid
    batch_id = str(uuid.uuid4())

    try:
        from testune_trade_sys.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(batch_id, req.chapters))
    except ImportError:
        logger.warning("TTS module not found — run will be a no-op")

    return {"batch_id": batch_id}


@router.get("/tts/progress/{batch_id}")
async def tts_progress(batch_id: str):
    """SSE stream for TTS batch progress."""
    async def event_stream():
        try:
            from testune_trade_sys.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                await asyncio.sleep(1)
        except ImportError:
            import json
            yield f"data: {json.dumps({'batch_id': batch_id, 'total': 0, 'completed': 0, 'chapters': {}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/tts/history")
async def tts_history(page: int = 1, limit: int = 50):
    """Get Test & Tune batch run history."""
    db = get_db_service()
    if not db:
        return {"data": [], "total": 0}
    try:
        data = db.get_tts_history(page=page, limit=limit)
        total = db.count_tts_runs()
        return {"data": data, "total": total}
    except Exception:
        return {"data": [], "total": 0}


# ─── RAG Pipeline ────────────────────────────────────────────────────────

@router.get("/rag/sources")
async def rag_sources():
    """List knowledge base sources."""
    engine = get_rag_engine()
    if not engine:
        return []
    try:
        names = await asyncio.to_thread(engine._vs.list_sources)
        results = []
        for name in names:
            details = await asyncio.to_thread(engine._vs.get_source_details, name)
            results.append({
                "id": name,
                "name": name,
                "type": name.rsplit(".", 1)[-1] if "." in name else "pdf",
                "doc_count": 1,
                "chunk_count": details.get("chunks", 0),
                "page_count": details.get("page_count"),
                "ingested_at": details.get("ingested_at"),
                "file_size_bytes": details.get("file_size_bytes"),
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/upload")
async def rag_upload(files: List[UploadFile] = File(...)):
    """Upload and ingest documents asynchronously in background threads."""
    from rag_pipeline.ingestion.background_ingest import get_ingestion_manager

    mgr = get_ingestion_manager()
    tasks = []
    for file in files:
        content = await file.read()
        task = mgr.submit(file.filename or "unknown", content)
        tasks.append({"task_id": task.task_id, "file_name": task.file_name, "status": task.status.value})
    return {"submitted": len(tasks), "tasks": tasks}


@router.get("/rag/ingest-status")
async def rag_ingest_status():
    """Poll ingestion task status for all active and recently completed tasks."""
    from rag_pipeline.ingestion.background_ingest import get_ingestion_manager

    mgr = get_ingestion_manager()
    active = mgr.get_active_tasks()
    recent = mgr.get_recently_completed(max_age_s=300)
    all_tasks = active + recent
    return [
        {
            "task_id": t.task_id,
            "file_name": t.file_name,
            "status": t.status.value,
            "stage": t.stage,
            "stage_pct": t.stage_pct,
            "error": t.error,
        }
        for t in all_tasks
    ]


@router.delete("/rag/sources/{source_id}")
async def rag_delete_source(source_id: str):
    """Delete a document source from the knowledge base."""
    engine = get_rag_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="RAG engine unavailable")
    try:
        from rag_pipeline.ingestion.pdf_ingestion import PDFIngestionService

        svc = PDFIngestionService(
            vector_store=engine._vs,
            config=engine._config,
            embedding_service=engine._embedder,
            on_change_callback=engine.invalidate_cache,
        )
        deleted = await asyncio.to_thread(svc.delete_source, source_id)
        return {"deleted": True, "chunks_removed": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/query")
async def rag_query(
    q: str,
    rag: str = "true",
    sources: str = "",
    token: str = "",
):
    """SSE streaming RAG query with real token-by-token LLM output.

    Uses ``query_stream()`` — the same streaming pipeline as the
    Streamlit UI — so retrieval, context-building, and LLM generation
    are identical.
    """
    engine = get_rag_engine()
    rag_enabled = rag.lower() == "true"
    source_ids = [s for s in sources.split(",") if s] if sources else None

    async def event_stream():
        import json

        if not engine or not rag_enabled:
            try:
                from rag_pipeline.llm import get_llm_response
                response = await asyncio.to_thread(get_llm_response, q)
                yield f"event: token\ndata: {json.dumps(response)}\n\n"
                yield f"event: done\ndata: done\n\n"
            except Exception as e:
                yield f"event: token\ndata: {json.dumps(f'Error: {e}')}\n\n"
                yield f"event: done\ndata: done\n\n"
            return

        try:
            source_filter = source_ids[0] if source_ids and len(source_ids) == 1 else None

            # Use query_stream() — real token-by-token streaming,
            # identical pipeline to Streamlit UI.
            stream_gen = engine.query_stream(q, source_filter=source_filter)

            # query_stream() is a blocking generator; iterate in a
            # thread so we don't block the asyncio event loop.
            import queue, threading

            token_queue: queue.Queue = queue.Queue()
            _SENTINEL = object()

            def _run_stream():
                try:
                    for tok in stream_gen:
                        token_queue.put(tok)
                except Exception as exc:
                    token_queue.put(exc)
                finally:
                    token_queue.put(_SENTINEL)

            thread = threading.Thread(target=_run_stream, daemon=True)
            thread.start()

            while True:
                # Wait for the next token (with a generous timeout
                # to cover model-loading / prompt-eval pauses).
                try:
                    item = await asyncio.to_thread(token_queue.get, True, 300)
                except Exception:
                    break

                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    yield f"event: token\ndata: {json.dumps(f'Error: {item}')}\n\n"
                    break

                yield f"event: token\ndata: {json.dumps(item)}\n\n"

            yield f"event: done\ndata: done\n\n"
        except Exception as e:
            logger.exception("RAG query SSE error")
            yield f"event: token\ndata: {json.dumps(f'Error: {e}')}\n\n"
            yield f"event: done\ndata: done\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Market Ticker Prices ────────────────────────────────────────────────

_ticker_price_cache: Dict[str, Any] = {}   # cache_key -> response dict
_ticker_cache_ts: Dict[str, float] = {}    # cache_key -> epoch timestamp
_TICKER_CACHE_TTL_OPEN = 10    # seconds – during market hours
_TICKER_CACHE_TTL_CLOSED = 120 # seconds – after market close

def _is_market_open(market: str) -> bool:
    """Check if the stock market is currently open."""
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    if market == "IND":
        # NSE: 9:15 AM – 3:30 PM IST (UTC+5:30), Mon–Fri
        ist = now_utc + timedelta(hours=5, minutes=30)
        if ist.weekday() >= 5:
            return False
        t = ist.hour * 60 + ist.minute
        return 9 * 60 + 15 <= t < 15 * 60 + 30
    else:
        # NYSE/NASDAQ: 9:30 AM – 4:00 PM ET (approx UTC-4/-5)
        # Use UTC-4 (EDT) as a safe approximation
        et = now_utc - timedelta(hours=4)
        if et.weekday() >= 5:
            return False
        t = et.hour * 60 + et.minute
        return 9 * 60 + 30 <= t < 16 * 60


@router.get("/market/ticker-prices")
async def market_ticker_prices(symbols: str, market: str = "US"):
    """Get current/last-traded prices for comma-separated ticker symbols."""
    import time
    try:
        import yfinance as yf
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        if not syms:
            return {"is_market_open": _is_market_open(market), "prices": []}

        # ── cache lookup ──
        cache_key = f"{market}:{',' .join(sorted(syms))}"
        now = time.monotonic()
        is_open = _is_market_open(market)
        ttl = _TICKER_CACHE_TTL_OPEN if is_open else _TICKER_CACHE_TTL_CLOSED

        if cache_key in _ticker_price_cache and (now - _ticker_cache_ts.get(cache_key, 0)) < ttl:
            return _ticker_price_cache[cache_key]

        # For IND market, append .NS suffix for NSE (with override map)
        if market == "IND":
            from utils import yf_nse_symbol
            yf_syms = [yf_nse_symbol(s) for s in syms]
        else:
            yf_syms = list(syms)

        def _fetch():
            # yf.download is ~4x faster than yf.Tickers for batch fetches
            df = yf.download(
                " ".join(yf_syms),
                period="2d",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
            results = []
            multi = len(yf_syms) > 1
            for orig, yf_sym in zip(syms, yf_syms):
                try:
                    close = df[yf_sym]["Close"].dropna() if multi else df["Close"].dropna()
                    if len(close) >= 2:
                        price, prev = float(close.iloc[-1]), float(close.iloc[-2])
                    elif len(close) == 1:
                        price, prev = float(close.iloc[-1]), float(close.iloc[-1])
                    else:
                        price, prev = 0, 0
                    change_pct = ((price - prev) / prev * 100) if prev else 0
                    results.append({"symbol": orig, "price": round(price, 2), "change_pct": round(change_pct, 2)})
                except Exception:
                    results.append({"symbol": orig, "price": 0, "change_pct": 0})
            return results

        prices = await asyncio.to_thread(_fetch)
        result = {"is_market_open": is_open, "prices": prices}

        # ── populate cache ──
        _ticker_price_cache[cache_key] = result
        _ticker_cache_ts[cache_key] = now

        return result
    except Exception as e:
        logger.error(f"ticker-prices error: {e}")
        # Return stale cache on error if available
        if cache_key in _ticker_price_cache:
            return _ticker_price_cache[cache_key]
        raise HTTPException(status_code=500, detail=str(e))
