"""
API v1 Gateway Router — maps Next.js frontend paths to existing service logic.

This router provides the /api/v1/* endpoints expected by the Next.js frontend,
delegating to existing internal modules. Missing functionality (DriveWealth,
FML, TTS chapters) is stubbed with minimal implementations.
"""

import asyncio
import logging
import math
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_db_service, get_kite_session, get_rag_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["API v1"])


def _sanitize_floats(obj):
    """Replace inf/nan floats with None so JSON serialization doesn't fail."""
    if isinstance(obj, float):
        return None if math.isinf(obj) or math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _metrics_to_dict(m) -> dict:
    """Convert a StockMetrics dataclass to a JSON-safe dict."""
    from dataclasses import asdict
    d = asdict(m)
    if d.get("timestamp"):
        d["timestamp"] = d["timestamp"].isoformat() if hasattr(d["timestamp"], "isoformat") else str(d["timestamp"])
    return _sanitize_floats(d)


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
    tickers: Optional[List[str]] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None


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
    from api.auth import authenticate_user_async, create_session_token
    ok, display_name, role = await authenticate_user_async(req.username, req.password)
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

    # Invalidate the in-memory credential cache so login uses the new hash
    from api.auth import invalidate_credentials_cache
    invalidate_credentials_cache()

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

        from services.sentiment import SentimentAnalyzer
        from services.metrics import MetricsCalculator
        from services.decision_engine import DecisionEngine

        aggregator = USNewsAggregator()
        analyzer = SentimentAnalyzer()
        calculator = MetricsCalculator()
        engine = DecisionEngine()

        news_items = await aggregator.fetch_news_for_tickers(req.tickers)
        analyzed = await asyncio.to_thread(analyzer.analyze_news_items, news_items)

        # Indian tickers need .NS suffix for yfinance / metrics lookups
        if req.market == "IND":
            from utils import yf_nse_symbol
            yf_tickers = {t: yf_nse_symbol(t) for t in req.tickers}
        else:
            yf_tickers = {t: t for t in req.tickers}

        metrics_map = {}
        for ticker in req.tickers:
            try:
                m = await asyncio.to_thread(calculator.get_stock_metrics, yf_tickers[ticker])
                metrics_map[ticker] = m
            except Exception:
                metrics_map[ticker] = None

        signals = []
        for item in analyzed:
            m = metrics_map.get(item.ticker)
            sig = engine.generate_signal(item, m)
            ni = sig.news_item
            signals.append({
                "news_item": {
                    "title": ni.title,
                    "summary": ni.summary,
                    "url": ni.url,
                    "timestamp": ni.timestamp.isoformat() if hasattr(ni.timestamp, "isoformat") else str(ni.timestamp),
                    "source": ni.source,
                    "ticker": ni.ticker,
                    "category": ni.category.value if ni.category else "general",
                    "sentiment_score": ni.sentiment_score,
                    "sentiment_label": ni.sentiment_label.value if ni.sentiment_label else None,
                    "sentiment_confidence": ni.sentiment_confidence,
                },
                "metrics": _metrics_to_dict(sig.metrics) if sig.metrics else None,
                "decision": sig.decision.value,
                "decision_score": sig.decision_score,
                "reasoning": sig.reasoning,
                "timestamp": sig.timestamp.isoformat() if hasattr(sig.timestamp, "isoformat") else str(sig.timestamp),
            })

        summary = {"total": len(signals), "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0}
        for s in signals:
            key = s.get("decision", "hold").lower()
            if key in summary:
                summary[key] += 1

        # Persist to DB if available
        db = get_db_service()
        run_id = None
        if db:
            try:
                run_id = db.start_analysis_run(
                    run_type="stock_analysis",
                    tickers=req.tickers,
                    market=req.market,
                )
                if run_id:
                    db.save_signals(signals, analysis_run_id=run_id, market=req.market)
                    run_id = str(run_id)
            except Exception as e:
                logger.warning("Failed to save analysis run: %s", e)

        return _sanitize_floats({"run_id": run_id, "signals": signals, "summary": summary})
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
    from services.metrics import MetricsCalculator
    calc = MetricsCalculator()
    results = []
    for ticker in tickers.split(","):
        ticker = ticker.strip()
        if not ticker:
            continue
        try:
            m = await asyncio.to_thread(calc.get_stock_metrics, ticker)
            if m:
                results.append(_metrics_to_dict(m))
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


@router.get("/macro/portfolio-risk")
async def macro_portfolio_risk(market: str = "IND"):
    """Get portfolio risk snapshot (drawdown, vol, concentration)."""
    try:
        from services.portfolio_vol_monitor import assess_portfolio_risk

        # Attempt to gather live position data from Kite (IND) or DriveWealth (US)
        position_values: dict = {}
        instrument_vols: dict = {}
        total_capital = 500_000.0
        peak_equity = None

        if market == "IND":
            try:
                from auth.shared_session import get_kite
                kite = get_kite()
                if kite:
                    positions = kite.positions().get("net", [])
                    for p in positions:
                        sym = p.get("tradingsymbol", "")
                        qty = p.get("quantity", 0)
                        ltp = p.get("last_price", 0)
                        if qty != 0 and ltp > 0:
                            position_values[sym] = abs(qty * ltp)
                            instrument_vols[sym] = 0.02  # ~32% annual vol default
            except Exception:
                pass

        snap = await asyncio.to_thread(
            assess_portfolio_risk,
            position_values=position_values,
            instrument_daily_vols=instrument_vols,
            total_capital=total_capital,
            peak_equity=peak_equity,
        )
        return {
            "timestamp": snap.timestamp,
            "portfolio_daily_vol": snap.portfolio_daily_vol,
            "portfolio_annual_vol_pct": snap.portfolio_annual_vol_pct,
            "target_annual_vol_pct": snap.target_annual_vol_pct,
            "vol_ratio": snap.vol_ratio,
            "hhi": snap.hhi,
            "largest_position_pct": snap.largest_position_pct,
            "peak_equity": snap.peak_equity,
            "current_equity": snap.current_equity,
            "drawdown_pct": snap.drawdown_pct,
            "risk_level": snap.risk_level.value if hasattr(snap.risk_level, 'value') else str(snap.risk_level),
            "scale_factor": snap.scale_factor,
            "emergency_liquidate": snap.emergency_liquidate,
            "alerts": snap.alerts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Backtest ────────────────────────────────────────────────────────────

@router.get("/backtest/strategies")
async def backtest_strategies(market: str = "US"):
    """List available trading strategies (lightweight, no heavy imports)."""
    try:
        from trading_strategies import list_strategies, get_strategy

        result = []
        for info in list_strategies():
            try:
                strategy_cls = get_strategy(info["id"])
                params_raw = strategy_cls.get_parameters() if strategy_cls else {}
                params = [
                    {"name": k, **v}
                    for k, v in params_raw.items()
                ]
            except Exception:
                params = []
            result.append({
                "id": info["id"],
                "name": info.get("name", info["id"]),
                "category": info.get("category", "general"),
                "description": info.get("description", ""),
                "parameters": params,
            })

        return result
    except Exception as e:
        logger.error("Strategy listing error: %s", e, exc_info=True)
        return []


@router.post("/backtest/run")
async def backtest_run(req: BacktestRunRequest):
    """Run a strategy backtest using the strategy registry directly."""
    import uuid as _uuid
    from datetime import datetime, timedelta
    from trading_strategies import get_strategy

    try:
        # ── Normalise Indian tickers to yfinance format (.NS) ──
        # Raw NSE symbols (e.g. "SBIN", "MARUTI") fail on Yahoo Finance
        # without the .NS suffix.  US tickers are left untouched.
        if req.market == "IND":
            from utils import yf_nse_symbol
            req.tickers = [
                yf_nse_symbol(t) if not t.upper().endswith((".NS", ".BO")) else t
                for t in req.tickers
            ]

        strategy_cls = get_strategy(req.strategy_id)
        if strategy_cls is None:
            raise HTTPException(status_code=404, detail=f"Strategy '{req.strategy_id}' not found")

        # Resolve date range: explicit dates take priority, else derive from period
        end_date = req.end_date
        start_date = req.start_date
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            period_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730, "5y": 1825}
            days = period_map.get(req.period, 365)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Build kwargs: strategy-specific params + required run() args
        run_kwargs = {
            "tickers": req.tickers,
            "start_date": start_date,
            "end_date": end_date,
            "capital": req.initial_capital,
            **req.params,
        }

        strategy = strategy_cls()
        result = await asyncio.to_thread(strategy.run, **run_kwargs)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message or "Strategy execution failed")

        # Extract flat metrics (result.metrics may be nested by ticker)
        metrics = result.metrics or {}

        # Detect per-ticker nesting: {"AAPL": {sharpe: ...}, "MSFT": {...}}
        ticker_metrics = {}
        flat_metrics = {}
        for t in req.tickers:
            if t in metrics and isinstance(metrics[t], dict):
                ticker_metrics[t] = metrics[t]

        if ticker_metrics:
            # Aggregate across all tickers for top-level summary
            agg_keys = ["total_return", "sharpe_ratio", "sortino_ratio",
                        "max_drawdown", "total_trades", "win_rate", "final_value"]
            n = len(ticker_metrics)
            agg: dict = {}
            for key in agg_keys:
                vals = [float(tm.get(key, 0)) for tm in ticker_metrics.values()]
                if key == "total_trades":
                    agg[key] = int(sum(vals))
                elif key == "max_drawdown":
                    agg[key] = min(vals)          # worst drawdown
                elif key == "final_value":
                    agg[key] = sum(vals)           # total portfolio value
                else:
                    agg[key] = sum(vals) / n       # average
            m = agg
        else:
            m = metrics
            ticker_metrics = {}

        # Build equity_curve from portfolio DataFrame
        equity_curve = []
        if result.portfolio is not None and not result.portfolio.empty:
            df = result.portfolio
            date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
            value_col = next((c for c in df.columns if c.lower() in ("value", "portfolio_value", "equity", "total")), None)
            dd_col = next((c for c in df.columns if "drawdown" in c.lower()), None)
            if date_col and value_col:
                for _, row in df.iterrows():
                    equity_curve.append({
                        "date": str(row[date_col])[:10],
                        "value": float(row[value_col]),
                        "drawdown": float(row[dd_col]) if dd_col else 0.0,
                    })

        # Build signals list from signals DataFrame
        signals = []
        if result.signals is not None and not result.signals.empty:
            df = result.signals
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                signals.append({
                    "date": str(row_dict.get("date", row_dict.get("datetime", "")))[:10],
                    "ticker": str(row_dict.get("ticker", row_dict.get("symbol", ""))),
                    "signal": str(row_dict.get("signal", row_dict.get("action", ""))),
                    "price": float(row_dict.get("price", row_dict.get("close", 0))),
                    "quantity": int(row_dict.get("quantity", row_dict.get("qty", 0))),
                })

        # Build charts list
        charts = []
        for c in (result.charts or []):
            charts.append({
                "type": c.chart_type,
                "data": c.data,
                "title": c.title,
            })

        response = {
            "id": _uuid.uuid4().hex,
            "strategy_id": req.strategy_id,
            "strategy_name": getattr(strategy_cls, "name", req.strategy_id),
            "tickers": req.tickers,
            "start_date": start_date,
            "end_date": end_date,
            "total_return": float(m.get("total_return", 0)),
            "sharpe_ratio": float(m.get("sharpe_ratio", 0)),
            "sortino_ratio": float(m.get("sortino_ratio", 0)),
            "max_drawdown": float(m.get("max_drawdown", 0)),
            "total_trades": int(m.get("total_trades", 0)),
            "win_rate": float(m.get("win_rate", 0)),
            "final_value": float(m.get("final_value", req.initial_capital)),
            "initial_capital": req.initial_capital,
            "charts": charts,
            "signals": signals,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "per_ticker": {
                t: {
                    "total_return": float(tm.get("total_return", 0)),
                    "sharpe_ratio": float(tm.get("sharpe_ratio", 0)),
                    "sortino_ratio": float(tm.get("sortino_ratio", 0)),
                    "max_drawdown": float(tm.get("max_drawdown", 0)),
                    "total_trades": int(tm.get("total_trades", 0)),
                    "win_rate": float(tm.get("win_rate", 0)),
                    "final_value": float(tm.get("final_value", 0)),
                }
                for t, tm in ticker_metrics.items()
            },
            "created_at": datetime.now().isoformat(),
        }

        # Persist to database
        db = get_db_service()
        if db:
            try:
                db.save_backtest_result(result=response, market=req.market)
            except Exception as e:
                logger.warning("Failed to save backtest: %s", e)

        # Upload charts to R2 / MinIO object storage (non-blocking)
        if result.charts:
            try:
                from services.storage.minio_service import get_minio_service
                minio_svc = get_minio_service()
                if minio_svc.is_available:
                    saved = await asyncio.to_thread(
                        minio_svc.save_backtest_charts,
                        run_id=response["id"],
                        charts=result.charts,
                        strategy_name=response.get("strategy_name", req.strategy_id),
                    )
                    logger.info("Saved %d chart(s) to R2 for backtest %s",
                                len(saved) if saved else 0, response["id"])
            except Exception as e:
                logger.warning("R2 chart upload failed (non-fatal): %s", e)

        return response
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
    """Run the multi-layer verdict engine via IntegratedScorer.

    Checks the scheduler verdict cache first; only tickers with a cache
    miss are evaluated live (saves 60-90s per cached ticker).
    """
    try:
        from scheduler import get_cached_verdict
        from services.integrated_scorer import IntegratedScorer

        # --- Serve cached verdicts where available ---
        # Gap C fix: validate cache age — reject entries older than 30 min
        cached_results = []
        uncached_tickers = []
        _MAX_CACHE_AGE_MIN = 30
        for t in req.tickers:
            cached = get_cached_verdict(t)
            if cached:
                try:
                    from datetime import datetime as _dt
                    cached_at = _dt.fromisoformat(cached.get("cached_at", ""))
                    age_min = (_dt.now(cached_at.tzinfo) - cached_at).total_seconds() / 60
                    if age_min > _MAX_CACHE_AGE_MIN:
                        logger.info(
                            "Verdict cache stale for %s (%.0f min old) — re-evaluating",
                            t, age_min,
                        )
                        uncached_tickers.append(t)
                        continue
                except Exception:
                    pass  # if cached_at missing/invalid, still serve the cache
                cached_results.append(cached)
            else:
                uncached_tickers.append(t)

        # --- Evaluate only the cache-miss tickers ---
        live_results = []
        if uncached_tickers:
            scorer = IntegratedScorer(weights=req.weights)
            date_range = tuple(req.date_range) if req.date_range and req.date_range[0] else None

            try:
                verdicts = await asyncio.wait_for(
                    asyncio.to_thread(
                        scorer.evaluate,
                        tickers=uncached_tickers,
                        market=req.market,
                        date_range=date_range,
                        skip_layers=req.skip_layers,
                    ),
                    timeout=540,  # 9-minute hard cap
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Verdict timed out after 9 minutes")

            for v in verdicts:
                ls = v.layer_scores or {}
                live_results.append({
                    "ticker": v.ticker,
                    "core_score": ls.get("core", 0) or 0,
                    "strategy_score": ls.get("strategy", 0) or 0,
                    "ml_score": ls.get("ml_features", 0) or 0,
                    "rl_score": ls.get("rl_bot", 0) or 0,
                    "robustness_score": ls.get("robustness", 0) or 0,
                    "weighted_score": v.final_score,
                    "verdict": v.classification,
                    "layer_details": v.layer_details,
                    "strategy_breakdown": v.layer_details.get("strategy", {}),
                })

        return cached_results + live_results
    except HTTPException:
        raise
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
        from kite_connect.nse.screener import NSEScreener, ScreenerConfig
        from kite_connect.trading.risk_manager import RiskManager, RiskConfig
        from kite_connect.core.config import INDEX_CONSTITUENTS

        # Map frontend field names → ScreenerConfig field names
        scfg = req.screener
        screener_cfg = ScreenerConfig(
            min_price=scfg.get("min_price", 100),
            min_avg_volume=int(scfg.get("min_avg_volume", 500_000)),
            min_beta=scfg.get("min_beta", 1.0),
            max_workers=int(scfg.get("workers", 8)),
            breakout_vol_mult=scfg.get("volume_multiplier", 1.5),
            history_days=int(scfg.get("lookback_days", 250)),
            index_mode=scfg.get("index_mode", False),
        )

        # Map frontend field names → RiskConfig field names
        rcfg = req.risk
        risk_cfg = RiskConfig(
            total_capital=rcfg.get("total_capital", 500_000),
            max_open_trades=int(rcfg.get("max_open_trades", 6)),
            risk_per_trade_pct=rcfg.get("risk_per_trade_pct", 2) / 100,  # UI sends 2 → config wants 0.02
            min_rr_ratio=rcfg.get("min_rr_ratio", 2.5),
            sl_method=rcfg.get("stop_loss_method", "tighter"),
        )

        screener = NSEScreener(config=screener_cfg)
        risk_mgr = RiskManager(config=risk_cfg)

        # Carver: inject VolatilityTarget when enabled
        try:
            from config import Config as _Cfg
            if getattr(_Cfg, "CARVER_ENABLED", False):
                from services.volatility_target import VolatilityTarget, VolatilityTargetConfig
                vt = VolatilityTarget(VolatilityTargetConfig(
                    initial_capital=risk_cfg.total_capital,
                    annual_vol_target_pct=getattr(_Cfg, "CARVER_ANNUAL_VOL_TARGET", 0.20),
                ))
                risk_mgr = RiskManager(config=risk_cfg, volatility_target=vt)
        except Exception:
            pass

        # Default to NIFTY50 when no tickers provided
        tickers = req.tickers if req.tickers else list(INDEX_CONSTITUENTS.get("NIFTY50", []))

        df = await asyncio.to_thread(screener.screen, tickers)
        stocks_raw = df.to_dict("records") if not df.empty else []

        # Map backend field names → frontend expectations
        stocks = []
        for s in stocks_raw:
            stocks.append({
                **s,
                "ticker": s.get("symbol", ""),
                "price": s.get("close", 0),
                "passed": True,  # all returned stocks passed Stage 1
            })

        plans = risk_mgr.plan_trades(df) if stocks else []
        plan_dicts = []
        for p in plans:
            d = p.to_dict()
            plan_dicts.append({
                **d,
                "ticker": d.get("symbol", ""),
                "risk": d.get("risk_amount", 0),
                "reward": d.get("reward_amount", 0),
            })

        return {"stocks": stocks, "trade_plans": plan_dicts, "summary": {"screened": len(stocks), "passed": len(stocks)}}
    except Exception as e:
        logger.error("Screener error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screener/execute")
async def screener_execute(req: Dict[str, Any]):
    """Execute trade plans via Kite.

    SAFETY: Requires IntegratedScorer verdicts before order placement.
    Only BUY/STRONG_BUY symbols are forwarded to the order manager.
    """
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite session not active")
    try:
        plans = req.get("plans", [])
        if not plans:
            return {"orders": [], "message": "No plans provided"}

        # ── Verdict enforcement: score all symbols first ──
        from services.integrated_scorer import IntegratedScorer
        from datetime import date, timedelta

        symbols = list({p.get("symbol", "") for p in plans if p.get("symbol")})
        ns_tickers = [f"{s}.NS" for s in symbols]

        scorer = IntegratedScorer()
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365)
        verdicts = scorer.evaluate(
            tickers=ns_tickers, market="IND",
            date_range=(str(start_dt), str(end_dt)),
            skip_layers=["rag"],
        )

        buy_tags = {"BUY", "STRONG_BUY"}
        approved = {
            v.ticker.replace(".NS", "").replace(".BO", "")
            for v in verdicts if v.classification in buy_tags
        }

        # Filter plans to only approved symbols
        filtered_plans = [p for p in plans if p.get("symbol") in approved]
        blocked = len(plans) - len(filtered_plans)
        if blocked > 0:
            logger.info("Verdict filter blocked %d/%d plans (non-BUY)", blocked, len(plans))

        if not filtered_plans:
            return {"orders": [], "message": f"No plans passed verdict filter ({blocked} blocked)"}

        from kite_connect.trading.order_service import place_order

        order_results = []
        for plan in filtered_plans:
            try:
                res = await asyncio.to_thread(
                    place_order,
                    kite,
                    symbol=plan.get("symbol", ""),
                    exchange=plan.get("exchange", "NSE"),
                    transaction_type=plan.get("transaction_type", "BUY"),
                    quantity=int(plan.get("quantity", 0)),
                    order_type=plan.get("order_type", "MARKET"),
                    product=plan.get("product", "CNC"),
                    price=plan.get("price"),
                    trigger_price=plan.get("trigger_price"),
                )
                order_results.append(res)
            except Exception as e:
                order_results.append({"success": False, "error": str(e), "symbol": plan.get("symbol")})
        return {
            "orders": order_results,
            "verdict_blocked": blocked,
            "total_placed": len(filtered_plans),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/monitor")
async def screener_monitor():
    """Get trade monitor summary."""
    try:
        from kite_connect.trading.trade_monitor import TradeMonitor
        monitor = TradeMonitor()
        return monitor.summary()
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
    from api.dependencies import set_kite_session, is_kite_token_expiring_soon

    # If already active and not expiring soon, return immediately
    existing = get_kite_session()
    if existing and not is_kite_token_expiring_soon():
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
    # Skip Selenium on containerised environments (HF Spaces, Docker) where
    # no browser is available — use HTTP-based login instead.
    _in_container = os.path.exists("/.dockerenv") or os.getenv("SPACE_ID")
    if _in_container:
        # Step 2a: HTTP-based login (no browser needed, ~3-5s)
        logger.info("Container detected — using HTTP-based Kite login")
        try:
            from kite_connect.auth.kite_session import http_login_kite
            kite = await asyncio.to_thread(http_login_kite)
            if kite:
                set_kite_session(kite)
                profile = await asyncio.to_thread(kite.profile)
                return {"success": True, "profile": profile}
            logger.warning("HTTP-based Kite login returned None")
        except Exception as e:
            logger.warning("HTTP-based Kite login failed: %s", e)
    else:
        # Step 2b: Selenium + TOTP auto-login (local dev with browser)
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


@router.get("/kite/session/status")
async def kite_session_status():
    """Return Kite session health: active, remaining time, expiring flag."""
    from api.dependencies import is_kite_token_expiring_soon, kite_token_remaining_seconds

    kite = get_kite_session()
    if not kite:
        return {"active": False, "remaining_seconds": 0, "expiring_soon": True}

    remaining = kite_token_remaining_seconds()
    expiring = is_kite_token_expiring_soon()
    return {
        "active": True,
        "remaining_seconds": remaining,
        "remaining_minutes": remaining // 60,
        "expiring_soon": expiring,
    }


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


@router.get("/kite/portfolio/pnl")
async def kite_portfolio_pnl():
    """Real-time portfolio P&L summary.

    Aggregates net positions and holdings into a single P&L view
    with total invested, current value, unrealised P&L, and
    day change.
    """
    kite = get_kite_session()
    if not kite:
        raise HTTPException(status_code=503, detail="Kite session not active")
    try:
        positions_data = await asyncio.to_thread(kite.positions)
        holdings_data = await asyncio.to_thread(kite.holdings)

        net_positions = positions_data.get("net", [])
        total_pnl = 0.0
        total_invested = 0.0
        total_current = 0.0
        day_pnl = 0.0
        position_details = []

        for p in net_positions:
            qty = p.get("quantity", 0)
            if qty == 0:
                continue
            avg = p.get("average_price", 0)
            ltp = p.get("last_price", 0)
            pnl = p.get("pnl", 0)
            day_m2m = p.get("day_m2m", 0)
            invested = abs(qty) * avg
            current = abs(qty) * ltp

            total_pnl += pnl
            total_invested += invested
            total_current += current
            day_pnl += day_m2m

            position_details.append({
                "symbol": p.get("tradingsymbol", ""),
                "quantity": qty,
                "avg_price": round(avg, 2),
                "ltp": round(ltp, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round((pnl / invested * 100) if invested else 0, 2),
                "day_change": round(day_m2m, 2),
            })

        # Add holdings (CNC delivery positions)
        for h in (holdings_data or []):
            qty = h.get("quantity", 0)
            if qty == 0:
                continue
            avg = h.get("average_price", 0)
            ltp = h.get("last_price", 0)
            pnl = h.get("pnl", 0)
            day_change = h.get("day_change", 0)
            invested = qty * avg
            current = qty * ltp

            total_pnl += pnl
            total_invested += invested
            total_current += current
            day_pnl += day_change * qty

            position_details.append({
                "symbol": h.get("tradingsymbol", ""),
                "quantity": qty,
                "avg_price": round(avg, 2),
                "ltp": round(ltp, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round((pnl / invested * 100) if invested else 0, 2),
                "day_change": round(day_change * qty, 2),
                "holding": True,
            })

        return {
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(
                (total_pnl / total_invested * 100) if total_invested else 0, 2
            ),
            "day_pnl": round(day_pnl, 2),
            "positions": position_details,
        }
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
        token = await asyncio.to_thread(client.authenticate)
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
        positions = await asyncio.to_thread(client.list_positions, _dw_session["account_id"])
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
        payload = {
            "accountNo": _dw_session["account_id"],
            "symbol": req.symbol,
            "side": req.side,
            "type": req.order_type,
            "quantity": str(req.quantity),
        }
        if req.limit_price is not None:
            payload["price"] = str(req.limit_price)
        result = await asyncio.to_thread(client.create_order, payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drivewealth/carver-orders")
async def dw_carver_orders(tickers: Optional[list] = None):
    """Generate Carver vol-targeted trade plans and optionally place via DriveWealth.

    Runs the full Carver pipeline (EWMAC → forecast → vol-target → sizing)
    on the requested US tickers, then submits BUY orders through DriveWealth.
    If DriveWealth is not connected, returns dry-run plans only.
    """
    try:
        from config import Config
        if not getattr(Config, "CARVER_US_ENABLED", False):
            raise HTTPException(status_code=400, detail="Carver US is not enabled")

        from services.us_carver_pipeline import run_us_carver_pipeline, DEFAULT_US_CARVER_TICKERS

        syms = tickers or DEFAULT_US_CARVER_TICKERS
        result = await asyncio.to_thread(run_us_carver_pipeline, syms)

        placed_orders = []
        client = _dw_session.get("client")
        account_id = _dw_session.get("account_id")

        if client and account_id and client.is_authenticated:
            for plan in result.trade_plans:
                try:
                    order_payload = {
                        "accountNo": account_id,
                        "symbol": plan["symbol"],
                        "side": "BUY",
                        "type": "MARKET",
                        "quantity": str(plan["quantity"]),
                    }
                    order_result = await asyncio.to_thread(client.create_order, order_payload)
                    placed_orders.append({
                        "symbol": plan["symbol"],
                        "quantity": plan["quantity"],
                        "status": "placed",
                        "order_id": order_result.get("orderID", ""),
                    })
                except Exception as exc:
                    placed_orders.append({
                        "symbol": plan["symbol"],
                        "quantity": plan["quantity"],
                        "status": "failed",
                        "error": str(exc),
                    })

        return {
            "success": True,
            "trade_plans": result.trade_plans,
            "orders_placed": placed_orders,
            "dry_run": not bool(client and account_id),
            "pipeline_log": result.pipeline_log,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Financial ML ────────────────────────────────────────────────────────

@router.get("/fml/chapters")
async def fml_chapters():
    """List available Financial ML chapters."""
    try:
        from references.financial_ml.applied import get_chapters
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
        from references.financial_ml.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(
            batch_id, req.chapters,
            tickers=req.tickers,
            date_start=req.date_start,
            date_end=req.date_end,
        ))
    except ImportError:
        logger.warning("FML module not found — run will be a no-op")

    return {"batch_id": batch_id}


@router.post("/fml/abort/{batch_id}")
async def fml_abort(batch_id: str):
    """Abort a running FML batch."""
    try:
        from references.financial_ml.applied import abort_batch
        ok = abort_batch(batch_id)
        return {"aborted": ok}
    except ImportError:
        raise HTTPException(404, "FML module not available")


@router.get("/fml/progress/{batch_id}")
async def fml_progress(batch_id: str):
    """SSE stream for FML batch progress."""
    async def event_stream():
        try:
            from references.financial_ml.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                    if progress.get("status") == "aborted":
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
        from references.testune.applied import get_chapters
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
        from references.testune.applied import run_chapters_async
        asyncio.create_task(run_chapters_async(
            batch_id, req.chapters,
            tickers=req.tickers,
            date_start=req.date_start,
            date_end=req.date_end,
        ))
    except ImportError:
        logger.warning("TTS module not found — run will be a no-op")

    return {"batch_id": batch_id}


@router.post("/tts/abort/{batch_id}")
async def tts_abort(batch_id: str):
    """Abort a running TTS batch."""
    try:
        from references.testune.applied import abort_batch
        ok = abort_batch(batch_id)
        return {"aborted": ok}
    except ImportError:
        raise HTTPException(404, "TTS module not available")


@router.get("/tts/progress/{batch_id}")
async def tts_progress(batch_id: str):
    """SSE stream for TTS batch progress."""
    async def event_stream():
        try:
            from references.testune.applied import get_batch_progress
            import json
            while True:
                progress = get_batch_progress(batch_id)
                if progress:
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress.get("completed", 0) >= progress.get("total", 1):
                        break
                    if progress.get("status") == "aborted":
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
                from rag_pipeline.llm.llm_service import create_llm_backend
                llm = create_llm_backend()
                tokens = llm.generate_stream(q, "")
                for tok in tokens:
                    yield f"event: token\ndata: {json.dumps(tok)}\n\n"
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

_ticker_price_cache: Dict[str, Any] = {}   # L1 in-memory: cache_key -> response dict
_ticker_cache_ts: Dict[str, float] = {}    # L1 in-memory: cache_key -> monotonic ts
_ticker_cache_lock = asyncio.Lock()        # guards concurrent dict access
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
    cache_key = ""
    try:
        import yfinance as yf
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        if not syms:
            return {"is_market_open": _is_market_open(market), "prices": []}

        # ── cache lookup (lock-protected) ──
        cache_key = f"{market}:{',' .join(sorted(syms))}"
        now = time.monotonic()
        is_open = _is_market_open(market)
        ttl = _TICKER_CACHE_TTL_OPEN if is_open else _TICKER_CACHE_TTL_CLOSED

        async with _ticker_cache_lock:
            if cache_key in _ticker_price_cache and (now - _ticker_cache_ts.get(cache_key, 0)) < ttl:
                return _ticker_price_cache[cache_key]

        # L2: Redis (cross-restart persistence)
        try:
            from infrastructure.cache import cache as _redis_cache
            redis_val = _redis_cache.get(f"price:{cache_key}")
            if redis_val is not None:
                async with _ticker_cache_lock:
                    _ticker_price_cache[cache_key] = redis_val
                    _ticker_cache_ts[cache_key] = now
                return redis_val
        except Exception:
            pass

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

        # ── populate cache (L1 + L2, lock-protected) ──
        async with _ticker_cache_lock:
            _ticker_price_cache[cache_key] = result
            _ticker_cache_ts[cache_key] = now
        try:
            from infrastructure.cache import cache as _redis_cache
            _redis_cache.set(f"price:{cache_key}", result, ttl=ttl)
        except Exception:
            pass

        return result
    except Exception as e:
        logger.error(f"ticker-prices error: {e}")
        # Return stale cache on error if available
        async with _ticker_cache_lock:
            stale = _ticker_price_cache.get(cache_key)
        if stale is not None:
            return stale
        raise HTTPException(status_code=500, detail=str(e))


# ─── RL Bot ─────────────────────────────────────────────────────────────


class RLTrainRequest(BaseModel):
    tickers: List[str]
    algorithm: str = "PPO"
    reward_type: str = "hybrid"
    total_timesteps: int = 500000
    lookback: int = 60
    train_days: int = 504
    test_days: int = 63
    folds: int = 6
    initial_capital: float = 100000


class RLEvalRequest(BaseModel):
    ticker: str
    algorithm: str = "PPO"
    eval_days: int = 252


class RLSignalRequest(BaseModel):
    ticker: str
    algorithm: str = "PPO"


@router.post("/rl-bot/train")
async def rl_bot_train(req: RLTrainRequest):
    """Train RL agents for given tickers."""
    try:
        from services.rl_bot.train_rl_agent import train_multi_ticker, TrainConfig

        cfg = TrainConfig(
            algorithm=req.algorithm,
            total_timesteps=req.total_timesteps,
            reward_type=req.reward_type,
            lookback=req.lookback,
            train_days=req.train_days,
            test_days=req.test_days,
            total_folds=req.folds,
            initial_capital=req.initial_capital,
        )

        results = await asyncio.to_thread(train_multi_ticker, req.tickers, cfg)

        out = {}
        for ticker, r in results.items():
            out[ticker] = {
                "algorithm": r.algorithm,
                "model_path": r.model_path,
                "avg_test_return": round(r.avg_test_return, 2),
                "avg_test_sharpe": round(r.avg_test_sharpe, 2),
                "avg_test_drawdown": round(r.avg_test_drawdown, 2),
                "folds": [
                    {
                        "fold": f.fold,
                        "train_period": f"{f.train_start} → {f.train_end}",
                        "test_period": f"{f.test_start} → {f.test_end}",
                        "return_pct": round(f.test_return_pct, 2),
                        "sharpe": round(f.test_sharpe, 2),
                        "max_dd_pct": round(f.test_max_drawdown_pct, 2),
                        "trades": f.test_n_trades,
                        "win_rate": round(f.test_win_rate * 100, 1),
                    }
                    for f in r.folds
                ],
            }

        return {"results": out}
    except Exception as e:
        logger.error("RL train error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rl-bot/evaluate")
async def rl_bot_evaluate(req: RLEvalRequest):
    """Evaluate a trained RL agent on recent data."""
    try:
        from pathlib import Path

        safe_ticker = req.ticker.replace(".", "_").replace(":", "_")
        model_dir = Path("data") / "rl_models"
        model_path = str(model_dir / f"{safe_ticker}_{req.algorithm.lower()}")

        # Check model exists
        if not (model_dir / f"{safe_ticker}_{req.algorithm.lower()}.zip").exists():
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for {req.ticker} ({req.algorithm}). Train first.",
            )

        from services.rl_bot.evaluate_agent import evaluate_agent
        from dataclasses import asdict

        metrics, signals, trades = await asyncio.to_thread(
            evaluate_agent,
            req.ticker,
            model_path,
            req.algorithm,
            eval_days=req.eval_days,
        )

        return _sanitize_floats({
            "metrics": asdict(metrics),
            "signals": [
                {"date": s.date, "action": s.action, "confidence": round(s.confidence, 3)}
                for s in signals[-30:]
            ],
            "trades": trades[-50:],
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RL evaluate error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rl-bot/signal")
async def rl_bot_signal(req: RLSignalRequest):
    """Get latest RL signal for a ticker."""
    try:
        from pathlib import Path
        from services.rl_bot.evaluate_agent import get_latest_signal

        safe_ticker = req.ticker.replace(".", "_").replace(":", "_")
        model_dir = Path("data") / "rl_models"
        model_path = str(model_dir / f"{safe_ticker}_{req.algorithm.lower()}")

        if not (model_dir / f"{safe_ticker}_{req.algorithm.lower()}.zip").exists():
            return {"action": "HOLD", "confidence": 0.0, "status": "no_model"}

        signal = await asyncio.to_thread(
            get_latest_signal, req.ticker, model_path, req.algorithm,
        )

        return {
            "ticker": signal.ticker,
            "date": signal.date,
            "action": signal.action,
            "confidence": round(signal.confidence, 3),
            "portfolio_value": signal.portfolio_value,
            "position": signal.position,
        }
    except Exception as e:
        logger.error("RL signal error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rl-bot/models")
async def rl_bot_models():
    """List all trained RL models."""
    from pathlib import Path

    model_dir = Path("data") / "rl_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for f in model_dir.glob("*.zip"):
        parts = f.stem.rsplit("_", 1)
        ticker = parts[0].replace("_", ".") if len(parts) == 2 else f.stem
        algo = parts[1].upper() if len(parts) == 2 else "UNKNOWN"
        models.append({
            "ticker": ticker,
            "algorithm": algo,
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
        })

    return {"models": models}


@router.post("/rl-bot/upload-data")
async def rl_bot_upload_data(file: UploadFile = File(...)):
    """Upload a CSV file with OHLCV data for RL training.

    Expected columns: Date, Open, High, Low, Close, Volume
    Optional: Ticker column for multi-stock files.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    import pandas as pd
    from io import StringIO
    from pathlib import Path

    upload_dir = Path("data") / "rl_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        text = content.decode("utf-8")
        df = pd.read_csv(StringIO(text))

        # Normalize column names
        col_map = {}
        for col in df.columns:
            low = col.strip().lower()
            if low in ("date", "datetime", "timestamp"):
                col_map[col] = "Date"
            elif low in ("open", "o"):
                col_map[col] = "Open"
            elif low in ("high", "h"):
                col_map[col] = "High"
            elif low in ("low", "l"):
                col_map[col] = "Low"
            elif low in ("close", "c", "adj close", "adj_close"):
                col_map[col] = "Close"
            elif low in ("volume", "vol", "v"):
                col_map[col] = "Volume"
            elif low in ("ticker", "symbol"):
                col_map[col] = "Ticker"
        df.rename(columns=col_map, inplace=True)

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing)}. "
                       f"Found: {', '.join(df.columns.tolist())}",
            )

        # Detect tickers in file
        tickers = []
        if "Ticker" in df.columns:
            tickers = df["Ticker"].dropna().unique().tolist()

        # Save the file
        safe_name = file.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        save_path = upload_dir / safe_name
        df.to_csv(save_path, index=False)

        return {
            "filename": safe_name,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "tickers": tickers[:50],
            "date_range": {
                "start": str(df["Date"].iloc[0]) if "Date" in df.columns else None,
                "end": str(df["Date"].iloc[-1]) if "Date" in df.columns else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RL upload error: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")


@router.get("/rl-bot/uploads")
async def rl_bot_list_uploads():
    """List uploaded CSV data files."""
    from pathlib import Path

    upload_dir = Path("data") / "rl_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for f in sorted(upload_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        files.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
        })

    return {"files": files}


@router.get("/rl-bot/portfolio-analysis")
async def rl_bot_portfolio_analysis():
    """
    Analyze the Green Energy Theme portfolio and generate Nifty buy predictions.
    Studies constituent stock selection patterns, factor characteristics, and
    applies the learned model to the broader Nifty universe.
    """
    try:
        from services.rl_bot.theme_analyzer import run_portfolio_analysis

        result = run_portfolio_analysis()
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Portfolio analysis error: %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
