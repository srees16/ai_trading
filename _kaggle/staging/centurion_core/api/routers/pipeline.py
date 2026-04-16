"""
Pipeline API router — programmatic access to the NSE screening + scoring pipeline.

Endpoints
---------
POST /ind-stocks/pipeline/screen
    Run the 3-stage screener on NIFTY50+Next50 universe.

POST /ind-stocks/pipeline/full
    Screen + IntegratedScorer verdict + (optionally) place orders via Kite.

GET  /ind-stocks/pipeline/latest
    Return the most recent scheduled / manual pipeline run from cache.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_kite_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ind-stocks/pipeline", tags=["IND Pipeline"])


# ═══════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════

class PipelineRequest(BaseModel):
    """Request body for pipeline endpoints."""
    symbols: Optional[List[str]] = Field(
        None,
        description="Override symbol list; defaults to NIFTY50+Next50 if omitted.",
    )
    auto_place: bool = Field(
        False,
        description="If True AND Kite is authenticated, place live orders.",
    )
    index_mode: bool = Field(
        True,
        description="Use relaxed screener filters for blue-chip universe.",
    )


class VerdictItem(BaseModel):
    ticker: str
    score: float
    classification: str
    confidence: float


class PlanItem(BaseModel):
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    target_price: float
    quantity: int
    rr_ratio: float


class OrderItem(BaseModel):
    symbol: str
    side: str
    quantity: int
    order_id: Optional[str] = None
    success: bool
    error: Optional[str] = None


class PipelineResponse(BaseModel):
    universe_size: int = 0
    screened_count: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    orders_placed: int = 0
    orders_failed: int = 0
    verdicts: List[VerdictItem] = []
    plans: List[PlanItem] = []
    orders: List[OrderItem] = []


class LatestRunResponse(BaseModel):
    run_type: Optional[str] = None
    timestamp: Optional[str] = None
    universe_size: int = 0
    screened_count: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    status: str = "no_data"


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@router.post("/screen", response_model=PipelineResponse)
async def run_screen(req: PipelineRequest):
    """Run the 3-stage NSE screener (no verdicts, no orders)."""
    try:
        from kite_connect.nse.nse_universe import get_nse_universe
        from kite_connect.nse.screener import NSEScreener, ScreenerConfig
        from kite_connect.trading.risk_manager import RiskManager, RiskConfig

        symbols = req.symbols or get_nse_universe()
        cfg = ScreenerConfig(index_mode=req.index_mode)
        screener = NSEScreener(config=cfg)
        screened_df = screener.screen(symbols)

        plans = []
        if not screened_df.empty:
            kite = get_kite_session()

            # Carver-aware: use VolatilityTarget when enabled
            vol_target = None
            try:
                from config import Config
                if getattr(Config, "CARVER_ENABLED", False):
                    from services.volatility_target import VolatilityTarget, VolatilityTargetConfig
                    vt_cfg = VolatilityTargetConfig(
                        initial_capital=getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000.0),
                        annual_vol_target_pct=getattr(Config, "CARVER_ANNUAL_VOL_TARGET", 0.20),
                    )
                    vol_target = VolatilityTarget(vt_cfg)
            except Exception:
                pass

            rm = RiskManager(kite=kite, volatility_target=vol_target)
            plans = rm.plan_trades(screened_df)

        return PipelineResponse(
            universe_size=len(symbols),
            screened_count=len(screened_df),
            plans=[PlanItem(
                symbol=p.symbol, side=p.side,
                entry_price=p.entry_price, stop_loss=p.stop_loss,
                target_price=p.target_price, quantity=p.quantity,
                rr_ratio=p.rr_ratio,
            ) for p in plans],
        )
    except Exception as exc:
        logger.exception("Pipeline /screen failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/full", response_model=PipelineResponse)
async def run_full_pipeline(req: PipelineRequest):
    """Screen → IntegratedScorer → (optionally) place orders."""
    try:
        from kite_connect.nse.nse_universe import get_nse_universe
        from kite_connect.trading.auto_executor import AutoExecutor
        from kite_connect.nse.screener import ScreenerConfig
        from kite_connect.trading.risk_manager import RiskConfig
        from services.integrated_scorer import IntegratedScorer

        kite = get_kite_session() if req.auto_place else None
        symbols = req.symbols or get_nse_universe()

        # Step 1: Screen
        scfg = ScreenerConfig(index_mode=req.index_mode)
        executor = AutoExecutor(kite=kite, screener_cfg=scfg, auto_place=False)
        report = executor.run(symbols=symbols)

        if report.screened_df.empty:
            return PipelineResponse(
                universe_size=report.universe_size,
                screened_count=0,
            )

        # Step 2: Verdict
        ns_tickers = [f"{s}.NS" for s in report.screened_df["symbol"].tolist()]
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365)
        scorer = IntegratedScorer()
        verdicts = scorer.evaluate(
            tickers=ns_tickers, market="IND",
            date_range=(str(start_dt), str(end_dt)),
        )

        verdict_items = [
            VerdictItem(
                ticker=v.ticker, score=round(v.final_score, 3),
                classification=v.classification,
                confidence=round(v.confidence, 2),
            )
            for v in verdicts
        ]
        buy_verdicts = [v for v in verdicts if v.classification in ("BUY", "STRONG_BUY")]
        sell_verdicts = [v for v in verdicts if v.classification in ("SELL", "STRONG_SELL")]

        # Step 3: Place orders (if requested and Kite available)
        orders_resp: List[OrderItem] = []
        orders_placed = 0
        orders_failed = 0

        if req.auto_place and kite and buy_verdicts:
            signal_dict = {
                v.ticker.replace(".NS", "").replace(".BO", ""): v.classification
                for v in verdicts
            }
            buy_syms = [v.ticker.replace(".NS", "").replace(".BO", "") for v in buy_verdicts]
            pre_screened = report.screened_df[
                report.screened_df["symbol"].isin(buy_syms)
            ].copy()

            order_exec = AutoExecutor(kite=kite, screener_cfg=scfg, auto_place=True)
            order_report = order_exec.run(
                symbols=buy_syms,
                signal_verdicts=signal_dict,
                pre_screened_df=pre_screened,
            )
            orders_placed = order_report.orders_placed
            orders_failed = order_report.orders_failed
            orders_resp = [
                OrderItem(
                    symbol=o.symbol, side=o.side, quantity=o.quantity,
                    order_id=o.order_id, success=o.success, error=o.error,
                )
                for o in order_report.order_results
            ]

        return PipelineResponse(
            universe_size=report.universe_size,
            screened_count=report.screened_count,
            buy_signals=len(buy_verdicts),
            sell_signals=len(sell_verdicts),
            orders_placed=orders_placed,
            orders_failed=orders_failed,
            verdicts=verdict_items,
            plans=[PlanItem(
                symbol=p.symbol, side=p.side,
                entry_price=p.entry_price, stop_loss=p.stop_loss,
                target_price=p.target_price, quantity=p.quantity,
                rr_ratio=p.rr_ratio,
            ) for p in report.trade_plans],
            orders=orders_resp,
        )
    except Exception as exc:
        logger.exception("Pipeline /full failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/latest", response_model=LatestRunResponse)
async def get_latest_run(run_type: Optional[str] = None):
    """Return the most recent scheduled pipeline run from the cache DB."""
    try:
        from scheduler import get_latest_run
        row = get_latest_run(run_type)
        if row is None:
            return LatestRunResponse()
        return LatestRunResponse(
            run_type=row.get("run_type"),
            timestamp=row.get("timestamp"),
            universe_size=row.get("universe_size", 0),
            screened_count=row.get("screened_count", 0),
            buy_signals=row.get("buy_signals", 0),
            sell_signals=row.get("sell_signals", 0),
            status=row.get("status", "unknown"),
        )
    except Exception as exc:
        logger.debug("Latest run fetch failed: %s", exc)
        return LatestRunResponse()


@router.get("/carver/status")
async def carver_status():
    """Return Carver framework configuration and wiring status."""
    try:
        from config import Config
        enabled = getattr(Config, "CARVER_ENABLED", False)
        return {
            "carver_enabled": enabled,
            "annual_vol_target": getattr(Config, "CARVER_ANNUAL_VOL_TARGET", 0.20),
            "initial_capital": getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000.0),
            "default_idm": getattr(Config, "CARVER_DEFAULT_IDM", 1.6),
            "max_leverage": getattr(Config, "CARVER_MAX_LEVERAGE", 1.0),
            "inertia_threshold": getattr(Config, "CARVER_INERTIA_THRESHOLD", 0.10),
            "cost_speed_limit_factor": getattr(Config, "CARVER_COST_SPEED_LIMIT", 3.0),
            "trade_horizon": getattr(Config, "CARVER_TRADE_HORIZON", "swing"),
            "modules_available": _check_carver_modules(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/carver/efficiency")
async def carver_efficiency_report(symbols: Optional[List[str]] = None):
    """Run the Carver calibration backtest and return the efficiency report.

    This computes before vs after metrics showing the impact of the
    Carver Systematic Trading framework redesign.
    """
    try:
        from services.carver_calibration import CarverCalibrator, generate_efficiency_report
        from config import Config
        from utils import download_ind_ohlcv
        from kite_connect.nse.nse_universe import get_nse_universe

        tickers = symbols or get_nse_universe()[:20]  # Default: top 20

        # Fetch OHLCV data
        ohlcv_cache = {}
        for sym in tickers:
            try:
                df = download_ind_ohlcv(sym, period="1y")
                if df is not None and len(df) >= 120:
                    ohlcv_cache[sym] = df
            except Exception:
                pass

        if not ohlcv_cache:
            raise HTTPException(status_code=400, detail="No OHLCV data available for calibration")

        calibrator = CarverCalibrator(
            annual_vol_target=getattr(Config, "CARVER_ANNUAL_VOL_TARGET", 0.20),
            initial_capital=getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000.0),
        )
        report = calibrator.run_expanding_backtest(ohlcv_cache)
        text_report = generate_efficiency_report(report)

        return {
            "report_text": text_report,
            "backtest_sharpe": report.backtest_sharpe,
            "backtest_sortino": report.backtest_sortino,
            "backtest_max_drawdown_pct": report.backtest_max_drawdown_pct,
            "backtest_annual_return_pct": report.backtest_annual_return_pct,
            "n_symbols": report.n_symbols,
            "n_days": report.n_days,
            "ewmac_scalars": report.ewmac_scalars,
            "calibrated_fdm": report.calibrated_fdm,
            "calibrated_idm": report.calibrated_idm,
            "before_metrics": report.before_metrics,
            "after_metrics": report.after_metrics,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Carver efficiency report failed")
        raise HTTPException(status_code=500, detail=str(exc))


def _check_carver_modules() -> Dict[str, bool]:
    """Check which Carver modules are importable."""
    modules = {
        "instrument_volatility": "services.instrument_volatility",
        "volatility_target": "services.volatility_target",
        "forecast_scalar": "services.forecast_scalar",
        "forecast_combiner": "services.forecast_combiner",
        "position_sizer": "services.position_sizer",
        "instrument_weights": "services.instrument_weights",
        "ewmac": "strategies.ewmac",
        "carry_rule": "strategies.carry_rule",
        "vol_trailing_stop": "services.vol_trailing_stop",
        "cost_speed_limit": "services.cost_speed_limit",
        "portfolio_vol_monitor": "services.portfolio_vol_monitor",
        "carver_calibration": "services.carver_calibration",
        "carver_pipeline": "services.carver_pipeline",
    }
    result = {}
    for name, mod_path in modules.items():
        try:
            __import__(mod_path)
            result[name] = True
        except ImportError:
            result[name] = False
    return result


# ═══════════════════════════════════════════════════════════════
# G4: Walk-Forward Validation (on-demand trigger)
# ═══════════════════════════════════════════════════════════════

class WalkForwardResponse(BaseModel):
    strategies_tested: int = 0
    overfit_count: int = 0
    params_saved: int = 0
    status: str = "pending"
    details: Optional[Dict[str, Any]] = None


@router.post("/walk-forward", response_model=WalkForwardResponse)
async def trigger_walk_forward():
    """G4: Trigger walk-forward validation for all strategies.

    This runs the same audit that the Saturday scheduler job performs,
    but on demand.  Results are saved to data/wf_params/ and
    strategy_decay_state.json is updated.
    """
    try:
        from scheduler import run_walk_forward_audit
        run_walk_forward_audit()

        # Count saved params
        from pathlib import Path
        wf_dir = Path(__file__).resolve().parent.parent.parent / "data" / "wf_params"
        params_saved = len(list(wf_dir.glob("*.json"))) if wf_dir.exists() else 0

        return WalkForwardResponse(
            strategies_tested=1,  # placeholder — actual count in audit log
            params_saved=params_saved,
            status="success",
        )
    except Exception as exc:
        logger.exception("Walk-forward trigger failed")
        raise HTTPException(status_code=500, detail=str(exc))
