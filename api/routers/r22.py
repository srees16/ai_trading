"""R22 router — Bull-Run Capital Infusion backtest + alert API."""

import asyncio
import logging

from fastapi import APIRouter

from api.schemas.r22 import (
    R22BacktestRequest,
    R22BacktestResponse,
    R22AlertEvent,
    R22InfusionEvent,
    R22InfusionSummary,
    R22Metrics,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/r22", tags=["R22 Capital Infusion"])


def _extract_metrics(result: dict) -> R22Metrics:
    return R22Metrics(
        sharpe=result.get("sharpe", 0),
        sortino=result.get("sortino", 0),
        calmar=result.get("calmar", 0),
        cagr_pct=result.get("annual_return_pct", 0),
        max_drawdown_pct=result.get("max_drawdown_pct", 0),
        total_return_pct=result.get("total_return_pct", 0),
        total_trades=result.get("n_trades", 0),
        win_rate=result.get("win_rate", 0),
        profit_factor=result.get("profit_factor", 0),
    )


def _run_backtest(capital: float, start_date: str, end_date: str,
                  infuse: bool, infusion_amount: float,
                  cooldown_days: int, bull_confirm_days: int) -> dict:
    """Run R22 backtest (blocking — meant to be called via asyncio.to_thread)."""
    import importlib
    import services.full_pipeline_backtest as bt_mod

    # Reset module state
    importlib.reload(bt_mod)

    # R21A base config
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25
    bt_mod._R21A_REGIME_DEFEND = 0.55
    bt_mod._SAVE_FORECASTS_MODE = False

    # R22 config
    bt_mod._R22_BULL_INFUSION = infuse
    bt_mod._R22_INFUSION_AMOUNT = infusion_amount
    bt_mod._R22_INFUSION_COOLDOWN_DAYS = cooldown_days
    bt_mod._R22_BULL_CONFIRM_DAYS = bull_confirm_days

    # Harvest OFF
    bt_mod._HARVEST_ENABLED = False
    bt_mod._HARVEST_DIP_BUYER = False
    bt_mod._HARVEST_PROFIT_TAKER = False

    return bt_mod.run_full_backtest(
        tickers=None,
        capital=capital,
        period="13y",
        market="IND",
        verbose=False,
        start_date=start_date,
        end_date=end_date,
    )


@router.post("/backtest", response_model=R22BacktestResponse)
async def run_r22_backtest(req: R22BacktestRequest):
    """Run R22 backtest with optional capital infusion, plus R21A baseline for comparison."""

    # Run R22 (with infusion)
    r22_result = await asyncio.to_thread(
        _run_backtest,
        capital=req.capital,
        start_date=req.start_date,
        end_date=req.end_date,
        infuse=req.infuse,
        infusion_amount=req.infusion_amount,
        cooldown_days=req.cooldown_days,
        bull_confirm_days=req.bull_confirm_days,
    )

    # Run R21A baseline (no infusion) for comparison
    r21a_result = await asyncio.to_thread(
        _run_backtest,
        capital=req.capital,
        start_date=req.start_date,
        end_date=req.end_date,
        infuse=False,
        infusion_amount=0,
        cooldown_days=200,
        bull_confirm_days=5,
    )

    # Extract R22 infusion data
    r22_data = r22_result.get("r22_bull_infusion") or {}
    alert_events = [
        R22AlertEvent(day=e[0], date=e[1])
        for e in r22_data.get("alert_events", [])
    ]
    infusion_events = [
        R22InfusionEvent(day=e[0], amount=e[1], equity_before=e[2], equity_after=e[3])
        for e in r22_data.get("infusion_events", [])
    ]

    return R22BacktestResponse(
        metrics=_extract_metrics(r22_result),
        r21a_benchmark=_extract_metrics(r21a_result),
        infusion_summary=R22InfusionSummary(
            enabled=req.infuse,
            infusion_amount=req.infusion_amount,
            total_infused=r22_data.get("total_infused", 0),
            n_alerts=r22_data.get("n_alerts", 0),
            n_infusions=r22_data.get("n_infusions", 0),
            alert_events=alert_events,
            infusion_events=infusion_events,
        ),
        daily_equity=r22_result.get("daily_equity", []),
        r21a_daily_equity=r21a_result.get("daily_equity", []),
    )


@router.get("/alerts")
async def get_latest_bull_alerts():
    """Get the latest bull-run alert events from the most recent backtest.

    In production, this would check live regime state. For now, returns
    sample data indicating when infusion opportunities were detected.
    """
    # This is a placeholder for live regime detection.
    # In production, this should:
    # 1. Check the live equity curve SMA200
    # 2. Detect bear→bull transition
    # 3. Return active alert if within window
    return {
        "has_active_alert": False,
        "message": "No active bull-run alert. Monitoring regime...",
        "last_check": None,
    }
