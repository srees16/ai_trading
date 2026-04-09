"""Portfolio router — Dual-strategy (Centurion Compounder + Harvest) API."""

import asyncio
import logging

from fastapi import APIRouter

from api.schemas.portfolio import (
    HarvestParams,
    PortfolioBacktestRequest,
    PortfolioBacktestResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.post("/backtest", response_model=PortfolioBacktestResponse)
async def run_portfolio_backtest(req: PortfolioBacktestRequest):
    """Run dual-strategy backtest with user-defined capital split."""
    from services.portfolio_allocator import run_dual_backtest, HARVEST_PRESETS

    hp = None
    if req.harvest_params:
        hp = req.harvest_params.model_dump()

    result = await asyncio.to_thread(
        run_dual_backtest,
        total_capital=req.total_capital,
        compounder_pct=req.compounder_pct,
        harvest_params=hp,
        start_date=req.start_date,
        end_date=req.end_date,
    )

    return result


@router.get("/presets")
async def get_harvest_presets():
    """Return available Harvest parameter presets."""
    from services.portfolio_allocator import HARVEST_PRESETS
    return {"presets": HARVEST_PRESETS}
