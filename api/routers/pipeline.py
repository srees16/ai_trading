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
            rm = RiskManager(kite=kite)
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
