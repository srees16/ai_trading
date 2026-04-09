"""Schemas for the Dual-Strategy Portfolio API (Centurion Compounder + Harvest)."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────

class HarvestParams(BaseModel):
    """Tunable Centurion Harvest parameters."""
    inject_pct: float = Field(0.20, ge=0.05, le=0.50, description="Fraction of base capital to inject at bear→bull crossover")
    book_pct: float = Field(0.15, ge=0.05, le=0.50, description="Fraction of gains to book in sustained bull")
    sustain_days: int = Field(30, ge=10, le=120, description="Days above SMA200 before profit booking triggers")
    min_gain_to_book: float = Field(0.10, ge=0.0, le=0.50, description="Min gain (fraction) above invested capital before booking")
    inject_cooldown_days: int = Field(200, ge=30, le=500, description="Min days between capital injections")
    preset: Optional[str] = Field(None, description="Optional preset: 'conservative', 'balanced', 'aggressive'")


class PortfolioBacktestRequest(BaseModel):
    """Request for a dual-strategy backtest."""
    total_capital: float = Field(1_000_000, ge=100_000, description="Total capital in ₹")
    compounder_pct: float = Field(50.0, ge=0.0, le=100.0, description="% allocated to Centurion Compounder")
    harvest_params: Optional[HarvestParams] = None
    start_date: str = Field("2012-01-01", description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field("2025-12-31", description="Backtest end date (YYYY-MM-DD)")


# ── Response Models ───────────────────────────────────────────────────────

class EquityPoint(BaseModel):
    day: int
    equity: float


class HarvestEvent(BaseModel):
    day: int
    amount: float
    equity_before: float
    equity_after: float
    event_type: str  # "inject" or "book"


class StrategyMetrics(BaseModel):
    """Key performance metrics for one strategy leg."""
    strategy_name: str
    capital_allocated: float
    final_equity: float
    sharpe: float
    sortino: float
    calmar: float
    cagr_pct: float
    max_drawdown_pct: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float


class HarvestSummary(BaseModel):
    """Capital rotation event summary for Centurion Harvest."""
    total_injected: float
    total_booked: float
    net_extracted: float
    inject_events: List[HarvestEvent]
    book_events: List[HarvestEvent]


class PortfolioBacktestResponse(BaseModel):
    """Result of a dual-strategy backtest."""
    total_capital: float
    compounder_pct: float
    harvest_pct: float

    compounder: StrategyMetrics
    harvest: StrategyMetrics

    # Combined wealth = CC equity + CH equity + CH net extracted
    combined_wealth: float
    combined_return_pct: float

    # Equity curves for charting
    compounder_equity: List[EquityPoint]
    harvest_equity: List[EquityPoint]

    # Harvest events
    harvest_summary: Optional[HarvestSummary] = None

    execution_time_sec: float
