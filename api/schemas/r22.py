"""Schemas for R22 Bull-Run Capital Infusion API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class R22BacktestRequest(BaseModel):
    """Request for an R22 capital infusion backtest."""
    capital: float = Field(500_000, ge=100_000, description="Starting capital in ₹")
    infuse: bool = Field(True, description="Whether to actually infuse capital (False = alerts only)")
    infusion_amount: float = Field(50_000, ge=10_000, le=500_000, description="₹ amount per infusion")
    cooldown_days: int = Field(200, ge=30, le=500, description="Min trading days between infusions")
    bull_confirm_days: int = Field(5, ge=3, le=30, description="Consecutive bull days to confirm regime change")
    start_date: str = Field("2012-01-01", description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field("2025-12-31", description="Backtest end date (YYYY-MM-DD)")


class R22AlertEvent(BaseModel):
    day: int
    date: str


class R22InfusionEvent(BaseModel):
    day: int
    amount: float
    equity_before: float
    equity_after: float


class R22InfusionSummary(BaseModel):
    enabled: bool
    infusion_amount: float
    total_infused: float
    n_alerts: int
    n_infusions: int
    alert_events: List[R22AlertEvent]
    infusion_events: List[R22InfusionEvent]


class R22Metrics(BaseModel):
    sharpe: float
    sortino: float
    calmar: float
    cagr_pct: float
    max_drawdown_pct: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float


class R22BacktestResponse(BaseModel):
    metrics: R22Metrics
    r21a_benchmark: R22Metrics
    infusion_summary: R22InfusionSummary
    daily_equity: List[float]
    r21a_daily_equity: List[float]
