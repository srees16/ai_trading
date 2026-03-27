"""
Cost Speed Limit — Carver Chapter 12 cost-aware trading filter.

Rule: Only trade if expected annual Sharpe ratio contribution
exceeds the cost of trading by a factor of 3:

    cost_per_trade = turnover × (spread + commission + slippage)
    annual_cost = n_trades_per_year × cost_per_trade
    speed_limit_ok = (expected_SR_contribution / annual_cost_drag) > 3

This prevents:
  - Churning low-conviction positions
  - EWMAC(16,64) signals on illiquid mid-caps where spread eats alpha
  - Re-balancing when position inertia should apply

For NSE equities:
  - Brokerage: ~0.03% (₹20 flat cap for Zerodha)
  - STT: 0.1% (on sell delivery)
  - Exchange charges: ~0.00325%
  - GST on brokerage: 18%
  - Stamp duty: 0.015% (buy) / 0.003% (sell)
  - Typical total round-trip: ~0.25-0.35% for delivery trades
  - Estimated spread+slippage: 0.10-0.40% depending on liquidity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """Cost parameters for NSE equity delivery trades."""
    # Round-trip transaction cost as fraction (brokerage + STT + charges)
    round_trip_cost_pct: float = 0.0030   # 0.30% round trip
    # Estimated spread + slippage (added on top)
    spread_slippage_pct: float = 0.0020   # 0.20% for liquid large-caps
    # Speed limit factor: cost_drag must be < SR / speed_limit_factor
    speed_limit_factor: float = 3.0       # Carver: "cost must be < SR/3"
    # Annual turnover estimate for each rule variation
    # EWMAC(16,64): ~12-18 trades/year   → turnover ~30× position
    # EWMAC(64,256): ~4-6 trades/year    → turnover ~10×
    # Carry: ~2-4 trades/year             → turnover ~6×
    # Screener: ~6-10 trades/year         → turnover ~18×
    # These are per-instrument turnovers (one-way)
    default_annual_turnover: float = 15.0  # weighted average one-way trades/year


@dataclass
class CostCheckResult:
    """Result of a cost speed limit check."""
    symbol: str
    allowed: bool
    total_cost_pct: float       # round-trip cost for this trade
    annual_cost_drag: float     # estimated annual cost drag (fraction)
    min_sr_required: float      # minimum SR to justify this cost
    estimated_sr: float         # estimated SR contribution from forecast
    reason: str = ""


def estimate_trade_cost(
    price: float,
    quantity: int,
    is_sell: bool = False,
    config: Optional[CostConfig] = None,
) -> float:
    """Estimate total cost for a single trade in ₹.

    Parameters
    ----------
    price : float
        Trade price per share.
    quantity : int
        Number of shares.
    is_sell : bool
        True for sell (STT is higher on sell delivery).
    config : CostConfig | None

    Returns
    -------
    float
        Estimated cost in ₹.
    """
    cfg = config or CostConfig()
    turnover = price * quantity
    cost_pct = cfg.round_trip_cost_pct / 2 + cfg.spread_slippage_pct / 2
    if is_sell:
        cost_pct += 0.001  # extra STT on sell
    return turnover * cost_pct


def check_speed_limit(
    symbol: str,
    combined_forecast: float,
    annual_vol_target_pct: float = 0.20,
    annual_turnover: Optional[float] = None,
    config: Optional[CostConfig] = None,
) -> CostCheckResult:
    """Check whether a trade passes the Carver cost speed limit.

    The estimated Sharpe contribution of a forecast must exceed
    3× the cost drag from trading at the expected frequency.

    Parameters
    ----------
    symbol : str
        Instrument ticker.
    combined_forecast : float
        Combined forecast value (-20 to +20).
    annual_vol_target_pct : float
        Annual portfolio volatility target (fraction, e.g. 0.20).
    annual_turnover : float | None
        Estimated annual one-way trades for this instrument.
    config : CostConfig | None

    Returns
    -------
    CostCheckResult
    """
    cfg = config or CostConfig()
    turnover = annual_turnover or cfg.default_annual_turnover

    # Total cost per round-trip trade (fraction)
    cost_per_trade = cfg.round_trip_cost_pct + cfg.spread_slippage_pct

    # Annual cost drag = turnover × cost_per_trade
    annual_cost_drag = turnover * cost_per_trade

    # Estimated SR contribution from this forecast
    # Forecast of 10 = neutral conviction → SR ≈ 0.20 (half-Kelly target)
    # Scale linearly: forecast of 20 → SR ≈ 0.40
    forecast_strength = abs(combined_forecast) / 10.0
    estimated_sr = forecast_strength * annual_vol_target_pct

    # Speed limit: SR contribution must be > factor × cost drag
    min_sr = cfg.speed_limit_factor * annual_cost_drag
    allowed = estimated_sr >= min_sr

    reason = ""
    if not allowed:
        reason = (
            f"Cost speed limit: SR {estimated_sr:.3f} < {min_sr:.3f} "
            f"(cost drag {annual_cost_drag:.3f} × {cfg.speed_limit_factor:.0f})"
        )
        logger.info("Speed limit blocks %s: %s", symbol, reason)

    return CostCheckResult(
        symbol=symbol,
        allowed=allowed,
        total_cost_pct=cost_per_trade,
        annual_cost_drag=annual_cost_drag,
        min_sr_required=min_sr,
        estimated_sr=estimated_sr,
        reason=reason,
    )


def filter_by_cost(
    forecasts: dict[str, float],
    annual_vol_target_pct: float = 0.20,
    config: Optional[CostConfig] = None,
) -> dict[str, float]:
    """Filter forecasts by cost speed limit, returning only those that pass.

    Parameters
    ----------
    forecasts : dict[str, float]
        {symbol: combined_forecast}.
    annual_vol_target_pct : float
        Annual vol target (fraction).
    config : CostConfig | None

    Returns
    -------
    dict[str, float]
        Filtered forecasts where cost speed limit is satisfied.
    """
    passed = {}
    blocked = 0
    for sym, fc in forecasts.items():
        result = check_speed_limit(sym, fc, annual_vol_target_pct, config=config)
        if result.allowed:
            passed[sym] = fc
        else:
            blocked += 1

    if blocked > 0:
        logger.info("Cost speed limit: %d/%d symbols blocked", blocked, len(forecasts))
    return passed
