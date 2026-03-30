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
    speed_multiplier: float     # 0.0-1.0 graduated dampening (1.0 = full pass)
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
    # Consistent one-way cost: half of round-trip + half of spread
    base_cost_pct = cfg.round_trip_cost_pct / 2 + cfg.spread_slippage_pct / 2
    # STT is 0.1% on sell-side only for delivery trades; buyer pays no STT
    stt_pct = 0.001 if is_sell else 0.0
    cost_pct = base_cost_pct + stt_pct
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

    # Annual cost drag = turnover × cost_per_trade (one-way adjusted)
    annual_cost_drag = turnover * (cost_per_trade / 2)

    # Estimated SR contribution from this forecast
    # Forecast of 10 = neutral conviction → SR ≈ 0.20 (half-Kelly target)
    # Scale linearly: forecast of 20 → SR ≈ 0.40
    forecast_strength = abs(combined_forecast) / 10.0
    estimated_sr = forecast_strength * annual_vol_target_pct

    # Speed limit: SR contribution must be > factor × cost drag
    min_sr = cfg.speed_limit_factor * annual_cost_drag

    # Graduated dampening instead of binary allow/block:
    # ratio >= 1.0 → full pass (multiplier=1.0)
    # ratio in (0.5, 1.0) → proportional dampening
    # ratio <= 0.5 → blocked (multiplier=0.0)
    if min_sr <= 0:
        ratio = 2.0
    else:
        ratio = estimated_sr / min_sr

    if ratio >= 1.0:
        speed_multiplier = 1.0
        allowed = True
    elif ratio > 0.5:
        speed_multiplier = (ratio - 0.5) / 0.5  # linear 0→1 in the 0.5-1.0 band
        allowed = True
    else:
        speed_multiplier = 0.0
        allowed = False

    reason = ""
    if not allowed:
        reason = (
            f"Cost speed limit: SR {estimated_sr:.3f} < {min_sr:.3f} "
            f"(cost drag {annual_cost_drag:.3f} × {cfg.speed_limit_factor:.0f})"
        )
        logger.info("Speed limit blocks %s: %s", symbol, reason)
    elif speed_multiplier < 1.0:
        reason = (
            f"Cost speed dampening: SR ratio {ratio:.2f} → multiplier {speed_multiplier:.2f}"
        )
        logger.info("Speed limit dampens %s: %s", symbol, reason)

    return CostCheckResult(
        symbol=symbol,
        allowed=allowed,
        speed_multiplier=round(speed_multiplier, 3),
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


# ═══════════════════════════════════════════════════════════════
# Gap B8: Forecast Capacity / Liquidity Check
# ═══════════════════════════════════════════════════════════════

def check_forecast_capacity(
    symbol: str,
    position_value: float,
    avg_daily_volume_value: float,
    max_adv_pct: float = 0.05,
) -> float:
    """Check if position size is within liquidity capacity.

    Returns a dampening multiplier (0.0 to 1.0) based on the ratio
    of position value to average daily traded value.

    Parameters
    ----------
    symbol : str
        Instrument ticker.
    position_value : float
        Target position notional value (₹).
    avg_daily_volume_value : float
        20-day average daily traded value in ₹.
    max_adv_pct : float
        Maximum position as fraction of ADV (default 5%).

    Returns
    -------
    float
        Capacity multiplier: 1.0 if within limit, linear dampening
        down to 0.0 as position approaches 2× the limit.
    """
    if avg_daily_volume_value <= 0:
        logger.warning("Capacity check: %s has zero ADV — blocking", symbol)
        return 0.0

    ratio = position_value / avg_daily_volume_value

    if ratio <= max_adv_pct:
        return 1.0
    elif ratio <= max_adv_pct * 2:
        # Linear dampening between 1× and 2× the limit
        mult = 1.0 - (ratio - max_adv_pct) / max_adv_pct
        logger.info(
            "Capacity dampening %s: position=%.0f, ADV=%.0f, ratio=%.1f%%, mult=%.2f",
            symbol, position_value, avg_daily_volume_value, ratio * 100, mult,
        )
        return max(0.0, mult)
    else:
        logger.info(
            "Capacity blocked %s: position=%.0f is %.1f%% of ADV=%.0f (> %d%%)",
            symbol, position_value, ratio * 100, avg_daily_volume_value,
            int(max_adv_pct * 200),
        )
        return 0.0
