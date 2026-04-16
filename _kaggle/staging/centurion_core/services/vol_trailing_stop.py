"""
Volatility-Based Trailing Stop — Carver-style adaptive exits.

Replaces fixed-percentage trailing stops with volatility-aware stops:

    stop_distance = multiplier × daily_price_volatility × price

This ensures stops are:
  - Wider for volatile stocks (avoid whipsaw)
  - Tighter for low-vol stocks (protect profits)
  - Automatically adaptive as volatility regime changes

Default: 2.5 × daily_vol from peak since entry (for swing)
         3.5 × daily_vol for positional (wider to ride trends)

Also implements a profit-lock mechanism:
  - Once unrealised profit > activation_threshold × daily_vol,
    the stop ratchets up to lock in a minimum gain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

@dataclass
class VolTrailingStopConfig:
    """Volatility trailing stop parameters."""
    # Stop distance as multiple of daily volatility
    vol_multiplier_swing: float = 2.5       # Swing trades: stop at 2.5σ
    vol_multiplier_positional: float = 3.5  # Positional: stop at 3.5σ (wider)

    # Profit-lock: activate tighter stop once profit exceeds threshold
    profit_lock_activation_vol: float = 4.0  # Activate after 4σ profit
    profit_lock_distance_vol: float = 1.5    # Lock stop at 1.5σ below peak

    # Hard floor: never allow stop further than this % below entry
    max_stop_distance_pct: float = 0.12      # 12% max distance from peak

    # Hard ceiling: never allow stop tighter than this % below peak
    min_stop_distance_pct: float = 0.02      # 2% minimum distance from peak

    # Contra-regime: tighter profit-lock in bull to book profits earlier
    bull_profit_lock_activation_vol: float = 3.0   # Activate after 3σ in bull (vs 4σ default)
    bull_profit_lock_distance_vol: float = 1.0     # Lock at 1.0σ in bull (vs 1.5σ default)


@dataclass
class TrailingStopState:
    """Tracks trailing stop state for a single position."""
    symbol: str
    entry_price: float
    peak_price: float           # highest price since entry
    current_stop: float         # current trailing stop level
    daily_vol: float            # latest daily price volatility (fraction)
    vol_multiplier: float       # which multiplier is active
    profit_locked: bool = False # True once profit-lock activated
    trade_horizon: str = "swing"


def compute_trailing_stop(
    entry_price: float,
    current_price: float,
    peak_price: float,
    daily_price_vol: float,
    previous_stop: Optional[float] = None,
    trade_horizon: str = "swing",
    config: Optional[VolTrailingStopConfig] = None,
    regime: str = "",
) -> TrailingStopState:
    """Compute the volatility-based trailing stop for a position.

    Parameters
    ----------
    entry_price : float
        Original entry price.
    current_price : float
        Current market price.
    peak_price : float
        Highest price since entry (updated externally).
    daily_price_vol : float
        Daily price volatility as a fraction (e.g. 0.02 for 2%).
    previous_stop : float | None
        Previous trailing stop level (stop only ratchets up).
    trade_horizon : str
        ``"swing"`` or ``"positional"``.
    config : VolTrailingStopConfig | None
    regime : str
        Current market regime (e.g. ``"trending_bull"``).
        In bull regimes, tighter profit-lock parameters are used to
        book profits earlier for contra-regime recycling.

    Returns
    -------
    TrailingStopState
    """
    cfg = config or VolTrailingStopConfig()

    # Update peak
    peak = max(peak_price, current_price)

    # Select vol multiplier based on trade horizon
    if trade_horizon == "positional":
        vol_mult = cfg.vol_multiplier_positional
    else:
        vol_mult = cfg.vol_multiplier_swing

    # Guard: ensure daily_vol is positive
    if daily_price_vol <= 0:
        daily_price_vol = 0.02  # fallback 2%

    # Base stop distance = vol_multiplier × daily_vol × peak_price
    stop_distance_pct = vol_mult * daily_price_vol
    stop_distance_pct = min(stop_distance_pct, cfg.max_stop_distance_pct)
    stop_distance_pct = max(stop_distance_pct, cfg.min_stop_distance_pct)

    base_stop = peak * (1 - stop_distance_pct)

    # Profit-lock check: if unrealised profit > threshold, tighten stop
    profit_locked = False
    profit_vol_units = (peak - entry_price) / (daily_price_vol * entry_price) if daily_price_vol > 0 and entry_price > 0 else 0

    # Contra-regime: use tighter thresholds in bull to book profits earlier
    _regime_lower = (regime or "").lower()
    _is_bull = "bull" in _regime_lower
    _activation = cfg.bull_profit_lock_activation_vol if _is_bull else cfg.profit_lock_activation_vol
    _lock_dist = cfg.bull_profit_lock_distance_vol if _is_bull else cfg.profit_lock_distance_vol

    if profit_vol_units >= _activation:
        profit_locked = True
        lock_distance_pct = _lock_dist * daily_price_vol
        lock_distance_pct = max(lock_distance_pct, cfg.min_stop_distance_pct)
        lock_stop = peak * (1 - lock_distance_pct)
        base_stop = max(base_stop, lock_stop)

    # Stop only ratchets up (never moves down)
    if previous_stop is not None:
        new_stop = max(base_stop, previous_stop)
    else:
        new_stop = base_stop

    # Floor: stop never below entry (break-even guarantee after profit-lock)
    if profit_locked:
        new_stop = max(new_stop, entry_price)

    return TrailingStopState(
        symbol="",
        entry_price=entry_price,
        peak_price=peak,
        current_stop=round(new_stop, 2),
        daily_vol=daily_price_vol,
        vol_multiplier=vol_mult,
        profit_locked=profit_locked,
        trade_horizon=trade_horizon,
    )


def should_exit(state: TrailingStopState, current_price: float) -> bool:
    """Return True if current price has breached the trailing stop."""
    return current_price <= state.current_stop
