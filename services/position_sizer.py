"""
Carver Position Sizer — Volatility-targeted continuous position sizing.

Implements Carver Chapter 11 'Position Sizing':

    vol_scalar  = daily_cash_vol_target / instrument_value_volatility
    subsys_pos  = (combined_forecast / 10) × vol_scalar
    portfolio_pos = subsys_pos × instrument_weight × IDM

Position inertia: only trade if change > INERTIA_THRESHOLD (10%) to
avoid excessive turnover from small forecast jiggles.

Final quantity is rounded to the nearest whole share for NSE.

Integration:
  - Uses VolatilityTarget  →  daily_cash_vol_target
  - Uses instrument_volatility  →  instrument_value_volatility
  - Uses forecast_combiner  →  combined_forecast (-20 to +20)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
FORECAST_SCALAR = 10.0          # Carver: forecast is expressed in units of 10
INERTIA_THRESHOLD = 0.10        # Only re-trade if target changes > 10%
MAX_FORECAST_ABS = 20.0         # Hard cap on forecast magnitude
MAX_LEVERAGE = 2.0              # Absolute leverage limit (notional / capital)


@dataclass
class PositionSize:
    """Result of the Carver position sizing formula for one instrument."""
    symbol: str
    combined_forecast: float      # capped combined forecast (-20 to +20)
    vol_scalar: float             # daily_cash_vol_target / instr_value_vol
    subsystem_position: float     # (forecast / 10) × vol_scalar  (fractional shares)
    instrument_weight: float      # 0 to 1
    idm: float                    # instrument diversification multiplier
    portfolio_position: float     # subsys × weight × IDM  (fractional shares)
    target_quantity: int          # rounded for NSE
    current_quantity: int = 0     # current holding for inertia check
    trade_required: bool = True   # False if within inertia threshold
    trade_delta: int = 0          # shares to buy (+) or sell (−)
    notional_value: float = 0.0   # target_quantity × price
    price: float = 0.0


@dataclass
class PositionSizerConfig:
    """Top-level config for the Carver position sizer."""
    inertia_threshold: float = INERTIA_THRESHOLD
    max_leverage: float = MAX_LEVERAGE
    # Default instrument weights when no optimised weights available
    default_instrument_weight: float = 0.10   # 10% each (implies ~10 position portfolio)
    # Instrument Diversification Multiplier
    # For 6-10 instruments at avg correlation ~0.4, IDM ≈ 1.5-1.9
    default_idm: float = 1.6


def compute_position_size(
    symbol: str,
    combined_forecast: float,
    instrument_value_vol: float,
    daily_cash_vol_target: float,
    price: float,
    capital: float,
    instrument_weight: float = 0.10,
    idm: float = 1.6,
    current_quantity: int = 0,
    inertia_threshold: float = INERTIA_THRESHOLD,
    max_leverage: float = MAX_LEVERAGE,
) -> PositionSize:
    """Compute Carver-style position size for a single instrument.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    combined_forecast : float
        Combined forecast value (−20 to +20).
    instrument_value_vol : float
        Daily instrument value volatility = price × daily_price_vol (₹).
    daily_cash_vol_target : float
        Daily cash vol target from VolatilityTarget (₹).
    price : float
        Current price of the instrument.
    capital : float
        Current total capital (for leverage limit).
    instrument_weight : float
        Portfolio weight [0, 1].
    idm : float
        Instrument diversification multiplier.
    current_quantity : int
        Current holding quantity (for inertia check).
    inertia_threshold : float
        Minimum % change in target to trigger a trade.
    max_leverage : float
        Maximum portfolio leverage.

    Returns
    -------
    PositionSize
    """
    # Guard: avoid division by zero
    if instrument_value_vol <= 0 or price <= 0 or daily_cash_vol_target <= 0:
        return PositionSize(
            symbol=symbol,
            combined_forecast=combined_forecast,
            vol_scalar=0.0,
            subsystem_position=0.0,
            instrument_weight=instrument_weight,
            idm=idm,
            portfolio_position=0.0,
            target_quantity=0,
            current_quantity=current_quantity,
            trade_required=False,
            trade_delta=0,
            price=price,
        )

    # ── Core Carver formulas ──────────────────────────────────
    # Step 1: vol_scalar
    vol_scalar = daily_cash_vol_target / instrument_value_vol

    # Step 2: subsystem position (before instrument weight / IDM)
    capped_forecast = max(-MAX_FORECAST_ABS, min(combined_forecast, MAX_FORECAST_ABS))
    subsystem_position = (capped_forecast / FORECAST_SCALAR) * vol_scalar

    # Step 3: portfolio position = subsystem × weight × IDM
    portfolio_position = subsystem_position * instrument_weight * idm

    # Step 4: Round to whole shares
    target_quantity = round(portfolio_position)

    # Step 5: Leverage limit — cap notional at max_leverage × capital
    if capital > 0:
        max_notional = capital * max_leverage
        max_qty_by_leverage = int(max_notional / price)
        target_quantity = max(-max_qty_by_leverage, min(target_quantity, max_qty_by_leverage))

    # Step 6: For NSE long-only, floor at 0 (no shorting)
    target_quantity = max(0, target_quantity)

    notional = target_quantity * price

    # Step 7: Position inertia — asymmetric thresholds
    # Scale-UP: full inertia (10%) to avoid noise-driven entry increases
    # Scale-DOWN: lower threshold (5%) to allow faster profit-taking / loss-cutting
    trade_required = True
    if current_quantity > 0:
        pct_change = abs(target_quantity - current_quantity) / current_quantity
        scaling_down = target_quantity < current_quantity
        effective_threshold = (inertia_threshold * 0.5) if scaling_down else inertia_threshold
        if pct_change < effective_threshold:
            trade_required = False
            target_quantity = current_quantity  # keep existing
            notional = target_quantity * price
    elif target_quantity == 0 and current_quantity == 0:
        trade_required = False

    trade_delta = target_quantity - current_quantity

    return PositionSize(
        symbol=symbol,
        combined_forecast=capped_forecast,
        vol_scalar=round(vol_scalar, 4),
        subsystem_position=round(subsystem_position, 2),
        instrument_weight=instrument_weight,
        idm=idm,
        portfolio_position=round(portfolio_position, 2),
        target_quantity=target_quantity,
        current_quantity=current_quantity,
        trade_required=trade_required,
        trade_delta=trade_delta,
        notional_value=round(notional, 2),
        price=price,
    )


def compute_position_sizes_batch(
    forecasts: Dict[str, float],
    volatilities: Dict[str, float],
    prices: Dict[str, float],
    daily_cash_vol_target: float,
    capital: float,
    instrument_weights: Optional[Dict[str, float]] = None,
    idm: float = 1.6,
    current_holdings: Optional[Dict[str, int]] = None,
    config: Optional[PositionSizerConfig] = None,
) -> Dict[str, PositionSize]:
    """Batch position sizing for all instruments.

    Parameters
    ----------
    forecasts : dict[str, float]
        Combined forecasts per symbol.
    volatilities : dict[str, float]
        Instrument value volatility per symbol (₹).
    prices : dict[str, float]
        Current prices per symbol.
    daily_cash_vol_target : float
        From VolatilityTarget.
    capital : float
        Current total capital.
    instrument_weights : dict[str, float] | None
        Per-symbol weights [0, 1].  Default: equal weight.
    idm : float
        Instrument diversification multiplier.
    current_holdings : dict[str, int] | None
        Current holding quantities for inertia check.
    config : PositionSizerConfig | None

    Returns
    -------
    dict[str, PositionSize]
    """
    cfg = config or PositionSizerConfig()
    current_holdings = current_holdings or {}

    # Default equal weights if not provided
    if instrument_weights is None:
        n = len(forecasts)
        w = 1.0 / n if n > 0 else cfg.default_instrument_weight
        instrument_weights = {sym: w for sym in forecasts}

    effective_idm = idm or cfg.default_idm

    results: Dict[str, PositionSize] = {}
    total_notional = 0.0

    for sym, forecast in forecasts.items():
        vol = volatilities.get(sym, 0.0)
        price = prices.get(sym, 0.0)
        weight = instrument_weights.get(sym, cfg.default_instrument_weight)
        current_qty = current_holdings.get(sym, 0)

        ps = compute_position_size(
            symbol=sym,
            combined_forecast=forecast,
            instrument_value_vol=vol,
            daily_cash_vol_target=daily_cash_vol_target,
            price=price,
            capital=capital,
            instrument_weight=weight,
            idm=effective_idm,
            current_quantity=current_qty,
            inertia_threshold=cfg.inertia_threshold,
            max_leverage=cfg.max_leverage,
        )
        results[sym] = ps
        total_notional += ps.notional_value

    # Log summary
    trades_needed = sum(1 for ps in results.values() if ps.trade_required and ps.trade_delta != 0)
    logger.info(
        "Position sizing: %d instruments, %d trades needed, "
        "total notional ₹%.0f / capital ₹%.0f (%.0f%% deployed)",
        len(results), trades_needed, total_notional, capital,
        (total_notional / capital * 100) if capital > 0 else 0,
    )
    return results
