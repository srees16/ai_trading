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

# Vince regime shrink/stretch multipliers (Vince Ch.8: rotating markets)
# DISABLED: Set to 1.0 for all regimes.  Regime de-risking is handled by:
#   1) Carver vol targeting (auto-shrinks when instrument vol rises)
#   2) Regime-adaptive leverage caps (Bull=7x, Range=5x, Bear=2x, Crisis=0.5x)
#   3) Portfolio vol monitor (smooth quadratic DD curve)
#   4) VIX pipeline scaling (caution/panic gates)
#   5) Correlation spike detection
#   6) Vince insurance floor (20% DD smooth halt)
# Multiplied regime adjustments on top of these 6 layers over-constrains CAGR.
VINCE_REGIME_MULTIPLIERS = {
    "TRENDING_BULL":  1.00,     # full — all regimes at 1.0, regime caps do the work
    "TRENDING_BEAR":  1.00,     # leverage cap limits to 2x (29% of max)
    "RANGE_BOUND":    1.00,     # leverage cap limits to 5x (71% of max)
    "HIGH_VOLATILITY": 1.00,    # vol targeting auto-shrinks + leverage cap at 2x
    "CRISIS":         1.00,     # leverage cap limits to 0.5x (7% of max) + VIX panic gate
}

# G12: Read max leverage from Config; fallback to 1.0 (not 2.0)
try:
    from config import Config as _Cfg
    MAX_LEVERAGE = getattr(_Cfg, "CARVER_MAX_LEVERAGE", 1.0)
except Exception:
    MAX_LEVERAGE = 1.0


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
    regime: str = "",
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

    # Step 3b: Vince regime shrink/stretch adjustment
    if regime:
        try:
            from config import Config
            if getattr(Config, "VINCE_REGIME_SHRINK_ENABLED", False):
                regime_key = regime.upper().replace(" ", "_")
                regime_mult = VINCE_REGIME_MULTIPLIERS.get(regime_key, 1.0)
                portfolio_position *= regime_mult
        except Exception:
            pass

    # Step 4: Round to whole shares
    target_quantity = round(portfolio_position)

    # Step 5: Leverage limit — cap notional at max_leverage × capital
    # GAP-3 FIX: Apply regime-adaptive leverage cap before position sizing
    # so max_leverage respects bull/bear/crisis limits from Config.
    if capital > 0:
        try:
            from config import Config as _LevCfg
            regime_upper = (regime or "").upper().replace(" ", "_")
            if regime_upper in ("BEAR", "HIGH_VOLATILITY"):
                regime_lev_cap = getattr(_LevCfg, 'LEVERAGE_BEAR_MAX', 2.0)
            elif regime_upper == "CRISIS":
                regime_lev_cap = getattr(_LevCfg, 'LEVERAGE_CRISIS_MAX', 0.5)
            else:
                regime_lev_cap = getattr(_LevCfg, 'LEVERAGE_BULL_MAX', max_leverage)
            effective_leverage = min(max_leverage, regime_lev_cap)
        except Exception:
            effective_leverage = max_leverage
        max_notional = capital * effective_leverage
        max_qty_by_leverage = int(max_notional / price)
        target_quantity = max(-max_qty_by_leverage, min(target_quantity, max_qty_by_leverage))

    # Step 6: Floor at 0 unless short selling is enabled
    try:
        from config import Config
        allow_short = getattr(Config, "SHORT_SELLING_ENABLED", False)
    except Exception:
        allow_short = False
    if not allow_short:
        target_quantity = max(0, target_quantity)

    notional = target_quantity * price

    # Step 7: Position inertia — asymmetric thresholds
    # Scale-UP: full inertia (10%) to avoid noise-driven entry increases
    # Scale-DOWN: lower threshold (5%) to allow faster profit-taking / loss-cutting
    # FIX: Inertia must NOT override leverage limit — clamp after inertia restore
    trade_required = True
    if current_quantity > 0:
        pct_change = abs(target_quantity - current_quantity) / current_quantity
        scaling_down = target_quantity < current_quantity
        effective_threshold = (inertia_threshold * 0.5) if scaling_down else inertia_threshold
        if pct_change < effective_threshold:
            trade_required = False
            # Restore current qty but re-enforce leverage cap
            restored_qty = current_quantity
            if capital > 0:
                max_notional = capital * max_leverage
                max_qty_by_leverage = int(max_notional / price)
                restored_qty = min(restored_qty, max_qty_by_leverage)
            target_quantity = max(0, restored_qty)
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
    regime: str = "",
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
            regime=regime,
        )
        results[sym] = ps
        total_notional += ps.notional_value

    # ── Tier 1 Gap 3: Gross notional ceiling ──
    # FIX-6A: Read max leverage from Config — was hardcoded 2.0, blocked 50%+ CAGR.
    try:
        from config import Config as _GrossCfg
        gross_cap_multiplier = getattr(_GrossCfg, 'CARVER_MAX_LEVERAGE', 2.0)
        # GAP-3: Apply regime-adaptive leverage cap to batch sizing too
        _regime_hint = getattr(_GrossCfg, '_CURRENT_REGIME', '')
        if _regime_hint:
            _ru = _regime_hint.upper().replace(' ', '_')
            if _ru in ('BEAR', 'HIGH_VOLATILITY'):
                gross_cap_multiplier = min(gross_cap_multiplier,
                                           getattr(_GrossCfg, 'LEVERAGE_BEAR_MAX', 2.0))
            elif _ru == 'CRISIS':
                gross_cap_multiplier = min(gross_cap_multiplier,
                                           getattr(_GrossCfg, 'LEVERAGE_CRISIS_MAX', 0.5))
    except Exception:
        gross_cap_multiplier = 2.0
    max_notional = gross_cap_multiplier * capital
    if capital > 0 and total_notional > max_notional:
        excess_ratio = total_notional / max_notional  # e.g. 1.5 = 50% over
        logger.warning(
            "Gross notional %.0f exceeds %.1fx capital %.0f -- scaling by forecast strength",
            total_notional, gross_cap_multiplier, capital,
        )
        from dataclasses import replace as _dc_replace
        # Compute per-instrument scale: weaker forecasts get cut more
        forecast_abs = {sym: abs(ps.combined_forecast) + 1e-6 for sym, ps in results.items()}
        max_fc = max(forecast_abs.values())
        for sym, ps in results.items():
            # Proportional scale: strongest forecast gets minimal cut
            fc_ratio = forecast_abs[sym] / max_fc  # 0 to 1
            # Blend: uniform_scale * (1 - alpha) + proportional * alpha
            uniform_scale = max_notional / total_notional
            proportional_scale = min(1.0, uniform_scale * (0.5 + 0.5 * fc_ratio))
            scaled_qty = int(ps.target_quantity * proportional_scale)
            results[sym] = _dc_replace(
                ps,
                target_quantity=scaled_qty,
                trade_delta=scaled_qty - ps.current_quantity,
                trade_required=abs(scaled_qty - ps.current_quantity) > 0,
                notional_value=abs(scaled_qty) * ps.price if ps.price else 0.0,
            )
        # G1 FIX: Recompute total_notional from scaled positions
        total_notional = sum(
            ps.notional_value for ps in results.values()
        )

    # ── Vince Leverage Space Cap ──────────────────────────────
    # Apply Vince secure_f as an additional leverage ceiling.
    # If secure_f recommends lower leverage than Carver sizing,
    # scale positions down to respect drawdown constraints.
    try:
        from config import Config as _VCfg
        if getattr(_VCfg, 'VINCE_REGIME_SHRINK_ENABLED', False) and capital > 0:
            from strategies.vince_leverage import (
                compute_active_equity_ratio,
                compute_leverage_from_vince,
            )
            hwm = getattr(_VCfg, '_HWM', capital)  # HWM tracked externally
            insurance = getattr(
                _VCfg, 'VINCE_INSURANCE_PCT_IND', 0.15
            )
            active_ratio, _ = compute_active_equity_ratio(
                capital, hwm, insurance
            )
            if active_ratio < 1.0:
                # Equity is below HWM — scale all positions by active ratio
                from dataclasses import replace as _dc_replace2
                scale = max(0.0, active_ratio)
                for sym, ps in results.items():
                    scaled_qty = max(0, int(ps.target_quantity * scale))
                    results[sym] = _dc_replace2(
                        ps,
                        target_quantity=scaled_qty,
                        trade_delta=scaled_qty - ps.current_quantity,
                        trade_required=abs(scaled_qty - ps.current_quantity) > 0,
                        notional_value=abs(scaled_qty) * ps.price if ps.price else 0.0,
                    )
                logger.info(
                    "Vince active equity %.2f (HWM=%.0f, floor=%.0f) → scaled by %.1f%%",
                    active_ratio, hwm, hwm * (1 - insurance), scale * 100,
                )
    except ImportError:
        pass
    except Exception as _ve:
        logger.warning("Vince leverage cap skipped: %s", _ve)

    # Log summary
    trades_needed = sum(1 for ps in results.values() if ps.trade_required and ps.trade_delta != 0)
    logger.info(
        "Position sizing: %d instruments, %d trades needed, "
        "total notional ₹%.0f / capital ₹%.0f (%.0f%% deployed)",
        len(results), trades_needed, total_notional, capital,
        (total_notional / capital * 100) if capital > 0 else 0,
    )
    return results
