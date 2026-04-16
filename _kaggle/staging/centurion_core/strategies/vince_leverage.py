"""
Vince Leverage Space Trading Model — From Ralph Vince's
"The Leverage Space Trading Model: Reconciling Portfolio Management
Strategies and Economic Theory".

Implements:
  1. Optimal f — Kelly-generalized fraction that maximizes geometric growth
  2. Secure f — Max-drawdown-constrained f (Vince Ch. 4)
  3. Leverage Space — Multi-dimensional f-vector for portfolio (Vince Ch. 5)
  4. Drawdown Management — Active equity curve with insurance floor (Vince Ch. 8)
  5. Regime-Adaptive Leverage — shrink/stretch f per market regime (Vince Ch. 8)

Integration:
  - IND: Caps position_sizer's Carver size by secure_f → Kite auto-order
  - US: Provides leverage_recommendation in API results for manual sizing
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class VinceAnalysis:
    """Complete Vince leverage space analysis for a single symbol."""
    symbol: str
    optimal_f: float           # Optimal fraction (0-1) for max geometric growth
    secure_f: float            # Drawdown-constrained fraction (≤ optimal_f)
    kelly_fraction: float      # Traditional Kelly for comparison
    largest_loss: float        # Worst single-trade loss (denominator for f)
    win_rate: float            # Historical win rate
    avg_win_loss_ratio: float  # Avg win / avg loss
    expected_geometric_growth: float  # TWR^(1/N) - 1 at secure_f
    max_drawdown_at_f: float   # Expected max DD at the chosen f
    leverage_recommendation: float  # Final recommended leverage multiplier
    active_equity_ratio: float # Current equity / HWM (1.0 = at peak)
    insurance_floor_pct: float # % of HWM reserved as floor


@dataclass
class PortfolioLeverageSpace:
    """Multi-asset leverage space analysis."""
    per_symbol: Dict[str, VinceAnalysis]
    portfolio_optimal_f: float  # Portfolio-level optimal f
    portfolio_secure_f: float   # Portfolio-level secure f
    portfolio_leverage: float   # Recommended portfolio leverage
    joint_drawdown_limit: float # Joint max drawdown tolerance


# ═══════════════════════════════════════════════════════════════
# Optimal f (Vince Ch. 3: "The Optimal f")
# ═══════════════════════════════════════════════════════════════

def compute_optimal_f(
    trade_returns: np.ndarray,
    precision: float = 0.01,
) -> Tuple[float, float]:
    """
    Find optimal f that maximizes Terminal Wealth Relative (TWR).

    Vince Ch. 3: "The optimal f is that fraction of your equity
    which, when risked on each trade, maximizes the geometric
    mean return."

    TWR = Product[(1 + f * return_i / |largest_loss|)]

    Args:
        trade_returns: array of trade P&L (positive = win, negative = loss)
        precision: search step size

    Returns:
        (optimal_f, max_twr_growth)
    """
    if len(trade_returns) < 10:
        return 0.0, 0.0

    largest_loss = abs(float(np.min(trade_returns)))
    if largest_loss < 1e-10:
        return 0.0, 0.0

    best_f = 0.0
    best_growth = 0.0

    # Search f from 0.01 to 1.0
    f = precision
    while f <= 1.0:
        twr = 1.0
        for ret in trade_returns:
            hpr = 1.0 + f * (-ret / largest_loss)
            if hpr <= 0:
                twr = 0.0
                break
            twr *= hpr

        if twr > 0:
            geometric_growth = twr ** (1.0 / len(trade_returns)) - 1.0
            if geometric_growth > best_growth:
                best_growth = geometric_growth
                best_f = f

        f += precision

    return round(best_f, 4), round(best_growth, 6)


# ═══════════════════════════════════════════════════════════════
# Secure f (Vince Ch. 4: "Drawdown Constraint")
# ═══════════════════════════════════════════════════════════════

def compute_secure_f(
    trade_returns: np.ndarray,
    max_dd_tolerance: float = 0.20,
    optimal_f: Optional[float] = None,
    precision: float = 0.01,
) -> float:
    """
    Compute secure f — the largest f that keeps expected drawdown
    within the specified tolerance.

    Vince Ch. 4: "Secure f is optimal f reduced to limit drawdown
    to a tolerable level. You sacrifice some geometric growth for
    drawdown control."

    Method: simulate equity curve at each f, measure max drawdown,
    select the largest f where DD ≤ tolerance.

    Args:
        trade_returns: array of trade P&L
        max_dd_tolerance: maximum acceptable drawdown (0.20 = 20%)
        optimal_f: pre-computed optimal f (skip search above it)
        precision: search step size

    Returns:
        secure_f value (0 to optimal_f)
    """
    if len(trade_returns) < 10:
        return 0.0

    largest_loss = abs(float(np.min(trade_returns)))
    if largest_loss < 1e-10:
        return 0.0

    if optimal_f is None:
        optimal_f, _ = compute_optimal_f(trade_returns, precision)

    if optimal_f <= 0:
        return 0.0

    upper = optimal_f
    best_f = 0.0

    f = precision
    while f <= upper:
        # Simulate equity curve
        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        for ret in trade_returns:
            hpr = 1.0 + f * (-ret / largest_loss)
            if hpr <= 0:
                max_dd = 1.0
                break
            equity *= hpr
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        if max_dd <= max_dd_tolerance:
            best_f = f

        f += precision

    return round(best_f, 4)


# ═══════════════════════════════════════════════════════════════
# Kelly Fraction (for comparison)
# ═══════════════════════════════════════════════════════════════

def compute_kelly_fraction(
    win_rate: float,
    avg_win_loss_ratio: float,
) -> float:
    """
    Traditional Kelly criterion: f* = (p × b - q) / b
    where p = win_rate, q = 1-p, b = avg_win/avg_loss.
    """
    if avg_win_loss_ratio <= 0:
        return 0.0
    q = 1.0 - win_rate
    f = (win_rate * avg_win_loss_ratio - q) / avg_win_loss_ratio
    return max(0.0, min(1.0, round(f, 4)))


# ═══════════════════════════════════════════════════════════════
# Active Equity Curve Management (Vince Ch. 8)
# ═══════════════════════════════════════════════════════════════

def compute_active_equity_ratio(
    current_equity: float,
    high_water_mark: float,
    insurance_pct: float = 0.15,
) -> Tuple[float, float]:
    """
    Active equity management with insurance floor.

    Vince Ch. 8: "Active equity = current equity - insurance floor.
    When active equity approaches zero, you must stop trading."

    Insurance floor = HWM × (1 - insurance_pct)
    Active ratio = (equity - floor) / (HWM - floor)

    When ratio → 0: halt trading (equity at floor)
    When ratio = 1.0: full sizing at HWM
    When ratio > 1.0: new HWM → floor rises

    Args:
        current_equity: current portfolio value
        high_water_mark: highest equity ever reached
        insurance_pct: fraction of HWM protected (0.15 = protect 85%)

    Returns:
        (active_ratio [0, 1+], floor_value)
    """
    if high_water_mark <= 0:
        return 1.0, 0.0

    floor = high_water_mark * (1.0 - insurance_pct)
    active_range = high_water_mark - floor
    if active_range <= 0:
        return 0.0, floor

    active_equity = current_equity - floor
    ratio = max(0.0, active_equity / active_range)

    return round(ratio, 4), round(floor, 2)


def compute_leverage_from_vince(
    secure_f: float,
    active_equity_ratio: float,
    max_leverage: float = 4.0,
    regime: str = "",
) -> float:
    """
    Convert Vince secure_f + active equity ratio into a leverage multiplier.

    Leverage = secure_f × active_equity_ratio × regime_multiplier

    Capped at max_leverage. In crisis/bear, regime multiplier
    aggressively dampens.

    Args:
        secure_f: drawdown-constrained fraction (0-1)
        active_equity_ratio: from active equity curve (0-1+)
        max_leverage: hard cap on leverage
        regime: current market regime

    Returns:
        leverage multiplier (0 to max_leverage)
    """
    # Regime multipliers (from Vince Ch. 8: rotating markets)
    regime_mults = {
        "TRENDING_BULL": 1.00,
        "TRENDING_BEAR": 0.38,
        "RANGE_BOUND": 0.73,
        "HIGH_VOLATILITY": 0.27,
        "CRISIS": 0.04,
    }

    regime_mult = regime_mults.get(regime.upper().replace(" ", "_"), 0.75)

    # Base leverage from secure_f (scale: f=0.3 → ~2× leverage)
    base_leverage = min(max_leverage, secure_f * max_leverage * 1.5)

    # Dampen by active equity ratio
    leverage = base_leverage * min(1.0, active_equity_ratio) * regime_mult

    return round(max(0.0, min(max_leverage, leverage)), 4)


# ═══════════════════════════════════════════════════════════════
# Drawdown Estimation (Vince Ch. 6)
# ═══════════════════════════════════════════════════════════════

def estimate_max_drawdown_at_f(
    trade_returns: np.ndarray,
    f: float,
    n_simulations: int = 500,
    sequence_length: int = 100,
) -> float:
    """
    Monte Carlo estimate of expected max drawdown at a given f.

    Vince Ch. 6: "The probability of a drawdown of magnitude D
    approaches 1 as the number of trades increases."

    Shuffles trade returns and simulates equity curves to estimate
    the expected worst drawdown.

    Returns:
        expected max drawdown (0 to 1)
    """
    if len(trade_returns) < 5 or f <= 0:
        return 0.0

    largest_loss = abs(float(np.min(trade_returns)))
    if largest_loss < 1e-10:
        return 0.0

    max_dds = []
    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        shuffled = rng.choice(trade_returns, size=sequence_length, replace=True)
        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        for ret in shuffled:
            hpr = 1.0 + f * (-ret / largest_loss)
            if hpr <= 0:
                max_dd = 1.0
                break
            equity *= hpr
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        max_dds.append(max_dd)

    return round(float(np.mean(max_dds)), 4)


# ═══════════════════════════════════════════════════════════════
# Single-Symbol Analysis
# ═══════════════════════════════════════════════════════════════

def compute_vince_analysis(
    symbol: str,
    trade_returns: np.ndarray,
    current_equity: float,
    high_water_mark: float,
    max_dd_tolerance: float = 0.20,
    insurance_pct: float = 0.15,
    max_leverage: float = 4.0,
    regime: str = "",
) -> VinceAnalysis:
    """
    Full Vince leverage space analysis for a single symbol.

    Args:
        symbol: ticker
        trade_returns: historical trade P&L array
        current_equity: current portfolio value
        high_water_mark: highest portfolio value
        max_dd_tolerance: max DD for secure_f
        insurance_pct: active equity insurance pct
        max_leverage: hard leverage cap
        regime: current market regime

    Returns:
        VinceAnalysis with all metrics
    """
    if len(trade_returns) < 10:
        active_ratio, _ = compute_active_equity_ratio(
            current_equity, high_water_mark, insurance_pct
        )
        return VinceAnalysis(
            symbol=symbol, optimal_f=0.0, secure_f=0.0,
            kelly_fraction=0.0, largest_loss=0.0,
            win_rate=0.0, avg_win_loss_ratio=0.0,
            expected_geometric_growth=0.0,
            max_drawdown_at_f=0.0,
            leverage_recommendation=0.0,
            active_equity_ratio=active_ratio,
            insurance_floor_pct=insurance_pct,
        )

    # Win/loss stats
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = abs(float(np.mean(losses))) if len(losses) > 0 else 1e-10
    avg_wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # Optimal f
    opt_f, opt_growth = compute_optimal_f(trade_returns)

    # Secure f (DD-constrained)
    sec_f = compute_secure_f(trade_returns, max_dd_tolerance, opt_f)

    # Kelly for comparison
    kelly = compute_kelly_fraction(win_rate, avg_wl_ratio)

    # Active equity
    active_ratio, floor_val = compute_active_equity_ratio(
        current_equity, high_water_mark, insurance_pct
    )

    # Expected DD at secure_f
    est_dd = estimate_max_drawdown_at_f(trade_returns, sec_f)

    # Leverage recommendation
    lev = compute_leverage_from_vince(
        sec_f, active_ratio, max_leverage, regime
    )

    largest_loss = abs(float(np.min(trade_returns)))

    logger.info(
        "Vince %s: opt_f=%.3f sec_f=%.3f kelly=%.3f active=%.2f → lev=%.2f× (regime=%s)",
        symbol, opt_f, sec_f, kelly, active_ratio, lev, regime or "N/A",
    )

    return VinceAnalysis(
        symbol=symbol,
        optimal_f=opt_f,
        secure_f=sec_f,
        kelly_fraction=kelly,
        largest_loss=largest_loss,
        win_rate=win_rate,
        avg_win_loss_ratio=round(avg_wl_ratio, 3),
        expected_geometric_growth=opt_growth,
        max_drawdown_at_f=est_dd,
        leverage_recommendation=lev,
        active_equity_ratio=active_ratio,
        insurance_floor_pct=insurance_pct,
    )


# ═══════════════════════════════════════════════════════════════
# Portfolio Leverage Space (Vince Ch. 5)
# ═══════════════════════════════════════════════════════════════

def compute_portfolio_leverage_space(
    trade_returns_dict: Dict[str, np.ndarray],
    current_equity: float,
    high_water_mark: float,
    max_dd_tolerance: float = 0.20,
    insurance_pct: float = 0.15,
    max_leverage: float = 4.0,
    regime: str = "",
) -> PortfolioLeverageSpace:
    """
    Multi-asset leverage space — find portfolio-level optimal f.

    Vince Ch. 5: "The leverage space for a portfolio of N markets
    is an N-dimensional space where each axis is the f for that
    market."

    Simplified approach: compute per-symbol Vince analysis, then
    combine via weighted average of secure_f values.

    Returns:
        PortfolioLeverageSpace with per-symbol and portfolio metrics
    """
    per_symbol = {}
    secure_fs = []
    weights = []

    for symbol, returns in trade_returns_dict.items():
        analysis = compute_vince_analysis(
            symbol, returns,
            current_equity, high_water_mark,
            max_dd_tolerance, insurance_pct,
            max_leverage, regime,
        )
        per_symbol[symbol] = analysis
        if analysis.secure_f > 0:
            secure_fs.append(analysis.secure_f)
            weights.append(1.0)

    # Portfolio-level f: conservative — use harmonic mean of secure_f values
    if secure_fs:
        n = len(secure_fs)
        harmonic = n / sum(1.0 / f for f in secure_fs)
        portfolio_opt = float(np.mean(secure_fs))
        portfolio_sec = min(harmonic, portfolio_opt)
    else:
        portfolio_opt = 0.0
        portfolio_sec = 0.0

    # Portfolio active equity
    active_ratio, _ = compute_active_equity_ratio(
        current_equity, high_water_mark, insurance_pct
    )

    portfolio_lev = compute_leverage_from_vince(
        portfolio_sec, active_ratio, max_leverage, regime
    )

    # Joint DD limit = tolerance scaled by active ratio
    joint_dd = max_dd_tolerance * min(1.0, active_ratio)

    return PortfolioLeverageSpace(
        per_symbol=per_symbol,
        portfolio_optimal_f=round(portfolio_opt, 4),
        portfolio_secure_f=round(portfolio_sec, 4),
        portfolio_leverage=portfolio_lev,
        joint_drawdown_limit=round(joint_dd, 4),
    )


# ═══════════════════════════════════════════════════════════════
# Batch Processing for Pipeline Integration
# ═══════════════════════════════════════════════════════════════

def compute_vince_leverage_batch(
    ohlcv_dict: Dict[str, pd.DataFrame],
    current_equity: float,
    high_water_mark: float,
    max_dd_tolerance: float = 0.20,
    insurance_pct: float = 0.15,
    max_leverage: float = 4.0,
    regime: str = "",
    trade_lookback: int = 60,
) -> Dict[str, VinceAnalysis]:
    """
    Compute Vince leverage analysis for multiple symbols using
    OHLCV return series as proxy for trade returns.

    For symbols without actual trade history, daily returns
    serve as a conservative proxy.

    Args:
        ohlcv_dict: {symbol: OHLCV DataFrame}
        current_equity: current portfolio value
        high_water_mark: highest portfolio value
        max_dd_tolerance: max drawdown tolerance
        insurance_pct: active equity insurance
        max_leverage: hard cap
        regime: market regime
        trade_lookback: days of returns to use

    Returns:
        {symbol: VinceAnalysis}
    """
    results = {}

    for symbol, df in ohlcv_dict.items():
        if df is None or len(df) < 20:
            results[symbol] = VinceAnalysis(
                symbol=symbol, optimal_f=0.0, secure_f=0.0,
                kelly_fraction=0.0, largest_loss=0.0,
                win_rate=0.0, avg_win_loss_ratio=0.0,
                expected_geometric_growth=0.0,
                max_drawdown_at_f=0.0, leverage_recommendation=0.0,
                active_equity_ratio=1.0, insurance_floor_pct=insurance_pct,
            )
            continue

        close = df["Close"].values.astype(float)
        returns = np.diff(close[-trade_lookback:]) / np.maximum(
            np.abs(close[-trade_lookback:-1]), 1e-10
        )

        results[symbol] = compute_vince_analysis(
            symbol, returns,
            current_equity, high_water_mark,
            max_dd_tolerance, insurance_pct,
            max_leverage, regime,
        )

    return results
