"""
Vince Leverage Space Trading Model — From Ralph Vince's
"The Leverage Space Trading Model: Reconciling Portfolio
Management Strategies and Economic Theory".

Implements growth-optimal position sizing with drawdown constraints:

  1. Optimal f — Growth-maximizing fraction per trade (Kelly generalization)
  2. Terminal Wealth Relative (TWR) — Geometric growth measure
  3. Leverage Space Portfolio — Multi-asset optimal f allocation
  4. Drawdown-Constrained Optimization — Maximize TWR within DD limit
  5. Monte Carlo Probability of Profit — Simulate N paths at given horizon
  6. Dynamic Leverage Adjustment — Continuous f recalibration

Integration:
  - IND: replaces heuristic Vince regime multipliers → dynamic sizing → Kite
  - US: Monte Carlo probability + optimal f in API → display on UI
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class OptimalFResult:
    """Result of optimal f computation for a single instrument."""
    symbol: str
    optimal_f: float               # Growth-maximizing fraction (0.0-1.0)
    half_f: float                  # Conservative half-Kelly
    twr: float                     # Terminal Wealth Relative at optimal f
    geometric_mean: float          # Geometric mean return at optimal f
    max_dd_at_optimal: float       # Expected max drawdown at optimal f
    safe_f: float                  # DD-constrained safe f (≤ max_dd_target)
    trade_count: int               # Number of trades in sample


@dataclass
class LeverageSpaceResult:
    """Multi-asset leverage space portfolio result."""
    optimal_fs: Dict[str, float]   # Per-symbol optimal f
    safe_fs: Dict[str, float]      # Per-symbol DD-constrained f
    portfolio_twr: float           # Portfolio-level TWR
    portfolio_geometric_mean: float
    portfolio_max_dd: float
    portfolio_sharpe: float
    monte_carlo_profit_prob: float  # P(profit) at horizon


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    n_simulations: int
    profit_probability: float      # P(final equity > initial)
    median_return: float           # Median terminal return %
    percentile_5: float            # 5th percentile return %
    percentile_95: float           # 95th percentile return %
    median_max_dd: float           # Median max drawdown %
    expected_cagr: float           # Expected CAGR from MC simulations


# ═══════════════════════════════════════════════════════════════
# Optimal f (Vince Ch. 1-3)
# ═══════════════════════════════════════════════════════════════

def compute_optimal_f(
    trade_returns: np.ndarray,
    max_dd_target: float = 0.15,
) -> OptimalFResult:
    """
    Compute the growth-optimal fraction (optimal f) for a trade series.

    Vince Ch. 1: "Geometric mean maximization is the single criterion
    for maximizing long-term growth."

    The optimal f maximizes the Terminal Wealth Relative (TWR):
        TWR = Π(1 + f × return_i / worst_loss)

    The geometric mean is G = TWR^(1/N) - 1.

    Args:
        trade_returns: array of individual trade returns (e.g., [0.02, -0.01, 0.03])
        max_dd_target: maximum acceptable drawdown (default 15%)

    Returns:
        OptimalFResult with optimal f, safe f, TWR, geometric mean
    """
    if len(trade_returns) < 5:
        return OptimalFResult(
            symbol="", optimal_f=0.01, half_f=0.005,
            twr=1.0, geometric_mean=0.0, max_dd_at_optimal=0.0,
            safe_f=0.01, trade_count=len(trade_returns),
        )

    worst_loss = abs(float(np.min(trade_returns)))
    if worst_loss == 0:
        worst_loss = 0.01  # Prevent division by zero

    # Search for optimal f via grid search (Vince's approach)
    best_f = 0.01
    best_twr = 0.0
    best_gmean = 0.0

    for f_pct in range(1, 100):  # 1% to 99%
        f = f_pct / 100.0
        twr = _compute_twr(trade_returns, f, worst_loss)

        if twr > best_twr:
            best_twr = twr
            best_f = f
            n = len(trade_returns)
            best_gmean = twr ** (1.0 / n) - 1.0 if n > 0 else 0.0

    # Compute max drawdown at optimal f
    dd_at_optimal = _compute_max_dd_at_f(trade_returns, best_f, worst_loss)

    # Find safe f (DD-constrained)
    safe_f = _find_safe_f(trade_returns, worst_loss, max_dd_target)

    return OptimalFResult(
        symbol="",
        optimal_f=best_f,
        half_f=best_f / 2.0,
        twr=best_twr,
        geometric_mean=best_gmean,
        max_dd_at_optimal=dd_at_optimal,
        safe_f=safe_f,
        trade_count=len(trade_returns),
    )


def _compute_twr(
    returns: np.ndarray, f: float, worst_loss: float
) -> float:
    """
    Terminal Wealth Relative at a given f.

    Vince Ch. 2: TWR = Π(1 + f × R_i / |worst_loss|)
    """
    twr = 1.0
    for r in returns:
        hpr = 1.0 + f * r / worst_loss
        if hpr <= 0:
            return 0.0  # Ruin
        twr *= hpr
    return twr


def _compute_max_dd_at_f(
    returns: np.ndarray, f: float, worst_loss: float
) -> float:
    """Compute max drawdown when trading at fraction f."""
    equity = 1.0
    peak = 1.0
    max_dd = 0.0

    for r in returns:
        hpr = 1.0 + f * r / worst_loss
        if hpr <= 0:
            return 1.0  # Total ruin
        equity *= hpr
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def _find_safe_f(
    returns: np.ndarray, worst_loss: float, max_dd_target: float
) -> float:
    """
    Find the largest f whose max drawdown ≤ max_dd_target.

    Vince Ch. 5: "Risk metrics in leverage space must include
    drawdown constraints."
    """
    safe_f = 0.01
    for f_pct in range(1, 100):
        f = f_pct / 100.0
        dd = _compute_max_dd_at_f(returns, f, worst_loss)
        if dd <= max_dd_target:
            safe_f = f
        else:
            break
    return safe_f


# ═══════════════════════════════════════════════════════════════
# Leverage Space Portfolio (Vince Ch. 4)
# ═══════════════════════════════════════════════════════════════

def compute_leverage_space_portfolio(
    trade_returns_dict: Dict[str, np.ndarray],
    max_dd_target: float = 0.15,
    max_total_f: float = 1.0,
) -> LeverageSpaceResult:
    """
    Multi-asset optimal f with drawdown constraint.

    Vince Ch. 4: "Multiple, simultaneous f values define the
    leverage space. The optimal portfolio maximizes TWR across
    the entire leverage space."

    Args:
        trade_returns_dict: {symbol: trade_returns_array}
        max_dd_target: max acceptable portfolio drawdown
        max_total_f: sum of all f values cannot exceed this

    Returns:
        LeverageSpaceResult with per-symbol f and portfolio metrics
    """
    # Compute individual optimal f for each symbol
    individual_results = {}
    for symbol, returns in trade_returns_dict.items():
        result = compute_optimal_f(returns, max_dd_target)
        result.symbol = symbol
        individual_results[symbol] = result

    n_symbols = len(individual_results)
    if n_symbols == 0:
        return LeverageSpaceResult(
            optimal_fs={}, safe_fs={},
            portfolio_twr=1.0, portfolio_geometric_mean=0.0,
            portfolio_max_dd=0.0, portfolio_sharpe=0.0,
            monte_carlo_profit_prob=0.5,
        )

    # Scale individual safe_f values to respect max_total_f
    total_safe_f = sum(r.safe_f for r in individual_results.values())
    scale = min(1.0, max_total_f / max(total_safe_f, 1e-10))

    optimal_fs = {}
    safe_fs = {}
    for symbol, result in individual_results.items():
        optimal_fs[symbol] = result.optimal_f
        safe_fs[symbol] = result.safe_f * scale

    # Simulate portfolio equity curve using safe_f allocations
    max_len = max(len(r) for r in trade_returns_dict.values())
    equity_curve = np.ones(max_len)

    for t in range(max_len):
        period_return = 0.0
        for symbol, returns in trade_returns_dict.items():
            if t < len(returns):
                f = safe_fs[symbol]
                worst = abs(float(np.min(returns)))
                if worst > 0:
                    period_return += f * returns[t] / worst

        equity_curve[t] = equity_curve[t - 1] * (1.0 + period_return) if t > 0 else 1.0 + period_return

    # Portfolio metrics
    portfolio_twr = float(equity_curve[-1]) if len(equity_curve) > 0 else 1.0
    n = len(equity_curve)
    portfolio_gmean = portfolio_twr ** (1.0 / max(1, n)) - 1.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.maximum(peak, 1e-10)
    portfolio_max_dd = float(np.max(dd))

    daily_returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10) if n > 1 else np.array([0.0])
    std = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1.0
    sharpe = float(np.mean(daily_returns) / max(std, 1e-10) * math.sqrt(252))

    # Monte Carlo
    mc = monte_carlo_simulation(daily_returns, n_simulations=2000, horizon_days=252)

    return LeverageSpaceResult(
        optimal_fs=optimal_fs,
        safe_fs=safe_fs,
        portfolio_twr=portfolio_twr,
        portfolio_geometric_mean=portfolio_gmean,
        portfolio_max_dd=portfolio_max_dd,
        portfolio_sharpe=sharpe,
        monte_carlo_profit_prob=mc.profit_probability,
    )


# ═══════════════════════════════════════════════════════════════
# Monte Carlo Simulation (Vince Ch. 7)
# ═══════════════════════════════════════════════════════════════

def monte_carlo_simulation(
    daily_returns: np.ndarray,
    n_simulations: int = 5000,
    horizon_days: int = 252,
) -> MonteCarloResult:
    """
    Monte Carlo simulation of future equity paths.

    Vince Ch. 7: "Maximizing the probability of profit requires
    Monte Carlo evaluation of the leverage space."

    Resamples historical daily returns with replacement to simulate
    N possible equity paths over the given horizon.

    Returns:
        MonteCarloResult with profit probability, median return,
        confidence intervals, and expected CAGR.
    """
    if len(daily_returns) < 10:
        return MonteCarloResult(
            n_simulations=0, profit_probability=0.5,
            median_return=0.0, percentile_5=0.0, percentile_95=0.0,
            median_max_dd=0.0, expected_cagr=0.0,
        )

    rng = np.random.default_rng(42)
    terminal_returns = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)

    for sim in range(n_simulations):
        # Resample returns with replacement
        sampled = rng.choice(daily_returns, size=horizon_days, replace=True)
        equity = np.cumprod(1.0 + sampled)

        terminal_returns[sim] = float(equity[-1] - 1.0) * 100.0  # %

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-10)
        max_drawdowns[sim] = float(np.max(dd)) * 100.0

    profit_prob = float(np.mean(terminal_returns > 0))

    # Expected CAGR from median terminal return
    median_terminal = float(np.percentile(terminal_returns, 50))
    years = horizon_days / 252.0
    if median_terminal > -100:
        expected_cagr = ((1.0 + median_terminal / 100.0) ** (1.0 / years) - 1.0) * 100.0
    else:
        expected_cagr = -100.0

    return MonteCarloResult(
        n_simulations=n_simulations,
        profit_probability=profit_prob,
        median_return=float(np.percentile(terminal_returns, 50)),
        percentile_5=float(np.percentile(terminal_returns, 5)),
        percentile_95=float(np.percentile(terminal_returns, 95)),
        median_max_dd=float(np.percentile(max_drawdowns, 50)),
        expected_cagr=expected_cagr,
    )


# ═══════════════════════════════════════════════════════════════
# Dynamic Leverage Adjustment
# ═══════════════════════════════════════════════════════════════

def compute_dynamic_leverage(
    recent_returns: np.ndarray,
    base_f: float,
    max_dd_target: float = 0.15,
    lookback: int = 50,
) -> float:
    """
    Dynamically adjust leverage fraction based on recent performance.

    Vince: The optimal f changes as the distribution of returns
    shifts. Recalibrate using the most recent trade returns.

    Args:
        recent_returns: most recent N trade returns
        base_f: the base (static) optimal f
        max_dd_target: max acceptable DD
        lookback: number of recent trades to consider

    Returns:
        Adjusted f value for current conditions
    """
    if len(recent_returns) < lookback:
        return base_f * 0.5  # Conservative when insufficient data

    window = recent_returns[-lookback:]
    result = compute_optimal_f(window, max_dd_target)

    # Blend: 60% recent optimal, 40% base to avoid whiplash
    blended_f = 0.6 * result.safe_f + 0.4 * base_f

    # Floor: never more than 2× base, never less than 10% of base
    return float(max(base_f * 0.10, min(base_f * 2.0, blended_f)))


# ═══════════════════════════════════════════════════════════════
# Integration helper: convert Vince f to Carver position scale
# ═══════════════════════════════════════════════════════════════

def vince_f_to_position_scale(
    optimal_f_result: OptimalFResult,
    use_safe: bool = True,
) -> float:
    """
    Convert Vince optimal f / safe f to a multiplicative position
    scale factor for the Carver pipeline.

    The Carver pipeline uses: portfolio_pos = subsys_pos × weight × IDM
    We multiply by vince_scale to adjust leverage.

    Scale = safe_f / 0.10 (baseline 10% risk fraction)
    Capped at [0.1, 2.0] to prevent extreme positions.
    """
    f = optimal_f_result.safe_f if use_safe else optimal_f_result.half_f
    baseline = 0.10  # 10% as baseline risk fraction

    scale = f / baseline
    return float(max(0.1, min(2.0, scale)))


def compute_vince_regime_multipliers(
    trade_returns_dict: Dict[str, np.ndarray],
    max_dd_target: float = 0.15,
) -> Dict[str, float]:
    """
    Compute per-symbol Vince leverage multipliers from trade history.

    Replaces the heuristic regime multipliers (bull=1.0, bear=0.38)
    with data-driven optimal f values.

    Returns:
        {symbol: position_scale_multiplier}
    """
    multipliers = {}
    for symbol, returns in trade_returns_dict.items():
        result = compute_optimal_f(returns, max_dd_target)
        result.symbol = symbol
        multipliers[symbol] = vince_f_to_position_scale(result, use_safe=True)

        logger.info(
            "Vince %s: optimal_f=%.3f safe_f=%.3f half_f=%.3f DD@opt=%.1f%% → scale=%.2f",
            symbol, result.optimal_f, result.safe_f, result.half_f,
            result.max_dd_at_optimal * 100, multipliers[symbol],
        )

    return multipliers
