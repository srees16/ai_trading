"""
Chapter 6: Estimating Future Performance II — Trade Analysis
=============================================================
Testing and Tuning Market Trading Systems — Timothy Masters

Algorithms:
  - Profit per bar vs per trade vs per time analysis (pp.200–213)
  - Parametric and nonparametric confidence intervals (pp.214–227)
  - BCa bootstrap confidence intervals (pp.228–236, BOOT_CONF.CPP)
  - Lower bound for mean future returns (pp.237–253)
  - Bounding future returns via empirical quantiles (pp.247–266)
  - Bounding drawdown (pp.267–286, DRAWDOWN program)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_data import (
    get_close_series,
    generate_returns,
    generate_random_trading_system,
)


# ======================================================================
# Profit Per Bar / Per Trade / Per Time  (pp.200–213)
# ======================================================================

def profit_analysis(trade_returns, trade_durations=None, n_bars=None):
    """Analyse profit on a per-bar, per-trade, and per-time basis.

    Parameters
    ----------
    trade_returns   : array-like – return for each completed trade
    trade_durations : array-like or None – duration (bars) of each trade
    n_bars          : int or None – total bars in evaluation period

    Returns
    -------
    dict with per_trade, per_bar, per_time metrics
    """
    trade_returns = np.asarray(trade_returns, dtype=float)
    n_trades = len(trade_returns)

    if trade_durations is not None:
        trade_durations = np.asarray(trade_durations, dtype=float)
    else:
        trade_durations = np.ones(n_trades) * 5  # default 5 bars per trade

    total_duration = trade_durations.sum()
    if n_bars is None:
        n_bars = int(total_duration * 1.2)  # assume 80% time-in-market

    total_return = trade_returns.sum()
    mean_per_trade = trade_returns.mean()
    mean_per_bar = total_return / max(n_bars, 1)
    mean_per_time = total_return / max(total_duration, 1)

    return {
        "n_trades": n_trades,
        "total_return": total_return,
        "mean_per_trade": mean_per_trade,
        "mean_per_bar": mean_per_bar,
        "mean_per_time_in_market": mean_per_time,
        "std_per_trade": trade_returns.std(),
        "pct_time_in_market": total_duration / max(n_bars, 1),
        "sr_per_trade": mean_per_trade / (trade_returns.std() + 1e-10),
    }


# ======================================================================
# Parametric Confidence Intervals  (pp.221–227)
# ======================================================================

def parametric_confidence(returns, confidence=0.95):
    """Compute parametric confidence interval for the mean return.

    Uses the t-distribution (Eq 6-1 through 6-5).

    Parameters
    ----------
    returns    : array-like – return series
    confidence : float – confidence level (e.g. 0.95)

    Returns
    -------
    dict with mean, std, lower, upper, t_stat, p_value
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    mean = returns.mean()
    std = returns.std(ddof=1)
    se = std / np.sqrt(n)

    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)

    lower = mean - t_crit * se
    upper = mean + t_crit * se

    # One-sided test: is mean > 0?
    t_stat = mean / se if se > 0 else 0.0
    p_value = 1 - t_dist.cdf(t_stat, df=n - 1)

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "lower": lower,
        "upper": upper,
        "t_stat": t_stat,
        "p_value": p_value,
        "n": n,
        "confidence": confidence,
    }


# ======================================================================
# BCa Bootstrap Confidence Intervals  (pp.228–236, BOOT_CONF.CPP)
# ======================================================================

def bca_bootstrap(data, stat_fn=None, n_boot=2000, confidence=0.95, seed=42):
    """BCa (bias-corrected and accelerated) bootstrap confidence interval.

    Converted from BOOT_CONF.CPP (pp.232–236).

    Parameters
    ----------
    data       : array-like – observed data
    stat_fn    : callable or None – statistic function f(data) → scalar
                 Default: np.mean
    n_boot     : int
    confidence : float
    seed       : int

    Returns
    -------
    dict with 'observed', 'lower', 'upper', 'bias_correction',
              'acceleration', 'boot_distribution'
    """
    if stat_fn is None:
        stat_fn = np.mean

    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    observed = stat_fn(data)

    # Bootstrap replications
    boot_stats = np.zeros(n_boot)
    for b in range(n_boot):
        sample = data[rng.integers(0, n, n)]
        boot_stats[b] = stat_fn(sample)

    # Bias correction (z0)
    prop_less = np.mean(boot_stats < observed)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop_less)

    # Acceleration (a) via jackknife
    jackknife_stats = np.zeros(n)
    for i in range(n):
        jk_data = np.delete(data, i)
        jackknife_stats[i] = stat_fn(jk_data)

    jk_mean = jackknife_stats.mean()
    diff = jk_mean - jackknife_stats
    a_num = np.sum(diff ** 3)
    a_den = 6.0 * (np.sum(diff ** 2) ** 1.5)
    acceleration = a_num / a_den if a_den > 0 else 0.0

    # Adjusted percentiles
    alpha = 1 - confidence
    z_alpha_lo = norm.ppf(alpha / 2)
    z_alpha_hi = norm.ppf(1 - alpha / 2)

    def _adj_percentile(z_alpha):
        num = z0 + z_alpha
        denom = 1 - acceleration * num
        if abs(denom) < 1e-10:
            denom = 1e-10
        adj_z = z0 + num / denom
        return norm.cdf(adj_z) * 100

    lo_pct = _adj_percentile(z_alpha_lo)
    hi_pct = _adj_percentile(z_alpha_hi)

    lo_pct = np.clip(lo_pct, 0.01, 99.99)
    hi_pct = np.clip(hi_pct, 0.01, 99.99)

    lower = np.percentile(boot_stats, lo_pct)
    upper = np.percentile(boot_stats, hi_pct)

    return {
        "observed": observed,
        "lower": lower,
        "upper": upper,
        "bias_correction": z0,
        "acceleration": acceleration,
        "boot_mean": boot_stats.mean(),
        "boot_std": boot_stats.std(),
        "confidence": confidence,
        "boot_distribution": boot_stats,
    }


# ======================================================================
# Lower Bound for Mean Future Returns  (pp.214–244)
# ======================================================================

def lower_bound_mean_return(returns, confidence=0.95, method="parametric"):
    """Compute a lower confidence bound for the mean future return.

    Parameters
    ----------
    returns    : array-like
    confidence : float
    method     : str – 'parametric' or 'bootstrap'

    Returns
    -------
    dict with 'lower_bound', 'method', 'confidence', etc.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    mean = returns.mean()

    if method == "parametric":
        std = returns.std(ddof=1)
        se = std / np.sqrt(n)
        t_crit = t_dist.ppf(confidence, df=n - 1)
        lower = mean - t_crit * se
        return {
            "lower_bound": lower,
            "mean": mean,
            "se": se,
            "method": "parametric",
            "confidence": confidence,
        }
    else:
        bca = bca_bootstrap(returns, np.mean, n_boot=2000,
                            confidence=confidence)
        return {
            "lower_bound": bca["lower"],
            "mean": bca["observed"],
            "method": "BCa bootstrap",
            "confidence": confidence,
            "bias_correction": bca["bias_correction"],
        }


# ======================================================================
# Bounding Future Returns via Empirical Quantiles  (pp.247–266)
# ======================================================================

def bound_future_returns(returns, quantile=0.05, confidence=0.95, n_boot=2000,
                         seed=42):
    """Estimate a lower bound on future returns at a given quantile.

    Uses bootstrap to get a confidence interval for the specified
    quantile of the return distribution.

    Parameters
    ----------
    returns    : array-like – return series
    quantile   : float – quantile to bound (e.g. 0.05 for 5th percentile)
    confidence : float – confidence level
    n_boot     : int
    seed       : int

    Returns
    -------
    dict with 'quantile_estimate', 'lower_ci', 'upper_ci'
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    observed_quantile = np.percentile(returns, quantile * 100)

    boot_quantiles = np.zeros(n_boot)
    for b in range(n_boot):
        sample = returns[rng.integers(0, n, n)]
        boot_quantiles[b] = np.percentile(sample, quantile * 100)

    alpha = 1 - confidence
    lower_ci = np.percentile(boot_quantiles, alpha / 2 * 100)
    upper_ci = np.percentile(boot_quantiles, (1 - alpha / 2) * 100)

    return {
        "quantile": quantile,
        "quantile_estimate": observed_quantile,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "confidence": confidence,
        "boot_mean": boot_quantiles.mean(),
        "boot_std": boot_quantiles.std(),
    }


# ======================================================================
# Bounding Drawdown  (pp.267–286, DRAWDOWN program)
# ======================================================================

def compute_drawdown(returns):
    """Compute drawdown series and max drawdown from returns.

    Parameters
    ----------
    returns : array-like – return series

    Returns
    -------
    dict with 'drawdown_series', 'max_drawdown', 'max_dd_duration'
    """
    returns = np.asarray(returns, dtype=float)
    equity = np.cumsum(returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max

    # Duration of max drawdown
    max_dd = drawdown.min()
    dd_end = np.argmin(drawdown)
    dd_start = np.argmax(equity[:dd_end + 1]) if dd_end > 0 else 0

    return {
        "drawdown_series": drawdown,
        "equity_curve": equity,
        "max_drawdown": max_dd,
        "max_dd_start": int(dd_start),
        "max_dd_end": int(dd_end),
        "max_dd_duration": int(dd_end - dd_start),
    }


def bootstrap_drawdown_bound(returns, confidence=0.95, n_boot=2000, seed=42):
    """Bootstrap upper bound on maximum drawdown.

    Converted from the DRAWDOWN program (pp.272–282).

    Bootstraps the return series (block bootstrap to preserve some
    temporal structure) and computes max drawdown for each replicate.

    Parameters
    ----------
    returns    : array-like
    confidence : float
    n_boot     : int
    seed       : int

    Returns
    -------
    dict with 'observed_max_dd', 'bound', 'boot_distribution'
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    block_size = max(10, int(np.sqrt(n)))

    observed_dd = compute_drawdown(returns)["max_drawdown"]

    boot_max_dd = np.zeros(n_boot)
    for b in range(n_boot):
        # Block bootstrap
        boot_ret = np.zeros(n)
        pos = 0
        while pos < n:
            start = rng.integers(0, max(n - block_size, 1))
            block_len = min(block_size, n - pos)
            boot_ret[pos: pos + block_len] = returns[start: start + block_len]
            pos += block_len
        boot_max_dd[b] = compute_drawdown(boot_ret)["max_drawdown"]

    bound = np.percentile(boot_max_dd, (1 - confidence) * 100)

    return {
        "observed_max_dd": observed_dd,
        "bound": bound,
        "confidence": confidence,
        "boot_mean_dd": boot_max_dd.mean(),
        "boot_std_dd": boot_max_dd.std(),
        "boot_distribution": boot_max_dd,
        "interpretation": (
            f"With {confidence:.0%} confidence, max drawdown will not exceed "
            f"{abs(bound):.4f}"
        ),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("Chapter 6 – Estimating Future Performance II: Trade Analysis")
    print("  (Testing and Tuning Market Trading Systems)")
    print("=" * 70)

    prices = get_close_series("SPY")
    returns = prices.pct_change().dropna().values

    # --- 1. Profit Analysis ---
    print("\n--- Profit per Bar / Trade / Time ---")
    trades = generate_random_trading_system(300, win_rate=0.55,
                                            avg_win=0.8, avg_loss=-0.6, seed=42)
    durations = np.random.default_rng(42).integers(2, 20, 300).astype(float)
    profit = profit_analysis(trades.values, durations, n_bars=2000)
    for k, v in profit.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- 2. Parametric Confidence Interval ---
    print("\n--- Parametric Confidence Interval (95%) ---")
    ci = parametric_confidence(returns, confidence=0.95)
    print(f"  Mean:   {ci['mean']:.6f}")
    print(f"  SE:     {ci['se']:.6f}")
    print(f"  95% CI: [{ci['lower']:.6f}, {ci['upper']:.6f}]")
    print(f"  t-stat: {ci['t_stat']:.4f}")
    print(f"  p-val (>0): {ci['p_value']:.4f}")

    # --- 3. BCa Bootstrap ---
    print("\n--- BCa Bootstrap Confidence Interval ---")
    bca = bca_bootstrap(returns, np.mean, n_boot=2000, confidence=0.95)
    print(f"  Observed mean:     {bca['observed']:.6f}")
    print(f"  BCa 95% CI:        [{bca['lower']:.6f}, {bca['upper']:.6f}]")
    print(f"  Bias correction:   {bca['bias_correction']:.4f}")
    print(f"  Acceleration:      {bca['acceleration']:.6f}")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(bca["boot_distribution"], bins=50, alpha=0.7, color="steelblue",
             edgecolor="white")
    ax1.axvline(bca["observed"], color="tomato", linewidth=2,
                label=f"Observed ({bca['observed']:.5f})")
    ax1.axvline(bca["lower"], color="orange", linestyle="--",
                label=f"Lower ({bca['lower']:.5f})")
    ax1.axvline(bca["upper"], color="orange", linestyle="--",
                label=f"Upper ({bca['upper']:.5f})")
    ax1.set_title("BCa Bootstrap Distribution of Mean Return")
    ax1.legend(fontsize=8)
    fig1.tight_layout()

    # --- 4. Lower Bound ---
    print("\n--- Lower Bound for Mean Return ---")
    for method in ["parametric", "bootstrap"]:
        lb = lower_bound_mean_return(returns, confidence=0.95, method=method)
        print(f"  {lb['method']:20s}: lower={lb['lower_bound']:.6f}, mean={lb['mean']:.6f}")

    # --- 5. Bounding Future Returns ---
    print("\n--- Bounding Future Returns (5th percentile) ---")
    br = bound_future_returns(returns, quantile=0.05, confidence=0.95)
    print(f"  5th percentile estimate:  {br['quantile_estimate']:.6f}")
    print(f"  95% CI for 5th pct:       [{br['lower_ci']:.6f}, {br['upper_ci']:.6f}]")

    # --- 6. Drawdown Bounds ---
    print("\n--- Drawdown Analysis & Bounds ---")
    dd = compute_drawdown(returns)
    print(f"  Max drawdown:    {dd['max_drawdown']:.6f}")
    print(f"  DD duration:     {dd['max_dd_duration']} bars")

    dd_bound = bootstrap_drawdown_bound(returns, confidence=0.95, n_boot=1000)
    print(f"  {dd_bound['interpretation']}")

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax2a.plot(dd["equity_curve"], color="steelblue", linewidth=0.8)
    ax2a.set_title("Equity Curve")
    ax2a.set_ylabel("Cumulative Return")
    ax2b.fill_between(range(len(dd["drawdown_series"])),
                      dd["drawdown_series"], 0, color="tomato", alpha=0.5)
    ax2b.axhline(dd_bound["bound"], color="darkred", linestyle="--",
                 label=f"95% Bound ({dd_bound['bound']:.4f})")
    ax2b.set_title("Drawdown")
    ax2b.set_ylabel("Drawdown")
    ax2b.set_xlabel("Bar")
    ax2b.legend()
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(dd_bound["boot_distribution"], bins=40, alpha=0.7,
             color="tomato", edgecolor="white")
    ax3.axvline(dd["max_drawdown"], color="darkred", linewidth=2,
                label=f"Observed ({dd['max_drawdown']:.4f})")
    ax3.axvline(dd_bound["bound"], color="orange", linestyle="--",
                label=f"95% Bound ({dd_bound['bound']:.4f})")
    ax3.set_title("Bootstrap Distribution of Max Drawdown")
    ax3.legend(fontsize=8)
    fig3.tight_layout()

    print("\nChapter 6 complete.")


if __name__ == "__main__":
    main()
