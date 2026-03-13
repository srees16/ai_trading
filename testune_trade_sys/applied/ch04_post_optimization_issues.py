"""
Chapter 4: Post-optimization Issues
=====================================
Testing and Tuning Market Trading Systems — Timothy Masters

Algorithms:
  - StocBias: cheap stochastic bias estimation (pp.98–102)
  - Cheap parameter relationships (pp.102–113)
  - Parameter sensitivity curves (pp.114–126)
  - OEX trading system integration
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_data import (
    get_close_series,
    generate_returns,
    generate_random_trading_system,
)


# ======================================================================
# StocBias — Cheap Stochastic Bias Estimates  (pp.98–102)
# ======================================================================

class StocBias:
    """Estimate optimisation bias without a full walkforward.

    The idea (Timothy Masters, pp.98–101): after optimising a trading
    system over the full sample, we estimate how much of the apparent
    performance is due to overfitting by resampling the data (bootstrap
    or permutation) and re-optimising on each resample.

    The bias is:
        bias  ≈ mean(resampled_optima) - population_expectation

    Since the population expectation of a random strategy is ≈ 0, the
    bias is simply the mean of the re-optimised performance on resamples.

    Parameters
    ----------
    n_resamples : int  – number of bootstrap / permutation trials
    seed        : int  – random seed
    """

    def __init__(self, n_resamples=200, seed=42):
        self.n_resamples = n_resamples
        self.seed = seed

    def estimate_bias(self, returns, strategy_fn, metric_fn=None):
        """Estimate the optimisation bias of a trading strategy.

        Parameters
        ----------
        returns     : np.ndarray  – daily returns (n,)
        strategy_fn : callable    – f(returns) → optimised metric value
        metric_fn   : callable or None – if None, strategy_fn returns the metric

        Returns
        -------
        dict with 'observed', 'bias_estimate', 'debiased', 'resample_stats'
        """
        rng = np.random.default_rng(self.seed)
        returns = np.asarray(returns, dtype=float)
        n = len(returns)

        # Observed (in-sample) optimised metric
        observed = strategy_fn(returns)

        # Bootstrap resampling
        resample_metrics = np.zeros(self.n_resamples)
        for i in range(self.n_resamples):
            # Permute returns to destroy temporal structure
            perm_returns = rng.permutation(returns)
            resample_metrics[i] = strategy_fn(perm_returns)

        bias_estimate = resample_metrics.mean()
        debiased = observed - bias_estimate

        return {
            "observed": observed,
            "bias_estimate": bias_estimate,
            "debiased": debiased,
            "bias_std": resample_metrics.std(),
            "resample_mean": resample_metrics.mean(),
            "resample_std": resample_metrics.std(),
            "resample_quantiles": {
                "5%": np.percentile(resample_metrics, 5),
                "25%": np.percentile(resample_metrics, 25),
                "50%": np.percentile(resample_metrics, 50),
                "75%": np.percentile(resample_metrics, 75),
                "95%": np.percentile(resample_metrics, 95),
            },
        }


# ======================================================================
# Cheap Parameter Relationships  (pp.102–113)
# ======================================================================

def parameter_relationships(returns, param_ranges, strategy_fn):
    """Evaluate a trading system across a grid of parameter values.

    For every pair of parameters, compute the strategy metric on a 2-D
    grid.  This reveals interactions and sensitivities.

    Parameters
    ----------
    returns      : np.ndarray – daily returns
    param_ranges : dict       – {param_name: [values]}
    strategy_fn  : callable   – f(returns, **params) → metric

    Returns
    -------
    dict with:
      'grid'   : dict of (param_i, param_j) → 2-D metric array
      'best'   : dict of best parameters
      'scores' : flat list of (params, metric) tuples
    """
    from itertools import product as iproduct

    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    all_combos = list(iproduct(*param_values))
    scores = []

    for combo in all_combos:
        params = dict(zip(param_names, combo))
        metric = strategy_fn(returns, **params)
        scores.append((params, metric))

    # Find best
    best_params, best_metric = max(scores, key=lambda x: x[1])

    # Build 2-D grid for each pair of parameters
    grids = {}
    if len(param_names) >= 2:
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                pi, pj = param_names[i], param_names[j]
                vi, vj = param_values[i], param_values[j]
                grid = np.full((len(vi), len(vj)), np.nan)
                for score_params, metric_val in scores:
                    ri = vi.index(score_params[pi]) if score_params[pi] in vi else -1
                    rj = vj.index(score_params[pj]) if score_params[pj] in vj else -1
                    if ri >= 0 and rj >= 0:
                        # Average over other params if needed
                        if np.isnan(grid[ri, rj]):
                            grid[ri, rj] = metric_val
                        else:
                            grid[ri, rj] = (grid[ri, rj] + metric_val) / 2
                grids[(pi, pj)] = {
                    "grid": grid,
                    "row_values": vi,
                    "col_values": vj,
                }

    return {
        "grids": grids,
        "best_params": best_params,
        "best_metric": best_metric,
        "scores": scores,
    }


# ======================================================================
# Parameter Sensitivity Curves  (pp.114–126)
# ======================================================================

def parameter_sensitivity(returns, param_name, param_values, strategy_fn,
                          fixed_params=None):
    """Compute a 1-D sensitivity curve for a single parameter.

    Parameters
    ----------
    returns       : np.ndarray
    param_name    : str – the parameter to vary
    param_values  : list – values to test
    strategy_fn   : callable – f(returns, **params) → metric
    fixed_params  : dict or None – fixed values for other params

    Returns
    -------
    pd.DataFrame with columns [param_name, 'metric']
    """
    fixed = fixed_params or {}
    results = []
    for val in param_values:
        params = {**fixed, param_name: val}
        metric = strategy_fn(returns, **params)
        results.append({param_name: val, "metric": metric})
    return pd.DataFrame(results)


# ======================================================================
# Simple SMA-Crossover Strategy for Demonstration
# ======================================================================

def _sma_strategy_metric(returns, short_window=10, long_window=50):
    """Annualised Sharpe ratio of an SMA-crossover strategy.

    This is a simple strategy function compatible with the parameter
    analysis utilities above.
    """
    prices = (1 + pd.Series(returns)).cumprod() * 100
    short_ma = prices.rolling(int(short_window)).mean()
    long_ma = prices.rolling(int(long_window)).mean()
    signal = np.where(short_ma > long_ma, 1, -1)
    strat_ret = pd.Series(returns) * pd.Series(signal).shift(1)
    strat_ret = strat_ret.dropna()
    if strat_ret.std() == 0:
        return 0.0
    return float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))


def _best_sma_sharpe(returns, short_range=None, long_range=None):
    """Optimise SMA crossover over a grid and return best Sharpe.

    Used as the strategy_fn for StocBias.
    """
    short_range = short_range or [5, 10, 20, 30]
    long_range = long_range or [40, 60, 80, 100]
    best = -np.inf
    for s in short_range:
        for l in long_range:
            if s >= l:
                continue
            sr = _sma_strategy_metric(returns, short_window=s, long_window=l)
            if sr > best:
                best = sr
    return best


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("Chapter 4 – Post-optimization Issues")
    print("  (Testing and Tuning Market Trading Systems)")
    print("=" * 70)

    # --- 1. StocBias ---
    print("\n--- StocBias: Cheap Bias Estimation ---")
    prices = get_close_series("SPY")
    returns = prices.pct_change().dropna().values

    bias_estimator = StocBias(n_resamples=100, seed=42)
    result = bias_estimator.estimate_bias(returns, _best_sma_sharpe)

    print(f"  Observed (in-sample) best Sharpe:  {result['observed']:.4f}")
    print(f"  Estimated bias:                    {result['bias_estimate']:.4f}")
    print(f"  Debiased Sharpe:                   {result['debiased']:.4f}")
    print(f"  Resample Sharpe std:               {result['resample_std']:.4f}")
    for q, v in result["resample_quantiles"].items():
        print(f"    Resample {q}: {v:.4f}")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.axvline(result["observed"], color="tomato", linewidth=2,
                label=f"Observed ({result['observed']:.3f})")
    ax1.axvline(result["debiased"], color="seagreen", linewidth=2,
                linestyle="--", label=f"Debiased ({result['debiased']:.3f})")
    # We show the range of resample metrics
    ax1.set_xlabel("Sharpe Ratio")
    ax1.set_title("StocBias: Optimisation Bias Estimation")
    ax1.legend()
    fig1.tight_layout()

    # --- 2. Parameter Sensitivity ---
    print("\n--- Parameter Sensitivity Curves ---")
    short_values = list(range(5, 55, 5))
    sens_df = parameter_sensitivity(
        returns,
        param_name="short_window",
        param_values=short_values,
        strategy_fn=_sma_strategy_metric,
        fixed_params={"long_window": 50},
    )
    print(f"  Best short_window: {sens_df.loc[sens_df['metric'].idxmax(), 'short_window']}")
    print(f"  Best Sharpe:       {sens_df['metric'].max():.4f}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(sens_df["short_window"], sens_df["metric"],
             marker="o", color="steelblue")
    ax2.set_xlabel("Short Window")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Parameter Sensitivity: Short MA Window (Long=50)")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    fig2.tight_layout()

    # --- 3. Parameter Relationships (2-D grid) ---
    print("\n--- Parameter Relationships (2-D Grid) ---")
    param_ranges = {
        "short_window": [5, 10, 15, 20, 25, 30],
        "long_window": [40, 50, 60, 80, 100],
    }
    grid_result = parameter_relationships(returns, param_ranges,
                                          _sma_strategy_metric)
    print(f"  Best parameters: {grid_result['best_params']}")
    print(f"  Best metric:     {grid_result['best_metric']:.4f}")

    if grid_result["grids"]:
        key = list(grid_result["grids"].keys())[0]
        gdata = grid_result["grids"][key]

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        im = ax3.imshow(gdata["grid"], aspect="auto", cmap="RdYlGn",
                        origin="lower")
        ax3.set_xticks(range(len(gdata["col_values"])))
        ax3.set_xticklabels(gdata["col_values"])
        ax3.set_yticks(range(len(gdata["row_values"])))
        ax3.set_yticklabels(gdata["row_values"])
        ax3.set_xlabel(key[1])
        ax3.set_ylabel(key[0])
        ax3.set_title("Parameter Relationship Heatmap (Sharpe Ratio)")
        plt.colorbar(im, ax=ax3, label="Sharpe Ratio")
        fig3.tight_layout()

    print("\nChapter 4 complete.")


if __name__ == "__main__":
    main()
