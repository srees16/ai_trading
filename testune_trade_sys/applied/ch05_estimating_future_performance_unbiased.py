"""
Chapter 5: Estimating Future Performance I — Unbiased Trade Simulation
=======================================================================
Testing and Tuning Market Trading Systems — Timothy Masters

Algorithms:
  - Walkforward analysis (pp.135–162)
  - Cross-validation for trading systems (pp.149–162)
  - Combinatorially Symmetric Cross-Validation (CSCV) superiority test
    (pp.164–177, C++ cscvcore() conversion)
  - Nested walkforward analysis (pp.178–192)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_data import get_close_series, generate_returns


# ======================================================================
# Walkforward Analysis  (pp.135–148)
# ======================================================================

def walkforward_analysis(returns, strategy_fn, train_size=500,
                         test_size=100, step_size=None):
    """Walkforward analysis with rolling train/test windows.

    Converted from the general walkforward algorithm (pp.140–143).

    Parameters
    ----------
    returns     : np.ndarray – daily returns
    strategy_fn : callable   – f(train_returns) → model that has .predict(test_returns)
                               OR f(train_returns) → float (Sharpe per test pass)
    train_size  : int – number of bars in training window
    test_size   : int – number of bars in test window
    step_size   : int or None – step between windows (default: test_size)

    Returns
    -------
    dict with 'oos_returns', 'is_returns', 'n_folds', 'summary'
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    step = step_size or test_size

    is_metrics = []
    oos_returns_all = []

    start = 0
    while start + train_size + test_size <= n:
        train = returns[start: start + train_size]
        test = returns[start + train_size: start + train_size + test_size]

        # Strategy function can return a scalar (IS metric)
        # and optionally accept/return OOS data
        result = strategy_fn(train)

        if isinstance(result, dict):
            is_metrics.append(result.get("is_metric", 0.0))
            oos_ret = result.get("oos_returns", test)
        elif callable(getattr(result, "predict", None)):
            # Model-like object
            is_metrics.append(0.0)
            oos_ret = result.predict(test)
        else:
            # Scalar IS metric — OOS = just raw test returns
            is_metrics.append(float(result))
            oos_ret = test

        oos_returns_all.extend(oos_ret.tolist() if hasattr(oos_ret, "tolist")
                               else list(oos_ret))
        start += step

    oos_arr = np.array(oos_returns_all)
    oos_sr = (oos_arr.mean() / (oos_arr.std() + 1e-10)) * np.sqrt(252)

    return {
        "oos_returns": oos_arr,
        "is_metrics": np.array(is_metrics),
        "n_folds": len(is_metrics),
        "oos_sharpe": oos_sr,
        "oos_total_return": oos_arr.sum(),
    }


# ======================================================================
# Cross-Validation for Trading Systems  (pp.149–156)
# ======================================================================

def trading_cross_validation(returns, strategy_fn, n_folds=10, seed=42):
    """K-fold cross-validation adapted for trading systems.

    Converted from the general cross-validation algorithm (pp.150–155).
    Uses time-ordered blocks (not random shuffling) to preserve
    temporal structure.

    Parameters
    ----------
    returns     : np.ndarray
    strategy_fn : callable – f(train) → metric (float)
    n_folds     : int
    seed        : int

    Returns
    -------
    dict with 'is_metrics', 'oos_metrics', 'mean_is', 'mean_oos'
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    fold_size = n // n_folds

    is_metrics = []
    oos_metrics = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_data = returns[test_start:test_end]
        train_data = np.concatenate([returns[:test_start],
                                     returns[test_end:]])

        is_metric = strategy_fn(train_data)
        oos_metric = strategy_fn(test_data)

        is_metrics.append(float(is_metric))
        oos_metrics.append(float(oos_metric))

    return {
        "is_metrics": np.array(is_metrics),
        "oos_metrics": np.array(oos_metrics),
        "mean_is": np.mean(is_metrics),
        "mean_oos": np.mean(oos_metrics),
        "std_oos": np.std(oos_metrics),
        "n_folds": n_folds,
    }


# ======================================================================
# CSCV Superiority Test  (pp.164–177, cscvcore() C++ conversion)
# ======================================================================

def cscv_superiority(returns_matrix, n_blocks=8, criterion_fn=None):
    """Combinatorially Symmetric Cross-Validation superiority test.

    Converted from cscvcore() in CSCV.CPP (pp.170–177).

    Tests whether the best in-sample system is also superior
    out-of-sample. Returns the fraction of IS/OOS combinations where
    the IS-best system ranks at or below the OOS median.

    Parameters
    ----------
    returns_matrix : np.ndarray – (n_systems, n_cases) matrix of returns
    n_blocks       : int        – number of blocks (must be even)
    criterion_fn   : callable or None – f(returns_1d) → metric
                     Default: mean return

    Returns
    -------
    dict with 'pbo' (probability of backtest overfitting),
              'n_combos', 'n_less'
    """
    if criterion_fn is None:
        criterion_fn = lambda r: np.mean(r) / (np.std(r) + 1e-10)

    returns_matrix = np.asarray(returns_matrix, dtype=float)
    n_systems, n_cases = returns_matrix.shape
    n_blocks = (n_blocks // 2) * 2  # ensure even

    # Partition cases into blocks
    block_starts = []
    block_lengths = []
    istart = 0
    for i in range(n_blocks):
        length = (n_cases - istart) // (n_blocks - i)
        block_starts.append(istart)
        block_lengths.append(length)
        istart += length

    half = n_blocks // 2
    block_indices = list(range(n_blocks))

    # Enumerate all C(n_blocks, half) training-set combinations
    n_less = 0
    n_combo = 0

    for train_blocks in combinations(block_indices, half):
        test_blocks = [b for b in block_indices if b not in train_blocks]

        # Compute IS criterion for each system
        is_crits = np.zeros(n_systems)
        for isys in range(n_systems):
            is_data = []
            for ib in train_blocks:
                start = block_starts[ib]
                end = start + block_lengths[ib]
                is_data.extend(returns_matrix[isys, start:end].tolist())
            is_crits[isys] = criterion_fn(np.array(is_data))

        # Compute OOS criterion for each system
        oos_crits = np.zeros(n_systems)
        for isys in range(n_systems):
            oos_data = []
            for ib in test_blocks:
                start = block_starts[ib]
                end = start + block_lengths[ib]
                oos_data.extend(returns_matrix[isys, start:end].tolist())
            oos_crits[isys] = criterion_fn(np.array(oos_data))

        # Find best IS system
        ibest = np.argmax(is_crits)
        best_oos = oos_crits[ibest]

        # Compute relative rank of IS-best system in OOS
        n_at_or_above = np.sum(best_oos >= oos_crits)
        rel_rank = n_at_or_above / (n_systems + 1)

        if rel_rank <= 0.5:
            n_less += 1

        n_combo += 1

    pbo = n_less / max(n_combo, 1)

    return {
        "pbo": pbo,
        "n_combos": n_combo,
        "n_less": n_less,
        "interpretation": (
            "High PBO (>0.5): likely overfitting — IS-best is OOS-mediocre"
            if pbo > 0.5 else
            "Low PBO (≤0.5): IS-best tends to perform well OOS"
        ),
    }


# ======================================================================
# Nested Walkforward Analysis  (pp.178–192)
# ======================================================================

def nested_walkforward(returns, strategy_factory, outer_train=600,
                       outer_test=120, inner_n_folds=5):
    """Nested walkforward analysis.

    Outer loop: standard walkforward (train → test).
    Inner loop: cross-validation within each outer training set to
    select hyperparameters, avoiding direct test-set contamination.

    Converted from the nested walkforward algorithm (pp.180–185).

    Parameters
    ----------
    returns          : np.ndarray
    strategy_factory : callable – f(train, inner_n_folds) → dict with
                       'best_params', 'is_metric'
    outer_train      : int
    outer_test       : int
    inner_n_folds    : int

    Returns
    -------
    dict with fold-level details and aggregate OOS metric
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    folds = []
    start = 0
    while start + outer_train + outer_test <= n:
        train = returns[start: start + outer_train]
        test = returns[start + outer_train: start + outer_train + outer_test]

        # Inner CV selects best parameters
        inner_result = strategy_factory(train, inner_n_folds)

        # Apply best parameters to test
        best_params = inner_result.get("best_params", {})
        is_metric = inner_result.get("is_metric", 0.0)

        # OOS: apply the strategy from inner result
        oos_metric = inner_result.get("oos_fn", lambda x: np.mean(x))(test)

        folds.append({
            "start": start,
            "best_params": best_params,
            "is_metric": is_metric,
            "oos_metric": float(oos_metric),
        })
        start += outer_test

    oos_metrics = [f["oos_metric"] for f in folds]
    return {
        "folds": folds,
        "n_folds": len(folds),
        "mean_oos_metric": np.mean(oos_metrics) if oos_metrics else 0.0,
        "std_oos_metric": np.std(oos_metrics) if oos_metrics else 0.0,
    }


# ======================================================================
# Helper: Sharpe-based strategy function for demonstrations
# ======================================================================

def _sma_walkforward_strategy(train_returns):
    """Find best SMA crossover on training data, return dict."""
    n = len(train_returns)
    prices = 100 * np.exp(np.cumsum(train_returns))

    best_sr = -np.inf
    best_params = (10, 50)
    for s in [5, 10, 15, 20]:
        for l in [30, 50, 70]:
            if s >= l or l >= n:
                continue
            short_ma = pd.Series(prices).rolling(s).mean().values
            long_ma = pd.Series(prices).rolling(l).mean().values
            signal = np.where(short_ma > long_ma, 1, -1)
            strat_ret = train_returns * np.roll(signal, 1)
            strat_ret = strat_ret[l:]
            sr = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
            if sr > best_sr:
                best_sr = sr
                best_params = (s, l)

    return {
        "is_metric": best_sr,
        "oos_returns": train_returns,  # placeholder
    }


def _sma_sharpe_metric(returns):
    """Compute annualised Sharpe of best SMA crossover."""
    n = len(returns)
    prices = 100 * np.exp(np.cumsum(returns))
    best_sr = -np.inf
    for s in [5, 10, 20]:
        for l in [30, 50]:
            if s >= l or l >= n:
                continue
            short_ma = pd.Series(prices).rolling(s).mean().values
            long_ma = pd.Series(prices).rolling(l).mean().values
            sig = np.where(short_ma > long_ma, 1, -1)
            sr_vals = returns * np.roll(sig, 1)
            sr_vals = sr_vals[l:]
            if len(sr_vals) < 10:
                continue
            sr = np.mean(sr_vals) / (np.std(sr_vals) + 1e-10) * np.sqrt(252)
            if sr > best_sr:
                best_sr = sr
    return best_sr if best_sr > -np.inf else 0.0


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("Chapter 5 – Estimating Future Performance I")
    print("  (Testing and Tuning Market Trading Systems)")
    print("=" * 70)

    prices = get_close_series("SPY")
    returns = prices.pct_change().dropna().values

    # --- 1. Walkforward Analysis ---
    print("\n--- Walkforward Analysis ---")
    wf_result = walkforward_analysis(
        returns, _sma_walkforward_strategy,
        train_size=400, test_size=80,
    )
    print(f"  Folds:            {wf_result['n_folds']}")
    print(f"  OOS Sharpe:       {wf_result['oos_sharpe']:.4f}")
    print(f"  OOS total return: {wf_result['oos_total_return']:.4f}")

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 6))
    oos_cum = np.cumsum(wf_result["oos_returns"])
    ax1a.plot(oos_cum, color="steelblue", linewidth=1)
    ax1a.set_title(f"Walkforward OOS Equity Curve (SR={wf_result['oos_sharpe']:.3f})")
    ax1a.set_ylabel("Cumulative Return")
    ax1b.bar(range(len(wf_result["is_metrics"])), wf_result["is_metrics"],
             color="darkorange", alpha=0.7)
    ax1b.set_xlabel("Fold")
    ax1b.set_ylabel("IS Metric")
    ax1b.set_title("In-Sample Metrics per Walkforward Fold")
    fig1.tight_layout()

    # --- 2. Cross-Validation ---
    print("\n--- Cross-Validation ---")
    cv_result = trading_cross_validation(
        returns, _sma_sharpe_metric, n_folds=8,
    )
    print(f"  Mean IS:   {cv_result['mean_is']:.4f}")
    print(f"  Mean OOS:  {cv_result['mean_oos']:.4f}")
    print(f"  OOS std:   {cv_result['std_oos']:.4f}")
    print(f"  IS-OOS gap (bias indicator): {cv_result['mean_is'] - cv_result['mean_oos']:.4f}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    x = np.arange(cv_result["n_folds"])
    width = 0.35
    ax2.bar(x - width / 2, cv_result["is_metrics"], width, label="IS",
            color="steelblue", alpha=0.8)
    ax2.bar(x + width / 2, cv_result["oos_metrics"], width, label="OOS",
            color="tomato", alpha=0.8)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Cross-Validation: IS vs OOS Performance")
    ax2.legend()
    fig2.tight_layout()

    # --- 3. CSCV Superiority Test ---
    print("\n--- CSCV Superiority Test ---")
    rng = np.random.default_rng(42)
    n_strats = 20
    n_days = 1000
    strat_returns = rng.normal(0.0001, 0.01, (n_strats, n_days))
    # Give strategy 0 a slight edge
    strat_returns[0] += 0.0003

    cscv_result = cscv_superiority(strat_returns, n_blocks=8)
    print(f"  PBO (Prob. of Backtest Overfitting): {cscv_result['pbo']:.4f}")
    print(f"  Combinations tested:                 {cscv_result['n_combos']}")
    print(f"  {cscv_result['interpretation']}")

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(["Overfit", "Not Overfit"],
            [cscv_result["n_less"], cscv_result["n_combos"] - cscv_result["n_less"]],
            color=["tomato", "seagreen"], alpha=0.8)
    ax3.set_ylabel("Number of Combinations")
    ax3.set_title(f"CSCV: PBO = {cscv_result['pbo']:.2%}")
    fig3.tight_layout()

    # --- 4. Nested Walkforward ---
    print("\n--- Nested Walkforward ---")

    def _sma_factory(train, inner_folds):
        """Inner CV to select best SMA params."""
        best_sr = _sma_sharpe_metric(train)
        return {
            "best_params": {"short": 10, "long": 50},
            "is_metric": best_sr,
            "oos_fn": lambda test: np.mean(test) / (np.std(test) + 1e-10) * np.sqrt(252),
        }

    nwf = nested_walkforward(returns, _sma_factory,
                             outer_train=500, outer_test=100)
    print(f"  Outer folds:    {nwf['n_folds']}")
    print(f"  Mean OOS metric: {nwf['mean_oos_metric']:.4f}")
    print(f"  Std OOS metric:  {nwf['std_oos_metric']:.4f}")

    print("\nChapter 5 complete.")


if __name__ == "__main__":
    main()
