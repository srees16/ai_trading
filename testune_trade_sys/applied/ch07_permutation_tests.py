"""
Chapter 7: Permutation Tests
==============================
Testing and Tuning Market Trading Systems — Timothy Masters

Algorithms:
  - Overview of permutation testing (pp.287–294)
  - Testing a fully specified trading system (pp.289–290)
  - Walkforward permutation testing a trading system factory (pp.291–293)
  - Simple price permutation (pp.302–303)
  - Permuting simple market prices (pp.303–305)
  - Permuting multiple markets with offset (pp.305–309)
  - Permuting price bars (OHLCV) (pp.309–314)
  - Partitioning total return of a trading system (pp.298–302)
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
    get_prices,
    generate_ohlcv_bars,
    generate_returns,
)


# ======================================================================
# Simple Permutation  (p.302)
# ======================================================================

def permute_returns(returns, seed=None):
    """Randomly permute a return series (destroys temporal structure).

    This is the simplest permutation method: shuffle the returns.
    The resulting series has the same distribution but no temporal
    dependencies — any profitable behavior must be due to random chance.

    Parameters
    ----------
    returns : array-like
    seed    : int or None

    Returns
    -------
    np.ndarray – shuffled returns
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float).copy()
    rng.shuffle(returns)
    return returns


# ======================================================================
# Permuting Simple Market Prices  (pp.303–305)
# ======================================================================

def permute_prices(prices, seed=None):
    """Permute a price series by shuffling the returns, then
    reconstructing prices from the shuffled returns.

    Preserves the return distribution and start/end price level
    (approximately).

    Parameters
    ----------
    prices : array-like – price series
    seed   : int or None

    Returns
    -------
    np.ndarray – permuted price series (same length)
    """
    prices = np.asarray(prices, dtype=float)
    returns = np.diff(np.log(prices))
    perm_returns = permute_returns(returns, seed=seed)
    perm_log_prices = np.concatenate([[np.log(prices[0])],
                                       np.cumsum(perm_returns)])
    return np.exp(perm_log_prices + np.log(prices[0]) - perm_log_prices[0])


# ======================================================================
# Permuting Multiple Markets with Offset  (pp.305–309)
# ======================================================================

def permute_multiple_markets(price_dict, seed=None):
    """Permute prices for multiple markets using a common offset.

    To preserve cross-market correlations within a bar while destroying
    temporal patterns, we use the same permutation index for all markets
    but with a random offset per market.

    Converted from the multi-market permutation algorithm (pp.305–309).

    Parameters
    ----------
    price_dict : dict {symbol: np.ndarray} – price series (same length)
    seed       : int or None

    Returns
    -------
    dict {symbol: np.ndarray} – permuted price series
    """
    rng = np.random.default_rng(seed)
    symbols = list(price_dict.keys())
    n = min(len(v) for v in price_dict.values())

    # Compute log returns for all markets
    log_returns = {}
    for sym in symbols:
        p = np.asarray(price_dict[sym][:n], dtype=float)
        log_returns[sym] = np.diff(np.log(p))

    n_ret = n - 1

    # Base permutation index
    base_perm = rng.permutation(n_ret)

    result = {}
    for sym in symbols:
        # Apply offset
        offset = rng.integers(0, n_ret)
        perm_idx = (base_perm + offset) % n_ret
        perm_ret = log_returns[sym][perm_idx]

        # Reconstruct prices
        start_price = price_dict[sym][0] if isinstance(price_dict[sym], np.ndarray) else float(price_dict[sym].iloc[0])
        perm_log = np.concatenate([[np.log(start_price)],
                                   np.log(start_price) + np.cumsum(perm_ret)])
        result[sym] = np.exp(perm_log)

    return result


# ======================================================================
# Permuting Price Bars (OHLCV)  (pp.309–314)
# ======================================================================

def permute_bars(ohlcv_df, seed=None):
    """Permute OHLCV price bars while preserving intra-bar relationships.

    Each bar's internal structure (Open/High/Low/Close ratios relative
    to Close) is kept intact, but the bar-to-bar returns are permuted.

    Converted from bar permutation algorithm (pp.309–314).

    Parameters
    ----------
    ohlcv_df : pd.DataFrame with columns ['Open','High','Low','Close','Volume']
    seed     : int or None

    Returns
    -------
    pd.DataFrame – permuted OHLCV data
    """
    rng = np.random.default_rng(seed)
    df = ohlcv_df.copy()

    close = df["Close"].values.astype(float)
    n = len(close)

    # Compute bar-to-bar log returns of Close
    log_ret = np.diff(np.log(close))

    # Compute intra-bar ratios (relative to Close)
    open_ratio = df["Open"].values / close
    high_ratio = df["High"].values / close
    low_ratio = df["Low"].values / close

    # Permute the close-to-close returns
    perm_ret = log_ret.copy()
    rng.shuffle(perm_ret)

    # Reconstruct Close
    new_close = np.zeros(n)
    new_close[0] = close[0]
    for i in range(1, n):
        new_close[i] = new_close[i - 1] * np.exp(perm_ret[i - 1])

    # Reconstruct OHLC using stored ratios, but with permuted bar order
    perm_idx = rng.permutation(n)
    new_open = new_close * open_ratio[perm_idx]
    new_high = new_close * high_ratio[perm_idx]
    new_low = new_close * low_ratio[perm_idx]

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    new_high = np.maximum(new_high, np.maximum(new_open, new_close))
    new_low = np.minimum(new_low, np.minimum(new_open, new_close))

    result = pd.DataFrame({
        "Open": new_open,
        "High": new_high,
        "Low": new_low,
        "Close": new_close,
        "Volume": df["Volume"].values[perm_idx],
    }, index=df.index)

    return result


# ======================================================================
# Permutation Test for a Trading System  (pp.289–297)
# ======================================================================

def permutation_test(returns, strategy_fn, n_perms=500, seed=42):
    """Permutation test: is the strategy statistically significant?

    Procedure:
      1. Compute the real strategy metric on the actual returns.
      2. For each permutation, shuffle the returns, re-apply the
         strategy, and record the metric.
      3. The p-value is the fraction of permuted metrics ≥ the real one.

    Parameters
    ----------
    returns     : array-like – daily returns
    strategy_fn : callable   – f(returns) → metric (higher = better)
    n_perms     : int
    seed        : int

    Returns
    -------
    dict with 'real_metric', 'p_value', 'perm_distribution', etc.
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    real_metric = strategy_fn(returns)

    perm_metrics = np.zeros(n_perms)
    for i in range(n_perms):
        perm_ret = returns.copy()
        rng.shuffle(perm_ret)
        perm_metrics[i] = strategy_fn(perm_ret)

    p_value = np.mean(perm_metrics >= real_metric)

    return {
        "real_metric": real_metric,
        "p_value": p_value,
        "mean_perm": perm_metrics.mean(),
        "std_perm": perm_metrics.std(),
        "perm_distribution": perm_metrics,
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
    }


# ======================================================================
# Walkforward Permutation Test  (pp.291–293)
# ======================================================================

def walkforward_permutation_test(returns, strategy_factory_fn,
                                 train_size=400, test_size=80,
                                 n_perms=200, seed=42):
    """Permutation test applied to a walkforward trading system factory.

    The entire walkforward process (train → optimise → test) is repeated
    on permuted data, so the test accounts for the optimisation itself.

    Parameters
    ----------
    returns              : array-like
    strategy_factory_fn  : callable – f(train) → callable predictor
                           The predictor is applied to test returns.
    train_size           : int
    test_size            : int
    n_perms              : int
    seed                 : int

    Returns
    -------
    dict with 'real_metric', 'p_value', etc.
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    def _wf_metric(ret):
        """Run walkforward on a return series and return total OOS return."""
        oos_total = 0.0
        start = 0
        n_folds = 0
        while start + train_size + test_size <= len(ret):
            train = ret[start: start + train_size]
            test = ret[start + train_size: start + train_size + test_size]
            predictor = strategy_factory_fn(train)
            if callable(predictor):
                oos_ret = predictor(test)
                oos_total += np.sum(oos_ret) if hasattr(oos_ret, "__len__") else oos_ret
            else:
                oos_total += np.sum(test)
            n_folds += 1
            start += test_size
        return oos_total

    real_metric = _wf_metric(returns)

    perm_metrics = np.zeros(n_perms)
    for i in range(n_perms):
        perm_ret = returns.copy()
        rng.shuffle(perm_ret)
        perm_metrics[i] = _wf_metric(perm_ret)

    p_value = np.mean(perm_metrics >= real_metric)

    return {
        "real_metric": real_metric,
        "p_value": p_value,
        "mean_perm": perm_metrics.mean(),
        "std_perm": perm_metrics.std(),
        "perm_distribution": perm_metrics,
        "significant_5pct": p_value < 0.05,
    }


# ======================================================================
# Partitioning Total Return  (pp.298–302)
# ======================================================================

def partition_return(returns, strategy_returns, n_perms=500, seed=42):
    """Partition the total strategy return into skill vs luck components.

    Uses permutation to estimate the expected return from luck alone,
    then attributes the remainder to skill.

    Parameters
    ----------
    returns          : array-like – raw market returns
    strategy_returns : array-like – strategy-weighted returns
    n_perms          : int
    seed             : int

    Returns
    -------
    dict with 'total_return', 'luck_component', 'skill_component',
              'luck_fraction', 'skill_fraction'
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    strategy_returns = np.asarray(strategy_returns, dtype=float)

    total = strategy_returns.sum()

    # Estimate luck: shuffle returns, keep same strategy weights
    perm_totals = np.zeros(n_perms)
    for i in range(n_perms):
        perm_ret = returns.copy()
        rng.shuffle(perm_ret)
        # Assume strategy_returns are market_returns * signals
        # With shuffled returns, the signal is uncorrelated
        perm_totals[i] = np.sum(perm_ret * np.sign(strategy_returns))

    luck = perm_totals.mean()
    skill = total - luck

    return {
        "total_return": total,
        "luck_component": luck,
        "skill_component": skill,
        "luck_fraction": luck / (abs(total) + 1e-10),
        "skill_fraction": skill / (abs(total) + 1e-10),
    }


# ======================================================================
# Helper Strategies for Demonstrations
# ======================================================================

def _sma_strategy_returns(returns, short=10, long=50):
    """Simple SMA crossover: returns the per-bar strategy returns."""
    prices = 100 * np.exp(np.cumsum(returns))
    short_ma = pd.Series(prices).rolling(short).mean().values
    long_ma = pd.Series(prices).rolling(long).mean().values
    signal = np.where(short_ma > long_ma, 1, -1)
    strat_ret = returns * np.roll(signal, 1)
    strat_ret[:long] = 0  # warmup
    return strat_ret


def _sma_sharpe(returns):
    """Sharpe of best SMA crossover."""
    best = -np.inf
    for s, l in [(10, 30), (10, 50), (20, 50), (20, 70)]:
        if l >= len(returns):
            continue
        sr = _sma_strategy_returns(returns, s, l)
        sr = sr[l:]
        if len(sr) < 10:
            continue
        metric = np.mean(sr) / (np.std(sr) + 1e-10) * np.sqrt(252)
        if metric > best:
            best = metric
    return best if best > -np.inf else 0.0


def _sma_factory(train):
    """Train an SMA system: find best params, return predictor."""
    best_params = (10, 50)
    best_sr = -np.inf
    for s, l in [(10, 30), (10, 50), (20, 50)]:
        if l >= len(train):
            continue
        sr = _sma_strategy_returns(train, s, l)
        sr = sr[l:]
        if len(sr) < 5:
            continue
        metric = np.mean(sr) / (np.std(sr) + 1e-10) * np.sqrt(252)
        if metric > best_sr:
            best_sr = metric
            best_params = (s, l)

    s, l = best_params

    def predictor(test_ret):
        return _sma_strategy_returns(test_ret, s, l)

    return predictor


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("Chapter 7 – Permutation Tests")
    print("  (Testing and Tuning Market Trading Systems)")
    print("=" * 70)

    prices = get_close_series("SPY")
    returns = prices.pct_change().dropna().values

    # --- 1. Permutation Test ---
    print("\n--- Permutation Test (SMA Crossover on SPY) ---")
    perm_result = permutation_test(returns, _sma_sharpe, n_perms=200, seed=42)
    print(f"  Real Sharpe:   {perm_result['real_metric']:.4f}")
    print(f"  Perm mean:     {perm_result['mean_perm']:.4f}")
    print(f"  Perm std:      {perm_result['std_perm']:.4f}")
    print(f"  p-value:       {perm_result['p_value']:.4f}")
    print(f"  Significant (5%): {perm_result['significant_5pct']}")
    print(f"  Significant (1%): {perm_result['significant_1pct']}")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(perm_result["perm_distribution"], bins=40, alpha=0.7,
             color="steelblue", edgecolor="white")
    ax1.axvline(perm_result["real_metric"], color="tomato", linewidth=2,
                label=f"Real ({perm_result['real_metric']:.3f})")
    ax1.set_title(f"Permutation Test: p={perm_result['p_value']:.3f}")
    ax1.set_xlabel("Sharpe Ratio")
    ax1.legend()
    fig1.tight_layout()

    # --- 2. Price Permutation ---
    print("\n--- Price Permutation ---")
    price_arr = prices.values
    perm_prices = permute_prices(price_arr, seed=42)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4))
    ax2a.plot(price_arr, color="steelblue", linewidth=0.8)
    ax2a.set_title("Original SPY Prices")
    ax2b.plot(perm_prices, color="tomato", linewidth=0.8)
    ax2b.set_title("Permuted SPY Prices")
    fig2.suptitle("Price Permutation — Same Distribution, No Pattern",
                  fontsize=12)
    fig2.tight_layout()

    # --- 3. Bar Permutation ---
    print("\n--- Bar Permutation (OHLCV) ---")
    bars = generate_ohlcv_bars(500, seed=42)
    perm_bars = permute_bars(bars, seed=42)
    print(f"  Original bars shape: {bars.shape}")
    print(f"  Permuted bars shape: {perm_bars.shape}")
    print(f"  Original Close mean: {bars['Close'].mean():.2f}")
    print(f"  Permuted Close mean: {perm_bars['Close'].mean():.2f}")

    # --- 4. Multiple Market Permutation ---
    print("\n--- Multiple Market Permutation ---")
    price_data = get_prices(["SPY", "QQQ"], start="2022-01-01", end="2024-01-01")
    price_dict = {}
    min_len = min(len(df) for df in price_data.values())
    for sym, df in price_data.items():
        price_dict[sym] = df["Close"].values[:min_len]

    if len(price_dict) >= 2:
        perm_multi = permute_multiple_markets(price_dict, seed=42)
        for sym in price_dict:
            orig_corr = np.corrcoef(np.diff(np.log(list(price_dict.values())[0])),
                                    np.diff(np.log(list(price_dict.values())[1])))[0, 1] if len(price_dict) >= 2 else 0
            print(f"  {sym}: original len={len(price_dict[sym])}, permuted len={len(perm_multi[sym])}")
        print(f"  Original cross-correlation: {orig_corr:.4f}")

    # --- 5. Walkforward Permutation Test ---
    print("\n--- Walkforward Permutation Test ---")
    wf_perm = walkforward_permutation_test(
        returns[:1000], _sma_factory,
        train_size=300, test_size=60, n_perms=100, seed=42,
    )
    print(f"  Real WF metric: {wf_perm['real_metric']:.4f}")
    print(f"  p-value:        {wf_perm['p_value']:.4f}")
    print(f"  Significant:    {wf_perm['significant_5pct']}")

    # --- 6. Return Partitioning ---
    print("\n--- Return Partitioning (Skill vs Luck) ---")
    strat_ret = _sma_strategy_returns(returns, short=10, long=50)
    partition = partition_return(returns, strat_ret, n_perms=200, seed=42)
    print(f"  Total return:    {partition['total_return']:.4f}")
    print(f"  Luck component:  {partition['luck_component']:.4f}")
    print(f"  Skill component: {partition['skill_component']:.4f}")
    print(f"  Skill fraction:  {partition['skill_fraction']:.2%}")

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(["Luck", "Skill"],
            [partition["luck_component"], partition["skill_component"]],
            color=["steelblue", "seagreen"], alpha=0.8)
    ax3.set_ylabel("Return")
    ax3.set_title("Return Decomposition: Skill vs Luck")
    fig3.tight_layout()

    print("\nChapter 7 complete.")


if __name__ == "__main__":
    main()
