"""
Chapter 2: Pre-optimization Issues
====================================
Testing and Tuning Market Trading Systems — Timothy Masters

Algorithms:
  - Stationarity assessment via gap analysis (STATN program)
  - Stationarity improvement via oscillating / extreme induction
  - Relative entropy of an indicator (Equation 2-1)
  - Entropy-based indicator quality assessment
  - Monotonic tail-only cleaning for outlier removal
  - ENTROPY program — full relative-entropy pipeline
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_data import (
    get_close_series,
    generate_ohlcv_bars,
    generate_indicator_series,
)

# ======================================================================
# Relative Entropy  (Equation 2-1, p.31 C++ conversion)
# ======================================================================

def relative_entropy(x, nbins=20):
    """Compute the relative entropy of a data vector.

    Relative entropy is the Shannon entropy divided by its theoretical
    maximum (log(nbins)), yielding a value in [0, 1].

    Converted from STATN.CPP / ENTROPY.CPP.

    Parameters
    ----------
    x     : array-like – data values
    nbins : int        – number of histogram bins

    Returns
    -------
    float – relative entropy in [0, 1]
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0

    xmin, xmax = x.min(), x.max()
    factor = (nbins - 1e-10) / (xmax - xmin + 1e-60)

    counts = np.zeros(nbins, dtype=int)
    for val in x:
        k = int(factor * (val - xmin))
        counts[k] += 1

    total = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            total += p * np.log(p)

    return -total / np.log(nbins)


# ======================================================================
# Gap Analysis for Stationarity  (STATN.CPP, pp.15–17)
# ======================================================================

def gap_analyze(indicators, quantile_frac=0.5, gap_boundaries=None):
    """Analyse stationarity by counting gap sizes between state transitions.

    For each indicator value, determine whether it is above or below the
    specified quantile.  Track how many consecutive bars remain in the
    same state.  Tally the counts into logarithmic bins.

    Converted from gap_analyze() in STATN.CPP.

    Parameters
    ----------
    indicators     : array-like – indicator time series
    quantile_frac  : float      – fractile for above/below threshold (0-1)
    gap_boundaries : list[int]  – bin boundaries (default: 1,2,4,8,...,512+)

    Returns
    -------
    dict with keys 'bins', 'counts', 'quantile_value'
    """
    indicators = np.asarray(indicators, dtype=float)
    n = len(indicators)

    if gap_boundaries is None:
        gap_boundaries = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    nbins = len(gap_boundaries) + 1  # last bin = "> max boundary"

    sorted_vals = np.sort(indicators)
    k = max(0, int(quantile_frac * (n + 1)) - 1)
    quantile_val = sorted_vals[min(k, n - 1)]

    # State: 0 = below quantile, 1 = above or equal
    states = (indicators >= quantile_val).astype(int)

    gap_counts = np.zeros(nbins, dtype=int)
    current_gap = 0

    for i in range(1, n):
        current_gap += 1
        if states[i] != states[i - 1]:
            # State changed — tally the gap
            binned = False
            for bi, boundary in enumerate(gap_boundaries):
                if current_gap <= boundary:
                    gap_counts[bi] += 1
                    binned = True
                    break
            if not binned:
                gap_counts[-1] += 1
            current_gap = 0

    # Final gap if any
    if current_gap > 0:
        binned = False
        for bi, boundary in enumerate(gap_boundaries):
            if current_gap <= boundary:
                gap_counts[bi] += 1
                binned = True
                break
        if not binned:
            gap_counts[-1] += 1

    bin_labels = [str(b) for b in gap_boundaries] + [f">{gap_boundaries[-1]}"]
    return {
        "bins": bin_labels,
        "counts": gap_counts.tolist(),
        "quantile_value": quantile_val,
    }


# ======================================================================
# Stationarity Assessment  (STATN program, pp.21–25)
# ======================================================================

def find_slope(prices, lookback):
    """Compute the slope of a least-squares regression line over a window."""
    n = len(prices)
    if n < lookback:
        return np.nan
    x = np.arange(lookback)
    y = prices[-lookback:]
    slope, _, _, _, _ = linregress(x, y)
    return slope


def stationarity_test(close, lookback=50, fractile=0.5, version=0):
    """Run the STATN stationarity test on a price series.

    Computes trend (slope) and volatility (ATR) indicators, then runs
    gap analysis on each.

    Parameters
    ----------
    close    : array-like – close prices
    lookback : int        – lookback window for indicators
    fractile : float      – quantile fractile (0–1)
    version  : int        – 0=raw, 1=differenced, >1=long-window differenced

    Returns
    -------
    dict with 'trend_gaps', 'volatility_gaps', 'trend', 'volatility'
    """
    close = np.asarray(close, dtype=float)
    n = len(close)

    if version == 0:
        full_lookback = lookback
    elif version == 1:
        full_lookback = 2 * lookback
    else:
        full_lookback = version * lookback

    nind = n - full_lookback + 1
    if nind < 10:
        raise ValueError("Not enough data for the specified lookback")

    trend = np.zeros(nind)
    volatility = np.zeros(nind)

    for i in range(nind):
        k = full_lookback - 1 + i
        window = close[k - lookback + 1: k + 1]

        if version == 0:
            trend[i] = find_slope(window, lookback)
        elif version == 1:
            past_window = close[k - 2 * lookback + 1: k - lookback + 1]
            trend[i] = find_slope(window, lookback) - find_slope(past_window, lookback)
        else:
            long_window = close[k - full_lookback + 1: k + 1]
            trend[i] = find_slope(window, lookback) - find_slope(long_window, full_lookback)

        # Volatility = average true range approximation
        high_low_range = np.max(window) - np.min(window)
        volatility[i] = high_low_range / (lookback * np.mean(window) + 1e-10)

    trend_gaps = gap_analyze(trend, fractile)
    vol_gaps = gap_analyze(volatility, fractile)

    return {
        "trend_gaps": trend_gaps,
        "volatility_gaps": vol_gaps,
        "trend": trend,
        "volatility": volatility,
        "nind": nind,
    }


# ======================================================================
# Improving Stationarity by Oscillating  (pp.25–27)
# ======================================================================

def oscillate_indicator(indicator, lookback):
    """Make an indicator more stationary by subtracting its lagged value.

    This is the Version=1 technique from STATN: subtracting the value
    from *lookback* bars ago converts a potentially trending indicator
    into an oscillator centred near zero.

    Parameters
    ----------
    indicator : array-like
    lookback  : int

    Returns
    -------
    np.ndarray – differenced indicator
    """
    indicator = np.asarray(indicator, dtype=float)
    return indicator[lookback:] - indicator[:-lookback]


# ======================================================================
# Monotonic Tail-Only Cleaning  (pp.37–39)
# ======================================================================

def monotonic_tail_clean(x, tail_fraction=0.05):
    """Clean outliers by monotonically compressing extreme tails.

    Values beyond the (1 - tail_fraction) quantile on either side are
    replaced with linearly spaced values between the boundary and the
    most extreme value, effectively reducing the impact of outliers
    while preserving their ordering.

    Parameters
    ----------
    x             : array-like – the values to clean
    tail_fraction : float      – fraction of data considered a tail (each side)

    Returns
    -------
    np.ndarray – cleaned values
    """
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    n_tail = max(1, int(n * tail_fraction))

    sorted_idx = np.argsort(x)

    # Lower tail
    lower_boundary = x[sorted_idx[n_tail]]
    lower_min = x[sorted_idx[0]]
    lower_indices = sorted_idx[:n_tail]
    if n_tail > 1:
        replacement = np.linspace(lower_min * 0.5 + lower_boundary * 0.5,
                                  lower_boundary, n_tail)
        x[lower_indices] = replacement[np.argsort(np.argsort(x[lower_indices]))]

    # Upper tail
    upper_boundary = x[sorted_idx[-(n_tail + 1)]]
    upper_max = x[sorted_idx[-1]]
    upper_indices = sorted_idx[-n_tail:]
    if n_tail > 1:
        replacement = np.linspace(upper_boundary,
                                  upper_max * 0.5 + upper_boundary * 0.5, n_tail)
        x[upper_indices] = replacement[np.argsort(np.argsort(x[upper_indices]))]

    return x


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("Chapter 2 – Pre-optimization Issues")
    print("  (Testing and Tuning Market Trading Systems)")
    print("=" * 70)

    # --- 1. Relative Entropy ---
    print("\n--- Relative Entropy of Indicators ---")
    indicators = generate_indicator_series(2000, n_indicators=4, seed=42)
    for col in indicators.columns:
        ent = relative_entropy(indicators[col].values, nbins=20)
        print(f"  {col}: entropy = {ent:.4f}")

    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 6))
    for ax, col in zip(axes1.flat, indicators.columns):
        ent = relative_entropy(indicators[col].values, nbins=20)
        ax.hist(indicators[col], bins=40, alpha=0.7, edgecolor="white",
                color="steelblue")
        ax.set_title(f"{col}  (entropy={ent:.3f})")
    fig1.suptitle("Indicator Entropy — Distribution Shape vs Information Content",
                  fontsize=12)
    fig1.tight_layout()

    # --- 2. Stationarity Test (STATN) ---
    print("\n--- Stationarity Test (STATN) ---")
    close = get_close_series("SPY")
    close_arr = close.values

    for ver in [0, 1]:
        result = stationarity_test(close_arr, lookback=50, fractile=0.5,
                                   version=ver)
        print(f"\n  Version {ver}:")
        tg = result["trend_gaps"]
        vg = result["volatility_gaps"]
        print(f"    Trend gaps:     {dict(zip(tg['bins'], tg['counts']))}")
        print(f"    Volatility gaps: {dict(zip(vg['bins'], vg['counts']))}")

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    result_v0 = stationarity_test(close_arr, lookback=50, fractile=0.5, version=0)
    ax2a.plot(result_v0["trend"], label="Trend (slope)", linewidth=0.8)
    ax2a.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2a.set_title("Trend Indicator (Version 0 — Raw)")
    ax2a.legend()
    ax2b.plot(result_v0["volatility"], label="Volatility (range/mean)",
              linewidth=0.8, color="darkorange")
    ax2b.set_title("Volatility Indicator")
    ax2b.legend()
    fig2.suptitle("STATN: Stationarity Diagnostics for SPY", fontsize=12)
    fig2.tight_layout()

    # --- 3. Gap Analysis Visualisation ---
    print("\n--- Gap Analysis ---")
    trend = result_v0["trend"]
    gaps = gap_analyze(trend, quantile_frac=0.5)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(gaps["bins"], gaps["counts"], color="steelblue", edgecolor="white")
    ax3.set_xlabel("Gap size (bars)")
    ax3.set_ylabel("Count")
    ax3.set_title("Trend Gap Distribution — Ideal Is Exponential Decay")
    fig3.tight_layout()

    # --- 4. Monotonic Tail Cleaning ---
    print("\n--- Monotonic Tail Cleaning ---")
    noisy = generate_indicator_series(1000, n_indicators=1, seed=99)["Ind_0"]
    # add extreme outliers
    rng = np.random.default_rng(99)
    noisy_arr = noisy.values.copy()
    outlier_idx = rng.choice(len(noisy_arr), 20, replace=False)
    noisy_arr[outlier_idx] = rng.normal(0, 10, 20)

    cleaned = monotonic_tail_clean(noisy_arr, tail_fraction=0.05)

    ent_before = relative_entropy(noisy_arr, nbins=20)
    ent_after = relative_entropy(cleaned, nbins=20)
    print(f"  Entropy before cleaning: {ent_before:.4f}")
    print(f"  Entropy after cleaning:  {ent_after:.4f}")
    print(f"  Improvement:             {ent_after - ent_before:+.4f}")

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 4))
    ax4a.hist(noisy_arr, bins=40, alpha=0.7, color="tomato", edgecolor="white")
    ax4a.set_title(f"Before Cleaning (ent={ent_before:.3f})")
    ax4b.hist(cleaned, bins=40, alpha=0.7, color="seagreen", edgecolor="white")
    ax4b.set_title(f"After Tail Cleaning (ent={ent_after:.3f})")
    fig4.suptitle("Monotonic Tail-Only Cleaning", fontsize=12)
    fig4.tight_layout()

    # --- 5. Oscillation for Stationarity ---
    print("\n--- Oscillating Indicator ---")
    oscillated = oscillate_indicator(result_v0["trend"], lookback=20)
    ent_raw = relative_entropy(result_v0["trend"], nbins=20)
    ent_osc = relative_entropy(oscillated, nbins=20)
    print(f"  Raw trend entropy:        {ent_raw:.4f}")
    print(f"  Oscillated trend entropy: {ent_osc:.4f}")

    print("\nChapter 2 complete.")


if __name__ == "__main__":
    main()
