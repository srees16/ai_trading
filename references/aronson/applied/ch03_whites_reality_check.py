"""
Chapter 3: White's Reality Check
=================================
Evidence-Based Technical Analysis — David Aronson

Demonstrates:
  1. The multiple-testing problem: best-of-N overfitting
  2. Bootstrap construction of the null distribution
  3. Computing the WRC p-value for the best signal
  4. Visualising the null distribution vs observed statistic

Reference: Aronson Ch. 8 — White's Reality Check uses circular block
bootstrap of the return series to build a distribution of the best
t-statistic under the null (no signal has skill). If observed best
exceeds the 95th percentile, the best signal is genuine.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import get_prices, SYMBOLS

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.aronson_validator import (
    detrend_returns,
    compute_signal_tstat,
    whites_reality_check,
)


def _build_signal_matrix(close: pd.Series, detrended: pd.Series) -> tuple:
    """Build matrix of signal returns for multiple SMA variants."""
    signals = {}
    for fast in [5, 10, 15, 20, 30]:
        for slow in [30, 50, 80, 100, 150]:
            if fast >= slow:
                continue
            sma_f = close.rolling(fast).mean()
            sma_s = close.rolling(slow).mean()
            sig = pd.Series(np.where(sma_f > sma_s, 1, -1), index=close.index).shift(1)
            common = sig.dropna().index.intersection(detrended.index)
            sig_ret = detrended.reindex(common) * sig.reindex(common)
            signals[f"SMA({fast}/{slow})"] = sig_ret

    # Align all to common index
    df = pd.DataFrame(signals).dropna()
    return df.values, list(df.columns)


if __name__ == "__main__":
    print("=" * 70)
    print("Aronson Ch.3: White's Reality Check")
    print("=" * 70)

    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]
        raw_ret = close.pct_change().dropna()
        detrended = detrend_returns(raw_ret)

        matrix, names = _build_signal_matrix(close, detrended)
        n_signals = matrix.shape[1]

        print(f"\n{'─' * 50}")
        print(f"  {sym}: {n_signals} signal variants, {matrix.shape[0]} observations")

        # Find best signal by t-stat
        best_t = -np.inf
        best_name = ""
        for i, name in enumerate(names):
            t, p = compute_signal_tstat(matrix[:, i])
            if t > best_t:
                best_t = t
                best_name = name

        print(f"  Best signal: {best_name} (t={best_t:.3f})")

        # White's Reality Check
        wrc_p, best_idx = whites_reality_check(matrix, n_bootstrap=2000, seed=42)
        print(f"  WRC p-value: {wrc_p:.4f}")
        if wrc_p < 0.05:
            print(f"  → PASS: Best signal survives WRC at 5% level")
        else:
            print(f"  → FAIL: Best signal does NOT survive WRC — likely data-snooped")

        # Monte Carlo to build null distribution for plot
        rng = np.random.default_rng(42)
        n_obs = matrix.shape[0]
        block_size = max(1, int(np.sqrt(n_obs)))
        null_best_ts = []
        for _ in range(2000):
            indices = []
            while len(indices) < n_obs:
                start = rng.integers(0, n_obs)
                indices.extend(range(start, min(start + block_size, n_obs)))
            indices = indices[:n_obs]
            boot_matrix = matrix[indices]
            # Compute best t-stat under null
            ts = []
            for j in range(n_signals):
                col = boot_matrix[:, j]
                mean = np.mean(col)
                std = np.std(col, ddof=1)
                if std > 0:
                    ts.append(mean / (std / np.sqrt(len(col))))
                else:
                    ts.append(0.0)
            null_best_ts.append(max(ts))

        # Plot null distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(null_best_ts, bins=50, alpha=0.7, color="steelblue", edgecolor="white",
                density=True, label="Null: best t-stat distribution")
        ax.axvline(best_t, color="red", linewidth=2, label=f"Observed best t={best_t:.2f}")
        pct95 = np.percentile(null_best_ts, 95)
        ax.axvline(pct95, color="green", linestyle="--", label=f"95th percentile={pct95:.2f}")
        ax.set_xlabel("Best t-statistic (bootstrap null)")
        ax.set_ylabel("Density")
        ax.set_title(f"{sym} — White's Reality Check (N={n_signals} signals, WRC p={wrc_p:.3f})")
        ax.legend(fontsize=8)
        plt.tight_layout()

    print("\n✓ Chapter 3 complete.")
