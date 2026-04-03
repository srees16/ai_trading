"""
Chapter 2: Benjamini-Hochberg FDR Control
==========================================
Evidence-Based Technical Analysis — David Aronson

Demonstrates:
  1. Generating N signal variants from parametric sweeps
  2. Computing raw p-values for each via t-test
  3. Applying BH-FDR procedure at q=0.10
  4. Comparing naive (unadjusted) vs FDR-corrected significance
  5. Visualising adjusted vs raw p-values

Reference: Aronson Ch. 6 — When testing many signals, raw p-values
overstate significance. BH-FDR controls the expected false discovery
proportion among rejected hypotheses.
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
    benjamini_hochberg,
)


def _generate_sma_variants(close: pd.Series, raw_ret: pd.Series) -> dict:
    """Generate SMA crossover signal returns for many (fast, slow) pairs."""
    variants = {}
    for fast in [5, 10, 15, 20, 30]:
        for slow in [30, 50, 80, 100, 150, 200]:
            if fast >= slow:
                continue
            sma_f = close.rolling(fast).mean()
            sma_s = close.rolling(slow).mean()
            sig = pd.Series(np.where(sma_f > sma_s, 1, -1), index=close.index).shift(1)
            common = sig.dropna().index.intersection(raw_ret.index)
            sig_ret = raw_ret.reindex(common) * sig.reindex(common)
            variants[f"SMA({fast}/{slow})"] = sig_ret.dropna().values
    return variants


if __name__ == "__main__":
    print("=" * 70)
    print("Aronson Ch.2: Benjamini-Hochberg FDR Control")
    print("=" * 70)

    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]
        raw_ret = close.pct_change().dropna()
        detrended = detrend_returns(raw_ret)

        # Generate many signal variants
        variants = _generate_sma_variants(close, detrended)
        n_signals = len(variants)
        print(f"\n{'─' * 50}")
        print(f"  {sym}: Testing {n_signals} SMA crossover parameter combos")

        # Compute raw p-values
        raw_pvalues = []
        for name, rets in variants.items():
            t, p = compute_signal_tstat(rets)
            raw_pvalues.append((name, p))

        # Count naive significant (p < 0.05)
        naive_sig = sum(1 for _, p in raw_pvalues if p < 0.05)

        # Apply BH-FDR
        bh_results = benjamini_hochberg(raw_pvalues, q=0.10)
        bh_sig = sum(1 for _, _, _, sig in bh_results if sig)

        print(f"  Naive p<0.05 significant: {naive_sig}/{n_signals}")
        print(f"  BH-FDR q=0.10 significant: {bh_sig}/{n_signals}")
        print(f"  False discoveries prevented: {naive_sig - bh_sig}")

        if bh_results:
            print(f"\n  {'Signal':<18} {'Raw p':>10} {'Adj p':>10} {'BH Sig?':>8}")
            print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*8}")
            for name, raw_p, adj_p, sig in bh_results[:15]:
                mark = "YES ✓" if sig else "no"
                print(f"  {name:<18} {raw_p:>10.6f} {adj_p:>10.6f} {mark:>8}")
            if len(bh_results) > 15:
                print(f"  ... and {len(bh_results) - 15} more")

        # Plot: raw vs adjusted p-values
        if bh_results:
            raw_ps = [r[1] for r in bh_results]
            adj_ps = [r[2] for r in bh_results]
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(bh_results))
            ax.scatter(x, raw_ps, s=20, label="Raw p", alpha=0.7, color="steelblue")
            ax.scatter(x, adj_ps, s=20, label="BH-adjusted p", alpha=0.7, color="darkorange")
            ax.axhline(0.05, color="red", linestyle="--", alpha=0.6, label="α=0.05")
            ax.axhline(0.10, color="green", linestyle="--", alpha=0.6, label="FDR q=0.10")
            ax.set_xlabel("Signal rank (sorted by raw p)")
            ax.set_ylabel("p-value")
            ax.set_title(f"{sym} — BH-FDR: Raw vs Adjusted p-values ({n_signals} signals)")
            ax.legend(fontsize=8)
            ax.set_ylim(-0.02, 1.02)
            plt.tight_layout()

    print("\n✓ Chapter 2 complete.")
