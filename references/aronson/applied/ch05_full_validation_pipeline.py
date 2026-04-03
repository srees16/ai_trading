"""
Chapter 5: Full Aronson Validation Pipeline
=============================================
Evidence-Based Technical Analysis — David Aronson

Demonstrates the complete end-to-end validation:
  1. Build multiple signal variants from real data
  2. Detrend returns
  3. Compute t-stat gate per signal
  4. Apply BH-FDR multi-test correction
  5. Run White's Reality Check on the best signal
  6. Estimate data-mining bias
  7. Compute composite confidence score
  8. Generate summary report with weight multipliers

Reference: Aronson Ch. 1,5,6,8,9 synthesised into the
AronsonValidator class pipeline.
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
    AronsonValidator,
    detrend_returns,
)


def _build_multi_signal_returns(close: pd.Series, detrended: pd.Series) -> dict:
    """Build diverse signal return series for testing."""
    signals = {}

    # SMA crossovers
    for fast, slow in [(5, 20), (10, 50), (20, 100), (30, 150)]:
        sma_f = close.rolling(fast).mean()
        sma_s = close.rolling(slow).mean()
        sig = pd.Series(np.where(sma_f > sma_s, 1, -1), index=close.index).shift(1)
        common = sig.dropna().index.intersection(detrended.index)
        signals[f"SMA_{fast}_{slow}"] = (detrended.reindex(common) * sig.reindex(common)).dropna()

    # Momentum
    for lb in [10, 20, 60]:
        mom = close.pct_change(lb)
        sig = pd.Series(np.where(mom > 0, 1, -1), index=close.index).shift(1)
        common = sig.dropna().index.intersection(detrended.index)
        signals[f"Mom_{lb}"] = (detrended.reindex(common) * sig.reindex(common)).dropna()

    # RSI
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        sig = pd.Series(0.0, index=close.index)
        sig[rsi < 30] = 1.0
        sig[rsi > 70] = -1.0
        sig = sig.shift(1)
        common = sig.dropna().index.intersection(detrended.index)
        sr = detrended.reindex(common) * sig.reindex(common)
        sr = sr[sr != 0].dropna()
        if len(sr) > 30:
            signals[f"RSI_{period}"] = sr

    # Mean reversion: Bollinger band bounce
    for window in [20, 50]:
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        z = (close - sma) / std.replace(0, np.nan)
        sig = pd.Series(0.0, index=close.index)
        sig[z < -2] = 1.0
        sig[z > 2] = -1.0
        sig = sig.shift(1)
        common = sig.dropna().index.intersection(detrended.index)
        sr = detrended.reindex(common) * sig.reindex(common)
        sr = sr[sr != 0].dropna()
        if len(sr) > 30:
            signals[f"BB_{window}"] = sr

    return signals


if __name__ == "__main__":
    print("=" * 70)
    print("Aronson Ch.5: Full Validation Pipeline")
    print("=" * 70)

    prices = get_prices()
    validator = AronsonValidator()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]
        raw_ret = close.pct_change().dropna()
        detrended = detrend_returns(raw_ret)

        # Build signal returns
        signal_returns = _build_multi_signal_returns(close, detrended)
        if not signal_returns:
            print(f"\n  {sym}: no valid signals generated — skipping")
            continue

        # Convert to arrays for AronsonValidator
        sig_dict = {name: sr.values for name, sr in signal_returns.items()}

        print(f"\n{'═' * 60}")
        print(f"  {sym}: {len(sig_dict)} signals to validate")
        print(f"{'═' * 60}")

        # Run full pipeline
        summary = validator.validate_signals(sig_dict, benchmark_returns=raw_ret.values)

        print(f"\n  WRC best signal: {summary.wrc_best_signal} (p={summary.wrc_best_p_value:.4f})")
        print(f"  Data-mining bias: {summary.dm_bias_estimate*100:.2f}% of Sharpe")
        print(f"  Signals validated: {summary.n_validated}/{summary.n_total}")

        print(f"\n  {'Signal':<18} {'t-stat':>8} {'BH adj-p':>10} {'BH Sig':>7} {'Weight':>8}")
        print(f"  {'─'*18} {'─'*8} {'─'*10} {'─'*7} {'─'*8}")
        for sv in sorted(summary.signals, key=lambda s: s.weight_multiplier, reverse=True):
            mark = "✓" if sv.bh_significant else " "
            print(f"  {sv.name:<18} {sv.t_stat:>8.3f} {sv.bh_adjusted_p:>10.4f} {mark:>7} {sv.weight_multiplier:>8.3f}")

        # Plot: weight multipliers
        names = [sv.name for sv in sorted(summary.signals, key=lambda s: s.weight_multiplier, reverse=True)]
        weights = [sv.weight_multiplier for sv in sorted(summary.signals, key=lambda s: s.weight_multiplier, reverse=True)]
        colors = ["green" if sv.bh_significant else "gray"
                  for sv in sorted(summary.signals, key=lambda s: s.weight_multiplier, reverse=True)]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, weights, color=colors, alpha=0.8, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Weight Multiplier")
        ax.set_title(f"{sym} — Aronson Validation: Signal Weights\n"
                     f"(green=BH-significant, validated {summary.n_validated}/{summary.n_total})")
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="Baseline=1.0")
        ax.legend(fontsize=8)
        plt.tight_layout()

    print("\n✓ Chapter 5 complete.")
