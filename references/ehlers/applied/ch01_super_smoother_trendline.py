"""
Chapter 1: Super Smoother & Instantaneous Trendline
=====================================================
Ehlers — Cybernetic Analysis for Stocks & Futures

Demonstrates:
  1. 2-pole Super Smoother vs SMA lag comparison
  2. 3-pole Super Smoother for long-period smoothing
  3. Instantaneous Trendline (zero-lag via dominant cycle)
  4. Price vs trendline crossover signals

Reference: Ehlers, Cybernetic Analysis Ch. 13
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

from strategies.ehlers_dsp import (
    super_smoother,
    three_pole_super_smoother,
    instantaneous_trendline,
)


if __name__ == "__main__":
    print("=" * 70)
    print("Ehlers Ch.1: Super Smoother & Instantaneous Trendline")
    print("=" * 70)

    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]

        # Compute indicators
        ss10 = super_smoother(close, period=10)
        ss20 = super_smoother(close, period=20)
        ss3p = three_pole_super_smoother(close, period=20)
        sma20 = close.rolling(20).mean()
        itrend = instantaneous_trendline(close)

        print(f"\n{'─' * 50}")
        print(f"  {sym}: {len(close)} bars")

        # Lag comparison: SMA(20) vs SuperSmoother(20)
        # Approximate lag = index of max cross-correlation
        ret = close.pct_change().dropna()
        sma_lag = int(20 / 2)
        ss_lag = int(20 / 4)
        print(f"  SMA(20) theoretical lag: {sma_lag} bars")
        print(f"  SuperSmoother(20) theoretical lag: ~{ss_lag} bars")
        print(f"  InstTrendline adapts period to dominant cycle")

        # Plot: Price + Smoothers
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        tail = min(500, len(close))

        ax = axes[0]
        ax.plot(close.index[-tail:], close.values[-tail:], alpha=0.4, color="gray", label="Close")
        ax.plot(ss10.index[-tail:], ss10.values[-tail:], linewidth=1.5, color="blue", label="SS(10)")
        ax.plot(ss20.index[-tail:], ss20.values[-tail:], linewidth=1.5, color="red", label="SS(20)")
        ax.plot(sma20.index[-tail:], sma20.values[-tail:], linewidth=1, linestyle="--", color="green", label="SMA(20)")
        ax.set_title(f"{sym} — Super Smoother vs SMA")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(close.index[-tail:], close.values[-tail:], alpha=0.4, color="gray", label="Close")
        ax.plot(itrend.index[-tail:], itrend.values[-tail:], linewidth=2, color="darkorange", label="Inst. Trendline")
        ax.plot(ss3p.index[-tail:], ss3p.values[-tail:], linewidth=1, color="purple", label="3-pole SS(20)")
        ax.set_title(f"{sym} — Instantaneous Trendline (Zero-Lag)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()

    print("\n✓ Chapter 1 complete.")
