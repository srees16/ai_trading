"""
Chapter 2: Fisher Transform & Cyber Cycle
==========================================
Ehlers — Cybernetic Analysis for Stocks & Futures

Demonstrates:
  1. Fisher Transform — Gaussian normalisation of price channel
  2. Fisher crossover signals (Fisher vs Trigger)
  3. Cyber Cycle Oscillator — pure cycle extraction
  4. Cycle-based overbought/oversold detection

Reference: Ehlers, Cybernetic Analysis Ch. 1 & Ch. 4
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

from strategies.ehlers_dsp import fisher_transform, cyber_cycle


if __name__ == "__main__":
    print("=" * 70)
    print("Ehlers Ch.2: Fisher Transform & Cyber Cycle")
    print("=" * 70)

    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]

        fisher, trigger = fisher_transform(close, period=10)
        cc = cyber_cycle(close, alpha=0.07)

        # Count Fisher crossover signals
        cross_up = ((fisher > trigger) & (fisher.shift(1) <= trigger.shift(1))).sum()
        cross_dn = ((fisher < trigger) & (fisher.shift(1) >= trigger.shift(1))).sum()

        print(f"\n{'─' * 50}")
        print(f"  {sym}: {len(close)} bars")
        print(f"  Fisher crossovers — Up: {cross_up}, Down: {cross_dn}")
        print(f"  Fisher range: [{fisher.min():.2f}, {fisher.max():.2f}]")
        print(f"  Cyber Cycle range: [{cc.min():.4f}, {cc.max():.4f}]")

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1, 1]})
        tail = min(500, len(close))

        ax = axes[0]
        ax.plot(close.index[-tail:], close.values[-tail:], color="gray", alpha=0.7, label="Close")
        ax.set_title(f"{sym} — Price")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(fisher.index[-tail:], fisher.values[-tail:], color="blue", linewidth=1.2, label="Fisher")
        ax.plot(trigger.index[-tail:], trigger.values[-tail:], color="red", linewidth=1, linestyle="--", label="Trigger")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.fill_between(fisher.index[-tail:], fisher.values[-tail:], trigger.values[-tail:],
                        where=fisher.values[-tail:] > trigger.values[-tail:],
                        alpha=0.15, color="green")
        ax.fill_between(fisher.index[-tail:], fisher.values[-tail:], trigger.values[-tail:],
                        where=fisher.values[-tail:] < trigger.values[-tail:],
                        alpha=0.15, color="red")
        ax.set_title(f"{sym} — Fisher Transform (period=10)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        cc_vals = cc.values[-tail:]
        ax.plot(cc.index[-tail:], cc_vals, color="purple", linewidth=1.2, label="Cyber Cycle")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.fill_between(cc.index[-tail:], cc_vals, 0,
                        where=cc_vals > 0, alpha=0.15, color="green")
        ax.fill_between(cc.index[-tail:], cc_vals, 0,
                        where=cc_vals < 0, alpha=0.15, color="red")
        ax.set_title(f"{sym} — Cyber Cycle Oscillator")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()

    print("\n✓ Chapter 2 complete.")
