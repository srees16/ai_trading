"""
Chapter 3: MAMA/FAMA & Adaptive RSI
=====================================
Ehlers — Cybernetic Analysis / Rocket Science for Traders

Demonstrates:
  1. MAMA (MESA Adaptive Moving Average) via Hilbert Transform phase
  2. FAMA (Following Adaptive Moving Average) — slower adaptive
  3. MAMA/FAMA crossover as trend signal
  4. Adaptive RSI — self-tuning period via dominant cycle
  5. Sinewave Indicator — leading turning point detector

Reference: Ehlers, Rocket Science Ch. 8 & Cybernetic Analysis Ch. 6
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
    mama_fama,
    adaptive_rsi,
    sinewave_indicator,
    dominant_cycle_period,
)


if __name__ == "__main__":
    print("=" * 70)
    print("Ehlers Ch.3: MAMA/FAMA & Adaptive RSI")
    print("=" * 70)

    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]

        mama_s, fama_s = mama_fama(close)
        arsi = adaptive_rsi(close)
        sine_s, lead_s = sinewave_indicator(close)
        dc = dominant_cycle_period(close)

        # MAMA/FAMA crossover count
        cross_up = ((mama_s > fama_s) & (mama_s.shift(1) <= fama_s.shift(1))).sum()
        cross_dn = ((mama_s < fama_s) & (mama_s.shift(1) >= fama_s.shift(1))).sum()

        print(f"\n{'─' * 50}")
        print(f"  {sym}: {len(close)} bars")
        print(f"  MAMA/FAMA crossovers — Up: {cross_up}, Down: {cross_dn}")
        print(f"  Dominant cycle (latest): {dc.iloc[-1]:.1f} bars")
        print(f"  Adaptive RSI (latest): {arsi.iloc[-1]:.1f}")

        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [2, 1, 1, 1]})
        tail = min(500, len(close))

        ax = axes[0]
        ax.plot(close.index[-tail:], close.values[-tail:], color="gray", alpha=0.5, label="Close")
        ax.plot(mama_s.index[-tail:], mama_s.values[-tail:], color="blue", linewidth=1.5, label="MAMA")
        ax.plot(fama_s.index[-tail:], fama_s.values[-tail:], color="red", linewidth=1, label="FAMA")
        ax.fill_between(mama_s.index[-tail:], mama_s.values[-tail:], fama_s.values[-tail:],
                        where=mama_s.values[-tail:] > fama_s.values[-tail:],
                        alpha=0.1, color="green")
        ax.fill_between(mama_s.index[-tail:], mama_s.values[-tail:], fama_s.values[-tail:],
                        where=mama_s.values[-tail:] < fama_s.values[-tail:],
                        alpha=0.1, color="red")
        ax.set_title(f"{sym} — MAMA/FAMA Adaptive Moving Averages")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        arsi_vals = arsi.values[-tail:]
        ax.plot(arsi.index[-tail:], arsi_vals, color="darkorange", linewidth=1.2, label="Adaptive RSI")
        ax.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax.axhline(50, color="gray", linestyle="-", alpha=0.3)
        ax.fill_between(arsi.index[-tail:], arsi_vals, 70, where=arsi_vals > 70, alpha=0.15, color="red")
        ax.fill_between(arsi.index[-tail:], arsi_vals, 30, where=arsi_vals < 30, alpha=0.15, color="green")
        ax.set_title(f"{sym} — Adaptive RSI (self-tuning period)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(sine_s.index[-tail:], sine_s.values[-tail:], color="blue", label="Sine")
        ax.plot(lead_s.index[-tail:], lead_s.values[-tail:], color="red", linestyle="--", label="Lead Sine")
        ax.axhline(0, color="gray", alpha=0.3)
        ax.set_title(f"{sym} — Sinewave Indicator (leading)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[3]
        ax.plot(dc.index[-tail:], dc.values[-tail:], color="purple", linewidth=1.2, label="Dominant Cycle")
        ax.set_title(f"{sym} — Measured Dominant Cycle Period")
        ax.set_ylabel("Bars")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()

    print("\n✓ Chapter 3 complete.")
