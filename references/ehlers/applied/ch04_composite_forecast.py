"""
Chapter 4: Composite Ehlers Forecast
======================================
Ehlers — Cybernetic Analysis / Rocket Science for Traders

Demonstrates:
  1. Computing all 10 Ehlers DSP indicators simultaneously
  2. Combining into a Carver-compatible composite forecast (±20)
  3. Signal-to-Noise Ratio filter (only trade when SNR > threshold)
  4. Relative Vigor Index (RVI) confirmation
  5. Full EhlersAnalysis dataclass output

Reference: All Ehlers indicators combined per compute_ehlers_analysis()
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
    compute_ehlers_analysis,
    compute_ehlers_forecast_batch,
    signal_to_noise_ratio,
    relative_vigor_index,
)


if __name__ == "__main__":
    print("=" * 70)
    print("Ehlers Ch.4: Composite Ehlers Forecast")
    print("=" * 70)

    prices = get_prices()

    forecast_data = {}

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]

        # Full analysis
        analysis = compute_ehlers_analysis(df)
        if not analysis:
            print(f"  {sym}: insufficient data for analysis")
            continue

        print(f"\n{'─' * 50}")
        print(f"  {sym}: Ehlers DSP Analysis")
        print(f"  {'─'*40}")
        print(f"  Super Smoother:      {analysis.super_smoother:.2f}")
        print(f"  Fisher Transform:    {analysis.fisher_transform:.3f} (trigger: {analysis.fisher_trigger:.3f})")
        print(f"  Instantaneous Trend: {analysis.instantaneous_trendline:.2f}")
        print(f"  Cyber Cycle:         {analysis.cyber_cycle:.4f}")
        print(f"  MAMA / FAMA:         {analysis.mama:.2f} / {analysis.fama:.2f}")
        print(f"  Sinewave:            {analysis.sinewave:.3f} (lead: {analysis.leadsine:.3f})")
        print(f"  RVI (signal):        {analysis.rvi:.4f} ({analysis.rvi_signal:.4f})")
        print(f"  SNR:                 {analysis.snr:.1f} dB")
        print(f"  Adaptive RSI:        {analysis.adaptive_rsi:.1f}")
        print(f"  Dominant Cycle:      {analysis.dominant_cycle:.1f} bars")
        print(f"  COMPOSITE FORECAST:  {analysis.composite_forecast:+.1f}")

        forecast_data[sym] = analysis.composite_forecast

        # Compute rolling SNR and RVI for plots
        close = df["Close"]
        snr = signal_to_noise_ratio(close, period=10)
        rvi_s, rvi_sig = relative_vigor_index(df, period=10)

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1, 1]})
        tail = min(400, len(close))

        ax = axes[0]
        ax.plot(close.index[-tail:], close.values[-tail:], color="gray", alpha=0.5, label="Close")
        ax.set_title(f"{sym} — Price (Composite Forecast: {analysis.composite_forecast:+.1f})")
        direction = "BULLISH ▲" if analysis.composite_forecast > 5 else "BEARISH ▼" if analysis.composite_forecast < -5 else "NEUTRAL ●"
        color = "green" if analysis.composite_forecast > 5 else "red" if analysis.composite_forecast < -5 else "gray"
        ax.text(0.98, 0.95, direction, transform=ax.transAxes, fontsize=14,
                verticalalignment="top", horizontalalignment="right",
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(snr.index[-tail:], snr.values[-tail:], color="blue", linewidth=1.2, label="SNR (dB)")
        ax.axhline(6, color="green", linestyle="--", alpha=0.5, label="Trade threshold (6 dB)")
        ax.set_title(f"{sym} — Signal-to-Noise Ratio")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        if rvi_s is not None:
            ax.plot(rvi_s.index[-tail:], rvi_s.values[-tail:], color="darkorange", label="RVI")
            ax.plot(rvi_sig.index[-tail:], rvi_sig.values[-tail:], color="purple", linestyle="--", label="RVI Signal")
            ax.axhline(0, color="gray", alpha=0.3)
        ax.set_title(f"{sym} — Relative Vigor Index")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()

    # Summary heatmap
    if forecast_data:
        print(f"\n{'═' * 50}")
        print(f"  Forecast Summary")
        print(f"{'═' * 50}")
        for sym, fc in sorted(forecast_data.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * max(1, int(abs(fc)))
            direction = "▲" if fc > 0 else "▼" if fc < 0 else "●"
            print(f"  {sym:<10} {fc:>+6.1f} {direction} {bar}")

    print("\n✓ Chapter 4 complete.")
