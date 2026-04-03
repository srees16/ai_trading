"""
Chapter 1: Optimal f & Terminal Wealth Relative
=================================================
Ralph Vince — The Leverage Space Trading Model

Demonstrates:
  1. Computing optimal f for each ticker's return series
  2. TWR curve as function of f (the leverage space)
  3. Geometric mean maximisation
  4. Safe f (drawdown-constrained)
  5. Half-Kelly conservative sizing

Reference: Vince Ch. 1-3 — "Geometric mean maximization is the single
criterion for maximizing long-term growth."
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

from services.vince_leverage_space import compute_optimal_f, _compute_twr


if __name__ == "__main__":
    print("=" * 70)
    print("Vince Ch.1: Optimal f & Terminal Wealth Relative")
    print("=" * 70)

    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]
        daily_ret = close.pct_change().dropna().values

        # Compute optimal f
        result = compute_optimal_f(daily_ret, max_dd_target=0.15)
        result.symbol = sym

        print(f"\n{'─' * 50}")
        print(f"  {sym}: {len(daily_ret)} daily returns")
        print(f"  Worst single-day loss: {np.min(daily_ret)*100:.2f}%")
        print(f"  {'─'*40}")
        print(f"  Optimal f:         {result.optimal_f:.2f}")
        print(f"  Half-Kelly f:      {result.half_f:.2f}")
        print(f"  Safe f (DD≤15%):   {result.safe_f:.2f}")
        print(f"  TWR at optimal f:  {result.twr:.4f}")
        print(f"  Geometric mean:    {result.geometric_mean*100:.4f}% per trade")
        print(f"  Max DD at opt f:   {result.max_dd_at_optimal*100:.1f}%")

        # Build TWR curve across f values
        worst_loss = abs(float(np.min(daily_ret)))
        if worst_loss == 0:
            worst_loss = 0.01

        f_range = np.arange(0.01, 1.0, 0.01)
        twr_curve = []
        gmean_curve = []
        dd_curve = []
        n = len(daily_ret)

        for f in f_range:
            twr = _compute_twr(daily_ret, f, worst_loss)
            twr_curve.append(twr)
            gmean = (twr ** (1.0 / n) - 1.0) if twr > 0 and n > 0 else -1.0
            gmean_curve.append(gmean)

            # Drawdown at this f
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            for r in daily_ret:
                hpr = 1.0 + f * r / worst_loss
                if hpr <= 0:
                    max_dd = 1.0
                    break
                equity *= hpr
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
            dd_curve.append(max_dd)

        # Plot: TWR / Geometric Mean / Drawdown vs f
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.plot(f_range, twr_curve, color="steelblue", linewidth=2)
        ax.axvline(result.optimal_f, color="red", linestyle="--", label=f"Optimal f={result.optimal_f:.2f}")
        ax.axvline(result.safe_f, color="green", linestyle="--", label=f"Safe f={result.safe_f:.2f}")
        ax.axvline(result.half_f, color="orange", linestyle=":", label=f"Half-f={result.half_f:.2f}")
        ax.set_xlabel("f (fraction risked)")
        ax.set_ylabel("TWR")
        ax.set_title(f"{sym} — Terminal Wealth Relative")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

        ax = axes[1]
        ax.plot(f_range, [g * 100 for g in gmean_curve], color="darkorange", linewidth=2)
        ax.axvline(result.optimal_f, color="red", linestyle="--")
        ax.axvline(result.safe_f, color="green", linestyle="--")
        ax.set_xlabel("f (fraction risked)")
        ax.set_ylabel("Geometric Mean (%)")
        ax.set_title(f"{sym} — Growth Rate")
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(f_range, [d * 100 for d in dd_curve], color="red", linewidth=2)
        ax.axhline(15, color="green", linestyle="--", label="Max DD target (15%)")
        ax.axvline(result.safe_f, color="green", linestyle="--", label=f"Safe f={result.safe_f:.2f}")
        ax.set_xlabel("f (fraction risked)")
        ax.set_ylabel("Max Drawdown (%)")
        ax.set_title(f"{sym} — Drawdown vs Leverage")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        plt.suptitle(f"{sym} — Vince Leverage Space Analysis", fontsize=13)
        plt.tight_layout()

    print("\n✓ Chapter 1 complete.")
