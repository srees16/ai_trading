"""
Chapter 2: Leverage Space Portfolio
=====================================
Ralph Vince — The Leverage Space Trading Model

Demonstrates:
  1. Multi-asset optimal f allocation
  2. Per-symbol safe f with drawdown constraint
  3. Portfolio-level TWR and geometric mean
  4. Comparison of optimal f vs safe f vs equal-weight
  5. Equity curve simulations

Reference: Vince Ch. 4 — "Multiple, simultaneous f values define the
leverage space. The optimal portfolio maximizes TWR across the entire
leverage space."
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

from services.vince_leverage_space import (
    compute_optimal_f,
    compute_leverage_space_portfolio,
)


if __name__ == "__main__":
    print("=" * 70)
    print("Vince Ch.2: Leverage Space Portfolio")
    print("=" * 70)

    prices = get_prices()

    # Collect daily returns for all symbols
    all_returns = {}
    for sym in SYMBOLS:
        if sym not in prices:
            continue
        close = prices[sym]["Close"]
        ret = close.pct_change().dropna().values
        if len(ret) > 50:
            all_returns[sym] = ret

    if len(all_returns) < 2:
        print("  Need at least 2 symbols with data — skipping")
    else:
        # Individual optimal f
        print(f"\n  {'Symbol':<10} {'Opt f':>7} {'Safe f':>7} {'Half f':>7} {'TWR':>10} {'MaxDD%':>7}")
        print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*10} {'─'*7}")

        for sym, ret in all_returns.items():
            r = compute_optimal_f(ret, max_dd_target=0.15)
            print(f"  {sym:<10} {r.optimal_f:>7.2f} {r.safe_f:>7.2f} {r.half_f:>7.2f} {r.twr:>10.2f} {r.max_dd_at_optimal*100:>6.1f}%")

        # Portfolio optimisation
        for dd_target in [0.10, 0.15, 0.25]:
            result = compute_leverage_space_portfolio(
                all_returns, max_dd_target=dd_target, max_total_f=1.0
            )

            print(f"\n{'═' * 50}")
            print(f"  Portfolio (max DD = {dd_target*100:.0f}%)")
            print(f"{'═' * 50}")
            print(f"  Portfolio TWR:          {result.portfolio_twr:.4f}")
            print(f"  Portfolio geomean:      {result.portfolio_geometric_mean*100:.4f}% per period")
            print(f"  Portfolio max DD:       {result.portfolio_max_dd*100:.1f}%")
            print(f"  Portfolio Sharpe:       {result.portfolio_sharpe:.3f}")
            print(f"  MC profit probability:  {result.monte_carlo_profit_prob*100:.1f}%")

            print(f"\n  {'Symbol':<10} {'Optimal f':>10} {'Safe f':>10}")
            print(f"  {'─'*10} {'─'*10} {'─'*10}")
            for sym in sorted(result.optimal_fs):
                opt = result.optimal_fs[sym]
                safe = result.safe_fs.get(sym, 0)
                print(f"  {sym:<10} {opt:>10.3f} {safe:>10.3f}")

        # Plot: compare equity curves at different f levels
        # Use safe_f from 15% DD target
        result_15 = compute_leverage_space_portfolio(all_returns, max_dd_target=0.15)

        # Simulate portfolio equity curves
        min_len = min(len(v) for v in all_returns.values())
        syms = list(all_returns.keys())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Individual safe-f equity curves
        ax = axes[0]
        for sym in syms:
            ret = all_returns[sym][:min_len]
            safe_f = result_15.safe_fs.get(sym, 0.05)
            worst = abs(float(np.min(ret))) or 0.01
            equity = [1.0]
            for r in ret:
                hpr = 1.0 + safe_f * r / worst
                equity.append(equity[-1] * max(hpr, 0.01))
            ax.plot(equity, label=f"{sym} (f={safe_f:.2f})", alpha=0.8)
        ax.set_title("Individual Equity Curves (safe f, DD≤15%)")
        ax.set_xlabel("Trading days")
        ax.set_ylabel("Equity multiple")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # f allocation pie chart
        ax = axes[1]
        safe_fs = result_15.safe_fs
        labels = list(safe_fs.keys())
        sizes = [safe_fs[s] for s in labels]
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Safe f Allocation (DD≤15%, total f={sum(sizes):.2f})")
        else:
            ax.text(0.5, 0.5, "No allocation", ha="center", va="center")
            ax.set_title("Allocation (empty)")

        plt.tight_layout()

    print("\n✓ Chapter 2 complete.")
