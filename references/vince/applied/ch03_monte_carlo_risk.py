"""
Chapter 3: Monte Carlo Risk Simulation
========================================
Ralph Vince — The Leverage Space Trading Model / Math of Money Management

Demonstrates:
  1. Monte Carlo simulation of portfolio equity paths
  2. Probability of profit at various horizons
  3. CAGR distribution from simulated paths
  4. Max drawdown distribution
  5. Risk of ruin estimation

Reference: Vince Ch. 5-6 — "Risk metrics in leverage space must include
drawdown constraints and probability of achieving target."
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
    monte_carlo_simulation,
)


if __name__ == "__main__":
    print("=" * 70)
    print("Vince Ch.3: Monte Carlo Risk Simulation")
    print("=" * 70)

    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]
        daily_ret = close.pct_change().dropna().values

        if len(daily_ret) < 50:
            print(f"  {sym}: insufficient data — skipping")
            continue

        opt_f = compute_optimal_f(daily_ret, max_dd_target=0.15)

        print(f"\n{'═' * 50}")
        print(f"  {sym}: Monte Carlo Simulation")
        print(f"  Optimal f: {opt_f.optimal_f:.2f}, Safe f: {opt_f.safe_f:.2f}")
        print(f"{'═' * 50}")

        # Run MC at multiple f levels
        f_levels = {
            "safe_f": opt_f.safe_f,
            "half_kelly": opt_f.half_f,
            "quarter_kelly": opt_f.optimal_f / 4,
        }

        mc_results = {}
        for label, f_val in f_levels.items():
            mc = monte_carlo_simulation(
                trade_returns=daily_ret,
                f=f_val,
                n_simulations=5000,
                horizon_days=252 * 3,  # 3-year horizon
                seed=42,
            )
            mc_results[label] = mc

            print(f"\n  [{label}] f={f_val:.3f}")
            print(f"    P(profit) @ 3yr:   {mc.profit_probability*100:.1f}%")
            print(f"    Median return:     {mc.median_return*100:.1f}%")
            print(f"    5th percentile:    {mc.percentile_5*100:.1f}%")
            print(f"    95th percentile:   {mc.percentile_95*100:.1f}%")
            print(f"    Median max DD:     {mc.median_max_dd*100:.1f}%")
            print(f"    Expected CAGR:     {mc.expected_cagr*100:.1f}%")

        # Plot: MC equity paths for safe_f
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Sample equity paths
        ax = axes[0][0]
        rng = np.random.default_rng(42)
        worst = abs(float(np.min(daily_ret))) or 0.01
        f_safe = opt_f.safe_f
        n_paths = 100
        horizon = 252 * 3

        for _ in range(n_paths):
            path_rets = rng.choice(daily_ret, size=horizon, replace=True)
            equity = [1.0]
            for r in path_rets:
                hpr = 1.0 + f_safe * r / worst
                equity.append(equity[-1] * max(hpr, 0.001))
            ax.plot(equity, alpha=0.1, color="steelblue", linewidth=0.5)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"{sym} — MC Equity Paths (f={f_safe:.2f}, N=100)")
        ax.set_xlabel("Days")
        ax.set_ylabel("Equity multiple")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

        # Top-right: Terminal wealth distribution
        ax = axes[0][1]
        terminal_vals = []
        for _ in range(5000):
            path_rets = rng.choice(daily_ret, size=horizon, replace=True)
            equity = 1.0
            for r in path_rets:
                hpr = 1.0 + f_safe * r / worst
                equity *= max(hpr, 0.001)
            terminal_vals.append(equity)

        ax.hist(terminal_vals, bins=80, alpha=0.7, color="steelblue", edgecolor="white", density=True)
        ax.axvline(1.0, color="red", linestyle="--", label="Break-even")
        ax.axvline(np.median(terminal_vals), color="green", linestyle="-", linewidth=2,
                   label=f"Median={np.median(terminal_vals):.2f}x")
        ax.set_title(f"{sym} — Terminal Wealth Distribution (3yr)")
        ax.set_xlabel("Terminal equity multiple")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Bottom-left: Max DD distribution
        ax = axes[1][0]
        dd_vals = []
        for _ in range(5000):
            path_rets = rng.choice(daily_ret, size=horizon, replace=True)
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            for r in path_rets:
                hpr = 1.0 + f_safe * r / worst
                equity *= max(hpr, 0.001)
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
            dd_vals.append(max_dd * 100)

        ax.hist(dd_vals, bins=60, alpha=0.7, color="red", edgecolor="white", density=True)
        ax.axvline(15, color="green", linestyle="--", label="15% target")
        ax.axvline(np.median(dd_vals), color="blue", linestyle="-", linewidth=2,
                   label=f"Median={np.median(dd_vals):.1f}%")
        ax.set_title(f"{sym} — Max Drawdown Distribution (3yr)")
        ax.set_xlabel("Max Drawdown (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Bottom-right: Profit probability vs f
        ax = axes[1][1]
        f_range = np.arange(0.01, 0.50, 0.01)
        pp_by_f = []
        for f in f_range:
            profits = 0
            for _ in range(1000):
                path_rets = rng.choice(daily_ret, size=horizon, replace=True)
                equity = 1.0
                for r in path_rets:
                    hpr = 1.0 + f * r / worst
                    equity *= max(hpr, 0.001)
                if equity > 1.0:
                    profits += 1
            pp_by_f.append(profits / 1000 * 100)

        ax.plot(f_range * 100, pp_by_f, color="steelblue", linewidth=2)
        ax.axvline(opt_f.safe_f * 100, color="green", linestyle="--", label=f"Safe f={opt_f.safe_f:.2f}")
        ax.axvline(opt_f.half_f * 100, color="orange", linestyle="--", label=f"Half f={opt_f.half_f:.2f}")
        ax.set_xlabel("f (%)")
        ax.set_ylabel("P(profit) at 3yr (%)")
        ax.set_title(f"{sym} — Probability of Profit vs Leverage")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.suptitle(f"{sym} — Vince Monte Carlo Risk Analysis", fontsize=13)
        plt.tight_layout()

    print("\n✓ Chapter 3 complete.")
