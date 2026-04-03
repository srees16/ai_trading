"""
Chapter 4: Data-Mining Bias Estimation
=======================================
Evidence-Based Technical Analysis — David Aronson

Demonstrates:
  1. The data-mining bias formula: E[best] ≈ σ√(2·ln N)
  2. Empirical verification via Monte Carlo simulation
  3. Correcting observed Sharpe ratios for bias
  4. Visualising bias growth with number of strategies tested

Reference: Aronson Ch. 9 — When searching over N strategies, the
expected Sharpe ratio of the best strategy under pure noise is
σ√(2·ln N). This must be subtracted from observed performance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import get_prices, generate_returns, SYMBOLS

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.aronson_validator import estimate_data_mining_bias, trimmed_sharpe


if __name__ == "__main__":
    print("=" * 70)
    print("Aronson Ch.4: Data-Mining Bias Estimation")
    print("=" * 70)

    # ── Part 1: Theoretical bias curve ──────────────────────────
    print("\n[1] Theoretical bias: E[best Sharpe] ≈ σ√(2·ln N)")
    n_range = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
    bias_values = [estimate_data_mining_bias(n, sigma_best=1.0) for n in n_range]

    print(f"\n  {'N tested':>10} {'Expected bias (Sharpe units)':>30}")
    print(f"  {'─'*10} {'─'*30}")
    for n, b in zip(n_range, bias_values):
        print(f"  {n:>10d} {b:>30.4f}")

    # Plot bias curve
    fig, ax = plt.subplots(figsize=(8, 5))
    n_fine = np.logspace(0.5, 4, 200)
    bias_fine = [np.sqrt(2 * np.log(n)) for n in n_fine]
    ax.plot(n_fine, bias_fine, color="steelblue", linewidth=2, label="σ√(2·ln N)")
    ax.scatter(n_range, bias_values, color="red", s=50, zorder=5, label="Selected N")
    ax.set_xscale("log")
    ax.set_xlabel("Number of strategies tested (N)")
    ax.set_ylabel("Expected bias (Sharpe units)")
    ax.set_title("Data-Mining Bias: Expected Best Sharpe Under Null")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    # ── Part 2: Monte Carlo empirical verification ──────────────
    print("\n[2] Monte Carlo verification (10,000 sims per N)")
    rng = np.random.default_rng(42)

    mc_ns = [10, 50, 100, 500]
    mc_results = {}
    n_sims = 10000
    n_obs = 252  # 1 year of daily returns

    for n_strats in mc_ns:
        best_sharpes = []
        for _ in range(n_sims):
            # Generate N random equity curves (null: no alpha)
            sharpes = []
            for _ in range(n_strats):
                rets = rng.normal(0, 0.01, n_obs)  # mean=0, std=1%
                sr = np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252) if np.std(rets) > 0 else 0
                sharpes.append(sr)
            best_sharpes.append(max(sharpes))

        mc_results[n_strats] = best_sharpes
        empirical_mean = np.mean(best_sharpes)
        theoretical = estimate_data_mining_bias(n_strats, sigma_best=1.0)
        print(f"  N={n_strats:>4d}:  Empirical best SR = {empirical_mean:.3f}  |  Theory = {theoretical:.3f}")

    # Plot MC distributions
    fig, axes = plt.subplots(1, len(mc_ns), figsize=(4 * len(mc_ns), 4), sharey=True)
    for i, n_strats in enumerate(mc_ns):
        ax = axes[i]
        ax.hist(mc_results[n_strats], bins=40, alpha=0.7, color="steelblue", edgecolor="white", density=True)
        th = estimate_data_mining_bias(n_strats, sigma_best=1.0)
        ax.axvline(th, color="red", linestyle="--", linewidth=2, label=f"Theory={th:.2f}")
        ax.axvline(np.mean(mc_results[n_strats]), color="green", linestyle="-", linewidth=2, label=f"MC mean={np.mean(mc_results[n_strats]):.2f}")
        ax.set_title(f"N={n_strats}", fontsize=10)
        ax.set_xlabel("Best Sharpe Ratio")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Density")
    plt.suptitle("Monte Carlo: Best SR Distribution Under Null", fontsize=12)
    plt.tight_layout()

    # ── Part 3: Apply correction to real data ───────────────────
    print("\n[3] Bias-corrected Sharpe ratios on real symbols")
    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        close = prices[sym]["Close"]
        ret = close.pct_change().dropna()

        raw_sr = float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(252)) if len(ret) > 30 else 0.0
        trimmed_sr = trimmed_sharpe(ret.values, trim_pct=0.05)

        # Assume we scanned 50 parameter combos to find this signal
        n_tested = 50
        bias = estimate_data_mining_bias(n_tested, sigma_best=1.0)
        corrected_sr = max(0, raw_sr - bias)

        print(f"\n  {sym}:")
        print(f"    Raw Sharpe:       {raw_sr:.3f}")
        print(f"    Trimmed Sharpe:   {trimmed_sr:.3f}")
        print(f"    Bias (N={n_tested}):    {bias:.3f}")
        print(f"    Corrected Sharpe: {corrected_sr:.3f}")

    print("\n✓ Chapter 4 complete.")
