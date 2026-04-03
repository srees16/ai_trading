"""
Chapter 1: Detrended Returns & t-Stat Gating
=============================================
Evidence-Based Technical Analysis — David Aronson

Demonstrates:
  1. Detrending returns (zero-centering by rolling mean)
  2. Building simple technical signals (SMA crossover, RSI threshold, Momentum)
  3. Evaluating each signal via t-statistic gate (H0: mean excess return = 0)
  4. Visualising raw vs detrended return distributions

Reference: Aronson Ch. 1 & 5 — signal returns must be detrended to
remove confounding secular drift before statistical testing.
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

# ── Import core service functions ────────────────────────────────────────
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.aronson_validator import detrend_returns, compute_signal_tstat


def _sma_crossover_signal(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Return +1 when fast SMA > slow SMA, else -1."""
    sma_f = close.rolling(fast).mean()
    sma_s = close.rolling(slow).mean()
    sig = pd.Series(np.where(sma_f > sma_s, 1, -1), index=close.index)
    return sig.shift(1).dropna()


def _rsi_threshold_signal(close: pd.Series, period: int = 14, buy_threshold: float = 30, sell_threshold: float = 70) -> pd.Series:
    """Return +1 when RSI < buy_threshold (oversold), -1 when RSI > sell_threshold, else 0."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    sig = pd.Series(0.0, index=close.index)
    sig[rsi < buy_threshold] = 1.0
    sig[rsi > sell_threshold] = -1.0
    return sig.shift(1).dropna()


def _momentum_signal(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Return +1 when N-day return > 0, else -1."""
    mom = close.pct_change(lookback)
    sig = pd.Series(np.where(mom > 0, 1, -1), index=close.index)
    return sig.shift(1).dropna()


if __name__ == "__main__":
    print("=" * 70)
    print("Aronson Ch.1: Detrended Returns & t-Stat Gating")
    print("=" * 70)

    prices = get_prices()

    for sym in SYMBOLS:
        if sym not in prices:
            continue
        df = prices[sym]
        close = df["Close"]
        raw_ret = close.pct_change().dropna()
        detrended = detrend_returns(raw_ret)

        print(f"\n{'─' * 50}")
        print(f"  Symbol: {sym}  |  {len(raw_ret)} observations")
        print(f"  Raw mean:      {raw_ret.mean()*100:.4f}% daily")
        print(f"  Detrended mean: {detrended.mean()*100:.6f}% (should be ≈ 0)")

        # Build 3 simple signals
        signals = {
            "SMA(20/50)": _sma_crossover_signal(close),
            "RSI(14)": _rsi_threshold_signal(close),
            "Momentum(20)": _momentum_signal(close),
        }

        print(f"\n  {'Signal':<20} {'t-stat':>8} {'p-value':>10} {'Fires':>7} {'Pass?':>6}")
        print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*7} {'─'*6}")

        for name, sig in signals.items():
            # Align signal with returns
            common_idx = sig.index.intersection(raw_ret.index)
            sig_returns = detrended.reindex(common_idx) * sig.reindex(common_idx)
            sig_returns = sig_returns.dropna()

            t, p = compute_signal_tstat(sig_returns.values)
            n_fires = int((sig.reindex(common_idx).abs() > 0).sum())
            passed = "YES" if t >= 2.0 else "no"
            print(f"  {name:<20} {t:>8.3f} {p:>10.6f} {n_fires:>7d} {passed:>6}")

        # Plot: Raw vs Detrended return distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(raw_ret.values, bins=60, alpha=0.7, color="steelblue", edgecolor="white")
        axes[0].axvline(raw_ret.mean(), color="red", linestyle="--", label=f"mean={raw_ret.mean()*100:.3f}%")
        axes[0].set_title(f"{sym} — Raw Returns")
        axes[0].legend(fontsize=8)

        axes[1].hist(detrended.values, bins=60, alpha=0.7, color="darkorange", edgecolor="white")
        axes[1].axvline(detrended.mean(), color="red", linestyle="--", label=f"mean={detrended.mean()*100:.5f}%")
        axes[1].set_title(f"{sym} — Detrended Returns")
        axes[1].legend(fontsize=8)
        plt.tight_layout()

    print("\n✓ Chapter 1 complete.")
