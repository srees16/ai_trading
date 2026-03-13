"""
Chapter 1: Introduction
========================
Testing and Tuning Market Trading Systems — Timothy Masters

Overview of trading system concepts: market returns, two types of automated
trading systems (indicator-based vs ML-based), future leak dangers, and the
percent-wins fallacy.  This chapter is primarily theoretical — the code here
provides utility functions and illustrative calculations referenced later.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_data import (
    get_close_series,
    generate_returns,
    generate_random_trading_system,
)

# ======================================================================
# Market Returns — log vs simple returns
# ======================================================================

def compute_log_returns(prices):
    """Compute log returns from a price series.

    Parameters
    ----------
    prices : pd.Series – price series

    Returns
    -------
    pd.Series – log returns
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices):
    """Compute simple (arithmetic) returns from a price series."""
    return prices.pct_change().dropna()


# ======================================================================
# Future Leak Detection — autocorrelation of returns
# ======================================================================

def detect_future_leak(returns, max_lags=10):
    """Check for suspicious autocorrelation that may indicate future leak.

    If returns have unusually high positive autocorrelation at lag 1, the
    system may be peeking into the future.

    Parameters
    ----------
    returns  : pd.Series – return series
    max_lags : int       – number of lags to test

    Returns
    -------
    pd.DataFrame with columns ['lag', 'autocorrelation', 'suspicious']
    """
    results = []
    n = len(returns)
    threshold = 2.0 / np.sqrt(n)  # approximate 95% CI for white noise
    for lag in range(1, max_lags + 1):
        ac = returns.autocorr(lag=lag)
        results.append({
            "lag": lag,
            "autocorrelation": ac,
            "threshold_95pct": threshold,
            "suspicious": abs(ac) > threshold,
        })
    return pd.DataFrame(results)


# ======================================================================
# Percent Wins Fallacy — illustrate why win rate alone is misleading
# ======================================================================

def percent_wins_analysis(trade_returns):
    """Demonstrate the percent-wins fallacy.

    A system can have a high win rate but still lose money if the average
    loss is much larger than the average win. Conversely, a low win rate
    can still be profitable with large winners.

    Parameters
    ----------
    trade_returns : pd.Series – per-trade returns

    Returns
    -------
    dict with win_rate, avg_win, avg_loss, profit_factor, expectancy, etc.
    """
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]

    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    # Expectancy: average return per trade
    expectancy = trade_returns.mean()

    # Kelly fraction: f* = (p * b - q) / b
    # where p = win_rate, q = 1 - p, b = avg_win / abs(avg_loss)
    if abs(avg_loss) > 0 and avg_win > 0:
        b = avg_win / abs(avg_loss)
        kelly = (win_rate * b - (1 - win_rate)) / b
    else:
        kelly = 0.0

    return {
        "n_trades": len(trade_returns),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "total_return": trade_returns.sum(),
        "kelly_fraction": kelly,
    }


# ======================================================================
# Two Types of Trading Systems
# ======================================================================

def indicator_based_system(prices, short_window=10, long_window=50):
    """Simple moving average crossover — indicator-based system example.

    Parameters
    ----------
    prices       : pd.Series  – price series
    short_window : int        – short MA window
    long_window  : int        – long MA window

    Returns
    -------
    signals : pd.DataFrame with columns ['price','short_ma','long_ma','signal','returns']
    """
    df = pd.DataFrame({"price": prices})
    df["short_ma"] = prices.rolling(short_window).mean()
    df["long_ma"] = prices.rolling(long_window).mean()
    df["signal"] = 0
    df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1
    df.loc[df["short_ma"] <= df["long_ma"], "signal"] = -1
    df["returns"] = df["signal"].shift(1) * compute_simple_returns(prices)
    return df.dropna()


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("Chapter 1 – Introduction")
    print("  (Testing and Tuning Market Trading Systems)")
    print("=" * 70)

    # 1. Market returns
    print("\n--- Market Returns ---")
    prices = get_close_series("SPY")
    log_ret = compute_log_returns(prices)
    simple_ret = compute_simple_returns(prices)
    print(f"  SPY: {len(prices)} bars")
    print(f"  Log returns:    mean={log_ret.mean():.6f}, std={log_ret.std():.6f}")
    print(f"  Simple returns: mean={simple_ret.mean():.6f}, std={simple_ret.std():.6f}")

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    axes1[0].hist(log_ret, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    axes1[0].set_title("Log Returns Distribution")
    axes1[0].set_xlabel("Return")
    axes1[1].plot(prices.index, prices.values, color="steelblue", linewidth=0.8)
    axes1[1].set_title("SPY Price History")
    axes1[1].set_ylabel("Price ($)")
    fig1.tight_layout()

    # 2. Future leak detection
    print("\n--- Future Leak Detection ---")
    ac_df = detect_future_leak(simple_ret)
    for _, row in ac_df.iterrows():
        flag = " *** SUSPICIOUS" if row["suspicious"] else ""
        print(f"  Lag {int(row['lag']):2d}: autocorr={row['autocorrelation']:+.4f}{flag}")

    # 3. Percent wins fallacy — two contrasting systems
    print("\n--- Percent Wins Fallacy ---")
    sys_a = generate_random_trading_system(500, win_rate=0.80,
                                           avg_win=0.3, avg_loss=-1.2, seed=1)
    sys_b = generate_random_trading_system(500, win_rate=0.35,
                                           avg_win=3.0, avg_loss=-0.5, seed=2)

    analysis_a = percent_wins_analysis(sys_a)
    analysis_b = percent_wins_analysis(sys_b)

    print(f"\n  System A (high win rate):  WR={analysis_a['win_rate']:.1%}  "
          f"PF={analysis_a['profit_factor']:.2f}  "
          f"Total={analysis_a['total_return']:+.2f}")
    print(f"  System B (low win rate):   WR={analysis_b['win_rate']:.1%}  "
          f"PF={analysis_b['profit_factor']:.2f}  "
          f"Total={analysis_b['total_return']:+.2f}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    cum_a = sys_a.cumsum()
    cum_b = sys_b.cumsum()
    axes2[0].plot(cum_a.values, label=f"WR={analysis_a['win_rate']:.0%}", color="crimson")
    axes2[0].set_title("System A (High Win Rate)")
    axes2[0].set_ylabel("Cumulative Return")
    axes2[0].legend()
    axes2[1].plot(cum_b.values, label=f"WR={analysis_b['win_rate']:.0%}", color="seagreen")
    axes2[1].set_title("System B (Low Win Rate)")
    axes2[1].set_ylabel("Cumulative Return")
    axes2[1].legend()
    fig2.suptitle("Percent Wins Fallacy", fontsize=13)
    fig2.tight_layout()

    # 4. Indicator-based system
    print("\n--- Indicator-Based System (SMA Crossover) ---")
    sma_df = indicator_based_system(prices, short_window=20, long_window=50)
    total_ret = sma_df["returns"].sum()
    sr = sma_df["returns"].mean() / (sma_df["returns"].std() + 1e-10) * np.sqrt(252)
    print(f"  Total return:   {total_ret:.4f}")
    print(f"  Annualised SR:  {sr:.4f}")

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(sma_df.index, sma_df["price"], label="Price", alpha=0.5, linewidth=0.8)
    ax3.plot(sma_df.index, sma_df["short_ma"], label="SMA(20)", linewidth=1)
    ax3.plot(sma_df.index, sma_df["long_ma"], label="SMA(50)", linewidth=1)
    ax3.set_title("SMA Crossover System")
    ax3.legend()
    fig3.tight_layout()

    print("\nChapter 1 complete.")


if __name__ == "__main__":
    main()
