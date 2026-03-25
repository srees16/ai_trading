"""
sample_data.py – Shared data-generation module for TTMTS chapter scripts.

Based on: "Testing and Tuning Market Trading Systems" by Timothy Masters.

Uses yfinance to download real market data and provides synthetic generators
for cases where real data is not appropriate (e.g., Ornstein-Uhlenbeck
processes, random trading systems).

Delegates to ``shared.sample_data_base`` for implementation — this thin
wrapper only defines the TTS-specific configuration (env var prefix,
default symbols, cache directory).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.sample_data_base import (  # noqa: E402
    get_prices as _get_prices,
    get_close_series as _get_close_series,
    get_multi_close as _get_multi_close,
    generate_ohlcv_bars as _generate_ohlcv_bars,
    generate_returns as _generate_returns,
    generate_ou_process as _generate_ou_process,
)

# ---------------------------------------------------------------------------
# Configuration — env vars override defaults when set by the runner
# ---------------------------------------------------------------------------
_env_tickers = os.environ.get("TTS_TICKERS")
SYMBOLS = [t.strip() for t in _env_tickers.split(",") if t.strip()] if _env_tickers else ["SPY", "QQQ", "IWM", "DIA"]
DEFAULT_START = os.environ.get("TTS_DATE_START", "2020-01-01")
DEFAULT_END = os.environ.get("TTS_DATE_END", "2024-12-31")
CACHE_DIR = Path(__file__).parent / "_cache"

# ---------------------------------------------------------------------------
# Thin wrappers that bind TTS configuration
# ---------------------------------------------------------------------------

def get_prices(symbols=None, start=None, end=None, interval="1d"):
    return _get_prices(symbols or SYMBOLS, DEFAULT_START, DEFAULT_END, CACHE_DIR,
                       start=start, end=end, interval=interval)

def get_close_series(symbol=None, start=None, end=None):
    return _get_close_series(SYMBOLS, DEFAULT_START, DEFAULT_END, CACHE_DIR,
                             symbol=symbol, start=start, end=end)

def get_multi_close(symbols=None, start=None, end=None):
    return _get_multi_close(SYMBOLS, DEFAULT_START, DEFAULT_END, CACHE_DIR,
                            target_symbols=symbols, start=start, end=end)

def generate_ohlcv_bars(n_bars=2000, seed=42):
    return _generate_ohlcv_bars(DEFAULT_START, n_bars=n_bars, seed=seed)

def generate_returns(n=2000, n_assets=1, seed=42):
    """Generate synthetic daily returns. Returns Series for n_assets=1."""
    df = _generate_returns(DEFAULT_START, n=n, n_assets=max(n_assets, 2), seed=seed)
    if n_assets == 1:
        return pd.Series(df.iloc[:, 0].values, index=df.index[:n], name="returns")
    return df

def generate_ou_process(n=2000, theta=0.1, mu=100.0, sigma=2.0, seed=42):
    return _generate_ou_process(DEFAULT_START, n=n, theta=theta, mu=mu,
                                sigma=sigma, seed=seed)


def generate_random_trading_system(n_trades=500, win_rate=0.55,
                                   avg_win=1.0, avg_loss=-0.8, seed=42):
    """Generate synthetic trade returns for a random trading system.

    Parameters
    ----------
    n_trades : int   – number of trades
    win_rate : float – probability of a winning trade
    avg_win  : float – average return of a winning trade (%)
    avg_loss : float – average return of a losing trade (%)
    seed     : int   – random seed

    Returns
    -------
    trades : pd.Series of per-trade returns
    """
    rng = np.random.default_rng(seed)
    wins = rng.random(n_trades) < win_rate
    returns = np.where(
        wins,
        rng.exponential(avg_win, n_trades),
        -rng.exponential(abs(avg_loss), n_trades),
    )
    return pd.Series(returns, name="trade_return")


def generate_indicator_series(n=2000, n_indicators=3, seed=42):
    """Generate synthetic indicator time series with varying entropy.

    Returns a DataFrame with columns ['Ind_0','Ind_1',...] where
    Ind_0 has nearly uniform distribution (high entropy),
    Ind_1 has moderate entropy,
    and Ind_2+ have increasingly skewed distributions (low entropy).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    data = {}
    for i in range(n_indicators):
        if i == 0:
            # Uniform distribution (high entropy)
            data[f"Ind_{i}"] = rng.uniform(-1, 1, n)
        elif i == 1:
            # Normal (moderate entropy)
            data[f"Ind_{i}"] = rng.normal(0, 1, n)
        else:
            # Increasingly skewed (low entropy via outliers)
            base = rng.normal(0, 0.3, n)
            n_outliers = max(1, n // (50 * i))
            idx = rng.choice(n, n_outliers, replace=False)
            base[idx] = rng.normal(0, 5 * i, n_outliers)
            data[f"Ind_{i}"] = base
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("sample_data.py – self-test (TTMTS)")
    print("=" * 60)

    print("\n[1] Downloading real prices ...")
    prices = get_prices(["SPY"], start="2023-01-01", end="2024-01-01")
    for sym, df in prices.items():
        print(f"  {sym}: {len(df)} bars, columns={list(df.columns)}")

    print("\n[2] Synthetic OHLCV bars ...")
    bars = generate_ohlcv_bars(500)
    print(f"  shape={bars.shape}")

    print("\n[3] O-U process ...")
    ou = generate_ou_process(500)
    print(f"  shape={ou.shape}, mean={ou.mean():.2f}")

    print("\n[4] Random trading system ...")
    trades = generate_random_trading_system(200)
    print(f"  n={len(trades)}, mean={trades.mean():.4f}, win%={( trades > 0).mean():.2%}")

    print("\n[5] Indicator series ...")
    ind = generate_indicator_series(500, 3)
    print(f"  shape={ind.shape}")
