"""
sample_data_base.py – Shared data helpers for chapter-based book modules.

Provides yfinance download/cache, synthetic data generators, and
configuration-driven symbols/dates. Both ``financial_ML/sample_data``
and ``testune_trade_sys/sample_data`` delegate here to avoid duplication.
"""

import numpy as np
import os
import pandas as pd
import yfinance as yf
from pathlib import Path


# ---------------------------------------------------------------------------
# Real market data helpers
# ---------------------------------------------------------------------------

def get_prices(symbols, default_start, default_end, cache_dir,
               start=None, end=None, interval="1d"):
    """Download daily OHLCV data via yfinance and cache locally.

    Returns a dict  {symbol: DataFrame} with columns
    ['Open','High','Low','Close','Volume'].
    """
    start = start or default_start
    end = end or default_end
    cache_dir.mkdir(exist_ok=True)

    result = {}
    for sym in symbols:
        cache_file = cache_dir / f"{sym}_{start}_{end}_{interval}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            df = yf.download(sym, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"[sample_data] WARNING: no data for {sym}")
                continue
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_parquet(cache_file)
        result[sym] = df
    return result


def get_close_series(symbols, default_start, default_end, cache_dir,
                     symbol=None, start=None, end=None):
    """Return a single close-price Series with DatetimeIndex."""
    symbol = symbol or symbols[0]
    data = get_prices([symbol], default_start, default_end, cache_dir,
                      start=start, end=end)
    if symbol not in data:
        raise ValueError(f"No data downloaded for {symbol}")
    return data[symbol]["Close"].squeeze()


def get_multi_close(symbols, default_start, default_end, cache_dir,
                    target_symbols=None, start=None, end=None):
    """Return a DataFrame of close prices: columns = symbols, rows = dates."""
    data = get_prices(target_symbols or symbols, default_start, default_end,
                      cache_dir, start=start, end=end)
    closes = pd.DataFrame({sym: df["Close"].squeeze() for sym, df in data.items()})
    return closes.dropna()


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_tick_data(n_ticks=50000, seed=42):
    """Simulate tick-level trade data (price, volume, timestamp).

    Returns DataFrame with columns ['price','volume','dollar','timestamp'].
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-03 09:30", periods=n_ticks, freq="200ms")
    log_returns = rng.normal(0, 0.0002, n_ticks)
    prices = 100 * np.exp(np.cumsum(log_returns))
    volumes = rng.integers(1, 500, n_ticks).astype(float)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "volume": volumes,
        "dollar": prices * volumes,
    })
    return df.set_index("timestamp")


def generate_ohlcv_bars(default_start, n_bars=2000, seed=42):
    """Simulate daily OHLCV bars for testing.

    Returns DataFrame with DatetimeIndex and columns
    ['Open','High','Low','Close','Volume'].
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(default_start, periods=n_bars)
    log_ret = rng.normal(0.0003, 0.015, n_bars)
    close = 100 * np.exp(np.cumsum(log_ret))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
    opn = low + (high - low) * rng.random(n_bars)
    vol = rng.integers(1_000_000, 20_000_000, n_bars).astype(float)
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol,
    }, index=dates)


def generate_returns(default_start, n=2000, n_assets=4, seed=42):
    """Generate a DataFrame of synthetic daily returns for *n_assets* assets."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(default_start, periods=n)
    cols = [f"Asset_{i}" for i in range(n_assets)]
    data = rng.normal(0.0003, 0.015, (n, n_assets))
    return pd.DataFrame(data, index=dates, columns=cols)


def generate_classification_data(default_start, n_samples=5000, n_features=20,
                                 n_informative=5, seed=42):
    """Generate synthetic classification dataset for ML experiments."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=seed,
    )
    dates = pd.bdate_range(default_start, periods=n_samples)
    X = pd.DataFrame(X, index=dates,
                     columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(y, index=dates, name="label")
    return X, y


def generate_ou_process(default_start, n=2000, theta=0.1, mu=100.0,
                        sigma=2.0, seed=42):
    """Simulate an Ornstein-Uhlenbeck mean-reverting price process.

    Parameters
    ----------
    n     : int   – number of time steps
    theta : float – speed of mean reversion
    mu    : float – long-run mean
    sigma : float – volatility
    seed  : int   – random seed

    Returns
    -------
    prices : pd.Series with DatetimeIndex
    """
    rng = np.random.default_rng(seed)
    dt = 1.0
    prices = np.zeros(n)
    prices[0] = mu
    for t in range(1, n):
        prices[t] = (prices[t - 1]
                     + theta * (mu - prices[t - 1]) * dt
                     + sigma * np.sqrt(dt) * rng.standard_normal())
    dates = pd.bdate_range(default_start, periods=n)
    return pd.Series(prices, index=dates, name="price")
