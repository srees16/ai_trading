"""
sample_data.py – Shared data-generation module for TTMTS chapter scripts.

Based on: "Testing and Tuning Market Trading Systems" by Timothy Masters.

Uses yfinance to download real market data and provides synthetic generators
for cases where real data is not appropriate (e.g., Ornstein-Uhlenbeck
processes, random trading systems).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"
CACHE_DIR = Path(__file__).parent / "_cache"

# ---------------------------------------------------------------------------
# Real market data helpers
# ---------------------------------------------------------------------------

def get_prices(symbols=None, start=None, end=None, interval="1d"):
    """Download daily OHLCV data via yfinance and cache locally.

    Returns a dict  {symbol: DataFrame} with columns
    ['Open','High','Low','Close','Volume'].
    """
    symbols = symbols or SYMBOLS
    start = start or DEFAULT_START
    end = end or DEFAULT_END
    CACHE_DIR.mkdir(exist_ok=True)

    result = {}
    for sym in symbols:
        cache_file = CACHE_DIR / f"{sym}_{start}_{end}_{interval}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            df = yf.download(sym, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"[sample_data] WARNING: no data for {sym}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_parquet(cache_file)
        result[sym] = df
    return result


def get_close_series(symbol="SPY", start=None, end=None):
    """Return a single close-price Series with DatetimeIndex."""
    data = get_prices([symbol], start=start, end=end)
    if symbol not in data:
        raise ValueError(f"No data downloaded for {symbol}")
    return data[symbol]["Close"].squeeze()


def get_multi_close(symbols=None, start=None, end=None):
    """Return a DataFrame of close prices: columns = symbols, rows = dates."""
    data = get_prices(symbols, start=start, end=end)
    closes = pd.DataFrame({sym: df["Close"].squeeze() for sym, df in data.items()})
    return closes.dropna()


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_ohlcv_bars(n_bars=2000, seed=42):
    """Simulate daily OHLCV bars for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_bars)
    log_ret = rng.normal(0.0003, 0.015, n_bars)
    close = 100 * np.exp(np.cumsum(log_ret))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
    opn = low + (high - low) * rng.random(n_bars)
    vol = rng.integers(1_000_000, 20_000_000, n_bars).astype(float)
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol,
    }, index=dates)


def generate_returns(n=2000, n_assets=1, seed=42):
    """Generate a DataFrame of synthetic daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    if n_assets == 1:
        data = rng.normal(0.0003, 0.015, n)
        return pd.Series(data, index=dates, name="returns")
    cols = [f"Asset_{i}" for i in range(n_assets)]
    data = rng.normal(0.0003, 0.015, (n, n_assets))
    return pd.DataFrame(data, index=dates, columns=cols)


def generate_ou_process(n=2000, theta=0.1, mu=100.0, sigma=2.0, seed=42):
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
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(prices, index=dates, name="price")


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
