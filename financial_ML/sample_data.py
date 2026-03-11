"""
sample_data.py – Shared data-generation module for AFML chapter scripts.

Uses yfinance to download real market data and provides synthetic generators
for cases where real data is not appropriate (e.g., tick-level bars, futures).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOLS = ["MSFT", "GOOG", "NVDA", "AMD"]
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
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_parquet(cache_file)
        result[sym] = df
    return result


def get_close_series(symbol="MSFT", start=None, end=None):
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


def generate_ohlcv_bars(n_bars=2000, seed=42):
    """Simulate daily OHLCV bars for testing.

    Returns DataFrame with DatetimeIndex and columns
    ['Open','High','Low','Close','Volume'].
    """
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


def generate_returns(n=2000, n_assets=4, seed=42):
    """Generate a DataFrame of synthetic daily returns for *n_assets* assets."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols = [f"Asset_{i}" for i in range(n_assets)]
    data = rng.normal(0.0003, 0.015, (n, n_assets))
    return pd.DataFrame(data, index=dates, columns=cols)


def generate_classification_data(n_samples=5000, n_features=20,
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
    dates = pd.bdate_range("2020-01-02", periods=n_samples)
    X = pd.DataFrame(X, index=dates,
                     columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(y, index=dates, name="label")
    return X, y


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("sample_data.py – self-test")
    print("=" * 60)

    print("\n[1] Downloading real prices …")
    prices = get_prices(["MSFT", "GOOG"], start="2023-01-01", end="2024-01-01")
    for sym, df in prices.items():
        print(f"  {sym}: {len(df)} bars, columns={list(df.columns)}")

    print("\n[2] Synthetic tick data …")
    ticks = generate_tick_data(1000)
    print(f"  shape={ticks.shape}, cols={list(ticks.columns)}")

    print("\n[3] Synthetic OHLCV bars …")
    bars = generate_ohlcv_bars(500)
    print(f"  shape={bars.shape}, cols={list(bars.columns)}")

    print("\n[4] Synthetic returns …")
    rets = generate_returns(500, 4)
    print(f"  shape={rets.shape}, cols={list(rets.columns)}")

    print("\n[5] Classification data …")
    X, y = generate_classification_data(1000, 10, 3)
    print(f"  X.shape={X.shape}, y.shape={y.shape}, labels={sorted(y.unique())}")

    print("\nAll self-tests passed ✓")
