"""
sample_data.py – Shared data-generation module for AFML chapter scripts.

Uses yfinance to download real market data and provides synthetic generators
for cases where real data is not appropriate (e.g., tick-level bars, futures).

Delegates to ``shared.sample_data_base`` for implementation — this thin
wrapper only defines the FML-specific configuration (env var prefix,
default symbols, cache directory).
"""

import os
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.sample_data_base import (  # noqa: E402
    get_prices as _get_prices,
    get_close_series as _get_close_series,
    get_multi_close as _get_multi_close,
    generate_tick_data,
    generate_ohlcv_bars as _generate_ohlcv_bars,
    generate_returns as _generate_returns,
    generate_classification_data as _generate_classification_data,
)

# ---------------------------------------------------------------------------
# Configuration — env vars override defaults when set by the runner
# ---------------------------------------------------------------------------
_env_tickers = os.environ.get("FML_TICKERS")
SYMBOLS = [t.strip() for t in _env_tickers.split(",") if t.strip()] if _env_tickers else ["MSFT", "GOOG", "NVDA", "AMD"]
DEFAULT_START = os.environ.get("FML_DATE_START", "2020-01-01")
DEFAULT_END = os.environ.get("FML_DATE_END", "2024-12-31")
CACHE_DIR = Path(__file__).parent / "_cache"

# ---------------------------------------------------------------------------
# Thin wrappers that bind FML configuration
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

def generate_returns(n=2000, n_assets=4, seed=42):
    return _generate_returns(DEFAULT_START, n=n, n_assets=n_assets, seed=seed)

def generate_classification_data(n_samples=5000, n_features=20,
                                 n_informative=5, seed=42):
    return _generate_classification_data(DEFAULT_START, n_samples=n_samples,
                                         n_features=n_features,
                                         n_informative=n_informative, seed=seed)


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
