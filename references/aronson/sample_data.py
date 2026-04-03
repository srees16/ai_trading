"""
sample_data.py – Data helpers for Aronson EBTA chapter scripts.

Delegates to shared sample_data_base for yfinance download/cache.
"""

import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from references.sample_data_base import (  # noqa: E402
    get_prices as _get_prices,
    get_close_series as _get_close_series,
    get_multi_close as _get_multi_close,
    generate_returns as _generate_returns,
)

_env_tickers = os.environ.get("ARON_TICKERS")
SYMBOLS = [t.strip() for t in _env_tickers.split(",") if t.strip()] if _env_tickers else ["MSFT", "GOOG", "NVDA", "AMD"]
DEFAULT_START = os.environ.get("ARON_DATE_START", "2020-01-01")
DEFAULT_END = os.environ.get("ARON_DATE_END", "2024-12-31")
CACHE_DIR = Path(__file__).parent / "_cache"


def get_prices(symbols=None, start=None, end=None, interval="1d"):
    return _get_prices(symbols or SYMBOLS, DEFAULT_START, DEFAULT_END, CACHE_DIR,
                       start=start, end=end, interval=interval)

def get_close_series(symbol=None, start=None, end=None):
    return _get_close_series(SYMBOLS, DEFAULT_START, DEFAULT_END, CACHE_DIR,
                             symbol=symbol, start=start, end=end)

def get_multi_close(symbols=None, start=None, end=None):
    return _get_multi_close(SYMBOLS, DEFAULT_START, DEFAULT_END, CACHE_DIR,
                            target_symbols=symbols, start=start, end=end)

def generate_returns(n=2000, n_assets=4, seed=42):
    return _generate_returns(DEFAULT_START, n=n, n_assets=n_assets, seed=seed)
