"""
Binance Public API Data Fetcher

Downloads OHLCV kline data from the Binance public REST API.
No API key is required — uses the unauthenticated ``/api/v3/klines`` endpoint.

Per-symbol CSV caching ensures that:
* Only *missing* symbols are downloaded on subsequent calls.
* Adding a new symbol to an existing set doesn't re-download the others.
* Cached data is extended (appended) when the date range grows.

Usage::

    from binance_data import fetch_crypto_prices

    prices = fetch_crypto_prices(
        symbols={"ETHUSDT": "eth", "BTCUSDT": "btc"},
        start="2020-01-01",
    )
"""

import os
import time
import logging
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_LIMIT = 1000  # Binance maximum per request

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ============================================================
# Core Functions
# ============================================================
def _to_ms(dt_str: str) -> int:
    """Convert a date string (YYYY-MM-DD) to milliseconds since epoch (UTC)."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    start: str = "2017-08-17",
    end: str | None = None,
    pause: float = 0.25,
) -> pd.DataFrame:
    """
    Fetch all klines for *symbol* from Binance, paginating automatically.

    Parameters
    ----------
    symbol : str
        Binance trading pair, e.g. ``"BTCUSDT"``.
    interval : str
        Kline interval, e.g. ``"1d"`` (daily), ``"1h"`` (hourly).
    start : str
        Start date as ``"YYYY-MM-DD"`` (inclusive).
    end : str or None
        End date as ``"YYYY-MM-DD"`` (inclusive).  ``None`` means today.
    pause : float
        Seconds to sleep between paginated requests to avoid rate limits.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``.
        Index is reset (integer).  ``timestamp`` is timezone-aware (UTC).
    """
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        resp = requests.get(f"{BASE_URL}{KLINES_ENDPOINT}", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)

        # Next page starts 1 ms after the last candle's close time
        last_close_time = data[-1][6]  # close_time field
        current_start = last_close_time + 1

        if len(data) < MAX_LIMIT:
            break  # No more data to fetch

        time.sleep(pause)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ============================================================
# Per-Symbol Cache Helpers
# ============================================================
def _cache_path(symbol: str, interval: str = "1d") -> str:
    """Return path to the per-symbol cache CSV: data/<SYMBOL>_<interval>.csv"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{symbol}_{interval}.csv")


def _load_symbol_cache(
    symbol: str,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """Load cached klines for a single symbol, or ``None`` if missing."""
    path = _cache_path(symbol, interval)
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception as exc:
        logger.warning("Corrupt cache for %s, will re-download: %s", symbol, exc)
        return None


def _save_symbol_cache(
    symbol: str,
    df: pd.DataFrame,
    interval: str = "1d",
) -> None:
    """Persist klines for a single symbol to its cache file."""
    path = _cache_path(symbol, interval)
    df.to_csv(path, index=False)


def _fetch_with_incremental_cache(
    symbol: str,
    interval: str = "1d",
    start: str = "2017-08-17",
    end: str | None = None,
    pause: float = 0.25,
) -> pd.DataFrame:
    """
    Fetch klines for *symbol*, using a per-symbol CSV cache to avoid
    re-downloading data that already exists locally.

    * If no cache exists → full download.
    * If cache exists but doesn't cover the requested range →
      download only the missing tail (or head) and append.
    """
    requested_start = pd.Timestamp(start, tz="UTC")
    requested_end = (
        pd.Timestamp(end, tz="UTC")
        if end
        else pd.Timestamp.now(tz="UTC").normalize()
    )

    cached = _load_symbol_cache(symbol, interval)

    if cached is not None and not cached.empty:
        cache_min = cached["timestamp"].min()
        cache_max = cached["timestamp"].max()

        need_before = requested_start < cache_min
        need_after = requested_end > cache_max + pd.Timedelta(days=1)

        if not need_before and not need_after:
            # Cache fully covers the request
            mask = (cached["timestamp"] >= requested_start) & (
                cached["timestamp"] <= requested_end
            )
            return cached.loc[mask].reset_index(drop=True)

        parts = [cached]

        if need_before:
            head_end = (cache_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info("  %s: fetching earlier data %s → %s", symbol, start, head_end)
            head = fetch_klines(symbol, interval, start, head_end, pause)
            if not head.empty:
                parts.insert(0, head)

        if need_after:
            tail_start = (cache_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            tail_end = end  # None means today
            logger.info("  %s: fetching newer data %s → %s", symbol, tail_start, tail_end or "today")
            tail = fetch_klines(symbol, interval, tail_start, tail_end, pause)
            if not tail.empty:
                parts.append(tail)

        merged = pd.concat(parts, ignore_index=True)
        merged.drop_duplicates(subset=["timestamp"], inplace=True)
        merged.sort_values("timestamp", inplace=True)
        merged.reset_index(drop=True, inplace=True)

        _save_symbol_cache(symbol, merged, interval)

        mask = (merged["timestamp"] >= requested_start) & (
            merged["timestamp"] <= requested_end
        )
        return merged.loc[mask].reset_index(drop=True)

    # No cache — full download
    df = fetch_klines(symbol, interval, start, end, pause)
    if not df.empty:
        _save_symbol_cache(symbol, df, interval)
    return df


# ============================================================
# Public API
# ============================================================
def fetch_crypto_prices(
    symbols: Dict[str, str],
    interval: str = "1d",
    start: str = "2017-08-17",
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily close prices for multiple cryptocurrencies and return
    a single wide DataFrame.

    Parameters
    ----------
    symbols : dict
        **Required.** Mapping of Binance symbol → column name,
        e.g. ``{"ETHUSDT": "eth", "BTCUSDT": "btc"}``.
    interval : str
        Kline interval (default ``"1d"``).
    start : str
        Start date ``"YYYY-MM-DD"``.
    end : str or None
        End date ``"YYYY-MM-DD"``; ``None`` = today.
    cache : bool
        If ``True``, use per-symbol CSV caches so only missing data is
        downloaded.

    Returns
    -------
    pd.DataFrame
        Columns named after each crypto (e.g. ``eth``, ``btc``).
        Index is ``timestamp`` (DatetimeIndex, UTC).
    """
    if not symbols:
        raise ValueError(
            "symbols dict is required — e.g. {'ETHUSDT': 'eth', 'BTCUSDT': 'btc'}"
        )

    frames: dict[str, pd.Series] = {}

    for binance_symbol, col_name in symbols.items():
        logger.info("  Fetching %s (%s)…", binance_symbol, col_name)
        if cache:
            df = _fetch_with_incremental_cache(
                binance_symbol, interval, start, end,
            )
        else:
            df = fetch_klines(binance_symbol, interval, start, end)

        if df.empty:
            logger.warning("  %s returned no data — skipping", binance_symbol)
            continue

        series = df.set_index("timestamp")["close"].rename(col_name)
        frames[col_name] = series
        logger.info("  %s: %d rows", col_name, len(series))

    if not frames:
        return pd.DataFrame()

    # Combine into a single DataFrame, aligned on timestamp
    prices = pd.DataFrame(frames)
    prices.index.name = "timestamp"
    prices.sort_index(inplace=True)

    logger.info("Final prices shape: %s", prices.shape)
    return prices


# ============================================================
# Migrate legacy monolithic cache to per-symbol caches
# ============================================================
def migrate_legacy_cache() -> None:
    """
    One-time helper: reads the old ``crypto_prices_cache.csv`` and
    splits it into per-symbol files under ``data/``.
    """
    old_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "crypto_prices_cache.csv",
    )
    if not os.path.isfile(old_path):
        return

    logger.info("Migrating legacy cache %s → per-symbol files", old_path)
    cached = pd.read_csv(old_path, index_col="timestamp", parse_dates=True)

    # Guess the Binance symbol from column name
    _col_to_symbol = {
        "eth": "ETHUSDT",
        "btc": "BTCUSDT",
        "ltc": "LTCUSDT",
    }

    for col in cached.columns:
        symbol = _col_to_symbol.get(col.lower())
        if not symbol:
            symbol = f"{col.upper()}USDT"

        df = pd.DataFrame({
            "timestamp": cached.index,
            "open": float("nan"),
            "high": float("nan"),
            "low": float("nan"),
            "close": cached[col].values,
            "volume": float("nan"),
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.dropna(subset=["close"], inplace=True)
        _save_symbol_cache(symbol, df)
        logger.info("  Wrote %d rows for %s", len(df), symbol)

    # Rename legacy file so it's not loaded again
    renamed = old_path + ".migrated"
    os.rename(old_path, renamed)
    logger.info("Legacy cache renamed to %s", renamed)


# ============================================================
# Direct execution — quick test
# ============================================================
if __name__ == "__main__":
    # Migrate legacy cache if present
    migrate_legacy_cache()

    prices = fetch_crypto_prices(
        symbols={"ETHUSDT": "eth", "BTCUSDT": "btc", "LTCUSDT": "ltc"},
        start="2017-08-17",
    )
    print(prices.head(10))
    print(prices.tail(5))
    print(f"\nShape: {prices.shape}")
    print(f"Date range: {prices.index.min()} → {prices.index.max()}")
