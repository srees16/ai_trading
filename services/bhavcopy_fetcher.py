"""
NSE Bhavcopy OHLCV Fetcher.

Downloads and caches daily NSE Bhavcopy (equity) CSV files from
nsearchives.nseindia.com and builds per-symbol OHLCV DataFrames
identical to yfinance output format.

Designed as a reliable fallback for Indian stocks when yfinance
fails (rate-limits, delistings, ticker mismatches).

Fallback chain in MarketDataService:
    Kite Connect → **Bhavcopy** → yfinance

Data source:
    ``https://nsearchives.nseindia.com/content/historical/EQUITIES/{YYYY}/{MMM}/cm{DDMMMYYYY}bhav.csv.zip``

Cache:
    ``data/bhavcopy_cache/{YYYY}/{YYYYMMDD}.csv``  (unzipped, one file per trading day)
"""

from __future__ import annotations

import io
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────

_BHAV_BASE = (
    "https://nsearchives.nseindia.com/products/content"
)
_BHAV_URL_TPL = "{base}/sec_bhavdata_full_{ddmmyyyy}.csv"

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "bhavcopy_cache"

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# Column mapping: new Bhavcopy CSV → internal names
_COL_MAP = {
    "OPEN_PRICE": "OPEN",
    "HIGH_PRICE": "HIGH",
    "LOW_PRICE": "LOW",
    "CLOSE_PRICE": "CLOSE",
    "TTL_TRD_QNTY": "TOTTRDQTY",
}

_SESSION: Optional[requests.Session] = None


# ── Session management ─────────────────────────────────────────

def _get_session() -> requests.Session:
    """Return a requests session with NSE cookies pre-loaded."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    sess = requests.Session()
    sess.headers.update(_NSE_HEADERS)
    try:
        # Hit the main NSE page to get CSRF / session cookies
        sess.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    _SESSION = sess
    return sess


def _reset_session() -> None:
    """Force a fresh session on next call (e.g. after 403)."""
    global _SESSION
    _SESSION = None


# ── Single-day bhavcopy download ───────────────────────────────

def _cache_path(d: date) -> Path:
    """Return the local cache file path for a given date."""
    return _CACHE_DIR / str(d.year) / f"{d.strftime('%Y%m%d')}.csv"


def _download_bhavcopy(d: date) -> Optional[pd.DataFrame]:
    """Download and parse the bhavcopy CSV for a single trading day.

    Returns a DataFrame with normalised columns:
        SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY, …

    Returns None if the date is a holiday / weekend or download fails.
    """
    cache = _cache_path(d)
    if cache.exists():
        try:
            return pd.read_csv(cache)
        except Exception:
            cache.unlink(missing_ok=True)

    # Skip weekends
    if d.weekday() >= 5:
        return None

    url = _BHAV_URL_TPL.format(
        base=_BHAV_BASE,
        ddmmyyyy=d.strftime("%d%m%Y"),
    )

    sess = _get_session()
    for attempt in range(2):
        try:
            resp = sess.get(url, timeout=15)
            if resp.status_code == 403 and attempt == 0:
                _reset_session()
                sess = _get_session()
                continue
            if resp.status_code == 404:
                # Market holiday — no file
                logger.debug("Bhavcopy not found for %s (likely holiday)", d)
                return None
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 0:
                _reset_session()
                sess = _get_session()
                continue
            logger.warning("Bhavcopy download failed for %s: %s", d, exc)
            return None

    # Parse CSV directly (new format is uncompressed)
    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        logger.warning("Bhavcopy CSV parse failed for %s: %s", d, exc)
        return None

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Normalise column names to match legacy format
    df.rename(columns=_COL_MAP, inplace=True)

    # Strip whitespace from key string columns (NSE pads with spaces)
    for col in ("SYMBOL", "SERIES"):
        if col in df.columns:
            df[col] = df[col].str.strip()

    # Cache to disk
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)

    return df


# ── Multi-day OHLCV builder ───────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    start: date,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Build a yfinance-compatible OHLCV DataFrame for *symbol* from Bhavcopy.

    Parameters
    ----------
    symbol : str
        NSE trading symbol (e.g. ``"RELIANCE"``). .NS/.BO suffixes are
        stripped automatically.
    start : date
        First date (inclusive).
    end : date, optional
        Last date (inclusive). Defaults to today.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (``Date``)
        Empty DataFrame if no data found.
    """
    # Strip exchange suffix
    sym = symbol.upper()
    for sfx in (".NS", ".BO"):
        if sym.endswith(sfx):
            sym = sym[: -len(sfx)]
            break

    if end is None:
        end = date.today()

    rows: List[Dict] = []
    current = start

    while current <= end:
        df = _download_bhavcopy(current)
        if df is not None and not df.empty:
            # Filter to symbol + EQ series (regular equity)
            mask = (df["SYMBOL"] == sym) & (df["SERIES"].isin(["EQ", "BE"]))
            matched = df.loc[mask]
            if not matched.empty:
                row = matched.iloc[0]
                rows.append(
                    {
                        "Date": pd.Timestamp(current),
                        "Open": float(row["OPEN"]),
                        "High": float(row["HIGH"]),
                        "Low": float(row["LOW"]),
                        "Close": float(row["CLOSE"]),
                        "Volume": int(row["TOTTRDQTY"]),
                    }
                )
        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    result = pd.DataFrame(rows).set_index("Date")
    result.index.name = "Date"
    return result


def fetch_ohlcv_batch(
    symbols: List[str],
    start: date,
    end: Optional[date] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple symbols efficiently.

    Downloads each bhavcopy only once and filters for all requested
    symbols simultaneously — much faster than calling fetch_ohlcv()
    per symbol.

    Returns a dict of ``{original_ticker: DataFrame}``.
    """
    if end is None:
        end = date.today()

    # Normalise symbols: strip suffix, keep mapping back to original
    raw_map: Dict[str, str] = {}  # raw_upper → original ticker
    for s in symbols:
        raw = s.upper()
        for sfx in (".NS", ".BO"):
            if raw.endswith(sfx):
                raw = raw[: -len(sfx)]
                break
        raw_map[raw] = s

    raw_set = set(raw_map.keys())
    accum: Dict[str, List[Dict]] = {r: [] for r in raw_set}

    current = start
    while current <= end:
        df = _download_bhavcopy(current)
        if df is not None and not df.empty:
            mask = df["SYMBOL"].isin(raw_set) & df["SERIES"].isin(["EQ", "BE"])
            for _, row in df.loc[mask].iterrows():
                sym = row["SYMBOL"]
                accum[sym].append(
                    {
                        "Date": pd.Timestamp(current),
                        "Open": float(row["OPEN"]),
                        "High": float(row["HIGH"]),
                        "Low": float(row["LOW"]),
                        "Close": float(row["CLOSE"]),
                        "Volume": int(row["TOTTRDQTY"]),
                    }
                )
        current += timedelta(days=1)

    results: Dict[str, pd.DataFrame] = {}
    for raw, orig in raw_map.items():
        rows = accum.get(raw, [])
        if rows:
            result_df = pd.DataFrame(rows).set_index("Date")
            result_df.index.name = "Date"
            results[orig] = result_df

    return results
