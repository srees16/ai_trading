"""
NSE F&O Bhavcopy Fetcher — Real Open Interest Data.

Downloads daily F&O bhavcopy from NSE to get actual open interest data
instead of using volume as a proxy.

Data source:
    https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{YYYY}/{MMM}/fo{DDMMMYYYY}bhav.csv.zip
    
Fallback for newer format:
    https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{DDMMYYYY}.csv

Provides per-symbol OI change data for the oi_signal module.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fo_bhavcopy_cache"

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

_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    sess = requests.Session()
    sess.headers.update(_NSE_HEADERS)
    try:
        sess.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    _SESSION = sess
    return sess


def _cache_path(d: date) -> Path:
    return _CACHE_DIR / str(d.year) / f"fo_{d.strftime('%Y%m%d')}.csv"


def _download_fo_bhavcopy(d: date) -> Optional[pd.DataFrame]:
    """Download NSE F&O bhavcopy for a given date."""
    cache = _cache_path(d)
    if cache.exists():
        try:
            return pd.read_csv(str(cache))
        except Exception:
            pass

    # Try the zip format first
    month_str = d.strftime("%b").upper()
    day_str = d.strftime("%d%b%Y").upper()
    url = (
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
        f"{d.year}/{month_str}/fo{day_str}bhav.csv.zip"
    )

    sess = _get_session()
    df = None

    try:
        resp = sess.get(url, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 100:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
    except Exception as exc:
        logger.debug("F&O bhavcopy zip download failed for %s: %s", d, exc)

    if df is None:
        # Fallback: try flat CSV format
        dd_mm_yyyy = d.strftime("%d%m%Y")
        url2 = (
            f"https://nsearchives.nseindia.com/products/content/"
            f"sec_bhavdata_full_{dd_mm_yyyy}.csv"
        )
        try:
            resp = sess.get(url2, timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
        except Exception:
            pass

    if df is not None and not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(cache), index=False)
        logger.info("F&O bhavcopy cached for %s (%d rows)", d, len(df))

    return df


def _parse_oi_data(df: pd.DataFrame) -> Dict[str, Dict]:
    """Extract per-symbol OI data from F&O bhavcopy DataFrame.
    
    Returns {symbol: {"oi": int, "oi_change": int, "close": float, "volume": int}}
    """
    result: Dict[str, Dict] = {}

    # Column names vary by format — normalize
    cols = {c.strip().upper(): c for c in df.columns}

    # Try standard F&O bhavcopy columns
    symbol_col = cols.get("SYMBOL", cols.get("TRD_SYMBOL"))
    oi_col = cols.get("OPEN_INT", cols.get("OI", cols.get("OPEN_INTEREST")))
    chg_oi_col = cols.get("CHG_IN_OI", cols.get("CHANGE_IN_OI"))
    close_col = cols.get("CLOSE", cols.get("CLOSE_PRICE", cols.get("SETTLE_PR")))
    vol_col = cols.get("CONTRACTS", cols.get("VOLUME", cols.get("TTL_TRD_QNTY")))
    inst_col = cols.get("INSTRUMENT", cols.get("SERIES"))

    if not symbol_col or not oi_col:
        return result

    for _, row in df.iterrows():
        try:
            # Only stock futures (FUTSTK) for individual stock OI
            inst = str(row.get(inst_col, "")).strip() if inst_col else ""
            if inst and inst not in ("FUTSTK", "EQ", ""):
                continue

            sym = str(row[symbol_col]).strip()
            if not sym or sym == "nan":
                continue

            oi = int(float(row.get(oi_col, 0) or 0))
            chg_oi = int(float(row.get(chg_oi_col, 0) or 0)) if chg_oi_col else 0
            close = float(row.get(close_col, 0) or 0) if close_col else 0.0
            vol = int(float(row.get(vol_col, 0) or 0)) if vol_col else 0

            # Aggregate: for same symbol with multiple expiries, use near-month
            if sym not in result or oi > result[sym].get("oi", 0):
                result[sym] = {
                    "oi": oi,
                    "oi_change": chg_oi,
                    "close": close,
                    "volume": vol,
                }
        except Exception:
            continue

    return result


def fetch_fo_oi_data(
    target_date: Optional[date] = None,
    lookback_days: int = 2,
) -> Dict[str, Dict]:
    """Fetch F&O OI data with change percentages for oi_signal integration.
    
    Returns {symbol: {"oi_change_pct": float, "price_change_pct": float, "volume_ratio": float}}
    """
    if target_date is None:
        target_date = date.today()

    # Get today's and previous day's data for computing changes
    today_data = None
    prev_data = None

    for offset in range(lookback_days + 3):  # try a few days in case market was closed
        d = target_date - timedelta(days=offset)
        if d.weekday() >= 5:  # skip weekends
            continue
        df = _download_fo_bhavcopy(d)
        if df is not None and not df.empty:
            parsed = _parse_oi_data(df)
            if parsed:
                if today_data is None:
                    today_data = parsed
                elif prev_data is None:
                    prev_data = parsed
                    break

    if not today_data:
        logger.warning("No F&O bhavcopy data available")
        return {}

    result: Dict[str, Dict] = {}
    for sym, data in today_data.items():
        oi = data["oi"]
        close = data["close"]

        if oi <= 0 or close <= 0:
            continue

        # Compute OI change % vs previous day
        oi_change_pct = 0.0
        price_change_pct = 0.0
        volume_ratio = 1.0

        if prev_data and sym in prev_data:
            prev = prev_data[sym]
            if prev["oi"] > 0:
                oi_change_pct = ((oi - prev["oi"]) / prev["oi"]) * 100
            if prev["close"] > 0:
                price_change_pct = ((close - prev["close"]) / prev["close"]) * 100
            if prev.get("volume", 0) > 0 and data.get("volume", 0) > 0:
                volume_ratio = data["volume"] / prev["volume"]

        result[sym] = {
            "oi_change_pct": round(oi_change_pct, 2),
            "price_change_pct": round(price_change_pct, 2),
            "volume_ratio": round(min(3.0, max(0.3, volume_ratio)), 2),
        }

    logger.info("F&O OI data: %d symbols with real OI data", len(result))
    return result
