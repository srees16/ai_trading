"""
NSE Bhavcopy OHLCV Fetcher.

Downloads and caches daily NSE Bhavcopy (equity) CSV files from
nsearchives.nseindia.com and builds per-symbol OHLCV DataFrames
identical to yfinance output format.

Designed as a reliable fallback for Indian stocks when yfinance
fails (rate-limits, delistings, ticker mismatches).

Fallback chain in MarketDataService:
    Kite Connect → **Bhavcopy** → yfinance

URL formats:
    Legacy zip (2012–2023):
        ``https://nsearchives.nseindia.com/content/historical/EQUITIES/{YYYY}/{MMM}/cm{DDMMMYYYY}bhav.csv.zip``
    New CSV (2019+):
        ``https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{DDMMYYYY}.csv``

Cache:
    ``data/bhavcopy_cache/{YYYY}/{YYYYMMDD}.csv``  (normalised, one file per trading day)

Bulk download:
    ``download_full_history(start, end, max_workers)`` — parallel download of
    entire date range with progress reporting and automatic format selection.

Corporate actions:
    ``fetch_corporate_actions(symbol)`` — splits & bonuses from NSE API.
    ``adjust_ohlcv(df, symbol)`` — apply split/bonus adjustments to raw OHLCV.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────

# New-format CSV (works ~2019+)
_BHAV_NEW_BASE = "https://nsearchives.nseindia.com/products/content"
_BHAV_NEW_TPL = "{base}/sec_bhavdata_full_{ddmmyyyy}.csv"

# Legacy zip format (works 2012–2023)
_BHAV_LEGACY_TPL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES"
    "/{year}/{mon}/cm{ddmonyyyy}bhav.csv.zip"
)

# Corporate actions API
_CA_API = (
    "https://www.nseindia.com/api/corporates-corporateActions"
    "?index=equities&from_date={from_date}&to_date={to_date}&symbol={symbol}"
)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "bhavcopy_cache"
_CA_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "corporate_actions_cache"

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",  # no brotli — requests can't decode it natively
    "Referer": "https://www.nseindia.com/",
}

# Column mapping: new Bhavcopy CSV → normalised names
_NEW_COL_MAP = {
    "OPEN_PRICE": "OPEN",
    "HIGH_PRICE": "HIGH",
    "LOW_PRICE": "LOW",
    "CLOSE_PRICE": "CLOSE",
    "TTL_TRD_QNTY": "TOTTRDQTY",
}

_SESSION: Optional[requests.Session] = None
_SESSION_CREATED: float = 0.0
_SESSION_MAX_AGE = 300  # refresh session cookie every 5 minutes


# ── Session management ─────────────────────────────────────────

def _get_session() -> requests.Session:
    """Return a requests session with NSE cookies pre-loaded."""
    global _SESSION, _SESSION_CREATED
    now = time.monotonic()
    if _SESSION is not None and (now - _SESSION_CREATED) < _SESSION_MAX_AGE:
        return _SESSION
    sess = requests.Session()
    sess.headers.update(_NSE_HEADERS)
    try:
        sess.get("https://www.nseindia.com", timeout=10)
    except Exception as exc:
        logger.debug("NSE session cookie pre-fetch failed: %s", exc)
    _SESSION = sess
    _SESSION_CREATED = now
    return sess


def _reset_session() -> None:
    """Force a fresh session on next call (e.g. after 403)."""
    global _SESSION
    _SESSION = None


# ── Single-day bhavcopy download ───────────────────────────────

def _cache_path(d: date) -> Path:
    """Return the local cache file path for a given date."""
    return _CACHE_DIR / str(d.year) / f"{d.strftime('%Y%m%d')}.csv"


def _download_bhavcopy_legacy(d: date, sess: requests.Session) -> Optional[pd.DataFrame]:
    """Download bhavcopy using legacy zip format (2012–2023)."""
    mon = d.strftime("%b").upper()
    ddmonyyyy = d.strftime("%d%b%Y").upper()
    url = _BHAV_LEGACY_TPL.format(year=d.year, mon=mon, ddmonyyyy=ddmonyyyy)

    resp = sess.get(url, timeout=20)
    if resp.status_code in (404, 403):
        return None
    resp.raise_for_status()

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    except Exception as exc:
        logger.warning("Legacy bhavcopy zip parse failed for %s: %s", d, exc)
        return None

    # Legacy format already has SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY
    df.columns = df.columns.str.strip()
    for col in ("SYMBOL", "SERIES"):
        if col in df.columns:
            df[col] = df[col].str.strip()
    return df


def _download_bhavcopy_new(d: date, sess: requests.Session) -> Optional[pd.DataFrame]:
    """Download bhavcopy using new CSV format (2019+)."""
    url = _BHAV_NEW_TPL.format(base=_BHAV_NEW_BASE, ddmmyyyy=d.strftime("%d%m%Y"))

    resp = sess.get(url, timeout=20)
    if resp.status_code in (404, 403):
        return None
    resp.raise_for_status()

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        logger.warning("New bhavcopy CSV parse failed for %s: %s", d, exc)
        return None

    df.columns = df.columns.str.strip()
    df.rename(columns=_NEW_COL_MAP, inplace=True)
    for col in ("SYMBOL", "SERIES"):
        if col in df.columns:
            df[col] = df[col].str.strip()
    return df


def _download_bhavcopy(d: date) -> Optional[pd.DataFrame]:
    """Download and parse the bhavcopy CSV for a single trading day.

    Tries legacy zip format first (broader historical coverage),
    then falls back to new CSV format.

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

    sess = _get_session()

    # Try legacy zip first (works 2012–2023), then new CSV (2019+)
    df = None
    for attempt in range(2):
        try:
            df = _download_bhavcopy_legacy(d, sess)
            if df is not None:
                break
            df = _download_bhavcopy_new(d, sess)
            if df is not None:
                break
            # Both 404 → holiday
            return None
        except requests.RequestException:
            if attempt == 0:
                _reset_session()
                sess = _get_session()
                continue
            logger.warning("Bhavcopy download failed for %s after retries", d)
            return None

    if df is None or df.empty:
        return None

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


# ── Bulk historical download ──────────────────────────────────

def _trading_days(start: date, end: date) -> List[date]:
    """Generate all weekdays between start and end (inclusive)."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def download_full_history(
    start: date = date(2012, 1, 1),
    end: Optional[date] = None,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> int:
    """Download and cache all bhavcopy files from *start* to *end*.

    Parameters
    ----------
    start : date
        First date (inclusive). Default: 2012-01-01.
    end : date, optional
        Last date (inclusive). Default: today.
    max_workers : int
        Number of parallel download threads. Keep ≤ 5 to avoid
        NSE rate-limiting / IP bans.
    progress_callback : callable, optional
        Called as ``callback(downloaded, skipped, total)`` after each day.

    Returns
    -------
    int
        Number of new files downloaded (excludes cache hits).
    """
    if end is None:
        end = date.today()

    days = _trading_days(start, end)
    total = len(days)

    # Split into already-cached and to-download
    to_download = []
    skipped = 0
    for d in days:
        if _cache_path(d).exists():
            skipped += 1
        else:
            to_download.append(d)

    if not to_download:
        print(f"All {total} trading days already cached.")
        return 0

    print(f"Bhavcopy bulk download: {len(to_download)} new / {skipped} cached / {total} total")
    print(f"Date range: {start} to {end}")
    print(f"Workers: {max_workers}")

    downloaded = 0
    failed = 0

    def _fetch_one(d: date) -> Tuple[date, bool]:
        """Download a single day. Returns (date, success)."""
        try:
            df = _download_bhavcopy(d)
            return (d, df is not None and not df.empty)
        except Exception as exc:
            logger.warning("Failed %s: %s", d, exc)
            return (d, False)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, d): d for d in to_download}
        for future in as_completed(futures):
            d, success = future.result()
            if success:
                downloaded += 1
            else:
                failed += 1
            done = downloaded + failed
            if progress_callback:
                progress_callback(downloaded, skipped + failed, total)
            if done % 100 == 0 or done == len(to_download):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(to_download) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{len(to_download)}] "
                    f"downloaded={downloaded} failed={failed} "
                    f"rate={rate:.1f}/s ETA={eta / 60:.0f}m",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\nBhavcopy download complete: {downloaded} downloaded, {failed} failed "
          f"({elapsed / 60:.1f} min)")
    return downloaded


# ── Corporate actions (splits & bonuses) ───────────────────────

def _parse_split_ratio(subject: str) -> Optional[float]:
    """Parse a corporate action subject line into an adjustment factor.

    Examples:
        "Bonus 1:1"           → 2.0  (you get 1 share per 1 held)
        "Stock Split From Rs.10/- To Rs.2/-"  → 5.0
        "Sub-Division of shares from FV Rs.10 to FV Rs.2"  → 5.0
        "Bonus issue 3:1"     → 1.333...  (3 shares for every 1)

    Returns None if the subject is not a split/bonus.
    """
    subj = subject.lower()

    # Bonus: "bonus X:Y" means X new shares for every Y held → factor = (X+Y)/Y
    bonus_match = re.search(r'bonus\s+(?:issue\s+)?(\d+)\s*:\s*(\d+)', subj)
    if bonus_match:
        new, held = int(bonus_match.group(1)), int(bonus_match.group(2))
        if held > 0:
            return (new + held) / held

    # Stock split: "from Rs.X/- to Rs.Y/-" or "from FV Rs.X to FV Rs.Y"
    split_match = re.search(r'(?:from|fv)\s*(?:rs\.?\s*)?(\d+)\s*(?:/-)?\s*(?:to|per)\s*(?:rs\.?\s*)?(\d+)', subj)
    if split_match:
        old_fv, new_fv = int(split_match.group(1)), int(split_match.group(2))
        if new_fv > 0 and old_fv > new_fv:
            return old_fv / new_fv

    # Sub-division shorthand: "sub-division" or "sub division" + FV numbers
    if 'sub-division' in subj or 'sub division' in subj or 'split' in subj:
        fv_match = re.findall(r'(?:rs\.?\s*)?(\d+)', subj)
        if len(fv_match) >= 2:
            vals = [int(x) for x in fv_match]
            old_fv = max(vals)
            new_fv = min(v for v in vals if v > 0)
            if old_fv > new_fv:
                return old_fv / new_fv

    return None


def fetch_corporate_actions(
    symbol: str,
    start: date = date(2012, 1, 1),
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch split/bonus corporate actions for a symbol from NSE.

    Returns a DataFrame with columns: ex_date, factor, subject
    where factor is the adjustment multiplier (e.g. 2.0 for 1:1 bonus).
    Only includes splits and bonuses (not dividends).
    """
    sym = symbol.upper()
    for sfx in (".NS", ".BO"):
        if sym.endswith(sfx):
            sym = sym[: -len(sfx)]
            break

    if end is None:
        end = date.today()

    # Check cache
    _CA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CA_CACHE_DIR / f"{sym}.csv"
    if cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["ex_date"])
        # Use cache if it's recent (< 30 days old)
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime).date()
        if (date.today() - mtime).days < 30:
            return cached

    from_str = start.strftime("%d-%m-%Y")
    to_str = end.strftime("%d-%m-%Y")
    url = _CA_API.format(from_date=from_str, to_date=to_str, symbol=sym)

    # Corporate actions API is stricter than bhavcopy downloads — needs fresh
    # session cookies and JSON Accept header. Use a dedicated session.
    data = None
    for attempt in range(3):
        try:
            ca_sess = requests.Session()
            ca_hdrs = dict(_NSE_HEADERS)
            ca_hdrs["Accept"] = "application/json, text/javascript, */*; q=0.01"
            ca_hdrs["Accept-Encoding"] = "gzip, deflate"  # no brotli — requests can't decode it
            ca_sess.headers.update(ca_hdrs)
            ca_sess.get("https://www.nseindia.com", timeout=10)
            time.sleep(0.5)
            resp = ca_sess.get(url, timeout=15)
            if resp.status_code == 403:
                time.sleep(2)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            if attempt == 2:
                logger.warning("Corporate actions fetch failed for %s: %s", sym, exc)
                return pd.DataFrame(columns=["ex_date", "factor", "subject"])

    if data is None:
        return pd.DataFrame(columns=["ex_date", "factor", "subject"])

    actions = []
    for item in data:
        subject = item.get("subject", "")
        factor = _parse_split_ratio(subject)
        if factor is not None and factor > 1.0:
            ex_str = item.get("exDate", "")
            try:
                ex_date = pd.to_datetime(ex_str, dayfirst=True).date()
            except Exception:
                continue
            actions.append({
                "ex_date": pd.Timestamp(ex_date),
                "factor": factor,
                "subject": subject,
            })

    result = pd.DataFrame(actions, columns=["ex_date", "factor", "subject"])
    if not result.empty:
        result.sort_values("ex_date", inplace=True)
        result.reset_index(drop=True, inplace=True)

    # Cache
    result.to_csv(cache_file, index=False)
    return result


def adjust_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    start: date = date(2012, 1, 1),
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Apply corporate action adjustments to raw bhavcopy OHLCV.

    Adjusts Open, High, Low, Close backward from the most recent price
    (splits older prices by the cumulative factor). Volume is adjusted
    inversely (multiplied by the factor).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV with DatetimeIndex. Columns: Open, High, Low, Close, Volume.
    symbol : str
        NSE trading symbol.
    start, end : date
        Date range for fetching corporate actions.

    Returns
    -------
    pd.DataFrame
        Adjusted OHLCV (same shape, same index).
    """
    if df.empty:
        return df

    actions = fetch_corporate_actions(symbol, start=start, end=end)
    if actions.empty:
        return df

    result = df.copy()

    # Build cumulative backward adjustment factor per date
    # Most recent data = unadjusted (factor=1.0). Older data gets divided.
    adj = np.ones(len(result))
    for _, action in actions.iterrows():
        ex_ts = pd.Timestamp(action["ex_date"])
        factor = action["factor"]
        # All dates BEFORE ex_date get divided by factor
        mask = result.index < ex_ts
        adj[mask] /= factor

    for col in ("Open", "High", "Low", "Close"):
        if col in result.columns:
            result[col] = result[col] * adj

    if "Volume" in result.columns:
        # Volume scales inversely (more shares after split)
        result["Volume"] = (result["Volume"] / adj).astype(int)

    return result


# ── Convenience: adjusted batch OHLCV ──────────────────────────

def fetch_adjusted_ohlcv(
    symbol: str,
    start: date = date(2012, 1, 1),
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch and adjust OHLCV for a single symbol.

    Combines fetch_ohlcv + adjust_ohlcv into one call.
    """
    raw = fetch_ohlcv(symbol, start=start, end=end)
    if raw.empty:
        return raw
    return adjust_ohlcv(raw, symbol, start=start, end=end)


def fetch_adjusted_ohlcv_batch(
    symbols: List[str],
    start: date = date(2012, 1, 1),
    end: Optional[date] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch and adjust OHLCV for multiple symbols.

    Combines fetch_ohlcv_batch + per-symbol adjust_ohlcv.
    """
    raw_batch = fetch_ohlcv_batch(symbols, start=start, end=end)
    results = {}
    for ticker, df in raw_batch.items():
        results[ticker] = adjust_ohlcv(df, ticker, start=start, end=end)
    return results


# ── CLI entry point ────────────────────────────────────────────

def main():
    """Command-line bulk download: ``python -m services.bhavcopy_fetcher``."""
    import argparse

    parser = argparse.ArgumentParser(description="NSE Bhavcopy bulk downloader")
    parser.add_argument("--start", default="2012-01-01",
                        help="Start date YYYY-MM-DD (default: 2012-01-01)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads (default: 4, max recommended: 5)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    workers = min(args.workers, 8)  # safety cap

    download_full_history(start=start, end=end, max_workers=workers)


if __name__ == "__main__":
    main()
