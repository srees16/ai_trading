"""
Utility Functions Module for Centurion Capital LLC.

Provides helper functions for CSV processing, ticker validation,
and data handling across the application.
"""

import logging
from typing import Dict, List, Optional, Tuple
from io import StringIO
import pandas as pd

logger = logging.getLogger(__name__)

# ── yfinance NSE symbol overrides ──────────────────────────────
# Yahoo Finance occasionally changes or delists Indian tickers.
# Map the NSE trading symbol to the correct yfinance symbol (without .NS).
# Add entries here when yfinance stops recognizing an NSE symbol.
YF_NSE_SYMBOL_MAP = {
    "TATAMOTORS": "TMCV",
}


def yf_nse_symbol(nse_symbol: str) -> str:
    """Convert an NSE trading symbol to its yfinance ticker (with .NS).

    Idempotent — already-suffixed tickers (``.NS`` / ``.BO``) are
    returned as-is (after applying any override mapping).

    Applies ``YF_NSE_SYMBOL_MAP`` overrides before appending ``.NS``.

    >>> yf_nse_symbol("TATAMOTORS")
    'TMCV.NS'
    >>> yf_nse_symbol("RELIANCE")
    'RELIANCE.NS'
    >>> yf_nse_symbol("RELIANCE.NS")
    'RELIANCE.NS'
    """
    upper = nse_symbol.upper()
    # Strip existing exchange suffix before lookup
    raw = upper.replace(".NS", "").replace(".BO", "")
    mapped = YF_NSE_SYMBOL_MAP.get(raw, raw)
    return f"{mapped}.NS"


def parse_ticker_csv(file_content: str) -> List[str]:
    """
    Parse CSV file content to extract ticker symbols.
    
    Supports various CSV formats:
    - Single column with header (Ticker, Symbol, Stock, etc.)
    - Single column without header
    - Multiple columns (extracts first column or column named 'ticker'/'symbol')
    
    Args:
        file_content: String content of the CSV file
        
    Returns:
        List of unique ticker symbols (uppercase, stripped)
    """
    try:
        # Read CSV
        import pandas as pd
        df = pd.read_csv(StringIO(file_content))
        
        # Try to find ticker column
        ticker_col = None
        
        # Look for common ticker column names
        common_names = ['ticker', 'symbol', 'stock', 'tickers', 'symbols', 'stocks', 'name']
        for col in df.columns:
            if col.lower().strip() in common_names:
                ticker_col = col
                break
        
        # If no ticker column found, use first column
        if ticker_col is None:
            ticker_col = df.columns[0]
        
        # Extract tickers
        tickers = df[ticker_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
        
        # Filter out empty strings and invalid tickers
        # Allow exchange suffixes like .NS, .BO, .L (up to 15 chars total)
        tickers = [t for t in tickers if t and 1 <= len(t) <= 15]
        
        return tickers
    
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        # Try simple line-by-line parsing as fallback
        try:
            lines = file_content.strip().split('\n')
            tickers = []
            for line in lines:
                # Split by comma and take first item
                ticker = line.split(',')[0].strip().upper()
                if ticker and len(ticker) <= 15 and ticker.replace('.', '').replace('-', '').isalnum():
                    tickers.append(ticker)
            return list(set(tickers))
        except:
            return []


def validate_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate ticker symbols.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Tuple of (valid_tickers, invalid_tickers)
    """
    valid = []
    invalid = []
    
    for ticker in tickers:
        # Strip exchange suffix before length check (.NS, .BO, .L, etc.)
        base = ticker.split('.')[0] if '.' in ticker else ticker
        if base and 1 <= len(base) <= 15 and base.replace('-', '').isalnum():
            valid.append(ticker)
        else:
            invalid.append(ticker)
    
    return valid, invalid


def create_sample_csv() -> str:
    """
    Create a sample CSV content for user reference.
    
    Returns:
        Sample CSV string
    """
    sample = """Ticker,Company
AAPL,Apple Inc.
MSFT,Microsoft Corporation
GOOGL,Alphabet Inc.
AMZN,Amazon.com Inc.
TSLA,Tesla Inc.
NVDA,NVIDIA Corporation
META,Meta Platforms Inc.
JPM,JPMorgan Chase
V,Visa Inc.
WMT,Walmart Inc."""
    
    return sample


# ── Indian stock OHLCV download with Bhavcopy fallback ────────

_IND_SUFFIXES = (".NS", ".BO")


def _is_indian_ticker(ticker: str) -> bool:
    return ticker.upper().endswith(_IND_SUFFIXES)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate index, enforce canonical column names, sort by date, validate OHLC."""
    if df.empty:
        return df
    # Flatten MultiIndex columns from yfinance batch downloads
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Deduplicate on index (keep first occurrence)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    # Sort chronologically
    df = df.sort_index()

    # Gap D2: OHLC validation — ensure High >= max(Open,Close) and Low <= min(Open,Close)
    if all(c in df.columns for c in ("Open", "High", "Low", "Close")):
        max_oc = df[["Open", "Close"]].max(axis=1)
        min_oc = df[["Open", "Close"]].min(axis=1)
        bad_high = df["High"] < max_oc
        bad_low = df["Low"] > min_oc
        n_bad = int(bad_high.sum() + bad_low.sum())
        if n_bad > 0:
            logger.warning("OHLC validation: fixed %d bad bars", n_bad)
            df.loc[bad_high, "High"] = max_oc[bad_high]
            df.loc[bad_low, "Low"] = min_oc[bad_low]

    return df


def download_ind_ohlcv(
    ticker: str,
    *,
    period: str = "1y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download OHLCV for an Indian stock, Bhavcopy-first with yfinance fallback.

    For non-Indian tickers, delegates directly to yfinance.

    Parameters
    ----------
    ticker : str
        Symbol with or without ``.NS`` suffix.
    period : str
        yfinance period string (``"1y"``, ``"2y"``, etc.). Ignored if
        *start* is provided.
    start, end : str | None
        ISO date strings (``"2024-01-15"``).

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume (yfinance-compatible).
    """
    is_ind = _is_indian_ticker(ticker) or not any(c == "." for c in ticker)

    # For tickers that look Indian (no dot → NSE plain symbol), try Bhavcopy first
    if is_ind:
        df = _try_bhavcopy(ticker, period=period, start=start, end=end)
        if df is not None and not df.empty:
            return _normalize_ohlcv(df)

    # yfinance fallback
    df = _try_yfinance(ticker, period=period, start=start, end=end)
    if df is not None and not df.empty:
        return _normalize_ohlcv(df)

    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def download_us_ohlcv(
    ticker: str,
    *,
    period: str = "1y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download OHLCV for a US stock via yfinance (no .NS suffix).

    Parameters
    ----------
    ticker : str
        US ticker symbol (e.g. ``"AAPL"``, ``"MSFT"``).
    period : str
        yfinance period string (``"1y"``, ``"6mo"``, etc.).
    start, end : str | None
        ISO date strings.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume.
    """
    import yfinance as yf

    # US tickers: use as-is (no .NS suffix)
    sym = ticker.replace(".NS", "").replace(".BO", "")

    for attempt in range(2):
        try:
            df = yf.download(sym, period=period, start=start, end=end,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and not df.empty:
                return _normalize_ohlcv(df)
        except Exception as exc:
            exc_str = str(exc)
            if attempt == 0 and ("401" in exc_str or "Invalid Crumb" in exc_str):
                _clear_yfinance_crumb_cache()
                continue
            logger.debug("yfinance US failed for %s: %s", ticker, exc)
            break

    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def download_ind_ohlcv_batch(
    tickers: List[str],
    *,
    period: str = "1y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Batch OHLCV download for Indian stocks: Bhavcopy first, yfinance gap-fill."""
    from datetime import date as _date, timedelta

    if end:
        from datetime import datetime as _dt
        end_dt = _dt.strptime(end, "%Y-%m-%d").date()
    else:
        end_dt = _date.today()
    if start:
        from datetime import datetime as _dt
        start_dt = _dt.strptime(start, "%Y-%m-%d").date()
    else:
        _period_days = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825, "max": 3650,
        }
        start_dt = end_dt - timedelta(days=_period_days.get(period, 365))

    results: Dict[str, pd.DataFrame] = {}

    # 1. Try Bhavcopy for all tickers
    try:
        from services.bhavcopy_fetcher import fetch_ohlcv_batch
        bhav = fetch_ohlcv_batch(tickers, start=start_dt, end=end_dt)
        results.update(bhav)
    except Exception as exc:
        logger.debug("Bhavcopy batch failed: %s", exc)

    # 2. yfinance gap-fill for missed tickers
    missed = [t for t in tickers if t not in results]
    if missed:
        import yfinance as yf
        for t in missed:
            try:
                ns = t if "." in t else f"{t}.NS"
                df = yf.download(ns, period=period, start=start, end=end,
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    results[t] = df
            except Exception:
                pass

    # Normalize all DataFrames: dedup index, flatten columns, sort
    for t in list(results):
        results[t] = _normalize_ohlcv(results[t])

    return results


def _try_bhavcopy(
    ticker: str,
    *,
    period: str = "1y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Attempt to fetch OHLCV from Bhavcopy."""
    try:
        from datetime import date as _date, timedelta
        from services.bhavcopy_fetcher import fetch_ohlcv

        if end:
            from datetime import datetime as _dt
            end_dt = _dt.strptime(end, "%Y-%m-%d").date()
        else:
            end_dt = _date.today()
        if start:
            from datetime import datetime as _dt
            start_dt = _dt.strptime(start, "%Y-%m-%d").date()
        else:
            _period_days = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
                "1y": 365, "2y": 730, "5y": 1825, "max": 3650,
            }
            start_dt = end_dt - timedelta(days=_period_days.get(period, 365))

        return fetch_ohlcv(ticker, start=start_dt, end=end_dt)
    except Exception as exc:
        logger.debug("Bhavcopy fallback failed for %s: %s", ticker, exc)
        return None


def _try_yfinance(
    ticker: str,
    *,
    period: str = "1y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Attempt to fetch OHLCV from yfinance with retry on crumb/auth errors."""
    import yfinance as yf
    ns = ticker if "." in ticker else f"{ticker}.NS"

    for attempt in range(2):
        try:
            df = yf.download(ns, period=period, start=start, end=end,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if not df.empty else None
        except Exception as exc:
            exc_str = str(exc)
            # Retry once on 401 Invalid Crumb — clear yfinance cookie cache
            if attempt == 0 and ("401" in exc_str or "Invalid Crumb" in exc_str):
                logger.info("yfinance crumb expired for %s — clearing cache and retrying", ticker)
                _clear_yfinance_crumb_cache()
                continue
            logger.debug("yfinance failed for %s: %s", ticker, exc)
            return None
    return None


def _clear_yfinance_crumb_cache():
    """Clear yfinance's internal crumb/cookie cache to force re-authentication."""
    try:
        import yfinance as yf
        # yfinance ≥0.2.31 stores crumbs in a module-level cache
        if hasattr(yf, 'utils') and hasattr(yf.utils, 'get_all_by_isin'):
            pass  # older version, no cache to clear
        # Clear the session-level cookie jar used by yfinance
        if hasattr(yf, 'cache') and hasattr(yf.cache, 'get'):
            yf.cache.clear()
        # More reliable: clear the shared session's cookies
        if hasattr(yf, 'shared') and hasattr(yf.shared, '_requests_session'):
            s = yf.shared._requests_session
            if s and hasattr(s, 'cookies'):
                s.cookies.clear()
        # yfinance ≥0.2.36 — reset the crumb/cookie singleton
        if hasattr(yf, 'data') and hasattr(yf.data, 'YfData'):
            if hasattr(yf.data.YfData, '_crumb'):
                yf.data.YfData._crumb = None
            if hasattr(yf.data.YfData, '_cookie'):
                yf.data.YfData._cookie = None
    except Exception:
        pass  # best-effort cleanup


# ═══════════════════════════════════════════════════════════════
# Batch OHLCV Download — optimised for large universes (500-3000+)
# ═══════════════════════════════════════════════════════════════

def download_ohlcv_batch_parallel(
    tickers: List[str],
    *,
    market: str = "IND",
    period: str = "1y",
    start: Optional[str] = None,
    end: Optional[str] = None,
    batch_size: int = 50,
    max_workers: int = 8,
    progress_callback=None,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for a large list of tickers using yfinance batch
    mode with multi-threaded fallback.

    For IND: tries Bhavcopy first, then yfinance in batches of
    ``batch_size`` tickers per ``yf.download()`` call.

    For US: uses yfinance batch download directly (no suffix).

    Parameters
    ----------
    tickers : list[str]
        Stock symbols (plain, no suffix).
    market : str
        ``"IND"`` or ``"US"``.
    period, start, end : str
        Date range for yfinance.
    batch_size : int
        Number of tickers per yfinance batch call.
    max_workers : int
        Max threads for parallel batch processing.
    progress_callback : callable | None
        Called with (completed, total) for progress reporting.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping symbol -> OHLCV DataFrame.
    """
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from config import Config
        batch_size = getattr(Config, "OHLCV_DOWNLOAD_BATCH_SIZE", batch_size)
        max_workers = getattr(Config, "PIPELINE_MAX_WORKERS", max_workers)
    except Exception:
        pass

    results: Dict[str, pd.DataFrame] = {}

    # For IND: try Bhavcopy first (fast, no rate limit)
    remaining = list(tickers)
    if market.upper() == "IND":
        try:
            from services.bhavcopy_fetcher import fetch_ohlcv_batch
            from datetime import date as _date, timedelta, datetime as _dt

            end_dt = _dt.strptime(end, "%Y-%m-%d").date() if end else _date.today()
            _period_days = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
                "1y": 365, "2y": 730, "5y": 1825, "max": 3650,
            }
            start_dt = (
                _dt.strptime(start, "%Y-%m-%d").date()
                if start
                else end_dt - timedelta(days=_period_days.get(period, 365))
            )
            bhav = fetch_ohlcv_batch(tickers, start=start_dt, end=end_dt)
            for sym, df in bhav.items():
                results[sym] = _normalize_ohlcv(df)
            remaining = [t for t in tickers if t not in results]
            logger.info("Bhavcopy: %d/%d tickers, %d remaining for yfinance",
                        len(results), len(tickers), len(remaining))
        except Exception as exc:
            logger.debug("Bhavcopy batch failed: %s", exc)

    if not remaining:
        if progress_callback:
            progress_callback(len(tickers), len(tickers))
        return results

    # Split remaining into batches for yfinance
    suffix = ".NS" if market.upper() == "IND" else ""
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    def _download_batch(batch: List[str]) -> Dict[str, pd.DataFrame]:
        """Download a single batch of tickers via yfinance."""
        yf_syms = [f"{t}{suffix}" for t in batch]
        yf_str = " ".join(yf_syms)
        batch_result: Dict[str, pd.DataFrame] = {}
        try:
            raw = yf.download(
                yf_str, period=period, start=start, end=end,
                progress=False, auto_adjust=True, threads=True,
                group_by="ticker",
            )
            if raw is None or raw.empty:
                return batch_result

            if len(batch) == 1:
                # Single ticker — yfinance returns flat columns
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                if not raw.empty:
                    batch_result[batch[0]] = _normalize_ohlcv(raw)
            else:
                # Multi-ticker — yfinance returns MultiIndex columns (ticker, field)
                for plain, yf_sym in zip(batch, yf_syms):
                    try:
                        if yf_sym in raw.columns.get_level_values(0):
                            df = raw[yf_sym].copy()
                        elif plain in raw.columns.get_level_values(0):
                            df = raw[plain].copy()
                        else:
                            continue
                        df = df.dropna(how="all")
                        if not df.empty and len(df) >= 5:
                            batch_result[plain] = _normalize_ohlcv(df)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("yfinance batch download failed (%d tickers): %s",
                           len(batch), exc)
        return batch_result

    # Run batches in parallel threads
    completed_count = len(results)
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
        futures = {
            executor.submit(_download_batch, batch): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                batch_results = future.result()
                results.update(batch_results)
                completed_count += len(batch)
                if progress_callback:
                    progress_callback(min(completed_count, total), total)
            except Exception as exc:
                logger.warning("Batch download exception: %s", exc)
                completed_count += len(batch)

    logger.info(
        "Batch OHLCV download complete: %d/%d tickers (%s, batches=%d, workers=%d)",
        len(results), len(tickers), market, len(batches), max_workers,
    )
    return results
