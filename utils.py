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
            return df

    # yfinance fallback
    df = _try_yfinance(ticker, period=period, start=start, end=end)
    if df is not None and not df.empty:
        return df

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
    """Attempt to fetch OHLCV from yfinance."""
    try:
        import yfinance as yf
        ns = ticker if "." in ticker else f"{ticker}.NS"
        df = yf.download(ns, period=period, start=start, end=end,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except Exception as exc:
        logger.debug("yfinance failed for %s: %s", ticker, exc)
        return None
