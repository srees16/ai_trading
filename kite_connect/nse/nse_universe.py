"""
NSE Universe Downloader.

Downloads all equity symbols listed on NSE via yfinance /
Kite Connect instruments dump and returns them as a list ready
for screening.  Two strategies are provided:

1. **Kite instruments** (preferred when authenticated) — downloads the
   full instrument CSV from Zerodha, filters for ``NSE`` exchange and
   ``EQ`` segment.
2. **yfinance fallback** — uses a lightweight HTTP request to the NSE
   market-status API + NIFTY-500 constituents to seed a broad universe.
"""

import io
import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1.  Kite Connect instruments (broadest coverage)
# ═══════════════════════════════════════════════════════════════

def fetch_nse_symbols_from_kite(kite) -> List[str]:
    """
    Download the full Zerodha instrument list and return NSE equity
    trading symbols.

    Parameters
    ----------
    kite : KiteConnect
        Authenticated KiteConnect instance.

    Returns
    -------
    list[str]
        Trading symbols such as ``["RELIANCE", "TCS", ...]``
    """
    try:
        instruments = kite.instruments("NSE")
        df = pd.DataFrame(instruments)
        # Keep only equity segment (exclude ETFs, MFs, debt, etc.)
        eq_df = df[df["segment"] == "NSE"]
        eq_df = eq_df[eq_df["instrument_type"] == "EQ"]
        symbols = sorted(eq_df["tradingsymbol"].unique().tolist())
        logger.info("Fetched %d NSE equity symbols from Kite instruments", len(symbols))
        return symbols
    except Exception as exc:
        logger.error("Failed to fetch Kite instruments: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════
# 2.  yfinance fallback – broad NIFTY-500 universe
# ═══════════════════════════════════════════════════════════════

_NIFTY500_URL = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
)
_NIFTY50_URL = (
    "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
)
_NIFTY_NEXT50_URL = (
    "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv"
)

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_nse_symbols_nifty500() -> List[str]:
    """
    Download the NIFTY-500 constituent list from NSE archives.

    Returns
    -------
    list[str]
        NSE trading symbols (no suffix).
    """
    import requests  # deferred — keep module import-light

    try:
        session = requests.Session()
        # Pre-visit NSE homepage to obtain cookies
        session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)
        resp = session.get(_NIFTY500_URL, headers=_NSE_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        symbols = sorted(df[col].dropna().str.strip().unique().tolist())
        logger.info("Fetched %d NIFTY-500 symbols from NSE archives", len(symbols))
        return symbols
    except Exception as exc:
        logger.error("NIFTY-500 download failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def fetch_nse_index_symbols(url: str, label: str) -> List[str]:
    """
    Download an NSE index constituent CSV and return symbols.
    """
    import requests

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)
        resp = session.get(url, headers=_NSE_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        symbols = sorted(df[col].dropna().str.strip().unique().tolist())
        logger.info("Fetched %d %s symbols from NSE archives", len(symbols), label)
        return symbols
    except Exception as exc:
        logger.error("%s download failed: %s", label, exc)
        return []


def get_nse_default_tickers() -> List[str]:
    """
    Return NIFTY-50 + NIFTY-NEXT-50 symbols (100 stocks).

    This is the **default** universe for IND stock analysis — much
    faster than the full NIFTY-500 while covering all blue-chip and
    large-cap names that have reliable data on yfinance.

    Falls back to hardcoded lists when the NSE download fails.

    Returns
    -------
    list[str]
        Plain NSE symbols (no ``.NS`` suffix), deduplicated & sorted.
    """
    symbols: set = set()

    # Try downloading Nifty 50
    n50 = fetch_nse_index_symbols(_NIFTY50_URL, "NIFTY-50")
    if n50:
        symbols.update(n50)
    else:
        from kite_connect.core.config import INDEX_CONSTITUENTS
        symbols.update(INDEX_CONSTITUENTS.get("NIFTY50", []))

    # Try downloading Nifty Next 50
    nn50 = fetch_nse_index_symbols(_NIFTY_NEXT50_URL, "NIFTY-NEXT-50")
    if nn50:
        symbols.update(nn50)
    else:
        from kite_connect.core.config import INDEX_CONSTITUENTS
        symbols.update(INDEX_CONSTITUENTS.get("NIFTY_NEXT50", []))

    result = sorted(symbols)
    logger.info("Default IND universe: %d symbols (Nifty50 + Next50)", len(result))
    return result


def get_nse_universe(kite=None) -> List[str]:
    """
    Return the broadest available list of NSE equity symbols.

    * If *kite* is provided and authenticated → full instruments list.
    * Otherwise → NIFTY-50 + NIFTY-NEXT-50 (≈100 stocks).

    Parameters
    ----------
    kite : KiteConnect | None
        Optionally supply an authenticated Kite session.

    Returns
    -------
    list[str]
        Plain NSE symbols (no ``.NS`` suffix).
    """
    if kite is not None:
        symbols = fetch_nse_symbols_from_kite(kite)
        if symbols:
            return symbols

    # Default: Nifty50 + Nifty Next50 (≈100 stocks)
    return get_nse_default_tickers()
