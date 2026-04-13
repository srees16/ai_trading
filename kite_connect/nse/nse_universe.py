"""
NSE Universe Downloader.

Downloads equity symbols from NSE indices across Broad Market,
Sectoral, and Strategy & Thematic categories.  Three universe tiers
are provided:

1. **DEFAULT**  – NIFTY-50 + NIFTY-NEXT-50 (~100 stocks)
2. **NIFTY500** – NIFTY-500 constituents (~500 stocks)
3. **BROAD**    – Union of all NSE equity indices (~800-1200 unique
   stocks) covering Broad Market, Sectoral, and Strategy/Thematic
   indices.  This is the recommended tier for high-conviction
   stock picking across the full Indian market.

Data is fetched from NSE archives CSV endpoints and de-duplicated.
Falls back to Kite instruments or hardcoded lists when downloads fail.
"""

import io
import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# NSE Index Registry — Broad Market, Sectoral, Strategy & Thematic
# ═══════════════════════════════════════════════════════════════
_NSE_ARCHIVE_BASE = "https://archives.nseindia.com/content/indices"

# (csv_filename, label) — each is fetched from _NSE_ARCHIVE_BASE/csv_filename
# NSE CSV naming: ind_<indexname>list.csv
NSE_INDEX_REGISTRY: Dict[str, List[Tuple[str, str]]] = {
    # ── Broad Market Indices (21 — full niftyindices.com/broad-based-indices) ──
    "broad_market": [
        ("ind_nifty50list.csv",                      "NIFTY 50"),
        ("ind_niftynext50list.csv",                  "NIFTY Next 50"),
        ("ind_nifty100list.csv",                     "NIFTY 100"),
        ("ind_nifty200list.csv",                     "NIFTY 200"),
        ("ind_nifty500list.csv",                     "NIFTY 500"),
        ("ind_niftytotalmarket_list.csv",            "NIFTY Total Market"),
        ("ind_nifty500multicap502525list.csv",       "NIFTY500 Multicap 50:25:25"),
        ("ind_nifty500largemidsmallecwlist.csv",     "NIFTY500 LargeMidSmall Equal-Cap"),
        ("ind_niftymidcap50list.csv",                "NIFTY Midcap 50"),
        ("ind_niftymidcap100list.csv",               "NIFTY Midcap 100"),
        ("ind_niftymidcap150list.csv",               "NIFTY Midcap 150"),
        ("ind_niftymidcapselect_list.csv",           "NIFTY Midcap Select"),
        ("ind_niftysmallcap50list.csv",              "NIFTY Smallcap 50"),
        ("ind_niftysmallcap100list.csv",             "NIFTY Smallcap 100"),
        ("ind_niftysmallcap250list.csv",             "NIFTY Smallcap 250"),
        ("ind_niftysmallcap500list.csv",             "NIFTY Smallcap 500"),  # CSV 404; JSON API fallback
        ("ind_niftymicrocap250_list.csv",            "NIFTY Microcap 250"),
        ("ind_niftylargemidcap250list.csv",          "NIFTY LargeMidcap 250"),
        ("ind_niftymidsmallcap400list.csv",          "NIFTY MidSmallcap 400"),
        ("ind_niftyindiafpi150list.csv",             "NIFTY India FPI 150"),  # CSV 404; JSON API fallback
    ],
    # ── Sectoral Indices (25 — full niftyindices.com/sectoral-indices) ────────
    "sectoral": [
        ("ind_niftyautolist.csv",                    "NIFTY Auto"),
        ("ind_niftybanklist.csv",                    "NIFTY Bank"),
        ("ind_niftycementlist.csv",                  "NIFTY Cement"),  # CSV 404; JSON API fallback
        ("ind_niftychemicalslist.csv",               "NIFTY Chemicals"),  # CSV 404; JSON API fallback
        ("ind_niftyfinancialservices25_50list.csv",  "NIFTY Financial Services"),
        ("ind_niftyfinservexbanklist.csv",           "NIFTY Financial Services Ex Bank"),  # CSV 404; JSON API fallback
        ("ind_niftyfmcglist.csv",                    "NIFTY FMCG"),
        ("ind_niftyhealthcarelist.csv",              "NIFTY Healthcare"),
        ("ind_niftyitlist.csv",                      "NIFTY IT"),
        ("ind_niftymedialist.csv",                   "NIFTY Media"),
        ("ind_niftymetallist.csv",                   "NIFTY Metal"),
        ("ind_niftypharmalist.csv",                  "NIFTY Pharma"),
        ("ind_niftypvtbanklist.csv",                 "NIFTY Private Bank"),  # CSV 404; JSON API fallback
        ("ind_niftypsubanklist.csv",                 "NIFTY PSU Bank"),
        ("ind_niftyrealtylist.csv",                  "NIFTY Realty"),
        ("ind_niftyreitsrealtylist.csv",             "NIFTY REITs & Realty"),  # CSV 404; JSON API fallback
        ("ind_niftyconsumerdurableslist.csv",        "NIFTY Consumer Durables"),
        ("ind_niftyoilgaslist.csv",                  "NIFTY Oil and Gas"),
        ("ind_niftyenergylist.csv",                  "NIFTY Energy"),
        ("ind_nifty500healthcarelist.csv",           "NIFTY500 Healthcare"),  # CSV 404; JSON API fallback
        ("ind_niftymidsmallfinservlist.csv",          "NIFTY MidSmall Financial Services"),  # CSV 404; JSON API fallback
        ("ind_niftymidsmallhealthcare_list.csv",     "NIFTY MidSmall Healthcare"),
        ("ind_niftymidsmallit_telecomlist.csv",      "NIFTY MidSmall IT & Telecom"),  # CSV 404; JSON API fallback
        ("ind_niftyindiadefence_list.csv",           "NIFTY India Defence"),
        ("ind_niftyindiadigital_list.csv",           "NIFTY India Digital"),
    ],
    # ── Strategy & Thematic Indices ───────────────────────────
    "strategy_thematic": [
        ("ind_niftycommoditieslist.csv",             "NIFTY Commodities"),
        ("ind_niftyinfralist.csv",                   "NIFTY Infrastructure"),
        ("ind_niftymnclist.csv",                     "NIFTY MNC"),
        ("ind_niftycpselist.csv",                    "NIFTY CPSE"),
        ("ind_niftypselist.csv",                     "NIFTY PSE"),
        ("ind_niftyconsumptionlist.csv",             "NIFTY India Consumption"),
        ("ind_niftymidcap150quality50list.csv",      "NIFTY Midcap150 Quality 50"),
    ],
}

# ── JSON API fallback for indices whose CSV endpoint returns 404 ──
# NSE migrated some constituent lists to the v2 JSON API.
# Map: label used in NSE_INDEX_REGISTRY → NSE API index name.
_NSE_API_BASE = "https://www.nseindia.com/api/equity-stockIndices"
_NSE_API_INDEX_MAP: Dict[str, str] = {
    "NIFTY500 Multicap 50:25:25":        "NIFTY500 MULTICAP 50:25:25",
    "NIFTY500 LargeMidSmall Equal-Cap":  "NIFTY500 LARGEMIDSMALL EQUAL-CAP WEIGHTED",
    "NIFTY Smallcap 500":                "NIFTY SMALLCAP 500",  # Deprecated by NSE; covered by TotalMarket+Smallcap250
    "NIFTY India FPI 150":               "NIFTY INDIA FPI 150",
    "NIFTY Cement":                      "NIFTY CEMENT",  # Deprecated by NSE; stocks in broader sectoral indices
    "NIFTY Chemicals":                   "NIFTY CHEMICALS",
    "NIFTY Financial Services Ex Bank":  "NIFTY FINANCIAL SERVICES EX-BANK",
    "NIFTY Private Bank":                "NIFTY PRIVATE BANK",
    "NIFTY REITs & Realty":              "NIFTY REALTY",
    "NIFTY500 Healthcare":               "NIFTY500 HEALTHCARE",
    "NIFTY MidSmall Financial Services": "NIFTY MIDSMALL FINANCIAL SERVICES",
    "NIFTY MidSmall IT & Telecom":       "NIFTY MIDSMALL IT & TELECOM",
}


def _fetch_via_json_api(label: str, session) -> List[str]:
    """Fallback: fetch index constituents via NSE v2 JSON API."""
    api_name = _NSE_API_INDEX_MAP.get(label)
    if not api_name:
        return []
    import requests as _req
    try:
        url = f"{_NSE_API_BASE}?index={_req.utils.quote(api_name)}"
        resp = session.get(url, headers=_NSE_HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        stocks = data.get("data", [])
        symbols = [s["symbol"] for s in stocks
                    if s.get("symbol") and s["symbol"] != api_name]
        if symbols:
            logger.info("  %s: %d symbols (JSON API fallback)", label, len(symbols))
        return symbols
    except Exception as exc:
        logger.debug("  %s JSON API failed: %s", label, exc)
        return []


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
        logger.warning("%s download failed: %s", label, exc)
        return []


def _fetch_index_category(category: str, session=None) -> List[str]:
    """Fetch all symbols from a single index category (broad_market / sectoral / strategy_thematic)."""
    entries = NSE_INDEX_REGISTRY.get(category, [])
    if not entries:
        return []

    import requests

    if session is None:
        session = requests.Session()
        try:
            session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)
        except Exception:
            pass

    all_syms: set = set()
    for csv_name, label in entries:
        url = f"{_NSE_ARCHIVE_BASE}/{csv_name}"
        try:
            resp = session.get(url, headers=_NSE_HEADERS, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            col = "Symbol" if "Symbol" in df.columns else df.columns[2]
            syms = df[col].dropna().str.strip().unique().tolist()
            all_syms.update(syms)
            logger.debug("  %s: %d symbols", label, len(syms))
        except Exception as exc:
            # CSV failed — try JSON API fallback
            api_syms = _fetch_via_json_api(label, session)
            if api_syms:
                all_syms.update(api_syms)
            else:
                logger.warning("  %s download failed (%s): %s", label, csv_name, exc)
        # Small delay to avoid rate limiting from NSE
        time.sleep(0.3)

    return sorted(all_syms)


def get_nse_broad_universe() -> List[str]:
    """
    Return the FULL NSE universe from all index categories:
    Broad Market + Sectoral + Strategy & Thematic.

    This yields ~800-1200 unique stocks — the widest possible
    coverage for high-conviction stock picking.

    Returns
    -------
    list[str]
        Plain NSE symbols (no ``.NS`` suffix), deduplicated & sorted.
    """
    import requests

    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)
    except Exception:
        pass

    all_symbols: set = set()
    category_counts = {}

    for category in ("broad_market", "sectoral", "strategy_thematic"):
        syms = _fetch_index_category(category, session=session)
        category_counts[category] = len(syms)
        all_symbols.update(syms)

    result = sorted(all_symbols)
    logger.info(
        "Broad NSE universe: %d unique symbols "
        "(broad_market=%d, sectoral=%d, strategy_thematic=%d)",
        len(result),
        category_counts.get("broad_market", 0),
        category_counts.get("sectoral", 0),
        category_counts.get("strategy_thematic", 0),
    )
    return result


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


def get_nse_universe(kite=None, tier: Optional[str] = None) -> List[str]:
    """
    Return the NSE stock universe based on the configured tier.

    Parameters
    ----------
    kite : KiteConnect | None
        If provided and tier is not set, can be used for Kite instruments.
    tier : str | None
        ``"DEFAULT"`` (~100), ``"NIFTY500"`` (~500), ``"BROAD"`` (~800-1200).
        If ``None``, reads from ``Config.NSE_UNIVERSE_TIER``.

    Returns
    -------
    list[str]
        Plain NSE symbols (no ``.NS`` suffix).
    """
    if tier is None:
        try:
            from config import Config
            tier = getattr(Config, "NSE_UNIVERSE_TIER", "BROAD")
        except Exception:
            tier = "BROAD"

    tier = tier.upper().strip()

    if tier == "DEFAULT":
        return get_nse_default_tickers()
    elif tier == "NIFTY500":
        syms = fetch_nse_symbols_nifty500()
        if syms:
            return syms
        logger.warning("NIFTY500 fetch failed, falling back to DEFAULT")
        return get_nse_default_tickers()
    else:  # BROAD (default)
        syms = get_nse_broad_universe()
        if len(syms) >= 100:
            return syms
        # Fallback chain: NIFTY500 → DEFAULT
        logger.warning("Broad universe too small (%d), trying NIFTY500", len(syms))
        syms = fetch_nse_symbols_nifty500()
        if syms:
            return syms
        return get_nse_default_tickers()
