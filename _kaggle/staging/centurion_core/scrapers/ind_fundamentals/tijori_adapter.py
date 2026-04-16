"""
Tijori Finance adapter — Indian fundamental data fallback.

When yfinance returns ``None`` for key fundamental fields on .NS / .BO
tickers (PEG, ROE, EPS, FCF — happens ~70% of the time), this adapter
fetches the data from Tijori Finance's public pages.

Long-term these should migrate to BSE XBRL filings, but Tijori is
the quickest P0 fix with the broadest coverage of Indian companies.

Usage::

    adapter = TijoriAdapter()
    data = await adapter.fetch_fundamentals("RELIANCE")
    # data = {"trailingEps": 95.2, "returnOnEquity": 0.09, ...}
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Optional

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.tijorifinance.com/company/{slug}/financials"
_RATIOS_URL = "https://www.tijorifinance.com/company/{slug}/ratios"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
}

# Map NSE symbol → Tijori slug (lowercase, hyphens).
# Most tickers just need lowercasing; special cases handled in _to_slug.
_SLUG_OVERRIDES = {
    "HDFCBANK": "hdfc-bank",
    "ICICIBANK": "icici-bank",
    "KOTAKBANK": "kotak-mahindra-bank",
    "SBIN": "state-bank-of-india",
    "TATAMOTORS": "tata-motors",
    "TATASTEEL": "tata-steel",
    "BAJFINANCE": "bajaj-finance",
    "BAJAJFINSV": "bajaj-finserv",
    "HINDUNILVR": "hindustan-unilever",
    "M&M": "mahindra-and-mahindra",
    "LT": "larsen-and-toubro",
    "POWERGRID": "power-grid-corporation",
}


def _to_slug(nse_symbol: str) -> str:
    """Convert an NSE symbol to a Tijori URL slug."""
    clean = nse_symbol.upper().replace(".NS", "").replace(".BO", "")
    if clean in _SLUG_OVERRIDES:
        return _SLUG_OVERRIDES[clean]
    return clean.lower().replace("&", "-and-")


class TijoriAdapter:
    """
    Fetches fundamental data from Tijori Finance for Indian equities.

    Results are cached for 6 hours per ticker to avoid hammering
    the website.
    """

    _cache: Dict[str, dict] = {}
    _cache_ts: Dict[str, datetime] = {}
    _CACHE_TTL = timedelta(hours=6)

    async def fetch_fundamentals(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Fetch fundamental fields for *ticker*.

        Returns a dict with yfinance-compatible keys so it can be
        merged directly::

            info = stock.info
            tijori = await adapter.fetch_fundamentals(ticker)
            for k, v in tijori.items():
                if info.get(k) is None and v is not None:
                    info[k] = v

        Keys returned (when available):
            trailingEps, returnOnEquity, pegRatio, freeCashflow,
            revenueGrowth, earningsGrowth, debtToEquity, currentRatio,
            bookValue, dividendYield
        """
        clean = ticker.upper().replace(".NS", "").replace(".BO", "")
        now = datetime.utcnow()

        if (
            clean in self._cache
            and clean in self._cache_ts
            and (now - self._cache_ts[clean]) < self._CACHE_TTL
        ):
            return self._cache[clean]

        result: Dict[str, Optional[float]] = {}
        slug = _to_slug(ticker)

        try:
            result = await self._scrape_ratios(slug)
        except Exception as exc:
            logger.warning("TijoriAdapter: scrape failed for %s — %s", slug, exc)

        self._cache[clean] = result
        self._cache_ts[clean] = now

        logger.info(
            "TijoriAdapter: fetched %d fields for %s", len(result), clean,
        )
        return result

    async def _scrape_ratios(self, slug: str) -> Dict[str, Optional[float]]:
        """Scrape the Tijori ratios page for key financial metrics."""
        url = _RATIOS_URL.format(slug=slug)
        timeout = aiohttp.ClientTimeout(total=15)
        result: Dict[str, Optional[float]] = {}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=_HEADERS, ssl=False) as resp:
                if resp.status != 200:
                    logger.debug("TijoriAdapter: HTTP %d for %s", resp.status, slug)
                    return result
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Tijori renders ratios in <table> rows: <td>Label</td><td>Value</td>
        _FIELD_MAP = {
            "eps": "trailingEps",
            "earning per share": "trailingEps",
            "return on equity": "returnOnEquity",
            "roe": "returnOnEquity",
            "peg ratio": "pegRatio",
            "peg": "pegRatio",
            "free cash flow": "freeCashflow",
            "fcf": "freeCashflow",
            "revenue growth": "revenueGrowth",
            "earnings growth": "earningsGrowth",
            "debt to equity": "debtToEquity",
            "d/e ratio": "debtToEquity",
            "current ratio": "currentRatio",
            "book value": "bookValue",
            "dividend yield": "dividendYield",
        }

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            label = cells[0].get_text(strip=True).lower()
            value_text = cells[1].get_text(strip=True)

            for pattern, yf_key in _FIELD_MAP.items():
                if pattern in label:
                    val = self._parse_number(value_text)
                    if val is not None:
                        # Convert percentage fields to decimal
                        if yf_key in ("returnOnEquity", "revenueGrowth",
                                       "earningsGrowth", "dividendYield"):
                            if abs(val) > 1:
                                val = val / 100.0
                        result[yf_key] = val
                    break

        return result

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        """Parse a number from Tijori's display format (handles ₹, %, Cr, etc)."""
        if not text:
            return None
        # Remove currency symbols, commas, percentage, Cr/Lakh suffixes
        cleaned = re.sub(r"[₹,%]", "", text).strip()
        multiplier = 1.0
        if "cr" in cleaned.lower():
            cleaned = re.sub(r"(?i)\s*cr\.?", "", cleaned)
            multiplier = 1e7  # 1 crore = 10 million
        elif "lakh" in cleaned.lower() or "lac" in cleaned.lower():
            cleaned = re.sub(r"(?i)\s*(lakh|lac)\.?", "", cleaned)
            multiplier = 1e5
        cleaned = cleaned.replace(",", "").strip()
        try:
            return float(cleaned) * multiplier
        except (ValueError, TypeError):
            return None


async def enrich_yfinance_info(ticker: str, info: dict) -> dict:
    """
    Fill gaps in a yfinance ``info`` dict using Tijori data.

    Modifies *info* in-place and returns it for convenience.
    Only overwrites fields that are ``None`` in the original.
    """
    if not (ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")):
        return info

    adapter = TijoriAdapter()
    tijori = await adapter.fetch_fundamentals(ticker)

    for key, val in tijori.items():
        if info.get(key) is None and val is not None:
            info[key] = val
            logger.debug("TijoriAdapter: filled %s=%s for %s", key, val, ticker)

    return info
