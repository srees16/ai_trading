"""
RBI / NSDL macro-economic data scraper.

Fetches India-specific macro indicators that are NOT available via
yfinance:
  • RBI Repo Rate        — from RBI website / manual fallback
  • NSDL FPI daily flows — reuses FIIDIIFlows for institutional data
  • INR/USD exchange rate — yfinance USDINR=X (added to macro_indicators)
  • India CPI (YoY)      — from MOSPI / manual fallback

Provides a ``RBIMacroScraper`` with an async ``fetch()`` that returns
an ``IndiaMacroData`` dataclass.  Designed to complement the existing
``MacroIndicators`` module for India-market scoring.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_RBI_RATES_URL = "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx"
_RBI_KEY_RATES_URL = "https://rbi.org.in/Scripts/PublicationsView.aspx?id=22043"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,*/*",
}

# Manual fallback values (updated periodically)
_FALLBACK_REPO_RATE = 6.50       # RBI repo rate as of Jun 2025
_FALLBACK_REVERSE_REPO = 3.35    # standing deposit facility
_FALLBACK_CRR = 4.00             # cash reserve ratio
_FALLBACK_SLR = 18.00            # statutory liquidity ratio
_FALLBACK_CPI_YOY = 4.75         # India CPI YoY %


@dataclass
class IndiaMacroData:
    """India-specific macro indicators."""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # RBI policy rates
    repo_rate: Optional[float] = None
    reverse_repo_rate: Optional[float] = None
    crr: Optional[float] = None
    slr: Optional[float] = None

    # Inflation
    cpi_yoy: Optional[float] = None

    # FX
    inr_usd: Optional[float] = None

    # Source flags
    repo_source: str = "fallback"   # "rbi_live" or "fallback"
    cpi_source: str = "fallback"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "repo_rate": self.repo_rate,
            "reverse_repo_rate": self.reverse_repo_rate,
            "crr": self.crr,
            "slr": self.slr,
            "cpi_yoy": self.cpi_yoy,
            "inr_usd": self.inr_usd,
            "repo_source": self.repo_source,
            "cpi_source": self.cpi_source,
        }


class RBIMacroScraper:
    """
    Fetches India macro-economic data from RBI and related sources.

    Falls back to manually updated constants when live scraping fails,
    ensuring the pipeline is never blocked by upstream outages.

    Usage::

        scraper = RBIMacroScraper()
        data = await scraper.fetch()
    """

    _cached: Optional[IndiaMacroData] = None
    _cache_ts: Optional[datetime] = None
    _CACHE_TTL = timedelta(hours=6)  # RBI rates change infrequently

    async def fetch(self) -> IndiaMacroData:
        """Fetch India macro data (cached 6 hours)."""
        now = datetime.utcnow()
        if (
            self._cached is not None
            and self._cache_ts is not None
            and (now - self._cache_ts) < self._CACHE_TTL
        ):
            return self._cached

        data = IndiaMacroData(timestamp=now)

        # ── RBI Repo Rate ────────────────────────────────────────────
        try:
            await self._fetch_rbi_rates(data)
        except Exception as exc:
            logger.warning("RBIMacro: RBI rate scrape failed — %s", exc)

        # Fallback if live scrape didn't work
        if data.repo_rate is None:
            data.repo_rate = _FALLBACK_REPO_RATE
            data.reverse_repo_rate = _FALLBACK_REVERSE_REPO
            data.crr = _FALLBACK_CRR
            data.slr = _FALLBACK_SLR
            data.repo_source = "fallback"

        # ── CPI ──────────────────────────────────────────────────────
        try:
            await self._fetch_cpi(data)
        except Exception as exc:
            logger.warning("RBIMacro: CPI scrape failed — %s", exc)

        if data.cpi_yoy is None:
            data.cpi_yoy = _FALLBACK_CPI_YOY
            data.cpi_source = "fallback"

        # ── INR/USD via yfinance ─────────────────────────────────────
        try:
            import yfinance as yf
            fx = yf.Ticker("USDINR=X")
            hist = fx.history(period="2d")
            if not hist.empty:
                data.inr_usd = float(hist["Close"].iloc[-1])
        except Exception as exc:
            logger.debug("RBIMacro: INR/USD fetch failed — %s", exc)

        RBIMacroScraper._cached = data
        RBIMacroScraper._cache_ts = now

        logger.info(
            "RBIMacro: repo=%.2f%% (%s)  CPI=%.1f%% (%s)  INR/USD=%.2f",
            data.repo_rate or 0, data.repo_source,
            data.cpi_yoy or 0, data.cpi_source,
            data.inr_usd or 0,
        )
        return data

    async def _fetch_rbi_rates(self, data: IndiaMacroData) -> None:
        """Scrape RBI key policy rates from rbi.org.in."""
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                _RBI_KEY_RATES_URL, headers=_HEADERS, ssl=False,
            ) as resp:
                if resp.status != 200:
                    return
                html = await resp.text()

        # Look for "Repo Rate" followed by a percentage
        repo_match = re.search(
            r"(?i)repo\s*rate\s*[:\-–]?\s*([\d.]+)\s*%", html,
        )
        if repo_match:
            data.repo_rate = float(repo_match.group(1))
            data.repo_source = "rbi_live"

        reverse_match = re.search(
            r"(?i)(reverse\s*repo|standing\s*deposit\s*facility)\s*[:\-–]?\s*([\d.]+)\s*%",
            html,
        )
        if reverse_match:
            data.reverse_repo_rate = float(reverse_match.group(2))

        crr_match = re.search(r"(?i)CRR\s*[:\-–]?\s*([\d.]+)\s*%", html)
        if crr_match:
            data.crr = float(crr_match.group(1))

        slr_match = re.search(r"(?i)SLR\s*[:\-–]?\s*([\d.]+)\s*%", html)
        if slr_match:
            data.slr = float(slr_match.group(1))

    async def _fetch_cpi(self, data: IndiaMacroData) -> None:
        """
        Fetch India CPI YoY from MOSPI or tradingeconomics-style page.

        This is best-effort — CPI is released monthly with a lag, so
        the fallback value is often the most current anyway.
        """
        # Try a lightweight check via a known API
        url = "https://api.worldbank.org/v2/country/IND/indicator/FP.CPI.TOTL.ZG?format=json&per_page=1&date=2024"
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=_HEADERS, ssl=False) as resp:
                    if resp.status != 200:
                        return
                    payload = await resp.json(content_type=None)
            # World Bank returns [[metadata], [data_points]]
            if isinstance(payload, list) and len(payload) > 1:
                records = payload[1]
                if records and isinstance(records, list):
                    val = records[0].get("value")
                    if val is not None:
                        data.cpi_yoy = float(val)
                        data.cpi_source = "worldbank"
        except Exception:
            pass
