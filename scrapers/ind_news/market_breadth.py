"""
NSE Market Breadth scraper.

Fetches Advance / Decline / Unchanged counts from the NSE's
public JSON API to compute the Advance-Decline Ratio and
overall market breadth.

Primary source:
  • NSE market-status / market-turnover API
  • Fallback: NSE Bhavcopy CSV (end-of-day)

The A/D ratio is a key input to the India Fear & Greed index
(15% weight) and a useful breadth overlay for intraday
scoring.

Usage::

    from scrapers.ind_news.market_breadth import MarketBreadthScraper

    scraper = MarketBreadthScraper()
    snap = await scraper.fetch()
    # snap.advance_decline_ratio → 1.42
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_NSE_MARKET_STATUS_URL = "https://www.nseindia.com/api/marketStatus"
_NSE_MARKET_TURNOVER_URL = "https://www.nseindia.com/api/market-turnover"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


@dataclass
class BreadthSnapshot:
    """Market breadth data snapshot."""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    advances: Optional[int] = None
    declines: Optional[int] = None
    unchanged: Optional[int] = None
    total_traded: Optional[int] = None

    advance_decline_ratio: Optional[float] = None     # adv / dec
    breadth_pct: Optional[float] = None               # adv / total * 100

    # 52-week highs / lows (bonus breadth signals)
    new_highs: Optional[int] = None
    new_lows: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "advances": self.advances,
            "declines": self.declines,
            "unchanged": self.unchanged,
            "total_traded": self.total_traded,
            "advance_decline_ratio": self.advance_decline_ratio,
            "breadth_pct": self.breadth_pct,
            "new_highs": self.new_highs,
            "new_lows": self.new_lows,
        }


class MarketBreadthScraper:
    """
    Scrapes NSE advance/decline data.

    Cached for 10 minutes (breadth changes intraday).
    """

    _cached: Optional[BreadthSnapshot] = None
    _cache_ts: Optional[datetime] = None
    _CACHE_TTL = timedelta(minutes=10)

    async def fetch(self) -> BreadthSnapshot:
        """Fetch market breadth (cached 10 min)."""
        now = datetime.utcnow()
        if (
            self._cached is not None
            and self._cache_ts is not None
            and (now - self._cache_ts) < self._CACHE_TTL
        ):
            return self._cached

        snap = BreadthSnapshot(timestamp=now)

        # NSE requires cookies — we first hit the homepage to get them
        try:
            snap = await self._fetch_nse_api(snap)
        except Exception as exc:
            logger.warning("MarketBreadth: NSE API failed — %s", exc)

        # Fallback: try yfinance NSEI constituents OHLC approach
        if snap.advances is None:
            try:
                snap = await self._fallback_yfinance(snap)
            except Exception as exc:
                logger.debug("MarketBreadth: yfinance fallback failed — %s", exc)

        # Compute derived
        if snap.advances is not None and snap.declines is not None:
            if snap.declines > 0:
                snap.advance_decline_ratio = round(
                    snap.advances / snap.declines, 4
                )
            total = snap.advances + snap.declines + (snap.unchanged or 0)
            snap.total_traded = total
            if total > 0:
                snap.breadth_pct = round(snap.advances / total * 100, 1)

        MarketBreadthScraper._cached = snap
        MarketBreadthScraper._cache_ts = now

        logger.info(
            "MarketBreadth: Adv=%s Dec=%s A/D=%.2f",
            snap.advances, snap.declines, snap.advance_decline_ratio or 0,
        )
        return snap

    async def _fetch_nse_api(self, snap: BreadthSnapshot) -> BreadthSnapshot:
        """Fetch from NSE's marketStatus JSON API."""
        timeout = aiohttp.ClientTimeout(total=15)
        jar = aiohttp.CookieJar(unsafe=True)

        async with aiohttp.ClientSession(
            timeout=timeout, cookie_jar=jar,
        ) as session:
            # Step 1: Hit NSE homepage to get cookies
            async with session.get(
                "https://www.nseindia.com/", headers=_HEADERS, ssl=False,
            ) as homepage:
                if homepage.status != 200:
                    logger.debug("MarketBreadth: NSE homepage HTTP %d", homepage.status)

            # Step 2: Fetch market status with cookies
            async with session.get(
                _NSE_MARKET_STATUS_URL, headers=_HEADERS, ssl=False,
            ) as resp:
                if resp.status != 200:
                    logger.debug("MarketBreadth: marketStatus HTTP %d", resp.status)
                    return snap
                data = await resp.json(content_type=None)

        # NSE marketStatus response structure:
        # {"marketState": [{"market": "Capital Market", "marketStatus": "Open",
        #   "tradeDate": "...", "index": "NIFTY 50", ...}]}
        # OR the advance/decline is in a separate turnover endpoint.

        # Try to extract from marketState
        states = data.get("marketState", [])
        for state in states:
            if state.get("market") == "Capital Market":
                # Some NSE API versions include these directly
                adv = state.get("advances") or state.get("advance")
                dec = state.get("declines") or state.get("decline")
                unc = state.get("unchanged")

                if adv is not None:
                    snap.advances = int(adv)
                if dec is not None:
                    snap.declines = int(dec)
                if unc is not None:
                    snap.unchanged = int(unc)
                break

        return snap

    async def _fallback_yfinance(self, snap: BreadthSnapshot) -> BreadthSnapshot:
        """
        Fallback: Count advances/declines from a sample of Nifty-50
        constituents using yfinance.
        """
        import yfinance as yf

        # Use a representative sample of Nifty-50 stocks
        from kite_connect.core.config import INDEX_CONSTITUENTS

        nifty_stocks = INDEX_CONSTITUENTS.get("NIFTY50", [])[:50]
        if not nifty_stocks:
            return snap

        tickers = [f"{s}.NS" for s in nifty_stocks]
        try:
            data = yf.download(
                tickers, period="2d", progress=False, threads=True,
                group_by="ticker",
            )
        except Exception:
            return snap

        if data is None or data.empty:
            return snap

        advances = 0
        declines = 0
        unchanged = 0

        for t in tickers:
            try:
                if len(tickers) == 1:
                    closes = data["Close"].dropna()
                else:
                    closes = data[t]["Close"].dropna()
                if len(closes) < 2:
                    continue
                today = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                if today > prev:
                    advances += 1
                elif today < prev:
                    declines += 1
                else:
                    unchanged += 1
            except Exception:
                continue

        if advances + declines > 0:
            snap.advances = advances
            snap.declines = declines
            snap.unchanged = unchanged

        return snap
