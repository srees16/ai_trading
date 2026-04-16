"""
FII / DII daily flow scraper.

Scrapes NSDL FPI (Foreign Portfolio Investor) daily aggregate data
and derives DII flows from the NSE bulk-deal / activity reports.

Primary source:
  • NSDL FPI monitor — https://www.fpi.nsdl.co.in/web/Reports/Latest.aspx
  • Moneycontrol FII/DII page (fallback)

Exposes ``FIIDIIFlows`` class with a ``fetch()`` method that returns
a ``FlowSnapshot`` suitable for downstream consumption by the
India Fear & Greed index and the decision engine.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_MONEYCONTROL_FII_URL = (
    "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/data.json"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Referer": "https://www.moneycontrol.com/",
}

_NSDL_FPI_URL = "https://www.fpi.nsdl.co.in/web/Reports/Latest.aspx"


@dataclass
class FlowSnapshot:
    """Point-in-time FII / DII flow snapshot (all values in ₹ crore)."""

    date: datetime = field(default_factory=datetime.utcnow)

    fii_buy: Optional[float] = None
    fii_sell: Optional[float] = None
    fii_net: Optional[float] = None   # buy - sell

    dii_buy: Optional[float] = None
    dii_sell: Optional[float] = None
    dii_net: Optional[float] = None

    # Derived
    fii_dii_ratio: Optional[float] = None   # fii_net / dii_net (sign matters)
    flow_sentiment: Optional[float] = None  # -1 (heavy FII selling) … +1

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "fii_buy": self.fii_buy,
            "fii_sell": self.fii_sell,
            "fii_net": self.fii_net,
            "dii_buy": self.dii_buy,
            "dii_sell": self.dii_sell,
            "dii_net": self.dii_net,
            "fii_dii_ratio": self.fii_dii_ratio,
            "flow_sentiment": self.flow_sentiment,
        }


class FIIDIIFlows:
    """
    Fetches daily FII / DII flow data for the Indian equity market.

    Usage::

        flows = FIIDIIFlows()
        snap = await flows.fetch()
    """

    _cached: Optional[FlowSnapshot] = None
    _cache_ts: Optional[datetime] = None
    _CACHE_TTL = timedelta(minutes=30)

    async def fetch(self) -> FlowSnapshot:
        """Fetch latest FII/DII data (cached 30 min)."""
        now = datetime.utcnow()
        if (
            self._cached is not None
            and self._cache_ts is not None
            and (now - self._cache_ts) < self._CACHE_TTL
        ):
            return self._cached

        snap = FlowSnapshot(date=now)

        # Try Moneycontrol JSON endpoint first (most reliable)
        try:
            snap = await self._fetch_moneycontrol(snap)
        except Exception as exc:
            logger.warning("FIIDIIFlows: Moneycontrol fetch failed — %s", exc)

        # If Moneycontrol didn't work, try NSDL HTML scrape
        if snap.fii_net is None:
            try:
                snap = await self._fetch_nsdl_html(snap)
            except Exception as exc:
                logger.warning("FIIDIIFlows: NSDL fetch failed — %s", exc)

        # Compute derived metrics
        self._compute_sentiment(snap)

        FIIDIIFlows._cached = snap
        FIIDIIFlows._cache_ts = now

        logger.info(
            "FIIDIIFlows: FII_net=%.0f  DII_net=%.0f  sentiment=%.2f",
            snap.fii_net or 0, snap.dii_net or 0, snap.flow_sentiment or 0,
        )
        return snap

    # ── Moneycontrol JSON ────────────────────────────────────────────

    async def _fetch_moneycontrol(self, snap: FlowSnapshot) -> FlowSnapshot:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                _MONEYCONTROL_FII_URL, headers=_HEADERS, ssl=False,
            ) as resp:
                if resp.status != 200:
                    logger.debug("Moneycontrol FII: HTTP %d", resp.status)
                    return snap
                data = await resp.json(content_type=None)

        # Moneycontrol returns {"fpiData": [...], "diiData": [...]}
        # Each list has dicts with keys: buyVal, sellVal, netVal, date
        fpi = data.get("fpiData") or data.get("fiiData") or []
        dii = data.get("diiData") or []

        if fpi:
            latest_fpi = fpi[0] if isinstance(fpi, list) else {}
            snap.fii_buy = self._parse_crore(latest_fpi.get("buyVal"))
            snap.fii_sell = self._parse_crore(latest_fpi.get("sellVal"))
            snap.fii_net = self._parse_crore(latest_fpi.get("netVal"))

        if dii:
            latest_dii = dii[0] if isinstance(dii, list) else {}
            snap.dii_buy = self._parse_crore(latest_dii.get("buyVal"))
            snap.dii_sell = self._parse_crore(latest_dii.get("sellVal"))
            snap.dii_net = self._parse_crore(latest_dii.get("netVal"))

        return snap

    # ── NSDL HTML fallback ───────────────────────────────────────────

    async def _fetch_nsdl_html(self, snap: FlowSnapshot) -> FlowSnapshot:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                _NSDL_FPI_URL, headers=_HEADERS, ssl=False,
            ) as resp:
                if resp.status != 200:
                    return snap
                html = await resp.text()

        # Extract numbers from the NSDL report table via regex
        numbers = re.findall(r"[-+]?[\d,]+\.?\d*", html)
        # NSDL table typically has: Date, Buy, Sell, Net columns
        crore_vals = []
        for n in numbers:
            try:
                val = float(n.replace(",", ""))
                if abs(val) > 10:  # filter out small irrelevant numbers
                    crore_vals.append(val)
            except ValueError:
                continue

        # Best-effort mapping: first trio = FII buy/sell/net
        if len(crore_vals) >= 3:
            snap.fii_buy = crore_vals[0]
            snap.fii_sell = crore_vals[1]
            snap.fii_net = crore_vals[2]

        return snap

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_crore(val) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _compute_sentiment(snap: FlowSnapshot) -> None:
        """
        Map FII net flow to a sentiment score in [-1, +1].

        Heuristic thresholds (₹ crore):
          • FII net > +2000  →  +1.0  (strong institutional buying)
          • FII net > +500   →  +0.5
          • FII net ∈ [-500, +500] → 0.0 (neutral)
          • FII net < -500   → -0.5
          • FII net < -2000  → -1.0  (heavy selling)
        """
        if snap.fii_net is None:
            snap.flow_sentiment = None
            return

        net = snap.fii_net
        if net > 2000:
            snap.flow_sentiment = 1.0
        elif net > 500:
            snap.flow_sentiment = 0.5
        elif net > -500:
            snap.flow_sentiment = 0.0
        elif net > -2000:
            snap.flow_sentiment = -0.5
        else:
            snap.flow_sentiment = -1.0

        # FII/DII ratio (for reference)
        if snap.dii_net and snap.dii_net != 0:
            snap.fii_dii_ratio = snap.fii_net / abs(snap.dii_net)
