"""
FII/DII Flow Tracker for Indian Markets.

Fetches daily Foreign Institutional Investor (FII) and Domestic
Institutional Investor (DII) buy/sell data from NSDL and NSE.

Provides:
  - Daily net flow (FII buy - sell, DII buy - sell)
  - 5-day rolling net flow for trend detection
  - Gating signal: 5 consecutive days of FII selling > ₹2000 Cr
    suppresses BUY signals

Data source: NSE FII/DII statistics (public, no API key).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE: Optional[dict] = None
_CACHE_TS: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=2)


@dataclass
class FIIDIISnapshot:
    """Daily FII/DII flow data."""
    date: str
    fii_buy: float = 0.0       # ₹ Crores
    fii_sell: float = 0.0
    fii_net: float = 0.0
    dii_buy: float = 0.0
    dii_sell: float = 0.0
    dii_net: float = 0.0

    @property
    def institutional_net(self) -> float:
        return self.fii_net + self.dii_net


@dataclass
class FIIDIISignal:
    """Aggregated FII/DII signal for the scoring pipeline."""
    fii_5d_net: float = 0.0        # 5-day rolling FII net (₹ Cr)
    dii_5d_net: float = 0.0        # 5-day rolling DII net
    consecutive_fii_selling_days: int = 0
    is_fii_selling_pressure: bool = False   # ≥ 5 days of net FII selling
    is_heavy_fii_selling: bool = False      # FII selling > ₹2000 Cr/day for 5 days
    sentiment_score: float = 0.0           # -1 to +1 signal for scoring
    latest: Optional[FIIDIISnapshot] = None

    def to_dict(self) -> dict:
        return {
            "fii_5d_net_cr": round(self.fii_5d_net, 2),
            "dii_5d_net_cr": round(self.dii_5d_net, 2),
            "consecutive_fii_selling_days": self.consecutive_fii_selling_days,
            "is_fii_selling_pressure": self.is_fii_selling_pressure,
            "is_heavy_fii_selling": self.is_heavy_fii_selling,
            "sentiment_score": round(self.sentiment_score, 4),
        }


def fetch_fii_dii_data(days: int = 10) -> List[FIIDIISnapshot]:
    """Fetch recent FII/DII data from NSE.

    Tries the NSE API first, falls back to mock estimates based on
    NIFTY daily returns if the API is unavailable.
    """
    global _CACHE, _CACHE_TS

    now = datetime.utcnow()
    if _CACHE_TS and now - _CACHE_TS < _CACHE_TTL and _CACHE:
        return _CACHE.get("data", [])

    snapshots = _fetch_from_nse(days)

    if not snapshots:
        snapshots = _estimate_from_nifty(days)

    _CACHE = {"data": snapshots}
    _CACHE_TS = now
    return snapshots


def _fetch_from_nse(days: int) -> List[FIIDIISnapshot]:
    """Fetch from NSE FII/DII API."""
    try:
        import requests

        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            return []

        data = resp.json()
        snapshots: List[FIIDIISnapshot] = []

        for item in data:
            category = item.get("category", "")
            date_str = item.get("date", "")
            buy_val = float(item.get("buyValue", 0))
            sell_val = float(item.get("sellValue", 0))
            net_val = float(item.get("netValue", 0))

            if "FII" in category.upper() or "FPI" in category.upper():
                # Find or create snapshot for this date
                snap = next((s for s in snapshots if s.date == date_str), None)
                if not snap:
                    snap = FIIDIISnapshot(date=date_str)
                    snapshots.append(snap)
                snap.fii_buy = buy_val
                snap.fii_sell = sell_val
                snap.fii_net = net_val
            elif "DII" in category.upper():
                snap = next((s for s in snapshots if s.date == date_str), None)
                if not snap:
                    snap = FIIDIISnapshot(date=date_str)
                    snapshots.append(snap)
                snap.dii_buy = buy_val
                snap.dii_sell = sell_val
                snap.dii_net = net_val

        logger.info("Fetched %d days of FII/DII data from NSE", len(snapshots))
        return snapshots[-days:]

    except Exception as exc:
        logger.warning("NSE FII/DII fetch failed: %s", exc)
        return []


def _estimate_from_nifty(days: int) -> List[FIIDIISnapshot]:
    """Estimate FII/DII sentiment from NIFTY daily returns as fallback."""
    try:
        import yfinance as yf

        data = yf.download("^NSEI", period=f"{days + 5}d", progress=False)
        if data.empty:
            return []

        returns = data["Close"].pct_change().dropna().tail(days)
        snapshots = []

        for dt, ret in returns.items():
            ret_val = float(ret)
            # Rough estimate: positive NIFTY day → FII likely buying
            fii_net = ret_val * 5000  # Scale to approximate ₹ Cr
            dii_net = -fii_net * 0.3  # DII counter-trade

            snapshots.append(FIIDIISnapshot(
                date=str(dt.date()) if hasattr(dt, 'date') else str(dt),
                fii_net=round(fii_net, 2),
                dii_net=round(dii_net, 2),
            ))

        return snapshots

    except Exception:
        return []


def compute_fii_dii_signal() -> FIIDIISignal:
    """Compute the FII/DII gating signal for the scoring pipeline."""
    data = fetch_fii_dii_data(days=10)

    if not data:
        return FIIDIISignal()

    # 5-day rolling net
    recent_5 = data[-5:] if len(data) >= 5 else data
    fii_5d = sum(s.fii_net for s in recent_5)
    dii_5d = sum(s.dii_net for s in recent_5)

    # Consecutive FII selling days
    consecutive = 0
    heavy_selling = 0
    for s in reversed(data):
        if s.fii_net < 0:
            consecutive += 1
            if s.fii_net < -2000:  # > ₹2000 Cr selling
                heavy_selling += 1
        else:
            break

    is_pressure = consecutive >= 5
    is_heavy = heavy_selling >= 5

    # Sentiment score: -1 (heavy selling) to +1 (heavy buying)
    # Normalize 5-day FII net by ₹10,000 Cr scale
    raw = fii_5d / 10000.0
    sentiment = max(-1.0, min(1.0, raw))

    # Penalize for consecutive selling
    if consecutive >= 3:
        sentiment = min(sentiment, sentiment - consecutive * 0.05)
        sentiment = max(-1.0, sentiment)

    return FIIDIISignal(
        fii_5d_net=fii_5d,
        dii_5d_net=dii_5d,
        consecutive_fii_selling_days=consecutive,
        is_fii_selling_pressure=is_pressure,
        is_heavy_fii_selling=is_heavy,
        sentiment_score=round(sentiment, 4),
        latest=data[-1] if data else None,
    )
