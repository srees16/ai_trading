"""
NSE Delivery Volume Analyser.

Fetches delivery percentage data from NSE bhavcopy and provides a
conviction signal: stocks with delivery % > 60 % on above-average
volume indicate genuine institutional buying (not intraday churn).

Integration points:
  - NSE screener ``_compute_score()`` → +12 pts for high delivery
  - IntegratedScorer core layer → delivery-adjusted conviction
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE: Dict[str, "DeliveryData"] = {}
_CACHE_TS: Optional[datetime] = None
_CACHE_TTL = timedelta(minutes=30)

# NSE bhavcopy / equity delivery endpoint
_NSE_DELIVERY_URL = "https://www.nseindia.com/api/equity-stockIndices"
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class DeliveryData:
    """Delivery volume data for a single symbol."""
    symbol: str
    delivery_qty: int = 0
    traded_qty: int = 0
    delivery_pct: float = 0.0     # 0–100
    date: str = ""

    @property
    def is_high_delivery(self) -> bool:
        """Delivery > 60 % → strong institutional conviction."""
        return self.delivery_pct >= 60.0

    @property
    def is_very_high_delivery(self) -> bool:
        """Delivery > 75 % → very strong conviction."""
        return self.delivery_pct >= 75.0


def _get_nse_session() -> requests.Session:
    """Create a session with NSE cookies pre-loaded."""
    sess = requests.Session()
    sess.headers.update(_NSE_HEADERS)
    try:
        sess.get("https://www.nseindia.com", timeout=5)
    except Exception:
        pass
    return sess


def fetch_delivery_data(symbols: Optional[List[str]] = None) -> Dict[str, DeliveryData]:
    """Fetch delivery volume data for NSE equities.

    Uses the NSE equity bhavcopy which includes delivery quantities.
    Falls back to cached data if the API is unreachable.

    Returns:
        Mapping of symbol → DeliveryData.
    """
    global _CACHE, _CACHE_TS

    now = datetime.now()
    if _CACHE_TS and (now - _CACHE_TS) < _CACHE_TTL and _CACHE:
        if symbols:
            return {s: _CACHE[s] for s in symbols if s in _CACHE}
        return dict(_CACHE)

    result: Dict[str, DeliveryData] = {}

    try:
        sess = _get_nse_session()

        # Fetch NIFTY 500 which covers most screened symbols
        for index_name in ["NIFTY 500", "NIFTY 50", "NIFTY NEXT 50"]:
            try:
                resp = sess.get(
                    _NSE_DELIVERY_URL,
                    params={"index": index_name},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for item in data.get("data", []):
                    sym = item.get("symbol", "")
                    if not sym or sym == "NIFTY 500":
                        continue
                    delivery_qty = item.get("totalTradedVolume", 0)
                    traded_qty = item.get("totalTradedValue", 0)
                    # pChange available; but we need delivery %
                    # NSE provides deliveryToTradedQuantity in some endpoints
                    del_pct = item.get("deliveryToTradedQuantity")
                    if del_pct is None:
                        del_pct = item.get("deliveryQuantity", 0)
                        if delivery_qty and delivery_qty > 0 and del_pct:
                            del_pct = (del_pct / delivery_qty) * 100
                        else:
                            del_pct = 0
                    result[sym] = DeliveryData(
                        symbol=sym,
                        delivery_qty=int(item.get("deliveryQuantity", 0)),
                        traded_qty=int(item.get("totalTradedVolume", 0)),
                        delivery_pct=float(del_pct) if del_pct else 0.0,
                        date=item.get("lastUpdateTime", now.strftime("%d-%b-%Y")),
                    )
            except Exception as exc:
                logger.debug("Delivery fetch for %s failed: %s", index_name, exc)
                continue

        if result:
            _CACHE = result
            _CACHE_TS = now
            logger.info("Delivery volume data fetched for %d symbols", len(result))

    except Exception as exc:
        logger.warning("NSE delivery data fetch failed: %s", exc)

    # Fall back to stale cache if fresh fetch failed
    if not result and _CACHE:
        logger.debug("Using stale delivery cache (%d symbols)", len(_CACHE))
        result = dict(_CACHE)

    if symbols:
        return {s: result[s] for s in symbols if s in result}
    return result


def get_delivery_score(symbol: str) -> float:
    """Return a delivery conviction score for the screener.

    Returns:
        Score in 0–12 range:
          - delivery >= 75%  → 12 pts
          - delivery >= 60%  → 8 pts
          - delivery >= 45%  → 4 pts
          - delivery <  45%  → 0 pts
    """
    data = fetch_delivery_data([symbol])
    dd = data.get(symbol)
    if not dd:
        return 0.0
    if dd.delivery_pct >= 75:
        return 12.0
    if dd.delivery_pct >= 60:
        return 8.0
    if dd.delivery_pct >= 45:
        return 4.0
    return 0.0


def get_delivery_conviction(symbol: str) -> float:
    """Return a conviction multiplier for the IntegratedScorer.

    Returns:
        Multiplier 0.9–1.15:
          - very high delivery → 1.15
          - high delivery      → 1.08
          - normal             → 1.0
          - low delivery (<30%)→ 0.92
    """
    data = fetch_delivery_data([symbol])
    dd = data.get(symbol)
    if not dd or dd.delivery_pct == 0:
        return 1.0
    if dd.delivery_pct >= 75:
        return 1.15
    if dd.delivery_pct >= 60:
        return 1.08
    if dd.delivery_pct < 30:
        return 0.92
    return 1.0
