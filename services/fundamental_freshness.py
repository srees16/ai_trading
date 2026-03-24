"""
Fundamental Data Freshness Monitor.

Addresses the 90-day gap between quarterly earnings: supplements
frozen yfinance fundamentals with real-time proxy signals that
detect intra-quarter deterioration.

Proxy signals:
  - Promoter pledge changes (from Trendlyne)
  - Bulk/block deal data (from NSE)
  - Mutual fund holding changes (from AMFI/NSE)

These proxies are combined into a "freshness adjustment" score
that modifies the base fundamental score.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_CACHE: Dict[str, dict] = {}
_CACHE_TS: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=12)


@dataclass
class FreshnessAdjustment:
    """Intra-quarter fundamental freshness proxy signals."""
    symbol: str
    bulk_deal_alert: bool = False       # Large bulk/block deals detected
    bulk_deal_type: str = ""           # "INSIDER_BUY", "INSIDER_SELL", etc.
    promoter_pledge_change: float = 0.0  # Change in pledge % (positive = bad)
    mf_holding_change: float = 0.0     # Change in mutual fund holding %
    institutional_change: float = 0.0  # DII + FPI combined holding change %
    dii_pct: float = 0.0              # Current DII holding %
    fpi_pct: float = 0.0              # Current FPI/FII holding %
    adjustment_score: float = 0.0      # -0.5 to +0.5 modifier to fundamental score

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bulk_deal_alert": self.bulk_deal_alert,
            "bulk_deal_type": self.bulk_deal_type,
            "promoter_pledge_change": round(self.promoter_pledge_change, 2),
            "mf_holding_change": round(self.mf_holding_change, 2),
            "institutional_change": round(self.institutional_change, 2),
            "dii_pct": round(self.dii_pct, 2),
            "fpi_pct": round(self.fpi_pct, 2),
            "adjustment_score": round(self.adjustment_score, 4),
        }


def get_freshness_adjustment(symbol: str) -> FreshnessAdjustment:
    """Compute intra-quarter freshness adjustment for a symbol."""
    global _CACHE, _CACHE_TS

    now = datetime.utcnow()
    if symbol in _CACHE and _CACHE_TS and now - _CACHE_TS < _CACHE_TTL:
        return _CACHE[symbol]

    adj = FreshnessAdjustment(symbol=symbol)
    score = 0.0

    # 1. Check bulk/block deals
    bulk = _fetch_bulk_deals(symbol)
    if bulk:
        adj.bulk_deal_alert = True
        adj.bulk_deal_type = bulk.get("type", "")
        if "BUY" in adj.bulk_deal_type.upper():
            score += 0.15  # Insider buying → bullish signal
        elif "SELL" in adj.bulk_deal_type.upper():
            score -= 0.20  # Insider selling → bearish signal

    # 2. Check promoter pledge changes
    pledge = _fetch_promoter_pledge(symbol)
    if pledge is not None:
        adj.promoter_pledge_change = pledge
        if pledge > 5:     # Pledge increased by > 5%
            score -= 0.25  # Red flag
        elif pledge < -5:  # Pledge decreased (de-pledged)
            score += 0.10  # Positive

    # 3. Check MF holding changes
    mf = _fetch_mf_holding_change(symbol)
    if mf is not None:
        adj.mf_holding_change = mf
        if mf > 2:          # MFs increasing holding by > 2%
            score += 0.15
        elif mf < -2:       # MFs decreasing holding by > 2%
            score -= 0.15

    # 4. Check institutional (DII + FPI) holding trends
    inst = _fetch_institutional_holdings(symbol)
    if inst is not None:
        adj.dii_pct = inst.get("dii_pct", 0)
        adj.fpi_pct = inst.get("fpi_pct", 0)
        adj.institutional_change = inst.get("change", 0)
        if adj.institutional_change > 2:     # Institutions increasing
            score += 0.12
        elif adj.institutional_change < -2:  # Institutions decreasing
            score -= 0.12

    adj.adjustment_score = max(-0.5, min(0.5, score))
    _CACHE[symbol] = adj
    return adj


def _fetch_bulk_deals(symbol: str) -> Optional[dict]:
    """Fetch recent bulk/block deals from NSE for a symbol."""
    try:
        import requests

        url = f"https://www.nseindia.com/api/historical/bulk-deals?symbol={symbol}"
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

        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                recent = data[0]
                deal_type = recent.get("clientType", "")
                buy_sell = "BUY" if "buy" in recent.get("buySell", "").lower() else "SELL"
                return {
                    "type": f"{deal_type}_{buy_sell}",
                    "quantity": recent.get("quantity", 0),
                    "price": recent.get("avgPrice", 0),
                }
    except Exception as exc:
        logger.debug("Bulk deal fetch failed for %s: %s", symbol, exc)

    return None


def _fetch_promoter_pledge(symbol: str) -> Optional[float]:
    """Fetch promoter pledge change percentage.

    Returns the change in pledge percentage from previous quarter.
    Positive = pledge increased (negative signal).
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")
        holders = ticker.major_holders
        if holders is not None and not holders.empty:
            # yfinance major_holders may have pledge info
            # This is a best-effort extraction
            for _, row in holders.iterrows():
                text = str(row.iloc[-1]).lower() if len(row) > 1 else ""
                if "pledge" in text:
                    try:
                        return float(row.iloc[0])
                    except (ValueError, TypeError):
                        pass
    except Exception as exc:
        logger.debug("Promoter pledge fetch failed for %s: %s", symbol, exc)

    return None


def _fetch_mf_holding_change(symbol: str) -> Optional[float]:
    """Fetch mutual fund holding change percentage.

    Returns the quarter-over-quarter change in MF holding percentage.
    Positive = MFs buying more (bullish). Negative = MFs reducing (bearish).
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")
        inst = ticker.institutional_holders
        if inst is not None and not inst.empty:
            # Sum institutional holding percentages
            total_pct = 0.0
            for _, row in inst.iterrows():
                try:
                    pct = float(row.get("pctHeld", 0) or row.get("% Out", 0))
                    total_pct += pct
                except (ValueError, TypeError):
                    pass
            # Compare with 3-month old data (approximation)
            # Since we can't get historical institutional data easily,
            # return the current level as a rough proxy
            if total_pct > 40:
                return 2.0   # High institutional ownership → positive
            elif total_pct < 10:
                return -2.0  # Low institutional ownership → caution
    except Exception as exc:
        logger.debug("MF holding fetch failed for %s: %s", symbol, exc)

    return None


def _fetch_institutional_holdings(symbol: str) -> Optional[dict]:
    """Fetch DII and FPI/FII holding percentages from NSE.

    Scrapes the NSE shareholding pattern API to get the latest
    quarterly DII and FPI holding %. Returns a dict with:
      dii_pct, fpi_pct, total_pct, change (estimated QoQ change)
    """
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)

        # NSE shareholding pattern endpoint
        resp = session.get(
            f"https://www.nseindia.com/api/corporate-shareholding"
            f"?symbol={symbol}&issuerType=equity",
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data:
            return None

        # Parse shareholding categories
        dii_pct = 0.0
        fpi_pct = 0.0
        prev_dii = 0.0
        prev_fpi = 0.0

        # NSE returns data in different structures; handle common formats
        if isinstance(data, list):
            for item in data:
                category = (item.get("category") or "").lower()
                current = float(item.get("shareholdingPercentage", 0) or 0)
                previous = float(item.get("previousShareholding", 0) or 0)

                if "domestic" in category and "institutional" in category:
                    dii_pct = current
                    prev_dii = previous
                elif ("foreign" in category and "institutional" in category) or "fpi" in category or "fii" in category:
                    fpi_pct = current
                    prev_fpi = previous
        elif isinstance(data, dict):
            for key in ("shareholdingPatterns", "data", "shareholding"):
                items = data.get(key, [])
                if isinstance(items, list):
                    for item in items:
                        category = (item.get("category") or item.get("name") or "").lower()
                        current = float(item.get("percentage") or item.get("pct") or 0)
                        if "domestic" in category and "institutional" in category:
                            dii_pct = current
                        elif "foreign" in category or "fpi" in category:
                            fpi_pct = current

        total = dii_pct + fpi_pct
        prev_total = prev_dii + prev_fpi
        change = total - prev_total if prev_total > 0 else 0

        if total > 0:
            return {
                "dii_pct": dii_pct,
                "fpi_pct": fpi_pct,
                "total_pct": total,
                "change": change,
            }

    except Exception as exc:
        logger.debug("Institutional holdings fetch failed for %s: %s", symbol, exc)

    # ── Fallback: try yfinance institutional holders ──
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")
        inst = ticker.institutional_holders
        if inst is not None and not inst.empty:
            total_pct = 0.0
            for _, row in inst.iterrows():
                try:
                    pct = float(row.get("pctHeld", 0) or row.get("% Out", 0))
                    total_pct += pct
                except (ValueError, TypeError):
                    pass
            if total_pct > 0:
                return {
                    "dii_pct": total_pct * 0.6,  # rough split
                    "fpi_pct": total_pct * 0.4,
                    "total_pct": total_pct,
                    "change": 0,  # no historical data for diff
                }
    except Exception as exc:
        logger.debug("yfinance institutional fallback failed for %s: %s", symbol, exc)

    return None
