"""
Post-Earnings Momentum Detector.

After a positive earnings surprise (revenue beat > 5 % or profit beat > 10 %),
stocks often exhibit a 5-day post-earnings drift. This module detects such
events and provides a momentum boost to the scoring pipeline.

Integration points:
  - IntegratedScorer core layer → +0.12 boost for 5 trading days post-surprise
  - DecisionEngine → relaxes blackout suppression if surprise is positive
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_CACHE: Dict[str, "EarningsSurprise"] = {}
_CACHE_TS: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=6)


@dataclass
class EarningsSurprise:
    """Detected earnings surprise for a single symbol."""
    symbol: str
    result_date: str                 # YYYY-MM-DD
    revenue_surprise_pct: float = 0  # actual vs estimate, %
    profit_surprise_pct: float = 0
    is_positive: bool = False
    days_since: int = 999            # trading days since result

    @property
    def momentum_active(self) -> bool:
        """True if positive surprise within 5 trading days."""
        return self.is_positive and self.days_since <= 5

    @property
    def boost(self) -> float:
        """Score boost — decays linearly over 5 days."""
        if not self.momentum_active:
            return 0.0
        # Day 0 = +0.12, Day 5 = +0.02
        return max(0.02, 0.12 - (self.days_since * 0.02))


def _fetch_recent_results() -> Dict[str, EarningsSurprise]:
    """Scrape recent quarterly results from Moneycontrol / Trendlyne.

    Uses Trendlyne earnings calendar as primary source, falls back
    to a lightweight Moneycontrol scrape.
    """
    results: Dict[str, EarningsSurprise] = {}
    today = datetime.now().date()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/119.0 Safari/537.36"
        ),
    }

    # ── Source 1: Trendlyne board meetings / results ──
    try:
        resp = requests.get(
            "https://trendlyne.com/fundamentals/results-calendar/",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200 and resp.text:
            import re
            # Look for result data in the page (JSON blocks or table rows)
            # Trendlyne embeds result data in script tags
            json_blocks = re.findall(
                r'"symbol"\s*:\s*"([^"]+)".*?"resultDate"\s*:\s*"([^"]+)"'
                r'.*?"revSurprise"\s*:\s*([-\d.]+).*?"profitSurprise"\s*:\s*([-\d.]+)',
                resp.text,
                re.DOTALL,
            )
            for sym, rdate, rev_s, prof_s in json_blocks:
                try:
                    result_date = datetime.strptime(rdate, "%Y-%m-%d").date()
                    days = (today - result_date).days
                    # Approximate trading days (exclude weekends)
                    trading_days = int(days * 5 / 7)
                    rev_surp = float(rev_s)
                    prof_surp = float(prof_s)
                    is_pos = rev_surp > 5.0 or prof_surp > 10.0
                    results[sym.upper()] = EarningsSurprise(
                        symbol=sym.upper(),
                        result_date=str(result_date),
                        revenue_surprise_pct=rev_surp,
                        profit_surprise_pct=prof_surp,
                        is_positive=is_pos,
                        days_since=trading_days,
                    )
                except (ValueError, TypeError):
                    continue
    except Exception as exc:
        logger.debug("Trendlyne earnings fetch failed: %s", exc)

    # ── Source 2: NSE corporate filings for board meeting dates ──
    try:
        sess = requests.Session()
        sess.headers.update(headers)
        sess.get("https://www.nseindia.com", timeout=5)
        resp = sess.get(
            "https://www.nseindia.com/api/corporate-board-meetings",
            params={"index": "equities"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            for item in data if isinstance(data, list) else []:
                sym = item.get("symbol", "")
                purpose = (item.get("purpose") or "").lower()
                bm_date_str = item.get("bm_date", "")
                if "financial result" in purpose and sym and bm_date_str:
                    try:
                        bm_date = datetime.strptime(bm_date_str, "%d-%b-%Y").date()
                        days = (today - bm_date).days
                        trading_days = int(days * 5 / 7)
                        if trading_days <= 7 and sym not in results:
                            # We know they had results but don't have surprise data
                            # Mark as neutral — the delivery/price action will tell
                            results[sym] = EarningsSurprise(
                                symbol=sym,
                                result_date=str(bm_date),
                                days_since=trading_days,
                            )
                    except (ValueError, TypeError):
                        continue
    except Exception as exc:
        logger.debug("NSE board meetings fetch failed: %s", exc)

    return results


def get_earnings_surprises(
    symbols: Optional[List[str]] = None,
) -> Dict[str, EarningsSurprise]:
    """Get earnings surprise data, cached for 6 hours.

    Returns:
        Mapping of symbol → EarningsSurprise.
    """
    global _CACHE, _CACHE_TS

    now = datetime.now()
    if _CACHE_TS and (now - _CACHE_TS) < _CACHE_TTL and _CACHE:
        if symbols:
            return {s: _CACHE[s] for s in symbols if s in _CACHE}
        return dict(_CACHE)

    fresh = _fetch_recent_results()
    if fresh:
        _CACHE = fresh
        _CACHE_TS = now
        logger.info("Earnings surprise data fetched for %d symbols", len(fresh))

    result = _CACHE if _CACHE else {}
    if symbols:
        return {s: result[s] for s in symbols if s in result}
    return result


def get_post_earnings_boost(symbol: str) -> float:
    """Return the post-earnings momentum boost for a ticker.

    Returns:
        0.0–0.12 score boost (decays over 5 trading days).
    """
    data = get_earnings_surprises([symbol])
    surprise = data.get(symbol)
    if surprise and surprise.momentum_active:
        return surprise.boost
    return 0.0
