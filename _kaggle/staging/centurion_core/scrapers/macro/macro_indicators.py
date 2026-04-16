# Centurion Capital LLC — Macro-Economic Indicators
"""
Fetches key macro-economic indicators from public data sources (FRED via
yfinance proxies, Yahoo Finance market indices) to build a market-level
sentiment overlay.

Indicators scraped
──────────────────
* **VIX** (CBOE Volatility Index)          — fear gauge
* **US 10Y Treasury yield** (^TNX)         — risk-free rate proxy
* **US 2Y Treasury yield** (^IRX 13-wk) — short-end proxy for yield curve
* **S&P 500** (^GSPC)                      — broad market health
* **Gold** (GC=F)                          — safe-haven demand
* **DXY / USD Index** (DX-Y.NYB)           — dollar strength
* **India VIX** (^INDIAVIX)                — India-specific fear gauge
* **Nifty 50** (^NSEI)                     — Indian broad market
* **Crude Oil WTI** (CL=F)                 — energy cost proxy

The module exposes a single ``MacroIndicators`` class whose
``fetch()`` method returns a ``MacroSnapshot`` dataclass suitable for
downstream consumption by the ``DecisionEngine`` and the sentiment
analysis layer.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional

import yfinance as yf

logger = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────

@dataclass
class MacroSnapshot:
    """Point-in-time snapshot of macro-economic indicators."""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Volatility
    vix: Optional[float] = None
    india_vix: Optional[float] = None

    # Yields / rates
    us_10y_yield: Optional[float] = None
    us_13w_yield: Optional[float] = None          # proxy for short end
    yield_curve_spread: Optional[float] = None     # 10Y - 13W

    # Broad indices
    sp500_price: Optional[float] = None
    sp500_change_pct: Optional[float] = None       # 1-day % change
    nifty50_price: Optional[float] = None
    nifty50_change_pct: Optional[float] = None

    # Commodities / FX
    gold_price: Optional[float] = None
    crude_oil_price: Optional[float] = None
    dxy_index: Optional[float] = None

    # FII/DII flows (IND only)
    fii_net_crore: Optional[float] = None
    fii_flow_sentiment: Optional[float] = None  # -1 … +1

    # Derived sentiment
    macro_sentiment_score: Optional[float] = None   # -1 (fear) … +1 (greed)
    macro_sentiment_label: Optional[str] = None      # fearful / neutral / greedy

    def to_dict(self) -> Dict[str, object]:
        """Flat dict for easy DataFrame / JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "vix": self.vix,
            "india_vix": self.india_vix,
            "us_10y_yield": self.us_10y_yield,
            "us_13w_yield": self.us_13w_yield,
            "yield_curve_spread": self.yield_curve_spread,
            "sp500_price": self.sp500_price,
            "sp500_change_pct": self.sp500_change_pct,
            "nifty50_price": self.nifty50_price,
            "nifty50_change_pct": self.nifty50_change_pct,
            "gold_price": self.gold_price,
            "crude_oil_price": self.crude_oil_price,
            "dxy_index": self.dxy_index,
            "fii_net_crore": self.fii_net_crore,
            "fii_flow_sentiment": self.fii_flow_sentiment,
            "macro_sentiment_score": self.macro_sentiment_score,
            "macro_sentiment_label": self.macro_sentiment_label,
        }


# ── Ticker field mapping ──────────────────────────────────────────

_TICKERS: Dict[str, str] = {
    "^VIX":       "vix",
    "^INDIAVIX":  "india_vix",
    "^TNX":       "us_10y_yield",
    "^IRX":       "us_13w_yield",
    "^GSPC":      "sp500",
    "^NSEI":      "nifty50",
    "GC=F":       "gold_price",
    "CL=F":       "crude_oil_price",
    "DX-Y.NYB":   "dxy_index",
}


# ── Core class ───────────────────────────────────────────────────────

class MacroIndicators:
    """
    Fetches macro-economic indicators from Yahoo Finance and derives
    a composite market-sentiment score.

    Usage::

        mi = MacroIndicators()
        snap = mi.fetch(market="US")   # or "IND"
    """

    # In-process cache (class-level, per-market)
    _cached_snapshots: Dict[str, MacroSnapshot] = {}
    _cache_timestamps: Dict[str, datetime] = {}
    _CACHE_TTL = timedelta(minutes=15)

    def fetch(self, market: str = "US") -> MacroSnapshot:
        """
        Fetch latest macro indicators (cached for 15 min).

        Args:
            market: ``"US"`` or ``"IND"`` — controls which indices are
                    fetched and how the composite score is weighted.

        Returns:
            A :class:`MacroSnapshot` with all available fields populated.
        """
        now = datetime.utcnow()
        cached = MacroIndicators._cached_snapshots.get(market)
        cached_ts = MacroIndicators._cache_timestamps.get(market)
        if (
            cached is not None
            and cached_ts is not None
            and (now - cached_ts) < self._CACHE_TTL
        ):
            logger.debug("MacroIndicators: returning cached snapshot for %s", market)
            return cached

        snap = MacroSnapshot(timestamp=now)

        # ── Bulk download ────────────────────────────────────────────
        symbols = list(_TICKERS.keys())
        try:
            data = yf.download(
                symbols,
                period="5d",
                progress=False,
                threads=False,
                group_by="ticker",
            )
        except Exception as exc:
            logger.warning("MacroIndicators: yf.download failed — %s", exc)
            data = None

        if data is not None and not data.empty:
            for sym, field_key in _TICKERS.items():
                try:
                    if len(symbols) == 1:
                        col = data
                    else:
                        col = data[sym] if sym in data.columns.get_level_values(0) else None
                    if col is None or col.empty:
                        continue

                    closes = col["Close"].dropna()
                    if closes.empty:
                        continue
                    latest = float(closes.iloc[-1])

                    if field_key == "vix":
                        snap.vix = latest
                    elif field_key == "india_vix":
                        snap.india_vix = latest
                    elif field_key == "us_10y_yield":
                        snap.us_10y_yield = latest
                    elif field_key == "us_13w_yield":
                        snap.us_13w_yield = latest
                    elif field_key == "sp500":
                        snap.sp500_price = latest
                        if len(closes) >= 2:
                            prev = float(closes.iloc[-2])
                            snap.sp500_change_pct = (
                                (latest - prev) / prev * 100 if prev else None
                            )
                    elif field_key == "nifty50":
                        snap.nifty50_price = latest
                        if len(closes) >= 2:
                            prev = float(closes.iloc[-2])
                            snap.nifty50_change_pct = (
                                (latest - prev) / prev * 100 if prev else None
                            )
                    elif field_key == "gold_price":
                        snap.gold_price = latest
                    elif field_key == "crude_oil_price":
                        snap.crude_oil_price = latest
                    elif field_key == "dxy_index":
                        snap.dxy_index = latest
                except Exception as exc:
                    logger.debug("MacroIndicators: error parsing %s — %s", sym, exc)

        # Yield-curve spread
        if snap.us_10y_yield is not None and snap.us_13w_yield is not None:
            snap.yield_curve_spread = snap.us_10y_yield - snap.us_13w_yield

        # ── FII/DII flows (IND market only) ──────────────────────────
        if market == "IND":
            try:
                import asyncio
                from scrapers.ind_news.fii_dii_flows import FIIDIIFlows
                loop = asyncio.new_event_loop()
                try:
                    flow_snap = loop.run_until_complete(FIIDIIFlows().fetch())
                finally:
                    loop.close()
                snap.fii_net_crore = flow_snap.fii_net
                snap.fii_flow_sentiment = flow_snap.flow_sentiment
            except Exception as exc:
                logger.debug("MacroIndicators: FII/DII fetch failed — %s", exc)

        # ── Derive composite sentiment score ─────────────────────────
        snap.macro_sentiment_score = self._compute_sentiment(snap, market)
        if snap.macro_sentiment_score is not None:
            if snap.macro_sentiment_score >= 0.25:
                snap.macro_sentiment_label = "greedy"
            elif snap.macro_sentiment_score <= -0.25:
                snap.macro_sentiment_label = "fearful"
            else:
                snap.macro_sentiment_label = "neutral"

        # Cache only if we got meaningful data
        has_data = snap.vix is not None or snap.india_vix is not None or snap.sp500_price is not None
        if has_data:
            MacroIndicators._cached_snapshots[market] = snap
            MacroIndicators._cache_timestamps[market] = now

        logger.info(
            "MacroIndicators: VIX=%.1f  10Y=%.2f  S&P=%.0f  Macro=%s (%.2f)",
            snap.vix or 0,
            snap.us_10y_yield or 0,
            snap.sp500_price or 0,
            snap.macro_sentiment_label or "n/a",
            snap.macro_sentiment_score or 0,
        )
        return snap

    # ── Internal scoring ─────────────────────────────────────────────

    @staticmethod
    def _compute_sentiment(snap: MacroSnapshot, market: str) -> Optional[float]:
        """
        Derive a composite macro-sentiment score in [-1, +1].

        Components (equal weight where available):
        * VIX zone : < 15 +1, 15-20 +0.5, 20-30 -0.5, > 30 -1
        * Yield curve : positive spread +0.5, negative -0.5
        * Index momentum   : recent % change mapped to [-1,+1]
        * Gold / oil : rising gold fearful (-), rising oil mixed
        """
        scores = []

        # --- VIX ---
        vix = snap.india_vix if market == "IND" else snap.vix
        if vix is not None:
            if vix < 15:
                scores.append(1.0)
            elif vix < 20:
                scores.append(0.5)
            elif vix < 25:
                scores.append(0.0)
            elif vix < 30:
                scores.append(-0.5)
            else:
                scores.append(-1.0)

        # --- Yield curve ---
        if snap.yield_curve_spread is not None:
            if snap.yield_curve_spread > 0.5:
                scores.append(0.5)
            elif snap.yield_curve_spread > 0:
                scores.append(0.25)
            elif snap.yield_curve_spread > -0.5:
                scores.append(-0.25)
            else:
                scores.append(-0.5)

        # --- Index momentum ---
        change = snap.nifty50_change_pct if market == "IND" else snap.sp500_change_pct
        if change is not None:
            clamped = max(-3.0, min(3.0, change))  # cap at ±3%
            scores.append(clamped / 3.0)

        # --- FII/DII flow sentiment (IND only) ---
        if market == "IND" and snap.fii_flow_sentiment is not None:
            scores.append(snap.fii_flow_sentiment)

        if not scores:
            return None

        return max(-1.0, min(1.0, sum(scores) / len(scores)))
