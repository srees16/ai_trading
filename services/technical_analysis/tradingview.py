"""
TradingView Technical Analysis — multi-timeframe consensus.

Uses the ``tradingview_ta`` library to fetch TradingView's aggregate
BUY / SELL / NEUTRAL recommendation across 26 oscillators and moving
averages for multiple timeframes (1h, 4h, 1D, 1W).

Works identically for Indian (NSE:RELIANCE) and US (NASDAQ:AAPL) tickers.
No API key required — unlimited requests.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Timeframes to query — ordered from shortest to longest
_TIMEFRAMES = ["1h", "4h", "1d", "1W"]

# Exchange mapping: resolve ticker → (exchange, screener)
_EXCHANGE_MAP_IND = {
    ".NS": ("NSE", "india"),
    ".BO": ("BSE", "india"),
}

_US_EXCHANGES = {
    "NASDAQ": "NASDAQ",
    "NYSE": "NYSE",
    "AMEX": "AMEX",
}


@dataclass
class TVTimeframeResult:
    """TradingView consensus for a single timeframe."""
    timeframe: str
    recommendation: str = "NEUTRAL"        # BUY / SELL / STRONG_BUY / STRONG_SELL / NEUTRAL
    buy_count: int = 0                     # out of 26 indicators
    sell_count: int = 0
    neutral_count: int = 0
    oscillators_recommendation: str = "NEUTRAL"
    moving_averages_recommendation: str = "NEUTRAL"
    oscillators_buy: int = 0
    oscillators_sell: int = 0
    moving_averages_buy: int = 0
    moving_averages_sell: int = 0


@dataclass
class TVConsensus:
    """Multi-timeframe TradingView consensus for a ticker."""
    ticker: str
    exchange: str = ""
    screener: str = ""
    timeframes: Dict[str, TVTimeframeResult] = field(default_factory=dict)
    overall_score: float = 0.0           # -1 to +1 fused score
    available: bool = False              # True if at least one TF succeeded

    @property
    def summary(self) -> str:
        """Human-readable one-liner."""
        if not self.available:
            return f"TV unavailable for {self.ticker}"
        parts = []
        for tf in _TIMEFRAMES:
            r = self.timeframes.get(tf)
            if r:
                parts.append(f"{tf}={r.recommendation}")
        return f"TV({self.ticker}): {', '.join(parts)} → score={self.overall_score:+.2f}"


def fetch_tradingview_consensus(
    ticker: str,
    *,
    timeframes: list = None,
) -> TVConsensus:
    """Fetch TradingView multi-timeframe TA consensus.

    Args:
        ticker: Stock ticker (e.g. 'RELIANCE.NS', 'AAPL', 'MSFT')
        timeframes: Override timeframes. Default: 1h, 4h, 1D, 1W

    Returns:
        ``TVConsensus`` with per-timeframe breakdowns and a fused score.
    """
    result = TVConsensus(ticker=ticker)
    tfs = timeframes or _TIMEFRAMES

    # Resolve exchange and screener
    exchange, screener, symbol = _resolve_exchange(ticker)
    if not exchange:
        logger.warning("Cannot resolve exchange for %s — skipping TV consensus", ticker)
        return result

    result.exchange = exchange
    result.screener = screener

    try:
        from tradingview_ta import TA_Handler, Interval
    except ImportError:
        logger.warning("tradingview_ta not installed — TV consensus unavailable")
        return result

    interval_map = {
        "1h": Interval.INTERVAL_1_HOUR,
        "4h": Interval.INTERVAL_4_HOURS,
        "1d": Interval.INTERVAL_1_DAY,
        "1W": Interval.INTERVAL_1_WEEK,
    }

    scores = []

    for tf in tfs:
        interval = interval_map.get(tf)
        if not interval:
            continue

        try:
            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=interval,
            )
            analysis = handler.get_analysis()

            summary = analysis.summary
            osc = analysis.oscillators
            ma = analysis.moving_averages

            tf_result = TVTimeframeResult(
                timeframe=tf,
                recommendation=summary.get("RECOMMENDATION", "NEUTRAL"),
                buy_count=summary.get("BUY", 0),
                sell_count=summary.get("SELL", 0),
                neutral_count=summary.get("NEUTRAL", 0),
                oscillators_recommendation=osc.get("RECOMMENDATION", "NEUTRAL"),
                moving_averages_recommendation=ma.get("RECOMMENDATION", "NEUTRAL"),
                oscillators_buy=osc.get("BUY", 0),
                oscillators_sell=osc.get("SELL", 0),
                moving_averages_buy=ma.get("BUY", 0),
                moving_averages_sell=ma.get("SELL", 0),
            )

            result.timeframes[tf] = tf_result
            scores.append(_recommendation_to_score(tf_result.recommendation))

        except Exception as e:
            logger.debug("TV %s/%s failed: %s", ticker, tf, e)
            continue

    if scores:
        result.available = True
        # Weighted average: longer timeframes matter more
        # 1h=15%, 4h=20%, 1D=35%, 1W=30%
        weights = {"1h": 0.15, "4h": 0.20, "1d": 0.35, "1W": 0.30}
        weighted_sum = 0.0
        total_weight = 0.0
        for tf, tf_result in result.timeframes.items():
            w = weights.get(tf, 0.25)
            weighted_sum += _recommendation_to_score(tf_result.recommendation) * w
            total_weight += w
        result.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    logger.info("TV consensus for %s: %s", ticker, result.summary)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_exchange(ticker: str) -> tuple:
    """Resolve ticker → (exchange, screener, symbol).

    Returns ('', '', '') if unresolvable.
    """
    upper = ticker.upper()

    # Indian tickers (RELIANCE.NS → NSE, india, RELIANCE)
    for suffix, (exchange, screener) in _EXCHANGE_MAP_IND.items():
        if upper.endswith(suffix.upper()):
            symbol = ticker[:len(ticker) - len(suffix)]
            return exchange, screener, symbol

    # US tickers — try NASDAQ first, then NYSE
    # Strip any exchange prefix (e.g. 'NASDAQ:AAPL' → 'AAPL')
    if ":" in ticker:
        parts = ticker.split(":", 1)
        exchange = parts[0].upper()
        symbol = parts[1]
        return exchange, "america", symbol

    # Default: assume US stock, try NASDAQ first
    return "NASDAQ", "america", ticker


def _recommendation_to_score(rec: str) -> float:
    """Convert TradingView recommendation string to numeric score.

    STRONG_BUY = +1.0, BUY = +0.5, NEUTRAL = 0, SELL = -0.5, STRONG_SELL = -1.0
    """
    mapping = {
        "STRONG_BUY": 1.0,
        "BUY": 0.5,
        "NEUTRAL": 0.0,
        "SELL": -0.5,
        "STRONG_SELL": -1.0,
    }
    return mapping.get(rec, 0.0)
