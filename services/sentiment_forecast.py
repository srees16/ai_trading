"""
Sentiment Forecast — News sentiment as a Carver-scaled forecast source.

Aggregates recent news sentiment scores per ticker from scraped news articles
(IND: Moneycontrol, ET, Mint, etc. | US: Yahoo, Finviz, etc.) and converts
them to Carver forecast scale [-20, +20].

Pipeline:
  1. Query recent NewsItem objects from database/cache (7-day window)
  2. Filter by FinBERT confidence > threshold (0.85 default)
  3. Aggregate with exponential time-decay (half-life ≈ 3 days)
  4. Z-score normalize across universe
  5. Scale to [-20, +20]: forecast = z_score × SCALAR_SENTIMENT
  6. Cap at ±20

Integration:
  - Added as forecast source "sentiment" in forecast_combiner.py
  - Low correlation with trend-following (EWMAC ~0.15), moderate with PEAD (~0.30)
  - Weight: 3% (conservative until validated in walk-forward)

Regime behaviour:
  - BULL: amplifies existing momentum signals
  - BEAR: detects fear spikes → early warning
  - RANGE: noise filter (low confidence → suppressed)
  - CRISIS: extreme negative sentiment → contrarian buy signals when oversold
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.forecast_scalar import cap_forecast

logger = logging.getLogger(__name__)

# Scalar to convert z-scored sentiment to Carver forecast space
# avg|sentiment_zscore| ≈ 0.8  →  target avg|forecast| ≈ 10  →  scalar ≈ 12.5
SCALAR_SENTIMENT = 12.5

# Lookback window for news aggregation (days)
SENTIMENT_LOOKBACK_DAYS = 7

# Exponential decay half-life (days) — recent news matters more
DECAY_HALFLIFE_DAYS = 3.0

# Minimum number of articles required to produce a signal
MIN_ARTICLES = 2

# Default confidence threshold (overridden by Config if available)
DEFAULT_CONFIDENCE_THRESHOLD = 0.85


def _get_confidence_threshold() -> float:
    """Read sentiment confidence threshold from Config."""
    try:
        from config import Config
        return getattr(Config, "SENTIMENT_HIGH_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD)
    except Exception:
        return DEFAULT_CONFIDENCE_THRESHOLD


def _fetch_recent_news(ticker: str, lookback_days: int = SENTIMENT_LOOKBACK_DAYS) -> list:
    """Fetch recent news items for a ticker from the database.

    Returns list of dicts with keys: sentiment_score, confidence, timestamp.
    Falls back to empty list if no news data is available.
    """
    try:
        from database.service import DatabaseService
        db = DatabaseService()
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        news_items = db.get_news_by_ticker(ticker, since=cutoff)
        return [
            {
                "sentiment_score": getattr(item, "sentiment_score", None),
                "confidence": getattr(item, "sentiment_confidence", None),
                "timestamp": getattr(item, "timestamp", None),
            }
            for item in (news_items or [])
            if getattr(item, "sentiment_score", None) is not None
        ]
    except Exception:
        return []


def _fetch_recent_news_batch(tickers: List[str], lookback_days: int = SENTIMENT_LOOKBACK_DAYS) -> Dict[str, list]:
    """Fetch recent news for multiple tickers. Returns {ticker: [news_items]}."""
    try:
        from database.service import DatabaseService
        db = DatabaseService()
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        result: Dict[str, list] = {}
        for ticker in tickers:
            try:
                news_items = db.get_news_by_ticker(ticker, since=cutoff)
                result[ticker] = [
                    {
                        "sentiment_score": getattr(item, "sentiment_score", None),
                        "confidence": getattr(item, "sentiment_confidence", None),
                        "timestamp": getattr(item, "timestamp", None),
                    }
                    for item in (news_items or [])
                    if getattr(item, "sentiment_score", None) is not None
                ]
            except Exception:
                result[ticker] = []
        return result
    except Exception:
        return {t: [] for t in tickers}


def compute_sentiment_score(
    news_items: list,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Optional[float]:
    """Compute time-decay-weighted average sentiment for one ticker.

    Parameters
    ----------
    news_items : list of dicts
        Each dict has keys: sentiment_score, confidence, timestamp.
    confidence_threshold : float
        Minimum FinBERT confidence to include an article.

    Returns
    -------
    float or None
        Weighted average sentiment in [-1, +1], or None if insufficient data.
    """
    now = datetime.now(timezone.utc)
    decay_lambda = math.log(2) / DECAY_HALFLIFE_DAYS

    weighted_sum = 0.0
    weight_total = 0.0

    for item in news_items:
        score = item.get("sentiment_score")
        conf = item.get("confidence", 0.0)
        ts = item.get("timestamp")

        if score is None or conf is None:
            continue
        if conf < confidence_threshold:
            continue

        # Time decay weight
        if ts is not None:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = max(0, (now - ts).total_seconds() / 86400)
            decay = math.exp(-decay_lambda * age_days)
        else:
            decay = 0.5  # unknown age → assign moderate weight

        w = conf * decay
        weighted_sum += score * w
        weight_total += w

    if weight_total < 0.01 or weight_total == 0:
        return None

    count = sum(
        1 for item in news_items
        if (item.get("confidence") or 0) >= confidence_threshold
    )
    if count < MIN_ARTICLES:
        return None

    return weighted_sum / weight_total


def compute_sentiment_forecast(
    ticker: str,
    news_items: Optional[list] = None,
) -> Optional[float]:
    """Compute Carver-scaled sentiment forecast for one ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol.
    news_items : list or None
        Pre-fetched news items. If None, fetches from database.

    Returns
    -------
    float or None
        Forecast in [-20, +20], or None if insufficient data.
    """
    if news_items is None:
        news_items = _fetch_recent_news(ticker)

    conf_threshold = _get_confidence_threshold()
    raw_score = compute_sentiment_score(news_items, conf_threshold)

    if raw_score is None:
        return None

    # Scale raw [-1, +1] to forecast space
    # raw_score 0.5 → very bullish → forecast ≈ 6.25
    # raw_score -0.8 → very bearish → forecast ≈ -10
    scaled = raw_score * SCALAR_SENTIMENT
    return float(cap_forecast(scaled))


def compute_sentiment_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute sentiment-based forecasts for all symbols in the universe.

    This is the main entry point called by carver_pipeline.py.
    Uses the same Dict[str, float] return pattern as other forecast modules.

    Parameters
    ----------
    ohlcv_cache : Dict[str, pd.DataFrame]
        Symbol → OHLCV DataFrame. Used only for the symbol list;
        actual sentiment data comes from the news database.

    Returns
    -------
    Dict[str, float]
        {symbol: forecast} where forecast is in [-20, +20].
        Only symbols with sufficient news coverage are included.
    """
    tickers = list(ohlcv_cache.keys())
    if not tickers:
        return {}

    conf_threshold = _get_confidence_threshold()

    # Fetch news for all tickers
    all_news = _fetch_recent_news_batch(tickers)

    # Compute raw sentiment scores
    raw_scores: Dict[str, float] = {}
    for ticker in tickers:
        news_items = all_news.get(ticker, [])
        score = compute_sentiment_score(news_items, conf_threshold)
        if score is not None:
            raw_scores[ticker] = score

    if not raw_scores:
        logger.info("Sentiment: no tickers with sufficient news coverage")
        return {}

    # Z-score normalize across the universe for relative ranking
    values = np.array(list(raw_scores.values()))
    mean = float(np.mean(values))
    std = float(np.std(values))

    forecasts: Dict[str, float] = {}
    for ticker, score in raw_scores.items():
        if std > 0.01:
            z = (score - mean) / std
        else:
            z = score * 2.0  # If all similar, use raw × 2 as mild signal

        scaled = z * SCALAR_SENTIMENT
        forecasts[ticker] = float(cap_forecast(scaled))

    logger.info(
        "Sentiment: %d/%d tickers produced forecasts (avg=%.2f)",
        len(forecasts), len(tickers),
        np.mean(list(forecasts.values())) if forecasts else 0,
    )
    return forecasts
