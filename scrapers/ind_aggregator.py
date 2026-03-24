# Centurion Capital LLC - Indian News Aggregator
"""
Aggregator for Indian financial news scrapers.

Mirrors the ``USNewsAggregator`` for US stocks but uses Indian-specific
news sources: Moneycontrol, Economic Times, Mint, Business Standard,
Hindu Business Line, Zerodha Pulse, and NDTV Profit.

Shares the same ``ScraperCache`` singleton for caching, deduplication,
and adaptive rate-limiting.
"""

import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd

from scrapers import BaseNewsScraper
from scrapers.ind_news.moneycontrol import MoneycontrolScraper
from scrapers.ind_news.economic_times import EconomicTimesScraper
from scrapers.ind_news.livemint import LiveMintScraper
from scrapers.ind_news.business_standard import BusinessStandardScraper
from scrapers.ind_news.hindu_businessline import HinduBusinessLineScraper
from scrapers.ind_news.zerodha_pulse import ZerodhaPulseScraper
from scrapers.ind_news.ndtv_profit import NDTVProfitScraper
from scrapers.ind_news.google_news_india import GoogleNewsIndiaScraper
from scrapers.morningstar import MorningstarScraper
from scrapers.cache import ScraperCache, get_scraper_cache
from models import NewsItem
from config import Config

logger = logging.getLogger(__name__)

# Source reliability weights for importance ranking (0-1 scale)
_SOURCE_WEIGHTS: Dict[str, float] = {
    "Moneycontrol": 0.90,
    "Economic Times": 0.85,
    "Morningstar": 0.85,
    "Mint": 0.80,
    "Business Standard": 0.80,
    "Hindu Business Line": 0.75,
    "Zerodha Pulse": 0.70,
    "NDTV Profit": 0.75,
    "Google News India": 0.85,
}


class IndianNewsAggregator:
    """Aggregates news from Indian financial sources with caching, dedup & rate-limiting."""

    def __init__(self):
        self.scrapers: List[BaseNewsScraper] = [
            MoneycontrolScraper(),
            EconomicTimesScraper(),
            MorningstarScraper(),
            LiveMintScraper(),
            BusinessStandardScraper(),
            HinduBusinessLineScraper(),
            ZerodhaPulseScraper(),
            NDTVProfitScraper(),
            GoogleNewsIndiaScraper(),
        ]
        self._semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        self._cache: ScraperCache = get_scraper_cache()

    # ── internal helpers ─────────────────────────────────────────────

    async def _rate_limited_fetch(
        self, scraper: BaseNewsScraper, ticker: str
    ) -> List[NewsItem]:
        """Fetch news through semaphore with adaptive rate-limiting and caching."""
        source = scraper.source_name

        # 1) Check per-(ticker, source) in-process cache
        if self._cache.has_cached(ticker, source):
            logger.debug(
                "%s: Cache HIT for %s — returning %d cached items",
                source, ticker, len(self._cache.get_cached(ticker, source)),
            )
            return self._cache.get_cached(ticker, source)

        # 2) Adaptive delay
        wait = self._cache.time_until_next(source)
        if wait > 0:
            await asyncio.sleep(wait)

        # 3) Acquire semaphore and fetch
        async with self._semaphore:
            self._cache.record_request(source)
            t0 = time.monotonic()
            try:
                items = await scraper.fetch_news(ticker)
            except Exception as exc:
                self._cache.record_error(source)
                logger.warning(
                    "%s: Failed for %s — %s (backoff now %.1fs)",
                    source, ticker, type(exc).__name__,
                    self._cache.time_until_next(source),
                )
                return []

        elapsed = time.monotonic() - t0
        items = items if isinstance(items, list) else []

        if items:
            self._cache.record_success(source)

        # 4) Content-hash dedup before caching
        items = self._cache.deduplicate(items)

        # 5) Store in per-(ticker, source) cache
        self._cache.store(ticker, source, items)

        logger.debug(
            "%s: Fetched %d unique items for %s in %.2fs",
            source, len(items), ticker, elapsed,
        )
        return items

    # ── public API ───────────────────────────────────────────────────

    async def fetch_all_news(self, ticker: str) -> List[NewsItem]:
        """
        Fetch news from all Indian sources concurrently (rate-limited + cached).

        Args:
            ticker: Stock ticker symbol (e.g. "RELIANCE.NS")

        Returns:
            Deduplicated list of news items from all sources
        """
        logger.info(
            "Fetching Indian news for %s from %d sources…",
            ticker, len(self.scrapers),
        )
        tasks = [
            self._rate_limited_fetch(scraper, ticker)
            for scraper in self.scrapers
        ]
        results = await asyncio.gather(*tasks)

        all_news: List[NewsItem] = []
        for batch in results:
            all_news.extend(batch)

        logger.info("Total: %d unique articles collected for %s", len(all_news), ticker)
        return all_news

    async def fetch_news_for_tickers(
        self,
        tickers: List[str],
        *,
        cached_news: Dict[str, List[NewsItem]] | None = None,
    ) -> List[NewsItem]:
        """
        Fetch news for multiple tickers with three-layer caching.

        Identical interface to ``USNewsAggregator.fetch_news_for_tickers``
        so the two aggregators are interchangeable in the analysis pipeline.

        Args:
            tickers: List of stock ticker symbols
            cached_news: Optional pre-cached results to reuse

        Returns:
            Combined list of news items for all tickers
        """
        cached_news = cached_news or {}
        all_news: List[NewsItem] = []

        # ── Categorise tickers ───────────────────────────────────────
        tickers_from_caller_cache: List[str] = []
        tickers_from_scraper_cache: List[str] = []
        tickers_fresh_in_db: List[str] = []
        tickers_to_fetch: List[str] = []

        for t in tickers:
            if t in cached_news:
                tickers_from_caller_cache.append(t)
            elif self._cache.get_all_cached_for_ticker(t):
                tickers_from_scraper_cache.append(t)
            elif self._cache.is_fresh(t):
                tickers_fresh_in_db.append(t)
            else:
                tickers_to_fetch.append(t)

        # Log categorisation
        if tickers_from_caller_cache:
            logger.info(
                "Reusing caller-cached news for %d IND ticker(s): %s",
                len(tickers_from_caller_cache),
                ", ".join(tickers_from_caller_cache),
            )
        if tickers_from_scraper_cache:
            logger.info(
                "Reusing scraper-cached news for %d IND ticker(s): %s",
                len(tickers_from_scraper_cache),
                ", ".join(tickers_from_scraper_cache),
            )
        if tickers_fresh_in_db:
            logger.info(
                "Skipping %d DB-fresh IND ticker(s): %s",
                len(tickers_fresh_in_db),
                ", ".join(tickers_fresh_in_db),
            )
        if tickers_to_fetch:
            logger.info(
                "Fetching news for %d NEW IND ticker(s): %s",
                len(tickers_to_fetch),
                ", ".join(tickers_to_fetch),
            )

        # ── Gather from caller cache ─────────────────────────────────
        for t in tickers_from_caller_cache:
            all_news.extend(cached_news[t])

        # ── Gather from scraper cache ────────────────────────────────
        for t in tickers_from_scraper_cache:
            items = self._cache.get_all_cached_for_ticker(t)
            all_news.extend(items)

        fetch_list = tickers_to_fetch + tickers_fresh_in_db

        # ── Fetch only the tickers we don't already have ─────────────
        if fetch_list:
            tasks = [self.fetch_all_news(t) for t in fetch_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                ticker = fetch_list[i]
                if isinstance(result, list):
                    all_news.extend(result)
                    self._cache.record_fetch_to_db(
                        ticker,
                        record_count=len(result),
                    )
                elif isinstance(result, Exception):
                    logger.warning(
                        "Failed to fetch Indian news for %s: %s",
                        ticker, type(result).__name__,
                    )

        total_cached = (
            len(tickers_from_caller_cache)
            + len(tickers_from_scraper_cache)
        )
        logger.info(
            "Indian news fetch complete: %d total articles for %d ticker(s) "
            "(%d cached, %d fetched)",
            len(all_news), len(tickers),
            total_cached, len(fetch_list),
        )
        return all_news

    # ── ranking & convenience ────────────────────────────────────────

    @staticmethod
    def _rank_items(
        items: List[NewsItem],
        ticker: str,
        company: str = "",
    ) -> List[NewsItem]:
        """
        Compute ``relevance_score`` and assign ``importance_rank`` to each item.

        Importance combines:
        * Source reliability weight  (40 %)
        * Relevance score            (35 %)
        * Recency                    (25 %)
        """
        now = datetime.utcnow()

        for item in items:
            item.relevance_score = BaseNewsScraper.compute_relevance_score(
                item, ticker, company,
            )

            src_w = _SOURCE_WEIGHTS.get(item.source, 0.5)

            try:
                ts = item.timestamp.replace(tzinfo=None) if item.timestamp.tzinfo else item.timestamp
                age_hours = max((now - ts).total_seconds() / 3600.0, 0.0)
            except Exception:
                age_hours = 168.0  # 7 days fallback
            recency = max(1.0 - age_hours / (7 * 24), 0.0)

            item._importance = (
                0.40 * src_w
                + 0.35 * (item.relevance_score or 0.0)
                + 0.25 * recency
            )

        items.sort(key=lambda x: x._importance, reverse=True)

        for rank, item in enumerate(items, start=1):
            item.importance_rank = rank
            del item._importance

        return items

    async def get_stock_news(
        self,
        ticker: str,
        company: str = "",
    ) -> pd.DataFrame:
        """
        High-level convenience method: fetch, rank, and return a sorted DataFrame.

        Args:
            ticker: Stock ticker symbol (e.g. ``"RELIANCE.NS"``).
            company: Optional human-readable company name for relevance scoring.

        Returns:
            ``pandas.DataFrame`` sorted by ``importance_rank`` (ascending).
        """
        items = await self.fetch_all_news(ticker)
        items = self._rank_items(items, ticker, company)

        if not items:
            return pd.DataFrame()

        rows = [
            {
                "title": it.title,
                "summary": it.summary,
                "url": it.url,
                "timestamp": it.timestamp,
                "source": it.source,
                "ticker": it.ticker,
                "category": str(it.category),
                "sentiment_score": it.sentiment_score,
                "sentiment_label": str(it.sentiment_label) if it.sentiment_label else None,
                "sentiment_confidence": it.sentiment_confidence,
                "relevance_score": it.relevance_score,
                "importance_rank": it.importance_rank,
            }
            for it in items
        ]
        return pd.DataFrame(rows).sort_values("importance_rank")
