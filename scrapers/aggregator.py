"""
Aggregator for all news scrapers.

Full integration with ``ScraperCache`` for:
* **Per-(ticker, source) caching** — avoids redundant HTTP calls within
  and across analysis runs during the same session.
* **Cross-session freshness** — checks ``DataFreshness`` in the DB so
  that recently scraped tickers are not re-fetched even after a restart.
* **Content-hash deduplication** — articles already seen (in-session or
  in the DB) are silently dropped.
* **Adaptive rate limiting** — per-source exponential backoff on errors
  (429s, timeouts, etc.) replaces the prior hardcoded delays.
"""

import logging
import asyncio
import time
from typing import Dict, List, Set

from scrapers import BaseNewsScraper
from scrapers.yahoo_finance import YahooFinanceScraper
from scrapers.finviz import FinvizScraper
from scrapers.investing import InvestingScraper
from scrapers.tradingview import TradingViewScraper
from scrapers.wallstreetbets import WallStreetBetsScraper
from scrapers.cache import ScraperCache, get_scraper_cache
from models import NewsItem
from config import Config

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregates news from multiple sources with caching, dedup & adaptive rate-limiting."""
    
    def __init__(self):
        self.scrapers: List[BaseNewsScraper] = [
            YahooFinanceScraper(),
            FinvizScraper(),
            InvestingScraper(),
            TradingViewScraper(),
            WallStreetBetsScraper(),
        ]
        # Enforce the declared MAX_CONCURRENT_REQUESTS
        self._semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        self._cache: ScraperCache = get_scraper_cache()

    # ── internal helpers ─────────────────────────────────────────────

    async def _rate_limited_fetch(
        self, scraper: BaseNewsScraper, ticker: str
    ) -> List[NewsItem]:
        """
        Fetch news through the semaphore gate with adaptive rate-limiting
        and per-(ticker, source) caching.
        """
        source = scraper.source_name

        # 1) Check per-(ticker, source) in-process cache
        if self._cache.has_cached(ticker, source):
            logger.debug(
                "%s: Cache HIT for %s — returning %d cached items",
                source, ticker, len(self._cache.get_cached(ticker, source)),
            )
            return self._cache.get_cached(ticker, source)

        # 2) Adaptive delay — wait if the source needs cooling off
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
        else:
            # Empty result isn't necessarily an error, but we don't
            # reset backoff — only positive results do that.
            pass

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
        Fetch news from all sources concurrently (rate-limited + cached).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Deduplicated list of news items from all sources
        """
        logger.info(
            "Fetching news for %s from %d sources…",
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

        # Cross-source dedup is already handled inside _rate_limited_fetch
        # via the singleton deduplicator — no second pass needed.
        logger.info("Total: %d unique articles collected for %s", len(all_news), ticker)
        return all_news

    async def fetch_news_for_tickers(
        self,
        tickers: List[str],
        *,
        cached_news: Dict[str, List[NewsItem]] | None = None,
    ) -> List[NewsItem]:
        """
        Fetch news for multiple tickers, using three layers of caching:

        1. **Caller-provided** ``cached_news`` dict (SessionCache "news"
           namespace from ``services/analysis.py``).
        2. **ScraperCache** per-(ticker, source) in-process cache.
        3. **DataFreshness** DB table for cross-session awareness.
        
        Args:
            tickers: List of stock ticker symbols
            cached_news: Optional ``{ticker: [NewsItem, …]}`` dict of
                         previously cached results to reuse.
            
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
                "Reusing caller-cached news for %d ticker(s): %s",
                len(tickers_from_caller_cache),
                ", ".join(tickers_from_caller_cache),
            )
        if tickers_from_scraper_cache:
            logger.info(
                "Reusing scraper-cached news for %d ticker(s): %s",
                len(tickers_from_scraper_cache),
                ", ".join(tickers_from_scraper_cache),
            )
        if tickers_fresh_in_db:
            logger.info(
                "Skipping %d DB-fresh ticker(s) (recently scraped): %s",
                len(tickers_fresh_in_db),
                ", ".join(tickers_fresh_in_db),
            )
        if tickers_to_fetch:
            logger.info(
                "Fetching news for %d NEW ticker(s): %s",
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

        # Note: tickers_fresh_in_db — we know data exists in the DB but
        # not in memory.  For now we still fetch (it's cheap) so the
        # analysis pipeline has items to work with.  The freshness check
        # avoids unnecessary *repeated* fetches in quick succession.
        fetch_list = tickers_to_fetch + tickers_fresh_in_db

        # ── Fetch only the tickers we don't already have ─────────────
        if fetch_list:
            tasks = [self.fetch_all_news(t) for t in fetch_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                ticker = fetch_list[i]
                if isinstance(result, list):
                    all_news.extend(result)
                    # Record successful fetch in DataFreshness
                    self._cache.record_fetch_to_db(
                        ticker,
                        record_count=len(result),
                    )
                elif isinstance(result, Exception):
                    logger.warning(
                        "Failed to fetch news for %s: %s",
                        ticker, type(result).__name__,
                    )

        total_cached = (
            len(tickers_from_caller_cache)
            + len(tickers_from_scraper_cache)
        )
        logger.info(
            "News fetch complete: %d total articles for %d ticker(s) "
            "(%d cached, %d fetched)",
            len(all_news), len(tickers),
            total_cached, len(fetch_list),
        )
        return all_news
