"""
Scraper Cache — Session-Scoped Caching, Deduplication & Adaptive Rate Limiting.

Provides a single integration point between the scraper layer, the
in-process ``SessionCache``, and the database-backed ``DataFreshness``
table.  Any component that fetches external data should go through this
module so that:

* **Within a session** — repeated requests for the same (ticker, source)
  pair are served from the in-process ``SessionCache`` (sub-ms).
* **Across restarts** — the ``DataFreshness`` table is consulted so that
  data scraped minutes before a restart isn't immediately re-fetched.
* **Content-hash dedup** — articles whose SHA-256 hash has already been
  seen (in this session or in the DB) are silently dropped *before*
  they enter the analysis pipeline.
* **Adaptive rate limiting** — per-source error counters trigger
  exponential backoff so that a misbehaving scraper doesn't burn
  through API quotas.

Design constraints
──────────────────
* No external cache infrastructure (Redis, Memcached) required.
* Thread-safe for Streamlit's rare background-thread usage.
* Gracefully degrades when the database is unavailable — falls back to
  in-process cache only.
"""

import hashlib
import logging
import time
import threading
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple

from models import NewsItem
from config import Config
from services.cache import get_session_cache, SessionCache

logger = logging.getLogger(__name__)

# ─── Adaptive Rate Limiter ───────────────────────────────────────────


class _SourceRateLimiter:
    """
    Tracks per-source request timestamps and error counts to enforce
    adaptive backoff.

    Rules:
    * A minimum inter-request interval (``base_delay``) is always
      enforced.
    * On each consecutive error, the delay doubles (capped at
      ``max_delay``).
    * A successful request resets the error multiplier.
    """

    __slots__ = (
        "base_delay", "max_delay", "_lock",
        "_last_request_at", "_consecutive_errors",
    )

    def __init__(self, base_delay: float = 0.5, max_delay: float = 30.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._lock = threading.Lock()
        self._last_request_at: float = 0.0
        self._consecutive_errors: int = 0

    @property
    def current_delay(self) -> float:
        """Compute the current required delay with exponential backoff."""
        backoff = self.base_delay * (2 ** min(self._consecutive_errors, 6))
        return min(backoff, self.max_delay)

    def time_until_next(self) -> float:
        """Seconds the caller must wait before the next request."""
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.current_delay - elapsed
            return max(0.0, remaining)

    def record_request(self) -> None:
        """Mark that a request was just issued."""
        with self._lock:
            self._last_request_at = time.monotonic()

    def record_success(self) -> None:
        """Reset backoff on success."""
        with self._lock:
            self._consecutive_errors = 0

    def record_error(self) -> None:
        """Bump the backoff multiplier."""
        with self._lock:
            self._consecutive_errors += 1
            logger.debug(
                "Rate-limiter backoff now %.1fs (%d consecutive errors)",
                self.current_delay, self._consecutive_errors,
            )

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors


# ─── Content-Hash Deduplicator ───────────────────────────────────────


class _ContentDeduplicator:
    """
    Maintains a set of content hashes (SHA-256 of title + url) seen
    during this session.  Optionally checks the database ``news_items``
    table for historical duplicates.
    """

    def __init__(self):
        self._seen: Set[str] = set()
        self._lock = threading.Lock()
        # Pre-load hashes from DB on first use (lazy)
        self._db_loaded = False

    @staticmethod
    def compute_hash(title: str, url: str = "") -> str:
        return hashlib.sha256(
            (title.strip().lower() + "|" + (url or "").strip()).encode()
        ).hexdigest()

    def is_duplicate(self, title: str, url: str = "") -> bool:
        h = self.compute_hash(title, url)
        with self._lock:
            if h in self._seen:
                return True
            self._seen.add(h)
            return False

    def load_from_db(self) -> None:
        """Best-effort load of recent content hashes from the DB."""
        if self._db_loaded:
            return
        self._db_loaded = True
        try:
            from database.service import get_database_service
            db = get_database_service()
            if not db.is_available:
                return
            with db.session_scope() as session:
                from database.models import NewsItem as DBNewsItem
                from sqlalchemy import func
                from datetime import datetime, timedelta
                cutoff = datetime.utcnow() - timedelta(days=2)
                rows = session.query(DBNewsItem.content_hash).filter(
                    DBNewsItem.created_at >= cutoff,
                    DBNewsItem.content_hash.isnot(None),
                ).all()
                with self._lock:
                    for (h,) in rows:
                        self._seen.add(h)
                logger.info("Deduplicator pre-loaded %d hashes from DB", len(rows))
        except Exception as exc:
            logger.debug("Could not pre-load hashes from DB: %s", exc)

    @property
    def seen_count(self) -> int:
        return len(self._seen)


# ─── ScraperCache (public API) ───────────────────────────────────────


class ScraperCache:
    """
    Unified caching / dedup / rate-limiting facade for the scraper layer.

    Typical usage inside ``USNewsAggregator``::

        cache = ScraperCache()

        # Before scraping
        if cache.has_cached(ticker, source):
            return cache.get_cached(ticker, source)

        # After scraping
        items = cache.deduplicate(items)
        cache.store(ticker, source, items)
        cache.record_success(source)
    """

    _instance: Optional["ScraperCache"] = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialised = False
                    cls._instance = inst
        return cls._instance

    def __init__(self):
        if self._initialised:
            return

        self._session_cache: SessionCache = get_session_cache()
        self._deduplicator = _ContentDeduplicator()

        # Per-source rate limiters (keyed by source display name)
        self._limiters: Dict[str, _SourceRateLimiter] = {}
        self._limiter_defaults: Dict[str, float] = {
            # US sources
            "Finviz": 1.5,
            "Investing.com": 1.0,
            "TradingView": 0.5,
            "Yahoo Finance": 0.3,
            # Indian sources
            "Moneycontrol": 1.0,
            "Economic Times": 1.0,
            "Mint": 0.8,
            "Business Standard": 0.8,
            "Hindu Business Line": 0.8,
            "Zerodha Pulse": 0.5,
            "NDTV Profit": 0.8,
            "Google News India": 0.5,
        }

        self._news_ttl = timedelta(minutes=Config.NEWS_CACHE_TTL_MINUTES)
        self._initialised = True
        logger.info("ScraperCache initialised (news TTL=%s min)", Config.NEWS_CACHE_TTL_MINUTES)

    # ── Rate Limiter Management ──────────────────────────────────────

    def _get_limiter(self, source: str) -> _SourceRateLimiter:
        if source not in self._limiters:
            base = self._limiter_defaults.get(source, 0.5)
            self._limiters[source] = _SourceRateLimiter(base_delay=base)
        return self._limiters[source]

    def time_until_next(self, source: str) -> float:
        """Seconds to wait before issuing another request to *source*."""
        return self._get_limiter(source).time_until_next()

    def record_request(self, source: str) -> None:
        """Call immediately *before* issuing a request."""
        self._get_limiter(source).record_request()

    def record_success(self, source: str) -> None:
        """Call on a successful scrape to reset backoff."""
        self._get_limiter(source).record_success()

    def record_error(self, source: str) -> None:
        """Call on a scraper error to increase backoff."""
        self._get_limiter(source).record_error()
        self._record_db_error(source)

    # ── In-Process Cache (via SessionCache) ──────────────────────────

    def _cache_key(self, ticker: str, source: str) -> str:
        return f"{ticker.upper()}|{source}"

    def has_cached(self, ticker: str, source: str) -> bool:
        """Check if we already have cached news for (ticker, source)."""
        return self._session_cache.has("scraper_news", self._cache_key(ticker, source))

    def get_cached(self, ticker: str, source: str) -> List[NewsItem]:
        """Return cached news items or empty list."""
        return self._session_cache.get(
            "scraper_news", self._cache_key(ticker, source), default=[]
        )

    def store(self, ticker: str, source: str, items: List[NewsItem]) -> None:
        """Store news items in the per-(ticker, source) cache slot."""
        self._session_cache.put(
            "scraper_news",
            self._cache_key(ticker, source),
            items,
            ttl=self._news_ttl,
        )

    def get_all_cached_for_ticker(self, ticker: str) -> List[NewsItem]:
        """
        Aggregate cached news across all sources for a single ticker.
        """
        all_items: List[NewsItem] = []
        bucket = self._session_cache.get_all("scraper_news")
        prefix = f"{ticker.upper()}|"
        for key, items in bucket.items():
            if key.startswith(prefix):
                all_items.extend(items)
        return all_items

    def get_cached_tickers(self) -> Set[str]:
        """Return tickers that have *any* cached scraper data."""
        keys = self._session_cache.keys("scraper_news")
        return {k.split("|")[0] for k in keys}

    # ── Content-Hash Deduplication ───────────────────────────────────

    def deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        """
        Remove items whose content hash was already seen this session
        (or exists in the DB).  Lazily pre-loads DB hashes on first call.
        """
        self._deduplicator.load_from_db()
        unique: List[NewsItem] = []
        for item in items:
            if not self._deduplicator.is_duplicate(item.title, item.url):
                unique.append(item)
        dropped = len(items) - len(unique)
        if dropped:
            logger.debug("Dedup dropped %d duplicate article(s)", dropped)
        return unique

    # ── Cross-Session Freshness (DataFreshness table) ────────────────

    def is_fresh(self, ticker: str, max_age_minutes: int = None) -> bool:
        """
        Check the DB ``data_freshness`` table for a recent 'news' fetch
        for *ticker*.  Returns ``True`` if the data is still fresh.
        Falls back to ``False`` (= stale) when the DB is unavailable.
        """
        if max_age_minutes is None:
            max_age_minutes = Config.NEWS_CACHE_TTL_MINUTES
        try:
            from database.service import get_database_service
            return get_database_service().check_freshness(
                ticker, "news", max_age_minutes
            )
        except Exception:
            return False

    def record_fetch_to_db(
        self,
        ticker: str,
        record_count: int = 0,
        fetch_seconds: float = 0.0,
    ) -> None:
        """
        Persist a successful news fetch to the ``data_freshness`` table
        so that freshness survives process restarts.
        """
        try:
            from database.service import get_database_service
            get_database_service().record_fetch(
                ticker,
                data_type="news",
                record_count=record_count,
                fetch_seconds=fetch_seconds,
                refresh_minutes=Config.NEWS_CACHE_TTL_MINUTES,
            )
        except Exception as exc:
            logger.debug("Could not record fetch to DB: %s", exc)

    def _record_db_error(self, source: str) -> None:
        """Log a scraper error to ``data_freshness`` for the source."""
        try:
            from database.service import get_database_service
            get_database_service().record_fetch_error(
                ticker=source,  # use source name as pseudo-ticker for tracking
                data_type="scraper_health",
                error=f"Consecutive errors: {self._get_limiter(source).consecutive_errors}",
            )
        except Exception:
            pass

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        """Return cache + rate-limiter diagnostics."""
        return {
            "cached_tickers": len(self.get_cached_tickers()),
            "content_hashes": self._deduplicator.seen_count,
            "rate_limiters": {
                src: {
                    "delay": round(lim.current_delay, 2),
                    "errors": lim.consecutive_errors,
                }
                for src, lim in self._limiters.items()
            },
        }


def get_scraper_cache() -> ScraperCache:
    """Return the global ``ScraperCache`` singleton."""
    return ScraperCache()
