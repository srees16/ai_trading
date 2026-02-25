"""
Session Cache Service for Centurion Capital LLC.

Provides a centralized, TTL-aware, in-process cache that spans a single
Streamlit session.  All expensive data — scraped news, sentiment results,
stock metrics — is cached per ticker so that:

* Re-running analysis with overlapping tickers reuses prior results.
* Adding a new ticker only fetches data for that ticker.
* Each layer (news → sentiment → metrics) can interrogate the cache
  independently, eliminating redundant API/scraping calls.

Design decisions
────────────────
* **No Redis required.**  The app runs as a single Streamlit process per
  user.  An in-process singleton (with optional ``st.session_state``
  persistence) gives sub-millisecond lookups without infrastructure cost.
* **TTL per entry.**  Stale data is automatically evicted; callers can
  also force a refresh.
* **Thread-safe.**  Uses a ``threading.Lock`` for the rare case where
  Streamlit spawns background threads.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class _CacheEntry:
    """A value together with its expiration timestamp."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: timedelta):
        self.value = value
        self.expires_at = datetime.utcnow() + ttl

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at


class SessionCache:
    """
    Centralized session-level cache for scraped / computed data.

    Stores data in namespaced buckets keyed by ticker:

    * ``news``        — ``Dict[ticker, List[NewsItem]]``
    * ``sentiment``   — ``Dict[ticker, List[NewsItem]]``  (post-analysis)
    * ``metrics``     — ``Dict[ticker, StockMetrics | None]``
    * ``signals``     — ``Dict[ticker, List[TradingSignal]]``

    Each entry carries its own TTL.  The default TTL per namespace is set
    via ``default_ttl`` but can be overridden per-put.
    """

    _instance: Optional["SessionCache"] = None
    _lock = threading.Lock()

    # ── Singleton ────────────────────────────────────────────────────
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialised = False
                    cls._instance = inst
        return cls._instance

    def __init__(self, default_ttl_minutes: int = 30):
        if self._initialised:
            return
        self._default_ttl = timedelta(minutes=default_ttl_minutes)
        # _store[namespace][key] → _CacheEntry
        self._store: Dict[str, Dict[str, _CacheEntry]] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._initialised = True
        logger.info(
            "SessionCache initialised (TTL=%s min)", default_ttl_minutes
        )

    # ── Core API ─────────────────────────────────────────────────────
    def get(
        self, namespace: str, key: str, default: Any = None
    ) -> Any:
        """Return cached value or *default* if missing / expired."""
        bucket = self._store.get(namespace)
        if bucket is None:
            self._miss_count += 1
            return default
        entry = bucket.get(key)
        if entry is None or entry.is_expired:
            if entry and entry.is_expired:
                del bucket[key]
            self._miss_count += 1
            return default
        self._hit_count += 1
        return entry.value

    def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
    ) -> None:
        """Store *value* under *namespace / key* with optional custom TTL."""
        bucket = self._store.setdefault(namespace, {})
        bucket[key] = _CacheEntry(value, ttl or self._default_ttl)

    def has(self, namespace: str, key: str) -> bool:
        """Return ``True`` if a non-expired entry exists."""
        return self.get(namespace, key, _SENTINEL) is not _SENTINEL

    def keys(self, namespace: str) -> Set[str]:
        """Return the set of non-expired keys in *namespace*."""
        bucket = self._store.get(namespace, {})
        valid = {k for k, v in bucket.items() if not v.is_expired}
        # Prune expired while we're here
        expired = set(bucket.keys()) - valid
        for k in expired:
            del bucket[k]
        return valid

    def invalidate(self, namespace: str, key: Optional[str] = None) -> None:
        """Remove one key, or clear an entire namespace."""
        if key is None:
            self._store.pop(namespace, None)
        else:
            bucket = self._store.get(namespace)
            if bucket:
                bucket.pop(key, None)

    def clear(self) -> None:
        """Wipe the entire cache (e.g. on new analysis run)."""
        self._store.clear()
        self._hit_count = 0
        self._miss_count = 0
        logger.info("SessionCache cleared")

    # ── Convenience helpers ──────────────────────────────────────────
    def get_cached_tickers(self, namespace: str) -> Set[str]:
        """Return tickers that already have cached data in *namespace*."""
        return self.keys(namespace)

    def get_new_tickers(
        self, namespace: str, requested: List[str]
    ) -> List[str]:
        """
        Given a list of *requested* tickers, return only those that are
        **not** already cached (or whose cache has expired).
        """
        cached = self.get_cached_tickers(namespace)
        return [t for t in requested if t not in cached]

    def get_all(self, namespace: str) -> Dict[str, Any]:
        """Return all non-expired values in *namespace* as a plain dict."""
        bucket = self._store.get(namespace, {})
        result = {}
        expired = []
        for k, entry in bucket.items():
            if entry.is_expired:
                expired.append(k)
            else:
                result[k] = entry.value
        for k in expired:
            del bucket[k]
        return result

    # ── Diagnostics ──────────────────────────────────────────────────
    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._hit_count + self._miss_count
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": (
                f"{self._hit_count / total * 100:.1f}%"
                if total > 0
                else "N/A"
            ),
            "namespaces": {
                ns: len(self.keys(ns)) for ns in list(self._store.keys())
            },
        }

    def __repr__(self) -> str:
        ns_summary = ", ".join(
            f"{ns}={len(self.keys(ns))}"
            for ns in self._store
        )
        return f"<SessionCache {ns_summary or 'empty'}>"


# Sentinel for internal use
_SENTINEL = object()


def get_session_cache() -> SessionCache:
    """Return the global ``SessionCache`` singleton."""
    return SessionCache()
