# Centurion Capital LLC - Unified Cache Layer
"""
Redis-backed caching with in-memory fallback.

Uses Upstash Redis in production (free tier: 10K commands/day).
Falls back to a TTL-aware in-memory dict when Redis is unavailable.

Usage:
    from infrastructure.cache import cache

    # Set with TTL
    cache.set("regime:snapshot", data_dict, ttl=1800)

    # Get (returns None on miss)
    val = cache.get("regime:snapshot")

    # Delete
    cache.delete("regime:snapshot")
"""

import os
import json
import time
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing redis (works with Upstash via standard redis protocol)
# ---------------------------------------------------------------------------
try:
    import redis as _redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.debug("redis package not installed — using in-memory cache")


class _InMemoryCache:
    """Fallback TTL-aware in-memory cache."""

    def __init__(self):
        self._store: Dict[str, tuple] = {}  # key -> (value, expire_ts)

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expire_ts = entry
        if expire_ts and time.time() > expire_ts:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 0):
        expire_ts = time.time() + ttl if ttl > 0 else None
        self._store[key] = (value, expire_ts)

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def flush(self):
        self._store.clear()

    def keys(self, pattern: str = "*") -> list:
        now = time.time()
        # Simple glob: only support prefix* pattern
        prefix = pattern.rstrip("*")
        return [
            k for k, (_, exp) in self._store.items()
            if k.startswith(prefix) and (exp is None or now < exp)
        ]


class CacheService:
    """Unified cache with Redis backend and in-memory fallback."""

    def __init__(self):
        self._redis: Optional[Any] = None
        self._memory = _InMemoryCache()
        self._redis_url: Optional[str] = None  # resolved lazily
        self._prefix = "centurion:"

    def _resolve_url(self) -> str:
        """Lazily resolve Redis URL so dotenv has time to load."""
        if self._redis_url is None:
            self._redis_url = os.getenv("UPSTASH_REDIS_URL", os.getenv("REDIS_URL", ""))
        return self._redis_url

    @property
    def _client(self):
        """Lazy-init Redis connection."""
        url = self._resolve_url()
        if self._redis is None and REDIS_AVAILABLE and url:
            try:
                self._redis = _redis_lib.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                # Verify connectivity
                self._redis.ping()
                logger.info("Redis cache connected (Upstash)")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory fallback: {e}")
                self._redis = None
        return self._redis

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value. Returns None on miss."""
        full_key = self._key(key)
        client = self._client
        if client:
            try:
                raw = client.get(full_key)
                if raw is None:
                    return None
                return json.loads(raw)
            except Exception as e:
                logger.debug(f"Redis GET failed for {key}: {e}")
        return self._memory.get(full_key)

    def set(self, key: str, value: Any, ttl: int = 1800):
        """Set a cached value with TTL in seconds (default 30 min)."""
        full_key = self._key(key)
        serialized = json.dumps(value, default=str)
        client = self._client
        if client:
            try:
                client.setex(full_key, ttl, serialized) if ttl > 0 else client.set(full_key, serialized)
                return
            except Exception as e:
                logger.debug(f"Redis SET failed for {key}: {e}")
        # Fallback to in-memory
        self._memory.set(full_key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete a cached key."""
        full_key = self._key(key)
        client = self._client
        if client:
            try:
                return bool(client.delete(full_key))
            except Exception:
                pass
        return self._memory.delete(full_key)

    def flush_prefix(self, prefix: str):
        """Delete all keys matching a prefix."""
        full_prefix = self._key(prefix)
        client = self._client
        if client:
            try:
                cursor = 0
                while True:
                    cursor, keys = client.scan(cursor, match=f"{full_prefix}*", count=100)
                    if keys:
                        client.delete(*keys)
                    if cursor == 0:
                        break
                return
            except Exception:
                pass
        for k in self._memory.keys(f"{full_prefix}*"):
            self._memory.delete(k)

    @property
    def is_redis(self) -> bool:
        """True if using Redis (not in-memory fallback)."""
        return self._client is not None

    def health_check(self) -> dict:
        """Return cache health info."""
        client = self._client
        if client:
            try:
                info = client.info("memory")
                return {
                    "backend": "redis",
                    "healthy": True,
                    "used_memory_mb": round(info.get("used_memory", 0) / 1048576, 2),
                }
            except Exception as e:
                return {"backend": "redis", "healthy": False, "error": str(e)}
        return {
            "backend": "in-memory",
            "healthy": True,
            "keys": len(self._memory._store),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
cache = CacheService()
