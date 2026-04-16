"""
Shared FastAPI dependencies for Centurion Capital API.

Provides dependency-injection callables for database access,
authentication tokens, and service singletons.
"""

import logging
import time
from functools import lru_cache


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database service dependency
# ---------------------------------------------------------------------------

def get_db_service():
    """
    FastAPI dependency that yields the singleton DatabaseService.
    Returns None if the database is not configured.
    """
    try:
        from config import Config
        if not Config.is_database_configured():
            logger.debug("Database not configured — skipping")
            return None

        from database.service import get_database_service
        service = get_database_service()
        if service and service.is_available:
            return service
    except Exception as exc:
        logger.warning("Database unavailable: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Kite Connect session dependency — with token persistence & expiry tracking
# ---------------------------------------------------------------------------

_kite_instance = None
_kite_access_token: str | None = None
_kite_token_ts: float = 0.0          # epoch when token was obtained
_KITE_TOKEN_LIFETIME = 6 * 3600      # Kite access tokens are valid for ~6h
_KITE_REFRESH_BUFFER = 30 * 60       # re-auth 30 min before expiry

_CACHE_KEY_KITE_TOKEN = "kite:access_token"
_CACHE_KEY_KITE_TS = "kite:token_ts"


def get_kite_session():
    """
    FastAPI dependency that returns the active KiteConnect instance.
    Returns None if not authenticated.

    On first call after a restart, attempts to restore the session
    from the cache (Redis / in-memory).
    """
    global _kite_instance
    if _kite_instance is not None:
        return _kite_instance

    # Try to restore from cache (survives server restarts if Redis is up)
    _kite_instance = _restore_kite_from_cache()
    return _kite_instance


def set_kite_session(kite):
    """Store the authenticated KiteConnect instance and persist the token."""
    global _kite_instance, _kite_access_token, _kite_token_ts
    _kite_instance = kite

    if kite is not None:
        _kite_access_token = kite.access_token
        _kite_token_ts = time.time()
        _persist_kite_token(kite.access_token)
    else:
        _kite_access_token = None
        _kite_token_ts = 0.0
        _clear_kite_token()


def is_kite_token_expiring_soon() -> bool:
    """True if the Kite access token will expire within the refresh buffer."""
    if _kite_token_ts == 0.0:
        return True
    elapsed = time.time() - _kite_token_ts
    return elapsed >= (_KITE_TOKEN_LIFETIME - _KITE_REFRESH_BUFFER)


def kite_token_remaining_seconds() -> int:
    """Seconds until the current Kite token expires (approximate)."""
    if _kite_token_ts == 0.0:
        return 0
    remaining = _KITE_TOKEN_LIFETIME - (time.time() - _kite_token_ts)
    return max(0, int(remaining))


def _persist_kite_token(access_token: str):
    """Save the access token + timestamp to the unified cache."""
    try:
        from infrastructure.cache import cache
        cache.set(_CACHE_KEY_KITE_TOKEN, access_token, ttl=_KITE_TOKEN_LIFETIME)
        cache.set(_CACHE_KEY_KITE_TS, time.time(), ttl=_KITE_TOKEN_LIFETIME)
    except Exception as exc:
        logger.debug("Failed to persist Kite token to cache: %s", exc)


def _clear_kite_token():
    """Remove cached Kite token on logout / invalidation."""
    try:
        from infrastructure.cache import cache
        cache.delete(_CACHE_KEY_KITE_TOKEN)
        cache.delete(_CACHE_KEY_KITE_TS)
    except Exception:
        pass


def _restore_kite_from_cache():
    """Attempt to recreate a KiteConnect instance from cached access_token."""
    global _kite_access_token, _kite_token_ts
    try:
        from infrastructure.cache import cache
        token = cache.get(_CACHE_KEY_KITE_TOKEN)
        ts = cache.get(_CACHE_KEY_KITE_TS)
        if not token:
            return None

        # Check if token is still valid (not expired)
        ts = float(ts) if ts else 0.0
        if ts > 0 and (time.time() - ts) >= _KITE_TOKEN_LIFETIME:
            logger.info("Cached Kite token expired — clearing")
            _clear_kite_token()
            return None

        import os
        from kiteconnect import KiteConnect
        from kite_connect.core.config import API_KEY

        pool_cfg = {"pool_maxsize": int(os.getenv("KITE_POOL_MAXSIZE", "20"))}
        kite = KiteConnect(api_key=API_KEY, pool=pool_cfg)
        kite.set_access_token(token)

        # Verify the token is still valid with a lightweight API call
        kite.profile()

        _kite_access_token = token
        _kite_token_ts = ts
        logger.info("Restored Kite session from cache (%.0f min remaining)",
                     (_KITE_TOKEN_LIFETIME - (time.time() - ts)) / 60)
        return kite
    except Exception as exc:
        logger.debug("Could not restore Kite session from cache: %s", exc)
        _clear_kite_token()
        return None


# ---------------------------------------------------------------------------
# RAG Query Engine dependency (lazy singleton)
# ---------------------------------------------------------------------------

_rag_engine = None


def get_rag_engine():
    """
    FastAPI dependency that returns the RAG QueryEngine singleton.
    Lazily initialised on first call.
    """
    global _rag_engine
    if _rag_engine is None:
        try:
            from rag_pipeline.config import RAGConfig
            from rag_pipeline.core.query_engine import RAGQueryEngine
            from rag_pipeline.storage.vector_store import VectorStoreManager
            config = RAGConfig()
            vs = VectorStoreManager(config)
            _rag_engine = RAGQueryEngine(vector_store=vs, config=config)
            logger.info("RAG QueryEngine initialised")
        except Exception as exc:
            logger.error("Failed to initialise RAG engine: %s", exc)
            return None
    return _rag_engine


# ---------------------------------------------------------------------------
# Trading system dependency (lazy singleton)
# ---------------------------------------------------------------------------

_trading_system = None


def get_trading_system(tickers=None):
    """
    FastAPI dependency that returns an AlgoTradingSystem instance.
    """
    global _trading_system
    if _trading_system is None or tickers:
        try:
            from main import AlgoTradingSystem
            _trading_system = AlgoTradingSystem(tickers=tickers)
        except Exception as exc:
            logger.error("Failed to initialise trading system: %s", exc)
            return None
    return _trading_system


# ---------------------------------------------------------------------------
# Config dependency
# ---------------------------------------------------------------------------

@lru_cache()
def get_config():
    """Return the global Config singleton."""
    from config import Config
    return Config
