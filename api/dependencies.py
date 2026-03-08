"""
Shared FastAPI dependencies for Centurion Capital API.

Provides dependency-injection callables for database access,
authentication tokens, and service singletons.
"""

import logging
from functools import lru_cache
from typing import Generator, Optional

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
        from database.service import get_database_service
        service = get_database_service()
        if service and service.is_available:
            return service
    except Exception as exc:
        logger.warning("Database unavailable: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Kite Connect session dependency
# ---------------------------------------------------------------------------

_kite_instance = None


def get_kite_session():
    """
    FastAPI dependency that returns the active KiteConnect instance.
    Returns None if not authenticated.
    """
    global _kite_instance
    return _kite_instance


def set_kite_session(kite):
    """Store the authenticated KiteConnect instance."""
    global _kite_instance
    _kite_instance = kite


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
            config = RAGConfig()
            _rag_engine = RAGQueryEngine(config)
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
