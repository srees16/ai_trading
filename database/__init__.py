# Centurion Capital LLC - Database Module
"""
Enterprise-grade database layer with PostgreSQL + TimescaleDB support.

This module provides:
- Connection pooling and management
- SQLAlchemy ORM models with TimescaleDB hypertables
- Repository pattern for data access
- Service layer for business logic
"""

from database.connection import DatabaseManager, get_db_session
from database.models import (
    Base,
    StockSignal,
    FundamentalMetric,
    BacktestResult,
    NewsItem,
    AnalysisRun
)
from database.repositories import (
    SignalRepository,
    FundamentalRepository,
    BacktestRepository,
    NewsRepository,
    AnalysisRepository
)
from database.service import DatabaseService, get_database_service

__all__ = [
    # Connection
    'DatabaseManager',
    'get_db_session',
    # Models
    'Base',
    'StockSignal',
    'FundamentalMetric', 
    'BacktestResult',
    'NewsItem',
    'AnalysisRun',
    # Repositories
    'SignalRepository',
    'FundamentalRepository',
    'BacktestRepository',
    'NewsRepository',
    'AnalysisRepository',
    # Service
    'DatabaseService',
    'get_database_service',
]
