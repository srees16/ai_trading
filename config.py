"""
Configuration Module for Centurion Capital LLC.

Centralized configuration settings for all system components.
All values can be overridden via environment variables with
the CENTURION_ prefix.
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Global configuration with sensible defaults."""
    
    # =================================================================
    # Sentiment Analysis
    # =================================================================
    SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"
    SENTIMENT_HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    
    # =================================================================
    # Storage / Output
    # =================================================================
    OUTPUT_FILE: str = "daily_stock_news.xlsx"
    APPEND_MODE: bool = True
    
    # =================================================================
    # Web Scraping
    # =================================================================
    REQUEST_TIMEOUT: int = 10  # seconds
    MAX_CONCURRENT_REQUESTS: int = 5
    USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    )
    
    # =================================================================
    # Technical Analysis Parameters
    # =================================================================
    HISTORICAL_DAYS: int = 365
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: int = 2
    
    # =================================================================
    # Decision Engine Weights (must sum to 1.0)
    # =================================================================
    SENTIMENT_WEIGHT: float = 0.4
    FUNDAMENTAL_WEIGHT: float = 0.3
    TECHNICAL_WEIGHT: float = 0.3
    
    # =================================================================
    # Decision Thresholds
    # =================================================================
    STRONG_BUY_THRESHOLD: float = 0.7
    BUY_THRESHOLD: float = 0.4
    SELL_THRESHOLD: float = -0.4
    STRONG_SELL_THRESHOLD: float = -0.7
    
    # =================================================================
    # Notification
    # =================================================================
    NOTIFICATION_DURATION: int = 10  # seconds
    
    # =================================================================
    # Default Tickers
    # =================================================================
    DEFAULT_TICKERS: List[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "TSLA", "NVDA", "JPM", "V", "WMT"
    ]
    
    # =================================================================
    # News Keywords for Categorization
    # =================================================================
    BREAKING_KEYWORDS: List[str] = ["breaking", "urgent", "alert", "just in"]
    DEALS_KEYWORDS: List[str] = ["merger", "acquisition", "deal", "buyout", "takeover"]
    MACRO_KEYWORDS: List[str] = ["fed", "interest rate", "inflation", "gdp", "unemployment", "treasury"]
    EARNINGS_KEYWORDS: List[str] = ["earnings", "quarterly", "q1", "q2", "q3", "q4", "revenue", "profit"]
    
    # =================================================================
    # Database Configuration (PostgreSQL + TimescaleDB)
    # =================================================================
    
    # Database connection settings (override via environment variables)
    DB_HOST = os.getenv("CENTURION_DB_HOST", "localhost")
    DB_PORT = int(os.getenv("CENTURION_DB_PORT", "5432"))
    DB_NAME = os.getenv("CENTURION_DB_NAME", "centurion_trading")
    DB_USER = os.getenv("CENTURION_DB_USER", "centurion")
    DB_PASSWORD = os.getenv("CENTURION_DB_PASSWORD", "")
    
    # Connection string (can be overridden directly)
    DATABASE_URL = os.getenv(
        "CENTURION_DATABASE_URL",
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    
    # Connection pool settings
    DB_POOL_SIZE: int = int(os.getenv("CENTURION_DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("CENTURION_DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT: int = int(os.getenv("CENTURION_DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("CENTURION_DB_POOL_RECYCLE", "1800"))
    
    # Enable/disable database persistence
    DB_ENABLED: bool = os.getenv("CENTURION_DB_ENABLED", "true").lower() == "true"
    
    # TimescaleDB settings
    TIMESCALEDB_CHUNK_INTERVAL: str = os.getenv(
        "CENTURION_TIMESCALEDB_CHUNK_INTERVAL", 
        "7 days"
    )
    
    # Data retention settings
    DB_RETENTION_DAYS: int = int(os.getenv("CENTURION_DB_RETENTION_DAYS", "365"))
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get the database URL, building from components if not set directly."""
        if os.getenv("CENTURION_DATABASE_URL"):
            return os.getenv("CENTURION_DATABASE_URL")
        
        if cls.DB_PASSWORD:
            return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        else:
            return f"postgresql://{cls.DB_USER}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
    
    @classmethod
    def is_database_configured(cls) -> bool:
        """Check if database is properly configured."""
        return cls.DB_ENABLED and bool(cls.DB_PASSWORD or os.getenv("CENTURION_DATABASE_URL"))
