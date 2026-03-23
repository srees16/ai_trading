"""
Configuration Module for Centurion Capital LLC.

Centralized configuration settings for all system components.
All values can be overridden via environment variables with
the CENTURION_ prefix.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Global configuration with sensible defaults."""
    
    # =================================================================
    # Sentiment Analysis
    # =================================================================
    SENTIMENT_MODEL: str = "ProsusAI/finbert"
    SENTIMENT_HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    SENTIMENT_CONFIDENCE_FLOOR: float = 0.70  # discard scores below this confidence
    
    # =================================================================
    # Storage / Output
    # =================================================================
    OUTPUT_FILE: str = "daily_stock_news.xlsx"
    APPEND_MODE: bool = True
    
    # =================================================================
    # Web Scraping
    # =================================================================
    REQUEST_TIMEOUT: int = 10  # seconds
    MAX_CONCURRENT_REQUESTS: int = 5  # enforced via asyncio.Semaphore in aggregator
    USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    )
    
    # =================================================================
    # Session Cache
    # =================================================================
    CACHE_TTL_MINUTES: int = int(os.getenv("CENTURION_CACHE_TTL_MINUTES", "30"))
    # News cache has a shorter TTL than metrics since news is more volatile
    NEWS_CACHE_TTL_MINUTES: int = int(os.getenv("CENTURION_NEWS_CACHE_TTL_MINUTES", "30"))
    METRICS_CACHE_TTL_MINUTES: int = int(os.getenv("CENTURION_METRICS_CACHE_TTL_MINUTES", "30"))
    
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
    ADX_PERIOD: int = 14              # Average Directional Index
    ADX_TREND_THRESHOLD: float = 20.0 # ADX >= 20 → trending market
    OBV_SMA_PERIOD: int = 20          # On-Balance Volume smoothing
    VOLUME_SMA_PERIOD: int = 20       # Volume moving average for confirmation
    
    # =================================================================
    # Transaction Costs (round-trip, as fraction)
    # =================================================================
    TRANSACTION_COST_IND: float = 0.0013   # 13 bps NSE (STT + exchange + stamp + GST)
    TRANSACTION_COST_US: float = 0.001     # 10 bps US equities
    SLIPPAGE_MODEL_IND_BPS: float = 20.0   # 20 bps assumed slippage for NSE mid-caps
    SLIPPAGE_MODEL_US_BPS: float = 5.0     # 5 bps for US large-cap

    # =================================================================
    # Paper Trading Mode
    # =================================================================
    PAPER_TRADE_MODE: bool = False          # Set True to route orders to PaperTrader

    # =================================================================
    # Signal Freshness (data staleness gate)
    # =================================================================
    SIGNAL_FRESHNESS_MAX_HOURS: int = 4    # Discard signals older than 4 hours
    
    # =================================================================
    # Earnings Blackout Window
    # =================================================================
    EARNINGS_BLACKOUT_DAYS_BEFORE: int = 2  # Suppress BUY signals 2 days before
    EARNINGS_BLACKOUT_DAYS_AFTER: int = 1   # Suppress BUY signals 1 day after
    
    # =================================================================
    # NSE Circuit Breaker
    # =================================================================
    CIRCUIT_BREAKER_PCT: float = 0.20      # 20% daily move → circuit limit hit

    # =================================================================
    # VIX Regime Gate
    # =================================================================
    VIX_CAUTION_THRESHOLD: float = 20.0    # India VIX > 20 → reduce position sizes
    VIX_PANIC_THRESHOLD: float = 25.0      # India VIX > 25 → suppress new BUY signals
    VIX_POSITION_SCALE: float = 0.5        # Scale factor when VIX in caution zone
    NIFTY_BENCHMARK_TICKER: str = "^NSEI"  # NIFTY 50 index ticker for benchmarking

    # =================================================================
    # Minimum Strategy Quality Floor
    # =================================================================
    MIN_STRATEGY_SHARPE: float = 0.3       # Exclude strategies with Sharpe < 0.3 from voting
    
    # =================================================================
    # Sector Concentration Limit
    # =================================================================
    MAX_SECTOR_EXPOSURE_PCT: float = 0.40  # Max 40% capital in one sector
    MAX_TRADES_PER_SECTOR: int = 3         # Max 3 open trades per sector
    
    # =================================================================
    # Decision Engine Weights (must sum to 1.0)
    # Fundamentals + Technicals + Macro only — no sentiment.
    # =================================================================
    FUNDAMENTAL_WEIGHT: float = 0.40
    TECHNICAL_WEIGHT: float = 0.40
    MACRO_WEIGHT: float = 0.20
    
    # =================================================================
    # Decision Thresholds
    # Tightened from 0.4/0.7 to generate more actionable signals:
    # most stocks cluster in 0.1–0.3 under the averaging mechanics.
    # =================================================================
    STRONG_BUY_THRESHOLD: float = 0.55
    BUY_THRESHOLD: float = 0.30
    SELL_THRESHOLD: float = -0.30
    STRONG_SELL_THRESHOLD: float = -0.55
    
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
    # NSE Sector Mapping (NIFTY 50 + NIFTY Next 50 constituents)
    # Shared across risk_engine, portfolio_analyzer, and screener.
    # =================================================================
    NSE_SECTOR_MAP: Dict[str, str] = {
        # IT
        "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
        "TECHM": "IT", "LTIM": "IT", "COFORGE": "IT", "MPHASIS": "IT",
        "PERSISTENT": "IT", "LTTS": "IT",
        # Financials — Banks
        "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials",
        "KOTAKBANK": "Financials", "AXISBANK": "Financials", "INDUSINDBK": "Financials",
        "BANKBARODA": "Financials", "PNB": "Financials", "IDFCFIRSTB": "Financials",
        "FEDERALBNK": "Financials", "CANBK": "Financials", "AUBANK": "Financials",
        "BANDHANBNK": "Financials",
        # Financials — NBFC / Insurance
        "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials",
        "HDFCLIFE": "Financials", "SBILIFE": "Financials", "ICICIGI": "Financials",
        "CHOLAFIN": "Financials", "SHRIRAMFIN": "Financials",
        # Energy / O&G
        "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
        "POWERGRID": "Energy", "ADANIGREEN": "Energy", "ADANIENSOL": "Energy",
        "TATAPOWER": "Energy", "BPCL": "Energy", "IOC": "Energy",
        "ADANIENT": "Energy", "COALINDIA": "Energy", "GAIL": "Energy",
        "HINDPETRO": "Energy", "JSWENERGY": "Energy",
        # Auto
        "MARUTI": "Auto", "M&M": "Auto", "TATAMOTORS": "Auto",
        "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
        "BOSCHLTD": "Auto", "TVSMOTOR": "Auto",
        # Consumer / FMCG
        "HINDUNILVR": "Consumer", "ITC": "Consumer", "NESTLEIND": "Consumer",
        "TITAN": "Consumer", "BRITANNIA": "Consumer", "DABUR": "Consumer",
        "GODREJCP": "Consumer", "COLPAL": "Consumer", "TRENT": "Consumer",
        "MARICO": "Consumer", "UNITDSPR": "Consumer",
        # Pharma / Healthcare
        "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
        "DIVISLAB": "Pharma", "APOLLOHOSP": "Pharma", "MAXHEALTH": "Pharma",
        "TORNTPHARM": "Pharma",
        # Metals / Materials
        "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
        "VEDL": "Metals", "ULTRACEMCO": "Metals", "GRASIM": "Metals",
        "SHREECEM": "Metals", "AMBUJACEM": "Metals", "ADANIPORTS": "Metals",
        # Telecom
        "BHARTIARTL": "Telecom",
        # Infra / Capital Goods
        "LT": "Infra", "SIEMENS": "Infra", "ABB": "Infra",
        "HAL": "Infra", "BEL": "Infra",
        # Consumer Tech
        "ZOMATO": "Consumer Tech", "PAYTM": "Consumer Tech",
        "DMART": "Retail", "NAUKRI": "Consumer Tech",
        # Chemicals
        "PIDILITIND": "Chemicals", "SRF": "Chemicals",
    }

    # =================================================================
    # News Keywords for Categorization
    # =================================================================
    BREAKING_KEYWORDS: List[str] = ["breaking", "urgent", "alert", "just in"]
    DEALS_KEYWORDS: List[str] = ["merger", "acquisition", "deal", "buyout", "takeover"]
    MACRO_KEYWORDS: List[str] = ["fed", "interest rate", "inflation", "gdp", "unemployment", "treasury"]
    INDIA_MACRO_KEYWORDS: List[str] = ["rbi", "repo rate", "inflation", "gdp", "fii", "dii", "nifty", "sensex", "rupee", "gst"]
    EARNINGS_KEYWORDS: List[str] = ["earnings", "quarterly", "q1", "q2", "q3", "q4", "revenue", "profit"]
    
    # =================================================================
    # Database Configuration (PostgreSQL + TimescaleDB)
    # =================================================================
    
    # Database connection settings (override via environment variables)
    DB_HOST = os.getenv("CENTURION_DB_HOST", "localhost")
    DB_PORT = int(os.getenv("CENTURION_DB_PORT", "9003"))
    DB_NAME = os.getenv("CENTURION_DB_NAME", "centurion_trading")
    DB_USER = os.getenv("CENTURION_DB_USER", "")
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
