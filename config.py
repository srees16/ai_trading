"""
Configuration Module for Centurion Capital LLC.

Centralized configuration settings for all system components.
All values can be overridden via environment variables with
the CENTURION_ prefix.
"""

import os
from urllib.parse import quote_plus as _url_quote
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

_ENV = os.getenv("CENTURION_ENV", "development").lower()  # development | staging | production


class Config:
    """Global configuration with sensible defaults.

    Environment is set via CENTURION_ENV (development/staging/production).
    Production enforces stricter defaults.
    """

    ENV: str = _ENV
    DEBUG: bool = _ENV != "production"
    LOG_LEVEL: str = "DEBUG" if _ENV == "development" else "INFO"
    
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

    # Advanced TA Layer (hybrid tradingview-ta + ta library)
    TA_LOCAL_WEIGHT: float = 0.50      # Weight for local advanced indicators
    TA_TV_WEIGHT: float = 0.30         # Weight for TradingView consensus
    TA_XVAL_WEIGHT: float = 0.20       # Weight for cross-validation bonus
    TA_SKIP_TRADINGVIEW: bool = False   # Skip TV API calls (offline mode)
    SUPERTREND_PERIOD: int = 10
    SUPERTREND_MULTIPLIER: float = 3.0
    STOCH_RSI_PERIOD: int = 14
    WILLIAMS_R_PERIOD: int = 14
    CCI_PERIOD: int = 20
    MFI_PERIOD: int = 14
    ATR_PERIOD: int = 14
    KELTNER_PERIOD: int = 20
    KELTNER_ATR_MULT: float = 1.5
    CMF_PERIOD: int = 20
    
    # =================================================================
    # Transaction Costs (round-trip, as fraction)
    # =================================================================
    TRANSACTION_COST_IND: float = 0.0013   # 13 bps NSE (STT + exchange + stamp + GST)
    TRANSACTION_COST_US: float = 0.001     # 10 bps US equities
    SLIPPAGE_MODEL_IND_BPS: float = 20.0   # 20 bps assumed slippage for NSE mid-caps
    SLIPPAGE_IND_LARGECAP_BPS: float = 5.0  # NIFTY50 — very liquid
    SLIPPAGE_IND_MIDCAP_BPS: float = 20.0   # NIFTY_NEXT50 / upper mid-cap
    SLIPPAGE_IND_SMALLCAP_BPS: float = 50.0  # everything else
    SLIPPAGE_MODEL_US_BPS: float = 5.0     # 5 bps for US large-cap

    # =================================================================
    # Paper Trading Mode
    # =================================================================
    PAPER_TRADE_MODE: bool = True           # Tier 1: Paper mode ON for 4-week validation
    KILL_SWITCH: bool = False               # G2: Emergency halt — blocks ALL order placement
    SIGNAL_FRESHNESS_MAX_HOURS: int = 4      # Tier 1 Gap 5: reject OHLCV older than N hours

    # =================================================================
    # Carver Systematic Trading Framework (Robert Carver)
    # =================================================================
    CARVER_ENABLED: bool = True             # Enable Carver vol-targeted sizing (False = legacy Kelly)
    CARVER_ANNUAL_VOL_TARGET: float = 0.85  # 85% annual vol target (optimal for 50% CAGR at F&O leverage)
    CARVER_INITIAL_CAPITAL: float = 500_000.0  # Starting capital (₹)
    CARVER_DEFAULT_IDM: float = 2.0         # Instrument Diversification Multiplier (10-15 stocks)
    CARVER_MAX_LEVERAGE: float = 9.0        # F&O leverage (optimal for 50% CAGR — NIFTY stock futures)
    CARVER_INERTIA_THRESHOLD: float = 0.15  # 15% position change for re-trade (reduces churn at high leverage)
    CARVER_COST_SPEED_LIMIT: float = 3.0    # SR must exceed 3× cost drag
    CARVER_TRADE_HORIZON: str = "swing"     # "swing" (3σ bear/5σ bull) or "positional"

    # Drawdown thresholds (adjusted for high-leverage operation)
    PORTFOLIO_DRAWDOWN_WARNING: float = 0.25    # 25% DD → reduce to 70%
    PORTFOLIO_DRAWDOWN_CRITICAL: float = 0.40   # 40% DD → reduce to 50%
    PORTFOLIO_DRAWDOWN_HALT: float = 0.55       # 55% DD → halt all new trades

    # Carver — US Stocks overrides (USD-based)
    CARVER_US_ENABLED: bool = True          # Enable Carver for US stocks pipeline
    CARVER_US_INITIAL_CAPITAL: float = 10_000.0  # Starting capital ($USD)
    CARVER_US_ANNUAL_VOL_TARGET: float = 0.20    # 20% annual vol target
    CARVER_US_DEFAULT_IDM: float = 1.5      # IDM for US diversified basket
    CARVER_US_MAX_LEVERAGE: float = 1.0     # No leverage for swing equity
    CARVER_US_COST_ROUND_TRIP_PCT: float = 0.0010  # 10 bps round-trip (US zero-commission)
    CARVER_US_SPREAD_SLIPPAGE_PCT: float = 0.0005  # 5 bps spread+slippage (US large-cap)

    # =================================================================
    # Stock Universe Configuration
    # =================================================================
    # IND universe tier: "DEFAULT" (~100 NIFTY50+NEXT50),
    #   "NIFTY500" (~500), "BROAD" (~800-1200 all NSE indices)
    NSE_UNIVERSE_TIER: str = "BROAD"
    # Max symbols to process per pipeline run (0 = no limit)
    NSE_UNIVERSE_MAX_SYMBOLS: int = 0
    # OHLCV download batch size for yfinance (higher = faster but may rate-limit)
    OHLCV_DOWNLOAD_BATCH_SIZE: int = 50
    # Max parallel workers for CPU-bound signal computation
    PIPELINE_MAX_WORKERS: int = 8

    # US universe mode: "DEFAULT" (top-20), "NASDAQ100" (~100),
    #   "SP500" (~500), "NASDAQ_FULL" (~3000+)
    US_UNIVERSE_MODE: str = "NASDAQ_FULL"

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
    # Gap C6: Circuit breaker durations for all NSE tiers
    CIRCUIT_BREAKER_TIERS: list = [0.01, 0.02, 0.05, 0.10, 0.20]  # 1-20%
    CIRCUIT_BREAKER_RESET_SECONDS: int = 2700  # 45 min (NSE standard)

    # =================================================================
    # VIX Regime Gate — Gap C4: Unified thresholds
    # =================================================================
    VIX_CAUTION_THRESHOLD: float = 18.0    # India VIX > 18 → reduce position sizes
    VIX_PANIC_THRESHOLD: float = 25.0      # India VIX > 25 → suppress new BUY signals
    VIX_POSITION_SCALE: float = 0.5        # Scale factor when VIX in caution zone
    NIFTY_BENCHMARK_TICKER: str = "^NSEI"  # NIFTY 50 index ticker for benchmarking

    # =================================================================
    # Gap C3: Unified max open trades (single source of truth)
    # =================================================================
    MAX_OPEN_TRADES: int = 8               # Max concurrent positions

    # =================================================================
    # Gap C5: Time-based exit enforcement
    # =================================================================
    MAX_HOLD_DAYS_SWING: int = 15          # Max holding period for swing trades
    MAX_HOLD_DAYS_POSITIONAL: int = 60     # Max holding period for positional trades

    # =================================================================
    # Gap D3: Dynamic slippage model by market cap tier
    # =================================================================
    SLIPPAGE_LARGECAP_BPS: float = 5.0     # Large-cap (NIFTY 50) slippage
    SLIPPAGE_MIDCAP_BPS: float = 20.0      # Mid-cap (NIFTY Next 50) slippage
    SLIPPAGE_SMALLCAP_BPS: float = 50.0    # Small-cap slippage

    # =================================================================
    # HMM Regime Detection (Gap B1)
    # =================================================================
    HMM_ENABLED: bool = True               # Enable HMM regime detection
    HMM_N_STATES: int = 3                  # Number of hidden states
    HMM_MIN_CONFIDENCE: float = 0.6        # Min confidence to use HMM vs rule-based
    HMM_REFIT_DAYS: int = 30               # Refit model every N days

    # =================================================================
    # Minimum Strategy Quality Floor
    # =================================================================
    MIN_STRATEGY_SHARPE: float = 0.3       # Exclude strategies with Sharpe < 0.3 from voting
    
    # =================================================================
    # Sector Concentration Limit
    # =================================================================
    MAX_SECTOR_EXPOSURE_PCT: float = 0.30  # G13: Aligned to 30% matching risk_engine
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
        f"postgresql://{_url_quote(DB_USER)}:{_url_quote(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
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

    # =================================================================
    # RL Bot Configuration
    # =================================================================
    RL_ENABLED: bool = os.getenv("CENTURION_RL_ENABLED", "false").lower() == "true"
    RL_ALGORITHM: str = os.getenv("CENTURION_RL_ALGORITHM", "PPO")  # DQN | PPO | A2C
    RL_REWARD_TYPE: str = os.getenv("CENTURION_RL_REWARD_TYPE", "hybrid")
    RL_TOTAL_TIMESTEPS: int = int(os.getenv("CENTURION_RL_TIMESTEPS", "500000"))
    RL_LOOKBACK: int = int(os.getenv("CENTURION_RL_LOOKBACK", "60"))
    RL_TRAIN_DAYS: int = int(os.getenv("CENTURION_RL_TRAIN_DAYS", "504"))
    RL_TEST_DAYS: int = int(os.getenv("CENTURION_RL_TEST_DAYS", "63"))
    RL_WALK_FORWARD_FOLDS: int = int(os.getenv("CENTURION_RL_FOLDS", "6"))
    RL_LAYER_WEIGHT: float = float(os.getenv("CENTURION_RL_LAYER_WEIGHT", "0.15"))

    # =================================================================
    # Risk Metrics Configuration (Phase 0)
    # =================================================================
    COMPUTE_SORTINO: bool = True
    COMPUTE_CALMAR: bool = True
    COMPUTE_CVAR: bool = True
    CVAR_ALPHA: float = 0.05            # 5% tail for CVaR computation
    RISK_FREE_RATE_IND: float = 0.07    # India 10-year G-Sec yield
    RISK_FREE_RATE_US: float = 0.04     # US 10-year Treasury yield

    # =================================================================
    # Forecast Scalar Calibration
    # =================================================================
    AUTO_CALIBRATE_SCALARS: bool = True        # Enable auto-calibration of forecast scalars
    SCALAR_CALIBRATION_MAX_AGE_DAYS: int = 14  # Recalibrate if older than 14 days
    SCALAR_CALIBRATION_FILE: str = "data/calibrated_scalars.json"

    # =================================================================
    # Walk-Forward Transaction Cost Simulation
    # =================================================================
    WF_ROUND_TRIP_COST_IND: float = 0.004   # 0.40% NSE round-trip (STT + exchange + GST + slippage)
    WF_ROUND_TRIP_COST_US: float = 0.001    # 0.10% US round-trip

    # =================================================================
    # Phase 1 — Options Income
    # =================================================================
    OPTIONS_ENABLED: bool = False           # Master switch for options strategies
    OPTIONS_MAX_PORTFOLIO_PCT: float = 0.15  # Max 15% of portfolio in options premium
    OPTIONS_MAX_CONCURRENT: int = 5          # Max concurrent options positions
    OPTIONS_PROFIT_TARGET_PCT: float = 0.50  # Close at 50% of premium earned
    OPTIONS_DTE_MIN: int = 15               # Min days to expiry for new sells
    OPTIONS_DTE_MAX: int = 45               # Max days to expiry for new sells
    OPTIONS_DELTA_MAX: float = 0.30         # Max delta for short options
    OPTIONS_DELTA_ROLL_THRESHOLD: float = 0.40  # Roll when delta exceeds this
    OPTIONS_TAIL_HEDGE_ENABLED: bool = False # Enable tail hedge puts
    OPTIONS_TAIL_HEDGE_PCT: float = 0.02    # 2% of portfolio for tail hedges

    # =================================================================
    # Phase 2 — Short Selling
    # =================================================================
    SHORT_SELLING_ENABLED: bool = True       # G8: Enabled — regime-gated to bear only
    SHORT_MAX_PORTFOLIO_PCT: float = 0.20    # Max 20% of portfolio short
    SHORT_MAX_CONCURRENT: int = 3            # Max concurrent short positions
    SHORT_REGIME_REQUIRED: str = "bear"      # Only short in bear regime
    SHORT_MIN_FORECAST: float = -5.0         # Min forecast to trigger short trade plan
    SHORT_PRODUCT: str = "MIS"               # MIS (intraday) or NRML (overnight F&O)

    # =================================================================
    # Phase 3 — Leverage via Futures
    # =================================================================
    LEVERAGE_ENABLED: bool = True            # G11: Enabled — regime-adaptive leverage
    LEVERAGE_MAX: float = 1.5               # Absolute max leverage
    LEVERAGE_BULL_MAX: float = 1.3           # G11: Conservative 1.3x in bull regime
    LEVERAGE_RANGE_MAX: float = 1.15         # G11: Mild 1.15x in range-bound regime
    LEVERAGE_BEAR_MAX: float = 0.8           # Max leverage in bear regime
    FUTURES_INSTRUMENT: str = "NIFTY"        # NIFTY or BANKNIFTY
    FUTURES_LOT_SIZE: int = 25               # NIFTY lot size
    FUTURES_ROLLOVER_DAYS_BEFORE: int = 3    # Roll N days before expiry
    FUTURES_MARGIN_PCT: float = 0.12         # ~12% margin requirement
    MARGIN_WARNING_PCT: float = 0.70         # Warn at 70% margin utilisation
    MARGIN_CRITICAL_PCT: float = 0.85        # Critical at 85% margin utilisation

    # =================================================================
    # Phase 4 — Uncorrelated Alpha: Pairs Trading
    # =================================================================
    PAIRS_ENABLED: bool = True               # Activated: decorrelated alpha source
    PAIRS_ENTRY_Z: float = 2.0              # Z-score threshold for entry
    PAIRS_EXIT_Z: float = 0.5               # Z-score threshold for exit
    PAIRS_MAX_Z: float = 4.0                # Z-score stop-loss
    PAIRS_LOOKBACK: int = 60                 # Lookback days for spread calc
    PAIRS_MAX_CONCURRENT: int = 3            # Max concurrent pairs
    PAIRS_LIST: list = [
        ("HDFCBANK", "ICICIBANK"),
        ("TCS", "INFY"),
        ("RELIANCE", "ONGC"),
        ("SBIN", "PNB"),
        ("BHARTIARTL", "IDEA"),
    ]

    # =================================================================
    # Phase 4 — Uncorrelated Alpha: Event-Driven
    # =================================================================
    EVENT_DRIVEN_ENABLED: bool = False       # Master switch for event-driven signals
    EVENT_LOOKFORWARD_DAYS: int = 7          # Look-ahead window for events
    EVENT_EARNINGS_IV_LOW_RATIO: float = 0.8  # IV ratio below which vol expansion expected
    EVENT_EARNINGS_IV_HIGH_RATIO: float = 1.3  # IV ratio above which earnings priced in

    # =================================================================
    # Monte Carlo Permutation Test (Timothy Masters)
    # =================================================================
    MC_PERMUTATION_N_REPS: int = 5000       # Number of MC permutation trials
    MC_CENTER_RETURNS: bool = True           # Center returns to remove directional bias
    MC_NORMALIZE_TIME: bool = True           # sqrt(n) normalization for comparable p-values
    MC_SIGNIFICANCE_LEVEL: float = 0.05     # p-value threshold for significance
    MC_WF_PERM_N_REPS: int = 2000           # Fewer reps for walk-forward permutation (speed)
    MC_TOURNAMENT_N_REPS: int = 2000        # Fewer reps for tournament best-of-N (speed)

    @classmethod
    def get_database_url(cls) -> str:
        """Get the database URL, building from components if not set directly."""
        if os.getenv("CENTURION_DATABASE_URL"):
            return os.getenv("CENTURION_DATABASE_URL")
        
        if cls.DB_PASSWORD:
            return f"postgresql://{_url_quote(cls.DB_USER)}:{_url_quote(cls.DB_PASSWORD)}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        else:
            return f"postgresql://{_url_quote(cls.DB_USER)}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
    
    @classmethod
    def is_database_configured(cls) -> bool:
        """Check if database is properly configured.

        A database connection URL (CENTURION_DATABASE_URL or DATABASE_URL)
        or explicit DB_PASSWORD must be set.  When only a password is
        provided without a URL, the code falls back to host/port
        component-based connection which requires a reachable host.
        """
        if not cls.DB_ENABLED:
            return False
        has_url = bool(
            os.getenv("CENTURION_DATABASE_URL")
            or os.getenv("DATABASE_URL")
        )
        has_password = bool(cls.DB_PASSWORD)
        return has_url or has_password
