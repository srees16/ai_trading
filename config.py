"""
Configuration Module for Centurion Capital LLC.

Centralized configuration settings for all system components.
All values can be overridden via environment variables with
the CENTURION_ prefix.
"""

import os
from pathlib import Path
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
    TRANSACTION_COST_IND: float = 0.0022   # 22 bps NSE Zerodha delivery (STT + exchange + stamp + GST + SEBI)
    TRANSACTION_COST_US: float = 0.001     # 10 bps US equities
    SLIPPAGE_MODEL_IND_BPS: float = 20.0   # 20 bps assumed slippage for NSE mid-caps
    SLIPPAGE_IND_LARGECAP_BPS: float = 5.0  # NIFTY50 — very liquid
    SLIPPAGE_IND_MIDCAP_BPS: float = 20.0   # NIFTY_NEXT50 / upper mid-cap
    SLIPPAGE_IND_SMALLCAP_BPS: float = 50.0  # everything else
    SLIPPAGE_MODEL_US_BPS: float = 5.0     # 5 bps for US large-cap

    # Live Viability: Backtest realism controls
    SEVERE_BEAR_EXPOSURE_FLOOR: float = 0.50  # Phase B fix: 10% caused death spiral  # Phase 1: 10% of normal in severe_bear (not 0%)
    EXECUTION_GAP_ENABLED: bool = True         # Phase 4: T+1 open fill gap penalty
    EXECUTION_GAP_BPS: float = 0.0010          # Phase 4: 10 bps (realistic for liquid NIFTY500 stocks)
    PIT_UNIVERSE_ENABLED: bool = True          # Phase B: Point-in-time NIFTY500 (survivorship bias fix)

    # =================================================================
    # Paper Trading Mode
    # =================================================================
    PAPER_TRADE_MODE: bool = os.getenv("CENTURION_PAPER_TRADE", "true").lower() in ("true", "1", "yes")  # FIX-4: env-toggleable
    KILL_SWITCH: bool = False               # G2: Emergency halt — blocks ALL order placement
    KILL_SWITCH_AUTO_ENABLED: bool = True    # T0-2: Auto-trigger kill switch on DD>30% or VIX>40
    KILL_SWITCH_VIX_THRESHOLD: float = 40.0 # T0-2: VIX level that auto-triggers kill switch
    SIGNAL_FRESHNESS_MAX_HOURS: int = 4      # Tier 1 Gap 5: reject OHLCV older than N hours

    # =================================================================
    # Strategy Names
    # =================================================================
    STRATEGY_COMPOUNDER: str = "Centurion Compounder"   # R21A pure compounding (CC)
    STRATEGY_HARVEST: str = "Centurion Harvest"          # V4 income-generating overlay (CH)

    # =================================================================
    # Backtest Configuration
    # =================================================================
    BACKTEST_START_DATE: str = "2012-01-01"  # 13+ years: covers 2013 taper tantrum, 2015 China, 2018 IL&FS, 2020 COVID, 2022 rate hikes
    BACKTEST_END_DATE: str = "2025-12-31"    # End of backtest window (use "" for latest available)

    # =================================================================
    # Carver Systematic Trading Framework (Robert Carver)
    # =================================================================
    CARVER_ENABLED: bool = True             # Enable Carver vol-targeted sizing (False = legacy Kelly)
    CARVER_ANNUAL_VOL_TARGET: float = 0.40  # H2: \u2193 from 50% — lower vol target reduces DD
    CARVER_INITIAL_CAPITAL: float = 500_000.0  # Starting capital (₹)
    CARVER_DEFAULT_IDM: float = 1.3         # P1e: IDM 1.3 — calibrated for NIFTY avg_corr ~0.5; was 2.3
    CARVER_MAX_LEVERAGE: float = 2.0        # P1b: 2× — was 4×. Bull regime can selectively go to 3×.
    CARVER_INERTIA_THRESHOLD: float = 0.25  # H5: \u2191 from 20% — wider inertia reduces churn
    CARVER_COST_SPEED_LIMIT: float = 3.0    # SR must exceed 3× cost drag
    CARVER_TRADE_HORIZON: str = "swing"     # "swing" (3σ bear/5σ bull) or "positional"

    # =================================================================
    # R21A Regime-Adaptive Vol Scaling (centralized source of truth)
    # =================================================================
    R21A_REGIME_VOL: bool = True            # Enable equity-curve regime scaling
    R21A_REGIME_BOOST: float = 1.25         # Uptrend vol multiplier (equity > SMA200×1.02)
    R21A_REGIME_DEFEND: float = 0.55        # Downtrend vol multiplier (equity < SMA200×0.98)
    R21A_SMA_LOOKBACK: int = 200            # SMA lookback for equity-curve regime

    # =================================================================
    # Phase 1 Enhancements — Signal Quality Improvements
    # =================================================================
    # Meta-labeling (AFML Ch.3) — filter false signals via RF meta-classifier
    META_LABELING_ENABLED: bool = True      # Wire meta-labeling into backtest loop
    META_LABEL_MIN_PROBABILITY: float = 0.50  # Only trade when meta_prob > this

    # Regime-adaptive stops (replaces fixed 5σ) — M3: tightened for better DD control
    STOP_SIGMA_BULL: float = 3.0            # RESTORED: tighter stops chopped valid trades
    STOP_SIGMA_BEAR: float = 2.0            # Rapid exit in downtrend (kept)
    STOP_SIGMA_NEUTRAL: float = 4.0         # RESTORED from 3.0
    STOP_SIGMA_STRONG_TREND: float = 5.0    # RESTORED from 4.0

    # Time-based exits — regime-adaptive max hold days
    TIME_EXIT_ENABLED: bool = True

    # Tiered signal recompute frequency (trading days)
    RECOMPUTE_FREQ_FAST: int = 5            # ewmac_8_32, breakout (every 5 days for Kaggle speed)
    RECOMPUTE_FREQ_MEDIUM: int = 10         # momentum, ehlers_dsp, acceleration, penfold_trend
    RECOMPUTE_FREQ_SLOW: int = 20           # carver_value, ewmac_64_256, mean_reversion

    # Forecast-proportional position sizing (replaces flat 1/N)
    FORECAST_PROPORTIONAL_SIZING: bool = True
    FORECAST_SIZING_FLOOR: float = 3.0      # min |forecast| for sizing (prevents tiny positions)

    # Empirical FDM toggle
    EMPIRICAL_FDM_ENABLED: bool = True

    # Smooth bear defense — sigmoid interpolation (replaces binary threshold)
    SMOOTH_BEAR_DEFENSE: bool = True
    SMOOTH_DEFENSE_STEEPNESS: float = 10.0  # sigmoid steepness parameter

    # Sector concentration enforcement in backtest
    MAX_STOCKS_PER_SECTOR: int = 3          # Max positions per GICS sector
    SECTOR_ENFORCEMENT_ENABLED: bool = True

    # Cost-aware inertia (replaces fixed 20%)
    COST_AWARE_INERTIA: bool = True
    INERTIA_ALPHA_COST_RATIO: float = 2.0   # Only trade if expected_alpha > N × expected_cost

    # Block bootstrap for Sharpe CI
    BOOTSTRAP_BLOCK_LENGTH: int = 40  # L4: ~2× trading month, better autocorrelation
    CHECKPOINT_INTERVAL_DAYS: int = 50  # L1: configurable checkpoint frequency

    # Turnover penalty for weight optimization
    TURNOVER_PENALTY_LAMBDA: float = 0.005  # Sharpe_adj = Sharpe - λ × annual_turnover

    # =================================================================
    # Phase 2 — Multi-Timeframe & Multi-Asset
    # =================================================================
    MULTI_TIMEFRAME_ENABLED: bool = True
    WEEKLY_SIGNAL_WEIGHT: float = 0.25      # Weight for weekly TF signals in combined forecast
    MONTHLY_SIGNAL_WEIGHT: float = 0.10     # Weight for monthly TF signals
    DAILY_SIGNAL_WEIGHT: float = 0.65       # Weight for daily TF signals (remainder)

    # Multi-asset diversification (NSE-listed ETFs/futures)
    MULTI_ASSET_ENABLED: bool = True
    MULTI_ASSET_TICKERS_IND: List[str] = [
        "GOLDBEES.NS",       # Gold ETF — uncorrelated with equity
        "GOLDIETF.NS",       # Gold ETF alternate
        "CPSEETF.NS",        # CPSE ETF — govt enterprises
        "LIQUIDBEES.NS",     # Liquid fund — near-cash parking
    ]
    MULTI_ASSET_MAX_ALLOCATION: float = 0.15  # Max 15% of portfolio in non-equity

    # New Alpha Sources (Phase 4) — C5: non-zero initial weights
    NEW_ALPHA_SOURCES_ENABLED: bool = True   # Master switch for 6 new sources
    CRYPTO_TICKER: str = "BTC-USD"           # Bitcoin ticker for crypto correlation
    CALENDAR_EFFECTS_WEIGHT: float = 0.02    # C5: starter weight (was 0.00)
    FUNDAMENTAL_MOMENTUM_WEIGHT: float = 0.03  # C5: starter weight
    INSIDER_ACTIVITY_WEIGHT: float = 0.02    # C5: starter weight
    DISPERSION_WEIGHT: float = 0.02          # C5: starter weight
    GOLD_EQUITY_ROTATION_WEIGHT: float = 0.02  # C5: starter weight
    CRYPTO_CORRELATION_WEIGHT: float = 0.01  # C5: smallest — most speculative

    # =================================================================
    # Godmode Gap Fixes (April 2026)
    # =================================================================
    # M1: Minimum forecast strength gate — DISABLED (was filtering valid trades)
    MIN_FORECAST_GATE_BULL: float = 0.0      # DISABLED: gate killed returns
    MIN_FORECAST_GATE_NEUTRAL: float = 0.0   # DISABLED
    MIN_FORECAST_GATE_BEAR: float = 0.0      # DISABLED

    # M8: Distribution shift detector integration
    DISTRIBUTION_SHIFT_ENABLED: bool = True   # Wire shift detector into backtest loop

    # C1: PBO/CSCV parameters
    PBO_N_PARTITIONS: int = 10               # CSCV partitions (even number)
    PBO_OVERFIT_THRESHOLD: float = 0.35      # PBO > 35% = likely overfit

    # H1/H2: True peak DD tracking (always on — critical fix)
    TRUE_PEAK_DD_HALT: float = 0.35          # Hard halt at 35% TRUE DD from absolute peak

    # C6: Risk-managed momentum — DISABLED (double-stacks with regime blend)
    RISK_MANAGED_MOMENTUM_ENABLED: bool = False

    # =================================================================
    # Phase 3 — Dynamic Leverage & Risk
    # =================================================================
    DYNAMIC_LEVERAGE_ENABLED: bool = True
    LEVERAGE_BULL_CONFIRMED: float = 2.0    # H4: \u2193 from 2.5 — less aggressive to reduce DD
    LEVERAGE_NEUTRAL: float = 1.5           # H4: \u2193 from 2.0 — more conservative
    LEVERAGE_BEAR: float = 0.5              # H4: \u2193 from 1.0 — half leverage in bear
    BULL_CONFIRM_DAYS_LEVERAGE: int = 20    # Consecutive days to confirm bull for leverage boost

    # Vince Money Management — active/inactive equity insurance
    # Floor = HWM × insurance_pct.  0.15 = protect 15% of HWM as floor.
    # At HWM: full sizing.  At floor (15% DD): sizing → 0 (smooth halt).
    VINCE_INSURANCE_PCT_IND: float = 0.12   # IND: 12% floor (P4: was 20%, freed 8pp active equity → +2-3pp CAGR)
    VINCE_INSURANCE_PCT_US: float = 0.10    # US: 10% floor (more conservative for manual)
    VINCE_REGIME_SHRINK_ENABLED: bool = True  # Enable Vince shrink/stretch per regime

    # Drawdown thresholds — restored to pre-Godmode levels (true peak tracking still active)
    PORTFOLIO_DRAWDOWN_WARNING: float = 0.15    # RESTORED from 0.10 (too sensitive)
    PORTFOLIO_DRAWDOWN_CRITICAL: float = 0.25   # RESTORED from 0.20
    PORTFOLIO_DRAWDOWN_HALT: float = 0.35       # H2: hard halt (unchanged, but now from TRUE peak)

    # Carver — US Stocks overrides (USD-based)
    CARVER_US_ENABLED: bool = True          # Enable Carver for US stocks pipeline
    CARVER_US_INITIAL_CAPITAL: float = 10_000.0  # Starting capital ($USD)
    CARVER_US_ANNUAL_VOL_TARGET: float = 0.55    # 55% annual vol target (T2-1: raised from 20%)
    CARVER_US_DEFAULT_IDM: float = 2.0      # IDM for US diversified basket (T2: raised from 1.5)
    CARVER_US_MAX_LEVERAGE: float = 2.0     # 2× leverage for active equity (T2-1: raised from 1.0)
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
    # Earnings Blackout Window
    # =================================================================
    EARNINGS_BLACKOUT_DAYS_BEFORE: int = 2  # Suppress BUY signals 2 days before
    EARNINGS_BLACKOUT_DAYS_AFTER: int = 1   # Suppress BUY signals 1 day after
    
    # =================================================================
    # NSE Circuit Breaker
    # =================================================================
    CIRCUIT_BREAKER_PCT: float = 0.20      # 20% daily move → circuit limit hit
    CIRCUIT_BREAKER_RESET_SECONDS: int = 2700  # 45 min (NSE standard)

    # =================================================================
    # VIX Regime Gate — Gap C4: Unified thresholds
    # =================================================================
    VIX_CAUTION_THRESHOLD: float = 20.0    # India VIX > 20 → reduce position sizes
    VIX_PANIC_THRESHOLD: float = 30.0      # India VIX > 30 → suppress new BUY signals
    VIX_POSITION_SCALE: float = 0.5        # Scale factor when VIX in caution zone
    VIX_PIPELINE_SCALING_ENABLED: bool = True  # Enable VIX scaling in Carver pipeline (not just risk_manager)
    NIFTY_BENCHMARK_TICKER: str = "^NSEI"  # NIFTY 50 index ticker for benchmarking

    # =================================================================
    # Gap C3: Unified max open trades (single source of truth)
    # =================================================================
    MAX_OPEN_TRADES: int = 25              # G3: raised from 12 — 100-stock universe needs more slots for diversification

    # =================================================================
    # Gap C5: Time-based exit enforcement
    # =================================================================
    MAX_HOLD_DAYS_SWING: int = 15          # Max holding period for swing trades (default, overridden per-regime)
    MAX_HOLD_DAYS_POSITIONAL: int = 60     # Max holding period for positional trades

    # A5: Regime-adaptive hold days — faster exits in bear/crisis, longer holds in bull
    REGIME_HOLD_DAYS_SWING: dict = None  # populated below

    @staticmethod
    def get_regime_hold_days(regime: str, horizon: str = "swing") -> int:
        """P5: Return max hold days for given regime and horizon.
        
        Tuned from signal quality audit (April 2026):
          - SIDEWAYS 20D: Sharpe 0.85 (best) → hold 20 days to capture full MR cycle
          - BULL 10D: Sharpe 0.73 → hold 12 days (lock profits before reversal)
          - BEAR 5D: Sharpe -0.01 → hold 5 days (rapid exit, signals broken)
        """
        if horizon != "swing":
            return 60
        _HOLD = {
            "trending_bull":    12,   # P5: was 18 → 12 (BULL 10D Sharpe=0.73, lock profits)
            "trending_bear":     5,   # P5: was 7 → 5 (BEAR signals broken, rapid exit)
            "range_bound":      20,   # P5: was 10 → 20 (SIDEWAYS 20D Sharpe=0.85, best regime!)
            "high_volatility":   5,   # P5: was 5, unchanged (chaos = fast exits)
            "crisis":            3,   # Unchanged — emergency exits
        }
        return _HOLD.get((regime or "").lower().strip(), 15)

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
    # Loaded from data/nse_sector_map.json — editable without code changes.
    # Shared across risk_engine, portfolio_analyzer, and screener.
    # =================================================================
    NSE_SECTOR_MAP: Dict[str, str] = {}  # populated below class body

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
    RL_ENABLED: bool = os.getenv("CENTURION_RL_ENABLED", "true").lower() == "true"
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
    OPTIONS_ENABLED: bool = True            # Master switch for options strategies (CSP + covered calls)
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
    # Phase 2 — Short Selling (F&O Only)
    # =================================================================
    # P2: Enabled via Futures & Options — naked short selling is prohibited in India.
    # Shorts are executed as NRML (overnight F&O) positions, not MIS equity.
    SHORT_SELLING_ENABLED: bool = True       # P2: ENABLED via F&O (legal in India)
    SHORT_SELLING_MODE: str = "FNO"          # P2: "FNO" = futures/options only (no naked equity shorts)
    SHORT_MAX_PORTFOLIO_PCT: float = 0.25    # P2: Max 25% of portfolio short (raised from 20% for bear alpha)
    SHORT_MAX_CONCURRENT: int = 5            # P2: Max concurrent short positions (raised from 3)
    SHORT_REGIME_REQUIRED: str = "bear"      # Only short in bear regime
    SHORT_MIN_FORECAST: float = -5.0         # Min forecast to trigger short trade plan
    SHORT_PRODUCT: str = "NRML"              # P2: NRML for overnight F&O (was MIS intraday)

    # Options-Based Bear Hedging (replaces direct shorting for IND)
    OPTIONS_HEDGE_ENABLED: bool = True       # Enable put-buying / call-selling in bear regime
    OPTIONS_HEDGE_MAX_PORTFOLIO_PCT: float = 0.05  # Max 5% of portfolio for hedge premium
    OPTIONS_HEDGE_STRATEGY: str = "protective_put"  # protective_put | covered_call | collar
    OPTIONS_HEDGE_REGIME_REQUIRED: str = "bear"     # Activate hedges only in bear/crisis regime
    OPTIONS_HEDGE_MIN_VIX: float = 18.0      # Min VIX to consider hedges (cheap vol = buy puts)
    OPTIONS_HEDGE_MAX_VIX: float = 35.0      # Max VIX: don't buy expensive puts above this

    # =================================================================
    # Phase 3 — Leverage via Futures
    # =================================================================
    LEVERAGE_ENABLED: bool = True            # G11: Enabled — regime-adaptive leverage
    LEVERAGE_MAX: float = 2.0               # P1b: 2× absolute cap (synced with CARVER_MAX_LEVERAGE; was 4×)
    LEVERAGE_BULL_MAX: float = 4.0           # Recal: 4× in strong bull — capture trend with controlled risk
    LEVERAGE_RANGE_MAX: float = 3.0          # Recal: 3× in range-bound — alpha capture with DD layers
    LEVERAGE_BEAR_MAX: float = 1.5           # Recal: 1.5× in bear — defensive, stops+VIX gate provide DD protection
    LEVERAGE_CRISIS_MAX: float = 0.5         # 50% sizing in crisis — near-cash (VIX panic gate also blocks entries)
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
    EVENT_DRIVEN_ENABLED: bool = True        # Master switch for event-driven signals
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


# ── Populate NSE_SECTOR_MAP from external JSON ──────────────────────
def _load_nse_sector_map() -> Dict[str, str]:
    import json
    _path = Path(__file__).parent / "data" / "nse_sector_map.json"
    try:
        return json.loads(_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

Config.NSE_SECTOR_MAP = _load_nse_sector_map()
