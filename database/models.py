# Centurion Capital LLC - Database Models
"""
SQLAlchemy ORM models for PostgreSQL + TimescaleDB.

These models are designed for:
- Time-series data with TimescaleDB hypertables
- Financial analysis persistence
- Audit trails and data lineage
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text,
    ForeignKey, Index, Enum as SQLEnum, JSON, Numeric, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class DecisionType(enum.Enum):
    """Trading decision types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SentimentType(enum.Enum):
    """Sentiment classification types."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class AnalysisStatus(enum.Enum):
    """Analysis run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TimestampMixin:
    """Mixin for automatic timestamp fields."""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class AnalysisRun(Base, TimestampMixin):
    """
    Tracks individual analysis runs for audit and lineage.
    """
    __tablename__ = 'analysis_runs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_type = Column(String(50), nullable=False)  # 'stock_analysis', 'backtest', 'fundamental'
    status = Column(SQLEnum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False)
    
    # Market identifier (US, IND)
    market = Column(String(10), nullable=False, default='US', server_default='US', index=True)
    
    # Input parameters
    tickers = Column(ARRAY(String), nullable=False)
    parameters = Column(JSONB, default={})
    
    # Results summary
    total_signals = Column(Integer, default=0)
    total_news_items = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    
    # Error tracking
    error_message = Column(Text)
    error_traceback = Column(Text)
    
    # Metadata
    user_id = Column(String(100))  # For multi-user support
    source = Column(String(50), default='web')  # 'web', 'api', 'scheduled'
    
    # Relationships
    signals = relationship("StockSignal", back_populates="analysis_run", lazy="dynamic")
    news_items = relationship("NewsItem", back_populates="analysis_run", lazy="dynamic")
    
    __table_args__ = (
        Index('idx_analysis_runs_status', 'status'),
        Index('idx_analysis_runs_type', 'run_type'),
        Index('idx_analysis_runs_created', 'created_at'),
    )


class NewsItem(Base, TimestampMixin):
    """
    Stores scraped news items with sentiment analysis.
    TimescaleDB hypertable on published_at for time-series queries.
    """
    __tablename__ = 'news_items'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_run_id = Column(UUID(as_uuid=True), ForeignKey('analysis_runs.id'), nullable=True)
    
    # Market identifier (US, IND)
    market = Column(String(10), nullable=False, default='US', server_default='US', index=True)
    
    # Core news data
    ticker = Column(String(20), nullable=False, index=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    url = Column(Text)
    source = Column(String(100), nullable=False)
    author = Column(String(200))
    
    # Timestamps
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    scraped_at = Column(DateTime(timezone=True), default=func.now())
    
    # Sentiment analysis results
    sentiment_label = Column(SQLEnum(SentimentType))
    sentiment_confidence = Column(Float)
    sentiment_scores = Column(JSONB)  # {positive: 0.x, neutral: 0.x, negative: 0.x}
    
    # Content analysis
    keywords = Column(ARRAY(String))
    entities = Column(JSONB)  # Named entities extracted
    
    # Deduplication
    content_hash = Column(String(64), index=True)  # SHA256 of title+content
    
    # Relationships
    analysis_run = relationship("AnalysisRun", back_populates="news_items")
    signal = relationship("StockSignal", back_populates="news_item", uselist=False)
    
    __table_args__ = (
        Index('idx_news_ticker_date', 'ticker', 'published_at'),
        Index('idx_news_source', 'source'),
        UniqueConstraint('content_hash', 'ticker', name='uq_news_content_ticker'),
    )


class StockSignal(Base, TimestampMixin):
    """
    Trading signals generated from analysis.
    TimescaleDB hypertable on created_at for time-series queries.
    """
    __tablename__ = 'stock_signals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_run_id = Column(UUID(as_uuid=True), ForeignKey('analysis_runs.id'), nullable=True)
    news_item_id = Column(UUID(as_uuid=True), ForeignKey('news_items.id'), nullable=True)
    
    # Market identifier (US, IND)
    market = Column(String(10), nullable=False, default='US', server_default='US', index=True)
    
    # Core signal data
    ticker = Column(String(20), nullable=False, index=True)
    decision = Column(SQLEnum(DecisionType), nullable=False)
    decision_score = Column(Float, nullable=False)
    reasoning = Column(Text)
    
    # Associated metrics at signal time
    current_price = Column(Numeric(12, 4))
    price_change_pct = Column(Float)
    volume = Column(Integer)
    
    # Technical indicators
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    sma_20 = Column(Numeric(12, 4))
    sma_50 = Column(Numeric(12, 4))
    sma_200 = Column(Numeric(12, 4))
    bollinger_upper = Column(Numeric(12, 4))
    bollinger_lower = Column(Numeric(12, 4))
    
    # Fundamental scores
    altman_z_score = Column(Float)
    beneish_m_score = Column(Float)
    piotroski_f_score = Column(Integer)
    
    # Decision factors (for explainability)
    decision_factors = Column(JSONB)  # Breakdown of what contributed to decision
    
    # Tags for filtering
    tags = Column(ARRAY(String))
    
    # Relationships
    analysis_run = relationship("AnalysisRun", back_populates="signals")
    news_item = relationship("NewsItem", back_populates="signal")
    
    __table_args__ = (
        Index('idx_signals_ticker_date', 'ticker', 'created_at'),
        Index('idx_signals_decision', 'decision'),
        Index('idx_signals_score', 'decision_score'),
    )


class FundamentalMetric(Base, TimestampMixin):
    """
    Fundamental analysis metrics history.
    TimescaleDB hypertable on recorded_at for time-series queries.
    """
    __tablename__ = 'fundamental_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Market identifier (US, IND)
    market = Column(String(10), nullable=False, default='US', server_default='US', index=True)
    
    # Core data
    ticker = Column(String(20), nullable=False, index=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    
    # Price data
    current_price = Column(Numeric(12, 4))
    market_cap = Column(Numeric(20, 2))
    enterprise_value = Column(Numeric(20, 2))
    
    # Valuation ratios
    pe_ratio = Column(Float)
    forward_pe = Column(Float)
    peg_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    ev_to_ebitda = Column(Float)
    ev_to_revenue = Column(Float)
    
    # Financial health scores
    altman_z_score = Column(Float)
    beneish_m_score = Column(Float)
    piotroski_f_score = Column(Integer)
    
    # Profitability
    profit_margin = Column(Float)
    operating_margin = Column(Float)
    return_on_equity = Column(Float)
    return_on_assets = Column(Float)
    
    # Financial position
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    debt_to_equity = Column(Float)
    debt_to_assets = Column(Float)
    interest_coverage = Column(Float)
    
    # Growth metrics
    revenue_growth = Column(Float)
    earnings_growth = Column(Float)
    
    # Dividend data
    dividend_yield = Column(Float)
    payout_ratio = Column(Float)
    
    # Raw data for recalculation
    raw_financials = Column(JSONB)
    
    # Data source
    data_source = Column(String(50), default='yfinance')
    
    __table_args__ = (
        Index('idx_fundamental_ticker_date', 'ticker', 'recorded_at'),
        UniqueConstraint('ticker', 'recorded_at', name='uq_fundamental_ticker_date'),
    )


class BacktestResult(Base, TimestampMixin):
    """
    Backtesting results storage.
    """
    __tablename__ = 'backtest_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Market identifier (US, IND)
    market = Column(String(10), nullable=False, default='US', server_default='US', index=True)
    
    # Strategy identification
    strategy_id = Column(String(100), nullable=False, index=True)
    strategy_name = Column(String(200), nullable=False)
    strategy_category = Column(String(100))
    strategy_version = Column(String(20), default='1.0')
    
    # Test parameters
    tickers = Column(ARRAY(String), nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    initial_capital = Column(Numeric(15, 2), nullable=False)
    
    # Strategy parameters
    parameters = Column(JSONB, nullable=False, default={})
    
    # Results status
    success = Column(Boolean, default=False)
    error_message = Column(Text)
    
    # Performance metrics
    total_return = Column(Float)
    annualized_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    calmar_ratio = Column(Float)
    
    # Trading statistics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    
    # Detailed metrics (full breakdown)
    metrics = Column(JSONB)
    
    # Trade signals generated
    signals = Column(JSONB)  # List of trade signals
    
    # Equity curve data points (sampled for storage efficiency)
    equity_curve = Column(JSONB)
    
    # Execution metadata
    execution_time_seconds = Column(Float)
    data_points_processed = Column(Integer)
    
    # Relationships to normalised detail tables
    trade_details = relationship(
        'BacktestTrade', back_populates='backtest',
        cascade='all, delete-orphan', lazy='dynamic',
    )
    equity_points = relationship(
        'BacktestEquityPoint', back_populates='backtest',
        cascade='all, delete-orphan', lazy='dynamic',
    )
    daily_returns = relationship(
        'BacktestDailyReturn', back_populates='backtest',
        cascade='all, delete-orphan', lazy='dynamic',
    )
    
    __table_args__ = (
        Index('idx_backtest_strategy', 'strategy_id'),
        Index('idx_backtest_tickers', 'tickers', postgresql_using='gin'),
        Index('idx_backtest_date_range', 'start_date', 'end_date'),
    )


class UserWatchlist(Base, TimestampMixin):
    """
    User-specific watchlist for tracking stocks.
    """
    __tablename__ = 'user_watchlists'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    tickers = Column(ARRAY(String), nullable=False)
    is_default = Column(Boolean, default=False)
    
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='uq_watchlist_user_name'),
    )


class AlertConfiguration(Base, TimestampMixin):
    """
    Alert configurations for automated notifications.
    """
    __tablename__ = 'alert_configurations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Alert target
    ticker = Column(String(20), nullable=False)
    alert_type = Column(String(50), nullable=False)  # 'price', 'signal', 'news', 'fundamental'
    
    # Conditions
    conditions = Column(JSONB, nullable=False)  # {field: 'price', operator: '>', value: 100}
    
    # Notification settings
    notification_channels = Column(ARRAY(String))  # ['email', 'sms', 'push']
    is_active = Column(Boolean, default=True)
    
    # Tracking
    last_triggered_at = Column(DateTime(timezone=True))
    trigger_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_alert_user_ticker', 'user_id', 'ticker'),
        Index('idx_alert_active', 'is_active'),
    )


# =====================================================================
# Bronze Layer — Raw/Unprocessed Data (Medallion Architecture)
# =====================================================================

class RawScrapedNews(Base):
    """
    Bronze layer: Raw scraped news items before enrichment.

    Stores the unprocessed scraper output exactly as received.
    Downstream processing promotes rows into the enriched NewsItem
    (silver layer) after sentiment analysis, deduplication, and
    entity extraction.
    """
    __tablename__ = 'raw_scraped_news'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Source metadata
    ticker = Column(String(20), nullable=False, index=True)
    source = Column(String(100), nullable=False)
    scraper_name = Column(String(100))  # e.g. 'finviz', 'yahoo_finance'

    # Raw content as received
    raw_title = Column(Text, nullable=False)
    raw_content = Column(Text)
    raw_url = Column(Text)
    raw_author = Column(String(200))
    raw_published_at = Column(DateTime(timezone=True))

    # Deduplication
    content_hash = Column(String(64), index=True)

    # Processing state
    is_processed = Column(Boolean, default=False, index=True)
    processed_at = Column(DateTime(timezone=True))
    enriched_news_id = Column(UUID(as_uuid=True), ForeignKey('news_items.id'), nullable=True)

    # Ingestion timestamp
    ingested_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_raw_news_ticker_ingested', 'ticker', 'ingested_at'),
        Index('idx_raw_news_unprocessed', 'is_processed',
              postgresql_where=Column('is_processed') == False),
    )


# =====================================================================
# Normalized Backtest Detail Tables (JSONB Extraction)
# =====================================================================

class BacktestTrade(Base):
    """
    Individual trade entries extracted from BacktestResult.signals JSONB.
    Enables SQL-level querying, filtering, and aggregation of trades.
    """
    __tablename__ = 'backtest_trades'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey('backtest_results.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )

    # Trade details
    trade_number = Column(Integer, nullable=False)
    trade_type = Column(String(10), nullable=False)   # 'BUY' / 'SELL'
    ticker = Column(String(20), nullable=False)
    entry_date = Column(DateTime(timezone=True))
    exit_date = Column(DateTime(timezone=True))
    entry_price = Column(Numeric(12, 4))
    exit_price = Column(Numeric(12, 4))
    quantity = Column(Numeric(15, 4))
    pnl = Column(Float)
    pnl_pct = Column(Float)
    holding_period_days = Column(Integer)

    # Parent relationship
    backtest = relationship('BacktestResult', back_populates='trade_details')

    __table_args__ = (
        Index('idx_bt_trade_backtest', 'backtest_id'),
        Index('idx_bt_trade_ticker', 'ticker'),
    )


class BacktestEquityPoint(Base):
    """
    Time-series equity curve data extracted from BacktestResult.equity_curve JSONB.
    """
    __tablename__ = 'backtest_equity_points'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey('backtest_results.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )

    point_date = Column(DateTime(timezone=True), nullable=False)
    portfolio_value = Column(Numeric(15, 2), nullable=False)
    drawdown = Column(Float)
    benchmark_value = Column(Numeric(15, 2))

    # Parent relationship
    backtest = relationship('BacktestResult', back_populates='equity_points')

    __table_args__ = (
        Index('idx_bt_equity_backtest_date', 'backtest_id', 'point_date'),
    )


class BacktestDailyReturn(Base):
    """
    Daily return values extracted from BacktestResult.metrics.daily_returns JSONB.
    """
    __tablename__ = 'backtest_daily_returns'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey('backtest_results.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )

    return_date = Column(DateTime(timezone=True), nullable=False)
    daily_return = Column(Float, nullable=False)
    cumulative_return = Column(Float)

    # Parent relationship
    backtest = relationship('BacktestResult', back_populates='daily_returns')

    __table_args__ = (
        Index('idx_bt_daily_ret_backtest_date', 'backtest_id', 'return_date'),
    )


# =====================================================================
# Gold Layer — Pre-Aggregated Summary Tables
# =====================================================================

class StrategyPerformanceSummary(Base, TimestampMixin):
    """
    Gold layer: Pre-aggregated strategy performance metrics.

    Refreshed on every new backtest completion. Enables instant
    dashboard queries without scanning the full backtest_results table.
    """
    __tablename__ = 'strategy_performance_summary'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(String(100), nullable=False, unique=True, index=True)
    strategy_name = Column(String(200), nullable=False)

    # Aggregate counts
    total_backtests = Column(Integer, default=0)
    successful_backtests = Column(Integer, default=0)

    # Return statistics
    avg_return = Column(Float)
    best_return = Column(Float)
    worst_return = Column(Float)
    median_return = Column(Float)

    # Risk statistics
    avg_sharpe = Column(Float)
    avg_sortino = Column(Float)
    avg_max_drawdown = Column(Float)
    avg_calmar = Column(Float)

    # Trading statistics
    avg_win_rate = Column(Float)
    avg_profit_factor = Column(Float)
    avg_total_trades = Column(Float)

    # Last refresh metadata
    last_backtest_at = Column(DateTime(timezone=True))
    last_refreshed_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_strategy_perf_id', 'strategy_id'),
    )


# =====================================================================
# Data Freshness Tracking
# =====================================================================

class DataFreshness(Base):
    """
    Tracks when each data type was last fetched per ticker.

    Used to avoid redundant API/scraper calls and to surface
    stale-data warnings in the UI.
    """
    __tablename__ = 'data_freshness'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20), nullable=False)
    data_type = Column(String(50), nullable=False)  # 'news', 'fundamentals', 'price', 'sentiment'

    # Freshness timestamps
    last_fetched_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    next_refresh_at = Column(DateTime(timezone=True))

    # Statistics
    fetch_count = Column(Integer, default=1)
    avg_fetch_seconds = Column(Float)
    last_fetch_seconds = Column(Float)
    last_record_count = Column(Integer)

    # Error tracking
    consecutive_errors = Column(Integer, default=0)
    last_error = Column(Text)
    last_error_at = Column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint('ticker', 'data_type', name='uq_freshness_ticker_type'),
        Index('idx_freshness_ticker', 'ticker'),
        Index('idx_freshness_next_refresh', 'next_refresh_at'),
    )
