"""
Pydantic schemas for US Stocks API endpoints.

Covers: analysis, news scraping, sentiment, metrics, decision engine,
strategies, and backtesting.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    """Request body for running stock analysis."""
    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        examples=[["AAPL", "MSFT", "GOOGL"]],
        description="List of US stock ticker symbols to analyze",
    )


class NewsItemResponse(BaseModel):
    """Single news item with sentiment."""
    title: str
    summary: str
    url: str
    timestamp: Optional[datetime] = None
    source: str
    ticker: str
    category: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_confidence: Optional[float] = None


class StockMetricsResponse(BaseModel):
    """Stock fundamental and technical metrics."""
    ticker: str
    timestamp: Optional[datetime] = None
    # Fundamentals
    peg_ratio: Optional[float] = None
    roe: Optional[float] = None
    eps: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dcf_value: Optional[float] = None
    intrinsic_value: Optional[float] = None
    altman_z_score: Optional[float] = None
    beneish_m_score: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    # Technicals
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_price: Optional[float] = None


class TradingSignalResponse(BaseModel):
    """Complete trading signal with analysis results."""
    ticker: str
    source: str
    title: str
    url: str
    category: str
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_confidence: Optional[float] = None
    decision: str
    decision_score: float
    reasoning: str
    current_price: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    peg_ratio: Optional[float] = None
    roe: Optional[float] = None
    eps: Optional[float] = None
    altman_z_score: Optional[float] = None
    beneish_m_score: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    timestamp: Optional[datetime] = None


class AnalysisResponse(BaseModel):
    """Response for stock analysis containing all trading signals."""
    success: bool = True
    ticker_count: int
    signal_count: int
    signals: List[TradingSignalResponse]
    cache_stats: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# News Scraping
# ---------------------------------------------------------------------------

class ScrapeNewsRequest(BaseModel):
    """Request body for scraping news."""
    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        examples=[["AAPL", "TSLA"]],
    )


class ScrapeNewsResponse(BaseModel):
    """Response containing scraped news items."""
    success: bool = True
    total: int
    items: List[NewsItemResponse]


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

class SentimentRequest(BaseModel):
    """Request body for sentiment analysis on free text."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text snippets to analyze sentiment for",
    )


class SentimentResult(BaseModel):
    """Single sentiment analysis result."""
    text: str
    label: str
    score: float
    confidence: float


class SentimentResponse(BaseModel):
    """Response for sentiment analysis."""
    success: bool = True
    results: List[SentimentResult]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricsRequest(BaseModel):
    """Request body for calculating stock metrics."""
    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        examples=[["AAPL"]],
    )


class MetricsResponse(BaseModel):
    """Response containing stock metrics."""
    success: bool = True
    metrics: List[StockMetricsResponse]


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    """Request body for generating a trading decision."""
    ticker: str = Field(..., examples=["AAPL"])
    news_title: str = Field(..., description="News headline")
    news_summary: str = Field(default="", description="News article summary")
    news_url: str = Field(default="", description="Article URL")
    news_source: str = Field(default="api", description="News source name")
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    sentiment_label: Optional[str] = Field(None, description="positive/negative/neutral")
    sentiment_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class DecisionResponse(BaseModel):
    """Response with trading decision."""
    success: bool = True
    ticker: str
    decision: str
    decision_score: float
    reasoning: str
    sentiment_score: float
    fundamental_score: float
    technical_score: float


# ---------------------------------------------------------------------------
# Strategies & Backtesting
# ---------------------------------------------------------------------------

class StrategyInfo(BaseModel):
    """Strategy metadata."""
    id: str
    name: str
    description: str
    category: str
    requires_sentiment: bool = False
    min_tickers: int = 1


class StrategyListResponse(BaseModel):
    """Response listing available strategies."""
    success: bool = True
    count: int
    strategies: List[StrategyInfo]


class BacktestRequest(BaseModel):
    """Request body for running a backtest."""
    strategy_id: str = Field(..., examples=["macd"])
    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        examples=[["AAPL", "MSFT"]],
    )
    start_date: Optional[str] = Field(
        None,
        examples=["2024-01-01"],
        description="Start date in YYYY-MM-DD format",
    )
    end_date: Optional[str] = Field(
        None,
        examples=["2025-01-01"],
        description="End date in YYYY-MM-DD format",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters",
    )
    initial_capital: float = Field(100000.0, gt=0)


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    profit_factor: Optional[float] = None
    annualized_return: Optional[float] = None
    volatility: Optional[float] = None


class BacktestResponse(BaseModel):
    """Response for a completed backtest."""
    success: bool = True
    strategy_id: str
    tickers: List[str]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Database history
# ---------------------------------------------------------------------------

class AnalysisHistoryResponse(BaseModel):
    """Response listing past analysis runs."""
    success: bool = True
    runs: List[Dict[str, Any]]
    total: int
