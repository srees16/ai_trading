"""
Data Models Module for Centurion Capital LLC.

Defines core data structures and interfaces for the algorithmic trading
alert system including news items, stock metrics, and trading signals.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum


class NewsCategory(Enum):
    """News category classification types."""
    BREAKING = "breaking"
    DEALS_MA = "deals_ma"
    MACRO_ECONOMIC = "macro_economic"
    EARNINGS = "earnings"
    GENERAL = "general"

    def __str__(self) -> str:
        return self.value


class SentimentLabel(Enum):
    """Sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class DecisionTag(Enum):
    """Trading decision tags with severity ordering."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

    @property
    def is_bullish(self) -> bool:
        """Check if decision is bullish."""
        return self in (DecisionTag.STRONG_BUY, DecisionTag.BUY)

    @property
    def is_bearish(self) -> bool:
        """Check if decision is bearish."""
        return self in (DecisionTag.STRONG_SELL, DecisionTag.SELL)

    def __str__(self) -> str:
        return self.value


@dataclass
class NewsItem:
    """Represents a single news item with optional sentiment analysis."""
    title: str
    summary: str
    url: str
    timestamp: datetime
    source: str
    ticker: str
    category: NewsCategory = NewsCategory.GENERAL
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[SentimentLabel] = None
    sentiment_confidence: Optional[float] = None
    
    def __repr__(self) -> str:
        return (
            f"NewsItem(ticker={self.ticker!r}, source={self.source!r}, "
            f"sentiment={self.sentiment_label}, title={self.title[:40]!r}...)"
        )
    
    def is_highly_positive(self) -> bool:
        """Check if news is highly positive (confidence > 0.85)."""
        return (
            self.sentiment_label == SentimentLabel.POSITIVE 
            and self.sentiment_confidence is not None 
            and self.sentiment_confidence > 0.85
        )
    
    def is_highly_negative(self) -> bool:
        """Check if news is highly negative (confidence > 0.85)."""
        return (
            self.sentiment_label == SentimentLabel.NEGATIVE 
            and self.sentiment_confidence is not None 
            and self.sentiment_confidence > 0.85
        )


@dataclass
class StockMetrics:
    """Stock fundamental and technical metrics container."""
    ticker: str
    timestamp: datetime
    
    # Fundamentals
    peg_ratio: Optional[float] = None
    roe: Optional[float] = None
    eps: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dcf_value: Optional[float] = None
    intrinsic_value: Optional[float] = None
    
    # Advanced Fundamental Metrics
    altman_z_score: Optional[float] = None
    beneish_m_score: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    
    # Technicals
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    fibonacci_levels: Optional[dict] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_price: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"StockMetrics(ticker={self.ticker!r}, "
            f"price={self.current_price}, rsi={self.rsi}, "
            f"z_score={self.altman_z_score}, f_score={self.piotroski_f_score})"
        )


@dataclass
class TradingSignal:
    """Complete trading signal with news, metrics, and decision."""
    news_item: NewsItem
    metrics: Optional[StockMetrics]
    decision: DecisionTag
    decision_score: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self) -> str:
        return (
            f"TradingSignal(ticker={self.news_item.ticker!r}, "
            f"decision={self.decision.value}, score={self.decision_score:.2f})"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'ticker': self.news_item.ticker,
            'source': self.news_item.source,
            'title': self.news_item.title,
            'url': self.news_item.url,
            'category': self.news_item.category.value,
            'sentiment_label': self.news_item.sentiment_label.value if self.news_item.sentiment_label else None,
            'sentiment_score': self.news_item.sentiment_score,
            'sentiment_confidence': self.news_item.sentiment_confidence,
            'decision': self.decision.value,
            'decision_score': self.decision_score,
            'reasoning': self.reasoning,
            'current_price': self.metrics.current_price if self.metrics else None,
            'rsi': self.metrics.rsi if self.metrics else None,
            'macd': self.metrics.macd if self.metrics else None,
            'peg_ratio': self.metrics.peg_ratio if self.metrics else None,
            'roe': self.metrics.roe if self.metrics else None,
            'eps': self.metrics.eps if self.metrics else None,
            'altman_z_score': self.metrics.altman_z_score if self.metrics else None,
            'beneish_m_score': self.metrics.beneish_m_score if self.metrics else None,
            'piotroski_f_score': self.metrics.piotroski_f_score if self.metrics else None,
        }
