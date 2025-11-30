"""
Data models and interfaces for the algo-trading alert system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class NewsCategory(Enum):
    """News category types."""
    BREAKING = "breaking"
    DEALS_MA = "deals_ma"
    MACRO_ECONOMIC = "macro_economic"
    EARNINGS = "earnings"
    GENERAL = "general"


class SentimentLabel(Enum):
    """Sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class DecisionTag(Enum):
    """Trading decision tags."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class NewsItem:
    """Represents a single news item."""
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
    """Stock fundamental and technical metrics."""
    ticker: str
    timestamp: datetime
    
    # Fundamentals
    peg_ratio: Optional[float] = None
    roe: Optional[float] = None
    eps: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dcf_value: Optional[float] = None
    intrinsic_value: Optional[float] = None
    
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


@dataclass
class TradingSignal:
    """Complete trading signal with news, metrics, and decision."""
    news_item: NewsItem
    metrics: Optional[StockMetrics]
    decision: DecisionTag
    decision_score: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
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
        }
