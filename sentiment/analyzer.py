"""
Sentiment Analysis Module.

Analyzes news sentiment using FinBERT (ProsusAI/finbert) transformer
model to classify financial text as positive, negative, or neutral.

The heavy ``transformers`` import and model load are deferred to
first use so that importing this module is near-instant.
"""

import logging
import threading
from typing import List, Tuple

from config import Config
from models import NewsItem, SentimentLabel

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of news items using FinBERT (ProsusAI/finbert).

    The transformer pipeline is loaded lazily on the first call to
    :meth:`analyze` so that constructing the object is fast and the
    ~440 MB model download / load only happens when actually needed.
    """

    _shared_pipeline = None  # class-level cache across instances
    _lock = threading.Lock()  # protects lazy import + model init

    def __init__(self):
        """Initialize the sentiment analyzer (model loaded on first use)."""
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the transformer pipeline on first access."""
        if self._pipeline is not None:
            return self._pipeline

        with SentimentAnalyzer._lock:
            # Double-check after acquiring lock
            if SentimentAnalyzer._shared_pipeline is not None:
                self._pipeline = SentimentAnalyzer._shared_pipeline
                logger.info("Reusing cached sentiment model")
                return self._pipeline

            logger.info("Loading sentiment analysis model...")
            from transformers import pipeline as _hf_pipeline

            SentimentAnalyzer._shared_pipeline = _hf_pipeline(
                "sentiment-analysis",
                model=Config.SENTIMENT_MODEL,
                device=-1,  # CPU; set to 0 for GPU
            )
            self._pipeline = SentimentAnalyzer._shared_pipeline
            logger.info("Sentiment model loaded successfully")
            return self._pipeline
    
    def analyze(self, text: str) -> Tuple[float, SentimentLabel, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_score, sentiment_label, confidence)
            sentiment_score: -1 to 1 (negative to positive)
            sentiment_label: POSITIVE, NEGATIVE, or NEUTRAL
            confidence: 0 to 1
        """
        try:
            # Truncate text if too long
            text = text[:512]
            
            result = self.pipeline(text)[0]
            label = result['label'].upper()
            confidence = result['score']
            
            # Confidence gating: treat low-confidence predictions as neutral
            if confidence < Config.SENTIMENT_CONFIDENCE_FLOOR:
                return 0.0, SentimentLabel.NEUTRAL, confidence

            # Convert to our format (FinBERT returns lowercase labels)
            if label == 'POSITIVE':
                sentiment_score = confidence
                sentiment_label = SentimentLabel.POSITIVE
            elif label == 'NEGATIVE':
                sentiment_score = -confidence
                sentiment_label = SentimentLabel.NEGATIVE
            else:
                sentiment_score = 0.0
                sentiment_label = SentimentLabel.NEUTRAL
            
            return sentiment_score, sentiment_label, confidence
        
        except Exception as e:
            logger.error("Error analyzing sentiment: %s", e)
            return 0.0, SentimentLabel.NEUTRAL, 0.5
    
    def analyze_news_item(self, news_item: NewsItem) -> NewsItem:
        """
        Analyze sentiment of a news item and update it.

        Applies exponential time-decay: recent articles carry more weight.
        Half-life = 7 days (λ = 0.1).
        """
        # Combine title and summary for analysis
        text = f"{news_item.title}. {news_item.summary}"

        sentiment_score, sentiment_label, confidence = self.analyze(text)

        # ── Recency weighting (#12) ──────────────────────────
        # Apply exponential decay based on article age.
        # weight = e^(-λ × days_old) where λ = 0.1 (half-life ~7 days)
        if news_item.timestamp:
            try:
                from datetime import datetime
                import math
                now = datetime.now()
                if hasattr(news_item.timestamp, 'timestamp'):
                    age_days = (now - news_item.timestamp).total_seconds() / 86400
                else:
                    age_days = 0
                decay = math.exp(-0.1 * max(0, age_days))
                sentiment_score *= decay
            except Exception:
                pass  # Use undecayed score if date parsing fails

        news_item.sentiment_score = sentiment_score
        news_item.sentiment_label = sentiment_label
        news_item.sentiment_confidence = confidence

        return news_item
    
    def analyze_news_items(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """
        Analyze sentiment for multiple news items.
        
        Args:
            news_items: List of NewsItem objects
            
        Returns:
            List of updated NewsItem objects with sentiment information
        """
        analyzed_items = []
        
        for news_item in news_items:
            analyzed_item = self.analyze_news_item(news_item)
            analyzed_items.append(analyzed_item)
        
        return analyzed_items
