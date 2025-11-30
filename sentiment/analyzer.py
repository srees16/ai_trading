"""
Sentiment analysis module using transformers.
"""

from transformers import pipeline
from typing import List, Tuple
from models import NewsItem, SentimentLabel
from config import Config


class SentimentAnalyzer:
    """Analyzes sentiment of news items using DistilBERT."""
    
    def __init__(self):
        """Initialize the sentiment analysis pipeline."""
        print("Loading sentiment analysis model...")
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=Config.SENTIMENT_MODEL,
            device=-1  # Use CPU (-1), set to 0 for GPU
        )
        print("Sentiment model loaded successfully.")
    
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
            label = result['label']
            confidence = result['score']
            
            # Convert to our format
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
            print(f"Error analyzing sentiment: {e}")
            return 0.0, SentimentLabel.NEUTRAL, 0.5
    
    def analyze_news_item(self, news_item: NewsItem) -> NewsItem:
        """
        Analyze sentiment of a news item and update it.
        
        Args:
            news_item: NewsItem to analyze
            
        Returns:
            Updated NewsItem with sentiment information
        """
        # Combine title and summary for analysis
        text = f"{news_item.title}. {news_item.summary}"
        
        sentiment_score, sentiment_label, confidence = self.analyze(text)
        
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
