# Centurion Capital LLC - News Repository
"""
Repository for news items data access.
"""

import logging
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func

from database.models import NewsItem, SentimentType
from database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class NewsRepository(BaseRepository[NewsItem]):
    """Repository for NewsItem entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, NewsItem)
    
    def get_by_ticker(
        self,
        ticker: str,
        limit: int = 100,
        days_back: int = 7
    ) -> List[NewsItem]:
        """
        Get news for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum results
            days_back: How many days back to look
            
        Returns:
            List of news items
        """
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        return self.session.query(NewsItem).filter(
            and_(
                NewsItem.ticker == ticker.upper(),
                NewsItem.published_at >= cutoff
            )
        ).order_by(desc(NewsItem.published_at)).limit(limit).all()
    
    def get_by_source(
        self,
        source: str,
        limit: int = 100
    ) -> List[NewsItem]:
        """
        Get news from a specific source.
        
        Args:
            source: News source name
            limit: Maximum results
            
        Returns:
            List of news items
        """
        return self.session.query(NewsItem).filter(
            NewsItem.source == source
        ).order_by(desc(NewsItem.published_at)).limit(limit).all()
    
    def get_by_sentiment(
        self,
        sentiment: SentimentType,
        limit: int = 100
    ) -> List[NewsItem]:
        """
        Get news by sentiment label.
        
        Args:
            sentiment: Sentiment type to filter
            limit: Maximum results
            
        Returns:
            List of news items
        """
        return self.session.query(NewsItem).filter(
            NewsItem.sentiment_label == sentiment
        ).order_by(desc(NewsItem.published_at)).limit(limit).all()
    
    def check_duplicate(self, title: str, content: str, ticker: str) -> bool:
        """
        Check if news item already exists.
        
        Args:
            title: News title
            content: News content
            ticker: Stock ticker
            
        Returns:
            True if duplicate exists
        """
        content_hash = self._generate_hash(title, content)
        existing = self.session.query(NewsItem).filter(
            and_(
                NewsItem.content_hash == content_hash,
                NewsItem.ticker == ticker.upper()
            )
        ).first()
        return existing is not None
    
    def create_with_dedup(self, news_item: NewsItem) -> Optional[NewsItem]:
        """
        Create news item with deduplication.
        
        Args:
            news_item: NewsItem to create
            
        Returns:
            Created item or None if duplicate
        """
        content_hash = self._generate_hash(news_item.title, news_item.content or '')
        news_item.content_hash = content_hash
        
        if not self.check_duplicate(news_item.title, news_item.content or '', news_item.ticker):
            return self.create(news_item)
        return None
    
    def _generate_hash(self, title: str, content: str) -> str:
        """Generate content hash for deduplication."""
        combined = f"{title}|{content}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()
    
    def get_sentiment_summary(
        self,
        ticker: str = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get sentiment distribution summary.
        
        Args:
            ticker: Optional ticker filter
            days: Number of days to analyze
            
        Returns:
            Summary with sentiment distribution
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = self.session.query(
            NewsItem.sentiment_label,
            func.count(NewsItem.id).label('count'),
            func.avg(NewsItem.sentiment_confidence).label('avg_confidence')
        ).filter(NewsItem.published_at >= cutoff)
        
        if ticker:
            query = query.filter(NewsItem.ticker == ticker.upper())
        
        results = query.group_by(NewsItem.sentiment_label).all()
        
        summary = {
            'period_days': days,
            'ticker': ticker,
            'sentiments': {}
        }
        
        for sentiment, count, avg_conf in results:
            if sentiment:
                summary['sentiments'][sentiment.value] = {
                    'count': count,
                    'avg_confidence': float(avg_conf) if avg_conf else 0
                }
        
        return summary
    
    def get_source_statistics(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get statistics by news source.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of source statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        results = self.session.query(
            NewsItem.source,
            func.count(NewsItem.id).label('total'),
            func.count(func.distinct(NewsItem.ticker)).label('unique_tickers')
        ).filter(
            NewsItem.published_at >= cutoff
        ).group_by(NewsItem.source).all()
        
        return [
            {
                'source': source,
                'total_articles': total,
                'unique_tickers': unique_tickers
            }
            for source, total, unique_tickers in results
        ]
