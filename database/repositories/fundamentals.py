# Centurion Capital LLC - Fundamental Metrics Repository
"""
Repository for fundamental analysis metrics data access.
"""

import logging
from typing import List, Optional
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func

from database.models import FundamentalMetric
from database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class FundamentalRepository(BaseRepository[FundamentalMetric]):
    """Repository for FundamentalMetric entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, FundamentalMetric)
    
    def get_latest_by_ticker(self, ticker: str) -> Optional[FundamentalMetric]:
        """
        Get most recent fundamental metrics for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Latest FundamentalMetric or None
        """
        return self.session.query(FundamentalMetric).filter(
            FundamentalMetric.ticker == ticker.upper()
        ).order_by(desc(FundamentalMetric.recorded_at)).first()
    
    def get_history(
        self,
        ticker: str,
        limit: int = 100,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[FundamentalMetric]:
        """
        Get fundamental metrics history for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum results
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of metrics ordered by date
        """
        query = self.session.query(FundamentalMetric).filter(
            FundamentalMetric.ticker == ticker.upper()
        )
        
        if start_date:
            query = query.filter(FundamentalMetric.recorded_at >= start_date)
        if end_date:
            query = query.filter(FundamentalMetric.recorded_at <= end_date)
        
        return query.order_by(desc(FundamentalMetric.recorded_at)).limit(limit).all()
    
    def get_multiple_tickers(
        self,
        tickers: List[str],
        latest_only: bool = True
    ) -> List[FundamentalMetric]:
        """
        Get fundamental metrics for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            latest_only: If True, only get latest for each ticker
            
        Returns:
            List of metrics
        """
        tickers_upper = [t.upper() for t in tickers]
        
        if latest_only:
            # Subquery to get latest date for each ticker
            subquery = self.session.query(
                FundamentalMetric.ticker,
                func.max(FundamentalMetric.recorded_at).label('max_date')
            ).filter(
                FundamentalMetric.ticker.in_(tickers_upper)
            ).group_by(FundamentalMetric.ticker).subquery()
            
            return self.session.query(FundamentalMetric).join(
                subquery,
                and_(
                    FundamentalMetric.ticker == subquery.c.ticker,
                    FundamentalMetric.recorded_at == subquery.c.max_date
                )
            ).all()
        else:
            return self.session.query(FundamentalMetric).filter(
                FundamentalMetric.ticker.in_(tickers_upper)
            ).order_by(
                FundamentalMetric.ticker,
                desc(FundamentalMetric.recorded_at)
            ).all()
    
    def upsert(self, metric: FundamentalMetric) -> FundamentalMetric:
        """
        Insert or update fundamental metric.
        Uses ticker + recorded_at as unique key.
        
        Args:
            metric: FundamentalMetric to upsert
            
        Returns:
            Upserted metric
        """
        existing = self.session.query(FundamentalMetric).filter(
            and_(
                FundamentalMetric.ticker == metric.ticker,
                FundamentalMetric.recorded_at == metric.recorded_at
            )
        ).first()
        
        if existing:
            # Update existing record
            for key, value in metric.__dict__.items():
                if not key.startswith('_') and key not in ['id', 'created_at']:
                    setattr(existing, key, value)
            self.session.flush()
            return existing
        else:
            return self.create(metric)
