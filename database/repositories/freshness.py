# Centurion Capital LLC - Data Freshness Repository
"""
Repository for tracking data freshness per ticker and data type.
Enables smart refresh decisions to avoid redundant API/scraper calls.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_

from database.models import DataFreshness
from database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class FreshnessRepository(BaseRepository[DataFreshness]):
    """Repository for DataFreshness tracking."""
    
    def __init__(self, session: Session):
        super().__init__(session, DataFreshness)
    
    def get_freshness(
        self,
        ticker: str,
        data_type: str,
    ) -> Optional[DataFreshness]:
        """
        Get freshness record for a specific ticker and data type.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Data category ('news', 'fundamentals', 'price', 'sentiment')
            
        Returns:
            DataFreshness record or None
        """
        return self.session.query(DataFreshness).filter(
            and_(
                DataFreshness.ticker == ticker.upper(),
                DataFreshness.data_type == data_type
            )
        ).first()
    
    def is_stale(
        self,
        ticker: str,
        data_type: str,
        max_age_minutes: int = 30
    ) -> bool:
        """
        Check if data for a ticker/type is stale or missing.
        
        Args:
            ticker: Stock ticker
            data_type: Data category
            max_age_minutes: Maximum acceptable age in minutes
            
        Returns:
            True if data is stale or doesn't exist
        """
        record = self.get_freshness(ticker, data_type)
        if not record:
            return True
        
        age = datetime.utcnow() - record.last_fetched_at.replace(tzinfo=None)
        return age > timedelta(minutes=max_age_minutes)
    
    def record_fetch(
        self,
        ticker: str,
        data_type: str,
        record_count: int = 0,
        fetch_seconds: float = 0.0,
        refresh_minutes: int = 30,
    ) -> DataFreshness:
        """
        Record a successful data fetch (upsert).
        
        Args:
            ticker: Stock ticker
            data_type: Data category
            record_count: Number of records fetched
            fetch_seconds: Time taken for fetch
            refresh_minutes: Minutes until next refresh
            
        Returns:
            Updated DataFreshness record
        """
        ticker = ticker.upper()
        record = self.get_freshness(ticker, data_type)
        now = datetime.utcnow()
        
        if record:
            # Update existing
            record.last_fetched_at = now
            record.next_refresh_at = now + timedelta(minutes=refresh_minutes)
            record.fetch_count = (record.fetch_count or 0) + 1
            record.last_fetch_seconds = fetch_seconds
            record.last_record_count = record_count
            record.consecutive_errors = 0
            record.last_error = None
            record.last_error_at = None
            
            # Running average of fetch time
            if record.avg_fetch_seconds and record.fetch_count > 1:
                record.avg_fetch_seconds = (
                    (record.avg_fetch_seconds * (record.fetch_count - 1) + fetch_seconds)
                    / record.fetch_count
                )
            else:
                record.avg_fetch_seconds = fetch_seconds
            
            self.session.flush()
            return record
        else:
            # Create new
            fresh = DataFreshness(
                ticker=ticker,
                data_type=data_type,
                last_fetched_at=now,
                next_refresh_at=now + timedelta(minutes=refresh_minutes),
                fetch_count=1,
                avg_fetch_seconds=fetch_seconds,
                last_fetch_seconds=fetch_seconds,
                last_record_count=record_count,
                consecutive_errors=0,
            )
            return self.create(fresh)
    
    def record_error(
        self,
        ticker: str,
        data_type: str,
        error_message: str,
    ) -> Optional[DataFreshness]:
        """
        Record a fetch error. Increments consecutive error count.
        
        Args:
            ticker: Stock ticker
            data_type: Data category
            error_message: Error description
            
        Returns:
            Updated DataFreshness record or None
        """
        ticker = ticker.upper()
        record = self.get_freshness(ticker, data_type)
        now = datetime.utcnow()
        
        if record:
            record.consecutive_errors = (record.consecutive_errors or 0) + 1
            record.last_error = error_message
            record.last_error_at = now
            self.session.flush()
            return record
        else:
            # Create a record even on first-time error
            fresh = DataFreshness(
                ticker=ticker,
                data_type=data_type,
                last_fetched_at=now,
                consecutive_errors=1,
                last_error=error_message,
                last_error_at=now,
            )
            return self.create(fresh)
    
    def get_stale_tickers(
        self,
        data_type: str,
        max_age_minutes: int = 30,
        limit: int = 100,
    ) -> List[DataFreshness]:
        """
        Get tickers that need refreshing.
        
        Args:
            data_type: Data category to check
            max_age_minutes: Threshold age in minutes
            limit: Max results
            
        Returns:
            List of stale DataFreshness records
        """
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        
        return self.session.query(DataFreshness).filter(
            and_(
                DataFreshness.data_type == data_type,
                DataFreshness.last_fetched_at < cutoff,
            )
        ).order_by(DataFreshness.last_fetched_at).limit(limit).all()
    
    def get_ticker_freshness(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get all freshness records for a ticker across data types.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            List of freshness info dicts
        """
        records = self.session.query(DataFreshness).filter(
            DataFreshness.ticker == ticker.upper()
        ).all()
        
        return [
            {
                'data_type': r.data_type,
                'last_fetched_at': r.last_fetched_at,
                'next_refresh_at': r.next_refresh_at,
                'fetch_count': r.fetch_count,
                'is_stale': (
                    r.next_refresh_at is not None
                    and datetime.utcnow() > r.next_refresh_at.replace(tzinfo=None)
                ),
                'consecutive_errors': r.consecutive_errors,
                'last_error': r.last_error,
            }
            for r in records
        ]
    
    def get_all_freshness_summary(self) -> Dict[str, Any]:
        """
        Get a summary of data freshness across all tickers.
        
        Returns:
            Summary dict with counts per data type
        """
        from sqlalchemy import func
        
        rows = self.session.query(
            DataFreshness.data_type,
            func.count(DataFreshness.id).label('total'),
            func.avg(
                func.extract('epoch', func.now() - DataFreshness.last_fetched_at)
            ).label('avg_age_seconds'),
            func.sum(
                func.cast(DataFreshness.consecutive_errors > 0, Integer)
            ).label('error_count'),
        ).group_by(DataFreshness.data_type).all()
        
        return {
            r.data_type: {
                'total_tickers': r.total,
                'avg_age_minutes': round(float(r.avg_age_seconds or 0) / 60, 1),
                'tickers_with_errors': int(r.error_count or 0),
            }
            for r in rows
        }
