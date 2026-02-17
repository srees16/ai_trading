# Centurion Capital LLC - Fundamental Metrics Repository
"""
Repository for fundamental analysis metrics data access.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
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
    
    def get_health_summary(
        self,
        tickers: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get financial health summary across tickers.
        
        Args:
            tickers: Optional list of tickers to filter
            
        Returns:
            Summary with health metrics breakdown
        """
        query = self.session.query(FundamentalMetric)
        
        if tickers:
            tickers_upper = [t.upper() for t in tickers]
            # Get latest for each ticker
            subquery = self.session.query(
                FundamentalMetric.ticker,
                func.max(FundamentalMetric.recorded_at).label('max_date')
            ).filter(
                FundamentalMetric.ticker.in_(tickers_upper)
            ).group_by(FundamentalMetric.ticker).subquery()
            
            metrics = self.session.query(FundamentalMetric).join(
                subquery,
                and_(
                    FundamentalMetric.ticker == subquery.c.ticker,
                    FundamentalMetric.recorded_at == subquery.c.max_date
                )
            ).all()
        else:
            # Get latest for all tickers
            subquery = self.session.query(
                FundamentalMetric.ticker,
                func.max(FundamentalMetric.recorded_at).label('max_date')
            ).group_by(FundamentalMetric.ticker).subquery()
            
            metrics = self.session.query(FundamentalMetric).join(
                subquery,
                and_(
                    FundamentalMetric.ticker == subquery.c.ticker,
                    FundamentalMetric.recorded_at == subquery.c.max_date
                )
            ).all()
        
        summary = {
            'total_stocks': len(metrics),
            'altman_z': {'safe': 0, 'grey': 0, 'distress': 0, 'na': 0},
            'beneish_m': {'unlikely': 0, 'likely': 0, 'na': 0},
            'piotroski_f': {'strong': 0, 'moderate': 0, 'weak': 0, 'na': 0}
        }
        
        for m in metrics:
            # Altman Z-Score
            if m.altman_z_score is not None:
                if m.altman_z_score > 2.99:
                    summary['altman_z']['safe'] += 1
                elif m.altman_z_score > 1.81:
                    summary['altman_z']['grey'] += 1
                else:
                    summary['altman_z']['distress'] += 1
            else:
                summary['altman_z']['na'] += 1
            
            # Beneish M-Score
            if m.beneish_m_score is not None:
                if m.beneish_m_score <= -2.22:
                    summary['beneish_m']['unlikely'] += 1
                else:
                    summary['beneish_m']['likely'] += 1
            else:
                summary['beneish_m']['na'] += 1
            
            # Piotroski F-Score
            if m.piotroski_f_score is not None:
                if m.piotroski_f_score >= 8:
                    summary['piotroski_f']['strong'] += 1
                elif m.piotroski_f_score >= 5:
                    summary['piotroski_f']['moderate'] += 1
                else:
                    summary['piotroski_f']['weak'] += 1
            else:
                summary['piotroski_f']['na'] += 1
        
        return summary
    
    def get_score_trends(
        self,
        ticker: str,
        days: int = 90
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get trend of financial scores over time.
        
        Args:
            ticker: Stock ticker
            days: Number of days to look back
            
        Returns:
            Dict with score trends
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.session.query(FundamentalMetric).filter(
            and_(
                FundamentalMetric.ticker == ticker.upper(),
                FundamentalMetric.recorded_at >= cutoff
            )
        ).order_by(FundamentalMetric.recorded_at).all()
        
        return {
            'ticker': ticker.upper(),
            'period_days': days,
            'data_points': len(metrics),
            'altman_z': [
                {'date': m.recorded_at, 'value': m.altman_z_score}
                for m in metrics if m.altman_z_score is not None
            ],
            'beneish_m': [
                {'date': m.recorded_at, 'value': m.beneish_m_score}
                for m in metrics if m.beneish_m_score is not None
            ],
            'piotroski_f': [
                {'date': m.recorded_at, 'value': m.piotroski_f_score}
                for m in metrics if m.piotroski_f_score is not None
            ]
        }
