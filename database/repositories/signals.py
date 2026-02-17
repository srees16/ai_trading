# Centurion Capital LLC - Signal Repository
"""
Repository for trading signal data access.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func

from database.models import StockSignal, DecisionType
from database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class SignalRepository(BaseRepository[StockSignal]):
    """Repository for StockSignal entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, StockSignal)
    
    def get_by_ticker(
        self,
        ticker: str,
        limit: int = 100,
        days_back: int = 30
    ) -> List[StockSignal]:
        """
        Get signals for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum results
            days_back: How many days back to look
            
        Returns:
            List of signals
        """
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        return self.session.query(StockSignal).filter(
            and_(
                StockSignal.ticker == ticker.upper(),
                StockSignal.created_at >= cutoff
            )
        ).order_by(desc(StockSignal.created_at)).limit(limit).all()
    
    def get_by_decision(
        self,
        decision: DecisionType,
        limit: int = 100
    ) -> List[StockSignal]:
        """
        Get signals by decision type.
        
        Args:
            decision: Decision type to filter
            limit: Maximum results
            
        Returns:
            List of signals
        """
        return self.session.query(StockSignal).filter(
            StockSignal.decision == decision
        ).order_by(desc(StockSignal.created_at)).limit(limit).all()
    
    def get_top_signals(
        self,
        limit: int = 10,
        decision_types: List[DecisionType] = None
    ) -> List[StockSignal]:
        """
        Get top signals by score.
        
        Args:
            limit: Maximum results
            decision_types: Filter by these decision types
            
        Returns:
            List of top signals
        """
        query = self.session.query(StockSignal)
        
        if decision_types:
            query = query.filter(StockSignal.decision.in_(decision_types))
        
        return query.order_by(desc(StockSignal.decision_score)).limit(limit).all()
    
    def get_by_analysis_run(
        self,
        analysis_run_id: UUID
    ) -> List[StockSignal]:
        """
        Get all signals from a specific analysis run.
        
        Args:
            analysis_run_id: The analysis run ID
            
        Returns:
            List of signals
        """
        return self.session.query(StockSignal).filter(
            StockSignal.analysis_run_id == analysis_run_id
        ).order_by(desc(StockSignal.decision_score)).all()
    
    def get_recent_signals_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get summary of recent signals.
        
        Args:
            days: Number of days to summarize
            
        Returns:
            Summary dict with counts by decision type
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        results = self.session.query(
            StockSignal.decision,
            func.count(StockSignal.id).label('count'),
            func.avg(StockSignal.decision_score).label('avg_score')
        ).filter(
            StockSignal.created_at >= cutoff
        ).group_by(StockSignal.decision).all()
        
        summary = {
            'period_days': days,
            'decisions': {}
        }
        
        for decision, count, avg_score in results:
            summary['decisions'][decision.value] = {
                'count': count,
                'avg_score': float(avg_score) if avg_score else 0
            }
        
        return summary
    
    def get_ticker_signal_history(
        self,
        ticker: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Get signal history for a ticker within date range.
        
        Args:
            ticker: Stock ticker
            start_date: Start of range
            end_date: End of range
            
        Returns:
            List of signal summaries
        """
        query = self.session.query(StockSignal).filter(
            StockSignal.ticker == ticker.upper()
        )
        
        if start_date:
            query = query.filter(StockSignal.created_at >= start_date)
        if end_date:
            query = query.filter(StockSignal.created_at <= end_date)
        
        signals = query.order_by(StockSignal.created_at).all()
        
        return [
            {
                'date': s.created_at,
                'decision': s.decision.value,
                'score': s.decision_score,
                'price': float(s.current_price) if s.current_price else None
            }
            for s in signals
        ]
