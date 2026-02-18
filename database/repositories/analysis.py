# Centurion Capital LLC - Analysis Run Repository
"""
Repository for analysis run tracking and audit trail.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func

from database.models import AnalysisRun, AnalysisStatus
from database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class AnalysisRepository(BaseRepository[AnalysisRun]):
    """Repository for AnalysisRun entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, AnalysisRun)
    
    def create_run(
        self,
        run_type: str,
        tickers: List[str],
        parameters: Dict[str, Any] = None,
        user_id: str = None,
        source: str = 'web'
    ) -> AnalysisRun:
        """
        Create a new analysis run.
        
        Args:
            run_type: Type of analysis ('stock_analysis', 'backtest', 'fundamental')
            tickers: List of tickers being analyzed
            parameters: Analysis parameters
            user_id: User who initiated the run
            source: Source of the run ('web', 'api', 'scheduled')
            
        Returns:
            Created AnalysisRun
        """
        run = AnalysisRun(
            run_type=run_type,
            status=AnalysisStatus.PENDING,
            tickers=[t.upper() for t in tickers],
            parameters=parameters or {},
            user_id=user_id,
            source=source
        )
        return self.create(run)
    
    def start_run(self, run_id: UUID) -> AnalysisRun:
        """
        Mark a run as started.
        
        Args:
            run_id: Run ID
            
        Returns:
            Updated run
        """
        run = self.get_by_id(run_id)
        if run:
            run.status = AnalysisStatus.RUNNING
            run.started_at = datetime.now(timezone.utc)
            self.session.flush()
        return run
    
    def complete_run(
        self,
        run_id: UUID,
        total_signals: int = 0,
        total_news_items: int = 0
    ) -> AnalysisRun:
        """
        Mark a run as completed.
        
        Args:
            run_id: Run ID
            total_signals: Number of signals generated
            total_news_items: Number of news items processed
            
        Returns:
            Updated run
        """
        run = self.get_by_id(run_id)
        if run:
            run.status = AnalysisStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)
            run.total_signals = total_signals
            run.total_news_items = total_news_items
            if run.started_at:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            self.session.flush()
        return run
    
    def fail_run(
        self,
        run_id: UUID,
        error_message: str,
        error_traceback: str = None
    ) -> AnalysisRun:
        """
        Mark a run as failed.
        
        Args:
            run_id: Run ID
            error_message: Error message
            error_traceback: Full traceback
            
        Returns:
            Updated run
        """
        run = self.get_by_id(run_id)
        if run:
            run.status = AnalysisStatus.FAILED
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = error_message
            run.error_traceback = error_traceback
            if run.started_at:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            self.session.flush()
        return run
    
    def get_recent_runs(
        self,
        run_type: str = None,
        limit: int = 50,
        days: int = 7
    ) -> List[AnalysisRun]:
        """
        Get recent analysis runs.
        
        Args:
            run_type: Filter by run type
            limit: Maximum results
            days: Days back to look
            
        Returns:
            List of runs
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = self.session.query(AnalysisRun).filter(
            AnalysisRun.created_at >= cutoff
        )
        
        if run_type:
            query = query.filter(AnalysisRun.run_type == run_type)
        
        return query.order_by(desc(AnalysisRun.created_at)).limit(limit).all()
    
    def get_runs_by_status(
        self,
        status: AnalysisStatus,
        limit: int = 50
    ) -> List[AnalysisRun]:
        """
        Get runs by status.
        
        Args:
            status: Status to filter
            limit: Maximum results
            
        Returns:
            List of runs
        """
        return self.session.query(AnalysisRun).filter(
            AnalysisRun.status == status
        ).order_by(desc(AnalysisRun.created_at)).limit(limit).all()
    
    def get_run_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get statistics about analysis runs.
        
        Args:
            days: Days to analyze
            
        Returns:
            Statistics dict
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Status counts
        status_results = self.session.query(
            AnalysisRun.status,
            func.count(AnalysisRun.id).label('count')
        ).filter(
            AnalysisRun.created_at >= cutoff
        ).group_by(AnalysisRun.status).all()
        
        # Type counts
        type_results = self.session.query(
            AnalysisRun.run_type,
            func.count(AnalysisRun.id).label('count')
        ).filter(
            AnalysisRun.created_at >= cutoff
        ).group_by(AnalysisRun.run_type).all()
        
        # Average duration
        duration = self.session.query(
            func.avg(AnalysisRun.duration_seconds)
        ).filter(
            and_(
                AnalysisRun.created_at >= cutoff,
                AnalysisRun.duration_seconds.isnot(None)
            )
        ).scalar()
        
        return {
            'period_days': days,
            'by_status': {s.value: c for s, c in status_results},
            'by_type': {t: c for t, c in type_results},
            'avg_duration_seconds': float(duration) if duration else None,
            'total_runs': sum(c for _, c in status_results)
        }
    
    def cleanup_old_runs(self, days: int = 90) -> int:
        """
        Delete old analysis runs.
        
        Args:
            days: Delete runs older than this many days
            
        Returns:
            Number of deleted runs
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        deleted = self.session.query(AnalysisRun).filter(
            AnalysisRun.created_at < cutoff
        ).delete()
        
        self.session.flush()
        logger.info(f"Cleaned up {deleted} old analysis runs")
        return deleted
