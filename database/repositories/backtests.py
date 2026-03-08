# Centurion Capital LLC - Backtest Repository
"""
Repository for backtesting results data access.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func

from database.models import BacktestResult
from database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class BacktestRepository(BaseRepository[BacktestResult]):
    """Repository for BacktestResult entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, BacktestResult)
    
    def get_by_strategy(
        self,
        strategy_id: str,
        limit: int = 50
    ) -> List[BacktestResult]:
        """
        Get backtest results for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            limit: Maximum results
            
        Returns:
            List of backtest results
        """
        return self.session.query(BacktestResult).filter(
            BacktestResult.strategy_id == strategy_id
        ).order_by(desc(BacktestResult.created_at)).limit(limit).all()
    
    def get_by_ticker(
        self,
        ticker: str,
        limit: int = 50
    ) -> List[BacktestResult]:
        """
        Get backtest results containing a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum results
            
        Returns:
            List of backtest results
        """
        return self.session.query(BacktestResult).filter(
            BacktestResult.tickers.contains([ticker.upper()])
        ).order_by(desc(BacktestResult.created_at)).limit(limit).all()
    
    def get_top_performers(
        self,
        limit: int = 10,
        strategy_id: str = None,
        min_trades: int = 5
    ) -> List[BacktestResult]:
        """
        Get top performing backtest results.
        
        Args:
            limit: Maximum results
            strategy_id: Filter by strategy ID
            min_trades: Minimum number of trades required
            
        Returns:
            List of top performing backtests
        """
        query = self.session.query(BacktestResult).filter(
            and_(
                BacktestResult.success == True,
                BacktestResult.total_trades >= min_trades
            )
        )
        
        if strategy_id:
            query = query.filter(BacktestResult.strategy_id == strategy_id)
        
        return query.order_by(desc(BacktestResult.total_return)).limit(limit).all()
    
    def get_strategy_summary(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a strategy's backtest history.
        Uses a single SQL aggregate query instead of loading all rows.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Summary dict with aggregated metrics
        """
        result = self.session.query(
            func.min(BacktestResult.strategy_name).label('strategy_name'),
            func.count(BacktestResult.id).label('total_backtests'),
            func.avg(BacktestResult.total_return).label('avg_return'),
            func.avg(BacktestResult.sharpe_ratio).label('avg_sharpe'),
            func.max(BacktestResult.total_return).label('best_return'),
            func.min(BacktestResult.total_return).label('worst_return'),
            func.avg(BacktestResult.win_rate).label('avg_win_rate'),
        ).filter(
            and_(
                BacktestResult.strategy_id == strategy_id,
                BacktestResult.success == True
            )
        ).first()
        
        if not result or not result.total_backtests:
            return {
                'strategy_id': strategy_id,
                'total_backtests': 0,
                'avg_return': None,
                'avg_sharpe': None,
                'best_return': None,
                'worst_return': None
            }
        
        return {
            'strategy_id': strategy_id,
            'strategy_name': result.strategy_name,
            'total_backtests': result.total_backtests,
            'successful_backtests': result.total_backtests,  # already filtered
            'avg_return': float(result.avg_return) if result.avg_return is not None else None,
            'avg_sharpe': float(result.avg_sharpe) if result.avg_sharpe is not None else None,
            'best_return': float(result.best_return) if result.best_return is not None else None,
            'worst_return': float(result.worst_return) if result.worst_return is not None else None,
            'avg_win_rate': float(result.avg_win_rate) if result.avg_win_rate is not None else None,
        }
    
    def get_all_strategies_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary for all strategies in a single GROUP BY query.
        Eliminates the previous N+1 pattern.
        
        Returns:
            List of strategy summaries
        """
        rows = self.session.query(
            BacktestResult.strategy_id,
            func.min(BacktestResult.strategy_name).label('strategy_name'),
            func.count(BacktestResult.id).label('total_backtests'),
            func.avg(BacktestResult.total_return).label('avg_return'),
            func.avg(BacktestResult.sharpe_ratio).label('avg_sharpe'),
            func.max(BacktestResult.total_return).label('best_return'),
            func.min(BacktestResult.total_return).label('worst_return'),
            func.avg(BacktestResult.win_rate).label('avg_win_rate'),
        ).filter(
            BacktestResult.success == True
        ).group_by(
            BacktestResult.strategy_id
        ).all()
        
        return [
            {
                'strategy_id': r.strategy_id,
                'strategy_name': r.strategy_name,
                'total_backtests': r.total_backtests,
                'successful_backtests': r.total_backtests,
                'avg_return': float(r.avg_return) if r.avg_return is not None else None,
                'avg_sharpe': float(r.avg_sharpe) if r.avg_sharpe is not None else None,
                'best_return': float(r.best_return) if r.best_return is not None else None,
                'worst_return': float(r.worst_return) if r.worst_return is not None else None,
                'avg_win_rate': float(r.avg_win_rate) if r.avg_win_rate is not None else None,
            }
            for r in rows
        ]
    
    def get_recent_backtests(
        self,
        days: int = 7,
        limit: int = 50,
        market: str = None,
    ) -> List[BacktestResult]:
        """
        Get recently run backtests.
        
        Args:
            days: Number of days back
            limit: Maximum results
            market: Filter by market ('US', 'IND')
            
        Returns:
            List of recent backtests
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = self.session.query(BacktestResult).filter(
            BacktestResult.created_at >= cutoff
        )
        
        if market:
            query = query.filter(BacktestResult.market == market)
        
        return query.order_by(desc(BacktestResult.created_at)).limit(limit).all()
    
    def compare_strategies(
        self,
        strategy_ids: List[str],
        ticker: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies side by side in a single SQL query.
        Eliminates the previous N+1 loop pattern.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            ticker: Optional ticker filter
            start_date: Optional date range start
            end_date: Optional date range end
            
        Returns:
            Comparison data for all strategies
        """
        query = self.session.query(
            BacktestResult.strategy_id,
            func.min(BacktestResult.strategy_name).label('strategy_name'),
            func.count(BacktestResult.id).label('backtest_count'),
            func.avg(BacktestResult.total_return).label('avg_return'),
            func.avg(BacktestResult.sharpe_ratio).label('avg_sharpe'),
            func.avg(BacktestResult.max_drawdown).label('avg_drawdown'),
            func.max(BacktestResult.total_return).label('best_return'),
            func.min(BacktestResult.total_return).label('worst_return'),
        ).filter(
            and_(
                BacktestResult.strategy_id.in_(strategy_ids),
                BacktestResult.success == True
            )
        )
        
        if ticker:
            query = query.filter(BacktestResult.tickers.contains([ticker.upper()]))
        if start_date:
            query = query.filter(BacktestResult.start_date >= start_date)
        if end_date:
            query = query.filter(BacktestResult.end_date <= end_date)
        
        rows = query.group_by(BacktestResult.strategy_id).all()
        
        return {
            'strategies': [
                {
                    'strategy_id': r.strategy_id,
                    'strategy_name': r.strategy_name,
                    'backtest_count': r.backtest_count,
                    'avg_return': float(r.avg_return) if r.avg_return is not None else None,
                    'avg_sharpe': float(r.avg_sharpe) if r.avg_sharpe is not None else None,
                    'avg_drawdown': float(r.avg_drawdown) if r.avg_drawdown is not None else None,
                    'best_return': float(r.best_return) if r.best_return is not None else None,
                    'worst_return': float(r.worst_return) if r.worst_return is not None else None,
                }
                for r in rows
            ]
        }
