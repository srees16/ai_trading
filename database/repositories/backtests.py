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
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Summary dict with aggregated metrics
        """
        results = self.session.query(BacktestResult).filter(
            and_(
                BacktestResult.strategy_id == strategy_id,
                BacktestResult.success == True
            )
        ).all()
        
        if not results:
            return {
                'strategy_id': strategy_id,
                'total_backtests': 0,
                'avg_return': None,
                'avg_sharpe': None,
                'best_return': None,
                'worst_return': None
            }
        
        returns = [r.total_return for r in results if r.total_return is not None]
        sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio is not None]
        
        return {
            'strategy_id': strategy_id,
            'strategy_name': results[0].strategy_name,
            'total_backtests': len(results),
            'successful_backtests': sum(1 for r in results if r.success),
            'avg_return': sum(returns) / len(returns) if returns else None,
            'avg_sharpe': sum(sharpes) / len(sharpes) if sharpes else None,
            'best_return': max(returns) if returns else None,
            'worst_return': min(returns) if returns else None,
            'avg_win_rate': sum(r.win_rate for r in results if r.win_rate) / len([r for r in results if r.win_rate]),
        }
    
    def get_all_strategies_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary for all strategies.
        
        Returns:
            List of strategy summaries
        """
        # Get distinct strategy IDs
        strategy_ids = self.session.query(
            BacktestResult.strategy_id
        ).distinct().all()
        
        return [
            self.get_strategy_summary(sid[0])
            for sid in strategy_ids
        ]
    
    def get_recent_backtests(
        self,
        days: int = 7,
        limit: int = 50
    ) -> List[BacktestResult]:
        """
        Get recently run backtests.
        
        Args:
            days: Number of days back
            limit: Maximum results
            
        Returns:
            List of recent backtests
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        return self.session.query(BacktestResult).filter(
            BacktestResult.created_at >= cutoff
        ).order_by(desc(BacktestResult.created_at)).limit(limit).all()
    
    def compare_strategies(
        self,
        strategy_ids: List[str],
        ticker: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies side by side.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            ticker: Optional ticker filter
            start_date: Optional date range start
            end_date: Optional date range end
            
        Returns:
            Comparison data for all strategies
        """
        comparison = {'strategies': []}
        
        for strategy_id in strategy_ids:
            query = self.session.query(BacktestResult).filter(
                and_(
                    BacktestResult.strategy_id == strategy_id,
                    BacktestResult.success == True
                )
            )
            
            if ticker:
                query = query.filter(BacktestResult.tickers.contains([ticker.upper()]))
            if start_date:
                query = query.filter(BacktestResult.start_date >= start_date)
            if end_date:
                query = query.filter(BacktestResult.end_date <= end_date)
            
            results = query.all()
            
            if results:
                returns = [r.total_return for r in results if r.total_return]
                sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio]
                drawdowns = [r.max_drawdown for r in results if r.max_drawdown]
                
                comparison['strategies'].append({
                    'strategy_id': strategy_id,
                    'strategy_name': results[0].strategy_name,
                    'backtest_count': len(results),
                    'avg_return': sum(returns) / len(returns) if returns else None,
                    'avg_sharpe': sum(sharpes) / len(sharpes) if sharpes else None,
                    'avg_drawdown': sum(drawdowns) / len(drawdowns) if drawdowns else None,
                    'best_return': max(returns) if returns else None,
                    'worst_return': min(returns) if returns else None,
                })
        
        return comparison
