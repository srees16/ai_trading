# Centurion Capital LLC - Database Service Layer
"""
High-level database service providing a unified API for business operations.
Coordinates repositories and manages transactions.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
from contextlib import contextmanager

from sqlalchemy.orm import Session

from database.connection import DatabaseManager
from database.models import (
    AnalysisRun, NewsItem, StockSignal, FundamentalMetric,
    BacktestResult, AnalysisStatus
)
from database.repositories import (
    AnalysisRepository, SignalRepository, NewsRepository,
    FundamentalRepository, BacktestRepository
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Unified database service for the Centurion trading platform.
    
    This service provides high-level operations that coordinate
    multiple repositories within proper transaction boundaries.
    
    Usage:
        service = DatabaseService()
        
        # Save analysis results
        service.save_analysis_results(
            tickers=['AAPL', 'GOOGL'],
            signals=[...],
            news_items=[...]
        )
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._db_manager = DatabaseManager()
        self._initialized = True
        logger.info("DatabaseService initialized")
    
    @property
    def is_available(self) -> bool:
        """Check if database is available."""
        return self._db_manager.health_check()
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        
        Usage:
            with service.session_scope() as session:
                # do work
        """
        session = self._db_manager.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            session.close()
    
    # =================================================================
    # Analysis Run Operations
    # =================================================================
    
    def start_analysis_run(
        self,
        run_type: str,
        tickers: List[str],
        parameters: Dict[str, Any] = None,
        user_id: str = None
    ) -> Optional[UUID]:
        """
        Start a new analysis run and return its ID for tracking.
        
        Args:
            run_type: Type of analysis ('stock_analysis', 'backtest', 'fundamental')
            tickers: List of tickers to analyze
            parameters: Analysis parameters
            user_id: Optional user identifier
            
        Returns:
            Run ID for tracking, or None if database unavailable
        """
        if not self.is_available:
            logger.warning("Database unavailable, skipping run tracking")
            return None
        
        try:
            with self.session_scope() as session:
                repo = AnalysisRepository(session)
                run = repo.create_run(
                    run_type=run_type,
                    tickers=tickers,
                    parameters=parameters,
                    user_id=user_id
                )
                repo.start_run(run.id)
                return run.id
        except Exception as e:
            logger.error(f"Failed to start analysis run: {e}")
            return None
    
    def complete_analysis_run(
        self,
        run_id: UUID,
        total_signals: int = 0,
        total_news_items: int = 0
    ) -> bool:
        """
        Mark an analysis run as completed.
        
        Args:
            run_id: Run ID to complete
            total_signals: Number of signals generated
            total_news_items: Number of news items processed
            
        Returns:
            True if successful
        """
        if not run_id:
            return False
        
        try:
            with self.session_scope() as session:
                repo = AnalysisRepository(session)
                repo.complete_run(run_id, total_signals, total_news_items)
                return True
        except Exception as e:
            logger.error(f"Failed to complete analysis run: {e}")
            return False
    
    def fail_analysis_run(
        self,
        run_id: UUID,
        error_message: str,
        error_traceback: str = None
    ) -> bool:
        """Mark an analysis run as failed."""
        if not run_id:
            return False
        
        try:
            with self.session_scope() as session:
                repo = AnalysisRepository(session)
                repo.fail_run(run_id, error_message, error_traceback)
                return True
        except Exception as e:
            logger.error(f"Failed to mark run as failed: {e}")
            return False
    
    # =================================================================
    # Signal Operations
    # =================================================================
    
    def save_signals(
        self,
        signals: List[Dict[str, Any]],
        analysis_run_id: UUID = None
    ) -> int:
        """
        Save trading signals to database.
        
        Args:
            signals: List of signal dictionaries with keys:
                - ticker: Stock ticker
                - decision: 'BUY', 'SELL', or 'HOLD'
                - confidence: Confidence score (0-100)
                - strategy_name: Strategy that generated signal
                - reasons: List of reason strings
                - price_target: Optional target price
                - stop_loss: Optional stop loss
                - metadata: Optional additional data
            analysis_run_id: Optional run ID to link signals
            
        Returns:
            Number of signals saved
        """
        if not signals or not self.is_available:
            return 0
        
        try:
            with self.session_scope() as session:
                repo = SignalRepository(session)
                saved = 0
                
                for sig in signals:
                    # Map decision string to enum
                    decision_str = sig.get('decision', 'HOLD').upper()
                    
                    signal = StockSignal(
                        analysis_run_id=analysis_run_id,
                        ticker=sig.get('ticker', '').upper(),
                        decision=decision_str,
                        decision_score=sig.get('confidence', 0.0) / 100.0 if sig.get('confidence', 0) > 1 else sig.get('confidence', 0.0),
                        reasoning='; '.join(sig.get('reasons', [])) if isinstance(sig.get('reasons'), list) else sig.get('reasons', ''),
                        current_price=sig.get('current_price') or sig.get('price_target'),
                        decision_factors=sig.get('metadata', {}),
                        tags=sig.get('tags', [])
                    )
                    repo.create(signal)
                    saved += 1
                
                logger.info(f"Saved {saved} signals to database")
                return saved
                
        except Exception as e:
            logger.error(f"Failed to save signals: {e}")
            return 0
    
    def get_latest_signals(
        self,
        tickers: List[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get latest signals, optionally filtered by tickers."""
        try:
            with self.session_scope() as session:
                repo = SignalRepository(session)
                
                if tickers:
                    signals = []
                    for ticker in tickers:
                        ticker_signals = repo.get_by_ticker(ticker, limit=10)
                        signals.extend(ticker_signals)
                else:
                    signals = repo.get_recent_signals_summary(limit=limit)
                    return signals if isinstance(signals, list) else []
                
                return [
                    {
                        'ticker': s.ticker,
                        'decision': s.decision,
                        'confidence': s.confidence_score,
                        'strategy': s.strategy_name,
                        'reasons': s.reasons,
                        'created_at': s.created_at.isoformat()
                    }
                    for s in signals
                ]
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return []
    
    # =================================================================
    # News Operations
    # =================================================================
    
    def save_news_items(
        self,
        news_items: List[Dict[str, Any]],
        analysis_run_id: UUID = None
    ) -> int:
        """
        Save news items to database with deduplication.
        
        Args:
            news_items: List of news dictionaries with keys:
                - ticker: Stock ticker
                - headline: News headline
                - summary: News summary
                - source: News source
                - url: Article URL
                - published_at: Publication datetime
                - sentiment_score: Sentiment (-1 to 1)
                - impact_score: Impact score (0-100)
            analysis_run_id: Optional run ID to link news
            
        Returns:
            Number of news items saved (excluding duplicates)
        """
        if not news_items or not self.is_available:
            return 0
        
        try:
            with self.session_scope() as session:
                repo = NewsRepository(session)
                saved = 0
                
                for item in news_items:
                    # Deduplication check
                    headline = item.get('headline', item.get('title', ''))
                    if repo.check_duplicate(
                        ticker=item.get('ticker', ''),
                        headline=headline,
                        url=item.get('url', '')
                    ):
                        continue
                    
                    news = NewsItem(
                        analysis_run_id=analysis_run_id,
                        ticker=item.get('ticker', '').upper(),
                        title=headline,
                        content=item.get('summary', item.get('content', '')),
                        source=item.get('source', 'unknown'),
                        url=item.get('url', ''),
                        published_at=item.get('published_at'),
                        sentiment_label=item.get('sentiment_label'),
                        sentiment_confidence=item.get('sentiment_score'),
                        entities=item.get('entities', {}),
                    )
                    repo.create(news)
                    saved += 1
                
                logger.info(f"Saved {saved} news items to database")
                return saved
                
        except Exception as e:
            logger.error(f"Failed to save news items: {e}")
            return 0
    
    # =================================================================
    # Fundamental Metrics Operations
    # =================================================================
    
    def save_fundamental_metrics(
        self,
        metrics: List[Dict[str, Any]],
        analysis_run_id: UUID = None
    ) -> int:
        """
        Save fundamental metrics with upsert logic.
        
        Args:
            metrics: List of metric dictionaries
            analysis_run_id: Optional run ID
            
        Returns:
            Number of metrics saved
        """
        if not metrics or not self.is_available:
            return 0
        
        try:
            with self.session_scope() as session:
                repo = FundamentalRepository(session)
                saved = 0
                
                for m in metrics:
                    # Create FundamentalMetric object
                    metric = FundamentalMetric(
                        ticker=m.get('ticker', '').upper(),
                        recorded_at=datetime.utcnow(),
                        current_price=m.get('current_price'),
                        market_cap=m.get('market_cap'),
                        enterprise_value=m.get('enterprise_value'),
                        pe_ratio=m.get('pe_ratio'),
                        forward_pe=m.get('forward_pe'),
                        peg_ratio=m.get('peg_ratio'),
                        pb_ratio=m.get('pb_ratio'),
                        ps_ratio=m.get('ps_ratio'),
                        ev_to_ebitda=m.get('ev_ebitda'),
                        altman_z_score=m.get('altman_z_score'),
                        beneish_m_score=m.get('beneish_m_score'),
                        piotroski_f_score=m.get('piotroski_f_score'),
                        profit_margin=m.get('profit_margin'),
                        operating_margin=m.get('operating_margin'),
                        return_on_equity=m.get('roe'),
                        return_on_assets=m.get('roa'),
                        current_ratio=m.get('current_ratio'),
                        quick_ratio=m.get('quick_ratio'),
                        debt_to_equity=m.get('debt_to_equity'),
                        revenue_growth=m.get('revenue_growth'),
                        earnings_growth=m.get('earnings_growth'),
                        dividend_yield=m.get('dividend_yield'),
                        payout_ratio=m.get('payout_ratio'),
                        data_source=m.get('data_source', 'yahoo_finance')
                    )
                    result = repo.upsert(metric)
                    if result:
                        saved += 1
                
                logger.info(f"Saved {saved} fundamental metrics")
                return saved
                
        except Exception as e:
            logger.error(f"Failed to save fundamental metrics: {e}")
            return 0
    
    # =================================================================
    # Backtest Operations
    # =================================================================
    
    def save_backtest_result(
        self,
        result: Dict[str, Any],
        analysis_run_id: UUID = None
    ) -> bool:
        """
        Save a backtest result.
        
        Args:
            result: Backtest result dictionary with keys:
                - strategy_name: Name of strategy
                - ticker: Ticker tested
                - start_date/end_date: Test period
                - initial_capital: Starting capital
                - final_value: Ending portfolio value
                - total_return: Total return percentage
                - annualized_return: Annualized return
                - max_drawdown: Maximum drawdown
                - sharpe_ratio: Sharpe ratio
                - sortino_ratio: Sortino ratio
                - win_rate: Win rate
                - total_trades: Number of trades
                - parameters: Strategy parameters
                - daily_returns: List of daily returns
            analysis_run_id: Optional run ID
            
        Returns:
            True if saved successfully
        """
        if not result or not self.is_available:
            return False
        
        try:
            with self.session_scope() as session:
                repo = BacktestRepository(session)
                
                # Handle both single ticker and list of tickers
                tickers_input = result.get('tickers') or result.get('ticker', '')
                if isinstance(tickers_input, str):
                    tickers_list = [tickers_input.upper()] if tickers_input else []
                else:
                    tickers_list = [t.upper() for t in tickers_input if t]
                
                backtest = BacktestResult(
                    analysis_run_id=analysis_run_id,
                    strategy_id=result.get('strategy_id', result.get('strategy_name', 'unknown').lower().replace(' ', '_')),
                    strategy_name=result.get('strategy_name', 'unknown'),
                    tickers=tickers_list,
                    start_date=result.get('start_date'),
                    end_date=result.get('end_date'),
                    initial_capital=result.get('initial_capital', 10000),
                    final_value=result.get('final_value'),
                    total_return=result.get('total_return'),
                    annualized_return=result.get('annualized_return'),
                    max_drawdown=result.get('max_drawdown'),
                    sharpe_ratio=result.get('sharpe_ratio'),
                    sortino_ratio=result.get('sortino_ratio'),
                    calmar_ratio=result.get('calmar_ratio'),
                    win_rate=result.get('win_rate'),
                    profit_factor=result.get('profit_factor'),
                    total_trades=result.get('total_trades', 0),
                    winning_trades=result.get('winning_trades'),
                    losing_trades=result.get('losing_trades'),
                    avg_win=result.get('avg_win'),
                    avg_loss=result.get('avg_loss'),
                    largest_win=result.get('largest_win'),
                    largest_loss=result.get('largest_loss'),
                    parameters=result.get('parameters', {}),
                    daily_returns=result.get('daily_returns', []),
                    trade_log=result.get('trade_log', [])
                )
                repo.create(backtest)
                logger.info(f"Saved backtest result for {tickers_list}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save backtest result: {e}")
            return False
    
    def get_strategy_performance(
        self,
        strategy_name: str = None,
        ticker: str = None
    ) -> List[Dict[str, Any]]:
        """Get historical backtest performance."""
        try:
            with self.session_scope() as session:
                repo = BacktestRepository(session)
                
                if strategy_name:
                    results = repo.get_by_strategy(strategy_name, limit=50)
                elif ticker:
                    results = repo.get_all(limit=50)
                    results = [r for r in results if r.ticker == ticker.upper()]
                else:
                    return repo.get_strategy_summary()
                
                return [
                    {
                        'strategy': r.strategy_name,
                        'ticker': r.ticker,
                        'total_return': r.total_return,
                        'sharpe_ratio': r.sharpe_ratio,
                        'max_drawdown': r.max_drawdown,
                        'win_rate': r.win_rate,
                        'created_at': r.created_at.isoformat()
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return []
    
    # =================================================================
    # Combined Analysis Save
    # =================================================================
    
    def save_complete_analysis(
        self,
        tickers: List[str],
        signals: List[Dict[str, Any]] = None,
        news_items: List[Dict[str, Any]] = None,
        fundamental_metrics: List[Dict[str, Any]] = None,
        backtest_results: List[Dict[str, Any]] = None,
        parameters: Dict[str, Any] = None,
        run_type: str = 'stock_analysis'
    ) -> Tuple[UUID, Dict[str, int]]:
        """
        Save complete analysis results in a single transaction.
        
        This is the primary method for saving analysis results from app.py.
        
        Args:
            tickers: List of analyzed tickers
            signals: Trading signals list
            news_items: News items list
            fundamental_metrics: Fundamental metrics list
            backtest_results: Backtest results list
            parameters: Analysis parameters
            run_type: Type of analysis
            
        Returns:
            Tuple of (run_id, counts_dict)
        """
        counts = {
            'signals': 0,
            'news': 0,
            'fundamentals': 0,
            'backtests': 0
        }
        
        if not self.is_available:
            logger.warning("Database unavailable, analysis not saved")
            return None, counts
        
        # Start the analysis run
        run_id = self.start_analysis_run(
            run_type=run_type,
            tickers=tickers,
            parameters=parameters
        )
        
        try:
            # Save all components
            if signals:
                counts['signals'] = self.save_signals(signals, run_id)
            
            if news_items:
                counts['news'] = self.save_news_items(news_items, run_id)
            
            if fundamental_metrics:
                counts['fundamentals'] = self.save_fundamental_metrics(
                    fundamental_metrics, run_id
                )
            
            if backtest_results:
                for result in backtest_results:
                    if self.save_backtest_result(result, run_id):
                        counts['backtests'] += 1
            
            # Complete the run
            self.complete_analysis_run(
                run_id,
                total_signals=counts['signals'],
                total_news_items=counts['news']
            )
            
            logger.info(
                f"Analysis saved: {counts['signals']} signals, "
                f"{counts['news']} news, {counts['fundamentals']} metrics, "
                f"{counts['backtests']} backtests"
            )
            
            return run_id, counts
            
        except Exception as e:
            logger.error(f"Failed to save complete analysis: {e}")
            self.fail_analysis_run(run_id, str(e))
            return run_id, counts
    
    # =================================================================
    # Utility Methods
    # =================================================================
    
    def get_run_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get analysis run statistics."""
        try:
            with self.session_scope() as session:
                repo = AnalysisRepository(session)
                return repo.get_run_statistics(days)
        except Exception as e:
            logger.error(f"Failed to get run statistics: {e}")
            return {}
    
    def initialize_database(self) -> bool:
        """Initialize database tables and hypertables."""
        return self._db_manager.create_tables()
    
    def close(self):
        """Close database connections."""
        self._db_manager.close()


# Global service instance
_service_instance = None


def get_database_service() -> DatabaseService:
    """
    Get the global DatabaseService instance.
    
    Usage:
        from database.service import get_database_service
        
        db = get_database_service()
        db.save_signals([...])
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = DatabaseService()
    return _service_instance
