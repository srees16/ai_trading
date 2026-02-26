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
from sqlalchemy import Integer

from database.connection import DatabaseManager
from database.models import (
    NewsItem, StockSignal, FundamentalMetric,
    BacktestResult, AnalysisStatus, SentimentType,
    BacktestTrade, BacktestEquityPoint,
    BacktestDailyReturn, StrategyPerformanceSummary, DataFreshness,
)
from database.repositories import (
    AnalysisRepository, SignalRepository, NewsRepository,
    FundamentalRepository, BacktestRepository, FreshnessRepository,
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
        result = self._db_manager.health_check()
        return result.get('healthy', False) if isinstance(result, dict) else bool(result)
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        
        Usage:
            with service.session_scope() as session:
                # do work
        """
        with self._db_manager.get_session() as session:
            yield session
    
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
                    summary = item.get('summary', item.get('content', ''))
                    if repo.check_duplicate(
                        title=headline,
                        content=summary,
                        ticker=item.get('ticker', '')
                    ):
                        continue
                    
                    # Convert sentiment label string to enum
                    sentiment_raw = item.get('sentiment_label')
                    sentiment_enum = None
                    if sentiment_raw:
                        if isinstance(sentiment_raw, str):
                            try:
                                sentiment_enum = SentimentType(sentiment_raw.lower())
                            except ValueError:
                                sentiment_enum = SentimentType.NEUTRAL
                        else:
                            sentiment_enum = sentiment_raw

                    news = NewsItem(
                        analysis_run_id=analysis_run_id,
                        ticker=item.get('ticker', '').upper(),
                        title=headline,
                        content=item.get('summary', item.get('content', '')),
                        source=item.get('source', 'unknown'),
                        url=item.get('url', ''),
                        published_at=item.get('published_at'),
                        sentiment_label=sentiment_enum,
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
        Save a backtest result with normalised detail tables.
        
        Persists the core BacktestResult row **and** populates the
        normalised BacktestTrade, BacktestEquityPoint, and
        BacktestDailyReturn tables extracted from JSONB data.
        After saving, refreshes the StrategyPerformanceSummary.
        
        Args:
            result: Backtest result dictionary
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
                
                # --- Separate large arrays from scalar metrics --------
                scalar_metrics = {
                    k: result[k]
                    for k in ('final_value', 'largest_win', 'largest_loss')
                    if k in result
                }
                
                backtest = BacktestResult(
                    strategy_id=result.get('strategy_id', result.get('strategy_name', 'unknown').lower().replace(' ', '_')),
                    strategy_name=result.get('strategy_name', 'unknown'),
                    tickers=tickers_list,
                    start_date=result.get('start_date'),
                    end_date=result.get('end_date'),
                    initial_capital=result.get('initial_capital', 10000),
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
                    parameters=result.get('parameters', {}),
                    metrics=scalar_metrics,
                    success=True,
                )
                repo.create(backtest)
                
                # --- Normalised trade signals -------------------------
                trade_log = result.get('trade_log') or result.get('signals') or []
                for idx, trade in enumerate(trade_log):
                    if not isinstance(trade, dict):
                        continue
                    session.add(BacktestTrade(
                        backtest_id=backtest.id,
                        trade_number=idx + 1,
                        trade_type=str(trade.get('type', trade.get('action', 'BUY'))).upper(),
                        ticker=str(trade.get('ticker', tickers_list[0] if tickers_list else '')),
                        entry_date=trade.get('entry_date'),
                        exit_date=trade.get('exit_date'),
                        entry_price=trade.get('entry_price'),
                        exit_price=trade.get('exit_price'),
                        quantity=trade.get('quantity') or trade.get('shares'),
                        pnl=trade.get('pnl') or trade.get('profit'),
                        pnl_pct=trade.get('pnl_pct') or trade.get('return_pct'),
                        holding_period_days=trade.get('holding_period_days'),
                    ))
                
                # --- Normalised equity curve --------------------------
                equity_data = result.get('equity_curve') or []
                for pt in equity_data:
                    if not isinstance(pt, dict):
                        continue
                    session.add(BacktestEquityPoint(
                        backtest_id=backtest.id,
                        point_date=pt.get('date'),
                        portfolio_value=pt.get('value') or pt.get('portfolio_value'),
                        drawdown=pt.get('drawdown'),
                        benchmark_value=pt.get('benchmark'),
                    ))
                
                # --- Normalised daily returns -------------------------
                daily_returns = result.get('daily_returns') or []
                for dr in daily_returns:
                    if isinstance(dr, dict):
                        session.add(BacktestDailyReturn(
                            backtest_id=backtest.id,
                            return_date=dr.get('date'),
                            daily_return=dr.get('return', dr.get('daily_return', 0)),
                            cumulative_return=dr.get('cumulative_return'),
                        ))
                    elif isinstance(dr, (int, float)):
                        # Simple list of floats — no date info available
                        session.add(BacktestDailyReturn(
                            backtest_id=backtest.id,
                            return_date=None,
                            daily_return=float(dr),
                        ))
                
                session.flush()
                
                # --- Refresh pre-aggregated summary -------------------
                self._refresh_strategy_summary(session, backtest.strategy_id)
                
                logger.info(f"Saved backtest result for {tickers_list}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save backtest result: {e}")
            return False
    
    # =================================================================
    # Pre-Aggregated Summary Refresh
    # =================================================================
    
    def _refresh_strategy_summary(self, session: Session, strategy_id: str):
        """
        Refresh the StrategyPerformanceSummary row for a strategy.
        Called after each backtest save.
        """
        from sqlalchemy import func as sqla_func
        
        try:
            row = session.query(
                sqla_func.count(BacktestResult.id).label('total'),
                sqla_func.sum(
                    sqla_func.cast(BacktestResult.success == True, Integer)
                ).label('successful'),
                sqla_func.min(BacktestResult.strategy_name).label('name'),
                sqla_func.avg(BacktestResult.total_return).label('avg_return'),
                sqla_func.max(BacktestResult.total_return).label('best_return'),
                sqla_func.min(BacktestResult.total_return).label('worst_return'),
                sqla_func.avg(BacktestResult.sharpe_ratio).label('avg_sharpe'),
                sqla_func.avg(BacktestResult.sortino_ratio).label('avg_sortino'),
                sqla_func.avg(BacktestResult.max_drawdown).label('avg_mdd'),
                sqla_func.avg(BacktestResult.calmar_ratio).label('avg_calmar'),
                sqla_func.avg(BacktestResult.win_rate).label('avg_wr'),
                sqla_func.avg(BacktestResult.profit_factor).label('avg_pf'),
                sqla_func.avg(BacktestResult.total_trades).label('avg_trades'),
                sqla_func.max(BacktestResult.created_at).label('last_bt'),
            ).filter(
                BacktestResult.strategy_id == strategy_id,
                BacktestResult.success == True,
            ).first()
            
            if not row or not row.total:
                return
            
            summary = session.query(StrategyPerformanceSummary).filter(
                StrategyPerformanceSummary.strategy_id == strategy_id
            ).first()
            
            now = datetime.utcnow()
            
            if summary:
                summary.strategy_name = row.name
                summary.total_backtests = row.total
                summary.successful_backtests = int(row.successful or 0)
                summary.avg_return = float(row.avg_return) if row.avg_return else None
                summary.best_return = float(row.best_return) if row.best_return else None
                summary.worst_return = float(row.worst_return) if row.worst_return else None
                summary.avg_sharpe = float(row.avg_sharpe) if row.avg_sharpe else None
                summary.avg_sortino = float(row.avg_sortino) if row.avg_sortino else None
                summary.avg_max_drawdown = float(row.avg_mdd) if row.avg_mdd else None
                summary.avg_calmar = float(row.avg_calmar) if row.avg_calmar else None
                summary.avg_win_rate = float(row.avg_wr) if row.avg_wr else None
                summary.avg_profit_factor = float(row.avg_pf) if row.avg_pf else None
                summary.avg_total_trades = float(row.avg_trades) if row.avg_trades else None
                summary.last_backtest_at = row.last_bt
                summary.last_refreshed_at = now
            else:
                summary = StrategyPerformanceSummary(
                    strategy_id=strategy_id,
                    strategy_name=row.name,
                    total_backtests=row.total,
                    successful_backtests=int(row.successful or 0),
                    avg_return=float(row.avg_return) if row.avg_return else None,
                    best_return=float(row.best_return) if row.best_return else None,
                    worst_return=float(row.worst_return) if row.worst_return else None,
                    avg_sharpe=float(row.avg_sharpe) if row.avg_sharpe else None,
                    avg_sortino=float(row.avg_sortino) if row.avg_sortino else None,
                    avg_max_drawdown=float(row.avg_mdd) if row.avg_mdd else None,
                    avg_calmar=float(row.avg_calmar) if row.avg_calmar else None,
                    avg_win_rate=float(row.avg_wr) if row.avg_wr else None,
                    avg_profit_factor=float(row.avg_pf) if row.avg_pf else None,
                    avg_total_trades=float(row.avg_trades) if row.avg_trades else None,
                    last_backtest_at=row.last_bt,
                    last_refreshed_at=now,
                )
                session.add(summary)
            
            session.flush()
        except Exception as e:
            logger.warning(f"Failed to refresh strategy summary for {strategy_id}: {e}")
    
    # =================================================================
    # Data Freshness Operations
    # =================================================================
    
    def check_freshness(
        self,
        ticker: str,
        data_type: str,
        max_age_minutes: int = 30
    ) -> bool:
        """
        Check if data for a ticker/type needs refreshing.
        
        Returns:
            True if data is fresh (no refresh needed)
        """
        if not self.is_available:
            return False
        try:
            with self.session_scope() as session:
                repo = FreshnessRepository(session)
                return not repo.is_stale(ticker, data_type, max_age_minutes)
        except Exception as e:
            logger.warning(f"Freshness check failed: {e}")
            return False
    
    def record_fetch(
        self,
        ticker: str,
        data_type: str,
        record_count: int = 0,
        fetch_seconds: float = 0.0,
        refresh_minutes: int = 30,
    ):
        """Record a successful data fetch for freshness tracking."""
        if not self.is_available:
            return
        try:
            with self.session_scope() as session:
                repo = FreshnessRepository(session)
                repo.record_fetch(ticker, data_type, record_count, fetch_seconds, refresh_minutes)
        except Exception as e:
            logger.warning(f"Failed to record fetch: {e}")
    
    def record_fetch_error(self, ticker: str, data_type: str, error: str):
        """Record a failed data fetch for freshness tracking."""
        if not self.is_available:
            return
        try:
            with self.session_scope() as session:
                repo = FreshnessRepository(session)
                repo.record_error(ticker, data_type, error)
        except Exception as e:
            logger.warning(f"Failed to record fetch error: {e}")
    
    def get_ticker_freshness(self, ticker: str) -> List[Dict[str, Any]]:
        """Get freshness info for all data types for a ticker."""
        if not self.is_available:
            return []
        try:
            with self.session_scope() as session:
                repo = FreshnessRepository(session)
                return repo.get_ticker_freshness(ticker)
        except Exception as e:
            logger.warning(f"Failed to get freshness: {e}")
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
