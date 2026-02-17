"""
Analysis Service Module for Centurion Capital LLC.

Contains the stock analysis execution logic and database persistence.
"""

import streamlit as st
import logging
from typing import List, Any, Optional, Tuple, Dict

from config import Config
from main import AlgoTradingSystem
from ui.components import render_completion_banner

logger = logging.getLogger(__name__)

# Database availability check
try:
    from database.service import get_database_service
    DB_AVAILABLE = Config.is_database_configured()
except ImportError:
    DB_AVAILABLE = False
    get_database_service = None
    logger.warning("Database module not available - results won't be persisted")


async def run_analysis_async(tickers: List[str]) -> List[Any]:
    """
    Run the stock analysis asynchronously.
    
    Args:
        tickers: List of stock ticker symbols to analyze
    
    Returns:
        List of TradingSignal objects
    """
    system = AlgoTradingSystem(tickers=tickers)
    
    # Progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with st.spinner('🔄 Analyzing...'):
        # Fetch news
        signals = await _execute_analysis(
            system, tickers, progress_placeholder, status_placeholder
        )
        
        if not signals:
            return []
        
        # Save results
        save_path = _save_results(system, signals, tickers)
        db_saved = _save_to_database(signals, tickers)
        
        # Clear progress indicators
        progress_placeholder.progress(100)
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Show completion banner
        render_completion_banner(len(tickers), len(signals), db_saved)
        
        # Show save location if available
        if save_path:
            st.caption(f"💾 Results saved to: {save_path}")
        
        return signals


async def _execute_analysis(
    system: AlgoTradingSystem,
    tickers: List[str],
    progress_placeholder,
    status_placeholder
) -> List[Any]:
    """
    Execute the main analysis pipeline.
    
    Args:
        system: AlgoTradingSystem instance
        tickers: List of tickers
        progress_placeholder: Streamlit placeholder for progress bar
        status_placeholder: Streamlit placeholder for status messages
    
    Returns:
        List of TradingSignal objects
    """
    # Fetch news
    status_placeholder.info("📰 Scraping news from multiple sources...")
    all_news = await system.news_aggregator.fetch_news_for_tickers(tickers)
    progress_placeholder.progress(20)
    
    if not all_news:
        status_placeholder.warning("⚠️ No news found")
        return []
    
    status_placeholder.success(f"✓ Collected {len(all_news)} news items")
    
    # Analyze sentiment
    status_placeholder.info("🧠 Analyzing sentiment...")
    analyzed_news = system.sentiment_analyzer.analyze_news_items(all_news)
    progress_placeholder.progress(40)
    status_placeholder.success(f"✓ Analyzed {len(analyzed_news)} items")
    
    # Calculate metrics and generate signals
    status_placeholder.info("📊 Calculating metrics and generating signals...")
    signals = []
    
    for i, news_item in enumerate(analyzed_news):
        metrics = system.metrics_calculator.get_stock_metrics(news_item.ticker)
        signal = system.decision_engine.generate_signal(news_item, metrics)
        signals.append(signal)
        
        # Update progress
        progress = 40 + int((i + 1) / len(analyzed_news) * 50)
        progress_placeholder.progress(min(progress, 90))
    
    status_placeholder.success(f"✓ Generated {len(signals)} trading signals")
    
    return signals


def _save_results(system: AlgoTradingSystem, signals: List[Any], tickers: List[str]) -> Optional[str]:
    """
    Save analysis results to file.
    
    Args:
        system: AlgoTradingSystem instance
        signals: List of TradingSignal objects
        tickers: List of tickers analyzed
    
    Returns:
        Path where results were saved, or None
    """
    try:
        return system.storage_manager.save_signals(signals, append=Config.APPEND_MODE)
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None


def _save_to_database(signals: List[Any], tickers: List[str]) -> bool:
    """
    Save analysis results to database if available.
    
    Args:
        signals: List of TradingSignal objects
        tickers: List of tickers analyzed
    
    Returns:
        True if saved successfully, False otherwise
    """
    if not DB_AVAILABLE or not get_database_service:
        return False
    
    try:
        db_service = get_database_service()
        
        # Prepare data for database
        signal_data, news_data, fundamental_data = _prepare_database_data(signals)
        
        # Save to database
        run_id, counts = db_service.save_complete_analysis(
            tickers=tickers,
            signals=signal_data,
            news_items=news_data,
            fundamental_metrics=fundamental_data,
            parameters={'analysis_type': 'stock_analysis'},
            run_type='stock_analysis'
        )
        
        if run_id:
            logger.info(f"Saved to database: {counts}")
            return True
        
    except Exception as e:
        logger.error(f"Database save failed: {e}")
    
    return False


def _prepare_database_data(signals: List[Any]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Prepare signal data for database storage.
    
    Args:
        signals: List of TradingSignal objects
    
    Returns:
        Tuple of (signal_data, news_data, fundamental_data)
    """
    signal_data = []
    news_data = []
    fundamental_data = []
    seen_tickers = set()
    
    for signal in signals:
        # Extract signal info
        signal_data.append({
            'ticker': signal.news_item.ticker,
            'decision': signal.decision.value,
            'confidence': abs(signal.decision_score) * 100,
            'strategy_name': 'ensemble_analysis',
            'reasons': signal.reasoning,
            'price_target': (
                getattr(signal.metrics, 'current_price', None)
                if signal.metrics else None
            ),
            'current_price': (
                getattr(signal.metrics, 'current_price', None)
                if signal.metrics else None
            ),
            'metadata': {
                'sentiment_score': signal.news_item.sentiment_score,
                'sentiment_label': (
                    signal.news_item.sentiment_label.value
                    if signal.news_item.sentiment_label else None
                ),
            }
        })
        
        # Extract news info
        news_data.append({
            'ticker': signal.news_item.ticker,
            'headline': signal.news_item.title,
            'summary': signal.news_item.summary or '',
            'source': signal.news_item.source,
            'url': signal.news_item.url or '',
            'published_at': signal.news_item.timestamp,
            'sentiment_score': signal.news_item.sentiment_score,
            'sentiment_label': (
                signal.news_item.sentiment_label.value
                if signal.news_item.sentiment_label else None
            ),
        })
        
        # Extract fundamental metrics (once per ticker)
        ticker = signal.news_item.ticker
        if ticker not in seen_tickers and signal.metrics:
            seen_tickers.add(ticker)
            fundamental_data.append(
                _extract_fundamental_metrics(ticker, signal.metrics)
            )
    
    return signal_data, news_data, fundamental_data


def _extract_fundamental_metrics(ticker: str, metrics: Any) -> Dict:
    """
    Extract fundamental metrics for database storage.
    
    Args:
        ticker: Stock ticker symbol
        metrics: StockMetrics object
    
    Returns:
        Dictionary of fundamental metrics
    """
    m = metrics
    
    # Calculate health score from fundamental scores
    health_score = None
    score_count = 0
    score_sum = 0
    
    if m.altman_z_score is not None:
        # Normalize Z-score: >2.99 = 100, <1.81 = 0
        z_normalized = min(100, max(0, (m.altman_z_score - 1.81) / (2.99 - 1.81) * 100))
        score_sum += z_normalized
        score_count += 1
    
    if m.piotroski_f_score is not None:
        # F-score is 0-9, normalize to 0-100
        f_normalized = (m.piotroski_f_score / 9) * 100
        score_sum += f_normalized
        score_count += 1
    
    if m.beneish_m_score is not None:
        # M-score: <-2.22 is good (100), >-2.22 is bad (0)
        m_normalized = 100 if m.beneish_m_score < -2.22 else 0
        score_sum += m_normalized
        score_count += 1
    
    if score_count > 0:
        health_score = score_sum / score_count
    
    return {
        'ticker': ticker,
        'pe_ratio': getattr(m, 'pe_ratio', None),
        'pb_ratio': getattr(m, 'pb_ratio', None),
        'market_cap': getattr(m, 'market_cap', None),
        'health_score': health_score,
        'data_source': 'yahoo_finance'
    }
