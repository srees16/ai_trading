"""
Analysis Page Module for Centurion Capital LLC.

Contains the stock analysis results page rendering.
"""

import asyncio
import streamlit as st
import logging
from typing import List, Any

from ui.components import (
    render_page_header,
    render_footer,
    render_tickers_being_analyzed,
    render_metrics_cards,
    render_analysis_navigation_buttons,
)
from ui.charts import render_decision_chart, render_sentiment_chart, render_score_distribution
from ui.tables import render_simple_summary_table, render_signals_table, render_top_signals
from services.analysis import run_analysis_async

logger = logging.getLogger(__name__)


def render_analysis_page():
    """Render the analysis progress and results page."""
    render_page_header("📈 Stock Analysis")
    
    # Show stocks being analyzed
    if st.session_state.tickers:
        tickers = st.session_state.tickers
        ticker_mode = st.session_state.get('ticker_mode', 'Default Tickers')
        render_tickers_being_analyzed(tickers, ticker_mode)
    
    st.markdown("---")
    
    # Run analysis if not complete
    if not st.session_state.analysis_complete:
        st.session_state.signals = asyncio.run(
            run_analysis_async(st.session_state.tickers)
        )
        st.session_state.analysis_complete = True
        st.rerun()
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.signals:
        _render_analysis_results(st.session_state.signals)
    elif st.session_state.analysis_complete and not st.session_state.signals:
        _render_no_signals_warning()
    
    render_footer()


def _render_analysis_results(signals: List[Any]):
    """
    Render analysis results with charts and tables.
    
    Args:
        signals: List of TradingSignal objects
    """
    # Navigation buttons
    render_analysis_navigation_buttons()
    
    st.markdown("---")
    
    # Simple summary table at the top
    render_simple_summary_table(signals)
    
    st.markdown("---")
    
    # Metrics cards
    render_metrics_cards(signals)
    
    st.markdown("---")
    
    # Charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📋 Detailed Table",
        "🔝 Top Signals",
        "📈 Sentiment Charts"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            render_decision_chart(signals)
        with col2:
            render_score_distribution(signals)
    
    with tab2:
        render_signals_table(signals)
    
    with tab3:
        render_top_signals(signals)
    
    with tab4:
        render_sentiment_chart(signals)


def _render_no_signals_warning():
    """Render warning when no signals were generated."""
    st.warning("⚠️ No signals were generated. Please try different tickers.")
    
    if st.button("← Back to Main", key="back_no_signals"):
        st.session_state.current_page = 'main'
        st.rerun()
