"""
Analysis Page Module for Centurion Capital LLC.

Contains the stock analysis results page rendering.
"""

import asyncio
import logging
from typing import Any, List

import streamlit as st

from ui.components import (
    render_page_header,
    render_footer,
    render_metrics_cards,
    render_analysis_navigation_buttons,
    spinner_html,
)

logger = logging.getLogger(__name__)


def render_analysis_page():
    """Render the analysis progress and results page."""
    render_page_header("📈 US Stock Analysis")
    
    st.markdown("---")
    
    # Run analysis if not complete
    if not st.session_state.analysis_complete:
        _user = st.session_state.get('username', 'unknown')
        _tickers = st.session_state.tickers
        logger.info("[user=%s] Analysis started for %d tickers: %s",
                    _user, len(_tickers), ', '.join(_tickers))
        # Show a multi-colour spinning wheel *immediately* —
        # the CSS animation renders in the browser while heavy
        # Python imports and network calls happen server-side.
        spinner_slot = st.empty()
        spinner_slot.markdown(spinner_html("Analyzing… — 0%"), unsafe_allow_html=True)

        def _on_progress(pct: int, label: str):
            spinner_slot.markdown(
                spinner_html(f"Analyzing… — {pct}%"),
                unsafe_allow_html=True,
            )

        from services.analysis import run_analysis_async   # deferred (heavy)

        st.session_state.signals = asyncio.run(
            run_analysis_async(st.session_state.tickers, progress_callback=_on_progress)
        )
        st.session_state.analysis_complete = True
        logger.info("[user=%s] Analysis completed — %d signals generated",
                    st.session_state.get('username', 'unknown'),
                    len(st.session_state.signals))
        spinner_slot.empty()
        st.rerun()
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.signals:
        # Lazy-import chart / table helpers (they pull in pandas + plotly)
        from ui.charts import render_decision_chart, render_sentiment_chart, render_score_distribution
        from ui.tables import render_simple_summary_table, render_signals_table, render_top_signals
        _render_analysis_results(
            st.session_state.signals,
            render_decision_chart, render_sentiment_chart,
            render_score_distribution, render_simple_summary_table,
            render_signals_table, render_top_signals,
        )
    elif st.session_state.analysis_complete and not st.session_state.signals:
        _render_no_signals_warning()
    
    render_footer()


def _render_analysis_results(
    signals: List[Any],
    render_decision_chart,
    render_sentiment_chart,
    render_score_distribution,
    render_simple_summary_table,
    render_signals_table,
    render_top_signals,
):
    """
    Render analysis results with charts and tables.
    
    Args:
        signals: List of TradingSignal objects
        render_*: Lazily-imported rendering callables
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
    
    if st.button("🏠 Main", key="back_no_signals"):
        logger.info("[user=%s] Clicked 'Main' (no signals fallback)",
                    st.session_state.get('username', 'unknown'))
        st.session_state.current_page = 'main'
        st.rerun()
