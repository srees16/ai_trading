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
    render_tickers_being_analyzed,
    render_metrics_cards,
    render_analysis_navigation_buttons,
)

logger = logging.getLogger(__name__)

# ── CSS-only multi-colour spinning wheel (renders instantly) ────────
_SPINNER_CSS = """
<style>
@keyframes centurion-spin {
  0%   { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
@keyframes centurion-colors {
  0%   { border-top-color: #4fc3f7; }   /* light-blue  */
  25%  { border-top-color: #ab47bc; }   /* purple      */
  50%  { border-top-color: #66bb6a; }   /* green       */
  75%  { border-top-color: #ffa726; }   /* orange      */
  100% { border-top-color: #4fc3f7; }   /* light-blue  */
}
.centurion-spinner-overlay {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 0 2rem 0;
}
.centurion-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(120,120,120,0.18);
  border-top: 4px solid #4fc3f7;
  border-radius: 50%;
  animation: centurion-spin 0.8s linear infinite,
             centurion-colors 3s ease-in-out infinite;
}
.centurion-spinner-label {
  margin-top: 1.1rem;
  font-size: 1.05rem;
  color: #4a4a5a;
  letter-spacing: 0.03em;
}
</style>
"""

_SPINNER_HTML = """
<div class="centurion-spinner-overlay">
  <div class="centurion-spinner"></div>
  <div class="centurion-spinner-label">Analyzing&hellip;</div>
</div>
"""


def render_analysis_page():
    """Render the analysis progress and results page."""
    render_page_header("📈 US Stock Analysis")
    
    # Show stocks being analyzed
    if st.session_state.tickers:
        tickers = st.session_state.tickers
        ticker_mode = st.session_state.get('ticker_mode', 'Default Tickers')
        render_tickers_being_analyzed(tickers, ticker_mode)
    
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
        spinner_slot.markdown(_SPINNER_CSS + _SPINNER_HTML, unsafe_allow_html=True)

        from services.analysis import run_analysis_async   # deferred (heavy)

        st.session_state.signals = asyncio.run(
            run_analysis_async(st.session_state.tickers)
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
