"""
Fundamental Analysis Page Module for Centurion Capital LLC.

Contains the fundamental metrics analysis page rendering.
"""

import streamlit as st
from typing import Dict, Any

from ui.components import (
    render_page_header,
    render_footer,
    render_navigation_buttons,
    render_no_data_warning,
    render_score_interpretations_table,
    render_tickers_being_analyzed,
)
from ui.charts import render_fundamental_charts, render_fundamental_summary_metrics
from ui.tables import render_fundamental_table


def render_fundamental_page():
    """Render the fundamental analysis page."""
    render_page_header(
        "📊 Fundamental Analysis",
        description="Altman Z-Score • Beneish M-Score • Piotroski F-Score"
    )

    # Show stocks being analyzed
    tickers = st.session_state.get('tickers', [])
    if tickers:
        render_tickers_being_analyzed(tickers, st.session_state.get('ticker_mode', 'Default Tickers'))
    
    # Navigation buttons
    render_navigation_buttons(
        current_page='fundamental',
        back_key_suffix='from_fundamental'
    )
    
    st.markdown("---")
    
    signals = st.session_state.get('signals', [])
    
    if not signals:
        render_no_data_warning("fundamental")
        render_footer()
        return
    
    # Score interpretations (centered)
    _, interp_col, _ = st.columns([0.5, 4, 0.5])
    with interp_col:
        render_score_interpretations_table()
    
    st.markdown("---")
    
    # Group by stock to avoid duplicates
    stock_metrics = _extract_stock_metrics(signals)
    
    if not stock_metrics:
        st.info("No fundamental data available for the analyzed stocks.")
        render_footer()
        return
    
    # Display main metrics table (centered)
    _, center_col, _ = st.columns([0.5, 4, 0.5])
    with center_col:
        st.subheader("📋 All Stocks Overview")
        render_fundamental_table(stock_metrics)
    
    st.markdown("---")
    
    # Three charts side by side
    render_fundamental_charts(stock_metrics)
    
    st.markdown("---")
    
    # Summary metrics
    render_fundamental_summary_metrics(stock_metrics)
    
    render_footer()


def _extract_stock_metrics(signals) -> Dict[str, Any]:
    """
    Extract unique stock metrics from signals.
    
    Args:
        signals: List of TradingSignal objects
    
    Returns:
        Dictionary mapping ticker to metrics object
    """
    stock_metrics = {}
    
    for signal in signals:
        ticker = signal.news_item.ticker
        if ticker not in stock_metrics and signal.metrics:
            stock_metrics[ticker] = signal.metrics
    
    return stock_metrics
