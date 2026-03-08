"""
Fundamental Analysis Page Module for Centurion Capital LLC.

Contains the fundamental metrics analysis page rendering.
"""

import logging
from typing import Any, Dict

import streamlit as st

from ui.components import (
    render_page_header,
    render_footer,
    render_navigation_buttons,
    render_ind_navigation_buttons,
    render_stock_ticker_ribbon,
    render_vix_indicator,
    render_no_data_warning,
    render_score_interpretations_table,
)

logger = logging.getLogger(__name__)

# Lazy-loaded at render time (pandas + plotly ~70 s on slow machines)
render_fundamental_charts = None          # type: ignore[assignment]
render_fundamental_summary_metrics = None  # type: ignore[assignment]
render_fundamental_table = None           # type: ignore[assignment]
_heavy_loaded = False


def _ensure_heavy():
    global _heavy_loaded
    if _heavy_loaded:
        return
    from ui.charts import render_fundamental_charts as _rfc, render_fundamental_summary_metrics as _rfsm
    from ui.tables import render_fundamental_table as _rft
    globals()['render_fundamental_charts'] = _rfc
    globals()['render_fundamental_summary_metrics'] = _rfsm
    globals()['render_fundamental_table'] = _rft
    _heavy_loaded = True


def render_fundamental_page():
    """Render the fundamental analysis page."""
    _ensure_heavy()
    logger.info("[user=%s] Viewing Fundamental Analysis page",
                st.session_state.get('username', 'unknown'))
    market = st.session_state.get('current_market', 'US')
    market_label = "Indian" if market == 'IND' else "US"
    render_page_header(
        f"{market_label} Fundamental Analysis",
        description=" Altman Z-Score • Beneish M-Score • Piotroski F-Score"
    )

    # Navigation buttons
    if market == 'IND':
        render_stock_ticker_ribbon(market="IND")
        render_vix_indicator(market="IND")
        render_ind_navigation_buttons(current_page='fundamental', back_key_suffix='from_fundamental')
    else:
        render_stock_ticker_ribbon(market="US")
        render_vix_indicator(market="US")
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
        st.subheader(" All Stocks Overview")
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
