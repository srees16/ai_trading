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
    render_ind_navigation_buttons,
    render_stock_ticker_ribbon,
    render_vix_indicator,
    spinner_html,
)

logger = logging.getLogger(__name__)


def render_analysis_page():
    """Render the analysis progress and results page."""
    market = st.session_state.get('current_market', 'US')
    market_label = "Indian" if market == 'IND' else "US"
    render_page_header(f"{market_label} Stock Analysis")
    
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
        spinner_slot.markdown(spinner_html("Loading analysis engine"), unsafe_allow_html=True)

        def _on_progress(pct: int, label: str):
            spinner_slot.markdown(
                spinner_html(f"{label} — {pct}%"),
                unsafe_allow_html=True,
            )

        from services.analysis import run_analysis_async   # deferred (heavy)
        spinner_slot.markdown(spinner_html("Starting analysis…"), unsafe_allow_html=True)

        st.session_state.signals = asyncio.run(
            run_analysis_async(
                st.session_state.tickers,
                progress_callback=_on_progress,
                market=st.session_state.get("current_market", "US"),
            )
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

    # Carver US efficiency panel (Before vs After)
    market = st.session_state.get('current_market', 'US')
    if market == 'US':
        _render_us_carver_efficiency_section()
    
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
    market = st.session_state.get('current_market', 'US')
    if market == 'IND':
        render_stock_ticker_ribbon(market="IND")
        render_vix_indicator(market="IND")
        render_ind_navigation_buttons(current_page='analysis', back_key_suffix='from_analysis')
    else:
        render_stock_ticker_ribbon(market="US")
        render_vix_indicator(market="US")
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
        "Overview",
        "Detailed Table",
        "Top Signals",
        "Sentiment Charts"
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
    st.warning("No signals were generated. Please try different tickers.")
    
    if st.button(" Main", key="back_no_signals"):
        logger.info("[user=%s] Clicked 'Main' (no signals fallback)",
                    st.session_state.get('username', 'unknown'))
        st.session_state.current_page = 'main'
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# Carver US Efficiency Report (Before vs After)
# ═══════════════════════════════════════════════════════════════

def _render_us_carver_efficiency_section():
    """Render Carver framework efficiency comparison panel for US stocks."""
    try:
        from config import Config
        if not getattr(Config, "CARVER_US_ENABLED", False):
            return
    except Exception:
        return

    with st.expander("📊 Carver Framework (US) — Before vs After Analysis", expanded=False):
        st.markdown(
            "Compares the legacy sentiment-driven sizing with the Carver Systematic "
            "Trading framework (EWMAC + FDM + vol-targeted sizing) for US equities."
        )

        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("**BEFORE (Legacy)**")
            st.markdown("""
            - Signal: News sentiment → BUY/SELL/HOLD
            - Sizing: Fixed position (no vol-targeting)
            - Stop-loss: None (manual)
            - Vol target: None
            - Diversification: Ad-hoc
            - Cost mgmt: None
            """)
        with col_a:
            st.markdown("**AFTER (Carver)**")
            st.markdown("""
            - Forecast: EWMAC unified -20/+20 scale
            - Sizing: (f/10) × vol_scalar × w × IDM
            - Stop-loss: 2.5 × daily_vol (adaptive)
            - Vol target: 20% annual, Half-Kelly
            - Diversification: FDM + IDM multipliers
            - Cost mgmt: SR > 3× cost drag (US: 15 bps)
            """)

        if st.button("Run US Calibration Backtest", key="us_carver_calibrate"):
            with st.spinner("Running Carver expanding-window backtest on US stocks …"):
                try:
                    from services.us_carver_pipeline import run_us_carver_backtest

                    report = run_us_carver_backtest()

                    if "error" in report:
                        st.warning(report["error"])
                        return

                    # Display metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sharpe Ratio", f"{report['backtest_sharpe']:.3f}")
                    m2.metric("Sortino Ratio", f"{report['backtest_sortino']:.3f}")
                    m3.metric("Max Drawdown", f"{report['backtest_max_drawdown_pct']:.1f}%")
                    m4.metric("Annual Return", f"{report['backtest_annual_return_pct']:.1f}%")

                    st.markdown(
                        f"**Symbols:** {report['n_symbols']} | "
                        f"**Days:** {report['n_days']} | "
                        f"**Capital:** ${report['initial_capital']:,.0f} USD"
                    )

                    # Calibrated parameters
                    if report.get("ewmac_scalars"):
                        st.markdown("**Calibrated EWMAC Scalars:**")
                        for k, v in sorted(report["ewmac_scalars"].items()):
                            st.text(f"  {k}: {v:.3f}")

                    # Full report text
                    if report.get("report_text"):
                        st.code(report["report_text"], language="text")

                except Exception as exc:
                    st.error(f"US calibration failed: {exc}")
                    logger.exception("US Carver calibration UI error")
