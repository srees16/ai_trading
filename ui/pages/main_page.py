"""
Main Page Module for Centurion Capital LLC.

Contains the main dashboard and control panel rendering.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, List

import streamlit as st

from config import Config
from ui.components import (
    render_header,
    render_footer,
    render_metrics_cards,
    render_navigation_buttons,
    render_ribbon_and_vix,
    spinner_html,
)
from utils import parse_ticker_csv, validate_tickers, create_sample_csv

logger = logging.getLogger(__name__)


def render_main_page():
    """Render the main application page."""
    logger.info("[user=%s] Viewing US Stocks main page",
                st.session_state.get('username', 'unknown'))
    render_header()

    # Scrolling ribbon + VIX indicator (merged into single DOM element)
    render_ribbon_and_vix(market="US")

    # Navigation buttons
    render_navigation_buttons(
        current_page='main',
        back_key_suffix='from_main',
    )

    # Render control panel
    render_control_panel()

    # Footer on main page
    render_footer()


def render_control_panel():
    """Render the control panel with stock selection and settings."""
    # Tighten vertical gaps inside the two-column control area
    st.markdown(
        """<style>
        /* Reduce whitespace around radio buttons, expanders, and text areas */
        [data-testid="stRadio"] { margin-top: -0.2rem; margin-bottom: -0.3rem; }
        [data-testid="stExpander"] { margin-top: -0.3rem; margin-bottom: -0.3rem; }
        [data-testid="stTextArea"] { margin-top: -0.3rem; }
        [data-testid="stFileUploader"] { margin-top: -0.3rem; }
        [data-testid="stSelectbox"] { margin-bottom: -0.3rem; }
        [data-testid="stCheckbox"] { margin-top: -0.2rem; margin-bottom: -0.2rem; }
        </style>""",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 1])
    
    tickers = []
    
    with col1:
        tickers = _render_ticker_selection()
    
    with col2:
        _render_output_settings()
    
    st.session_state.tickers = tickers
    
    # Run Analysis section — full width below the settings (tighter spacing)
    st.markdown('<div style="margin-top: -1.5rem;"></div>', unsafe_allow_html=True)
    run_clicked = _render_run_controls(tickers)
    
    if run_clicked and len(tickers) > 0:
        logger.info("[user=%s] Clicked 'Run Analysis' with %d tickers: %s",
                    st.session_state.get('username', 'unknown'),
                    len(tickers), ', '.join(tickers))
        st.session_state.analysis_complete = False
        st.session_state.signals = []
        st.session_state.progress_messages = []
        # Snapshot the tickers the user chose for *this* analysis run.
        # This is the authoritative list that backtesting will use,
        # immune to the main-page widgets overwriting session tickers
        # during casual navigation.
        st.session_state.analysis_tickers = list(tickers)
        # Bump the analysis run counter so the backtesting cache knows
        # a fresh analysis was requested and will recompute strategies.
        st.session_state.analysis_run_id = st.session_state.get('analysis_run_id', 0) + 1

        # Run analysis inline
        _run_and_render_analysis(tickers)

    elif (not st.session_state.get('analysis_complete', True)
          and st.session_state.get('analysis_tickers')):
        # Pending analysis triggered from another page (e.g. Screener)
        _run_and_render_analysis(st.session_state.analysis_tickers)

    elif st.session_state.get('analysis_complete') and st.session_state.get('signals'):
        # Re-render previous results on page re-visit
        _render_analysis_results(st.session_state.signals)


def _render_ticker_selection() -> List[str]:
    """
    Render ticker selection controls.
    
    Returns:
        List of selected tickers
    """
    st.markdown("**Select Stocks**")
    
    ticker_mode = st.radio(
        "Input method:",
        ["Default Tickers", "Manual Entry", "Upload CSV"],
        help="Select how you want to specify the stocks to analyze",
        horizontal=True
    )
    
    st.session_state.ticker_mode = ticker_mode
    
    tickers = []
    
    if ticker_mode == "Default Tickers":
        tickers = _handle_default_tickers()
    elif ticker_mode == "Manual Entry":
        tickers = _handle_manual_entry()
    elif ticker_mode == "Upload CSV":
        tickers = _handle_csv_upload()
    
    return tickers


def _handle_default_tickers() -> List[str]:
    """Handle default tickers selection."""
    with st.expander("View default tickers"):
        st.write(", ".join(Config.DEFAULT_TICKERS))
    return Config.DEFAULT_TICKERS


def _handle_manual_entry() -> List[str]:
    """Handle manual ticker entry."""
    ticker_input = st.text_area(
        "Enter tickers (comma-separated):",
        value="GOOGL, TSLA",
        height=80,
        help="Enter stock ticker symbols separated by commas"
    )
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    return tickers


def _handle_csv_upload() -> List[str]:
    """Handle CSV file upload for tickers."""
    with st.expander("View CSV format example"):
        st.code(create_sample_csv(), language="csv")
        st.download_button(
            label=" Download Sample CSV",
            data=create_sample_csv(),
            file_name="sample_tickers.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing stock ticker symbols"
    )
    
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue().decode('utf-8')
            parsed_tickers = parse_ticker_csv(file_content)
            
            if parsed_tickers:
                valid_tickers, invalid_tickers = validate_tickers(parsed_tickers)
                st.success(f"Found {len(valid_tickers)} valid ticker(s)")
                
                if invalid_tickers:
                    st.warning(f"Skipped {len(invalid_tickers)} invalid ticker(s)")
                    with st.expander("View invalid tickers"):
                        st.write(", ".join(invalid_tickers))
                
                with st.expander("View uploaded tickers"):
                    st.write(", ".join(valid_tickers))
                
                return valid_tickers
            else:
                st.error("No valid tickers found in CSV")
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            st.error(f"Error parsing CSV: {e}")
    
    return []


def _render_output_settings():
    """Render output settings controls."""
    st.markdown(" **Output Settings**")
    
    output_format = st.selectbox(
        "Output format:",
        ["Excel (.xlsx)", "CSV (.csv)"],
        help="Choose the output file format"
    )
    
    chk_col1, chk_col2 = st.columns(2)
    with chk_col1:
        use_custom_path = st.checkbox(
            "Use custom save location",
            value=False,
            help="Choose a custom directory to save the output file"
        )
    with chk_col2:
        append_mode = st.checkbox(
            "Append to existing file",
            value=Config.APPEND_MODE,
            help="Append results to existing file instead of overwriting"
        )
        Config.APPEND_MODE = append_mode
    
    if use_custom_path:
        custom_path = st.text_input(
            "Custom save path:",
            value=str(Path.cwd()),
            help="Enter the full directory path where you want to save the file"
        )
        filename = st.text_input(
            "Filename:",
            value="daily_stock_news",
            help="Enter the filename (without extension)"
        )
        extension = ".xlsx" if output_format == "Excel (.xlsx)" else ".csv"
        full_path = Path(custom_path) / f"{filename}{extension}"
        Config.OUTPUT_FILE = str(full_path)
        st.caption(f"Save to: `{full_path}`")
    else:
        default_filename = (
            "daily_stock_news.xlsx"
            if output_format == "Excel (.xlsx)"
            else "daily_stock_news.csv"
        )
        Config.OUTPUT_FILE = default_filename
        default_path = Path.cwd() / default_filename
        st.caption(f"Save to: `{default_path}`")


def _render_run_controls(tickers: List[str]) -> bool:
    """
    Render run analysis controls.
    
    Args:
        tickers: List of selected tickers
    
    Returns:
        True if run button was clicked
    """
    # Status + Run button (left-aligned)
    if not tickers:
        st.warning("No tickers selected")
    
    btn_col, _ = st.columns([1, 2])
    
    with btn_col:
        run_button = st.button(
            "Run Analysis",
            type="primary",
            disabled=len(tickers) == 0
        )
    
    return run_button


def _run_and_render_analysis(tickers: List[str]):
    """Run analysis inline and display results below the button."""
    _user = st.session_state.get('username', 'unknown')
    logger.info("[user=%s] Analysis started for %d tickers: %s",
                _user, len(tickers), ', '.join(tickers))

    spinner_slot = st.empty()
    spinner_slot.markdown(spinner_html("Loading analysis engine"), unsafe_allow_html=True)

    def _on_progress(pct: int, label: str):
        spinner_slot.markdown(
            spinner_html(f"{label} — {pct}%"),
            unsafe_allow_html=True,
        )

    from services.analysis import run_analysis_async  # deferred (heavy)
    spinner_slot.markdown(spinner_html("Starting analysis…"), unsafe_allow_html=True)

    st.session_state.signals = asyncio.run(
        run_analysis_async(
            tickers,
            progress_callback=_on_progress,
            market=st.session_state.get("current_market", "US"),
        )
    )
    st.session_state.analysis_complete = True
    logger.info("[user=%s] Analysis completed — %d signals generated",
                _user, len(st.session_state.signals))
    spinner_slot.empty()
    st.rerun()


def _render_analysis_results(signals: List[Any]):
    """Render analysis results inline on the main page."""
    from ui.charts import render_decision_chart, render_score_distribution
    from ui.tables import render_simple_summary_table, render_signals_table, render_top_signals

    st.markdown("---")
    render_simple_summary_table(signals)
    st.markdown("---")
    render_metrics_cards(signals)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Overview",
        "Detailed Table",
        "Top Signals",
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
