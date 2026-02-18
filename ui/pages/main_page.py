"""
Main Page Module for Centurion Capital LLC.

Contains the main dashboard and control panel rendering.
"""

import streamlit as st
from pathlib import Path
from typing import List, Tuple
import logging

from config import Config
from utils import parse_ticker_csv, validate_tickers, create_sample_csv
from ui.components import (
    render_header,
    render_footer,
    render_features_section,
    render_how_to_use_section,
)

logger = logging.getLogger(__name__)


def render_main_page():
    """Render the main application page."""
    render_header()
    
    st.markdown("---")
    
    # Welcome screen - features section
    render_features_section()
    render_how_to_use_section()
    
    st.markdown("---")
    
    # Render control panel
    render_control_panel()
    
    # Footer on main page
    render_footer()


def render_control_panel():
    """Render the control panel with stock selection and settings."""
    st.subheader("📊 Stock Selection & Settings")
    
    col1, col2 = st.columns([1, 1])
    
    tickers = []
    
    with col1:
        tickers = _render_ticker_selection()
    
    with col2:
        _render_output_settings()
    
    st.session_state.tickers = tickers
    
    # Run Analysis section — full width below the settings
    st.markdown("---")
    run_clicked = _render_run_controls(tickers)
    
    if run_clicked and len(tickers) > 0:
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
        st.session_state.current_page = 'analysis'
        st.rerun()


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
    st.info(f"Using {len(Config.DEFAULT_TICKERS)} default tickers")
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
    with st.expander("📄 View CSV format example"):
        st.code(create_sample_csv(), language="csv")
        st.download_button(
            label="⬇️ Download Sample CSV",
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
                st.success(f"✓ Found {len(valid_tickers)} valid ticker(s)")
                
                if invalid_tickers:
                    st.warning(f"⚠️ Skipped {len(invalid_tickers)} invalid ticker(s)")
                    with st.expander("View invalid tickers"):
                        st.write(", ".join(invalid_tickers))
                
                with st.expander("View uploaded tickers"):
                    st.write(", ".join(valid_tickers))
                
                return valid_tickers
            else:
                st.error("❌ No valid tickers found in CSV")
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            st.error(f"❌ Error parsing CSV: {e}")
    
    return []


def _render_output_settings():
    """Render output settings controls."""
    st.markdown("**Output Settings**")
    
    output_format = st.selectbox(
        "Output format:",
        ["Excel (.xlsx)", "CSV (.csv)"],
        help="Choose the output file format"
    )
    
    use_custom_path = st.checkbox(
        "Use custom save location",
        value=False,
        help="Choose a custom directory to save the output file"
    )
    
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
        st.caption(f"📁 Save to: `{full_path}`")
    else:
        default_filename = (
            "daily_stock_news.xlsx"
            if output_format == "Excel (.xlsx)"
            else "daily_stock_news.csv"
        )
        Config.OUTPUT_FILE = default_filename
        default_path = Path.cwd() / default_filename
        st.caption(f"📁 Save to: `{default_path}`")
    
    append_mode = st.checkbox(
        "Append to existing file",
        value=Config.APPEND_MODE,
        help="Append results to existing file instead of overwriting"
    )
    Config.APPEND_MODE = append_mode


def _render_run_controls(tickers: List[str]) -> bool:
    """
    Render run analysis and navigation controls.
    
    Args:
        tickers: List of selected tickers
    
    Returns:
        True if run button was clicked
    """
    # Status + Run button row
    status_col, btn_col = st.columns([1.5, 1])
    
    with status_col:
        if tickers:
            st.success(f"📈 {len(tickers)} stock(s) ready")
        else:
            st.warning("⚠️ No tickers selected")
    
    with btn_col:
        st.write("")  # Align vertically with status
        run_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=len(tickers) == 0
        )
    
    # Navigation buttons — smaller, in a centered row
    has_results = st.session_state.get('analysis_complete', False) and st.session_state.get('signals')
    
    st.markdown(
        """<style>
        div[data-testid="stHorizontalBlock"]:has(> div > div > button[key^="nav_"]) button,
        button[key="nav_fundamental"], button[key="nav_backtest"], button[key="nav_history"], button[key="nav_results"] {
            font-size: 0.85rem !important;
            padding: 0.35rem 0.5rem !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    
    if has_results:
        _, nav_col0, nav_col1, nav_col2, nav_col3, _ = st.columns([0.6, 1, 1, 1, 1, 0.6])
        with nav_col0:
            if st.button("📈 Analysis Results", key="nav_results", use_container_width=True):
                st.session_state.current_page = 'analysis'
                st.rerun()
    else:
        _, nav_col1, nav_col2, nav_col3, _ = st.columns([0.8, 1, 1, 1, 0.8])
    
    with nav_col1:
        if st.button("📊 Fundamental Analysis", key="nav_fundamental", use_container_width=True):
            st.session_state.current_page = 'fundamental'
            st.rerun()
    
    with nav_col2:
        if st.button("🔬 Backtest Strategy", key="nav_backtest", use_container_width=True):
            st.session_state.current_page = 'backtesting'
            st.rerun()
    
    with nav_col3:
        if st.button("📋 History", key="nav_history", use_container_width=True):
            st.session_state.current_page = 'history'
            st.rerun()
    
    return run_button
