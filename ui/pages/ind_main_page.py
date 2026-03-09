"""
Indian Stocks Main Page Module for Centurion Capital LLC.

Contains the Indian stocks dashboard and control panel with ticker selection.
Mirrors the US Stocks main page but with NSE/BSE defaults.
"""

import logging
from pathlib import Path
from typing import List

import streamlit as st

from config import Config
from ui.components import (
    render_header,
    render_footer,
    render_ind_navigation_buttons,
    render_vix_indicator,
    render_stock_ticker_ribbon,
)
from utils import parse_ticker_csv, validate_tickers, create_sample_csv

logger = logging.getLogger(__name__)

# Popular NSE large-cap tickers (yfinance format with .NS suffix)
IND_DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
]


def render_ind_main_page():
    """Render the Indian Stocks main application page."""
    logger.info("[user=%s] Viewing Ind Stocks main page",
                st.session_state.get('username', 'unknown'))
    render_header()

    # Scrolling ribbon — top 10 Indian stocks by market cap
    render_stock_ticker_ribbon(market="IND")

    # VIX indicator bar
    render_vix_indicator(market="IND")

    # Navigation buttons
    render_ind_navigation_buttons(
        current_page='main',
        back_key_suffix='from_ind_main',
    )

    # Tighten the gap between nav buttons and control panel
    st.markdown(
        '<div style="margin-top: -1.5rem;"></div>',
        unsafe_allow_html=True,
    )

    # Render control panel
    _render_control_panel()

    # Footer on main page
    render_footer()


def _render_control_panel():
    """Render the control panel with Indian stock selection and settings."""
    st.markdown(
        """<style>
        [data-testid="stRadio"] { margin-top: -0.5rem; margin-bottom: -0.8rem; }
        [data-testid="stExpander"] { margin-top: -0.6rem; margin-bottom: -0.6rem; }
        [data-testid="stTextArea"] { margin-top: -0.6rem; }
        [data-testid="stFileUploader"] { margin-top: -0.6rem; }
        [data-testid="stSelectbox"] { margin-bottom: -0.8rem; }
        [data-testid="stCheckbox"] { margin-top: -0.5rem; margin-bottom: -0.5rem; }
        [data-testid="stHorizontalBlock"] + [data-testid="stElementContainer"],
        [data-testid="stHorizontalBlock"] + div {
            margin-top: -1.5rem !important;
        }
        [data-testid="stAlert"] { margin-top: -0.5rem !important; margin-bottom: -0.5rem !important; }
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

    # Run Analysis section — full width below the settings
    st.markdown('<div style="margin-top: -3.5rem;"></div>', unsafe_allow_html=True)
    run_clicked = _render_run_controls(tickers)

    if run_clicked and len(tickers) > 0:
        logger.info("[user=%s] Clicked 'Run Analysis' (IND) with %d tickers: %s",
                    st.session_state.get('username', 'unknown'),
                    len(tickers), ', '.join(tickers))
        st.session_state.analysis_complete = False
        st.session_state.signals = []
        st.session_state.progress_messages = []
        st.session_state.analysis_tickers = list(tickers)
        st.session_state.analysis_run_id = st.session_state.get('analysis_run_id', 0) + 1
        st.session_state.current_page = 'analysis'
        st.rerun()


def _render_ticker_selection() -> List[str]:
    """Render ticker selection controls for Indian stocks."""
    st.markdown("**Select Stocks**")

    ticker_mode = st.radio(
        "Input method:",
        ["Default Tickers", "Manual Entry", "Upload CSV"],
        help="Select how you want to specify the Indian stocks to analyze",
        horizontal=True,
        key="ind_ticker_mode",
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
    """Handle default Indian tickers selection."""
    with st.expander("📋 View default tickers (NSE)"):
        display_names = [t.replace('.NS', '') for t in IND_DEFAULT_TICKERS]
        st.write(", ".join(display_names))
        st.caption("Tickers are automatically appended with .NS suffix for NSE data.")
    return IND_DEFAULT_TICKERS


def _handle_manual_entry() -> List[str]:
    """Handle manual Indian ticker entry."""
    ticker_input = st.text_area(
        "Enter NSE tickers (comma-separated):",
        value="RELIANCE, TCS, INFY, HDFCBANK",
        height=80,
        help="Enter NSE stock symbols separated by commas. The .NS suffix is added automatically.",
        key="ind_manual_tickers",
    )
    raw = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    # Auto-append .NS if missing
    tickers = [t if t.endswith('.NS') or t.endswith('.BO') else f"{t}.NS" for t in raw]
    return tickers


def _handle_csv_upload() -> List[str]:
    """Handle CSV file upload for Indian tickers."""
    with st.expander("View CSV format example"):
        sample = "Ticker\nRELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK\n"
        st.code(sample, language="csv")
        st.caption("Tickers in the CSV can be plain NSE symbols — .NS suffix is added automatically.")
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=sample,
            file_name="sample_ind_tickers.csv",
            mime="text/csv",
            key="ind_csv_download",
        )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing Indian stock ticker symbols",
        key="ind_csv_upload",
    )

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue().decode('utf-8')
            parsed_tickers = parse_ticker_csv(file_content)

            if parsed_tickers:
                # Auto-append .NS suffix
                tickers = [
                    t if t.endswith('.NS') or t.endswith('.BO') else f"{t}.NS"
                    for t in parsed_tickers
                ]
                valid_tickers, invalid_tickers = validate_tickers(tickers)
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
    st.markdown("⚙️ **Output Settings**")

    output_format = st.selectbox(
        "Output format:",
        ["Excel (.xlsx)", "CSV (.csv)"],
        help="Choose the output file format",
        key="ind_output_format",
    )

    chk_col1, chk_col2 = st.columns(2)
    with chk_col1:
        use_custom_path = st.checkbox(
            "Use custom save location",
            value=False,
            help="Choose a custom directory to save the output file",
            key="ind_custom_path_chk",
        )
    with chk_col2:
        append_mode = st.checkbox(
            "Append to existing file",
            value=Config.APPEND_MODE,
            help="Append results to existing file instead of overwriting",
            key="ind_append_mode_chk",
        )
        Config.APPEND_MODE = append_mode

    if use_custom_path:
        custom_path = st.text_input(
            "Custom save path:",
            value=str(Path.cwd()),
            help="Enter the full directory path where you want to save the file",
            key="ind_custom_path",
        )
        filename = st.text_input(
            "Filename:",
            value="ind_stock_news",
            help="Enter the filename (without extension)",
            key="ind_filename",
        )
        extension = ".xlsx" if output_format == "Excel (.xlsx)" else ".csv"
        full_path = Path(custom_path) / f"{filename}{extension}"
        Config.OUTPUT_FILE = str(full_path)
        st.caption(f"Save to: `{full_path}`")
    else:
        default_filename = (
            "ind_stock_news.xlsx"
            if output_format == "Excel (.xlsx)"
            else "ind_stock_news.csv"
        )
        Config.OUTPUT_FILE = default_filename
        default_path = Path.cwd() / default_filename
        st.caption(f"Save to: `{default_path}`")


def _render_run_controls(tickers: List[str]) -> bool:
    """Render run analysis controls."""
    if not tickers:
        st.warning("No tickers selected")

    btn_col, _ = st.columns([1, 2])

    with btn_col:
        run_button = st.button(
            "Run Analysis",
            type="primary",
            disabled=len(tickers) == 0,
            key="ind_run_analysis",
        )

    return run_button
