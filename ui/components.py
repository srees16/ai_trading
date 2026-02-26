"""
Reusable UI Components for Centurion Capital LLC.

Contains header, footer, navigation, and other reusable UI elements.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)


def load_logo_base64() -> str:
    """
    Load logo image and return base64 encoded HTML.
    
    Returns:
        HTML img tag with embedded base64 logo, or empty string if not found
    """
    _KEY = "_logo_b64_large"
    if _KEY in st.session_state:
        return st.session_state[_KEY]

    logo_path = Path(__file__).parent / "assets" / "centurion_logo.png"
    html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        html = f'<img src="data:image/png;base64,{logo_data}" style="height: 1.6rem; vertical-align: middle; margin-right: 0.2rem;">'
    st.session_state[_KEY] = html
    return html


def load_logo_base64_small() -> str:
    """
    Load logo image with smaller size for page headers.
    
    Returns:
        HTML img tag with embedded base64 logo (smaller), or empty string if not found
    """
    _KEY = "_logo_b64_small"
    if _KEY in st.session_state:
        return st.session_state[_KEY]

    logo_path = Path(__file__).parent / "assets" / "centurion_logo.png"
    html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        html = f'<img src="data:image/png;base64,{logo_data}" style="height: 1.4rem; vertical-align: middle; margin-right: 0.3rem;">'
    st.session_state[_KEY] = html
    return html


_HEADER_BAR_CSS = """
<style>
    .header-bar {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0f3460 100%);
        padding: 0.9rem 1.6rem;
        border-radius: 10px;
        margin-top: 0.6rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-left: 4px solid #4299e1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .header-bar h1 {
        color: #ffffff !important;
        font-size: 1.55rem !important;
        margin: 0 !important;
        font-weight: 800;
        letter-spacing: 0.3px;
        line-height: 1.3 !important;
    }
    .header-bar h1 img {
        filter: brightness(0) invert(1);
    }
    .header-bar .subtitle {
        color: #8b949e !important;
        font-size: 0.72rem !important;
        margin: 0.15rem 0 0 0;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        font-weight: 500;
    }
</style>
"""


def render_header_bar(subtitle: str = "", right_html: str = ""):
    """Render the dark gradient header bar used across all modules.

    Args:
        subtitle: Short uppercase subtitle shown below the company name.
        right_html: Optional HTML placed on the right side of the bar
                    (e.g. status pills).
    """
    logo_html = load_logo_base64_small()
    right_block = f'<div style="text-align:right">{right_html}</div>' if right_html else ""
    subtitle_block = (
        f'<p class="subtitle" style="color: #8b949e !important;">{subtitle}</p>'
        if subtitle else ""
    )

    st.markdown(_HEADER_BAR_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="header-bar">
        <div>
            <h1 style="color: #ffffff !important;">{logo_html} Centurion Capital LLC</h1>
            {subtitle_block}
        </div>
        {right_block}
    </div>
    """, unsafe_allow_html=True)


# ── Reusable spinner HTML helper ─────────────────────────────────

def spinner_html(label: str = "Processing…") -> str:
    """Return an HTML snippet for the unified Centurion spinner.

    The CSS classes (`centurion-spinner`, `spinner-wrapper`,
    `spinner-text`) are defined globally in ``ui/styles.py`` so no
    extra ``<style>`` block is needed.

    Args:
        label: The italic text shown next to the spinning ring.
    """
    return (
        '<div class="spinner-wrapper">'
        '  <div class="centurion-spinner"></div>'
        f'  <span class="spinner-text">{label}</span>'
        '</div>'
    )


def render_header():
    """Render the main application header with the dark header bar."""
    render_header_bar(subtitle="Algorithmic Trading · Event-Driven Alpha")


def render_page_header(title: str, subtitle: Optional[str] = None, description: Optional[str] = None):
    """
    Render a page header with the dark header bar.

    Args:
        title: Main page title (shown as subtitle on the bar)
        subtitle: Optional subtitle text
        description: Optional description text
    """
    # Build subtitle line from title / subtitle / description
    parts = [p for p in [title, subtitle, description] if p]
    bar_subtitle = " · ".join(parts) if parts else ""
    render_header_bar(subtitle=bar_subtitle)


def render_footer():
    """Render the application footer."""
    st.markdown(
        """
        <div class="footer">
            Copyright © 2026 Sreekanth S & Co. Ltd. All rights reserved.<br>
            For reprint rights: <strong>Centurion Capital LLC</strong>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_navigation_buttons(
    current_page: str,
    back_key_suffix: str = "",
    **_kwargs,
):
    """
    Render navigation buttons for all pages.

    Shows a button for every page except the one the user is currently on.
    The Stock Analysis button only appears when results are available.

    Args:
        current_page: Current page identifier
            ('main', 'analysis', 'fundamental', 'backtesting', 'history')
        back_key_suffix: Suffix for button keys to avoid duplicates
    """
    has_results = (
        st.session_state.get('analysis_complete', False)
        and st.session_state.get('signals')
    )

    # All possible navigation targets (id, label)
    all_pages = [
        ('main',         '🏠 Main'),
        ('analysis',     '📈 Stock Analysis'),
        ('fundamental',  '📊 Fundamental Analysis'),
        ('backtesting',  '🔬 Backtest Strategy'),
        ('crypto',       '₿ Crypto'),
        ('history',      '📋 History'),
    ]

    # Build visible buttons: skip current page; skip Analysis if no results
    buttons = [
        (pid, label)
        for pid, label in all_pages
        if pid != current_page
        and (pid != 'analysis' or has_results)
    ]

    n = len(buttons)
    if n == 0:
        return

    col_spec = [0.3] + [1] * n + [0.3]
    cols = st.columns(col_spec, gap="small")

    for i, (page_id, label) in enumerate(buttons):
        with cols[i + 1]:
            if st.button(
                label,
                key=f"nav_{page_id}_{back_key_suffix}",
                use_container_width=True,
            ):
                logger.info("[user=%s] Navigation: %s -> %s",
                            st.session_state.get('username', 'unknown'),
                            current_page, page_id)
                st.session_state.current_page = page_id
                st.rerun()


def render_analysis_navigation_buttons():
    """Render navigation buttons for analysis results page."""
    render_navigation_buttons(current_page='analysis', back_key_suffix='from_analysis')


def render_metrics_cards(signals: List[Any]):
    """
    Render metric cards showing decision counts.
    
    Args:
        signals: List of TradingSignal objects
    """
    if not signals:
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Count decisions
    decision_counts: Dict[str, int] = {}
    for signal in signals:
        decision = signal.decision.value
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    with col1:
        st.metric(
            "🚀 STRONG BUY",
            decision_counts.get('STRONG_BUY', 0),
            help="Stocks with strong buy signals"
        )
    
    with col2:
        st.metric(
            "📈 BUY",
            decision_counts.get('BUY', 0),
            help="Stocks with buy signals"
        )
    
    with col3:
        st.metric(
            "⏸️ HOLD",
            decision_counts.get('HOLD', 0),
            help="Stocks to hold"
        )
    
    with col4:
        st.metric(
            "📉 SELL",
            decision_counts.get('SELL', 0),
            help="Stocks with sell signals"
        )
    
    with col5:
        st.metric(
            "⚠️ STRONG SELL",
            decision_counts.get('STRONG_SELL', 0),
            help="Stocks with strong sell signals"
        )


def render_tickers_being_analyzed(tickers: List[str], ticker_mode: str):
    """
    Display the stocks currently selected for analysis.
    
    Args:
        tickers: List of stock ticker symbols
        ticker_mode: Input mode used (Default Tickers, Manual Entry, Upload CSV)
    """
    if not tickers:
        return
    
    source_icons = {
        "Default Tickers": "📋",
        "Manual Entry": "✏️",
        "Upload CSV": "📁"
    }
    icon = source_icons.get(ticker_mode, "📊")
    
    st.markdown(
        f'<p class="page-description">{icon} Analyzing {len(tickers)} stock(s): {", ".join(tickers)}</p>',
        unsafe_allow_html=True
    )


def render_features_section():
    """Render the features section on the main page."""
    st.markdown("### 🎯 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📰 Multi-Source News**")
        st.caption("Aggregates from Yahoo Finance, Finviz, Investing.com, and more")
    
    with col2:
        st.markdown("**🧠 AI Sentiment Analysis**")
        st.caption("DistilBERT-powered sentiment classification")
    
    with col3:
        st.markdown("**📊 Comprehensive Metrics**")
        st.caption("Fundamentals + Technicals analysis")


def render_how_to_use_section():
    """Render the how to use section on the main page."""
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Select Input Method**: Choose default tickers, enter manually, or upload a CSV
    2. **Configure Settings**: Select output format and append mode
    3. **Run Analysis**: Click the Run Analysis button
    4. **Review Results**: Explore charts, tables, and detailed signal information
    5. **Download Data**: Export results for further analysis
    """)


def render_no_data_warning(page_name: str = "analysis"):
    """
    Render a warning when no data is available.
    
    Args:
        page_name: Name of the page for context-specific messaging
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.warning("⚠️ No analysis data available.")
        
        if page_name == "fundamental":
            st.info("""
            **To view fundamental analysis:**
            1. Click **Back to Main**
            2. Select your stocks to analyze
            3. Click **Run Analysis**
            4. Return to this page to view detailed fundamental metrics
            """)
        elif page_name == "backtesting":
            st.info("""
            **To view backtest strategy:**
            1. Click **Back to Main**
            2. Select your stocks to analyze
            3. Click **Run Analysis**
            4. Return to this page to view backtesting results
            """)
        else:
            st.info("""
            **To view analysis:**
            1. Click **Back to Main**
            2. Select your stocks to analyze
            3. Click **Run Analysis**
            """)


def render_score_interpretations_table():
    """Render the score interpretations reference table."""
    st.markdown("""
    <div style="display: flex; justify-content: center;">
    <div>
    
    ### 📖 Score Interpretations
    
    | Score | What it Measures | Interpretation |
    |-------|------------------|----------------|
    | **Altman Z-Score** | Bankruptcy risk | >2.99 Safe, 1.81-2.99 Grey Zone, <1.81 Distress |
    | **Beneish M-Score** | Earnings manipulation | >-2.22 Likely manipulator, <-2.22 Unlikely |
    | **Piotroski F-Score** | Financial health (0-9) | 8-9 Strong, 5-7 Moderate, 0-4 Weak |
    
    </div>
    </div>
    """, unsafe_allow_html=True)


def get_decision_emoji(decision: str) -> str:
    """
    Get emoji for a decision type.
    
    Args:
        decision: Decision value
    
    Returns:
        Emoji string
    """
    emoji_map = {
        'STRONG_BUY': '🟢🟢',
        'BUY': '🟢',
        'HOLD': '🟡',
        'SELL': '🔴',
        'STRONG_SELL': '🔴🔴',
    }
    return emoji_map.get(decision, '⚪')
