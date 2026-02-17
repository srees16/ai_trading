"""
Reusable UI Components for Centurion Capital LLC.

Contains header, footer, navigation, and other reusable UI elements.
"""

import base64
import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime


def load_logo_base64() -> str:
    """
    Load logo image and return base64 encoded HTML.
    
    Returns:
        HTML img tag with embedded base64 logo, or empty string if not found
    """
    logo_path = Path(__file__).parent.parent / "centurion_logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{logo_data}" style="height: 3rem; vertical-align: middle; margin-right: 0.2rem;">'
    return ""


def load_logo_base64_small() -> str:
    """
    Load logo image with smaller size for page headers.
    
    Returns:
        HTML img tag with embedded base64 logo (smaller), or empty string if not found
    """
    logo_path = Path(__file__).parent.parent / "centurion_logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{logo_data}" style="height: 2.5rem; vertical-align: middle; margin-right: 0.5rem;">'
    return ""


def render_header():
    """Render the main application header with logo."""
    logo_html = load_logo_base64()
    
    st.markdown(
        f'<div class="main-header">{logo_html}Centurion Capital LLC</div>'
        f'<p class="main-subtitle">Enterprise AI Engine for Event-Driven Alpha</p>',
        unsafe_allow_html=True
    )


def render_page_header(title: str, subtitle: Optional[str] = None, description: Optional[str] = None):
    """
    Render a page header with logo.
    
    Args:
        title: Main page title (displayed with page-subtitle class)
        subtitle: Optional subtitle text
        description: Optional description text
    """
    logo_html = load_logo_base64_small()
    
    st.markdown(
        f'<div class="main-header">{logo_html}Centurion Capital LLC</div>',
        unsafe_allow_html=True
    )
    
    if title:
        st.markdown(f'<p class="page-subtitle">{title}</p>', unsafe_allow_html=True)
    
    if subtitle:
        st.markdown(f'<p class="main-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    
    if description:
        st.markdown(f'<p class="page-description">{description}</p>', unsafe_allow_html=True)


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
    show_back: bool = True,
    show_fundamental: bool = True,
    show_backtest: bool = True,
    back_key_suffix: str = ""
):
    """
    Render navigation buttons.
    
    Args:
        current_page: Current page identifier
        show_back: Show back to main button
        show_fundamental: Show fundamental analysis button
        show_backtest: Show backtesting button
        back_key_suffix: Suffix for button keys to avoid duplicates
    """
    _, nav_col1, nav_col2, nav_col3, _ = st.columns([1, 1.2, 0.5, 1.2, 1])
    
    with nav_col1:
        if show_back:
            if st.button("← Back to Main", key=f"back_{back_key_suffix}", use_container_width=True):
                st.session_state.current_page = 'main'
                st.rerun()
    
    with nav_col2:
        pass  # Spacer column
    
    with nav_col3:
        if current_page == 'fundamental' and show_backtest:
            if st.button("🔬 Backtesting", key=f"nav_backtest_{back_key_suffix}", use_container_width=True):
                st.session_state.current_page = 'backtesting'
                st.rerun()
        elif current_page == 'backtesting' and show_fundamental:
            if st.button("📊 Fundamentals", key=f"nav_fundamental_{back_key_suffix}", use_container_width=True):
                st.session_state.current_page = 'fundamental'
                st.rerun()


def render_analysis_navigation_buttons():
    """Render navigation buttons for analysis results page."""
    _, nav_col1, nav_col2, nav_col3, _ = st.columns([0.5, 1.2, 1.2, 1.2, 0.5])
    
    with nav_col1:
        if st.button("← Back to Main", key="back_from_analysis", use_container_width=True):
            st.session_state.current_page = 'main'
            st.rerun()
    
    with nav_col2:
        if st.button("📊 Fundamentals", key="nav_fundamental_from_analysis", use_container_width=True):
            st.session_state.current_page = 'fundamental'
            st.rerun()
    
    with nav_col3:
        if st.button("🔬 Backtesting", key="nav_backtest_from_analysis", use_container_width=True):
            st.session_state.current_page = 'backtesting'
            st.rerun()


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


def render_stock_tags(tickers: List[str]) -> str:
    """
    Render stock tickers as markdown tags.
    
    Args:
        tickers: List of stock ticker symbols
    
    Returns:
        Markdown string with tickers as code tags
    """
    return " ".join([f"`{ticker}`" for ticker in tickers])


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


def render_completion_banner(
    tickers_count: int,
    signals_count: int,
    db_saved: bool = False
):
    """
    Render analysis completion banner.
    
    Args:
        tickers_count: Number of tickers analyzed
        signals_count: Number of signals generated
        db_saved: Whether results were saved to database
    """
    db_badge = ' • 🗄️ Saved to DB' if db_saved else ''
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #00cc44, #00aa33);
            color: white;
            padding: 0.6rem 1rem;
            border-radius: 6px;
            text-align: center;
            margin: 0.5rem 0;
        ">
            <span style="font-weight: 600;">✅ Analysis Complete</span>
            <span style="margin-left: 1rem;">Analyzed <strong>{tickers_count}</strong> stocks • <strong>{signals_count}</strong> signals generated{db_badge}</span>
        </div>
        """,
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


def get_status_indicator(
    value: float,
    thresholds: Dict[str, float],
    labels: Dict[str, str]
) -> tuple:
    """
    Get status indicator based on value and thresholds.
    
    Args:
        value: The value to evaluate
        thresholds: Dictionary with threshold values
        labels: Dictionary with status labels
    
    Returns:
        Tuple of (display_value, status_label)
    """
    if value is None:
        return "N/A", "N/A"
    
    # Example for Z-score: thresholds={'safe': 2.99, 'grey': 1.81}
    # labels={'safe': '🟢 Safe', 'grey': '🟡 Grey Zone', 'distress': '🔴 Distress'}
    if 'safe' in thresholds and value > thresholds['safe']:
        return f"{value:.2f}", labels.get('safe', 'Safe')
    elif 'grey' in thresholds and value > thresholds['grey']:
        return f"{value:.2f}", labels.get('grey', 'Grey')
    else:
        return f"{value:.2f}", labels.get('distress', 'Distress')
