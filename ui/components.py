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
        padding: 0.55rem 1.6rem;
        border-radius: 10px;
        margin-top: 0.4rem;
        margin-bottom: 0.1rem;
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


def _inject_nav_button_css():
    """Inject CSS so navigation buttons shrink text to fit without overflow."""
    st.markdown(
        """<style>
        /* Eliminate Streamlit's implicit element container gaps */
        .vix-bar { margin-bottom: 0 !important; }
        .ribbon-wrap { margin-bottom: 0 !important; }

        /* Collapse Streamlit element wrappers around VIX / ribbon / nav */
        [data-testid="stElementContainer"]:has(.vix-bar),
        [data-testid="stElementContainer"]:has(.ribbon-wrap) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }

        /* Navigation button row: compact, no overflow */
        [data-testid="stHorizontalBlock"] button[kind="secondary"] {
            font-size: 0.78rem !important;
            padding: 0.25rem 0.15rem !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            min-width: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        /* On narrow screens allow text to wrap instead of truncate */
        @media (max-width: 900px) {
            [data-testid="stHorizontalBlock"] button[kind="secondary"] {
                white-space: normal !important;
                font-size: 0.72rem !important;
                line-height: 1.2 !important;
                padding: 0.2rem 0.1rem !important;
            }
        }
        </style>""",
        unsafe_allow_html=True,
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
    _inject_nav_button_css()
    has_results = (
        st.session_state.get('analysis_complete', False)
        and st.session_state.get('signals')
    )

    # All possible navigation targets (id, label)
    all_pages = [
        ('main',         'Main'),
        ('analysis',     'Stock Analysis'),
        ('fundamental',  'Fundamental Analysis'),
        ('backtesting',  'Backtest Strategy'),
        ('history',      'History'),
        ('us_holdings',  'Holdings'),
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


def render_ind_navigation_buttons(
    current_page: str = "main",
    back_key_suffix: str = "",
    **_kwargs,
):
    """Render navigation buttons for the Indian Stocks module.

    Shows a button for every Ind Stocks sub-page except the current one.
    The Stock Analysis button only appears when results are available.

    Args:
        current_page: Current page identifier
        back_key_suffix: Suffix for button keys to avoid duplicates
    """
    _inject_nav_button_css()
    has_results = (
        st.session_state.get('analysis_complete', False)
        and st.session_state.get('signals')
    )

    all_pages = [
        ('main',         'Main'),
        ('analysis',     'Analysis'),
        ('fundamental',  'Fundamentals'),
        ('backtesting',  'Backtest'),
        ('history',      'History'),
        ('options',      'Options'),
        ('ind_kite',     'Fly Kite'),
    ]

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
                key=f"ind_nav_{page_id}_{back_key_suffix}",
                use_container_width=True,
            ):
                logger.info("[user=%s] Ind Navigation: %s -> %s",
                            st.session_state.get('username', 'unknown'),
                            current_page, page_id)
                st.session_state.current_page = page_id
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
            "⚖️ HOLD",
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
            "🔴 STRONG SELL",
            decision_counts.get('STRONG_SELL', 0),
            help="Stocks with strong sell signals"
        )


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


# ── Top 10 stocks by market cap ──────────────────────────────────
_IND_TOP10 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
]

_US_TOP10 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "TSLA", "JPM", "V",
]

_RIBBON_TTL = 300  # seconds


def _fetch_ribbon_prices(market: str = "IND") -> list:
    """Fetch latest prices for top-10 stocks (cached 5 min per market)."""
    import time
    cache_key = f"_ribbon_prices_{market}"
    ts_key = f"_ribbon_ts_{market}"
    now = time.time()
    cached_ts = st.session_state.get(ts_key, 0)
    if (now - cached_ts) < _RIBBON_TTL and cache_key in st.session_state:
        return st.session_state[cache_key]

    import yfinance as yf
    tickers = _IND_TOP10 if market == "IND" else _US_TOP10
    currency = "₹" if market == "IND" else "$"
    items = []
    try:
        data = yf.download(
            tickers, period="2d", progress=False, threads=True, group_by="ticker",
        )
        for sym in tickers:
            try:
                col = data[sym] if sym in data.columns.get_level_values(0) else None
                if col is None or col.empty:
                    continue
                closes = col["Close"].dropna()
                if closes.empty:
                    continue
                price = float(closes.iloc[-1])
                chg_pct = None
                if len(closes) >= 2:
                    prev = float(closes.iloc[-2])
                    if prev:
                        chg_pct = (price - prev) / prev * 100
                display = sym.replace(".NS", "")
                items.append((display, price, chg_pct, currency))
            except Exception:
                continue
    except Exception as exc:
        logger.warning("Ribbon price fetch failed (%s): %s", market, exc)

    st.session_state[cache_key] = items
    st.session_state[ts_key] = now
    return items


def render_stock_ticker_ribbon(market: str = "IND"):
    """Render a scrolling ribbon of top-10 stock prices.

    Args:
        market: ``"IND"`` for Indian stocks, ``"US"`` for US stocks.
    """
    items = _fetch_ribbon_prices(market)
    if not items:
        return

    # Build ticker spans (duplicate for seamless loop)
    spans = []
    for name, price, chg, currency in items:
        if chg is not None:
            arrow = "▲" if chg >= 0 else "▼"
            chg_color = "#16a34a" if chg >= 0 else "#dc2626"
            chg_str = f'<span style="color:{chg_color}; font-weight:600;">{arrow}&nbsp;{chg:+.2f}%</span>'
        else:
            chg_str = ""
        spans.append(
            f'<span class="ribbon-item">'
            f'<span class="ribbon-sym">{name}</span>'
            f'<span class="ribbon-price">{currency}{price:,.1f}</span>'
            f'{chg_str}'
            f'</span>'
        )

    ticker_html = "&nbsp;&nbsp;&nbsp;".join(spans)
    # Duplicate content so the scroll loops seamlessly
    full_html = f"{ticker_html}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ticker_html}"

    st.markdown(f"""
    <style>
        .ribbon-wrap {{
            overflow: hidden;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
            border-top: 1px solid #e2e8f0;
            padding: 0.2rem 0;
            margin-bottom: 0.05rem;
            border-radius: 6px;
        }}
        .ribbon-track {{
            display: inline-block;
            white-space: nowrap;
            animation: ribbonScroll 30s linear infinite;
        }}
        .ribbon-item {{
            display: inline-block;
            margin: 0 1rem;
            font-size: 0.82rem;
        }}
        .ribbon-sym {{
            color: #1e293b;
            font-weight: 700;
            margin-right: 0.3rem;
        }}
        .ribbon-price {{
            color: #334155;
            font-weight: 500;
            margin-right: 0.25rem;
        }}
        @keyframes ribbonScroll {{
            0%   {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
    </style>
    <div class="ribbon-wrap">
        <div class="ribbon-track">
            {full_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_vix_indicator(market: str = "US"):
    """Render a live VIX indicator bar on the landing page.

    Args:
        market: ``"US"`` for CBOE VIX or ``"IND"`` for India VIX.
    """
    from scrapers.macro.macro_indicators import MacroIndicators

    try:
        snap = MacroIndicators().fetch(market=market)
    except Exception:
        return  # silently skip if data unavailable

    if market == "IND":
        vix_val = snap.india_vix
        vix_label = "India VIX"
        index_label = "Nifty 50"
        index_val = snap.nifty50_price
        index_chg = snap.nifty50_change_pct
    else:
        vix_val = snap.vix
        vix_label = "CBOE VIX"
        index_label = "S&P 500"
        index_val = snap.sp500_price
        index_chg = snap.sp500_change_pct

    # VIX color: green < 15, yellow 15-20, orange 20-25, red > 25
    if vix_val is None:
        vix_color = "#6b7280"
        vix_display = "N/A"
    elif vix_val < 15:
        vix_color = "#16a34a"
        vix_display = f"{vix_val:.1f}"
    elif vix_val < 20:
        vix_color = "#ca8a04"
        vix_display = f"{vix_val:.1f}"
    elif vix_val < 25:
        vix_color = "#ea580c"
        vix_display = f"{vix_val:.1f}"
    else:
        vix_color = "#dc2626"
        vix_display = f"{vix_val:.1f}"

    # Sentiment pill
    sent_label = snap.macro_sentiment_label or "n/a"
    sent_score = snap.macro_sentiment_score or 0
    if sent_label == "greedy":
        pill_bg, pill_fg = "#16a34a", "#fff"
    elif sent_label == "fearful":
        pill_bg, pill_fg = "#dc2626", "#fff"
    else:
        pill_bg, pill_fg = "#ca8a04", "#fff"

    # Index change arrow
    if index_chg is not None:
        chg_sign = "+" if index_chg >= 0 else ""
        chg_arrow = "▲" if index_chg >= 0 else "▼"
        chg_color = "#16a34a" if index_chg >= 0 else "#dc2626"
        chg_html = (
            f'<span style="color:{chg_color}; font-weight:600;">'
            f'{chg_arrow} {chg_sign}{index_chg:.2f}%</span>'
        )
    else:
        chg_html = ""

    index_html = ""
    if index_val is not None:
        index_html = (
            f'<span style="margin-left:1.5rem;">'
            f'<span style="color:#6b7280; font-size:0.78rem;">{index_label}</span> '
            f'<span style="color:#1f2937; font-weight:700;">{index_val:,.1f}</span> '
            f'{chg_html}</span>'
        )

    # Extra macro pills
    extras = []
    if snap.us_10y_yield is not None:
        extras.append(f'<span style="color:#6b7280; font-size:0.78rem;">10Y</span> '
                      f'<span style="color:#1f2937; font-weight:600;">{snap.us_10y_yield:.2f}%</span>')
    if snap.gold_price is not None:
        extras.append(f'<span style="color:#6b7280; font-size:0.78rem;">Gold</span> '
                      f'<span style="color:#1f2937; font-weight:600;">${snap.gold_price:,.0f}</span>')
    if snap.crude_oil_price is not None:
        extras.append(f'<span style="color:#6b7280; font-size:0.78rem;">Crude</span> '
                      f'<span style="color:#1f2937; font-weight:600;">${snap.crude_oil_price:.1f}</span>')
    extras_html = ""
    if extras:
        extras_html = '<span style="margin-left:1.5rem;">' + '&nbsp;&nbsp;|&nbsp;&nbsp;'.join(extras) + '</span>'

    st.markdown(f"""
    <style>
        .vix-bar {{
            background: #ffffff;
            padding: 0.35rem 1.2rem;
            border-radius: 8px;
            margin-bottom: 0rem;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.3rem 0;
            border-left: 4px solid {vix_color};
            box-shadow: 0 1px 4px rgba(0,0,0,0.10);
        }}
        .vix-bar .pill {{
            display: inline-block;
            padding: 0.1rem 0.55rem;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 0.6rem;
        }}
    </style>
    <div class="vix-bar">
        <span style="color:#374151; font-size:0.82rem; font-weight:600;">{vix_label}</span>
        <span style="color:{vix_color}; font-weight:800; font-size:1.1rem; margin-left:0.4rem;">{vix_display}</span>
        <span class="pill" style="background:{pill_bg}; color:{pill_fg};">{sent_label}</span>
        {index_html}
        {extras_html}
    </div>
    """, unsafe_allow_html=True)


def render_score_interpretations_table():
    """Render the score interpretations reference table."""
    st.markdown("""
    <div style="display: flex; justify-content: center;">
    <div>
    
    ### Score Interpretations
    
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
        'STRONG_BUY': '',
        'BUY': '',
        'HOLD': '',
        'SELL': '',
        'STRONG_SELL': '',
    }
    return emoji_map.get(decision, '')
