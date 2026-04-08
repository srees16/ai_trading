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
        margin-top: 0rem;
        margin-bottom: 0rem;
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
        /* ── Compact header→ribbon→VIX→nav stack ────────── */
        .header-bar { margin-top: 0 !important; margin-bottom: 0 !important; }
        .vix-bar    { margin-top: 0 !important; margin-bottom: 0 !important; }
        .ribbon-wrap{ margin-top: 0 !important; margin-bottom: 0 !important; }

        /* Reduce Streamlit top padding on the main block container */
        [data-testid="stMainBlockContainer"] {
            padding-top: 0.5rem !important;
        }

        /* Shrink the Streamlit vertical-block gap that spaces all
           children apart.  Default is ~1rem; we want near-zero for
           the header/ribbon/VIX/nav stack. */
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
            gap: 0.25rem !important;
        }

        /* Collapse Streamlit element wrappers around header / VIX / ribbon / nav */
        [data-testid="stElementContainer"]:has(.header-bar),
        [data-testid="stElementContainer"]:has(.vix-bar),
        [data-testid="stElementContainer"]:has(.ribbon-wrap),
        [data-testid="stElementContainer"]:has(.ribbon-vix-stack),
        [data-testid="stElementContainer"]:has(.centurion-nav) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        /* Remove extra space Streamlit inserts between stacked markdown blocks */
        [data-testid="stMarkdown"]:has(.header-bar),
        [data-testid="stMarkdown"]:has(.ribbon-wrap),
        [data-testid="stMarkdown"]:has(.ribbon-vix-stack),
        [data-testid="stMarkdown"]:has(.vix-bar),
        [data-testid="stMarkdown"]:has(.centurion-nav) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        /* Collapse Streamlit column wrappers inside nav row */
        [data-testid="stColumns"]:has(button[kind="secondary"]) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            gap: 0.25rem !important;
        }

        /* Zero margins on all header-stack element containers */
        [data-testid="stElementContainer"]:has(.header-bar),
        [data-testid="stElementContainer"]:has(.ribbon-wrap),
        [data-testid="stElementContainer"]:has(.ribbon-vix-stack),
        [data-testid="stElementContainer"]:has(.vix-bar),
        [data-testid="stElementContainer"]:has(.centurion-nav) {
            margin-bottom: 0 !important;
        }

        /* Compact thin separator after nav (replaces bulky st.markdown('---')) */
        .nav-sep {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 0.3rem 0 0.4rem 0;
        }
        [data-testid="stElementContainer"]:has(.nav-sep) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        /* Navigation button row: compact, no overflow — scoped to nav class */
        .centurion-nav button[kind="secondary"] {
            font-size: 0.78rem !important;
            padding: 0.28rem 0.3rem !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            min-width: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            line-height: 1.3 !important;
        }
        /* Active / current-page nav button */
        .centurion-nav button[kind="secondary"]:disabled {
            opacity: 1 !important;
            background-color: #0068d6 !important;
            color: #ffffff !important;
            border-color: #0068d6 !important;
            cursor: default !important;
            font-weight: 600 !important;
        }
        /* On narrow screens allow text to wrap instead of truncate */
        @media (max-width: 900px) {
            .centurion-nav button[kind="secondary"] {
                white-space: normal !important;
                font-size: 0.72rem !important;
                line-height: 1.2 !important;
                padding: 0.2rem 0.2rem !important;
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

    Args:
        current_page: Current page identifier
            ('main', 'fundamental', 'backtesting', 'history')
        back_key_suffix: Suffix for button keys to avoid duplicates
    """
    _inject_nav_button_css()
    # All possible navigation targets (id, label)
    all_pages = [
        ('main',         'Main'),
        ('fundamental',  'Fundamentals'),
        ('backtesting',  'Backtest'),
        ('verdict',      'Verdict'),
        ('us_holdings',  'Holdings'),
        ('history',      'History'),
    ]

    n = len(all_pages)
    if n == 0:
        return

    col_spec = [0.3] + [1] * n + [0.3]
    st.markdown('<div class="centurion-nav">', unsafe_allow_html=True)
    cols = st.columns(col_spec, gap="small")

    for i, (page_id, label) in enumerate(all_pages):
        is_active = page_id == current_page
        with cols[i + 1]:
            if st.button(
                label,
                key=f"nav_{page_id}_{back_key_suffix}",
                width="stretch",
                disabled=is_active,
            ):
                logger.info("[user=%s] Navigation: %s -> %s",
                            st.session_state.get('username', 'unknown'),
                            current_page, page_id)
                st.session_state.current_page = page_id
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def _inject_ind_compact_dropdown_css():
    """Shrink selectbox / multiselect widgets so they fit their text content."""
    st.markdown(
        """<style>
        /* ── Compact dropdowns for IND Stocks module ────────── */
        [data-testid="stSelectbox"] {
            max-width: 260px !important;
        }
        [data-testid="stSelectbox"] > div > div {
            min-height: 0 !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
            font-size: 0.85rem !important;
            line-height: 1.4 !important;
        }
        [data-testid="stSelectbox"] label {
            font-size: 0.82rem !important;
            margin-bottom: 0 !important;
        }
        [data-testid="stMultiSelect"] {
            max-width: 320px !important;
        }
        [data-testid="stMultiSelect"] > div > div {
            min-height: 0 !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
            font-size: 0.85rem !important;
            line-height: 1.4 !important;
        }
        [data-testid="stMultiSelect"] label {
            font-size: 0.82rem !important;
            margin-bottom: 0 !important;
        }
        /* Dropdown list items */
        [data-baseweb="select"] [role="option"] {
            font-size: 0.84rem !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
        }
        /* Selected tag chips in multiselect */
        [data-baseweb="tag"] {
            font-size: 0.78rem !important;
            padding: 0.1rem 0.3rem !important;
            margin: 0.1rem !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )


def render_ind_navigation_buttons(
    current_page: str = "main",
    back_key_suffix: str = "",
    **_kwargs,
):
    """Render navigation buttons for the Indian Stocks module.

    Shows a button for every Ind Stocks sub-page except the current one.

    Args:
        current_page: Current page identifier
        back_key_suffix: Suffix for button keys to avoid duplicates
    """
    _inject_nav_button_css()
    _inject_ind_compact_dropdown_css()
    all_pages = [
        ('main',         'Main'),
        ('fundamental',  'Fundamentals'),
        ('backtesting',  'Backtest'),
        ('screener',     'Screener'),
        ('paper_dashboard', 'Paper Trade'),
        ('ind_kite',     'Fly Kite'),
        ('options',      'Options'),
        ('history',      'History'),
    ]

    n = len(all_pages)
    if n == 0:
        return

    col_spec = [0.3] + [1] * n + [0.3]
    st.markdown('<div class="centurion-nav">', unsafe_allow_html=True)
    cols = st.columns(col_spec, gap="small")

    for i, (page_id, label) in enumerate(all_pages):
        is_active = page_id == current_page
        with cols[i + 1]:
            if st.button(
                label,
                key=f"ind_nav_{page_id}_{back_key_suffix}",
                width="stretch",
                disabled=is_active,
            ):
                logger.info("[user=%s] Ind Navigation: %s -> %s",
                            st.session_state.get('username', 'unknown'),
                            current_page, page_id)
                st.session_state.current_page = page_id
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


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
            " STRONG BUY",
            decision_counts.get('STRONG_BUY', 0),
            help="Stocks with strong buy signals"
        )
    
    with col2:
        st.metric(
            " BUY",
            decision_counts.get('BUY', 0),
            help="Stocks with buy signals"
        )
    
    with col3:
        st.metric(
            " HOLD",
            decision_counts.get('HOLD', 0),
            help="Stocks to hold"
        )
    
    with col4:
        st.metric(
            " SELL",
            decision_counts.get('SELL', 0),
            help="Stocks with sell signals"
        )
    
    with col5:
        st.metric(
            " STRONG SELL",
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
        st.warning(" No analysis data available.")
        
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


# ── NIFTY 50 constituents (yfinance tickers) ─────────────────────
_IND_NIFTY50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
    "BEL.NS", "BHARTIARTL.NS", "BPCL.NS", "BRITANNIA.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "ETERNAL.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS",
    "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
    "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS",
    "TMPV.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]

_US_TOP10 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "TSLA", "JPM", "V",
]

_RIBBON_TTL_MARKET_OPEN = 60    # refresh every 60s during market hours
_RIBBON_TTL_MARKET_CLOSED = 300 # refresh every 5min when market is closed

# NSE Trading Holidays – update annually from
# https://www.nseindia.com/resources/exchange-communication-holidays
_NSE_HOLIDAYS: set = {
    # 2025
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
    "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    # 2026
    "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
    "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24",
    "2026-12-25",
}


def _is_nse_market_open() -> bool:
    """Return True if NSE is currently open (Mon-Fri, 9:15-15:30 IST, excl. holidays)."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    if now.strftime("%Y-%m-%d") in _NSE_HOLIDAYS:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def _fetch_ribbon_prices(market: str = "IND") -> list:
    """Fetch latest prices for NIFTY 50 / US Top-10 (adaptive cache TTL)."""
    import time
    cache_key = f"_ribbon_prices_{market}"
    ts_key = f"_ribbon_ts_{market}"
    now = time.time()
    cached_ts = st.session_state.get(ts_key, 0)
    ttl = _RIBBON_TTL_MARKET_OPEN if (market == "IND" and _is_nse_market_open()) else _RIBBON_TTL_MARKET_CLOSED
    if (now - cached_ts) < ttl and cache_key in st.session_state:
        return st.session_state[cache_key]

    import yfinance as yf
    import logging as _logging
    # Suppress yfinance error noise for bulk ribbon fetch
    _yf_logger = _logging.getLogger("yfinance")
    _prev_level = _yf_logger.level
    _yf_logger.setLevel(_logging.CRITICAL)
    tickers = _IND_NIFTY50 if market == "IND" else _US_TOP10
    currency = "₹" if market == "IND" else "$"
    items = []
    try:
        if market == "IND":
            # Use Bhavcopy-backed batch download for Indian stocks
            from utils import download_ind_ohlcv_batch
            plain_syms = [s.replace(".NS", "") for s in tickers]
            ohlcv = download_ind_ohlcv_batch(plain_syms, period="5d")
            for sym in plain_syms:
                try:
                    df = ohlcv.get(sym)
                    if df is None or df.empty:
                        continue
                    closes = df["Close"].dropna()
                    if closes.empty:
                        continue
                    price = float(closes.iloc[-1])
                    chg_pct = None
                    if len(closes) >= 2:
                        prev = float(closes.iloc[-2])
                        if prev:
                            chg_pct = (price - prev) / prev * 100
                    items.append((sym, price, chg_pct, currency))
                except Exception:
                    continue
        else:
            import yfinance as yf
            data = yf.download(
                tickers, period="2d", progress=False, threads=False, group_by="ticker",
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
                    display = sym
                    items.append((display, price, chg_pct, currency))
                except Exception:
                    continue
    except Exception as exc:
        logger.warning("Ribbon price fetch failed (%s): %s", market, exc)
    finally:
        _yf_logger.setLevel(_prev_level)

    st.session_state[cache_key] = items
    st.session_state[ts_key] = now
    return items


def _build_ribbon_html(market: str = "IND") -> str:
    """Build the scrolling ribbon HTML+CSS string (no st.markdown call).

    Returns empty string if no price data available.
    """
    items = _fetch_ribbon_prices(market)
    if not items:
        return ""

    # Build ticker spans (duplicate for seamless loop)
    spans = []
    for name, price, chg, currency in items:
        if chg is not None:
            arrow = "" if chg >= 0 else ""
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
    full_html = f"{ticker_html}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ticker_html}"

    return f"""
    <style>
        .ribbon-wrap {{
            overflow: hidden;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
            border-top: 1px solid #e2e8f0;
            padding: 0.45rem 0;
            border-radius: 6px;
        }}
        .ribbon-track {{
            display: inline-block;
            white-space: nowrap;
            animation: ribbonScroll 90s linear infinite;
            line-height: 1.5;
        }}
        .ribbon-item {{
            display: inline-block;
            margin: 0 1rem;
            font-size: 0.82rem;
            line-height: 1.5;
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
    </div>"""


def render_stock_ticker_ribbon(market: str = "IND"):
    """Render a scrolling ribbon of top-10 stock prices.

    Args:
        market: ``"IND"`` for Indian stocks, ``"US"`` for US stocks.

    .. note:: Prefer :func:`render_ribbon_and_vix` which merges ribbon +
       VIX into a single DOM element to avoid Streamlit wrapper clipping.
    """
    html = _build_ribbon_html(market)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def _build_vix_html(market: str = "US") -> str:
    """Build the VIX indicator bar HTML+CSS string (no st.markdown call).

    Returns empty string if data unavailable.
    """
    from scrapers.macro.macro_indicators import MacroIndicators

    try:
        snap = MacroIndicators().fetch(market=market)
    except Exception:
        return ""

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
        chg_arrow = "" if index_chg >= 0 else ""
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

    return f"""
    <style>
        .vix-bar {{
            background: #ffffff;
            padding: 0.35rem 1.2rem;
            border-radius: 8px;
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
    </div>"""


def render_vix_indicator(market: str = "US"):
    """Render a live VIX indicator bar on the landing page.

    Args:
        market: ``"US"`` for CBOE VIX or ``"IND"`` for India VIX.

    .. note:: Prefer :func:`render_ribbon_and_vix` which merges ribbon +
       VIX into a single DOM element to avoid Streamlit wrapper clipping.
    """
    html = _build_vix_html(market)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def render_ribbon_and_vix(market: str = "IND"):
    """Render ribbon + VIX bar inside a **single** ``st.markdown`` call.

    This avoids the persistent overlap/clipping issue caused by Streamlit
    wrapping each ``st.markdown`` in its own ``stElementContainer`` div
    whose overflow constraints clip the ribbon against the VIX bar.

    Args:
        market: ``"IND"`` or ``"US"``.
    """
    ribbon_html = _build_ribbon_html(market)
    vix_html = _build_vix_html(market)

    if not ribbon_html and not vix_html:
        return

    # Combine into one block with a small gap between them
    combined = f"""
    <div class="ribbon-vix-stack" style="display:flex; flex-direction:column; gap:0;">
        {ribbon_html}
        {vix_html}
    </div>
    """
    st.markdown(combined, unsafe_allow_html=True)


def render_india_fear_greed():
    """Render an India Fear & Greed gauge widget on the IND landing page."""
    import streamlit as st

    try:
        import asyncio
        from scrapers.macro.india_fear_greed import IndiaFearGreedIndex
        from scrapers.macro.macro_indicators import MacroIndicators
        from scrapers.ind_news.fii_dii_flows import FIIDIIFlows

        snap = MacroIndicators().fetch(market="IND")

        loop = asyncio.new_event_loop()
        try:
            flow = loop.run_until_complete(FIIDIIFlows().fetch())
        finally:
            loop.close()

        fg = IndiaFearGreedIndex()
        loop2 = asyncio.new_event_loop()
        try:
            result = loop2.run_until_complete(fg.compute(
                india_vix=snap.india_vix,
                fii_net_crore=flow.fii_net,
                nifty_change_pct=snap.nifty50_change_pct,
            ))
        finally:
            loop2.close()

        score = result.score
        label = result.label or "N/A"
    except Exception:
        score = None
        label = "N/A"

    if score is None:
        return

    # Color based on score
    if score <= 20:
        color = "#dc2626"
    elif score <= 40:
        color = "#ea580c"
    elif score <= 60:
        color = "#ca8a04"
    elif score <= 80:
        color = "#16a34a"
    else:
        color = "#15803d"

    st.markdown(f"""
    <style>
        .fg-gauge {{
            background: #ffffff;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border-left: 4px solid {color};
            box-shadow: 0 1px 4px rgba(0,0,0,0.10);
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }}
        .fg-gauge .fg-score {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {color};
        }}
        .fg-gauge .fg-label {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #374151;
        }}
        .fg-gauge .fg-bar {{
            flex: 1;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            position: relative;
            min-width: 120px;
        }}
        .fg-gauge .fg-bar-fill {{
            height: 100%;
            border-radius: 4px;
            background: {color};
            width: {score:.0f}%;
        }}
    </style>
    <div class="fg-gauge">
        <span class="fg-label">India F&G</span>
        <span class="fg-score">{score:.0f}</span>
        <span style="color:#6b7280; font-size:0.75rem;">{label}</span>
        <div class="fg-bar"><div class="fg-bar-fill"></div></div>
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
