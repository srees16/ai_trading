"""
Streamlit UI - Index-wise Live Stock Market Data (Real-time via Kite Connect)

Displays stocks grouped by their index (NIFTY50, NIFTYBANK, NIFTYIT, NIFTYENERGY)
with real-time data fetched from Zerodha Kite Connect API.
Columns: Name, High, Low, Volume, LTP, Change (%).
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
import requests
import streamlit as st
import pandas as pd

# Suppress harmless Tornado WebSocket closed errors during st.rerun()
logging.getLogger("tornado.general").setLevel(logging.CRITICAL)

# Add kite_connect folder to path so we can import shared modules
sys.path.insert(0, os.path.dirname(__file__))

from kiteconnect import KiteConnect, exceptions as kite_exceptions
from core.config import REFRESH_INTERVAL
from core.db_service import get_connection
from auth.kite_session import create_kite_session
from trading.order_service import place_order, get_order_book, get_positions, get_holdings, cancel_order
from trading.rsi_strategy import scan_watchlist
from options.option_chain import discover_expiries, fetch_option_chain, INDEX_META


# ── Kite Connect Session ───────────────────────────────────────
@st.cache_resource
def get_kite_session():
    """Login to Kite Connect and return the kite instance."""
    return create_kite_session()


# ── Database ───────────────────────────────────────────────────
def get_db_connection():
    """Get a fresh DB connection (no caching to avoid stale connections)."""
    return get_connection()


def fetch_index_groups(conn):
    """Get all index group names."""
    cur = conn.cursor()
    cur.execute("SELECT id, index_name FROM index_groups ORDER BY id;")
    groups = cur.fetchall()
    cur.close()
    return groups


def fetch_stock_names_by_index(conn, index_group_id):
    """Get stock symbols belonging to an index group."""
    cur = conn.cursor()
    cur.execute(
        "SELECT stock_name FROM index_stocks WHERE index_group_id = %s ORDER BY stock_name;",
        (index_group_id,),
    )
    names = [r[0] for r in cur.fetchall()]
    cur.close()
    return names


def update_stocks_in_db(conn, quotes_data):
    """Update the stocks table with real-time quote data."""
    cur = conn.cursor()
    for symbol, data in quotes_data.items():
        cur.execute("""
            UPDATE stocks
            SET high = %s, low = %s, volume = %s, ltp = %s, change = %s, updated_at = NOW()
            WHERE name = %s;
        """, (
            data.get("high"),
            data.get("low"),
            data.get("volume"),
            data.get("ltp"),
            data.get("change_pct"),
            symbol,
        ))
    conn.commit()
    cur.close()


def fetch_stocks_from_db(conn, index_group_id):
    """Get stocks from DB for an index group (after real-time update)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT s.name, s.high, s.low, s.volume, s.ltp, s.change
        FROM stocks s
        INNER JOIN index_stocks ix ON ix.stock_name = s.name
        WHERE ix.index_group_id = %s
        ORDER BY s.name;
    """, (index_group_id,))
    rows = cur.fetchall()
    cur.close()
    return rows


# ── Real-time Data Fetch ───────────────────────────────────────
def _parse_quote(q):
    """Extract standardised quote dict from a single Kite quote response."""
    ohlc = q.get("ohlc", {})
    last_price = q.get("last_price", 0)
    close = ohlc.get("close", 0)
    change_pct = round(
        ((last_price - close) / close) * 100, 2
    ) if close else 0
    return {
        "high": ohlc.get("high") or last_price,
        "low": ohlc.get("low") or last_price,
        "volume": q.get("volume", 0),
        "ltp": last_price,
        "change_pct": change_pct,
    }


def fetch_realtime_quotes(kite, stock_symbols):
    """
    Fetch real-time quotes from Kite Connect API.
    Returns dict: {symbol: {high, low, volume, ltp, change_pct}}
    """
    if not stock_symbols:
        return {}

    # Build instrument list for Kite API (NSE:SYMBOL format)
    instruments = [f"NSE:{sym}" for sym in stock_symbols]

    # Kite API allows max ~500 instruments per call
    all_quotes = {}
    failed_symbols = []
    batch_size = 200
    for i in range(0, len(instruments), batch_size):
        batch = instruments[i : i + batch_size]
        try:
            quotes = kite.quote(batch)
            for inst_key, q in quotes.items():
                symbol = inst_key.replace("NSE:", "")
                all_quotes[symbol] = _parse_quote(q)
        except kite_exceptions.TokenException:
            # Access token expired mid-session — clear cache so next refresh re-logins
            st.cache_resource.clear()
            st.error("Kite session expired. Click 'Reconnect' to re-login.")
            return all_quotes
        except kite_exceptions.InputException as e:
            # Some symbols may be invalid — try them individually
            st.warning(f"Batch quote failed, trying individually: {e}")
            for inst in batch:
                try:
                    q = kite.quote([inst])
                    for inst_key, data in q.items():
                        symbol = inst_key.replace("NSE:", "")
                        all_quotes[symbol] = _parse_quote(data)
                except Exception:
                    failed_symbols.append(inst.replace("NSE:", ""))
        except Exception as e:
            st.warning(f"Could not fetch quotes: {e}")

    if failed_symbols:
        st.warning(f"⚠️ {len(failed_symbols)} symbols not found: {', '.join(failed_symbols)}")

    return all_quotes


# ── Main App ───────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Live Stocks - India", page_icon="📈", layout="wide")

    # ── Custom CSS for enterprise look ──
    st.markdown("""
    <style>
        /* ── Global compact overrides ── */
        .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
        .stMarkdown, .stText, p, span, label, li { font-size: 0.82rem !important; line-height: 1.3 !important; }
        h1, h2, h3, h4 { margin-top: 0 !important; margin-bottom: 0.2rem !important; }
        /* Shrink emoji sizing globally */
        .stMarkdown img.emoji, .stButton img.emoji { height: 0.9em !important; width: 0.9em !important; }
        /* Reduce vertical gaps between Streamlit elements */
        div[data-testid="stVerticalBlock"] > div { padding-top: 0 !important; padding-bottom: 0 !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] { gap: 0.25rem !important; }
        .element-container { margin-bottom: 0.15rem !important; }
        /* Reduce expander internal padding */
        details[data-testid="stExpander"] summary { padding: 0.3rem 0.6rem !important; font-size: 0.8rem !important; }
        details[data-testid="stExpander"] > div { padding: 0.3rem 0.6rem !important; }

        /* Header bar styling */
        .header-bar {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0f3460 100%);
            padding: 0.9rem 1.6rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-left: 4px solid #4299e1;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }
        .header-bar h1 {
            color: #ffffff;
            font-size: 1.55rem !important;
            margin: 0 !important;
            font-weight: 800;
            letter-spacing: 0.3px;
            line-height: 1.3 !important;
        }
        .header-bar .subtitle {
            color: #8b949e;
            font-size: 0.72rem !important;
            margin: 0.15rem 0 0 0;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            font-weight: 500;
        }
        .header-bar .live-pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            border-radius: 14px;
            padding: 0.2rem 0.65rem;
            font-size: 0.68rem !important;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .header-bar .live-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
        }
        /* Green — market open */
        .pill-open {
            background: rgba(72,187,120,0.12);
            color: #48bb78;
            border: 1px solid rgba(72,187,120,0.35);
        }
        .pill-open .live-dot {
            background: #48bb78;
            animation: blink 1.4s infinite;
        }
        /* Amber — pre-market / post-market */
        .pill-pre {
            background: rgba(236,201,75,0.12);
            color: #ecc94b;
            border: 1px solid rgba(236,201,75,0.35);
        }
        .pill-pre .live-dot {
            background: #ecc94b;
            animation: blink 1.4s infinite;
        }
        /* Grey — market closed */
        .pill-closed {
            background: rgba(160,174,192,0.12);
            color: #a0aec0;
            border: 1px solid rgba(160,174,192,0.35);
        }
        .pill-closed .live-dot {
            background: #a0aec0;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        /* Status badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.72rem !important;
            font-weight: 600;
        }
        .badge-success {
            background: rgba(72,187,120,0.15);
            color: #48bb78;
            border: 1px solid rgba(72,187,120,0.3);
        }
        .badge-info {
            background: rgba(66,153,225,0.15);
            color: #4299e1;
            border: 1px solid rgba(66,153,225,0.3);
        }
        .badge-warn {
            background: rgba(236,201,75,0.15);
            color: #ecc94b;
            border: 1px solid rgba(236,201,75,0.3);
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.35rem 0.6rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        div[data-testid="stMetric"] label {
            color: #718096;
            font-size: 0.65rem !important;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 0.9rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-size: 0.7rem !important;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: #f1f5f9;
            border-radius: 6px;
            padding: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 5px;
            font-weight: 600;
            font-size: 0.78rem !important;
            padding: 0.3rem 0.8rem;
        }
        .stTabs [aria-selected="true"] {
            background: #fff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        }

        /* Button styling */
        .stButton > button {
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.75rem !important;
            padding: 0.25rem 0.7rem;
            transition: all 0.15s;
        }

        /* Move sidebar to RIGHT side of the page */
        [data-testid="stSidebar"] {
            left: auto !important;
            right: 0 !important;
            min-width: 260px !important;
            max-width: 280px !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            padding: 0.25rem 0.6rem !important;
        }
        [data-testid="stSidebar"] h3 {
            color: #e2e8f0 !important;
            font-size: 0.82rem !important;
            margin-bottom: 0 !important;
            margin-top: 0 !important;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #a0aec0 !important;
            font-size: 0.62rem !important;
            margin-bottom: 0 !important;
            line-height: 1.1 !important;
        }
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stNumberInput,
        [data-testid="stSidebar"] .stTextInput {
            margin-bottom: 0 !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            font-size: 0.7rem !important;
            min-height: 1.6rem !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
            padding: 0.1rem 0.35rem !important;
        }
        [data-testid="stSidebar"] .stNumberInput input {
            font-size: 0.7rem !important;
            padding: 0.1rem 0.3rem !important;
            height: 1.6rem !important;
        }
        /* Remove dark stepper bands on number inputs */
        [data-testid="stSidebar"] .stNumberInput [data-baseweb="input"] {
            background: transparent !important;
            border-color: rgba(255,255,255,0.15) !important;
        }
        [data-testid="stSidebar"] .stNumberInput button {
            background: transparent !important;
            border-color: rgba(255,255,255,0.15) !important;
            color: #a0aec0 !important;
            padding: 0 !important;
            min-height: 0.7rem !important;
            height: 0.7rem !important;
        }
        [data-testid="stSidebar"] .stNumberInput button:hover {
            background: rgba(255,255,255,0.08) !important;
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            margin-top: 0.15rem !important;
            font-size: 0.74rem !important;
            padding: 0.3rem 0.5rem !important;
            letter-spacing: 0.4px;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.15rem 0 !important;
            border-color: rgba(255,255,255,0.08) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0.05rem !important;
        }
        [data-testid="stSidebarCollapsedControl"] {
            left: auto !important;
            right: 0.5rem !important;
        }

        /* Center and constrain dataframe tables */
        [data-testid="stDataFrame"] {
            max-width: 900px;
            margin: 0 auto;
        }
        [data-testid="stDataFrame"] table {
            font-size: 0.78rem !important;
        }

        /* Tighter input widgets */
        .stTextInput > div, .stSelectbox > div, .stNumberInput > div {
            margin-bottom: 0.1rem !important;
        }
        .stTextInput input, .stNumberInput input {
            font-size: 0.8rem !important;
            padding: 0.25rem 0.5rem !important;
        }
        .stSelectbox [data-baseweb="select"] {
            font-size: 0.8rem !important;
        }

        /* Control bar: vertically center all widgets in a row */
        .ctrl-row [data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
        .ctrl-row [data-testid="stVerticalBlock"] {
            justify-content: center;
            min-height: 2.2rem;
        }
        .ctrl-row .stMarkdown { margin: 0 !important; }
        .ctrl-row [data-testid="stSlider"] label,
        .ctrl-row [data-testid="stCheckbox"] label { font-size: 0.72rem !important; }
        .ctrl-row [data-testid="stSlider"] { padding-top: 0 !important; max-width: 160px; }
        .ctrl-row [data-testid="stSlider"] [data-baseweb="slider"] { margin-top: 0 !important; padding: 0 !important; max-width: 160px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Market session status (from NSE API) ──
    def _nse_market_status():
        """Fetch Capital Market status from NSE. Returns (pill_class, label)."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }
            sess = requests.Session()
            sess.get("https://www.nseindia.com", headers=headers, timeout=4)
            resp = sess.get(
                "https://www.nseindia.com/api/marketStatus",
                headers=headers, timeout=4,
            )
            resp.raise_for_status()
            for mkt in resp.json().get("marketState", []):
                if mkt.get("market") == "Capital Market":
                    status = (mkt.get("marketStatus") or "").lower()
                    if status in ("open", "live"):
                        return "pill-open", "Live"
                    elif "pre" in status:
                        return "pill-pre", "Pre-Open"
                    elif "close" in status:
                        return "pill-closed", "Closed"
                    else:
                        return "pill-pre", status.title()
        except Exception:
            pass
        return "pill-closed", "Closed"

    pill_class, pill_label = _nse_market_status()

    # ── Connect to Kite & DB ──
    try:
        kite = get_kite_session()
        st.session_state["kite"] = kite          # store for the dialog
        profile = kite.profile()
        kite_status = profile.get("user_id", "Connected")
        kite_connected = True
    except kite_exceptions.TokenException:
        st.cache_resource.clear()
        st.warning("Session expired. Reconnecting...")
        st.rerun()
        return
    except Exception as e:
        kite_status = "Disconnected"
        st.error(f"Kite Connect login failed: {e}")
        st.info("Run `py kite_token_store.py` first to generate a valid request token, then click Reconnect.")
        return

    # ── Header bar (rendered after Kite login so kite_status is available) ──
    st.markdown(f"""
    <div class="header-bar">
        <div>
            <h1>📈 Indian Stock Market</h1>
            <p class="subtitle">Real-time data · Zerodha Kite Connect</p>
        </div>
        <div style="text-align:right">
            <div class="live-pill {pill_class}"><span class="live-dot"></span> {pill_label}</div>
            <div class="live-pill pill-open" style="margin-top:4px"><span class="live-dot"></span> Connected — {kite_status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        conn = get_db_connection()
        groups = fetch_index_groups(conn)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return

    if not groups:
        st.warning("No index groups found. Run setup_livestocks_db.py first.")
        return

    # ── Top control bar: settings pushed left ──
    ctrl_container = st.container()
    with ctrl_container:
        st.markdown('<div class="ctrl-row">', unsafe_allow_html=True)
        c1, c_spacer, c2 = st.columns([1.5, 6, 1])

        with c1:
            refresh_secs = st.select_slider(
                "Refresh interval ⏱",
                options=[10, 15, 20, 30, 45, 60, 90, 120],
                value=REFRESH_INTERVAL,
                key="refresh_secs",
            )

        with c2:
            if st.button("🔄 Reconnect", width="stretch"):
                st.cache_resource.clear()
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Quotes badge (filled by the data fragment) ──
    quotes_badge_slot = st.empty()

    # ── Collect all stock names (used for sidebar dropdown + data fetch) ──
    all_stock_names = set()
    group_stocks_map = {}
    for group_id, group_name in groups:
        names = fetch_stock_names_by_index(conn, group_id)
        group_stocks_map[group_id] = names
        all_stock_names.update(names)
    sorted_stock_list = sorted(all_stock_names)
    conn.close()  # fragment opens its own connection

    # ── Right sidebar: Place Order panel (isolated fragment) ──
    @st.fragment
    def _order_panel():
        st.markdown("### 🛒 Place Order")

        # ── Symbol (full width, searchable) ──
        o_symbol = st.selectbox(
            "Symbol",
            options=sorted_stock_list,
            index=sorted_stock_list.index("RELIANCE") if "RELIANCE" in sorted_stock_list else 0,
            key="order_symbol",
        )

        # ── Exchange | Transaction (side-by-side) ──
        r1a, r1b = st.columns(2)
        o_exchange = r1a.selectbox("Exchange", ["NSE", "BSE"], key="o_exch")
        o_txn      = r1b.selectbox("Side", ["BUY", "SELL"], key="o_txn")

        # ── Order Type | Product (side-by-side) ──
        r2a, r2b = st.columns(2)
        o_type    = r2a.selectbox("Type", ["MARKET", "LIMIT", "SL", "SL-M"], key="o_type")
        o_product = r2b.selectbox("Product", ["CNC", "MIS", "NRML"], key="o_prod")

        # ── Quantity | Validity (side-by-side) ──
        r3a, r3b = st.columns(2)
        o_qty      = r3a.number_input("Qty", min_value=1, value=1, step=1, key="o_qty")
        o_validity = r3b.selectbox("Validity", ["DAY", "IOC"], key="o_val")

        # ── Price | Trigger (side-by-side, always visible) ──
        r4a, r4b = st.columns(2)
        o_price   = r4a.number_input("Price", min_value=0.0, value=0.0, step=0.05, key="o_price",
                                      help="For LIMIT / SL orders")
        o_trigger = r4b.number_input("Trigger", min_value=0.0, value=0.0, step=0.05, key="o_trig",
                                      help="For SL / SL-M orders")

        st.markdown("---")

        # Green button styling
        st.markdown("""
        <style>
            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background-color: #38a169 !important;
                border-color: #38a169 !important;
                color: #fff !important;
            }
            [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
                background-color: #2f855a !important;
                border-color: #2f855a !important;
            }
        </style>
        """, unsafe_allow_html=True)

        btn_label = f"Place {o_txn.capitalize()} Order"
        if st.button(btn_label, width="stretch", type="primary"):
            if o_symbol not in all_stock_names:
                st.error(f"❌ **{o_symbol}** is not a valid stock.")
            else:
                result = place_order(
                    kite, o_symbol, o_exchange, o_txn, o_qty,
                    order_type=o_type, product=o_product,
                    price=o_price if o_price > 0 else None,
                    trigger_price=o_trigger if o_trigger > 0 else None,
                    validity=o_validity,
                )
                if result["success"]:
                    st.success(f"✅ Order placed — ID: **{result['order_id']}**")
                else:
                    st.error(f"❌ {result['error']}")

    with st.sidebar:
        _order_panel()

    # ═══════════════════════════════════════════════════════════
    # Top-level tabs: Stocks | Options
    # ═══════════════════════════════════════════════════════════
    stocks_main_tab, options_main_tab = st.tabs(["📈 Stocks", "🔗 Options"])

    # ── STOCKS TAB ─────────────────────────────────────────────
    with stocks_main_tab:
        # ── Auto-refreshing data fragment ──
        _run_every = timedelta(seconds=refresh_secs)

        @st.fragment(run_every=_run_every)
        def _live_data_panel():
            """Fetch quotes & render tables. Runs on its own timer without full page rerun."""
            _conn = get_db_connection()

            with st.spinner("Fetching real-time quotes from Kite Connect..."):
                quotes = fetch_realtime_quotes(kite, list(all_stock_names))
    
            if quotes:
                update_stocks_in_db(_conn, quotes)
                quotes_badge_slot.markdown(
                    f'<span class="status-badge badge-info">📊 {len(quotes)} quotes</span>',
                    unsafe_allow_html=True,
                )
            else:
                quotes_badge_slot.markdown(
                    '<span class="status-badge badge-warn">⚠ Cached data</span>',
                    unsafe_allow_html=True,
                )
    
            # ── Display tabs ──
            tab_names = [name for _, name in groups]
            tabs = st.tabs(tab_names)
    
            for tab, (group_id, group_name) in zip(tabs, groups):
                with tab:
                    rows = fetch_stocks_from_db(_conn, group_id)
    
                    if not rows:
                        st.info(f"No stocks found in {group_name}.")
                        continue
    
                    df = pd.DataFrame(rows, columns=["Name", "High", "Low", "Volume", "LTP", "Change (%)"])
    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Stocks", len(df))
                    col2.metric("Avg LTP", f"₹{df['LTP'].mean():,.2f}" if df['LTP'].notna().any() else "—")
                    if df['Change (%)'].notna().any():
                        gainer_idx = df['Change (%)'].idxmax()
                        loser_idx  = df['Change (%)'].idxmin()
                        col3.metric(
                            "Top Gainer",
                            df.loc[gainer_idx, 'Name'],
                            f"{df['Change (%)'].max():+.2f}%",
                        )
                        col4.metric(
                            "Top Loser",
                            df.loc[loser_idx, 'Name'],
                            f"{df['Change (%)'].min():+.2f}%",
                            delta_color="inverse",
                        )
    
                    # Style the dataframe
                    def color_change(val):
                        if val is None or pd.isna(val):
                            return ""
                        return "color: #38a169; font-weight:600" if val > 0 \
                            else "color: #e53e3e; font-weight:600" if val < 0 else ""
    
                    styled_df = df.style.map(color_change, subset=["Change (%)"])
                    styled_df = styled_df.format({
                        "High":  "₹{:,.2f}",
                        "Low":   "₹{:,.2f}",
                        "LTP":   "₹{:,.2f}",
                        "Volume": "{:,.0f}",
                        "Change (%)": "{:+.2f}%",
                    }, na_rep="—")
    
                    st.dataframe(
                        styled_df,
                        hide_index=True,
                        column_config={
                            "Name":       st.column_config.TextColumn("Name", width="medium"),
                            "High":       st.column_config.TextColumn("High", width="small"),
                            "Low":        st.column_config.TextColumn("Low", width="small"),
                            "Volume":     st.column_config.TextColumn("Volume", width="small"),
                            "LTP":        st.column_config.TextColumn("LTP", width="small"),
                            "Change (%)": st.column_config.TextColumn("Change (%)", width="small"),
                        },
                    )
    
                    # ── Per-stock quick-trade buttons ──
                    with st.expander("⚡ Quick Trade", expanded=False):
                        qt_cols = st.columns([3, 2, 2, 1.5, 1.5])
                        qt_symbol = qt_cols[0].selectbox(
                            "Symbol", df["Name"].tolist(),
                            key=f"qt_sym_{group_id}",
                            label_visibility="collapsed",
                        )
                        qt_qty = qt_cols[1].number_input(
                            "Qty", min_value=1, value=1, step=1,
                            key=f"qt_qty_{group_id}",
                            label_visibility="collapsed",
                        )
                        qt_product = qt_cols[2].selectbox(
                            "Product", ["CNC", "MIS", "NRML"],
                            key=f"qt_prod_{group_id}",
                            label_visibility="collapsed",
                        )
                        if qt_cols[3].button("🟢 BUY", key=f"qt_buy_{group_id}", width="stretch"):
                            res = place_order(kite, qt_symbol, "NSE", "BUY", qt_qty,
                                              order_type="MARKET", product=qt_product)
                            if res["success"]:
                                st.success(f"BUY order placed — ID: {res['order_id']}")
                            else:
                                st.error(res["error"])
                        if qt_cols[4].button("🔴 SELL", key=f"qt_sell_{group_id}", width="stretch"):
                            res = place_order(kite, qt_symbol, "NSE", "SELL", qt_qty,
                                              order_type="MARKET", product=qt_product)
                            if res["success"]:
                                st.success(f"SELL order placed — ID: {res['order_id']}")
                            else:
                                st.error(res["error"])
    
            # ── Order Book / Positions / Holdings / RSI Strategy ──────
            st.markdown("")
            ob_tab, pos_tab, hold_tab, rsi_tab = st.tabs(["📋 Order Book", "📊 Positions", "💼 Holdings", "🧠 RSI Strategy"])
    
            with ob_tab:
                orders = get_order_book(kite)
                if orders:
                    ob_df = pd.DataFrame(orders)
                    display_cols = [c for c in [
                        "order_id", "tradingsymbol", "exchange", "transaction_type",
                        "quantity", "price", "order_type", "product", "status",
                        "status_message", "order_timestamp",
                    ] if c in ob_df.columns]
                    ob_df = ob_df[display_cols]
    
                    def style_status(val):
                        colors = {
                            "COMPLETE": "color:#38a169;font-weight:600",
                            "REJECTED": "color:#e53e3e;font-weight:600",
                            "CANCELLED": "color:#a0aec0;font-weight:600",
                            "OPEN":     "color:#4299e1;font-weight:600",
                        }
                        return colors.get(val, "")
    
                    styled_ob = ob_df.style.map(style_status, subset=["status"])
                    st.dataframe(styled_ob, hide_index=True)
    
                    pending = ob_df[ob_df["status"].isin(["OPEN", "TRIGGER PENDING"])]
                    if not pending.empty:
                        st.markdown("**Cancel a pending order:**")
                        cancel_cols = st.columns([4, 1])
                        cancel_id = cancel_cols[0].selectbox(
                            "Order", pending["order_id"].tolist(), label_visibility="collapsed",
                        )
                        if cancel_cols[1].button("❌ Cancel", width="stretch"):
                            res = cancel_order(kite, cancel_id)
                            if res["success"]:
                                st.success(f"Order {cancel_id} cancelled.")
                                st.rerun()
                            else:
                                st.error(res["error"])
                else:
                    st.info("No orders placed today.")
    
            with pos_tab:
                positions = get_positions(kite)
                net = positions.get("net", [])
                if net:
                    pos_df = pd.DataFrame(net)
                    display_cols = [c for c in [
                        "tradingsymbol", "exchange", "quantity", "average_price",
                        "last_price", "pnl", "product",
                    ] if c in pos_df.columns]
                    pos_df = pos_df[display_cols]
    
                    def style_pnl(val):
                        try:
                            v = float(val)
                            if v > 0: return "color:#38a169;font-weight:600"
                            if v < 0: return "color:#e53e3e;font-weight:600"
                        except (ValueError, TypeError):
                            pass
                        return ""
    
                    if "pnl" in pos_df.columns:
                        styled_pos = pos_df.style.map(style_pnl, subset=["pnl"])
                        styled_pos = styled_pos.format({"pnl": "{:+,.2f}", "average_price": "₹{:,.2f}", "last_price": "₹{:,.2f}"}, na_rep="—")
                    else:
                        styled_pos = pos_df.style
                    st.dataframe(styled_pos, hide_index=True)
                else:
                    st.info("No open positions.")
    
            with hold_tab:
                holdings = get_holdings(kite)
                if holdings:
                    hold_df = pd.DataFrame(holdings)
    
                    # ── Classify Kite vs Smallcase ─────────────────────────
                    # The holdings API does NOT include a "tag" field, so we
                    # cross-reference with the order book (which has tags)
                    # and persist discoveries in a local JSON cache.
                    _SC_CACHE = os.path.join(os.path.dirname(__file__), "smallcase_symbols.json")
    
                    def _load_sc_cache():
                        if os.path.isfile(_SC_CACHE):
                            try:
                                with open(_SC_CACHE, "r") as f:
                                    return set(json.load(f))
                            except (json.JSONDecodeError, TypeError):
                                pass
                        return set()
    
                    def _save_sc_cache(symbols):
                        with open(_SC_CACHE, "w") as f:
                            json.dump(sorted(symbols), f, indent=2)
    
                    def _detect_smallcase_symbols(kite_inst):
                        """Merge cached Smallcase symbols with new ones from today's orders."""
                        cached = _load_sc_cache()
                        try:
                            orders = kite_inst.orders() or []
                            for o in orders:
                                tag = o.get("tag", "")
                                sym = o.get("tradingsymbol", "")
                                if tag and str(tag).strip() and sym:
                                    cached.add(sym)
                        except Exception:
                            pass
                        cached.discard("")
                        if cached:
                            _save_sc_cache(cached)
                        return cached
    
                    sc_symbols = _detect_smallcase_symbols(kite)
                    hold_df["_source"] = hold_df["tradingsymbol"].apply(
                        lambda s: "Smallcase" if s in sc_symbols else "Kite"
                    )
    
                    # Filter pills (Kite-style)
                    kite_count = (hold_df["_source"] == "Kite").sum()
                    sc_count   = (hold_df["_source"] == "Smallcase").sum()
                    h_filter = st.radio(
                        "Source",
                        ["All", f"Kite ({kite_count})", f"Smallcase ({sc_count})"],
                        horizontal=True, label_visibility="collapsed", key="hold_filter",
                    )
    
                    if h_filter.startswith("Kite"):
                        hold_df = hold_df[hold_df["_source"] == "Kite"]
                    elif h_filter.startswith("Smallcase"):
                        hold_df = hold_df[hold_df["_source"] == "Smallcase"]
    
                    display_cols = [c for c in [
                        "tradingsymbol", "exchange", "quantity", "average_price",
                        "last_price", "pnl", "day_change_percentage", "t1_quantity",
                    ] if c in hold_df.columns]
                    hold_df = hold_df[display_cols]
    
                    # Rename for readability
                    col_rename = {
                        "tradingsymbol": "Symbol", "exchange": "Exch", "quantity": "Qty",
                        "average_price": "Avg Price", "last_price": "LTP", "pnl": "P&L",
                        "day_change_percentage": "Day %", "t1_quantity": "T1 Qty",
                    }
                    hold_df = hold_df.rename(columns={k: v for k, v in col_rename.items() if k in hold_df.columns})
    
                    if "P&L" in hold_df.columns:
                        def style_hold_pnl(val):
                            try:
                                v = float(val)
                                if v > 0: return "color:#38a169;font-weight:600"
                                if v < 0: return "color:#e53e3e;font-weight:600"
                            except (ValueError, TypeError):
                                pass
                            return ""
    
                        pnl_subsets = [c for c in ["P&L", "Day %"] if c in hold_df.columns]
                        styled_hold = hold_df.style.map(style_hold_pnl, subset=pnl_subsets)
                        fmt = {"P&L": "{:+,.2f}"}
                        if "Avg Price" in hold_df.columns:
                            fmt["Avg Price"] = "₹{:,.2f}"
                        if "LTP" in hold_df.columns:
                            fmt["LTP"] = "₹{:,.2f}"
                        if "Day %" in hold_df.columns:
                            fmt["Day %"] = "{:+.2f}%"
                        styled_hold = styled_hold.format(fmt, na_rep="—")
                    else:
                        styled_hold = hold_df.style
                    st.dataframe(styled_hold, hide_index=True)
    
                    # Summary row
                    if "P&L" in hold_df.columns and not hold_df.empty:
                        total_pnl = hold_df["P&L"].sum()
                        pnl_color = "#38a169" if total_pnl >= 0 else "#e53e3e"
                        st.markdown(
                            f'<div style="text-align:right;font-size:0.82rem">'
                            f'Total P&L: <b style="color:{pnl_color}">₹{total_pnl:+,.2f}</b>'
                            f' · {len(hold_df)} holdings</div>',
                            unsafe_allow_html=True,
                        )
    
                    # ── Manage Smallcase symbol cache ──
                    with st.expander("™ Manage Smallcase symbols", expanded=False):
                        st.caption(
                            "The Kite Holdings API doesn't tag Smallcase stocks. "
                            "Symbols are auto-detected from order tags on trading days. "
                            "You can also add or remove symbols manually below."
                        )
                        all_syms = sorted(hold_df["Symbol"].tolist()) if "Symbol" in hold_df.columns else []
                        if not all_syms:
                            all_syms = sorted(pd.DataFrame(holdings)["tradingsymbol"].tolist())
                        current_sc = _load_sc_cache()
                        selected = st.multiselect(
                            "Smallcase symbols",
                            options=all_syms,
                            default=[s for s in all_syms if s in current_sc],
                            key="sc_multiselect",
                        )
                        if st.button("Save", key="sc_save"):
                            _save_sc_cache(set(selected))
                            st.success(f"Saved {len(selected)} Smallcase symbols.")
                            st.rerun()
                else:
                    st.info("No holdings found.")
    
            # ── RSI Strategy Scanner ──
            with rsi_tab:
                st.markdown("##### 🧠 RSI Auto-Order Scanner")
                st.caption("Scans stocks for RSI signals. **BUY** when RSI < oversold & bullish reversal. **SELL** when RSI > overbought & bearish reversal.")
    
                # Strategy settings
                rsi_c1, rsi_c2, rsi_c3, rsi_c4 = st.columns(4)
                rsi_capital   = rsi_c1.number_input("₹ Capital / trade", min_value=1000, value=50000, step=5000, key="rsi_capital")
                rsi_max_loss  = rsi_c2.number_input("₹ Max loss / trade", min_value=50, value=500, step=50, key="rsi_maxloss")
                rsi_limit     = rsi_c3.number_input("Order limit", min_value=1, value=3, step=1, key="rsi_limit")
                rsi_order_type = rsi_c4.selectbox("Order type", ["MARKET", "LIMIT"], key="rsi_otype")
    
                rsi_c5, rsi_c6, rsi_c7, rsi_c8 = st.columns(4)
                rsi_low  = rsi_c5.number_input("RSI oversold", min_value=5, max_value=50, value=30, step=5, key="rsi_low")
                rsi_high = rsi_c6.number_input("RSI overbought", min_value=50, max_value=95, value=70, step=5, key="rsi_high")
                rsi_interval = rsi_c7.selectbox("Candle interval", ["5minute", "15minute", "30minute", "60minute", "day"], key="rsi_intv")
                rsi_auto = rsi_c8.toggle("Auto-place orders", value=False, key="rsi_auto")
    
                if rsi_auto:
                    st.warning("⚠️ **Live trading enabled** — orders will be placed automatically on signals.")
    
                scan_btn_col, scan_status_col = st.columns([1, 3])
                run_scan = scan_btn_col.button("🔍 Run Scan", width="stretch", key="rsi_scan_btn")
    
                if run_scan:
                    with st.spinner("Scanning watchlist for RSI signals..."):
                        scan_results = scan_watchlist(
                            kite, list(all_stock_names),
                            capital=rsi_capital, max_loss=rsi_max_loss,
                            order_limit=rsi_limit, order_type=rsi_order_type,
                            rsi_low=rsi_low, rsi_high=rsi_high,
                            interval=rsi_interval, lookback_days=30,
                            auto_place=rsi_auto,
                        )
    
                    # Build results table
                    scan_rows = []
                    for r in scan_results:
                        if r.get("rsi") is not None:
                            row = {
                                "Symbol": r["symbol"],
                                "RSI": r["rsi"],
                                "LTP": r.get("ltp", 0),
                                "Close": r.get("close", 0),
                                "Signal": r.get("signal") or "—",
                            }
                            if r.get("order"):
                                o = r["order"]
                                row["Order"] = f"ID: {o['order_id']}" if o["success"] else f"❌ {o['error']}"
                                row["Qty"] = o.get("qty", 0)
                                row["SL"] = o.get("trigger_price", 0)
                            else:
                                row["Order"] = "—"
                                row["Qty"] = ""
                                row["SL"] = ""
                            scan_rows.append(row)
                        elif r.get("error"):
                            scan_rows.append({"Symbol": r["symbol"], "RSI": "—", "LTP": "—",
                                              "Close": "—", "Signal": "—", "Order": r["error"],
                                              "Qty": "", "SL": ""})
    
                    if scan_rows:
                        scan_df = pd.DataFrame(scan_rows)
    
                        def color_rsi(val):
                            try:
                                v = float(val)
                                if v < rsi_low: return "color:#e53e3e;font-weight:700"
                                if v > rsi_high: return "color:#38a169;font-weight:700"
                            except (ValueError, TypeError):
                                pass
                            return ""
    
                        def color_signal(val):
                            if val == "BUY": return "color:#38a169;font-weight:700"
                            if val == "SELL": return "color:#e53e3e;font-weight:700"
                            return "color:#a0aec0"
    
                        styled_scan = scan_df.style.map(color_rsi, subset=["RSI"])
                        styled_scan = styled_scan.map(color_signal, subset=["Signal"])
                        st.dataframe(styled_scan, hide_index=True)
    
                        signals = [r for r in scan_results if r.get("signal")]
                        orders_ok = sum(1 for r in scan_results if r.get("order", {}).get("success"))
                        scan_status_col.markdown(
                            f'<span class="status-badge badge-info">{len(scan_rows)} scanned</span> '
                            f'<span class="status-badge badge-warn">{len(signals)} signals</span> '
                            f'<span class="status-badge badge-success">{orders_ok} orders placed</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No data returned. Ensure market is open and stocks have sufficient history.")
    
            _conn.close()

    _live_data_panel()

    # ── OPTIONS TAB ────────────────────────────────────────────
    with options_main_tab:
        _render_option_chain_tab(kite)


def _render_option_chain_tab(kite):
    """Render the Option Chain tab with controls and data grid."""

    st.markdown("##### 🔗 Option Chain — Live OI & LTP")

    # ── Controls row ──
    oc_c1, oc_c2, oc_c3, oc_c4, oc_c5 = st.columns([1.5, 2, 1.5, 1.5, 1])

    oc_index = oc_c1.selectbox(
        "Index", list(INDEX_META.keys()), key="oc_index",
    )

    # ── Expiry discovery (cached in session_state) ──
    cache_key = f"_oc_expiries_{oc_index}"
    if cache_key not in st.session_state or st.session_state.get(f"_oc_exp_stale_{oc_index}"):
        with st.spinner("Discovering expiries..."):
            st.session_state[cache_key] = discover_expiries(kite, oc_index)
            st.session_state[f"_oc_exp_stale_{oc_index}"] = False

    expiry_list = st.session_state.get(cache_key, [])
    oc_expiry = oc_c2.selectbox(
        "Expiry", expiry_list if expiry_list else ["—"],
        key="oc_expiry",
    )

    oc_strikes = oc_c3.number_input(
        "Strikes", min_value=6, max_value=40, value=20, step=2, key="oc_strikes",
    )

    oc_timeframe = oc_c4.selectbox(
        "OI Interval",
        ["5minute", "15minute", "30minute", "60minute", "day"],
        key="oc_timeframe",
    )

    oc_refresh = oc_c5.button("🔄 Refresh", key="oc_refresh_btn", width="stretch")

    # Also add a button to re-discover expiries
    if oc_c2.button("↻ Reload expiries", key="oc_reload_exp"):
        st.session_state[f"_oc_exp_stale_{oc_index}"] = True
        st.rerun()

    if not expiry_list or oc_expiry == "—":
        st.warning("No expiries found. Market may be closed or the index is not available.")
        return

    # ── Fetch option chain data ──
    oc_cache_key = f"_oc_data_{oc_index}_{oc_expiry}_{oc_strikes}_{oc_timeframe}"
    need_fetch = oc_refresh or oc_cache_key not in st.session_state

    if need_fetch:
        with st.spinner(f"Fetching {oc_index} option chain …"):
            oc_data = fetch_option_chain(
                kite, oc_index, oc_expiry, oc_strikes, oc_timeframe,
            )
            st.session_state[oc_cache_key] = oc_data
    else:
        oc_data = st.session_state[oc_cache_key]

    if not oc_data["strikes"]:
        st.warning("No strike data returned. Check expiry / market hours.")
        return

    # ── Summary metrics ──
    spot = oc_data["spot"]
    atm = oc_data["atm_strike"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot", f"₹{spot:,.2f}")
    m2.metric("ATM Strike", f"{atm:,}")

    total_ce_oi = sum(r["ce_oi"] for r in oc_data["strikes"])
    total_pe_oi = sum(r["pe_oi"] for r in oc_data["strikes"])
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi else 0
    m3.metric("PCR (OI)", f"{pcr:.2f}")
    m4.metric("Strikes", len(oc_data["strikes"]))

    # ── Build DataFrame ──
    import pandas as pd
    rows = []
    for r in oc_data["strikes"]:
        rows.append({
            "CE OI":     r["ce_oi"],
            "CE ΔOI":    r["ce_oi_chg"],
            "CE Vol":    r.get("ce_volume", 0),
            "CE LTP":    r["ce_ltp"],
            "CE Chg":    r.get("ce_change", 0),
            "Strike":    r["strike"],
            "PE Chg":    r.get("pe_change", 0),
            "PE LTP":    r["pe_ltp"],
            "PE Vol":    r.get("pe_volume", 0),
            "PE ΔOI":    r["pe_oi_chg"],
            "PE OI":     r["pe_oi"],
            "_is_atm":   r["is_atm"],
        })
    df = pd.DataFrame(rows)

    # ── Style the chain (Sensibull / NSE colour scheme) ──
    atm_strike = oc_data["atm_strike"]
    atm_indices = set(df.index[df["_is_atm"]].tolist())
    display_df = df.drop(columns=["_is_atm"])

    ce_cols = ["CE OI", "CE ΔOI", "CE Vol", "CE LTP", "CE Chg"]
    pe_cols = ["PE Chg", "PE LTP", "PE Vol", "PE ΔOI", "PE OI"]
    col_list = list(display_df.columns)

    def _style_oc(styler):
        """Apply Sensibull / NSE-style option chain colours.

        ITM CE  (strike ≤ ATM) → soft green tint on call columns
        ITM PE  (strike ≥ ATM) → soft red tint on put columns
        ATM row → golden highlight band across all columns
        OI Δ / LTP Chg → green=+  red=−
        """

        # ── Colours ──
        CE_ITM_BG  = "background:rgba(232,245,233,0.85)"   # soft green
        PE_ITM_BG  = "background:rgba(255,235,238,0.85)"   # soft red
        ATM_BG     = "background:rgba(255,235,59,0.28)"    # golden / yellow
        STRIKE_BG  = "background:rgba(236,239,241,0.55);font-weight:700"  # neutral grey
        GREEN_TXT  = "color:#2e7d32"                       # green text
        RED_TXT    = "color:#c62828"                       # red text
        BOLD_GREEN = "color:#1b5e20;font-weight:700"
        BOLD_RED   = "color:#b71c1c;font-weight:700"

        # ── Row-level shading (ITM + ATM) ──
        def _shade_row(row):
            strike = row["Strike"]
            styles = [""] * len(row)

            # ATM band overrides ITM shading
            if row.name in atm_indices:
                return [ATM_BG] * len(row)

            for i, col in enumerate(col_list):
                if col in ce_cols and strike <= atm_strike:
                    styles[i] = CE_ITM_BG
                elif col in pe_cols and strike >= atm_strike:
                    styles[i] = PE_ITM_BG
                elif col == "Strike":
                    styles[i] = STRIKE_BG
            return styles

        # ── Per-cell colour for change values ──
        def _color_change(val):
            try:
                v = float(val)
                if v > 0:
                    return GREEN_TXT
                if v < 0:
                    return RED_TXT
            except (ValueError, TypeError):
                pass
            return ""

        # ── OI Δ with intensity (large buildup = bold) ──
        def _color_oi_delta(val):
            try:
                v = int(val)
                if v > 500_000:
                    return f"{BOLD_GREEN};background:rgba(200,230,201,0.4)"
                if v > 0:
                    return GREEN_TXT
                if v < -500_000:
                    return f"{BOLD_RED};background:rgba(255,205,210,0.4)"
                if v < 0:
                    return RED_TXT
            except (ValueError, TypeError):
                pass
            return ""

        styler = styler.apply(_shade_row, axis=1)
        styler = styler.map(_color_change, subset=["CE Chg", "PE Chg"])
        styler = styler.map(_color_oi_delta, subset=["CE ΔOI", "PE ΔOI"])
        styler = styler.format({
            "CE OI":   "{:,.0f}",
            "CE ΔOI":  "{:+,.0f}",
            "CE Vol":  "{:,.0f}",
            "CE LTP":  "₹{:,.2f}",
            "CE Chg":  "{:+.2f}",
            "Strike":  "{:,.0f}",
            "PE Chg":  "{:+.2f}",
            "PE LTP":  "₹{:,.2f}",
            "PE Vol":  "{:,.0f}",
            "PE ΔOI":  "{:+,.0f}",
            "PE OI":   "{:,.0f}",
        }, na_rep="—")
        return styler

    styled = _style_oc(display_df.style)
    st.dataframe(
        styled,
        hide_index=True,
        height=min(len(display_df) * 36 + 40, 720),
        column_config={
            "CE OI":   st.column_config.TextColumn("CE OI",  width="small"),
            "CE ΔOI":  st.column_config.TextColumn("CE ΔOI", width="small"),
            "CE Vol":  st.column_config.TextColumn("CE Vol", width="small"),
            "CE LTP":  st.column_config.TextColumn("CE LTP", width="small"),
            "CE Chg":  st.column_config.TextColumn("CE Chg", width="small"),
            "Strike":  st.column_config.TextColumn("Strike", width="small"),
            "PE Chg":  st.column_config.TextColumn("PE Chg", width="small"),
            "PE LTP":  st.column_config.TextColumn("PE LTP", width="small"),
            "PE Vol":  st.column_config.TextColumn("PE Vol", width="small"),
            "PE ΔOI":  st.column_config.TextColumn("PE ΔOI", width="small"),
            "PE OI":   st.column_config.TextColumn("PE OI",  width="small"),
        },
        width="stretch",
    )

    # ── Quick Trade for Options ──
    with st.expander("⚡ Quick Option Trade", expanded=False):
        # Build list of tradeable strikes
        strike_list = [str(r["strike"]) for r in oc_data["strikes"]]
        qt_c1, qt_c2, qt_c3, qt_c4, qt_c5, qt_c6 = st.columns([2, 1.5, 1, 1, 1, 1])

        qt_strike = qt_c1.selectbox("Strike", strike_list, key="oqt_strike")
        qt_opt    = qt_c2.selectbox("Type", ["CE", "PE"], key="oqt_type")
        qt_side   = qt_c3.selectbox("Side", ["BUY", "SELL"], key="oqt_side")
        qt_qty    = qt_c4.number_input("Lot", min_value=1, value=1, step=1, key="oqt_lot")
        qt_prod   = qt_c5.selectbox("Prod", ["MIS", "NRML"], key="oqt_prod")

        # Build the NFO trading symbol
        nfo_symbol = f"{INDEX_META[oc_index]['prefix']}{oc_expiry}{qt_strike}{qt_opt}"
        st.caption(f"Symbol: **NFO:{nfo_symbol}**")

        # Lot sizes
        lot_sizes = {"NIFTY": 75, "BANKNIFTY": 30}
        actual_qty = qt_qty * lot_sizes.get(oc_index, 30)

        if qt_c6.button(
            f"{'🟢' if qt_side == 'BUY' else '🔴'} {qt_side}",
            key="oqt_go", width="stretch",
        ):
            from trading.order_service import place_order as _place_order
            res = _place_order(
                kite, nfo_symbol, "NFO", qt_side, actual_qty,
                order_type="MARKET", product=qt_prod,
            )
            if res["success"]:
                st.success(f"✅ Order placed — ID: **{res['order_id']}** ({actual_qty} qty)")
            else:
                st.error(f"❌ {res['error']}")


if __name__ == "__main__":
    main()
