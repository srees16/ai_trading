"""
run the app by using:
    streamlit run zerodha_live.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import streamlit as st

# Suppress harmless Tornado WebSocket closed errors during st.rerun()
logging.getLogger("tornado.general").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# Add kite_connect folder to path so we can import shared modules
# NOTE: *append* (not insert-at-0) so the project-root 'auth' package
# is never shadowed by kite_connect/auth/.
_KITE_DIR = os.path.dirname(__file__)
if _KITE_DIR not in sys.path:
    sys.path.append(_KITE_DIR)
# Add project root so we can import shared ui utilities
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
elif sys.path[0] != _PROJECT_ROOT:
    # Ensure project root stays at position 0
    sys.path.remove(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)

# ── Heavy imports are LAZY ──────────────────────────────────────────
# kiteconnect pulls in twisted+autobahn (~30 s on Windows).  We defer
# all heavy imports to first actual use so the login page isn't blocked.
from ui.components import load_logo_base64_small, render_header_bar, render_footer, spinner_html as _spinner_html

# Lazy singletons — populated on first call via _ensure_imports()
_kite_mod = None
_kite_exceptions = None

# Forward-declare names that _ensure_imports() injects into globals().
# This keeps Pylance / mypy happy while avoiding the actual imports.
pd = None                   # type: ignore[assignment]
requests = None             # type: ignore[assignment]
KiteConnect = None          # type: ignore[assignment]
kite_exceptions = None      # type: ignore[assignment]
REFRESH_INTERVAL = None     # type: ignore[assignment]
get_connection = None       # type: ignore[assignment]
create_kite_session = None  # type: ignore[assignment]
place_order = None          # type: ignore[assignment]
get_order_book = None       # type: ignore[assignment]
get_positions = None        # type: ignore[assignment]
get_holdings = None         # type: ignore[assignment]
cancel_order = None         # type: ignore[assignment]
scan_watchlist = None       # type: ignore[assignment]
discover_expiries = None    # type: ignore[assignment]
fetch_option_chain = None   # type: ignore[assignment]
INDEX_META = None           # type: ignore[assignment]


# ── Webhook service (lazy singleton) ────────────────────────
_webhook_service = None  # type: ignore[assignment]

def _get_webhook_service():
    """Return the webhook service singleton (lazy init)."""
    global _webhook_service
    if _webhook_service is None:
        from kite_connect.webhooks.service import WebhookService
        _webhook_service = WebhookService.get_instance()
    return _webhook_service


_heavy_imports_done = False  # Track whether the heavy (post-auth) imports have run
_heavy_imports_thread = None  # Background thread for pre-loading heavy imports


def _ensure_auth_imports():
    """Phase 1: Import ONLY what's needed for Kite authentication.

    This is the fast path — just kiteconnect + auth session.
    Called immediately when the user clicks "Start Kite Session"
    so the 2FA browser window appears quickly.
    """
    global _kite_mod, _kite_exceptions
    if _kite_mod is not None:
        return
    import kiteconnect as _km
    _kite_mod = _km
    _kite_exceptions = _km.exceptions
    glb = globals()
    glb['KiteConnect'] = _km.KiteConnect
    glb['kite_exceptions'] = _km.exceptions
    from kite_connect.auth.kite_session import create_kite_session as _cks
    glb['create_kite_session'] = _cks


def _do_heavy_imports():
    """The actual heavy import work — called from thread or inline."""
    global _heavy_imports_done
    if _heavy_imports_done:
        return
    glb = globals()
    import pandas as _pd
    glb['pd'] = _pd
    import requests as _req
    glb['requests'] = _req
    from core.config import REFRESH_INTERVAL as _ri
    glb['REFRESH_INTERVAL'] = _ri
    from core.db_service import get_connection as _gc
    glb['get_connection'] = _gc
    from trading.order_service import (
        place_order as _po, get_order_book as _gob,
        get_positions as _gp, get_holdings as _gh,
        cancel_order as _co,
    )
    glb['place_order'] = _po
    glb['get_order_book'] = _gob
    glb['get_positions'] = _gp
    glb['get_holdings'] = _gh
    glb['cancel_order'] = _co
    from trading.rsi_strategy import scan_watchlist as _sw
    glb['scan_watchlist'] = _sw
    from options.option_chain import (
        discover_expiries as _de, fetch_option_chain as _foc,
        INDEX_META as _im,
    )
    glb['discover_expiries'] = _de
    glb['fetch_option_chain'] = _foc
    glb['INDEX_META'] = _im
    _heavy_imports_done = True
    logger.info("Heavy imports completed")


def _ensure_heavy_imports():
    """Phase 2: Ensure heavy imports are ready.

    If the background pre-load thread is still running, wait for it.
    Otherwise, run imports inline (fast if already done).
    """
    global _heavy_imports_thread
    if _heavy_imports_done:
        return
    if _heavy_imports_thread is not None and _heavy_imports_thread.is_alive():
        _heavy_imports_thread.join()  # wait for background thread to finish
    else:
        _do_heavy_imports()


def _ensure_imports():
    """Legacy entry point — runs both phases."""
    _ensure_auth_imports()
    _ensure_heavy_imports()


# ── Kick off background pre-load immediately on module import ──
# This starts loading pandas, requests, etc. in a daemon thread
# while the landing page is being displayed, so by the time the
# user clicks "Start Kite Session" they're already (or nearly) loaded.
import threading as _threading
_heavy_imports_thread = _threading.Thread(
    target=_do_heavy_imports,
    daemon=True,
    name="heavy-imports-preload",
)
_heavy_imports_thread.start()


# ── Kite Connect Session ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_kite_session():
    """Login to Kite Connect and return the kite instance."""
    _ensure_auth_imports()
    return create_kite_session()


# ── Cached NSE market status ──────────────────────────────────
# TTL of 2 minutes avoids a synchronous NSE API call on every render
# while still updating reasonably often for market-open / close transitions.
@st.cache_data(ttl=120, show_spinner=False)
def _cached_nse_market_status():
    """Get market status from webhook service or NSE API fallback (cached 2 min)."""
    try:
        svc = _get_webhook_service()
        if svc._started:
            return svc.get_market_status()
    except Exception:
        pass
    # Fallback: direct NSE API call (only on first render before webhook boots)
    try:
        import requests as _req
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        sess = _req.Session()
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
    """Upsert the stocks table with real-time quote data.

    Uses INSERT … ON CONFLICT so that new symbols are created
    automatically and existing ones are updated with fresh prices.
    This fixes the old UPDATE-only approach that silently skipped
    stocks not yet in the table.
    """
    cur = conn.cursor()
    for symbol, data in quotes_data.items():
        cur.execute("""
            INSERT INTO stocks (name, high, low, volume, ltp, change, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (name) DO UPDATE
            SET high = EXCLUDED.high,
                low  = EXCLUDED.low,
                volume = EXCLUDED.volume,
                ltp  = EXCLUDED.ltp,
                change = EXCLUDED.change,
                updated_at = NOW();
        """, (
            symbol,
            data.get("high"),
            data.get("low"),
            data.get("volume"),
            data.get("ltp"),
            data.get("change_pct"),
        ))
    conn.commit()
    cur.close()


def fetch_stocks_from_db(conn, index_group_id):
    """Get stocks from DB for an index group.

    Uses LEFT JOIN so that stocks always appear in the tab even when
    the market is closed and the stocks table has no price data yet.
    Price columns will be NULL (shown as '—' in the UI).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT ix.stock_name,
               s.high,
               s.low,
               s.volume,
               s.ltp,
               s.change
        FROM index_stocks ix
        LEFT JOIN stocks s ON s.name = ix.stock_name
        WHERE ix.index_group_id = %s
        ORDER BY ix.stock_name;
    """, (index_group_id,))
    rows = cur.fetchall()
    cur.close()
    return rows


def seed_stocks_from_kite(kite, conn):
    """
    Populate the stocks table and index_stocks mapping from Kite instruments.

    This is the **bootstrap** step that fixes the empty-table problem:
      • Fetches all NSE equity instruments via ``kite.instruments("NSE")``
        (works even when the market is closed — no live data needed).
      • Inserts every equity symbol into the ``stocks`` table (name only,
        prices stay NULL until market opens).
      • Seeds ``index_stocks`` with the actual NIFTY 50 / NIFTY BANK /
        NIFTY IT / NIFTY ENERGY constituents from config.
      • Uses ON CONFLICT so it is safe to call repeatedly (idempotent).

    Called automatically on dashboard startup when index_stocks is empty.
    """
    from core.config import INDEX_CONSTITUENTS, INDEX_GROUPS

    # Close any pending transaction so we can safely switch to autocommit.
    conn.commit()
    prev_autocommit = conn.autocommit
    conn.autocommit = True

    cur = conn.cursor()

    # ── 1. Ensure UNIQUE constraint on stocks.name exists ──────
    #    The original schema only has a SERIAL PK; UPSERT needs this.
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'stocks'::regclass
                  AND contype = 'u'
                  AND conname = 'stocks_name_key'
            ) THEN
                ALTER TABLE stocks ADD CONSTRAINT stocks_name_key UNIQUE (name);
            END IF;
        END $$;
    """)

    # ── 2. Fetch all NSE equity instruments from Kite ──────────
    logger.info("Seeding stocks table from Kite instruments API…")
    instruments = kite.instruments("NSE")

    valid_names = set()
    inserted = 0
    for inst in instruments:
        name = inst.get("tradingsymbol", "")
        segment = inst.get("segment", "")
        inst_type = inst.get("instrument_type", "")
        # Only include equities, not indices / ETFs with odd names
        if not name or segment == "INDICES":
            continue
        if inst_type not in ("EQ", "BE", "SM", "ST", ""):
            continue
        # Skip junk names (SDLs, government bonds, etc.)
        if name.startswith("SDL") or name.startswith("2.5%"):
            continue
        valid_names.add(name)
        cur.execute("""
            INSERT INTO stocks (name) VALUES (%s)
            ON CONFLICT (name) DO NOTHING;
        """, (name,))
        inserted += 1

    logger.info("Stocks table seeded: %d instruments processed", inserted)

    # ── 3. Ensure index groups exist ───────────────────────────
    for idx_name in INDEX_GROUPS:
        cur.execute(
            "INSERT INTO index_groups (index_name) VALUES (%s) ON CONFLICT (index_name) DO NOTHING;",
            (idx_name,),
        )

    # ── 4. Seed index_stocks with actual constituents ──────────
    cur.execute("SELECT id, index_name FROM index_groups ORDER BY id;")
    group_rows = cur.fetchall()

    total_mapped = 0
    for group_id, group_name in group_rows:
        constituents = INDEX_CONSTITUENTS.get(group_name, [])
        for stock_name in constituents:
            if stock_name in valid_names:
                cur.execute("""
                    INSERT INTO index_stocks (index_group_id, stock_name)
                    VALUES (%s, %s)
                    ON CONFLICT (index_group_id, stock_name) DO NOTHING;
                """, (group_id, stock_name))
                total_mapped += 1
            else:
                logger.debug("Skipping %s for %s (not in instruments)", stock_name, group_name)

    logger.info(
        "Index mapping seeded: %d stock-group entries across %d groups",
        total_mapped, len(group_rows),
    )
    cur.close()

    # Restore previous autocommit state so callers are not surprised.
    conn.autocommit = prev_autocommit
    return total_mapped


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
def render_live_dashboard():
    """
    Render the live Indian stock market dashboard.

    Can be called from the main app router (app.py) or run standalone.
    Does NOT call st.set_page_config — the caller is responsible for that.
    """

    # ── Landing page gate: show intro until user clicks "Start Kite Session" ──
    if not st.session_state.get("kite_session_started", False):
        _render_landing_page()
        return

    _render_dashboard()

    render_footer()


def _render_landing_page():
    """Show an intro landing page before the Kite session is started."""
    render_header_bar(subtitle="Indian Equities · Zerodha Kite Connect")

    st.markdown("""
    <style>
        .landing-card {
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 0.5rem 1rem 1rem 1rem; margin: 0.6rem auto 0 auto; max-width: 720px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        div[data-testid="column"]:has(button) { margin-top: -0.6rem; }
        .landing-card h3 { color: #1a1a2e !important; margin-top: 0 !important; font-size: 1.15rem !important; }
        .landing-card ul { padding-left: 1.2rem; margin: 0.8rem 0; }
        .landing-card li { color: #2d3436 !important; font-size: 0.92rem; line-height: 1.75; }
        .landing-card li strong { color: #0f3460; }
    </style>

    <div class="landing-card">
        <h3>📈 Indian Equities — Live Dashboard</h3>
        <ul>
            <li><strong>Real-time quotes</strong> — NIFTY 50, Bank Nifty, IT &amp; Energy indices streamed via Zerodha Kite Connect</li>
            <li><strong>Live market status</strong> — automatic detection of pre-open, live, and post-market sessions from NSE</li>
            <li><strong>One-click order placement</strong> — place, modify, and cancel orders directly from the dashboard</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    _, col_btn, _ = st.columns([3, 1, 3])
    with col_btn:
        if st.button("Start Kite Session", type="primary", use_container_width=True):
            logger.info("[user=%s] Ind Stocks: Start Kite Session clicked", st.session_state.get('username', 'unknown'))
            st.session_state["kite_session_started"] = True
            st.rerun()

    render_footer()


def _render_dashboard():
    """Core dashboard — called after landing-page gate."""
    _ensure_auth_imports()  # fast: just kiteconnect + auth

    # ── Page-specific CSS (shared base styles come from apply_custom_styles) ──
    st.markdown("""
    <style>
        /* Header bar styling */
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

    # ── Step 1: Connect to Kite (fast — only needs kiteconnect) ──
    _auth_slot = st.empty()
    try:
        _auth_slot.markdown(
            _spinner_html("Connecting to Kite… (complete 2FA in the browser window)"),
            unsafe_allow_html=True,
        )
        kite = get_kite_session()
        _auth_slot.empty()
        st.session_state["kite"] = kite          # store for the dialog
        profile = kite.profile()
        kite_status = profile.get("user_id", "Connected")
        kite_connected = True
    except kite_exceptions.TokenException:
        _auth_slot.empty()
        st.cache_resource.clear()
        st.warning("Session expired. Reconnecting...")
        st.rerun()
        return
    except Exception as e:
        _auth_slot.empty()
        kite_status = "Disconnected"
        st.error(f"Kite Connect login failed: {e}")
        st.info("Run `py kite_token_store.py` first to generate a valid request token, then click Reconnect.")
        return

    # ── Step 2: Load remaining heavy modules (pandas, DB, trading) ──
    _load_slot = st.empty()
    _load_slot.markdown(
        _spinner_html("Loading dashboard modules…"),
        unsafe_allow_html=True,
    )
    _ensure_heavy_imports()
    _load_slot.empty()

    # ── Step 3: Market session status (needs `requests`, loaded above) ──
    pill_class, pill_label = _cached_nse_market_status()

    _logo_html = load_logo_base64_small()

    # ── Header bar (rendered after Kite login so kite_status is available) ──
    _pills_html = (
        f'<div class="live-pill {pill_class}"><span class="live-dot"></span> {pill_label}</div>'
        f'<div class="live-pill pill-open" style="margin-top:10px"><span class="live-dot"></span> Online</div>'
    )
    render_header_bar(
        subtitle="Real-time data · Zerodha Kite Connect",
        right_html=_pills_html,
    )

    try:
        from setup.db_setup import create_table as _ensure_tables
        _ensure_tables()  # no-op if tables already exist (CREATE TABLE IF NOT EXISTS)
        conn = get_db_connection()
        groups = fetch_index_groups(conn)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return

    # ── Auto-seed: populate stocks & index mappings if empty ───
    # This runs ONCE after 2FA login. Uses kite.instruments("NSE")
    # which works even when market is closed (no live data needed).
    if kite_connected and not st.session_state.get("_stocks_seeded"):
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM index_stocks;")
            ix_count = cur.fetchone()[0]
            cur.close()

            if ix_count == 0:
                with st.spinner("First launch — seeding stock database from Kite instruments…"):
                    seed_stocks_from_kite(kite, conn)
                    # Refresh groups since seed inserts groups too
                    groups = fetch_index_groups(conn)
                st.session_state["_stocks_seeded"] = True
                st.rerun()  # rerun so the fresh data is picked up
            else:
                st.session_state["_stocks_seeded"] = True
        except Exception as e:
            logger.error("Auto-seed failed: %s", e)
            st.warning(f"Could not auto-seed stock database: {e}")

    if not groups:
        st.warning("No index groups found. Run setup_livestocks_db.py first.")
        return

    # ── Top control bar: settings pushed left ──
    ctrl_container = st.container()
    with ctrl_container:
        st.markdown('<div class="ctrl-row">', unsafe_allow_html=True)
        c1, c_ws_status, c_spacer, c2 = st.columns([1.5, 2, 4, 1])

        with c1:
            refresh_secs = st.select_slider(
                "UI refresh ⏱",
                options=[2, 5, 10, 15, 20, 30, 45, 60],
                value=5,
                key="refresh_secs",
                help="How often the UI re-reads from the WebSocket cache (no API calls)",
            )

        with c_ws_status:
            _svc = _get_webhook_service()
            _mkt_open = _svc.market_is_open if _svc._started else False
            if _svc.is_streaming:
                _cached = _svc.quotes_count
                if _mkt_open:
                    st.markdown(
                        f'<span class="status-badge badge-success">'
                        f'🔴 WebSocket Live · {_cached} instruments'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="status-badge badge-info">'
                        f'🌙 WebSocket Connected · {_cached} instruments · Market Closed'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<span class="status-badge badge-warn">'
                    '📡 Polling mode (WebSocket starting…)'
                    '</span>',
                    unsafe_allow_html=True,
                )

        with c_spacer:
            quotes_badge_slot = st.empty()

        with c2:
            if st.button("🔄 Reconnect", use_container_width=True):
                logger.info("[user=%s] Ind Stocks: Reconnect clicked", st.session_state.get('username', 'unknown'))
                # Stop webhook service so it restarts with fresh session
                try:
                    _get_webhook_service().stop()
                except Exception:
                    pass
                st.session_state["_holdings_stale"] = True
                st.cache_resource.clear()
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Collect all stock names (used for sidebar dropdown + data fetch) ──
    all_stock_names = set()
    group_stocks_map = {}
    for group_id, group_name in groups:
        names = fetch_stock_names_by_index(conn, group_id)
        group_stocks_map[group_id] = names
        all_stock_names.update(names)
    sorted_stock_list = sorted(all_stock_names)
    conn.close()

    # ── Start webhook-based real-time streaming (replaces polling) ──
    # This runs ONCE after 2FA auth. The WebSocket pushes ticks instead
    # of us calling kite.quote() every N seconds.
    svc = _get_webhook_service()
    if not svc._started and kite_connected:
        try:
            def _on_session_expired(event):
                """Callback when session expires — triggers re-auth."""
                logger.warning("Webhook: session expired, will need re-auth")
                st.session_state["kite_session_started"] = False
                st.cache_resource.clear()

            svc.start(
                kite=kite,
                stock_symbols=list(all_stock_names),
                tick_mode="quote",          # OHLC data (no market depth)
                enable_nse_monitor=True,    # background NSE status tracking
                on_session_expired=_on_session_expired,
            )
            logger.info("Webhook streaming started for %d symbols", len(all_stock_names))
        except Exception as e:
            logger.error("Failed to start webhook service: %s", e)
            st.warning(f"Real-time streaming unavailable: {e}. Falling back to polling.")
    elif svc._started:
        # Update subscriptions if stock list changed
        svc._update_subscriptions(list(all_stock_names))

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
        if st.button(btn_label, use_container_width=True, type="primary"):
            logger.info("[user=%s] Ind Stocks: Place Order clicked — symbol=%s, side=%s, qty=%s, type=%s, product=%s",
                        st.session_state.get('username', 'unknown'), o_symbol, o_txn, o_qty, o_type, o_product)
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
        # ── Auto-refreshing stock data fragment ──
        _run_every = timedelta(seconds=refresh_secs)

        @st.fragment(run_every=_run_every)
        def _stock_quotes_panel():
            """Read real-time quotes from webhook cache & render stock data tables.

            Strategy:
              • Market OPEN  + WS streaming → read from UITickCache (zero API calls)
              • Market OPEN  + WS not yet connected → one-off REST kite.quote() bootstrap
              • Market CLOSED → show last-session values from DB (zero API/WS calls)
            """
            _conn = get_db_connection()

            svc = _get_webhook_service()
            _market_open = svc.market_is_open if svc._started else False

            if svc._started and svc.is_streaming:
                # ── WebSocket is connected ──
                quotes = svc.get_quotes()
                if quotes and _market_open:
                    # Market OPEN and live ticks flowing
                    update_stocks_in_db(_conn, quotes)
                    _last_update = svc.last_update_time
                    _age = time.time() - _last_update if _last_update else 0
                    _age_str = f"{_age:.0f}s ago" if _age < 60 else f"{_age/60:.1f}m ago"
                    quotes_badge_slot.markdown(
                        f'<span class="status-badge badge-success">'
                        f'🔴 Live · {len(quotes)} quotes · {_age_str}</span>',
                        unsafe_allow_html=True,
                    )
                elif _market_open:
                    # Market is open but we haven't received a tick yet
                    quotes_badge_slot.markdown(
                        '<span class="status-badge badge-warn">'
                        '⏳ Waiting for first tick…</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Market is closed — WS connected but quotes are stale
                    quotes_badge_slot.empty()
            elif _market_open:
                # Market is open but WS hasn't connected yet — one-off REST fallback
                _quotes_spinner = st.empty()
                _quotes_spinner.markdown(
                    _spinner_html("Fetching quotes via REST API (WebSocket connecting…)"),
                    unsafe_allow_html=True,
                )
                quotes = fetch_realtime_quotes(kite, list(all_stock_names))
                _quotes_spinner.empty()

                if quotes:
                    update_stocks_in_db(_conn, quotes)
                    quotes_badge_slot.markdown(
                        f'<span class="status-badge badge-info">'
                        f'📊 {len(quotes)} quotes (REST bootstrap)</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    quotes_badge_slot.markdown(
                        '<span class="status-badge badge-warn">'
                        '⚠ Could not fetch quotes</span>',
                        unsafe_allow_html=True,
                    )
            else:
                # Market is closed AND WS not connected — just show DB data
                quotes_badge_slot.empty()
    
            # ── Display tabs ──
            tab_names = [name for _, name in groups]
            tabs = st.tabs(tab_names)
    
            for tab, (group_id, group_name) in zip(tabs, groups):
                with tab:
                    rows = fetch_stocks_from_db(_conn, group_id)
    
                    if not rows:
                        st.info(
                            f"No stocks mapped to **{group_name}**. "
                            f"This will auto-populate on next dashboard restart after Kite login."
                        )
                        continue
    
                    df = pd.DataFrame(rows, columns=["Name", "High", "Low", "Volume", "LTP", "Change (%)"])
    
                    # Detect if we have any price data at all
                    _has_prices = df["LTP"].notna().any()
    
                    if not _has_prices:
                        st.markdown(
                            '<div style="background:#fefce8;border:1px solid #fde68a;border-radius:6px;'
                            'padding:0.5rem 0.8rem;margin-bottom:0.5rem;font-size:0.82rem;color:#92400e">'
                            '📴 <b>Market is closed</b> — showing stock names only. '
                            'Prices will update automatically when the market session is active.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Stocks", len(df))
                    col2.metric("Avg LTP", f"₹{df['LTP'].mean():,.2f}" if _has_prices else "—")
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
    
            _conn.close()

        _stock_quotes_panel()

        # ── Quick Trade (not auto-refreshed) ─────────────────────
        def _portfolio_panels():
            """Quick Trade, Order Book, Positions, Holdings, RSI — not auto-refreshed."""
            with st.expander("⚡ Quick Trade", expanded=False):
                qt_cols = st.columns([3, 2, 2, 1.5, 1.5])
                qt_symbol = qt_cols[0].selectbox(
                    "Symbol", sorted_stock_list,
                    key="qt_sym_global",
                    label_visibility="collapsed",
                )
                qt_qty = qt_cols[1].number_input(
                    "Qty", min_value=1, value=1, step=1,
                    key="qt_qty_global",
                    label_visibility="collapsed",
                )
                qt_product = qt_cols[2].selectbox(
                    "Product", ["CNC", "MIS", "NRML"],
                    key="qt_prod_global",
                    label_visibility="collapsed",
                )
                if qt_cols[3].button("🟢 BUY", key="qt_buy_global", use_container_width=True):
                    res = place_order(kite, qt_symbol, "NSE", "BUY", qt_qty,
                                      order_type="MARKET", product=qt_product)
                    if res["success"]:
                        st.success(f"BUY order placed — ID: {res['order_id']}")
                    else:
                        st.error(res["error"])
                if qt_cols[4].button("🔴 SELL", key="qt_sell_global", use_container_width=True):
                    res = place_order(kite, qt_symbol, "NSE", "SELL", qt_qty,
                                      order_type="MARKET", product=qt_product)
                    if res["success"]:
                        st.success(f"SELL order placed — ID: {res['order_id']}")
                    else:
                        st.error(res["error"])

            # ── Order Book / Positions / Holdings / RSI Strategy ──────
            st.markdown("")
            hold_tab, pos_tab, ob_tab, rsi_tab = st.tabs(["💼 Holdings", "📊 Positions", "📋 Order Book", "🧠 RSI Strategy"])

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
                        if cancel_cols[1].button("❌ Cancel", use_container_width=True):
                            logger.info("[user=%s] Ind Stocks: Cancel Order clicked — order_id=%s", st.session_state.get('username', 'unknown'), cancel_id)
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

                    # ── Overlay real-time prices from WebSocket ──
                    _pos_svc = _get_webhook_service()
                    _pos_quotes = _pos_svc.get_quotes() if _pos_svc._started else {}
                    if _pos_quotes and "tradingsymbol" in pos_df.columns:
                        for idx, row in pos_df.iterrows():
                            sym = row["tradingsymbol"]
                            ws_tick = _pos_quotes.get(sym)
                            if ws_tick and ws_tick.get("ltp"):
                                pos_df.at[idx, "last_price"] = ws_tick["ltp"]
                                avg = row.get("average_price", 0)
                                qty = row.get("quantity", 0)
                                if avg and qty:
                                    pos_df.at[idx, "pnl"] = round((ws_tick["ltp"] - avg) * qty, 2)

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
                # ── Fetch holdings once per session (portfolio structure is stable intraday) ──
                if "holdings_raw" not in st.session_state or st.session_state.get("_holdings_stale", True):
                    _raw_holdings = get_holdings(kite)
                    st.session_state["holdings_raw"] = _raw_holdings
                    st.session_state["_holdings_stale"] = False
                else:
                    _raw_holdings = st.session_state["holdings_raw"]

                if _raw_holdings:
                    hold_df = pd.DataFrame(_raw_holdings)

                    # ── Overlay real-time prices from WebSocket cache ──
                    _ws_svc = _get_webhook_service()
                    _ws_quotes = _ws_svc.get_quotes() if _ws_svc._started else {}
                    _overlay_count = 0
                    if _ws_quotes and "tradingsymbol" in hold_df.columns:
                        for idx, row in hold_df.iterrows():
                            sym = row["tradingsymbol"]
                            ws_tick = _ws_quotes.get(sym)
                            if ws_tick and ws_tick.get("ltp"):
                                ws_ltp = ws_tick["ltp"]
                                hold_df.at[idx, "last_price"] = ws_ltp
                                # Recalculate P&L with real-time LTP
                                avg = row.get("average_price", 0)
                                qty = row.get("quantity", 0)
                                if avg and qty:
                                    hold_df.at[idx, "pnl"] = round((ws_ltp - avg) * qty, 2)
                                # Recalculate day change %
                                close = row.get("close_price", 0)
                                if close:
                                    hold_df.at[idx, "day_change_percentage"] = round(
                                        ((ws_ltp - close) / close) * 100, 2
                                    )
                                _overlay_count += 1

                    if _overlay_count > 0:
                        st.markdown(
                            f'<span class="status-badge badge-success">'
                            f'🔴 Live · {_overlay_count}/{len(hold_df)} holdings updated via WebSocket</span>',
                            unsafe_allow_html=True,
                        )
                    elif not _ws_svc.market_is_open if _ws_svc._started else True:
                        st.markdown(
                            '<span class="status-badge badge-info">'
                            '🌙 Market Closed · showing last session values</span>',
                            unsafe_allow_html=True,
                        )

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
                            all_syms = sorted(pd.DataFrame(_raw_holdings)["tradingsymbol"].tolist())
                        current_sc = _load_sc_cache()
                        selected = st.multiselect(
                            "Smallcase symbols",
                            options=all_syms,
                            default=[s for s in all_syms if s in current_sc],
                            key="sc_multiselect",
                        )
                        if st.button("Save", key="sc_save"):
                            logger.info("[user=%s] Ind Stocks: Save Smallcase symbols clicked — count=%d", st.session_state.get('username', 'unknown'), len(selected))
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
                run_scan = scan_btn_col.button("🔍 Run Scan", use_container_width=True, key="rsi_scan_btn")

                if run_scan:
                    logger.info("[user=%s] Ind Stocks: Run RSI Scan clicked — capital=%s, rsi_low=%s, rsi_high=%s, interval=%s, auto=%s",
                                st.session_state.get('username', 'unknown'), rsi_capital, rsi_low, rsi_high, rsi_interval, rsi_auto)
                    _rsi_spinner = st.empty()
                    _rsi_spinner.markdown(_spinner_html("Scanning watchlist for RSI signals…"), unsafe_allow_html=True)
                    scan_results = scan_watchlist(
                        kite, list(all_stock_names),
                        capital=rsi_capital, max_loss=rsi_max_loss,
                        order_limit=rsi_limit, order_type=rsi_order_type,
                        rsi_low=rsi_low, rsi_high=rsi_high,
                        interval=rsi_interval, lookback_days=30,
                        auto_place=rsi_auto,
                    )
                    _rsi_spinner.empty()
    
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
    
        _portfolio_panels()

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
        _exp_spinner = st.empty()
        _exp_spinner.markdown(_spinner_html("Discovering expiries…"), unsafe_allow_html=True)
        st.session_state[cache_key] = discover_expiries(kite, oc_index)
        st.session_state[f"_oc_exp_stale_{oc_index}"] = False
        _exp_spinner.empty()

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

    oc_refresh = oc_c5.button("🔄 Refresh", key="oc_refresh_btn", use_container_width=True)
    if oc_refresh:
        logger.info("[user=%s] Ind Stocks: Refresh Option Chain clicked — index=%s", st.session_state.get('username', 'unknown'), oc_index)

    # Also add a button to re-discover expiries
    if oc_c2.button("↻ Reload expiries", key="oc_reload_exp"):
        logger.info("[user=%s] Ind Stocks: Reload Expiries clicked — index=%s", st.session_state.get('username', 'unknown'), oc_index)
        st.session_state[f"_oc_exp_stale_{oc_index}"] = True
        st.rerun()

    if not expiry_list or oc_expiry == "—":
        st.warning("No expiries found. Market may be closed or the index is not available.")
        return

    # ── Fetch option chain data ──
    oc_cache_key = f"_oc_data_{oc_index}_{oc_expiry}_{oc_strikes}_{oc_timeframe}"
    need_fetch = oc_refresh or oc_cache_key not in st.session_state

    if need_fetch:
        _oc_spinner = st.empty()
        _oc_spinner.markdown(_spinner_html(f"Fetching {oc_index} option chain…"), unsafe_allow_html=True)
        oc_data = fetch_option_chain(
            kite, oc_index, oc_expiry, oc_strikes, oc_timeframe,
        )
        st.session_state[oc_cache_key] = oc_data
        _oc_spinner.empty()
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
            key="oqt_go", use_container_width=True,
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
    st.set_page_config(page_title="Live Stocks - India", page_icon="📈", layout="wide")
    from ui.styles import apply_custom_styles
    apply_custom_styles()
    render_live_dashboard()
