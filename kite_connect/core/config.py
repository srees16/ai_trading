"""
Centralized configuration for all kite_connect modules.

All credentials, connection parameters, and shared constants live here
so they are defined in one place and imported everywhere else.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ── PostgreSQL Database ────────────────────────────────────────
DB_HOST = os.getenv("KITE_DB_HOST", "localhost")
DB_PORT = int(os.getenv("KITE_DB_PORT", "9003"))
DB_USER = os.getenv("KITE_DB_USER", "")
DB_PASSWORD = os.getenv("KITE_DB_PASSWORD", "")
DB_NAME = os.getenv("KITE_DB_NAME", "livestocks_ind")
TABLE_NAME = os.getenv("KITE_DB_TABLE", "stocks")

# ── Zerodha Kite Connect API ──────────────────────────────────
API_KEY = os.getenv("ZERODHA_API_KEY", "")
API_SECRET = os.getenv("ZERODHA_API_SECRET", "")
LOGIN_URL = f"https://kite.zerodha.com/connect/login?api_key={API_KEY}"

# ── Zerodha Login Credentials (for Selenium auto-fill) ────────
ZERODHA_USER_ID = os.getenv("ZERODHA_USER_ID", "")
ZERODHA_PASSWORD = os.getenv("ZERODHA_PASSWORD", "")

# ── NSE ────────────────────────────────────────────────────────
NSE_URL = "https://www.nseindia.com/market-data/live-equity-market"
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

# ── Index Groups ───────────────────────────────────────────────
INDEX_GROUPS = ["NIFTY50", "NIFTYBANK", "NIFTYIT", "NIFTYENERGY"]

# ── Actual NSE Index Constituents ──────────────────────────────
# Used to seed the index_stocks mapping table on first launch.
# These are the well-known constituents (updated periodically by NSE).
# Unknown symbols are silently skipped during seeding.
INDEX_CONSTITUENTS = {
    "NIFTY50": [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
        "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
        "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
        "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
        "INDUSINDBK", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
        "KOTAKBANK", "LT", "LTIM", "M&M", "MARUTI",
        "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
        "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATAMOTORS",
        "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT",
        "ULTRACEMCO", "WIPRO",
    ],
    "NIFTYBANK": [
        "AUBANK", "AXISBANK", "BANDHANBNK", "BANKBARODA", "CANBK",
        "FEDERALBNK", "HDFCBANK", "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK",
        "KOTAKBANK", "PNB", "SBIN",
    ],
    "NIFTYIT": [
        "COFORGE", "HCLTECH", "INFY", "LTIM", "LTTS",
        "MPHASIS", "PERSISTENT", "TCS", "TECHM", "WIPRO",
    ],
    "NIFTYENERGY": [
        "ADANIENSOL", "ADANIGREEN", "BPCL", "COALINDIA", "GAIL",
        "HINDPETRO", "IOC", "NTPC", "ONGC", "POWERGRID",
        "RELIANCE", "TATAPOWER",
    ],
}

# ── Streamlit ──────────────────────────────────────────────────
REFRESH_INTERVAL = 5  # seconds between UI re-renders (reads from WebSocket cache)

# ── WebSocket / Webhook Settings ───────────────────────────────
# Tick mode: "full" (OHLC + depth), "quote" (OHLC only), "ltp" (price only)
WS_TICK_MODE = os.getenv("WS_TICK_MODE", "quote")
# Batch interval: how often ticks are flushed to subscribers (seconds)
WS_BATCH_INTERVAL = float(os.getenv("WS_BATCH_INTERVAL", "0.5"))
# DB write throttle: minimum seconds between DB writes
WS_DB_WRITE_INTERVAL = float(os.getenv("WS_DB_WRITE_INTERVAL", "1.0"))
# NSE market status check interval (seconds) — only used by background monitor
NSE_STATUS_CHECK_INTERVAL = int(os.getenv("NSE_STATUS_CHECK_INTERVAL", "60"))
# Max WebSocket reconnect attempts before giving up
WS_RECONNECT_MAX_TRIES = int(os.getenv("WS_RECONNECT_MAX_TRIES", "50"))

# ── Paths ──────────────────────────────────────────────────────
KITE_APP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "auth", "kite_token_store.py"
)
