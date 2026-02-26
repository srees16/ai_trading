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
DB_PORT = int(os.getenv("KITE_DB_PORT", "5432"))
DB_USER = os.getenv("KITE_DB_USER", "postgres")
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

# ── Streamlit ──────────────────────────────────────────────────
REFRESH_INTERVAL = 5  # seconds between auto-refresh

# ── Paths ──────────────────────────────────────────────────────
KITE_APP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "auth", "kite_token_store.py"
)
