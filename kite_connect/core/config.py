"""
Centralized configuration for all kite_connect modules.

All credentials, connection parameters, and shared constants live here
so they are defined in one place and imported everywhere else.
"""

import os

# ── PostgreSQL Database ────────────────────────────────────────
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "postgres"
DB_PASSWORD = "superadmin1"
DB_NAME = "livestocks_ind"
TABLE_NAME = "stocks"

# ── Zerodha Kite Connect API ──────────────────────────────────
API_KEY = "hzcjwdgbs8wpon7p"
API_SECRET = "nz978uwv2jmkxbh9t5kf8nittl1cydvy"
LOGIN_URL = f"https://kite.zerodha.com/connect/login?api_key={API_KEY}"

# ── Zerodha Login Credentials (for Selenium auto-fill) ────────
ZERODHA_USER_ID = "auz459"
ZERODHA_PASSWORD = "Imbest1!"

# ── NSE ────────────────────────────────────────────────────────
NSE_URL = "https://www.nseindia.com/market-data/live-equity-market"
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

# ── Index Groups ───────────────────────────────────────────────
INDEX_GROUPS = ["NIFTY50", "NIFTYBANK", "NIFTYIT", "NIFTYENERGY"]

# ── Streamlit ──────────────────────────────────────────────────
REFRESH_INTERVAL = 30  # seconds between auto-refresh

# ── Paths ──────────────────────────────────────────────────────
KITE_APP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "auth", "kite_token_store.py"
)
