"""
Centurion Capital LLC - Algorithmic Trading Application.

Enterprise AI Engine for Event-Driven Alpha.

Usage:
    streamlit run app.py

Manual stock tickers example: RGTI, QUBT, QBTS, IONQ
"""

import sys
import os

# ── Guarantee project root is on sys.path BEFORE any local imports ──
# Use multiple strategies to find the project root reliably, even when
# Streamlit's script runner re-executes the file in an unusual context.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Also set as env-var so child processes / rerun cycles inherit it
os.environ.setdefault("PYTHONPATH", _PROJECT_ROOT)
# Ensure cwd matches so relative imports and file paths work
if os.getcwd() != _PROJECT_ROOT:
    os.chdir(_PROJECT_ROOT)

import logging
import threading

import streamlit as st

from auth.authenticator import check_authentication, render_user_menu
from services.session import initialize_session_state
from ui.styles import apply_custom_styles

def _preload_heavy_libs():
    try:
        import numpy      # noqa: F401
        import pandas      # noqa: F401
    except Exception:
        pass

threading.Thread(target=_preload_heavy_libs, daemon=True).start()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Centurion Capital LLC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def main():
    """Main Streamlit application entry point."""
    # Initialize session state
    initialize_session_state()

    # Authentication check - must be authenticated to access app.
    # NOTE: apply_custom_styles() is deliberately deferred to AFTER auth
    # so the login page doesn't pay the cost of base64-encoding the
    # background image (~90 KB JPEG).  The login form injects its own
    # lightweight CSS via Authenticator._get_login_css().
    if not check_authentication():
        return

    # Apply full app styles (background image, typography, layout)
    apply_custom_styles()

    # Render user menu (logout button, user info)
    render_user_menu()

    # ── Top-level app selector ──────────────────────────────────
    # Displayed as a compact radio bar right after login.
    APP_OPTIONS = {
        "trading_platform": "📈 US Stocks",
        "live_stocks":      "📈 Ind Stocks",
        "rag_engine":       "📚 RAG Engine",
    }

    current_app = st.session_state.get("current_app", "trading_platform")

    selected_app = st.radio(
        "Select Application",
        options=list(APP_OPTIONS.keys()),
        format_func=lambda k: APP_OPTIONS[k],
        index=list(APP_OPTIONS.keys()).index(current_app),
        horizontal=True,
        key="app_selector",
        label_visibility="collapsed",
    )

    if selected_app != current_app:
        logger.info("[user=%s] Switched module: %s -> %s",
                    st.session_state.get('username', 'unknown'),
                    APP_OPTIONS.get(current_app, current_app),
                    APP_OPTIONS.get(selected_app, selected_app))
        st.session_state["current_app"] = selected_app
        # Reset sub-page when switching apps
        st.session_state["current_page"] = "main"
        # Clear stale RAG one-shot keys so they don't leak across apps
        for _k in ("rag_resubmit_query",):
            st.session_state.pop(_k, None)
        st.rerun()

    # ── Route to selected application ───────────────────────────
    if selected_app == "live_stocks":
        logger.info("[user=%s] Rendering module: Ind Stocks",
                    st.session_state.get('username', 'unknown'))
        from kite_connect.zerodha_live import render_live_dashboard
        render_live_dashboard()

    elif selected_app == "rag_engine":
        logger.info("[user=%s] Rendering module: RAG Engine",
                    st.session_state.get('username', 'unknown'))
        from rag_pipeline.rag_page import render_rag_page
        render_rag_page()

    else:
        # Default: Trading Platform with sub-page routing
        _route_trading_platform()


def _route_trading_platform():
    """Route to the appropriate Trading Platform sub-page."""
    current_page = st.session_state.get('current_page', 'main')
    _user = st.session_state.get('username', 'unknown')
    logger.info("[user=%s] US Stocks sub-page: %s", _user, current_page)

    if current_page == 'analysis':
        from ui.pages.analysis_page import render_analysis_page
        render_analysis_page()
    elif current_page == 'fundamental':
        from ui.pages.fundamental_page import render_fundamental_page
        render_fundamental_page()
    elif current_page == 'backtesting':
        from ui.pages.backtesting_page import render_backtesting_page
        render_backtesting_page()
    elif current_page == 'crypto':
        from ui.pages.crypto_page import render_crypto_page
        render_crypto_page()
    elif current_page == 'history':
        from ui.pages.history_page import render_history_page
        render_history_page()
    else:
        from ui.pages.main_page import render_main_page
        render_main_page()


if __name__ == "__main__":
    main()
