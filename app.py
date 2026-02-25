"""
Centurion Capital LLC - Algorithmic Trading Application.

Enterprise AI Engine for Event-Driven Alpha.

Usage:
    streamlit run app.py

Manual stock tickers example: RGTI, QUBT, QBTS, IONQ
"""

import streamlit as st
import logging
from ui.styles import apply_custom_styles
from services.session import initialize_session_state
from auth.authenticator import check_authentication, render_user_menu

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
    # Apply custom styles
    apply_custom_styles()
    
    # Initialize session state
    initialize_session_state()
    
    # Authentication check - must be authenticated to access app
    if not check_authentication():
        return
    
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
        st.session_state["current_app"] = selected_app
        # Reset sub-page when switching apps
        st.session_state["current_page"] = "main"
        st.rerun()

    # ── Route to selected application ───────────────────────────
    if selected_app == "live_stocks":
        from kite_connect.zerodha_live import render_live_dashboard
        render_live_dashboard()

    elif selected_app == "rag_engine":
        from rag_pipeline.rag_page import render_rag_page
        render_rag_page()

    else:
        # Default: Trading Platform with sub-page routing
        _route_trading_platform()


def _route_trading_platform():
    """Route to the appropriate Trading Platform sub-page."""
    current_page = st.session_state.get('current_page', 'main')

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
