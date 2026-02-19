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
from ui.pages import (
    render_main_page,
    render_analysis_page,
    render_fundamental_page,
    render_backtesting_page,
    render_history_page,
)
from services.session import initialize_session_state
from auth import check_authentication, render_user_menu

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
    
    # Route to appropriate page
    current_page = st.session_state.get('current_page', 'main')
    
    page_routes = {
        'main': render_main_page,
        'analysis': render_analysis_page,
        'fundamental': render_fundamental_page,
        'backtesting': render_backtesting_page,
        'history': render_history_page,
    }
    
    # Get the page renderer and execute
    page_renderer = page_routes.get(current_page, render_main_page)
    page_renderer()


if __name__ == "__main__":
    main()
