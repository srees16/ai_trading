"""
Session Management Module for Centurion Capital LLC.

Handles Streamlit session state initialization and management,
including the SessionCache lifecycle.
"""

import streamlit as st
from typing import Any, List, Optional


def initialize_session_state():
    """Initialize all Streamlit session state variables."""
    _init_analysis_state()
    _init_backtest_state()
    _init_navigation_state()
    _init_cache()


def _init_analysis_state():
    """Initialize analysis-related session state variables."""
    defaults = {
        'analysis_complete': False,
        'signals': [],
        'progress_messages': [],
        'ticker_mode': "Default Tickers",

    }
    
    # Import here to avoid circular imports
    from config import Config
    
    if 'tickers' not in st.session_state:
        st.session_state.tickers = Config.DEFAULT_TICKERS
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _init_backtest_state():
    """Initialize backtesting-related session state variables."""
    defaults = {
        'backtest_result': None,
        'selected_strategy': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _init_navigation_state():
    """Initialize navigation-related session state variables."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'


def _init_cache():
    """
    Initialise the ``SessionCache`` singleton.

    The cache persists across Streamlit reruns within the same browser
    session.  It is only cleared explicitly (e.g. when the user starts a
    brand-new analysis with a completely different ticker set).
    """
    from services.cache import get_session_cache  # noqa: local import to avoid circular
    from config import Config

    cache = get_session_cache()
    # Apply configured TTL (in case env vars changed between restarts)
    from datetime import timedelta
    cache._default_ttl = timedelta(minutes=Config.CACHE_TTL_MINUTES)
