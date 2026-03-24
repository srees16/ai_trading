"""
Session Management Module for Centurion Capital LLC.

Handles Streamlit session state initialization and management,
including the SessionCache lifecycle.
"""

import streamlit as st


def initialize_session_state():
    """Initialize all Streamlit session state variables.

    RAG state is deliberately **not** initialised here.  It is deferred
    to ``ensure_rag_state()`` which ``render_rag_page()`` calls on
    entry.  This avoids importing the heavy RAG config/model stack
    when the user is on a non-RAG page.
    """
    _init_analysis_state()
    _init_backtest_state()
    _init_navigation_state()


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
    if 'current_app' not in st.session_state:
        st.session_state.current_app = 'trading_platform'
    if 'current_market' not in st.session_state:
        st.session_state.current_market = 'US'


def _init_rag_state():
    """Initialize RAG pipeline session state variables.

    Reads the default ``rag_enabled`` value from ``RAGConfig`` so the
    env-var ``CENTURION_RAG_ENABLED`` is respected instead of being
    hard-coded to ``True``.
    """
    # Import here to avoid circular imports.
    from rag_pipeline.config import RAGConfig as _RAGConfig

    defaults = {
        'rag_enabled': _RAGConfig().rag_enabled,
        'rag_query': '',
        'rag_response': None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def ensure_rag_state():
    """Lazily initialize RAG state — call once when entering the RAG page."""
    if "_rag_state_initialised" not in st.session_state:
        _init_rag_state()
        st.session_state["_rag_state_initialised"] = True
