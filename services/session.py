"""
Session Management Module for Centurion Capital LLC.

Handles Streamlit session state initialization and management.
"""

import streamlit as st
from typing import Any, Dict, List, Optional


def initialize_session_state():
    """Initialize all Streamlit session state variables."""
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


def reset_analysis_state():
    """Reset analysis state for a new analysis run."""
    st.session_state.analysis_complete = False
    st.session_state.signals = []
    st.session_state.progress_messages = []


def get_current_page() -> str:
    """
    Get the current page from session state.
    
    Returns:
        Current page identifier
    """
    return st.session_state.get('current_page', 'main')


def set_current_page(page: str):
    """
    Set the current page in session state.
    
    Args:
        page: Page identifier to navigate to
    """
    st.session_state.current_page = page


def get_tickers() -> List[str]:
    """
    Get the current tickers from session state.
    
    Returns:
        List of ticker symbols
    """
    return st.session_state.get('tickers', [])


def set_tickers(tickers: List[str]):
    """
    Set tickers in session state.
    
    Args:
        tickers: List of ticker symbols
    """
    st.session_state.tickers = tickers


def get_signals() -> List[Any]:
    """
    Get signals from session state.
    
    Returns:
        List of TradingSignal objects
    """
    return st.session_state.get('signals', [])


def set_signals(signals: List[Any]):
    """
    Set signals in session state.
    
    Args:
        signals: List of TradingSignal objects
    """
    st.session_state.signals = signals


def is_analysis_complete() -> bool:
    """
    Check if analysis is complete.
    
    Returns:
        True if analysis is complete
    """
    return st.session_state.get('analysis_complete', False)


def set_analysis_complete(complete: bool = True):
    """
    Set analysis completion status.
    
    Args:
        complete: Whether analysis is complete
    """
    st.session_state.analysis_complete = complete


def get_backtest_result() -> Optional[Any]:
    """
    Get backtest result from session state.
    
    Returns:
        BacktestResult object or None
    """
    return st.session_state.get('backtest_result', None)


def set_backtest_result(result: Any):
    """
    Set backtest result in session state.
    
    Args:
        result: BacktestResult object
    """
    st.session_state.backtest_result = result
