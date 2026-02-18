"""
Pages Module for Centurion Capital LLC.

Contains page rendering logic separated by functionality.
"""

from ui.pages.main_page import render_main_page
from ui.pages.analysis_page import render_analysis_page
from ui.pages.fundamental_page import render_fundamental_page
from ui.pages.backtesting_page import render_backtesting_page
from ui.pages.history_page import render_history_page

__all__ = [
    'render_main_page',
    'render_analysis_page',
    'render_fundamental_page',
    'render_backtesting_page',
    'render_history_page',
]
