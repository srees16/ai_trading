"""
UI module for Centurion Capital LLC Streamlit application.

This module provides:
- styles: CSS styling and theming
- components: Reusable UI components (header, footer, metrics, etc.)
- charts: Visualization and chart rendering functions
- tables: Data table rendering functions
- pages: Page-specific rendering logic

All sub-module imports are lazy to keep login-page load times fast.
"""

import importlib as _importlib

__all__ = [
    # Styles
    'apply_custom_styles',
    'get_background_css',
    # Components
    'render_header',
    'render_footer',
    'render_page_header',
    'load_logo_base64',
    'render_navigation_buttons',
    'render_metrics_cards',
    # Charts
    'render_decision_chart',
    'render_sentiment_chart',
    'render_score_distribution',
    'render_fundamental_charts',
    # Tables
    'render_simple_summary_table',
    'render_signals_table',
    'render_top_signals',
    'render_fundamental_table',
]

_LAZY_MAP: dict[str, tuple[str, str]] = {
    # styles
    'apply_custom_styles':       ('ui.styles',      'apply_custom_styles'),
    'get_background_css':        ('ui.styles',      'get_background_css'),
    # components
    'render_header':             ('ui.components',  'render_header'),
    'render_footer':             ('ui.components',  'render_footer'),
    'render_page_header':        ('ui.components',  'render_page_header'),
    'load_logo_base64':          ('ui.components',  'load_logo_base64'),
    'render_navigation_buttons': ('ui.components',  'render_navigation_buttons'),
    'render_metrics_cards':      ('ui.components',  'render_metrics_cards'),
    # charts
    'render_decision_chart':     ('ui.charts',      'render_decision_chart'),
    'render_sentiment_chart':    ('ui.charts',      'render_sentiment_chart'),
    'render_score_distribution': ('ui.charts',      'render_score_distribution'),
    'render_fundamental_charts': ('ui.charts',      'render_fundamental_charts'),
    # tables
    'render_simple_summary_table': ('ui.tables',    'render_simple_summary_table'),
    'render_signals_table':      ('ui.tables',      'render_signals_table'),
    'render_top_signals':        ('ui.tables',      'render_top_signals'),
    'render_fundamental_table':  ('ui.tables',      'render_fundamental_table'),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        mod_path, attr = _LAZY_MAP[name]
        module = _importlib.import_module(mod_path)
        obj = getattr(module, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
