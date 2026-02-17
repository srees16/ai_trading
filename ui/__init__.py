"""
UI module for Centurion Capital LLC Streamlit application.

This module provides:
- styles: CSS styling and theming
- components: Reusable UI components (header, footer, metrics, etc.)
- charts: Visualization and chart rendering functions
- tables: Data table rendering functions
- pages: Page-specific rendering logic
"""

from ui.styles import apply_custom_styles, get_background_css
from ui.components import (
    render_header,
    render_footer,
    render_page_header,
    load_logo_base64,
    render_navigation_buttons,
    render_metrics_cards,
)
from ui.charts import (
    render_decision_chart,
    render_sentiment_chart,
    render_score_distribution,
    render_fundamental_charts,
)
from ui.tables import (
    render_simple_summary_table,
    render_signals_table,
    render_top_signals,
    render_fundamental_table,
)

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
