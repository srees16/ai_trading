"""
CSS Styles Module for Centurion Capital LLC.

Contains all CSS styling definitions and theme configurations
for the Streamlit application.
"""

import base64
import streamlit as st
from pathlib import Path
from typing import Optional


# Color palette constants
class Colors:
    """Application color palette."""
    
    PRIMARY = "#1a1a2e"
    SECONDARY = "#333"
    ACCENT_GREEN = "#00cc44"
    ACCENT_GREEN_DARK = "#00aa33"
    ACCENT_GREEN_LIGHT = "#66ff99"
    ACCENT_YELLOW = "#ffcc00"
    ACCENT_ORANGE = "#ff9933"
    ACCENT_RED = "#ff3333"
    
    BACKGROUND = "#f0f2f6"
    BACKGROUND_WHITE = "#ffffff"
    BORDER = "#ccc"
    BORDER_LIGHT = "#e0e0e0"
    
    TEXT_PRIMARY = "#1a1a2e"
    TEXT_SECONDARY = "#333"
    TEXT_MUTED = "#444"
    TEXT_CAPTION = "#555"


# Decision color mapping for charts
DECISION_COLORS = {
    'STRONG_BUY': Colors.ACCENT_GREEN,
    'BUY': Colors.ACCENT_GREEN_LIGHT,
    'HOLD': Colors.ACCENT_YELLOW,
    'SELL': Colors.ACCENT_ORANGE,
    'STRONG_SELL': Colors.ACCENT_RED,
}

# Sentiment color mapping
SENTIMENT_COLORS = {
    'positive': Colors.ACCENT_GREEN,
    'neutral': Colors.ACCENT_YELLOW,
    'negative': Colors.ACCENT_RED,
}

# Health status color mapping
HEALTH_COLORS = {
    'Safe': Colors.ACCENT_GREEN,
    'Strong': Colors.ACCENT_GREEN,
    'Grey Zone': Colors.ACCENT_YELLOW,
    'Moderate': Colors.ACCENT_YELLOW,
    'Distress': Colors.ACCENT_RED,
    'Weak': Colors.ACCENT_RED,
    'Likely Manipulator': Colors.ACCENT_RED,
    'Unlikely': Colors.ACCENT_GREEN,
}


def get_background_base64() -> Optional[str]:
    """
    Load background image and return base64 encoded string.
    
    Returns:
        Base64 encoded string or None if image doesn't exist
    """
    bg_path = Path(__file__).parent.parent / "nature_bg.png"
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def get_background_css(bg_base64: Optional[str] = None) -> str:
    """
    Generate CSS for background image styling.
    
    Args:
        bg_base64: Base64 encoded background image (optional, will load if not provided)
    
    Returns:
        CSS string for background styling
    """
    if bg_base64 is None:
        bg_base64 = get_background_base64()
    
    if not bg_base64:
        return ""
    
    return f"""
    /* Background image on root */
    .stApp {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    /* White overlay on main content area */
    [data-testid="stMain"] {{
        background: rgba(255, 255, 255, 0.92) !important;
    }}
    .main .block-container {{
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 2rem 3rem !important;
        margin: 1rem;
    }}
    """


def get_typography_css() -> str:
    """Get CSS for typography styling."""
    return """
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1a1a2e !important;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        text-align: center;
        color: #333 !important;
        margin-top: 0;
        margin-bottom: 0;
        font-weight: 500;
    }
    .page-subtitle {
        font-size: 2rem;
        font-weight: 600;
        color: #2d3436 !important;
        text-align: center;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .page-description {
        text-align: center;
        color: #444 !important;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    /* Ensure all text is visible */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li {
        color: #1a1a2e !important;
    }
    label, .stRadio label, .stCheckbox label {
        color: #1a1a2e !important;
    }
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        color: #1a1a2e !important;
    }
    .stSubheader {
        margin-top: 0.5rem;
        color: #1a1a2e !important;
    }
    /* Caption text */
    .stCaption, small {
        color: #555 !important;
    }
    /* Code blocks */
    code {
        color: #1a1a2e !important;
        background-color: #f0f2f6 !important;
    }
    """


def get_button_css() -> str:
    """Get CSS for button styling."""
    return """
    /* Fix button text overflow */
    .stButton > button {
        white-space: nowrap;
        overflow: visible;
        text-overflow: clip;
        padding: 0.5rem 0.75rem;
        font-size: 0.9rem;
        background-color: #f0f2f6 !important;
        color: #1a1a2e !important;
        border: 1px solid #ccc !important;
    }
    .stButton > button:hover {
        background-color: #e0e2e6 !important;
        border-color: #999 !important;
    }
    /* Green Run Analysis button */
    .stButton > button[kind="primary"] {
        background-color: #00cc44 !important;
        border-color: #00cc44 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #00aa33 !important;
        border-color: #00aa33 !important;
    }
    """


def get_layout_css() -> str:
    """Get CSS for layout and spacing."""
    return """
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 0.5rem;
    }
    /* Reduce default spacing */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    /* User menu bar - seamless with background */
    .stApp > header {
        background: transparent !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    /* First row (user menu) should blend seamlessly */
    .main .block-container > div:first-child {
        margin-top: 0;
        padding-top: 0;
    }
    .stMarkdown {
        margin-bottom: 0;
    }
    hr {
        margin: 0.75rem 0;
        border-color: rgba(0, 0, 0, 0.2);
    }
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    """


def get_footer_css() -> str:
    """Get CSS for footer styling."""
    return """
    /* Footer styles - seamless with background */
    .footer {
        text-align: center;
        color: #1a1a2e !important;
        font-size: 0.85rem;
        padding: 1.5rem 0 1rem 0;
        margin-top: 2rem;
        border-top: 1px solid rgba(0, 0, 0, 0.15);
        background: transparent;
        text-shadow: 0 0 8px rgba(255, 255, 255, 0.9), 0 0 16px rgba(255, 255, 255, 0.7);
    }
    .footer a {
        color: #1a1a2e !important;
        text-decoration: none;
    }
    """


def get_data_elements_css() -> str:
    """Get CSS for data elements (tables, inputs, metrics)."""
    return """
    /* Enhanced readability for data elements */
    .stDataFrame, .stTable {
        background-color: #ffffff !important;
        border-radius: 0.5rem;
    }
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
    }
    [data-testid="stMetricLabel"] {
        color: #333 !important;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6 !important;
        color: #1a1a2e !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1a1a2e !important;
    }
    /* Radio buttons and checkboxes */
    .stRadio > div, .stCheckbox > div {
        color: #1a1a2e !important;
    }
    /* Info/Warning/Error boxes */
    .stInfo, .stWarning, .stError, .stSuccess {
        color: #1a1a2e !important;
    }
    /* Plotly charts background */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }
    """


def get_complete_css(bg_base64: Optional[str] = None) -> str:
    """
    Get complete CSS stylesheet.
    
    Args:
        bg_base64: Optional base64 encoded background image
    
    Returns:
        Complete CSS string
    """
    return "\n".join([
        get_background_css(bg_base64),
        get_typography_css(),
        get_button_css(),
        get_layout_css(),
        get_footer_css(),
        get_data_elements_css(),
    ])


def apply_custom_styles():
    """Apply all custom CSS styles to the Streamlit application."""
    bg_base64 = get_background_base64()
    css = get_complete_css(bg_base64)
    
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_decision_style(decision: str) -> str:
    """
    Get CSS style for a decision value.
    
    Args:
        decision: Decision value (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
    
    Returns:
        CSS style string
    """
    styles = {
        'STRONG_BUY': 'background-color: #00cc44; color: white; font-weight: bold',
        'BUY': 'background-color: #66ff99',
        'HOLD': 'background-color: #ffcc00',
        'SELL': 'background-color: #ff9933',
        'STRONG_SELL': 'background-color: #ff3333; color: white; font-weight: bold',
    }
    return styles.get(decision, '')


def get_signal_style(signal: str) -> str:
    """
    Get CSS style for a trading signal.
    
    Args:
        signal: Signal value (BUY, SELL)
    
    Returns:
        CSS style string
    """
    if signal == 'BUY':
        return 'background-color: #90EE90'
    elif signal == 'SELL':
        return 'background-color: #FFB6C1'
    return ''
