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
# ALWAYS force project root to position 0 — other modules (e.g.
# kite_connect/zerodha_live) append sub-package dirs that contain an
# 'auth/' folder which would shadow our top-level auth package.
if sys.path and sys.path[0] == _PROJECT_ROOT:
    pass  # already in pole position
else:
    # Remove any existing entry first, then re-insert at 0
    try:
        sys.path.remove(_PROJECT_ROOT)
    except ValueError:
        pass
    sys.path.insert(0, _PROJECT_ROOT)
# Also set as env-var so child processes / rerun cycles inherit it
os.environ.setdefault("PYTHONPATH", _PROJECT_ROOT)
# Ensure cwd matches so relative imports and file paths work
if os.getcwd() != _PROJECT_ROOT:
    os.chdir(_PROJECT_ROOT)

import logging
import time as _time

import streamlit as st

from auth.authenticator import check_authentication, render_user_menu
# Pin the top-level 'auth' package in sys.modules so that sub-packages
# like kite_connect/auth/ can never shadow it across Streamlit reruns.
import auth as _auth_pkg              # noqa: F811
sys.modules.setdefault('auth', _auth_pkg)

from services.session import initialize_session_state
from ui.styles import apply_custom_styles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

_LOG_INTERVAL = 5  # seconds
_last_module_log_ts: float = 0.0

def _throttled_module_info(msg: str, *args) -> None:
    """Log module-rendering messages at most once every _LOG_INTERVAL seconds."""
    global _last_module_log_ts
    now = _time.monotonic()
    if now - _last_module_log_ts >= _LOG_INTERVAL:
        logger.info(msg, *args)
        _last_module_log_ts = now


# ── Ollama model warm-up (runs exactly once per process, in background) ──
@st.cache_resource(show_spinner=False)
def _warmup_ollama() -> bool:
    """Kick off a background thread to pre-load the RAG LLM model.

    Sends a minimal ``/api/generate`` request with ``num_predict=1``
    so the model weights are loaded into RAM/VRAM *before* the first
    real user query.  Runs in a daemon thread so the UI renders
    immediately without waiting for the model to finish loading.
    Returns True once the thread is launched.
    """
    import threading

    def _do_warmup() -> None:
        import requests as _req
        import time as _t

        model = os.getenv(
            "RAG_MODEL",
            os.getenv("CENTURION_RAG_LLM_MODEL", "qwen2.5:3b"),
        )
        base_url = os.getenv(
            "CENTURION_RAG_LLM_URL", "http://localhost:11434"
        ).rstrip("/")

        logger.info("Ollama startup warm-up: loading model '%s' …", model)
        try:
            t0 = _t.monotonic()
            resp = _req.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "warmup",
                    "num_predict": 1,
                    "stream": False,
                },
                timeout=(10, 120),
            )
            resp.raise_for_status()
            elapsed = _t.monotonic() - t0
            logger.info(
                "Ollama startup warm-up complete: model=%s loaded in %.1fs",
                model, elapsed,
            )
        except _req.ConnectionError:
            logger.warning(
                "Ollama startup warm-up skipped: cannot connect to %s", base_url,
            )
        except Exception as exc:
            logger.warning("Ollama startup warm-up failed: %s", exc)

    t = threading.Thread(target=_do_warmup, daemon=True, name="ollama-warmup")
    t.start()
    return True


# Trigger warm-up at import time (first Streamlit process spin-up).

st.set_page_config(
    page_title="Centurion Capital LLC",
    page_icon="",
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
    
    # ── Ollama model warm-up (runs once, after login succeeds) ──
    _warmup_ollama()

    # Apply full app styles (background image, typography, layout)
    apply_custom_styles()

    # Render user menu (logout button, user info)
    render_user_menu()

    # ── Top-level app selector ──────────────────────────────────
    # Displayed as a compact radio bar right after login.
    APP_OPTIONS = {
        "trading_platform": "US Stocks",
        "live_stocks": "Ind Stocks",
        "finance_ml": "Financial ML",
        "crypto": "Crypto",
        "rag_engine": "RAG Engine"
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
        # NOTE: No st.rerun() here — selected_app already holds the new
        # value so the routing block below renders the correct module
        # immediately, saving an entire Streamlit round-trip (~1-2 s).

    # ── Route to selected application ───────────────────────────
    _user = st.session_state.get('username', 'unknown')

    if selected_app == "live_stocks":
        logger.info("[user=%s] Rendering module: Ind Stocks", _user)
        _route_ind_stocks()

    elif selected_app == "finance_ml":
        _throttled_module_info("[user=%s] Rendering module: Financial ML", _user)
        _get_renderer("finance_ml")()

    elif selected_app == "rag_engine":
        _throttled_module_info("[user=%s] Rendering module: RAG Engine", _user)
        _get_renderer("rag_engine")()

    elif selected_app == "crypto":
        logger.info("[user=%s] Rendering module: Crypto", _user)
        _get_renderer("crypto")()

    else:
        # Default: Trading Platform with sub-page routing
        _route_trading_platform()


# ── Cached module importers ─────────────────────────────────────
# @st.cache_resource ensures each render function is imported exactly
# once per process lifetime rather than on every Streamlit rerun.
@st.cache_resource
def _get_renderer(module_key: str):
    """Import and cache a module-level render function."""
    if module_key == "live_stocks":
        from kite_connect.zerodha_live import render_live_dashboard
        return render_live_dashboard
    elif module_key == "rag_engine":
        from rag_pipeline.rag_page import render_rag_page
        return render_rag_page
    elif module_key == "crypto":
        from ui.pages.crypto_page import render_crypto_page
        return render_crypto_page
    elif module_key == "finance_ml":
        from ui.pages.finance_ml_page import render_finance_ml_page
        return render_finance_ml_page
    elif module_key == "analysis":
        from ui.pages.analysis_page import render_analysis_page
        return render_analysis_page
    elif module_key == "fundamental":
        from ui.pages.fundamental_page import render_fundamental_page
        return render_fundamental_page
    elif module_key == "backtesting":
        from ui.pages.backtesting_page import render_backtesting_page
        return render_backtesting_page
    elif module_key == "history":
        from ui.pages.history_page import render_history_page
        return render_history_page
    elif module_key == "us_holdings":
        from ui.pages.us_holdings_page import render_us_holdings_page
        return render_us_holdings_page
    elif module_key == "ind_main":
        from ui.pages.ind_main_page import render_ind_main_page
        return render_ind_main_page
    elif module_key == "options":
        from ui.pages.options_page import render_options_page
        return render_options_page
    elif module_key == "main":
        from ui.pages.main_page import render_main_page
        return render_main_page
    raise ValueError(f"Unknown module: {module_key}")


def _route_trading_platform():
    """Route to the appropriate Trading Platform sub-page."""
    current_page = st.session_state.get('current_page', 'main')
    _user = st.session_state.get('username', 'unknown')
    logger.info("[user=%s] US Stocks sub-page: %s", _user, current_page)

    st.session_state['current_market'] = 'US'

    renderer = _get_renderer(current_page if current_page in (
        'analysis', 'fundamental', 'backtesting', 'history', 'us_holdings',
    ) else 'main')
    renderer()


def _route_ind_stocks():
    """Route to the appropriate Indian Stocks sub-page."""
    current_page = st.session_state.get('current_page', 'main')
    _user = st.session_state.get('username', 'unknown')
    logger.info("[user=%s] Ind Stocks sub-page: %s", _user, current_page)

    st.session_state['current_market'] = 'IND'

    if current_page == 'ind_kite':
        # Render the live Kite dashboard
        _get_renderer('live_stocks')()
    elif current_page == 'options':
        _get_renderer('options')()
    elif current_page in ('analysis', 'fundamental', 'backtesting', 'history'):
        # Reuse the same pages as US Stocks — they read current_market
        _get_renderer(current_page)()
    else:
        # Default: Indian Stocks main page (ticker selection)
        _get_renderer('ind_main')()


if __name__ == "__main__":
    main()
