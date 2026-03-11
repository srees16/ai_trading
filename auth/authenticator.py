"""
Authentication Module for Centurion Capital LLC.

Provides user authentication, session management, and heartbeat
monitoring with YAML-based credential storage.
"""

import base64
import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import bcrypt
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from auth.shared_session import (
    SHARED_COOKIE_MAX_AGE,
    SHARED_COOKIE_NAME,
    create_shared_token,
    verify_shared_token,
)
from config import Config
from trading_strategies import list_strategies

load_dotenv()

logger = logging.getLogger(__name__)

# Path to credentials file
CREDENTIALS_PATH = Path(__file__).parent / "credentials.yaml"


# ---------------------------------------------------------------------------
# Password hashing — bcrypt with transparent SHA-256 legacy migration
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a password using bcrypt (12 rounds, random salt).

    Returns a UTF-8 bcrypt hash string suitable for storage.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def _is_legacy_sha256(hashed: str) -> bool:
    """Return True if *hashed* looks like a bare SHA-256 hex digest."""
    return len(hashed) == 64 and all(c in '0123456789abcdef' for c in hashed)


def verify_password(password: str, hashed: str) -> bool:
    """Verify *password* against a stored hash.

    Supports both:
    - **bcrypt** hashes (``$2b$…`` / ``$2a$…``)
    - **legacy SHA-256** hex digests (64 hex chars, no salt)

    Legacy hashes are checked via constant-time comparison so timing
    attacks are still mitigated.  Callers should re-hash and persist
    the bcrypt version after a successful legacy match (see
    ``Authenticator.authenticate``).
    """
    if _is_legacy_sha256(hashed):
        # Constant-time compare against the old unsalted SHA-256
        legacy_hash = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(legacy_hash, hashed)
    # bcrypt path
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except (ValueError, TypeError):
        return False


def load_credentials() -> Dict:
    """Load credentials from YAML file.

    The parsed dict is cached in ``st.session_state`` so the YAML file
    is only read from disk once per session (called 2× per rerun
    via ``check_authentication`` and ``render_user_menu``).

    Call ``invalidate_credentials_cache()`` after programmatic changes to
    the YAML file (e.g. password resets) to force a reload.
    """
    _CACHE_KEY = "_cached_credentials"
    if _CACHE_KEY in st.session_state:
        return st.session_state[_CACHE_KEY]

    if not CREDENTIALS_PATH.exists():
        # Create default credentials file
        _DEFAULT_ADMIN_PW = "admin123"
        _DEFAULT_ANALYST_PW = "analyst123"
        admin_pw = os.getenv("CENTURION_DEFAULT_ADMIN_PASSWORD", _DEFAULT_ADMIN_PW)
        analyst_pw = os.getenv("CENTURION_DEFAULT_ANALYST_PASSWORD", _DEFAULT_ANALYST_PW)
        if admin_pw == _DEFAULT_ADMIN_PW or analyst_pw == _DEFAULT_ANALYST_PW:
            logger.warning(
                "Using default bootstrap passwords — set CENTURION_DEFAULT_ADMIN_PASSWORD "
                "and CENTURION_DEFAULT_ANALYST_PASSWORD in .env for production use."
            )
        default_creds = {
            'users': {
                'admin': {
                    'password': hash_password(admin_pw),
                    'name': 'Administrator',
                    'role': 'admin'
                },
                'analyst': {
                    'password': hash_password(analyst_pw),
                    'name': 'Stock Analyst',
                    'role': 'analyst'
                }
            },
            'settings': {
                'session_timeout_minutes': 60,
                'max_login_attempts': 3
            }
        }
        save_credentials(default_creds)
        logger.info("Created default credentials file")
        st.session_state[_CACHE_KEY] = default_creds
        return default_creds
    
    try:
        with open(CREDENTIALS_PATH, 'r') as f:
            import yaml
            creds = yaml.safe_load(f)
        st.session_state[_CACHE_KEY] = creds
        return creds
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
        fallback = {'users': {}, 'settings': {'session_timeout_minutes': 60}}
        st.session_state[_CACHE_KEY] = fallback
        return fallback


def invalidate_credentials_cache():
    """Clear the cached credentials so the next call re-reads the YAML."""
    st.session_state.pop("_cached_credentials", None)


def save_credentials(credentials: Dict) -> bool:
    """Save credentials to YAML file."""
    try:
        CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CREDENTIALS_PATH, 'w') as f:
            import yaml
            yaml.dump(credentials, f, default_flow_style=False)
        # Invalidate cache so next load_credentials() picks up changes
        st.session_state.pop("_cached_credentials", None)
        return True
    except Exception as e:
        logger.error(f"Error saving credentials: {e}")
        return False


class Authenticator:
    """Handle user authentication with session caching."""
    
    def __init__(self):
        self.credentials = load_credentials()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for authentication."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        if 'user_name' not in st.session_state:
            st.session_state.user_name = None
        if 'login_time' not in st.session_state:
            st.session_state.login_time = None
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = None
    
    def _update_heartbeat(self):
        """Update the last activity timestamp (heartbeat)."""
        if st.session_state.authenticated:
            st.session_state.last_activity = datetime.now()
    
    def _check_session_timeout(self) -> bool:
        """Check if session has timed out based on login time or inactivity."""
        if st.session_state.login_time is None:
            return True
        
        timeout_minutes = self.credentials.get('settings', {}).get('session_timeout_minutes', 60)
        inactivity_minutes = self.credentials.get('settings', {}).get('inactivity_timeout_minutes', 30)
        
        # Check absolute session expiry (from login time)
        expiry_time = st.session_state.login_time + timedelta(minutes=timeout_minutes)
        if datetime.now() > expiry_time:
            logger.info("Session expired (absolute timeout)")
            self.logout()
            return True
        
        # Check inactivity timeout (from last activity)
        if st.session_state.last_activity:
            inactivity_expiry = st.session_state.last_activity + timedelta(minutes=inactivity_minutes)
            if datetime.now() > inactivity_expiry:
                logger.info("Session expired (inactivity timeout)")
                self.logout()
                return True
        
        return False
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authenticate user credentials.
        
        Returns:
            Tuple of (success, message)
        """
        max_attempts = self.credentials.get('settings', {}).get('max_login_attempts', 3)
        
        if st.session_state.login_attempts >= max_attempts:
            return False, "Too many failed attempts. Please wait a few minutes."
        
        users = self.credentials.get('users', {})
        
        if username not in users:
            st.session_state.login_attempts += 1
            return False, "Invalid username or password"
        
        user = users[username]
        
        if not verify_password(password, user['password']):
            st.session_state.login_attempts += 1
            remaining = max_attempts - st.session_state.login_attempts
            return False, f"Invalid username or password. {remaining} attempts remaining."
        
        # Successful authentication
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_role = user.get('role', 'user')
        st.session_state.user_name = user.get('name', username)
        st.session_state.login_time = datetime.now()
        st.session_state.login_attempts = 0
        # Reset SSO flag so the shared cookie is written on next render
        st.session_state.pop("_sso_cookie_set", None)

        # ── Auto-migrate legacy SHA-256 hash bcrypt ────────────
        if _is_legacy_sha256(user['password']):
            new_hash = hash_password(password)
            self.credentials['users'][username]['password'] = new_hash
            save_credentials(self.credentials)
            logger.info("Migrated password hash for '%s' from SHA-256 to bcrypt", username)

        logger.info(f"User '{username}' logged in successfully")
        return True, f"Welcome, {st.session_state.user_name}!"
    
    def logout(self):
        """Log out the current user and clear all session data."""
        username = st.session_state.get('username', 'Unknown')
        
        # Clear authentication state
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.user_name = None
        st.session_state.login_time = None
        st.session_state.last_activity = None
        st.session_state.pop("_sso_cookie_set", None)
        
        # Clear all analysis/app data to ensure nothing persists
        keys_to_clear = [
            'analysis_complete', 'signals', 'tickers', 'progress_messages',
            'ticker_mode', 'backtest_result', 'selected_strategy', 'current_page'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear any other session keys that might contain sensitive data
        sensitive_prefixes = ['analysis_', 'backtest_', 'signal_', 'db_']
        keys_to_remove = [k for k in st.session_state.keys() 
                         if any(k.startswith(prefix) for prefix in sensitive_prefixes)]
        for key in keys_to_remove:
            del st.session_state[key]
        
        logger.info(f"User '{username}' logged out - session cleared")
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated and session is valid."""
        if not st.session_state.authenticated:
            return False
        
        if self._check_session_timeout():
            return False
        
        # Update heartbeat on successful auth check
        self._update_heartbeat()
        
        return True
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current authenticated user info."""
        if not self.is_authenticated():
            return None
        
        return {
            'username': st.session_state.username,
            'name': st.session_state.user_name,
            'role': st.session_state.user_role,
            'login_time': st.session_state.login_time
        }
    
    def render_login_form(self) -> bool:
        """
        Render a polished, enterprise-grade login form with branding.
        
        Returns:
            True if user is authenticated, False otherwise
        """
        if self.is_authenticated():
            return True
        
        # Load logo for branding
        logo_html = self._get_logo_html()
        
        # Inject login page CSS
        st.markdown(self._get_login_css(), unsafe_allow_html=True)
        
        # --- Layout: vertically centered card ---
        st.markdown("<div class='login-spacer'></div>", unsafe_allow_html=True)
        
        _, center_col, _ = st.columns([1.2, 1.6, 1.2])
        
        with center_col:
            # Dark gradient header bar (matches module landing pages)
            st.markdown(
                f"""
                <div class="login-header-bar">
                    {logo_html}
                    <div class="login-header-title">Centurion Capital LLC</div>
                    <div class="login-header-subtitle">ALGORITHMIC TRADING · EVENT-DRIVEN ALPHA</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Login form
            with st.form("login_form"):
                username = st.text_input(
                    "Username", placeholder="Username", label_visibility="collapsed"
                )
                password = st.text_input(
                    "Password", type="password", placeholder="Password", label_visibility="collapsed"
                )
                
                submitted = st.form_submit_button(" Sign In", type="primary",
                                                       use_container_width=True)
                
                if submitted:
                    if username and password:
                        success, message = self.authenticate(username, password)
                        if success:
                            logger.info("[user=%s] Login successful", username)
                            st.success(message)
                            st.rerun()
                        else:
                            logger.warning("[user=%s] Login failed: %s", username, message)
                            st.error(message)
                    else:
                        logger.warning("Login attempt with empty credentials")
                        st.warning("Please enter both username and password")
            
            # Footer
            st.markdown(
                "<p style='text-align:center; margin-top:1.5rem; font-size:0.75rem; color:#999;'>"
                " © 2026 Centurion Capital LLC · All rights reserved</p>",
                unsafe_allow_html=True,
            )
        
        return False

    # ------------------------------------------------------------------
    # Private helpers for login form
    # ------------------------------------------------------------------

    @staticmethod
    def _get_logo_html() -> str:
        """Return an <img> tag with the base64-encoded logo, or empty string.

        The result is cached in ``st.session_state`` so the image file is
        only read from disk once per session.
        """
        _CACHE_KEY = "_login_logo_html"
        if _CACHE_KEY in st.session_state:
            return st.session_state[_CACHE_KEY]

        logo_path = Path(__file__).parent.parent / "ui" / "assets" / "centurion_logo.png"
        html = ""
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            html = (
                f'<img src="data:image/png;base64,{data}" '
                f'style="height:4rem; margin-bottom:0.75rem; '
                f'filter:brightness(0) invert(1);" />'
            )
        st.session_state[_CACHE_KEY] = html
        return html

    @staticmethod
    def _get_login_css() -> str:
        """Return all CSS specific to the login page.

        Includes minimal layout resets (hide sidebar, reduce spacing) so
        the login page looks correct without the heavyweight
        ``apply_custom_styles()`` call (which base64-encodes the
        background image).
        """
        return """
        <style>
        /* ---------- minimal layout resets ---------- */
        [data-testid="stSidebar"] { display: none; }
        .block-container { padding-top: 0.25rem; padding-bottom: 0.5rem; }
        [data-testid="stHeader"] { background: transparent !important; }

        /* ---------- login page overrides ---------- */
        .login-spacer { height: 14vh; }

        /* Dark gradient header bar (same as module pages) */
        .login-header-bar {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0f3460 100%);
            padding: 1.5rem 1.6rem 1.2rem 1.6rem;
            border-radius: 10px 10px 0 0;
            margin-bottom: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-left: 4px solid #4299e1;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.25);
            text-align: center;
        }
        .login-header-title {
            color: #ffffff !important;
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: 0.3px;
            line-height: 1.3;
        }
        .login-header-subtitle {
            color: #8b949e !important;
            font-size: 0.72rem;
            margin-top: 0.2rem;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            font-weight: 500;
        }

        /* Form container polish – blended into dark band */
        [data-testid="stForm"] {
            background: linear-gradient(180deg, #0f3460 0%, #0d1117 100%) !important;
            border: none !important;
            border-left: 4px solid #4299e1 !important;
            border-radius: 0 0 10px 10px !important;
            padding: 1.25rem 1.5rem 1.75rem 1.5rem !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        }

        /* Input fields – dark themed */
        [data-testid="stForm"] input {
            border-radius: 8px !important;
            padding: 0.6rem 0.75rem !important;
            font-size: 0.95rem !important;
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: #1a202c !important;
            -webkit-text-fill-color: #1a202c !important;
        }
        [data-testid="stForm"] input[type="password"] {
            -webkit-text-security: disc !important;
            color: #1a202c !important;
            -webkit-text-fill-color: #1a202c !important;
        }
        [data-testid="stForm"] input::placeholder {
            color: #718096 !important;
        }
        [data-testid="stForm"] [data-testid="stTextInput"] {
            margin-bottom: 0.5rem;
        }
        /* Input labels / captions in dark form */
        [data-testid="stForm"] label,
        [data-testid="stForm"] .stTextInput label {
            color: #a0aec0 !important;
        }

        /* Hide 'Press Enter to submit form' caption */
        [data-testid="stForm"] .st-emotion-cache-1ny7cjd,
        [data-testid="stForm"] [data-testid="InputInstructions"],
        [data-testid="stForm"] .stFormSubmitContent > div:last-child small,
        [data-testid="stForm"] .st-emotion-cache-ue6h4q {
            display: none !important;
        }

        /* Sign In button */
        [data-testid="stForm"] button[kind="primary"] {
            background: linear-gradient(135deg, #00cc44, #00aa33) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.65rem 1rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px;
            margin-top: 0.5rem;
            transition: opacity 0.2s ease;
        }
        [data-testid="stForm"] button[kind="primary"]:hover {
            opacity: 0.9;
        }
        </style>
        """


def check_authentication() -> bool:
    """
    Check if user is authenticated. Call at the start of main().

    Supports cross-app SSO: if the user logged in via the FastAPI
    docs page (port 9001), the shared cookie is detected and the
    Streamlit session is auto-populated without re-entering
    credentials.

    Returns:
        True if authenticated, False otherwise (renders login form)
    """
    auth = Authenticator()

    # ── Already authenticated in this session ──────────────────
    if auth.is_authenticated():
        _ensure_shared_cookie_set()
        return True

    # ── SSO: check for token passed via query-param redirect ──
    sso_token = st.query_params.get("_sso")
    if sso_token:
        payload = verify_shared_token(sso_token)
        if payload:
            _sso_auto_login(payload)
            # Remove the one-time param and reload authenticated
            del st.query_params["_sso"]
            st.rerun()
        else:
            # Expired / invalid — clear the param
            del st.query_params["_sso"]

    # ── SSO: inject JS to read the shared cookie ──────────────
    # If the cookie exists (set by FastAPI login), the JS redirects
    # the browser to ?_sso=<token> which the block above catches
    # on the next render cycle.
    if not st.session_state.get("authenticated", False):
        _inject_sso_cookie_check_js()

    return auth.render_login_form()


def logout():
    """Log out the current user and clear the shared SSO cookie."""
    auth = Authenticator()
    auth.logout()
    _inject_delete_cookie_js()


# ---------------------------------------------------------------------------
# SSO helpers — shared cookie between Streamlit and FastAPI
# ---------------------------------------------------------------------------

def _sso_auto_login(payload: Dict):
    """Populate Streamlit session_state from a verified SSO token payload."""
    username = payload["u"]
    role = payload["r"]
    creds = load_credentials()
    user = creds.get("users", {}).get(username, {})
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.user_role = role
    st.session_state.user_name = user.get("name", username)
    st.session_state.login_time = datetime.now()
    st.session_state.last_activity = datetime.now()
    st.session_state.login_attempts = 0
    logger.info("SSO auto-login: user=%s role=%s", username, role)


def _ensure_shared_cookie_set():
    """Inject JS to set the shared SSO cookie once per authenticated session."""
    if st.session_state.get("_sso_cookie_set"):
        return
    username = st.session_state.get("username", "")
    role = st.session_state.get("user_role", "")
    if not username:
        return
    token = create_shared_token(username, role)
    cookie_name = SHARED_COOKIE_NAME
    max_age = SHARED_COOKIE_MAX_AGE
    components.html(
        f"""<script>
        document.cookie = "{cookie_name}={token}; path=/; max-age={max_age}; samesite=lax";
        </script>""",
        height=0,
        width=0,
    )
    st.session_state._sso_cookie_set = True


def _inject_sso_cookie_check_js():
    """Inject JS that reads the shared cookie and redirects for auto-login.

    If the ``centurion_session`` cookie exists (set by the FastAPI login),
    the script redirects the browser to ``?_sso=<token>`` which Streamlit
    picks up on the next render cycle via ``st.query_params``.
    """
    cookie_name = SHARED_COOKIE_NAME
    components.html(
        f"""<script>
        (function() {{
            var cookies = document.cookie.split(";");
            for (var i = 0; i < cookies.length; i++) {{
                var c = cookies[i].trim();
                if (c.startsWith("{cookie_name}=")) {{
                    var token = c.substring({len(cookie_name) + 1});
                    if (token && !window.location.search.includes("_sso=")) {{
                        var sep = window.location.search ? "&" : "?";
                        window.location.href = window.location.pathname + window.location.search + sep + "_sso=" + encodeURIComponent(token);
                    }}
                    break;
                }}
            }}
        }})();
        </script>""",
        height=0,
        width=0,
    )


def _inject_delete_cookie_js():
    """Inject JS to delete the shared SSO cookie on logout."""
    cookie_name = SHARED_COOKIE_NAME
    components.html(
        f"""<script>
        document.cookie = "{cookie_name}=; path=/; max-age=0; samesite=lax";
        </script>""",
        height=0,
        width=0,
    )


def check_system_health() -> dict:
    """
    Check if essential system functionalities are accessible.
    Returns a dict with status of each component.
    """
    health = {
        'session_active': False,
        'config_loaded': False,
        'strategies_available': False,
        'storage_accessible': False
    }
    
    # Check session
    health['session_active'] = st.session_state.get('authenticated', False)
    
    # Check config
    try:
        health['config_loaded'] = hasattr(Config, 'DEFAULT_TICKERS')
    except Exception:
        pass
    
    # Check strategies
    try:
        strategies = list_strategies()
        health['strategies_available'] = len(strategies) > 0
    except Exception:
        pass
    
    # Check storage
    try:
        health['storage_accessible'] = Path.cwd().exists()
    except Exception:
        pass
    
    return health


def render_user_menu():
    """Render user menu in the app (call after authentication check)."""
    if not st.session_state.get('authenticated', False):
        return
    
    # User info
    user_name = st.session_state.get('user_name', 'User')
    user_role = st.session_state.get('user_role', 'user')
    
    # Update heartbeat
    auth = Authenticator()
    auth._update_heartbeat()
    
    # Check system health
    health = check_system_health()
    all_healthy = all(health.values())
    health_icon = "" if all_healthy else ""
    
    # Custom CSS for seamless user menu (no white background)
    st.markdown("""
    <style>
    .user-menu-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.2rem 0.5rem;
        margin: -0.5rem -0.5rem 0 -0.5rem;
        background: transparent;
    }
    .user-info-left {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        color: #1a1a2e;
        font-weight: 500;
    }
    .health-status {
        font-size: 0.72rem;
        color: #666;
        margin-left: 6px;
    }
    /* Remove white box around user menu area */
    [data-testid="stHorizontalBlock"]:first-child {
        background: transparent !important;
        margin-top: 28px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create layout: user info left, logout right — compact single row
    col_user, col_logout = st.columns([5, 1], gap="small")
    
    with col_user:
        st.markdown(
            f"<div style='white-space: nowrap; padding: 0; margin: 0; color: #1a1a2e; font-size: 0.82rem; line-height: 1.4;'>"
            f"<strong>{user_name}</strong> "
            f"<span style='font-size: 0.72rem; margin-left: 6px; opacity: 0.75;'>{health_icon}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col_logout:
        if st.button("Logout", key="logout_btn", use_container_width=True):
            logger.info("[user=%s] Logged out", st.session_state.get('username', 'unknown'))
            logout()
            st.rerun()
