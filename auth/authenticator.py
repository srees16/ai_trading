"""
Authentication Module for Centurion Capital LLC.

Provides user authentication, session management, and heartbeat
monitoring with YAML-based credential storage.
"""

import hashlib
import hmac
import yaml
import streamlit as st
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Path to credentials file
CREDENTIALS_PATH = Path(__file__).parent / "credentials.yaml"


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return hmac.compare_digest(hash_password(password), hashed)


def load_credentials() -> Dict:
    """Load credentials from YAML file."""
    if not CREDENTIALS_PATH.exists():
        # Create default credentials file
        default_creds = {
            'users': {
                'admin': {
                    'password': hash_password('admin123'),
                    'name': 'Administrator',
                    'role': 'admin'
                },
                'analyst': {
                    'password': hash_password('analyst123'),
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
        return default_creds
    
    try:
        with open(CREDENTIALS_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
        return {'users': {}, 'settings': {'session_timeout_minutes': 60}}


def save_credentials(credentials: Dict) -> bool:
    """Save credentials to YAML file."""
    try:
        CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CREDENTIALS_PATH, 'w') as f:
            yaml.dump(credentials, f, default_flow_style=False)
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
    
    def add_user(self, username: str, password: str, name: str, role: str = 'analyst') -> bool:
        """Add a new user (admin only)."""
        if st.session_state.user_role != 'admin':
            logger.warning("Non-admin attempted to add user")
            return False
        
        if username in self.credentials['users']:
            return False
        
        self.credentials['users'][username] = {
            'password': hash_password(password),
            'name': name,
            'role': role
        }
        return save_credentials(self.credentials)
    
    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password."""
        users = self.credentials.get('users', {})
        
        if username not in users:
            return False, "User not found"
        
        if not verify_password(old_password, users[username]['password']):
            return False, "Current password is incorrect"
        
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters"
        
        self.credentials['users'][username]['password'] = hash_password(new_password)
        if save_credentials(self.credentials):
            return True, "Password changed successfully"
        return False, "Failed to save new password"
    
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
            # Branding header
            st.markdown(
                f"""
                <div class="login-card">
                    <div class="login-brand">
                        {logo_html}
                        <div class="login-title">Centurion Capital LLC</div>
                        <div class="login-tagline">Enterprise AI Engine for Event-Driven Alpha</div>
                    </div>
                    <div class="login-divider"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Login form
            with st.form("login_form"):
                st.markdown(
                    "<p style='text-align:center; margin:0 0 1rem 0; font-size:0.95rem; color:#555;'>"
                    "Sign in to continue</p>",
                    unsafe_allow_html=True,
                )
                username = st.text_input(
                    "Username", placeholder="Enter your username", label_visibility="collapsed"
                )
                password = st.text_input(
                    "Password", type="password", placeholder="Enter your password", label_visibility="collapsed"
                )
                
                submitted = st.form_submit_button(
                    "Sign In", use_container_width=True, type="primary"
                )
                
                if submitted:
                    if username and password:
                        success, message = self.authenticate(username, password)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter both username and password")
            
            # Footer
            st.markdown(
                "<p style='text-align:center; margin-top:1.5rem; font-size:0.75rem; color:#999;'>"
                "© 2026 Centurion Capital LLC · All rights reserved</p>",
                unsafe_allow_html=True,
            )
        
        return False

    # ------------------------------------------------------------------
    # Private helpers for login form
    # ------------------------------------------------------------------

    @staticmethod
    def _get_logo_html() -> str:
        """Return an <img> tag with the base64-encoded logo, or empty string."""
        import base64 as _b64

        logo_path = Path(__file__).parent.parent / "centurion_logo.png"
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                data = _b64.b64encode(f.read()).decode()
            return (
                f'<img src="data:image/png;base64,{data}" '
                f'style="height:4rem; margin-bottom:0.75rem;" />'
            )
        return ""

    @staticmethod
    def _get_login_css() -> str:
        """Return all CSS specific to the login page."""
        return """
        <style>
        /* ---------- login page overrides ---------- */
        .login-spacer { height: 4vh; }

        .login-card {
            text-align: center;
            padding: 0 0.5rem;
        }
        .login-brand {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .login-title {
            font-size: 1.9rem;
            font-weight: 700;
            color: #1a1a2e;
            letter-spacing: 0.5px;
        }
        .login-tagline {
            font-size: 0.85rem;
            color: #666;
            margin-top: 0.25rem;
            font-weight: 400;
        }
        .login-divider {
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, #00cc44, #00aa33);
            border-radius: 2px;
            margin: 1.25rem auto 0.5rem auto;
        }

        /* Form container polish */
        [data-testid="stForm"] {
            background: rgba(255,255,255,0.98) !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 12px !important;
            padding: 1.75rem 1.5rem !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06) !important;
        }

        /* Input fields */
        [data-testid="stForm"] input {
            border-radius: 8px !important;
            padding: 0.6rem 0.75rem !important;
            font-size: 0.95rem !important;
        }
        [data-testid="stForm"] [data-testid="stTextInput"] {
            margin-bottom: 0.5rem;
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
    
    Returns:
        True if authenticated, False otherwise (renders login form)
    """
    auth = Authenticator()
    return auth.render_login_form()


def logout():
    """Log out the current user."""
    auth = Authenticator()
    auth.logout()


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
        from config import Config
        health['config_loaded'] = hasattr(Config, 'DEFAULT_TICKERS')
    except Exception:
        pass
    
    # Check strategies
    try:
        from trading_strategies import list_strategies
        strategies = list_strategies()
        health['strategies_available'] = len(strategies) > 0
    except Exception:
        pass
    
    # Check storage
    try:
        from pathlib import Path
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
    health_icon = "🟢" if all_healthy else "🟡"
    
    # Custom CSS for seamless user menu (no white background)
    st.markdown("""
    <style>
    .user-menu-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 1rem;
        margin: -1rem -1rem 0.5rem -1rem;
        background: transparent;
    }
    .user-info-left {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 14px;
        color: #1a1a2e;
        font-weight: 500;
    }
    .health-status {
        font-size: 12px;
        color: #666;
        margin-left: 10px;
    }
    /* Remove white box around user menu area */
    [data-testid="stHorizontalBlock"]:first-child {
        background: transparent !important;
        margin-top: -10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create layout: user info left, health status center, logout right
    col_user, col_spacer, col_logout = st.columns([2, 4, 1])
    
    with col_user:
        st.markdown(
            f"<div style='white-space: nowrap; padding-top: 5px; margin-top: 5px; color: #1a1a2e;'>"
            f"👤 <strong>{user_name}</strong> <em>({user_role})</em> "
            f"<span style='font-size: 12px; margin-left: 8px;'>{health_icon} Session Active</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col_logout:
        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            logout()
            st.rerun()
