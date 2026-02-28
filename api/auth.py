"""
API Authentication — token-based session management for FastAPI docs.

Uses ``itsdangerous`` (bundled with Starlette) for signed,
time-limited session cookies.  Credential verification re-uses
the existing ``auth.authenticator.verify_password`` and the
YAML credential store.
"""

import hashlib
import hmac
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secret key — read from env or generate a per-process random key
# ---------------------------------------------------------------------------
_SECRET_KEY = os.getenv("CENTURION_API_SECRET_KEY", "")
if not _SECRET_KEY:
    import secrets as _secrets
    _SECRET_KEY = _secrets.token_hex(32)
    logger.info("Generated ephemeral API secret key (set CENTURION_API_SECRET_KEY for persistence)")

_SERIALIZER = URLSafeTimedSerializer(_SECRET_KEY, salt="centurion-api-docs")

# Session cookie name
SESSION_COOKIE = "centurion_api_session"

# Token max-age in seconds (default 8 hours)
TOKEN_MAX_AGE = int(os.getenv("CENTURION_API_TOKEN_MAX_AGE", 28800))


# ---------------------------------------------------------------------------
# Credential helpers (stand-alone, no Streamlit dependency)
# ---------------------------------------------------------------------------

CREDENTIALS_YAML = Path(__file__).resolve().parent.parent / "auth" / "credentials.yaml"


def _is_legacy_sha256(hashed: str) -> bool:
    return len(hashed) == 64 and all(c in "0123456789abcdef" for c in hashed)


def _verify_password(password: str, hashed: str) -> bool:
    if _is_legacy_sha256(hashed):
        legacy = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(legacy, hashed)
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except (ValueError, TypeError):
        return False


def _load_credentials_from_yaml() -> Dict:
    """Load credentials directly from YAML (no Streamlit)."""
    if not CREDENTIALS_YAML.exists():
        logger.warning("Credentials file not found: %s", CREDENTIALS_YAML)
        return {"users": {}}
    try:
        import yaml
        with open(CREDENTIALS_YAML, "r") as fh:
            return yaml.safe_load(fh) or {"users": {}}
    except Exception as exc:
        logger.error("Failed to load credentials: %s", exc)
        return {"users": {}}


def authenticate_user(username: str, password: str) -> Tuple[bool, str, str]:
    """
    Verify username/password against the YAML credential store.

    Returns (success, user_display_name, role).
    """
    creds = _load_credentials_from_yaml()
    users = creds.get("users", {})
    user = users.get(username)
    if user is None:
        return False, "", ""
    if not _verify_password(password, user.get("password", "")):
        return False, "", ""
    return True, user.get("name", username), user.get("role", "user")


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def create_session_token(username: str, role: str) -> str:
    """Create a signed, time-limited session token."""
    return _SERIALIZER.dumps({"u": username, "r": role})


def verify_session_token(token: str) -> Optional[Dict]:
    """
    Verify and decode a session token.

    Returns ``{"u": username, "r": role}`` on success, ``None`` on failure.
    """
    try:
        return _SERIALIZER.loads(token, max_age=TOKEN_MAX_AGE)
    except SignatureExpired:
        logger.debug("Session token expired")
        return None
    except BadSignature:
        logger.debug("Invalid session token")
        return None


# ---------------------------------------------------------------------------
# Login page HTML
# ---------------------------------------------------------------------------

LOGIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Centurion API — Login</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
      margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0f1117; color: #e0e0e0; display: flex; align-items: center;
      justify-content: center; min-height: 100vh;
  }
  .card {
      background: #1a1d28; border: 1px solid #2d2f3a; border-radius: 12px;
      padding: 2.5rem 2rem; width: 100%; max-width: 380px; box-shadow: 0 8px 32px rgba(0,0,0,.4);
  }
  h1 { margin: 0 0 .25rem; font-size: 1.35rem; color: #fff; text-align: center; }
  .subtitle { text-align: center; color: #888; font-size: .85rem; margin-bottom: 1.5rem; }
  label { display: block; font-size: .8rem; color: #aaa; margin-bottom: .3rem; }
  input[type=text], input[type=password] {
      width: 100%; padding: .6rem .75rem; border: 1px solid #3a3d4a; border-radius: 6px;
      background: #12141d; color: #e0e0e0; font-size: .95rem; margin-bottom: 1rem;
      outline: none; transition: border-color .2s;
  }
  input:focus { border-color: #4a8cff; }
  button {
      width: 100%; padding: .65rem; border: none; border-radius: 6px;
      background: #4a8cff; color: #fff; font-size: .95rem; font-weight: 600;
      cursor: pointer; transition: background .2s;
  }
  button:hover { background: #3a7cf5; }
  .error {
      background: #2d1a1a; border: 1px solid #5a2020; border-radius: 6px;
      padding: .5rem .75rem; margin-bottom: 1rem; font-size: .85rem; color: #ff6b6b;
      display: none;
  }
  .error.show { display: block; }
  .footer { text-align: center; margin-top: 1.2rem; font-size: .75rem; color: #555; }
</style>
</head>
<body>
<div class="card">
  <h1>Centurion Capital API</h1>
  <p class="subtitle">Sign in to access documentation</p>
  <div id="error" class="error"></div>
  <form id="loginForm">
    <label for="username">Username</label>
    <input type="text" id="username" name="username" autocomplete="username" required autofocus/>
    <label for="password">Password</label>
    <input type="password" id="password" name="password" autocomplete="current-password" required/>
    <button type="submit">Sign In</button>
  </form>
  <div class="footer">Centurion Capital LLC &copy; 2026</div>
</div>
<script>
document.getElementById('loginForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const errEl = document.getElementById('error');
    errEl.classList.remove('show');
    const body = new URLSearchParams({
        username: document.getElementById('username').value,
        password: document.getElementById('password').value,
    });
    try {
        const res = await fetch('/auth/login', { method: 'POST', body, redirect: 'follow' });
        if (res.redirected) { window.location.href = res.url; return; }
        const data = await res.json();
        if (!data.success) {
            errEl.textContent = data.detail || 'Authentication failed';
            errEl.classList.add('show');
        }
    } catch (err) {
        errEl.textContent = 'Network error — please try again';
        errEl.classList.add('show');
    }
});
</script>
</body>
</html>"""
