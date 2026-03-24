"""
Shared session token management for cross-app SSO.

Both Streamlit (port 9000) and FastAPI (port 9001) use this module
to create and verify the same signed session tokens, enabling
single sign-on via a shared browser cookie.

IMPORTANT: Both apps must be accessed via ``localhost`` (not
``127.0.0.1``) for the cookie to be shared across ports.
"""

import logging
import os
from typing import Dict, Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared secret — MUST be identical across Streamlit and FastAPI processes
# ---------------------------------------------------------------------------
_SECRET_KEY = os.getenv("CENTURION_API_SECRET_KEY", "")
if not _SECRET_KEY:
    # Deterministic fallback so both processes agree without config.
    # Safe for local development; set the env var in production.
    _SECRET_KEY = "centurion-local-dev-shared-secret"
    logger.debug("Using default shared session secret (set CENTURION_API_SECRET_KEY for production)")

_SERIALIZER = URLSafeTimedSerializer(_SECRET_KEY, salt="centurion-shared-session")

# ---------------------------------------------------------------------------
# Cookie configuration
# ---------------------------------------------------------------------------
SHARED_COOKIE_NAME = "centurion_session"
SHARED_COOKIE_MAX_AGE = int(os.getenv("CENTURION_API_TOKEN_MAX_AGE", "28800"))  # 8 hours


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def create_shared_token(username: str, role: str) -> str:
    """Create a signed, time-limited session token."""
    return _SERIALIZER.dumps({"u": username, "r": role})


def verify_shared_token(token: str) -> Optional[Dict]:
    """Verify and decode a shared session token.

    Returns ``{"u": username, "r": role}`` on success, ``None`` on failure.
    """
    try:
        return _SERIALIZER.loads(token, max_age=SHARED_COOKIE_MAX_AGE)
    except SignatureExpired:
        logger.debug("Shared session token expired")
        return None
    except BadSignature:
        logger.debug("Invalid shared session token")
        return None
