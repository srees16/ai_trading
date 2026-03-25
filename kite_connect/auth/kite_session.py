"""
Shared Kite Connect session management.

Consolidates the duplicated login logic into a single reusable function.
"""

import logging
import os
import sys
from urllib.parse import parse_qs, urlparse

# Append (not insert-at-0) so the project-root 'auth' package is not shadowed
_kite_dir = os.path.dirname(os.path.dirname(__file__))
if _kite_dir not in sys.path:
    sys.path.append(_kite_dir)

from kiteconnect import KiteConnect, exceptions as kite_exceptions

from core.config import API_KEY, API_SECRET, KITE_APP_FILE

logger = logging.getLogger(__name__)


def _read_request_token():
    """Read the latest request_token value stored in kite_token_store.py."""
    with open(KITE_APP_FILE, "r") as f:
        for line in f:
            if line.strip().startswith("request_token"):
                # Handle both request_token='...' and request_token = '...'
                return line.strip().split("=", 1)[1].strip().strip("'\"")
    return None


def try_stored_token():
    """
    Try to create a session using just the stored request_token.

    Returns an authenticated KiteConnect instance if the token is still
    valid, or ``None`` if it has expired / is invalid.
    """
    request_token = _read_request_token()
    if not request_token:
        return None
    pool_cfg = {"pool_maxsize": int(os.getenv("KITE_POOL_MAXSIZE", "20"))}
    kite = KiteConnect(api_key=API_KEY, pool=pool_cfg)
    try:
        data = kite.generate_session(
            request_token=request_token, api_secret=API_SECRET,
        )
        kite.set_access_token(data["access_token"])
        return kite
    except (kite_exceptions.TokenException, kite_exceptions.InputException):
        return None


def http_login_kite():
    """
    Login to Kite via HTTP requests — no Selenium / browser needed.

    Uses Zerodha's login + 2FA API endpoints directly, then captures the
    OAuth redirect to extract the ``request_token``.  Works in headless /
    containerised environments (Docker, HF Spaces).

    Requires env vars: ZERODHA_USER_ID, ZERODHA_PASSWORD, ZERODHA_TOTP_SECRET.

    Returns an authenticated KiteConnect instance, or ``None`` on failure.
    """
    import requests

    from core.config import ZERODHA_USER_ID, ZERODHA_PASSWORD, ZERODHA_TOTP_SECRET

    if not all([ZERODHA_USER_ID, ZERODHA_PASSWORD, ZERODHA_TOTP_SECRET, API_KEY, API_SECRET]):
        logger.warning("HTTP login: missing one or more Zerodha env vars")
        return None

    try:
        import pyotp
    except ImportError:
        logger.warning("HTTP login: pyotp not installed")
        return None

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "X-Kite-Version": "3",
    })

    try:
        # Step 1: POST credentials
        login_resp = session.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": ZERODHA_USER_ID, "password": ZERODHA_PASSWORD},
        )
        login_data = login_resp.json()
        if login_data.get("status") != "success":
            logger.warning("HTTP login step 1 failed: %s", login_data.get("message", "unknown"))
            return None

        request_id = login_data["data"]["request_id"]

        # Step 2: POST TOTP
        totp_code = pyotp.TOTP(ZERODHA_TOTP_SECRET).now()
        twofa_resp = session.post(
            "https://kite.zerodha.com/api/twofa",
            data={
                "request_id": request_id,
                "twofa_value": totp_code,
                "user_id": ZERODHA_USER_ID,
                "twofa_type": "totp",
            },
        )
        twofa_data = twofa_resp.json()
        if twofa_data.get("status") != "success":
            logger.warning("HTTP login step 2 (TOTP) failed: %s", twofa_data.get("message", "unknown"))
            return None

        # Step 3: Visit OAuth login URL and follow redirect chain.
        # Zerodha's flow: /connect/login → /connect/finish?sess_id=...
        #                  /connect/finish → <redirect_url>?request_token=...
        # We follow up to 5 hops (without requests auto-redirect) to
        # capture the final Location header that contains request_token.
        oauth_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={API_KEY}"
        request_token = None

        for _hop in range(5):
            resp = session.get(oauth_url, allow_redirects=False)
            if resp.status_code not in (301, 302, 303):
                logger.warning("HTTP login step 3: expected redirect, got %s (hop %d)", resp.status_code, _hop)
                break
            location = resp.headers.get("Location", "")
            parsed = urlparse(location)
            params = parse_qs(parsed.query)
            request_token = params.get("request_token", [None])[0]
            if request_token:
                break
            # Follow next hop (only within kite.zerodha.com)
            if parsed.netloc and "zerodha.com" not in parsed.netloc:
                break
            oauth_url = location

        if not request_token:
            logger.warning("HTTP login: no request_token after redirect chain")
            return None

        logger.info("HTTP login: obtained request_token successfully")

        # Step 4: Generate session
        pool_cfg = {"pool_maxsize": int(os.getenv("KITE_POOL_MAXSIZE", "20"))}
        kite = KiteConnect(api_key=API_KEY, pool=pool_cfg)
        data = kite.generate_session(request_token=request_token, api_secret=API_SECRET)
        kite.set_access_token(data["access_token"])
        return kite

    except Exception as e:
        logger.warning("HTTP login failed: %s", e)
        return None


def create_kite_session():
    """
    Create and return an authenticated ``KiteConnect`` instance.

    Reads the stored *request_token* from ``kite_token_store.py``, attempts
    to generate a session.  If the token has expired, the interactive login
    flow (``kite_auth.fetch_request_token``) is launched automatically to
    obtain a fresh token.

    Returns
    -------
    KiteConnect
        An authenticated Kite instance with access_token already set.
    """
    request_token = _read_request_token()
    # configure connection pool size to avoid pool-full warnings
    pool_cfg = {"pool_maxsize": int(os.getenv("KITE_POOL_MAXSIZE", "20"))}
    kite = KiteConnect(api_key=API_KEY, pool=pool_cfg)

    try:
        data = kite.generate_session(
            request_token=request_token, api_secret=API_SECRET,
        )
        kite.set_access_token(data["access_token"])
        return kite
    except (kite_exceptions.TokenException, kite_exceptions.InputException):
        # Token expired or invalid — launch login flow
        try:
            from kite_connect.auth.kite_auth import fetch_request_token
        except ImportError:
            from auth.kite_auth import fetch_request_token

        print("\n  [!] Token expired. Launching login flow...\n")
        new_token = fetch_request_token()

        pool_cfg = {"pool_maxsize": int(os.getenv("KITE_POOL_MAXSIZE", "20"))}
        kite = KiteConnect(api_key=API_KEY, pool=pool_cfg)
        data = kite.generate_session(
            request_token=new_token, api_secret=API_SECRET,
        )
        kite.set_access_token(data["access_token"])
        return kite
