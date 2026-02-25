"""
Shared Kite Connect session management.

Consolidates the duplicated login logic into a single reusable function.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kiteconnect import KiteConnect, exceptions as kite_exceptions
from core.config import API_KEY, API_SECRET, KITE_APP_FILE


def _read_request_token():
    """Read the latest request_token value stored in kite_token_store.py."""
    with open(KITE_APP_FILE, "r") as f:
        for line in f:
            if line.strip().startswith("request_token"):
                # Handle both request_token='...' and request_token = '...'
                return line.strip().split("=", 1)[1].strip().strip("'\"")
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
    kite = KiteConnect(api_key=API_KEY)

    try:
        data = kite.generate_session(
            request_token=request_token, api_secret=API_SECRET,
        )
        kite.set_access_token(data["access_token"])
        return kite
    except (kite_exceptions.TokenException, kite_exceptions.InputException):
        # Token expired or invalid — launch login flow
        from auth.kite_auth import fetch_request_token

        print("\n  [!] Token expired. Launching login flow...\n")
        new_token = fetch_request_token()

        kite = KiteConnect(api_key=API_KEY)
        data = kite.generate_session(
            request_token=new_token, api_secret=API_SECRET,
        )
        kite.set_access_token(data["access_token"])
        return kite
