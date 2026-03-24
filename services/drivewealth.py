"""
DriveWealth API Service for Centurion Capital LLC.

Handles authentication, account linking, positions, holdings, orders,
and cash queries against the DriveWealth back-office API.

Reference: https://developer.drivewealth.com/apis/reference/introduction
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# DriveWealth sandbox/production base URL
_BASE_URL = os.getenv(
    "DW_BASE_URL",
    "https://bo-api.drivewealth.io/back-office",
).rstrip("/")


class DriveWealthError(Exception):
    """Raised when a DriveWealth API call fails."""

    def __init__(self, message: str, status_code: int = 0, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class DriveWealthClient:
    """Thin wrapper around the DriveWealth Back Office REST API."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        app_key: str,
        base_url: str = _BASE_URL,
    ):
        self.base_url = base_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._app_key = app_key
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0  # epoch seconds
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "dw-client-app-key": self._app_key,
        })

    # ── Authentication ─────────────────────────────────────────

    def authenticate(self) -> Dict[str, Any]:
        """Create a session token (POST /auth/tokens).

        Stores the access_token internally so subsequent calls are
        automatically authenticated.
        """
        resp = self._session.post(
            f"{self.base_url}/auth/tokens",
            json={
                "clientID": self._client_id,
                "clientSecret": self._client_secret,
            },
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        self._raise_for_status(resp, "authenticate")
        data = resp.json()
        self._access_token = data.get("access_token")
        expires_in = int(data.get("expires_in", 3600))
        self._token_expiry = time.time() + expires_in - 60  # 60 s buffer
        self._session.headers["Authorization"] = f"Bearer {self._access_token}"
        logger.info("DriveWealth authenticated — token valid for %ds", expires_in)
        return data

    @property
    def is_authenticated(self) -> bool:
        return bool(self._access_token and time.time() < self._token_expiry)

    def _ensure_auth(self):
        if not self.is_authenticated:
            self.authenticate()

    # ── Users ──────────────────────────────────────────────────

    def create_user(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /users — create a new user."""
        self._ensure_auth()
        resp = self._session.post(
            f"{self.base_url}/users",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        self._raise_for_status(resp, "create_user")
        return resp.json()

    def get_user(self, user_id: str) -> Dict[str, Any]:
        """GET /users/:userID"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/users/{user_id}",
            timeout=15,
        )
        self._raise_for_status(resp, "get_user")
        return resp.json()

    def get_kyc(self, user_id: str) -> Dict[str, Any]:
        """GET /users/:userID/kyc-status"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/users/{user_id}/kyc-status",
            timeout=15,
        )
        self._raise_for_status(resp, "get_kyc")
        return resp.json()

    # ── Accounts ───────────────────────────────────────────────

    def create_account(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /accounts"""
        self._ensure_auth()
        resp = self._session.post(
            f"{self.base_url}/accounts",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        self._raise_for_status(resp, "create_account")
        return resp.json()

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """GET /accounts/:accountID"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/accounts/{account_id}",
            timeout=15,
        )
        self._raise_for_status(resp, "get_account")
        return resp.json()

    def list_user_accounts(self, user_id: str) -> Dict[str, Any]:
        """GET /users/:userID/accounts"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/users/{user_id}/accounts",
            timeout=15,
        )
        self._raise_for_status(resp, "list_user_accounts")
        return resp.json()

    def get_account_cash(self, account_id: str) -> Dict[str, Any]:
        """GET /accounts/:accountID/summary/money"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/accounts/{account_id}/summary/money",
            timeout=15,
        )
        self._raise_for_status(resp, "get_account_cash")
        return resp.json()

    # ── Positions (Holdings) ───────────────────────────────────

    def list_positions(self, account_id: str) -> Dict[str, Any]:
        """GET /accounts/:accountID/summary/positions"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/accounts/{account_id}/summary/positions",
            timeout=15,
        )
        self._raise_for_status(resp, "list_positions")
        return resp.json()

    # ── Orders ─────────────────────────────────────────────────

    def create_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /orders"""
        self._ensure_auth()
        resp = self._session.post(
            f"{self.base_url}/orders",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        self._raise_for_status(resp, "create_order")
        return resp.json()

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """GET /orders/:orderID"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/orders/{order_id}",
            timeout=15,
        )
        self._raise_for_status(resp, "get_order")
        return resp.json()

    def list_resting_orders(self, account_id: str) -> Dict[str, Any]:
        """GET /accounts/:accountID/summary/orders"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/accounts/{account_id}/summary/orders",
            timeout=15,
        )
        self._raise_for_status(resp, "list_resting_orders")
        return resp.json()

    # ── Transactions ───────────────────────────────────────────

    def list_transactions(self, account_id: str) -> Dict[str, Any]:
        """GET /accounts/:accountID/transactions"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/accounts/{account_id}/transactions",
            timeout=15,
        )
        self._raise_for_status(resp, "list_transactions")
        return resp.json()

    # ── Performance ────────────────────────────────────────────

    def get_performance(self, account_id: str) -> Dict[str, Any]:
        """GET /accounts/:accountID/performance/summary"""
        self._ensure_auth()
        resp = self._session.get(
            f"{self.base_url}/accounts/{account_id}/performance/summary",
            timeout=15,
        )
        self._raise_for_status(resp, "get_performance")
        return resp.json()

    # ── Instruments ────────────────────────────────────────────

    def search_instruments(self, symbol: str) -> Dict[str, Any]:
        """POST /instruments/search"""
        self._ensure_auth()
        resp = self._session.post(
            f"{self.base_url}/instruments/search",
            json={"symbol": symbol},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        self._raise_for_status(resp, "search_instruments")
        return resp.json()

    # ── Internal helpers ───────────────────────────────────────

    @staticmethod
    def _raise_for_status(resp: requests.Response, context: str):
        if resp.ok:
            return
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        msg = f"DriveWealth {context} failed [{resp.status_code}]: {body}"
        logger.error(msg)
        raise DriveWealthError(msg, status_code=resp.status_code, body=body)
