"""
DriveWealth Execution Bridge — T2-5.

Bridges US Carver pipeline trade plans to DriveWealth API for execution.
DriveWealth supports fractional shares, pre-market/after-hours orders,
and zero-commission equity trading.

Integration:
  - Reads trade plans from `run_us_carver_pipeline()` output
  - Validates against risk limits (position size, daily loss)
  - Submits fractional share orders via DriveWealth REST API
  - Tracks fill status and computes slippage

API Docs: https://developer.drivewealth.com/
Auth: Bearer token from DriveWealth OAuth2 flow

NOTE: This is a template module. DriveWealth API credentials must
be configured via environment variables or config before live use:
  - DRIVEWEALTH_API_KEY
  - DRIVEWEALTH_API_SECRET
  - DRIVEWEALTH_ACCOUNT_NO
  - DRIVEWEALTH_BASE_URL (sandbox or prod)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Base URLs
DW_SANDBOX_URL = "https://bo-api.drivewealth.io/v1"
DW_PRODUCTION_URL = "https://bo-api.drivewealth.io/v1"


@dataclass
class DWConfig:
    """DriveWealth connection configuration."""
    api_key: str = ""
    api_secret: str = ""
    account_no: str = ""
    base_url: str = DW_SANDBOX_URL
    session_token: str = ""
    max_order_value_usd: float = 5000.0   # Max single order
    max_daily_orders: int = 50
    dry_run: bool = True  # Paper trading mode by default

    @classmethod
    def from_env(cls) -> "DWConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.environ.get("DRIVEWEALTH_API_KEY", ""),
            api_secret=os.environ.get("DRIVEWEALTH_API_SECRET", ""),
            account_no=os.environ.get("DRIVEWEALTH_ACCOUNT_NO", ""),
            base_url=os.environ.get("DRIVEWEALTH_BASE_URL", DW_SANDBOX_URL),
            dry_run=os.environ.get("DRIVEWEALTH_DRY_RUN", "true").lower() == "true",
        )


@dataclass
class DWOrderResult:
    """Result of a single order submission."""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    order_id: str = ""
    status: str = "PENDING"   # PENDING, FILLED, PARTIAL, REJECTED, DRY_RUN
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    slippage_bps: float = 0.0
    error: str = ""
    timestamp: str = ""


@dataclass
class DWExecutionResult:
    """Result of batch execution."""
    orders: List[DWOrderResult] = field(default_factory=list)
    total_value_usd: float = 0.0
    total_slippage_bps: float = 0.0
    log: List[str] = field(default_factory=list)


class DriveWealthBridge:
    """DriveWealth API execution bridge.

    Parameters
    ----------
    config : DWConfig
        API configuration. Use DWConfig.from_env() for env-based config.
    """

    def __init__(self, config: Optional[DWConfig] = None):
        self.config = config or DWConfig.from_env()
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._orders_today = 0
        self._daily_value = 0.0

    def authenticate(self) -> bool:
        """Authenticate with DriveWealth and obtain session token."""
        if self.config.dry_run:
            logger.info("DriveWealth: DRY RUN mode — skipping auth")
            return True

        if not self.config.api_key or not self.config.api_secret:
            logger.error("DriveWealth: API credentials not configured")
            return False

        try:
            resp = self._session.post(
                f"{self.config.base_url}/userSessions",
                json={
                    "appTypeID": "2000",
                    "appVersion": "1.0",
                    "username": self.config.api_key,
                    "password": self.config.api_secret,
                    "languageID": "en_US",
                    "osVersion": "Win10",
                    "osType": "Windows",
                    "scrRes": "1920x1080",
                    "ipAddress": "127.0.0.1",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self.config.session_token = data.get("sessionKey", "")
            self._session.headers["x-mysolomeo-session-key"] = self.config.session_token
            logger.info("DriveWealth: Authenticated successfully")
            return True
        except Exception as e:
            logger.error("DriveWealth auth failed: %s", e)
            return False

    def get_account_balance(self) -> Optional[Dict[str, float]]:
        """Get current account balances."""
        if self.config.dry_run:
            return {"cash": 10000.0, "equity": 10000.0, "buying_power": 10000.0}

        try:
            resp = self._session.get(
                f"{self.config.base_url}/users/{self.config.account_no}/accountSummary",
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "cash": float(data.get("cash", {}).get("cashAvailableForTrade", 0)),
                "equity": float(data.get("equity", {}).get("equityValue", 0)),
                "buying_power": float(data.get("cash", {}).get("cashBalance", 0)),
            }
        except Exception as e:
            logger.error("DriveWealth balance check failed: %s", e)
            return None

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        expected_price: float = 0.0,
        order_type: str = "MARKET",
    ) -> DWOrderResult:
        """Place a single order via DriveWealth API.

        Parameters
        ----------
        symbol : str
            US ticker (e.g., "AAPL").
        side : str
            "BUY" or "SELL".
        quantity : float
            Number of shares (can be fractional).
        expected_price : float
            Expected fill price for slippage calculation.
        order_type : str
            "MARKET" or "LIMIT".

        Returns
        -------
        DWOrderResult
        """
        result = DWOrderResult(
            symbol=symbol,
            side=side,
            quantity=quantity,
            timestamp=datetime.now().isoformat(),
        )

        # Safety checks
        if self._orders_today >= self.config.max_daily_orders:
            result.status = "REJECTED"
            result.error = f"Daily order limit reached ({self.config.max_daily_orders})"
            return result

        order_value = abs(quantity * expected_price)
        if order_value > self.config.max_order_value_usd:
            result.status = "REJECTED"
            result.error = f"Order value ${order_value:.0f} exceeds max ${self.config.max_order_value_usd:.0f}"
            return result

        if self.config.dry_run:
            # Simulate fill at expected price with tiny slippage
            import random
            slippage = random.uniform(0.0, 0.001)  # 0-10 bps
            fill_price = expected_price * (1 + slippage) if side == "BUY" else expected_price * (1 - slippage)

            result.order_id = f"DRY-{symbol}-{int(time.time())}"
            result.status = "DRY_RUN"
            result.fill_price = round(fill_price, 4)
            result.fill_quantity = quantity
            result.slippage_bps = round(slippage * 10000, 2)

            self._orders_today += 1
            self._daily_value += order_value
            logger.info(
                "DriveWealth DRY RUN: %s %s %.4f @ $%.2f (slippage: %.1f bps)",
                side, symbol, quantity, fill_price, result.slippage_bps,
            )
            return result

        # Live order placement
        try:
            order_payload = {
                "accountNo": self.config.account_no,
                "symbol": symbol,
                "orderType": "1" if order_type == "MARKET" else "2",
                "side": "B" if side == "BUY" else "S",
                "quantity": str(round(quantity, 8)),
            }

            resp = self._session.post(
                f"{self.config.base_url}/orders",
                json=order_payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            result.order_id = data.get("orderID", "")
            result.status = data.get("orderStatus", "PENDING")

            if result.status in ("FILLED", "PARTIAL_FILL"):
                result.fill_price = float(data.get("avgPrice", expected_price))
                result.fill_quantity = float(data.get("cumQty", 0))
                if expected_price > 0:
                    result.slippage_bps = round(
                        abs(result.fill_price - expected_price) / expected_price * 10000, 2
                    )

            self._orders_today += 1
            self._daily_value += order_value

            logger.info(
                "DriveWealth order placed: %s %s %.4f → %s (ID: %s)",
                side, symbol, quantity, result.status, result.order_id,
            )
        except Exception as e:
            result.status = "REJECTED"
            result.error = str(e)
            logger.error("DriveWealth order failed: %s %s → %s", side, symbol, e)

        return result

    def execute_trade_plans(
        self,
        trade_plans: List[Dict[str, Any]],
        capital: float = 10000.0,
    ) -> DWExecutionResult:
        """Execute batch of trade plans from US Carver pipeline.

        Parameters
        ----------
        trade_plans : list
            List of trade plan dicts from `run_us_carver_pipeline()`.
        capital : float
            Current capital for position sizing validation.

        Returns
        -------
        DWExecutionResult
        """
        exec_result = DWExecutionResult()

        if not trade_plans:
            exec_result.log.append("No trade plans to execute")
            return exec_result

        # Authenticate if needed
        if not self.config.session_token and not self.config.dry_run:
            if not self.authenticate():
                exec_result.log.append("Authentication failed — aborting execution")
                return exec_result

        # Check account balance
        balance = self.get_account_balance()
        if balance:
            available = balance.get("buying_power", capital)
            exec_result.log.append(f"Account balance: ${available:,.2f} available")
        else:
            available = capital

        # Sort by forecast strength (strongest first)
        plans = sorted(trade_plans, key=lambda p: abs(p.get("forecast", 0)), reverse=True)

        total_deployed = 0.0
        for plan in plans:
            symbol = plan.get("symbol", "")
            side = plan.get("side", "BUY")
            qty = plan.get("quantity", 0)
            price = plan.get("entry_price", 0)

            if not symbol or qty <= 0 or price <= 0:
                continue

            order_value = qty * price
            if total_deployed + order_value > available * 0.95:  # 95% max deploy
                exec_result.log.append(f"  Skip {symbol}: would exceed 95% deployment")
                continue

            order_result = self.place_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                expected_price=price,
            )
            exec_result.orders.append(order_result)

            if order_result.status in ("FILLED", "PARTIAL_FILL", "DRY_RUN"):
                total_deployed += order_value

        exec_result.total_value_usd = round(total_deployed, 2)

        # Aggregate slippage
        filled = [o for o in exec_result.orders if o.status in ("FILLED", "PARTIAL_FILL", "DRY_RUN")]
        if filled:
            exec_result.total_slippage_bps = round(
                sum(o.slippage_bps for o in filled) / len(filled), 2
            )

        exec_result.log.append(
            f"Execution complete: {len(filled)}/{len(plans)} filled, "
            f"${total_deployed:,.2f} deployed, avg slippage {exec_result.total_slippage_bps:.1f} bps"
        )

        for line in exec_result.log:
            logger.info("DW: %s", line)

        return exec_result
