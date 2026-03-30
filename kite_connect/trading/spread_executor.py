"""
Spread Executor — Synchronized multi-leg order placement.

Handles:
  - Pairs trading (long leg + short leg simultaneously)
  - Options spreads (buy + sell legs)
  - Abort mechanism: if one leg fills and other doesn't within timeout,
    close the filled leg to avoid naked exposure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class LegOrder:
    """A single leg of a spread/pair trade."""
    symbol: str
    exchange: str               # "NSE" or "NFO"
    transaction_type: str       # "BUY" or "SELL"
    quantity: int
    product: str = "CNC"        # CNC, NRML, MIS
    order_type: str = "LIMIT"
    price: Optional[float] = None
    tag: str = ""


@dataclass
class SpreadResult:
    """Result of a spread execution attempt."""
    success: bool = False
    leg1_order_id: Optional[str] = None
    leg2_order_id: Optional[str] = None
    leg1_filled: bool = False
    leg2_filled: bool = False
    aborted: bool = False
    abort_order_id: Optional[str] = None
    error: str = ""


class SpreadExecutor:
    """Execute synchronized 2-leg trades with abort safety."""

    def __init__(self, kite, fill_timeout_seconds: int = 30):
        self.kite = kite
        self.fill_timeout = fill_timeout_seconds

    def execute_pair(self, leg1: LegOrder, leg2: LegOrder) -> SpreadResult:
        """Execute a 2-leg spread trade.

        Places both legs, then monitors for fills. If one fills and the
        other doesn't within timeout, aborts by closing the filled leg.
        """
        from kite_connect.trading.order_service import place_order

        result = SpreadResult()

        # Place leg 1
        r1 = place_order(
            self.kite, symbol=leg1.symbol, exchange=leg1.exchange,
            transaction_type=leg1.transaction_type, quantity=leg1.quantity,
            order_type=leg1.order_type, product=leg1.product,
            price=leg1.price,
        )
        if not r1.get("success"):
            result.error = f"Leg 1 failed: {r1.get('error')}"
            return result
        result.leg1_order_id = r1.get("order_id")

        # Place leg 2
        r2 = place_order(
            self.kite, symbol=leg2.symbol, exchange=leg2.exchange,
            transaction_type=leg2.transaction_type, quantity=leg2.quantity,
            order_type=leg2.order_type, product=leg2.product,
            price=leg2.price,
        )
        if not r2.get("success"):
            # Leg 2 failed — cancel leg 1
            result.error = f"Leg 2 failed: {r2.get('error')}"
            self._cancel_or_close(result.leg1_order_id, leg1)
            result.aborted = True
            return result
        result.leg2_order_id = r2.get("order_id")

        # Wait for fills
        deadline = time.time() + self.fill_timeout
        while time.time() < deadline:
            l1_status = self._check_fill(result.leg1_order_id)
            l2_status = self._check_fill(result.leg2_order_id)
            result.leg1_filled = l1_status == "COMPLETE"
            result.leg2_filled = l2_status == "COMPLETE"

            if result.leg1_filled and result.leg2_filled:
                result.success = True
                return result

            # If either is rejected/cancelled, abort
            if l1_status in ("REJECTED", "CANCELLED") or l2_status in ("REJECTED", "CANCELLED"):
                break

            time.sleep(2)

        # Timeout or rejection — abort unfilled legs
        if result.leg1_filled and not result.leg2_filled:
            logger.warning("Spread abort: Leg 2 unfilled — closing leg 1")
            self._cancel_or_close(result.leg2_order_id, leg2)
            abort_result = self._reverse_leg(leg1)
            result.abort_order_id = abort_result
            result.aborted = True
        elif result.leg2_filled and not result.leg1_filled:
            logger.warning("Spread abort: Leg 1 unfilled — closing leg 2")
            self._cancel_or_close(result.leg1_order_id, leg1)
            abort_result = self._reverse_leg(leg2)
            result.abort_order_id = abort_result
            result.aborted = True
        else:
            # Neither filled — cancel both
            self._cancel_or_close(result.leg1_order_id, leg1)
            self._cancel_or_close(result.leg2_order_id, leg2)
            result.error = "Both legs timed out"

        return result

    def _check_fill(self, order_id: str) -> str:
        """Check order status. Returns COMPLETE, OPEN, REJECTED, CANCELLED, etc."""
        if not order_id:
            return "UNKNOWN"
        try:
            history = self.kite.order_history(order_id)
            if history:
                return history[-1].get("status", "UNKNOWN")
        except Exception:
            pass
        return "UNKNOWN"

    def _cancel_or_close(self, order_id: str, leg: LegOrder):
        """Try to cancel an unfilled order."""
        if not order_id:
            return
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            logger.info("Cancelled order %s for %s", order_id, leg.symbol)
        except Exception as exc:
            logger.warning("Cancel failed for %s: %s", order_id, exc)

    def _reverse_leg(self, leg: LegOrder) -> Optional[str]:
        """Place a reverse order to close a filled leg."""
        from kite_connect.trading.order_service import place_order

        reverse_side = "SELL" if leg.transaction_type == "BUY" else "BUY"
        result = place_order(
            self.kite, symbol=leg.symbol, exchange=leg.exchange,
            transaction_type=reverse_side, quantity=leg.quantity,
            order_type="MARKET", product=leg.product,
        )
        if result.get("success"):
            return result.get("order_id")
        logger.error("Reverse leg failed for %s: %s", leg.symbol, result.get("error"))
        return None
