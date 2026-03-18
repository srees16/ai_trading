"""
Execution Engine Layer — Order routing & fill management.

Supports dual-mode execution:
  - live: routes to Kite Connect (IND) or DriveWealth (US)
  - paper/backtest: simulated fills via PaperBroker

Emits:
  - ``execution.order_placed``
  - ``execution.order_filled``
  - ``execution.order_rejected``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderIntent:
    ticker: str
    side: str  # BUY | SELL
    quantity: int
    order_type: str = "LIMIT"  # LIMIT | MARKET | SL-M
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    product: str = "CNC"  # CNC | MIS | NRML
    exchange: str = "NSE"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    order_id: str
    ticker: str
    status: str  # PLACED | FILLED | REJECTED
    fill_price: Optional[float] = None
    fill_quantity: int = 0
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionEngine:
    """
    Routes orders through the appropriate broker based on execution mode.
    """

    def __init__(self, market: str = "IND", kite=None):
        self.market = market
        self._kite = kite

    def execute(self, intent: OrderIntent) -> OrderResult:
        """Place an order (live or paper, based on execution context)."""
        from infrastructure.execution_context import execution_ctx
        from infrastructure.event_bus import event_bus
        from infrastructure.latency_tracker import latency_tracker

        with latency_tracker.measure("execution.place_order"):
            if execution_ctx.is_live:
                result = self._live_execute(intent)
            else:
                result = self._paper_execute(intent)

        event_bus.emit(
            f"execution.order_{result.status.lower()}",
            payload={
                "order_id": result.order_id,
                "ticker": intent.ticker,
                "side": intent.side,
                "quantity": intent.quantity,
                "status": result.status,
                "fill_price": result.fill_price,
                "mode": execution_ctx.mode,
            },
            source="execution_engine",
        )
        return result

    def execute_batch(self, intents: List[OrderIntent]) -> List[OrderResult]:
        """Execute multiple orders sequentially."""
        return [self.execute(intent) for intent in intents]

    def _live_execute(self, intent: OrderIntent) -> OrderResult:
        """Route to the real broker."""
        if self.market == "IND":
            return self._kite_execute(intent)
        else:
            # US market — placeholder for DriveWealth / other broker
            logger.warning("US live execution not configured — using paper")
            return self._paper_execute(intent)

    def _kite_execute(self, intent: OrderIntent) -> OrderResult:
        """Place order via Kite Connect."""
        try:
            from kite_connect.trading.order_service import OrderService

            kite = self._kite
            if not kite:
                return OrderResult(
                    order_id="", ticker=intent.ticker,
                    status="REJECTED", message="Kite session not available",
                )

            svc = OrderService(kite)
            order_id = svc.place_order(
                symbol=intent.ticker,
                exchange=intent.exchange,
                transaction_type=intent.side,
                quantity=intent.quantity,
                order_type=intent.order_type,
                product=intent.product,
                price=intent.price,
                trigger_price=intent.trigger_price,
            )
            return OrderResult(
                order_id=str(order_id), ticker=intent.ticker,
                status="PLACED", fill_quantity=intent.quantity,
                fill_price=intent.price,
            )
        except Exception as exc:
            logger.error("Kite order failed for %s: %s", intent.ticker, exc)
            return OrderResult(
                order_id="", ticker=intent.ticker,
                status="REJECTED", message=str(exc),
            )

    def _paper_execute(self, intent: OrderIntent) -> OrderResult:
        """Simulate a fill at the intent price."""
        import uuid
        return OrderResult(
            order_id=f"PAPER-{uuid.uuid4().hex[:8]}",
            ticker=intent.ticker,
            status="FILLED",
            fill_price=intent.price or 0.0,
            fill_quantity=intent.quantity,
            message="Paper trade fill",
        )
