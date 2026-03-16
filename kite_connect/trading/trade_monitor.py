"""
Post-Trade Monitoring Service for Zerodha Kite Connect.

Monitors open positions and pending orders during market hours:
- Polls order book for SL/TP fill status
- Re-places TP orders that expire (DAY validity → new day)
- Cancels orphaned SL when TP fills, and vice versa
- Emits events for filled orders (integrates with EventBus)

Designed for swing / long-term trades (CNC product).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MonitoredTrade:
    """Represents an active trade being monitored."""
    symbol: str
    side: str
    quantity: int
    entry_price: float
    stop_loss: float
    target_price: float
    entry_order_id: str
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    entry_filled: bool = False
    sl_triggered: bool = False
    tp_triggered: bool = False
    closed: bool = False
    opened_at: datetime = field(default_factory=datetime.now)

    @property
    def is_active(self) -> bool:
        return self.entry_filled and not self.closed


class TradeMonitor:
    """
    Polls Kite order book and manages SL/TP lifecycle.

    Usage::

        monitor = TradeMonitor(kite)
        monitor.register_trade(trade_info)
        # Call periodically during market hours:
        events = monitor.poll()
    """

    def __init__(self, kite=None):
        self.kite = kite
        self._trades: Dict[str, MonitoredTrade] = {}  # keyed by entry_order_id

    def register_trade(self, trade: MonitoredTrade) -> None:
        """Register a new trade for monitoring."""
        self._trades[trade.entry_order_id] = trade
        logger.info(
            "TradeMonitor: registered %s %s qty=%d entry=%.2f SL=%.2f TP=%.2f",
            trade.side, trade.symbol, trade.quantity,
            trade.entry_price, trade.stop_loss, trade.target_price,
        )

    @property
    def active_trades(self) -> List[MonitoredTrade]:
        return [t for t in self._trades.values() if t.is_active]

    @property
    def all_trades(self) -> List[MonitoredTrade]:
        return list(self._trades.values())

    def poll(self) -> List[Dict]:
        """
        Poll the order book and update trade statuses.

        Returns a list of event dicts for any state changes::

            [
                {"type": "ENTRY_FILLED", "symbol": "RELIANCE", ...},
                {"type": "SL_TRIGGERED", "symbol": "TCS", ...},
                {"type": "TP_FILLED", "symbol": "INFY", ...},
            ]
        """
        if self.kite is None:
            return []

        events: List[Dict] = []

        try:
            order_book = self.kite.orders() or []
        except Exception as exc:
            logger.warning("TradeMonitor: failed to fetch order book — %s", exc)
            return events

        # Build lookup: order_id → order status
        order_map: Dict[str, dict] = {}
        for order in order_book:
            oid = str(order.get("order_id", ""))
            if oid:
                order_map[oid] = order

        for trade in self._trades.values():
            if trade.closed:
                continue

            # Check entry fill
            if not trade.entry_filled:
                entry_order = order_map.get(trade.entry_order_id, {})
                if entry_order.get("status") == "COMPLETE":
                    trade.entry_filled = True
                    fill_price = entry_order.get("average_price", trade.entry_price)
                    events.append({
                        "type": "ENTRY_FILLED",
                        "symbol": trade.symbol,
                        "fill_price": fill_price,
                        "quantity": trade.quantity,
                        "order_id": trade.entry_order_id,
                    })
                    logger.info("Entry FILLED: %s @ %.2f", trade.symbol, fill_price)
                elif entry_order.get("status") in ("CANCELLED", "REJECTED"):
                    trade.closed = True
                    events.append({
                        "type": "ENTRY_REJECTED",
                        "symbol": trade.symbol,
                        "order_id": trade.entry_order_id,
                        "reason": entry_order.get("status_message", ""),
                    })
                continue  # Wait for entry to fill before checking SL/TP

            # Check SL
            if trade.sl_order_id and not trade.sl_triggered:
                sl_order = order_map.get(trade.sl_order_id, {})
                if sl_order.get("status") == "COMPLETE":
                    trade.sl_triggered = True
                    trade.closed = True
                    exit_price = sl_order.get("average_price", trade.stop_loss)
                    events.append({
                        "type": "SL_TRIGGERED",
                        "symbol": trade.symbol,
                        "exit_price": exit_price,
                        "order_id": trade.sl_order_id,
                    })
                    logger.info("SL triggered: %s @ %.2f", trade.symbol, exit_price)
                    # Cancel the orphaned TP order
                    self._cancel_order(trade.tp_order_id, trade.symbol, "TP")

            # Check TP
            if trade.tp_order_id and not trade.tp_triggered:
                tp_order = order_map.get(trade.tp_order_id, {})
                if tp_order.get("status") == "COMPLETE":
                    trade.tp_triggered = True
                    trade.closed = True
                    exit_price = tp_order.get("average_price", trade.target_price)
                    events.append({
                        "type": "TP_FILLED",
                        "symbol": trade.symbol,
                        "exit_price": exit_price,
                        "order_id": trade.tp_order_id,
                    })
                    logger.info("TP filled: %s @ %.2f", trade.symbol, exit_price)
                    # Cancel the orphaned SL order
                    self._cancel_order(trade.sl_order_id, trade.symbol, "SL")

                elif tp_order.get("status") in ("CANCELLED", "REJECTED"):
                    # TP expired (DAY validity) — re-place it
                    new_tp_id = self._replace_tp(trade)
                    if new_tp_id:
                        trade.tp_order_id = new_tp_id
                        events.append({
                            "type": "TP_REPLACED",
                            "symbol": trade.symbol,
                            "new_order_id": new_tp_id,
                        })

        return events

    def _cancel_order(self, order_id: Optional[str], symbol: str, label: str):
        """Cancel an orphaned order (SL when TP fills, or vice versa)."""
        if not order_id or not self.kite:
            return
        try:
            from kite_connect.trading.order_service import cancel_order
            cancel_order(self.kite, order_id)
            logger.info("Cancelled orphaned %s order %s for %s", label, order_id, symbol)
        except Exception as exc:
            logger.warning("Failed to cancel %s order %s: %s", label, order_id, exc)

    def _replace_tp(self, trade: MonitoredTrade) -> Optional[str]:
        """Re-place an expired TP order for the next trading day."""
        if not self.kite:
            return None
        try:
            from kite_connect.trading.order_service import place_order
            tp_side = "SELL" if trade.side == "BUY" else "BUY"
            resp = place_order(
                kite=self.kite,
                symbol=trade.symbol,
                exchange="NSE",
                transaction_type=tp_side,
                quantity=trade.quantity,
                order_type="LIMIT",
                product="CNC",
                price=trade.target_price,
            )
            if resp.get("success"):
                logger.info(
                    "TP re-placed for %s: target=%.2f, new_id=%s",
                    trade.symbol, trade.target_price, resp["order_id"],
                )
                return resp["order_id"]
            else:
                logger.error("TP re-place FAILED for %s: %s", trade.symbol, resp.get("error"))
        except Exception as exc:
            logger.error("TP re-place exception for %s: %s", trade.symbol, exc)
        return None

    def summary(self) -> Dict:
        """Return a summary of monitored trades."""
        active = self.active_trades
        return {
            "total_registered": len(self._trades),
            "active": len(active),
            "closed": len(self._trades) - len(active),
            "symbols": [t.symbol for t in active],
        }
