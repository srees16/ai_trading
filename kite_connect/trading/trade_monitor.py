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

import json
import logging
import sqlite3
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Persistence path for trade monitor state
_MONITOR_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "trade_monitor_state.sqlite3",
)


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
        self._init_state_db()
        self._restore_state()

    # ------------------------------------------------------------------
    # State persistence (crash recovery)
    # ------------------------------------------------------------------

    def _init_state_db(self):
        """Create the SQLite state table if it doesn't exist."""
        try:
            os.makedirs(os.path.dirname(_MONITOR_DB), exist_ok=True)
            with sqlite3.connect(_MONITOR_DB) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS monitored_trades (
                        entry_order_id TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
        except Exception as e:
            logger.warning("TradeMonitor: state DB init failed (non-fatal): %s", e)

    def _persist_state(self):
        """Persist all trades to SQLite for crash recovery."""
        try:
            with sqlite3.connect(_MONITOR_DB) as conn:
                # Clear and re-insert (simple, atomic via WAL)
                conn.execute("DELETE FROM monitored_trades")
                now = datetime.now().isoformat()
                for oid, trade in self._trades.items():
                    d = asdict(trade)
                    d["opened_at"] = trade.opened_at.isoformat()
                    conn.execute(
                        "INSERT INTO monitored_trades (entry_order_id, state_json, updated_at) VALUES (?, ?, ?)",
                        (oid, json.dumps(d), now),
                    )
        except Exception as e:
            logger.warning("TradeMonitor: persist failed (non-fatal): %s", e)

    def _restore_state(self):
        """Restore active trades from SQLite on startup."""
        try:
            with sqlite3.connect(_MONITOR_DB) as conn:
                rows = conn.execute("SELECT entry_order_id, state_json FROM monitored_trades").fetchall()
            restored = 0
            for oid, state_json in rows:
                d = json.loads(state_json)
                d["opened_at"] = datetime.fromisoformat(d["opened_at"])
                trade = MonitoredTrade(**d)
                if not trade.closed:
                    self._trades[oid] = trade
                    restored += 1
            if restored:
                logger.info("TradeMonitor: restored %d active trades from crash-recovery DB", restored)
        except Exception as e:
            logger.warning("TradeMonitor: restore failed (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Trade registration & queries
    # ------------------------------------------------------------------

    def register_trade(self, trade: MonitoredTrade) -> None:
        """Register a new trade for monitoring."""
        self._trades[trade.entry_order_id] = trade
        self._persist_state()
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
                    # Now place SL and TP orders (only after entry fills)
                    self._place_sl_for_trade(trade)
                    self._place_tp_for_trade(trade)
                elif entry_order.get("status") in ("CANCELLED", "REJECTED"):
                    trade.closed = True
                    events.append({
                        "type": "ENTRY_REJECTED",
                        "symbol": trade.symbol,
                        "order_id": trade.entry_order_id,
                        "reason": entry_order.get("status_message", ""),
                    })
                else:
                    # M1 fix: cancel stale unfilled entries (>2 hours old)
                    age_minutes = (datetime.now() - trade.opened_at).total_seconds() / 60
                    if age_minutes > 120:
                        self._cancel_order(trade.entry_order_id, trade.symbol, "ENTRY")
                        trade.closed = True
                        events.append({
                            "type": "ENTRY_CANCELLED_STALE",
                            "symbol": trade.symbol,
                            "order_id": trade.entry_order_id,
                            "age_minutes": int(age_minutes),
                        })
                        logger.info(
                            "Stale entry cancelled: %s (%.0f min old)",
                            trade.symbol, age_minutes,
                        )
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

            # ── R3: Trailing stop-loss ─────────────────────────
            # If price has moved > activation_pct above entry,
            # ratchet the SL up to trail_pct below current price.
            if trade.entry_filled and not trade.closed:
                trail_event = self._maybe_trail_sl(trade)
                if trail_event:
                    events.append(trail_event)

        # ── Corporate action adjustment (#2) ──────────────────
        # Check for pending SPLIT/BONUS on active trades and adjust
        # position sizes, SL, and TP accordingly.
        ca_events = self._check_corporate_actions()
        events.extend(ca_events)

        # ── Desktop notifications for SL/TP events ────────────
        self._dispatch_notifications(events)

        # ── Persist state after all mutations ─────────────────
        if events:
            self._persist_state()

        return events

    @staticmethod
    def _dispatch_notifications(events: List[Dict]):
        """Send desktop notifications for important trade events."""
        if not events:
            return
        try:
            from notifications.manager import NotificationManager
            nm = NotificationManager()
            for ev in events:
                etype = ev.get("type", "")
                if etype in ("SL_TRIGGERED", "TP_FILLED", "TRAILING_SL_UPDATED"):
                    nm.notify_sl_tp_event(
                        etype,
                        ev.get("symbol", ""),
                        ev.get("exit_price", ev.get("new_sl", 0)),
                    )
        except Exception:
            pass  # notifications are non-critical

    def _place_sl_for_trade(self, trade: MonitoredTrade) -> None:
        """Place a stop-loss order after entry fills."""
        if not self.kite:
            return
        try:
            from kite_connect.trading.order_service import place_order
            import time
            sl_side = "SELL" if trade.side == "BUY" else "BUY"
            resp = place_order(
                kite=self.kite,
                symbol=trade.symbol,
                exchange="NSE",
                transaction_type=sl_side,
                quantity=trade.quantity,
                order_type="SL",
                product="CNC",
                trigger_price=trade.stop_loss,
                price=round(trade.stop_loss * 0.99, 2),  # SL limit 1% below trigger
            )
            time.sleep(0.15)
            if resp.get("success"):
                trade.sl_order_id = resp["order_id"]
                logger.info("SL placed for %s: trigger=%.2f, id=%s",
                            trade.symbol, trade.stop_loss, resp["order_id"])
            else:
                logger.error("SL FAILED for %s: %s", trade.symbol, resp.get("error"))
        except Exception as exc:
            logger.error("SL exception for %s: %s", trade.symbol, exc)

    def _place_tp_for_trade(self, trade: MonitoredTrade) -> None:
        """Place a take-profit order after entry fills."""
        if not self.kite:
            return
        try:
            from kite_connect.trading.order_service import place_order
            import time
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
            time.sleep(0.15)
            if resp.get("success"):
                trade.tp_order_id = resp["order_id"]
                logger.info("TP placed for %s: target=%.2f, id=%s",
                            trade.symbol, trade.target_price, resp["order_id"])
            else:
                logger.error("TP FAILED for %s: %s", trade.symbol, resp.get("error"))
        except Exception as exc:
            logger.error("TP exception for %s: %s", trade.symbol, exc)

    def _maybe_trail_sl(self, trade: MonitoredTrade) -> Optional[Dict]:
        """Ratchet the stop-loss upward when price moves in our favour.

        Activation: current price > entry * (1 + activation_pct)
        New SL:     max(existing SL, current_price * (1 - trail_distance_pct))
        """
        if not self.kite or not trade.sl_order_id:
            return None
        try:
            activation_pct = 0.05   # 5% profit triggers trailing
            trail_pct = 0.03        # trail 3% below current price

            key = f"NSE:{trade.symbol}"
            ltp_data = self.kite.ltp([key])
            ltp = ltp_data.get(key, {}).get("last_price", 0)
            if ltp <= 0:
                return None

            # Not enough profit yet to activate trailing
            profit_pct = (ltp - trade.entry_price) / trade.entry_price
            if profit_pct < activation_pct:
                return None

            new_sl = round(ltp * (1 - trail_pct), 2)
            # Only ratchet up, never down
            if new_sl <= trade.stop_loss:
                return None

            # Modify existing SL order
            from kite_connect.trading.order_service import modify_order
            resp = modify_order(
                self.kite, trade.sl_order_id,
                trigger_price=new_sl,
                price=round(new_sl * 0.99, 2),
            )
            if resp.get("success"):
                old_sl = trade.stop_loss
                trade.stop_loss = new_sl
                logger.info(
                    "Trailing SL: %s ratcheted %.2f → %.2f (LTP=%.2f, profit=%.1f%%)",
                    trade.symbol, old_sl, new_sl, ltp, profit_pct * 100,
                )
                return {
                    "type": "TRAILING_SL_UPDATED",
                    "symbol": trade.symbol,
                    "old_sl": old_sl,
                    "new_sl": new_sl,
                    "ltp": ltp,
                    "profit_pct": round(profit_pct * 100, 1),
                }
        except Exception as exc:
            logger.debug("Trailing SL check failed for %s: %s", trade.symbol, exc)
        return None

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

    def _check_corporate_actions(self) -> List[Dict]:
        """Check for pending SPLIT/BONUS corporate actions on active trades.

        When a split or bonus is detected for a monitored position,
        the local trade record's quantity, entry_price, SL, and TP are
        adjusted and existing SL/TP Kite orders are cancelled and
        re-placed at the adjusted prices.
        """
        events: List[Dict] = []
        active = [t for t in self._trades.values() if t.is_active]
        if not active:
            return events

        try:
            from services.corporate_actions import get_actions_for_symbols, adjust_position
            symbols = [t.symbol for t in active]
            pending = get_actions_for_symbols(symbols)
            if not pending:
                return events

            for trade in active:
                action = pending.get(trade.symbol)
                if action is None:
                    continue

                adj = adjust_position(
                    qty=trade.quantity,
                    entry_price=trade.entry_price,
                    stop_loss=trade.stop_loss,
                    target_price=trade.target_price,
                    action=action,
                )

                old_qty = trade.quantity
                trade.quantity = adj["quantity"]
                trade.entry_price = adj["entry_price"]
                trade.stop_loss = adj["stop_loss"]
                trade.target_price = adj["target_price"]

                logger.info(
                    "Corporate action %s on %s: qty %d→%d, entry %.2f→%.2f",
                    action.action_type, trade.symbol,
                    old_qty, trade.quantity,
                    trade.entry_price, adj["entry_price"],
                )

                # Cancel and re-place SL/TP at adjusted prices
                self._cancel_order(trade.sl_order_id, trade.symbol, "SL")
                self._cancel_order(trade.tp_order_id, trade.symbol, "TP")
                trade.sl_order_id = None
                trade.tp_order_id = None
                self._place_sl_for_trade(trade)
                self._place_tp_for_trade(trade)

                events.append({
                    "type": "CORPORATE_ACTION_ADJUSTED",
                    "symbol": trade.symbol,
                    "action_type": action.action_type,
                    "old_qty": old_qty,
                    "new_qty": trade.quantity,
                    "description": action.description[:80],
                })

        except Exception as exc:
            logger.debug("Corporate action check failed (non-fatal): %s", exc)

        return events

    def summary(self) -> Dict:
        """Return a summary of monitored trades."""
        active = self.active_trades
        return {
            "total_registered": len(self._trades),
            "active": len(active),
            "closed": len(self._trades) - len(active),
            "symbols": [t.symbol for t in active],
        }
