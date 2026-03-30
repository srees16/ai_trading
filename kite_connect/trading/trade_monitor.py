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
    scaled_2r: bool = False    # G5: persisted scale-out state
    scaled_3r: bool = False    # G5: persisted scale-out state
    opened_at: datetime = field(default_factory=datetime.now)
    direction: str = "LONG"    # "LONG" or "SHORT" (Phase 2)

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

    def __init__(self, kite=None, risk_config=None):
        self.kite = kite
        self._risk_config = risk_config  # RiskConfig for trailing SL params
        self._trades: Dict[str, MonitoredTrade] = {}  # keyed by entry_order_id
        self._daily_sl_count: int = 0     # G14: SL hits today
        self._daily_sl_date: Optional[str] = None  # G14: date tracker
        self._halted: bool = False         # G14: True when max-loss-per-day breached
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

        # G14: Reset daily SL counter on new trading day
        today = datetime.now().strftime("%Y-%m-%d")
        if self._daily_sl_date != today:
            self._daily_sl_date = today
            self._daily_sl_count = 0
            self._halted = False

        # G14: If 3+ SL hits today, halt all new monitoring (existing SL/TP still fire)
        if self._halted:
            logger.warning("TradeMonitor: HALTED — %d SL hits today, skipping poll", self._daily_sl_count)
            return [{"type": "DAILY_HALT", "sl_count": self._daily_sl_count}]

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
                    # Check for partial fills
                    filled_qty = int(entry_order.get("filled_quantity", 0))
                    pending_qty = int(entry_order.get("pending_quantity", trade.quantity))

                    # M1 fix: cancel stale unfilled entries (>2 hours old)
                    age_minutes = (datetime.now() - trade.opened_at).total_seconds() / 60
                    if age_minutes > 120:
                        self._cancel_order(trade.entry_order_id, trade.symbol, "ENTRY")
                        if filled_qty > 0:
                            # Partial fill: accept what we got, adjust qty and proceed
                            trade.quantity = filled_qty
                            trade.entry_filled = True
                            fill_price = entry_order.get("average_price", trade.entry_price)
                            events.append({
                                "type": "ENTRY_PARTIAL_ACCEPTED",
                                "symbol": trade.symbol,
                                "filled_qty": filled_qty,
                                "original_qty": filled_qty + pending_qty,
                                "fill_price": fill_price,
                                "order_id": trade.entry_order_id,
                            })
                            logger.info(
                                "Partial fill accepted: %s %d/%d @ %.2f (stale cancelled)",
                                trade.symbol, filled_qty, filled_qty + pending_qty, fill_price,
                            )
                            self._place_sl_for_trade(trade)
                            self._place_tp_for_trade(trade)
                        else:
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
                    realized_pnl = (exit_price - trade.entry_price) * trade.quantity
                    events.append({
                        "type": "SL_TRIGGERED",
                        "symbol": trade.symbol,
                        "exit_price": exit_price,
                        "order_id": trade.sl_order_id,
                        "realized_pnl": round(realized_pnl, 2),
                    })
                    logger.info("SL triggered: %s @ %.2f (P&L: %.2f)", trade.symbol, exit_price, realized_pnl)
                    self._roll_capital(realized_pnl)
                    # G14: Track daily SL hits — halt at 3
                    self._daily_sl_count += 1
                    if self._daily_sl_count >= 3:
                        self._halted = True
                        logger.warning("G14: 3 SL hits today — HALTING new trades")
                        events.append({"type": "DAILY_HALT", "sl_count": self._daily_sl_count})
                    # Cancel the orphaned TP order
                    self._cancel_order(trade.tp_order_id, trade.symbol, "TP")

            # Check TP
            if trade.tp_order_id and not trade.tp_triggered:
                tp_order = order_map.get(trade.tp_order_id, {})
                if tp_order.get("status") == "COMPLETE":
                    trade.tp_triggered = True
                    trade.closed = True
                    exit_price = tp_order.get("average_price", trade.target_price)
                    realized_pnl = (exit_price - trade.entry_price) * trade.quantity
                    events.append({
                        "type": "TP_FILLED",
                        "symbol": trade.symbol,
                        "exit_price": exit_price,
                        "order_id": trade.tp_order_id,
                        "realized_pnl": round(realized_pnl, 2),
                    })
                    logger.info("TP filled: %s @ %.2f (P&L: %.2f)", trade.symbol, exit_price, realized_pnl)
                    self._roll_capital(realized_pnl)
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

            # ── Gap C7: Scale-out at 2R and 3R targets ─────────
            # Sell partial position (33% at 2R, 33% at 3R) to lock profits
            if trade.entry_filled and not trade.closed:
                scaleout_event = self._check_scale_out(trade)
                if scaleout_event:
                    events.append(scaleout_event)

            # ── P0 fix: Time-based forced exit ─────────────────
            # Force close positions held beyond max hold days
            if trade.entry_filled and not trade.closed:
                exit_event = self._check_hold_time_exit(trade)
                if exit_event:
                    events.append(exit_event)

        # ── Corporate action adjustment (#2) ──────────────────
        # Check for pending SPLIT/BONUS on active trades and adjust
        # position sizes, SL, and TP accordingly.
        ca_events = self._check_corporate_actions()
        events.extend(ca_events)

        # ── Desktop notifications for SL/TP events ────────────
        self._dispatch_notifications(events)

        # ── G15: Sync unrealized P&L to vol target ──────────
        self._sync_vol_target_unrealized()

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
            from services.notifications.manager import NotificationManager
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
        """Place a stop-loss order after entry fills.

        Gap D fix: retries up to 3 times with 1s backoff if SL placement
        fails, to prevent orphaned positions running without a stop-loss.
        """
        if not self.kite:
            return
        import time as _time
        from kite_connect.trading.order_service import place_order
        sl_side = "SELL" if trade.side == "BUY" else "BUY"

        _MAX_SL_RETRIES = 3
        for attempt in range(_MAX_SL_RETRIES):
            try:
                resp = place_order(
                    kite=self.kite,
                    symbol=trade.symbol,
                    exchange="NSE",
                    transaction_type=sl_side,
                    quantity=trade.quantity,
                    order_type="SL",
                    product="CNC",
                    trigger_price=trade.stop_loss,
                    price=round(trade.stop_loss * 0.99, 2),
                )
                _time.sleep(0.15)
                if resp.get("success"):
                    trade.sl_order_id = resp["order_id"]
                    logger.info("SL placed for %s: trigger=%.2f, id=%s",
                                trade.symbol, trade.stop_loss, resp["order_id"])
                    return  # success — exit retry loop
                else:
                    logger.warning(
                        "SL attempt %d/%d failed for %s: %s",
                        attempt + 1, _MAX_SL_RETRIES,
                        trade.symbol, resp.get("error"),
                    )
            except Exception as exc:
                logger.warning(
                    "SL attempt %d/%d exception for %s: %s",
                    attempt + 1, _MAX_SL_RETRIES, trade.symbol, exc,
                )
            if attempt < _MAX_SL_RETRIES - 1:
                _time.sleep(1)  # 1s backoff between retries

        # All retries exhausted — EMERGENCY: place MARKET sell to close unprotected position
        logger.error(
            "CRITICAL: SL placement FAILED after %d retries for %s — "
            "placing emergency MARKET SELL to protect capital",
            _MAX_SL_RETRIES, trade.symbol,
        )
        try:
            emergency_resp = place_order(
                kite=self.kite,
                symbol=trade.symbol,
                exchange="NSE",
                transaction_type="SELL" if trade.side == "BUY" else "BUY",
                quantity=trade.quantity,
                order_type="MARKET",
                product="CNC",
            )
            if emergency_resp.get("success"):
                trade.closed = True
                logger.warning(
                    "Emergency MARKET SELL executed for %s qty=%d — position closed",
                    trade.symbol, trade.quantity,
                )
            else:
                logger.error(
                    "Emergency MARKET SELL also FAILED for %s: %s — POSITION UNPROTECTED",
                    trade.symbol, emergency_resp.get("error"),
                )
        except Exception as em_exc:
            logger.error("Emergency MARKET SELL exception for %s: %s", trade.symbol, em_exc)

        try:
            from services.notifications.manager import NotificationManager
            NotificationManager().notify_critical(
                f"SL FAILED for {trade.symbol} after {_MAX_SL_RETRIES} retries — "
                f"emergency sell {'executed' if trade.closed else 'ALSO FAILED'}"
            )
        except Exception:
            pass

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
        """Ratchet the stop-loss using Carver vol-based adaptive trailing.

        For LONG: ratchets UP (new_sl > old_sl).
        For SHORT: ratchets DOWN (new_sl < old_sl).

        Uses vol_trailing_stop.compute_trailing_stop() which:
          - Scales stop distance by daily volatility (2.5σ swing, 3.5σ positional)
          - Activates profit-lock after 4σ gain (tightens to 1.5σ)
          - Guarantees break-even once profit-lock activates
          - Clamps stop between 2% (min) and 12% (max) of peak

        Falls back to simple percentage trail if vol computation unavailable.
        """
        if not self.kite or not trade.sl_order_id:
            return None

        is_short = getattr(trade, "direction", "LONG") == "SHORT"

        try:
            key = f"NSE:{trade.symbol}"
            ltp_data = self.kite.ltp([key])
            ltp = ltp_data.get(key, {}).get("last_price", 0)
            if ltp <= 0:
                return None

            # Attempt Carver vol-based trailing stop
            try:
                from services.vol_trailing_stop import compute_trailing_stop
                from services.instrument_volatility import daily_price_volatility
                from utils import download_ind_ohlcv
                from config import Config

                # Fetch recent close prices for vol computation
                df = download_ind_ohlcv(trade.symbol, period="3mo")
                if df is not None and len(df) >= 20:
                    close_series = df["Close"] if "Close" in df.columns else df["close"]
                    daily_vol = daily_price_volatility(close_series)
                else:
                    daily_vol = 0.02  # 2% fallback

                trade_horizon = getattr(Config, "CARVER_TRADE_HORIZON", "swing")

                if is_short:
                    # SHORT: track trough (lowest price since entry)
                    trough_price = min(getattr(trade, "_trough_price", trade.entry_price), ltp)
                    trade._trough_price = trough_price

                    # Mirror: compute as if LONG from trough, then flip
                    state = compute_trailing_stop(
                        entry_price=trade.entry_price,
                        current_price=trade.entry_price,  # dummy
                        peak_price=trade.entry_price,      # dummy
                        daily_price_vol=daily_vol,
                        trade_horizon=trade_horizon,
                    )
                    # For SHORT stop: entry + stop_distance, trail downward from there
                    stop_dist = trade.entry_price - state.current_stop
                    new_sl = trough_price + stop_dist
                else:
                    peak_price = max(getattr(trade, "_peak_price", trade.entry_price), ltp)
                    trade._peak_price = peak_price

                    state = compute_trailing_stop(
                        entry_price=trade.entry_price,
                        current_price=ltp,
                        peak_price=peak_price,
                        daily_price_vol=daily_vol,
                        previous_stop=trade.stop_loss,
                        trade_horizon=trade_horizon,
                    )
                    new_sl = state.current_stop
            except Exception:
                # Fallback: simple percentage trail
                cfg = self._risk_config
                activation_pct = getattr(cfg, "trailing_sl_activation_pct", 0.05) if cfg else 0.05
                trail_pct = getattr(cfg, "trailing_sl_distance_pct", 0.03) if cfg else 0.03

                if is_short:
                    profit_pct = (trade.entry_price - ltp) / trade.entry_price
                    if profit_pct < activation_pct:
                        return None
                    new_sl = round(ltp * (1 + trail_pct), 2)
                else:
                    profit_pct = (ltp - trade.entry_price) / trade.entry_price
                    if profit_pct < activation_pct:
                        return None
                    new_sl = round(ltp * (1 - trail_pct), 2)

            # Direction-aware ratchet: LONG ratchets up, SHORT ratchets down
            if is_short:
                if new_sl >= trade.stop_loss:
                    return None  # SHORT: only ratchet DOWN
            else:
                if new_sl <= trade.stop_loss:
                    return None  # LONG: only ratchet UP

            new_sl = round(new_sl, 2)

            # Modify existing SL order on exchange
            from kite_connect.trading.order_service import modify_order
            resp = modify_order(
                self.kite, trade.sl_order_id,
                trigger_price=new_sl,
                price=round(new_sl * (1.01 if is_short else 0.99), 2),
            )
            if resp.get("success"):
                old_sl = trade.stop_loss
                trade.stop_loss = new_sl
                if is_short:
                    profit_pct = (trade.entry_price - ltp) / trade.entry_price
                else:
                    profit_pct = (ltp - trade.entry_price) / trade.entry_price
                logger.info(
                    "Trailing SL (vol-based): %s [%s] ratcheted %.2f → %.2f (LTP=%.2f, profit=%.1f%%)",
                    trade.symbol, "SHORT" if is_short else "LONG",
                    old_sl, new_sl, ltp, profit_pct * 100,
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

    def _check_hold_time_exit(self, trade: MonitoredTrade) -> Optional[Dict]:
        """Gap C5: Force exit positions held beyond max hold days.

        Uses Config values: MAX_HOLD_DAYS_SWING=15, MAX_HOLD_DAYS_POSITIONAL=60.
        Places a MARKET sell order and cancels existing SL/TP orders.
        """
        if not self.kite or trade.closed:
            return None
        try:
            from config import Config
            hold_days = (datetime.now() - trade.opened_at).days
            horizon = getattr(Config, "CARVER_TRADE_HORIZON", "swing")
            max_days = getattr(Config, "MAX_HOLD_DAYS_POSITIONAL", 60) if horizon == "positional" else getattr(Config, "MAX_HOLD_DAYS_SWING", 15)

            if hold_days < max_days:
                return None

            # Force market exit
            from kite_connect.trading.order_service import place_order
            exit_side = "SELL" if trade.side == "BUY" else "BUY"
            resp = place_order(
                kite=self.kite,
                symbol=trade.symbol,
                exchange="NSE",
                transaction_type=exit_side,
                quantity=trade.quantity,
                order_type="MARKET",
                product="CNC",
            )

            # Cancel existing SL and TP
            self._cancel_order(trade.sl_order_id, trade.symbol, "SL")
            self._cancel_order(trade.tp_order_id, trade.symbol, "TP")
            trade.closed = True

            logger.info(
                "TIME EXIT: %s held %d days > %d max (%s) — forced market sell",
                trade.symbol, hold_days, max_days, horizon,
            )
            return {
                "type": "TIME_EXIT_FORCED",
                "symbol": trade.symbol,
                "hold_days": hold_days,
                "max_days": max_days,
                "horizon": horizon,
                "exit_order_id": resp.get("order_id") if resp.get("success") else None,
            }
        except Exception as exc:
            logger.warning("Hold time exit check failed for %s: %s", trade.symbol, exc)
        return None

    def _check_scale_out(self, trade: MonitoredTrade) -> Optional[Dict]:
        """Gap C7: Partial exit (scale-out) at 2R and 3R targets.

        Sells 33% of position at 2R, another 33% at 3R.
        Tracks scale-out state via trade attributes.
        """
        if not self.kite or trade.closed or trade.quantity <= 1:
            return None
        try:
            # Get current price
            key = f"NSE:{trade.symbol}"
            ltp_data = self.kite.ltp([key])
            current_price = ltp_data.get(key, {}).get("last_price", 0)
            if current_price <= 0:
                return None

            # Risk distance (1R)
            risk_1r = trade.entry_price - trade.stop_loss
            if risk_1r <= 0:
                return None

            # Current R-multiple
            profit = current_price - trade.entry_price
            r_multiple = profit / risk_1r

            # Track scale-out state (G5: persisted via dataclass fields)
            scaled_at_2r = trade.scaled_2r
            scaled_at_3r = trade.scaled_3r

            sell_qty = 0
            r_target = 0

            if r_multiple >= 3.0 and not scaled_at_3r:
                sell_qty = max(1, trade.quantity // 3)
                r_target = 3
                trade.scaled_3r = True
            elif r_multiple >= 2.0 and not scaled_at_2r:
                sell_qty = max(1, trade.quantity // 3)
                r_target = 2
                trade.scaled_2r = True

            if sell_qty <= 0:
                return None

            # Place partial SELL order
            from kite_connect.trading.order_service import place_order
            resp = place_order(
                kite=self.kite,
                symbol=trade.symbol,
                exchange="NSE",
                transaction_type="SELL",
                quantity=sell_qty,
                order_type="MARKET",
                product="CNC",
            )

            # Reduce monitored quantity
            trade.quantity -= sell_qty
            realized_pnl = (current_price - trade.entry_price) * sell_qty
            self._roll_capital(realized_pnl)

            logger.info(
                "SCALE-OUT %dR: %s sold %d/%d @ %.2f (P&L: %.2f)",
                r_target, trade.symbol, sell_qty,
                trade.quantity + sell_qty, current_price, realized_pnl,
            )
            return {
                "type": f"SCALE_OUT_{r_target}R",
                "symbol": trade.symbol,
                "quantity_sold": sell_qty,
                "remaining_qty": trade.quantity,
                "exit_price": current_price,
                "r_multiple": round(r_multiple, 1),
                "realized_pnl": round(realized_pnl, 2),
                "order_id": resp.get("order_id") if resp.get("success") else None,
            }
        except Exception as exc:
            logger.debug("Scale-out check failed for %s: %s", trade.symbol, exc)
        return None

    def _roll_capital(self, realized_pnl: float) -> None:
        """Roll realized P&L into VolatilityTarget for dynamic position sizing.

        Also persists peak equity to a JSON file for crash recovery.
        """
        try:
            from config import Config
            # Update cumulative realized P&L in config for cross-session tracking
            prev = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
            Config._CUMULATIVE_REALIZED_PNL = prev + realized_pnl

            # Update peak equity
            capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
            current_equity = capital + Config._CUMULATIVE_REALIZED_PNL
            peak = getattr(Config, "_PEAK_EQUITY", current_equity)
            Config._PEAK_EQUITY = max(peak, current_equity)

            # Persist peak equity + cumulative P&L to file
            _state_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "portfolio_state.json",
            )
            state = {
                "cumulative_realized_pnl": Config._CUMULATIVE_REALIZED_PNL,
                "peak_equity": Config._PEAK_EQUITY,
                "last_updated": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(_state_file), exist_ok=True)
            with open(_state_file, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(
                "Capital rolled: realized=%.2f, cumulative=%.2f, equity=%.0f, peak=%.0f",
                realized_pnl, Config._CUMULATIVE_REALIZED_PNL,
                current_equity, Config._PEAK_EQUITY,
            )
        except Exception as exc:
            logger.debug("Capital rollup failed (non-fatal): %s", exc)

    def _sync_vol_target_unrealized(self) -> None:
        """G15: Sync unrealized P&L into VolatilityTarget for responsive sizing.

        Computes total unrealized P&L across all active trades and feeds it
        to the vol target so that the position sizer can shrink/grow positions
        in real-time rather than waiting for trade closure.
        """
        active = self.active_trades
        if not active or not self.kite:
            return
        try:
            ltps = self.kite.ltp([f"NSE:{t.symbol}" for t in active])
            total_unreal = 0.0
            for t in active:
                ltp_info = ltps.get(f"NSE:{t.symbol}", {})
                cmp = ltp_info.get("last_price", t.entry_price)
                if t.direction == "SHORT":
                    pnl = (t.entry_price - cmp) * t.quantity
                else:
                    pnl = (cmp - t.entry_price) * t.quantity
                total_unreal += pnl

            from config import Config
            capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
            realized = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
            Config._CURRENT_EQUITY = capital + realized + total_unreal
            logger.debug("Vol-target sync: unrealized=%.0f equity=%.0f", total_unreal, Config._CURRENT_EQUITY)
        except Exception as exc:
            logger.debug("Vol-target unrealized sync failed (non-fatal): %s", exc)

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
