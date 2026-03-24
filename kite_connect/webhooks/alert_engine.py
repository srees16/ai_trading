"""
Price alert engine for Centurion Core — Indian Stocks.

Subscribes to TICK_BATCH events from the WebhookDispatcher and evaluates
user-defined conditions (price crosses threshold, % change exceeds limit,
volume spike, etc.).  When an alert triggers, it dispatches a STRATEGY_ALERT
event and optionally fires a desktop notification via NotificationManager.

Usage
-----
>>> from kite_connect.webhooks.alert_engine import PriceAlertEngine
>>> engine = PriceAlertEngine.get_instance()
>>> alert_id = engine.create_alert("RELIANCE", "price_above", 2800.0)
>>> engine.remove_alert(alert_id)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert condition enum (mirrors the Pydantic schema)
# ---------------------------------------------------------------------------

class AlertCondition(str, Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    CHANGE_PCT_ABOVE = "change_pct_above"
    CHANGE_PCT_BELOW = "change_pct_below"
    VOLUME_ABOVE = "volume_above"


# ---------------------------------------------------------------------------
# Alert record
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    condition: AlertCondition = AlertCondition.PRICE_ABOVE
    threshold: float = 0.0
    message: str = ""
    one_shot: bool = True
    active: bool = True
    created_at: float = field(default_factory=time.time)
    triggered_at: Optional[float] = None
    trigger_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "condition": self.condition.value,
            "threshold": self.threshold,
            "message": self.message,
            "one_shot": self.one_shot,
            "active": self.active,
            "created_at": self.created_at,
            "triggered_at": self.triggered_at,
            "trigger_count": self.trigger_count,
        }


# ---------------------------------------------------------------------------
# Alert engine singleton
# ---------------------------------------------------------------------------

class PriceAlertEngine:
    """
    Evaluates price alerts against incoming tick batches.

    Thread-safe — called from the WebhookDispatcher's ThreadPoolExecutor.
    """

    _instance: Optional["PriceAlertEngine"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def get_instance(cls) -> "PriceAlertEngine":
        return cls()

    def __init__(self):
        if self._initialized:
            return
        self._alerts: Dict[str, Alert] = {}
        self._alerts_lock = threading.Lock()
        self._on_trigger_callbacks: List[Callable] = []
        self._stats = {
            "ticks_evaluated": 0,
            "alerts_triggered": 0,
        }
        self._initialized = True
        logger.info("PriceAlertEngine initialised")

    # ── Alert CRUD ─────────────────────────────────────────────

    def create_alert(
        self,
        symbol: str,
        condition: str | AlertCondition,
        threshold: float,
        message: str = "",
        one_shot: bool = True,
    ) -> str:
        """
        Create and register a new price alert.

        Returns the alert_id.
        """
        if isinstance(condition, str):
            condition = AlertCondition(condition)

        if not message:
            message = f"{symbol} {condition.value} {threshold}"

        alert = Alert(
            symbol=symbol.upper(),
            condition=condition,
            threshold=threshold,
            message=message,
            one_shot=one_shot,
        )

        with self._alerts_lock:
            self._alerts[alert.alert_id] = alert

        logger.info(
            "Alert created: %s — %s %s %s",
            alert.alert_id[:8], symbol, condition.value, threshold,
        )
        return alert.alert_id

    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert by ID. Returns True if found."""
        with self._alerts_lock:
            removed = self._alerts.pop(alert_id, None)
        if removed:
            logger.info("Alert removed: %s", alert_id[:8])
        return removed is not None

    def get_alert(self, alert_id: str) -> Optional[Dict]:
        with self._alerts_lock:
            a = self._alerts.get(alert_id)
            return a.to_dict() if a else None

    def list_alerts(self, only_active: bool = False) -> List[Dict]:
        with self._alerts_lock:
            alerts = list(self._alerts.values())
        if only_active:
            alerts = [a for a in alerts if a.active]
        return [a.to_dict() for a in alerts]

    def clear_all(self) -> int:
        """Remove all alerts. Returns count removed."""
        with self._alerts_lock:
            count = len(self._alerts)
            self._alerts.clear()
        logger.info("All %d alerts cleared", count)
        return count

    # ── Trigger callback registration ──────────────────────────

    def on_trigger(self, callback: Callable) -> None:
        """Register a callback invoked when any alert triggers."""
        self._on_trigger_callbacks.append(callback)

    # ── Dispatcher callback (called from WebhookDispatcher) ────

    def __call__(self, event) -> None:
        """
        WebhookDispatcher callback entry point.

        Expected to receive TICK_BATCH events with
        payload = {"ticks": [{"name": ..., "ltp": ..., ...}, ...]}
        """
        from .events import EventType
        if event.event_type != EventType.TICK_BATCH:
            return

        ticks = event.payload.get("ticks", [])
        if not ticks:
            return

        # Build a lookup: symbol tick data
        tick_map: Dict[str, Dict] = {}
        for t in ticks:
            sym = t.get("name", "").upper()
            if sym:
                tick_map[sym] = t

        self._stats["ticks_evaluated"] += len(tick_map)

        # Evaluate each active alert
        triggered: List[tuple] = []  # (alert, tick_data)

        with self._alerts_lock:
            for alert in list(self._alerts.values()):
                if not alert.active:
                    continue
                tick = tick_map.get(alert.symbol)
                if tick is None:
                    continue
                if self._evaluate(alert, tick):
                    triggered.append((alert, tick))

        # Process triggered alerts (outside the lock)
        for alert, tick in triggered:
            self._fire_alert(alert, tick)

    # ── Evaluation logic ───────────────────────────────────────

    @staticmethod
    def _evaluate(alert: Alert, tick: Dict) -> bool:
        """Return True if the alert condition is met by the tick data."""
        cond = alert.condition
        threshold = alert.threshold

        if cond == AlertCondition.PRICE_ABOVE:
            return (tick.get("ltp") or 0.0) >= threshold

        elif cond == AlertCondition.PRICE_BELOW:
            return (tick.get("ltp") or 0.0) <= threshold

        elif cond == AlertCondition.CHANGE_PCT_ABOVE:
            return (tick.get("change_pct") or 0.0) >= threshold

        elif cond == AlertCondition.CHANGE_PCT_BELOW:
            return (tick.get("change_pct") or 0.0) <= threshold

        elif cond == AlertCondition.VOLUME_ABOVE:
            return (tick.get("volume") or 0) >= threshold

        return False

    def _fire_alert(self, alert: Alert, tick: Dict) -> None:
        """Process a triggered alert."""
        now = time.time()
        alert.triggered_at = now
        alert.trigger_count += 1
        self._stats["alerts_triggered"] += 1

        if alert.one_shot:
            alert.active = False

        logger.info(
            "ALERT TRIGGERED: %s — %s %s %s (ltp=%s)",
            alert.alert_id[:8],
            alert.symbol,
            alert.condition.value,
            alert.threshold,
            tick.get("ltp"),
        )

        # Dispatch a STRATEGY_ALERT event through the webhook system
        try:
            from .dispatcher import WebhookDispatcher
            from .events import EventType, WebhookEvent

            WebhookDispatcher().dispatch(WebhookEvent(
                event_type=EventType.STRATEGY_ALERT,
                payload={
                    "alert_id": alert.alert_id,
                    "symbol": alert.symbol,
                    "condition": alert.condition.value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "trigger_count": alert.trigger_count,
                    "tick": tick,
                },
                source="alert_engine",
            ))
        except Exception:
            logger.exception("Failed to dispatch alert event")

        # Fire desktop notification (best-effort)
        try:
            from notifications.manager import NotificationManager
            nm = NotificationManager()
            nm.send_notification(
                title=f"Price Alert: {alert.symbol}",
                message=alert.message,
                duration=5,
            )
        except Exception:
            logger.debug("Desktop notification unavailable", exc_info=True)

        # Fire external callbacks
        for cb in self._on_trigger_callbacks:
            try:
                cb(alert.to_dict(), tick)
            except Exception:
                logger.exception("Alert trigger callback failed")

    # ── Introspection ──────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def active_count(self) -> int:
        with self._alerts_lock:
            return sum(1 for a in self._alerts.values() if a.active)

    @property
    def total_count(self) -> int:
        with self._alerts_lock:
            return len(self._alerts)
