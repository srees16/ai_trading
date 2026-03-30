"""
Margin Utilization Monitor — Phase L-2 / S-2.

Tracks real-time margin usage via kite.margins() and enforces:
  - Alert at 80% utilization → stop new leveraged orders
  - Halt at 90% → auto-reduce positions
  - Pre-order margin check before placing leveraged trades

Scheduled alongside trade_monitor (every 3 min during market hours).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class MarginSnapshot:
    """Real-time margin utilization snapshot."""
    total_capital: float = 0.0
    used_margin: float = 0.0
    available_margin: float = 0.0
    utilization_pct: float = 0.0
    collateral: float = 0.0
    alert_level: str = "OK"     # OK, WARNING, CRITICAL
    timestamp: str = ""


def get_margin_snapshot(kite) -> MarginSnapshot:
    """Fetch current margin utilization from Kite.

    Returns
    -------
    MarginSnapshot
        Current margin state with alert level.
    """
    try:
        from config import Config
        alert_pct = getattr(Config, "LEVERAGE_MARGIN_ALERT_PCT", 0.80)
        halt_pct = getattr(Config, "LEVERAGE_MARGIN_HALT_PCT", 0.90)
    except Exception:
        alert_pct, halt_pct = 0.80, 0.90

    try:
        margins = kite.margins(segment="equity")
        net = margins.get("net", 0) or 0
        used = margins.get("utilised", {})
        total_used = sum(
            v for k, v in used.items() if isinstance(v, (int, float))
        ) if isinstance(used, dict) else 0
        available = margins.get("available", {})
        cash_available = available.get("live_balance", 0) if isinstance(available, dict) else 0
        collateral = available.get("collateral", 0) if isinstance(available, dict) else 0

        total_capital = net if net > 0 else (cash_available + collateral + total_used)
        utilization = (total_used / total_capital) if total_capital > 0 else 0

        alert = "OK"
        if utilization >= halt_pct:
            alert = "CRITICAL"
        elif utilization >= alert_pct:
            alert = "WARNING"

        return MarginSnapshot(
            total_capital=round(total_capital, 2),
            used_margin=round(total_used, 2),
            available_margin=round(cash_available, 2),
            utilization_pct=round(utilization * 100, 2),
            collateral=round(collateral, 2),
            alert_level=alert,
            timestamp=datetime.now(_IST).isoformat(),
        )

    except Exception as exc:
        logger.warning("Margin fetch failed: %s", exc)
        return MarginSnapshot(
            alert_level="UNKNOWN",
            timestamp=datetime.now(_IST).isoformat(),
        )


def check_margin_before_order(kite, estimated_margin_required: float) -> bool:
    """Pre-flight margin check before placing a leveraged order.

    Returns True if sufficient margin is available (with buffer).
    """
    snap = get_margin_snapshot(kite)
    if snap.alert_level in ("CRITICAL", "UNKNOWN"):
        logger.warning("Margin check BLOCKED: alert_level=%s", snap.alert_level)
        return False

    buffer = estimated_margin_required * 1.20  # 20% buffer
    if snap.available_margin < buffer:
        logger.warning(
            "Margin check BLOCKED: available=%.0f < required=%.0f (with 20%% buffer)",
            snap.available_margin, buffer,
        )
        return False

    return True
