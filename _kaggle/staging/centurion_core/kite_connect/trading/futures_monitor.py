"""
Futures Monitor — Phase L-2.

Monitors open futures positions for:
  - Auto-rollover before expiry
  - Margin utilisation alerts
  - P&L tracking of overlay positions
  - Regime-change driven de-leveraging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class FuturesPosition:
    """Snapshot of a single futures position."""
    tradingsymbol: str = ""
    quantity: int = 0
    average_price: float = 0.0
    last_price: float = 0.0
    pnl: float = 0.0
    m2m: float = 0.0
    days_to_expiry: int = 99


@dataclass
class FuturesMonitorResult:
    """Result of a monitoring poll."""
    positions: List[FuturesPosition] = field(default_factory=list)
    total_notional: float = 0.0
    total_pnl: float = 0.0
    total_margin_used: float = 0.0
    needs_rollover: List[str] = field(default_factory=list)
    needs_deleveraging: bool = False
    alerts: List[str] = field(default_factory=list)


def run_futures_monitor(kite, regime: str = "unknown") -> FuturesMonitorResult:
    """Poll Kite Connect for open futures positions and generate alerts.

    Parameters
    ----------
    kite : KiteConnect
        Authenticated Kite session.
    regime : str
        Current HMM regime for de-leveraging checks.
    """
    result = FuturesMonitorResult()

    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return result
    except Exception:
        return result

    try:
        positions = kite.positions().get("net", [])
    except Exception as exc:
        logger.error("Failed to fetch positions: %s", exc)
        result.alerts.append(f"Position fetch failed: {exc}")
        return result

    for pos in positions:
        tsym = pos.get("tradingsymbol", "")
        exchange = pos.get("exchange", "")
        if exchange != "NFO" or "FUT" not in tsym.upper():
            continue

        qty = pos.get("quantity", 0)
        if qty == 0:
            continue

        avg_price = pos.get("average_price", 0.0)
        last_price = pos.get("last_price", 0.0)
        pnl = pos.get("pnl", 0.0)
        m2m = pos.get("m2m", 0.0)

        fp = FuturesPosition(
            tradingsymbol=tsym,
            quantity=qty,
            average_price=avg_price,
            last_price=last_price,
            pnl=pnl,
            m2m=m2m,
        )

        # Estimate DTE from instrument lookup
        try:
            from services.futures_overlay import check_rollover_needed
            next_sym = check_rollover_needed(kite, tsym, days_before=3)
            if next_sym:
                result.needs_rollover.append(tsym)
                fp.days_to_expiry = 3  # approximate
        except Exception:
            pass

        result.positions.append(fp)
        lot_size = 25  # NIFTY lot size
        notional = abs(qty) * last_price if last_price > 0 else abs(qty) * avg_price
        result.total_notional += notional
        result.total_pnl += pnl

    # Margin check
    try:
        from kite_connect.trading.margin_monitor import get_margin_snapshot
        snapshot = get_margin_snapshot(kite)
        result.total_margin_used = snapshot.used
        if snapshot.alert_level == "CRITICAL":
            result.alerts.append(
                f"CRITICAL margin: {snapshot.utilisation_pct:.1f}% used "
                f"({snapshot.used:,.0f} / {snapshot.available + snapshot.used:,.0f})"
            )
        elif snapshot.alert_level == "WARNING":
            result.alerts.append(
                f"WARNING margin: {snapshot.utilisation_pct:.1f}% used"
            )
    except Exception as exc:
        logger.warning("Margin check in futures monitor failed: %s", exc)

    # De-leveraging check: if regime turned bear and we have long futures
    if regime.lower() == "bear":
        long_futures = [p for p in result.positions if p.quantity > 0]
        if long_futures:
            result.needs_deleveraging = True
            result.alerts.append(
                f"Regime=BEAR with {len(long_futures)} long futures. "
                "Consider reducing leverage."
            )

    if result.positions:
        logger.info(
            "Futures monitor: %d positions, notional=%.0f, PnL=%.0f, "
            "rollover=%d, deleverage=%s",
            len(result.positions),
            result.total_notional,
            result.total_pnl,
            len(result.needs_rollover),
            result.needs_deleveraging,
        )

    return result
