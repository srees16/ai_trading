"""
TWAP/VWAP Execution Engine — T4-5 Order Execution Optimization.

Splits large orders into smaller child orders to minimize market impact.

Strategies:
  - TWAP: Time-Weighted Average Price — evenly spaced slices over time window
  - VWAP: Volume-Weighted Average Price — slices proportional to historical volume profile

Activation threshold: orders > ₹5L notional (configurable).

Integration:
  - Called from auto_executor.py when order exceeds threshold
  - Child orders placed via order_service with delay between slices
  - Parent order tracking for fill aggregation
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# Default config
TWAP_MIN_NOTIONAL = 500_000  # ₹5L minimum for order splitting
TWAP_MAX_SLICES = 10         # Max number of child orders
TWAP_SLICE_INTERVAL_S = 120  # 2 minutes between slices
VWAP_LOOKBACK_DAYS = 20      # Volume profile lookback


@dataclass
class ExecutionSlice:
    """Single child order in a TWAP/VWAP execution."""
    slice_id: int
    symbol: str
    quantity: int
    side: str            # "BUY" or "SELL"
    order_type: str      # "LIMIT" or "MARKET"
    limit_price: float = 0.0
    scheduled_time: str = ""
    executed: bool = False
    fill_price: float = 0.0
    fill_quantity: int = 0
    order_id: str = ""


@dataclass
class ExecutionPlan:
    """Complete TWAP/VWAP execution plan."""
    strategy: str        # "TWAP" or "VWAP"
    symbol: str
    total_quantity: int
    side: str
    slices: List[ExecutionSlice] = field(default_factory=list)
    estimated_vwap: float = 0.0
    actual_vwap: float = 0.0
    total_filled: int = 0
    slippage_bps: float = 0.0
    status: str = "PENDING"  # PENDING, EXECUTING, COMPLETED, PARTIAL, FAILED


def should_use_algo_execution(
    notional_value: float,
    adv_value: float = 0.0,
    min_notional: float = TWAP_MIN_NOTIONAL,
) -> bool:
    """Check if order should use TWAP/VWAP instead of single order.

    Parameters
    ----------
    notional_value : float
        Order notional (quantity × price).
    adv_value : float
        Average Daily Volume in ₹. If order > 5% of ADV, use algo.
    min_notional : float
        Minimum notional threshold for algo execution.
    """
    if notional_value >= min_notional:
        return True
    if adv_value > 0 and notional_value > adv_value * 0.05:
        return True
    return False


def create_twap_plan(
    symbol: str,
    quantity: int,
    side: str,
    current_price: float,
    n_slices: int = 5,
    interval_seconds: int = TWAP_SLICE_INTERVAL_S,
) -> ExecutionPlan:
    """Create a TWAP execution plan.

    Splits order into equal-sized slices at regular intervals.
    Uses LIMIT orders at current price ± buffer to minimize slippage.
    """
    n_slices = min(n_slices, TWAP_MAX_SLICES)
    n_slices = max(2, n_slices)

    qty_per_slice = quantity // n_slices
    remainder = quantity - (qty_per_slice * n_slices)

    slices = []
    now = datetime.now(_IST)

    for i in range(n_slices):
        q = qty_per_slice + (1 if i < remainder else 0)
        if q <= 0:
            continue
        scheduled = now + timedelta(seconds=i * interval_seconds)
        # LIMIT price: +0.1% for BUY, -0.1% for SELL (aggression buffer)
        if side.upper() == "BUY":
            limit_px = round(current_price * 1.001, 2)
        else:
            limit_px = round(current_price * 0.999, 2)

        slices.append(ExecutionSlice(
            slice_id=i + 1,
            symbol=symbol,
            quantity=q,
            side=side.upper(),
            order_type="LIMIT",
            limit_price=limit_px,
            scheduled_time=scheduled.isoformat(),
        ))

    return ExecutionPlan(
        strategy="TWAP",
        symbol=symbol,
        total_quantity=quantity,
        side=side.upper(),
        slices=slices,
        estimated_vwap=current_price,
    )


def create_vwap_plan(
    symbol: str,
    quantity: int,
    side: str,
    current_price: float,
    volume_profile: Optional[List[float]] = None,
    n_slices: int = 5,
) -> ExecutionPlan:
    """Create a VWAP execution plan.

    Distributes order proportional to intraday volume profile.
    volume_profile: list of relative volumes per time bucket.
    """
    n_slices = min(n_slices, TWAP_MAX_SLICES)

    if volume_profile is None or len(volume_profile) < n_slices:
        # Default NSE intraday volume profile (U-shaped)
        # Morning: 25%, Mid-morning: 15%, Midday: 10%, Afternoon: 20%, Close: 30%
        volume_profile = [0.25, 0.15, 0.10, 0.20, 0.30]

    # Normalize profile to n_slices
    profile = volume_profile[:n_slices]
    total_wt = sum(profile)
    if total_wt <= 0:
        profile = [1.0 / n_slices] * n_slices
        total_wt = 1.0

    slices = []
    now = datetime.now(_IST)
    remaining = quantity

    for i, wt in enumerate(profile):
        frac = wt / total_wt
        q = max(1, round(quantity * frac)) if i < len(profile) - 1 else remaining
        q = min(q, remaining)
        remaining -= q
        if q <= 0:
            continue

        scheduled = now + timedelta(seconds=i * TWAP_SLICE_INTERVAL_S)
        if side.upper() == "BUY":
            limit_px = round(current_price * 1.001, 2)
        else:
            limit_px = round(current_price * 0.999, 2)

        slices.append(ExecutionSlice(
            slice_id=i + 1,
            symbol=symbol,
            quantity=q,
            side=side.upper(),
            order_type="LIMIT",
            limit_price=limit_px,
            scheduled_time=scheduled.isoformat(),
        ))

    return ExecutionPlan(
        strategy="VWAP",
        symbol=symbol,
        total_quantity=quantity,
        side=side.upper(),
        slices=slices,
        estimated_vwap=current_price,
    )


def execute_plan(plan: ExecutionPlan, kite, product: str = "CNC") -> ExecutionPlan:
    """Execute a TWAP/VWAP plan sequentially via Kite.

    Places each slice as a LIMIT order, waits for interval, then next slice.
    Tracks fills and computes actual VWAP + slippage.
    """
    if kite is None:
        logger.warning("TWAP/VWAP: No Kite connection — dry run")
        plan.status = "FAILED"
        return plan

    try:
        from kite_connect.trading.order_service import place_order
    except ImportError:
        logger.error("Cannot import order_service for TWAP/VWAP execution")
        plan.status = "FAILED"
        return plan

    plan.status = "EXECUTING"
    total_cost = 0.0
    total_filled = 0

    for sl in plan.slices:
        try:
            result = place_order(
                kite,
                symbol=sl.symbol,
                exchange="NSE",
                transaction_type=sl.side,
                quantity=sl.quantity,
                order_type=sl.order_type,
                price=sl.limit_price if sl.order_type == "LIMIT" else None,
                product=product,
                tag=f"{plan.strategy}_{sl.slice_id}",
            )
            if result.get("success"):
                sl.executed = True
                sl.order_id = result.get("order_id", "")
                # Assume fill at limit price (actual fill checked later via order book)
                sl.fill_price = sl.limit_price
                sl.fill_quantity = sl.quantity
                total_cost += sl.fill_price * sl.fill_quantity
                total_filled += sl.fill_quantity
                logger.info(
                    "%s slice %d/%d: %s %d × %s @ %.2f",
                    plan.strategy, sl.slice_id, len(plan.slices),
                    sl.side, sl.quantity, sl.symbol, sl.limit_price,
                )
            else:
                logger.warning(
                    "%s slice %d FAILED: %s",
                    plan.strategy, sl.slice_id, result.get("error", "unknown"),
                )
        except Exception as e:
            logger.error("%s slice %d error: %s", plan.strategy, sl.slice_id, e)

        # Wait between slices (don't wait after last)
        if sl.slice_id < len(plan.slices):
            time.sleep(TWAP_SLICE_INTERVAL_S)

    plan.total_filled = total_filled
    if total_filled > 0:
        plan.actual_vwap = total_cost / total_filled
        if plan.estimated_vwap > 0:
            plan.slippage_bps = abs(plan.actual_vwap - plan.estimated_vwap) / plan.estimated_vwap * 10000

    plan.status = "COMPLETED" if total_filled == plan.total_quantity else "PARTIAL"
    logger.info(
        "%s execution %s: filled %d/%d, VWAP=%.2f, slippage=%.1f bps",
        plan.strategy, plan.status, total_filled, plan.total_quantity,
        plan.actual_vwap, plan.slippage_bps,
    )

    return plan
