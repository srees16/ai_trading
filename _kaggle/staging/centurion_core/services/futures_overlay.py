"""
Futures Overlay — Phase L-2.

Uses NIFTY/BANKNIFTY futures for portfolio-level leverage when
the regime is favourable and equity positions are fully deployed.

Features:
  - Regime-adaptive leverage (bull=1.5x, range=1.2x, bear=0.8x)
  - Automatic rollover 3 days before expiry
  - Margin monitoring integration
  - Carry cost tracking (futures premium vs spot)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class FuturesOverlaySignal:
    """Signal to add or reduce futures exposure."""
    action: str = "HOLD"        # BUY_FUT, SELL_FUT, ROLL, HOLD
    instrument: str = ""        # e.g. "NIFTY26APRFUT"
    lots: int = 0
    estimated_margin: float = 0.0
    regime: str = "unknown"
    target_leverage: float = 1.0
    current_leverage: float = 1.0
    carry_cost_pct: float = 0.0


def compute_regime_leverage(regime: str, confidence: float = 0.5) -> float:
    """Compute target leverage based on HMM regime.

    Parameters
    ----------
    regime : str
        Current regime: "bull", "range", "bear"
    confidence : float
        HMM confidence (0-1). Below 0.5 → default to 1.0x.
    """
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return 1.0
        bull_max = getattr(Config, "LEVERAGE_BULL_MAX", 1.5)
        range_max = getattr(Config, "LEVERAGE_RANGE_MAX", 1.2)
        bear_max = getattr(Config, "LEVERAGE_BEAR_MAX", 0.8)
        absolute_max = getattr(Config, "LEVERAGE_MAX", 1.5)
    except Exception:
        return 1.0

    if confidence < 0.5:
        return 1.0

    regime_lower = regime.lower() if regime else "unknown"
    if regime_lower == "bull":
        target = bull_max
    elif regime_lower in ("range", "range-bound", "neutral"):
        target = range_max
    elif regime_lower == "bear":
        target = bear_max
    else:
        target = 1.0

    return min(target, absolute_max)


def compute_futures_overlay(
    portfolio_value: float,
    current_futures_notional: float,
    nifty_spot: float,
    regime: str,
    regime_confidence: float,
    nifty_fut_price: float = 0.0,
) -> FuturesOverlaySignal:
    """Compute the futures overlay signal.

    Parameters
    ----------
    portfolio_value : float
        Current equity portfolio value.
    current_futures_notional : float
        Existing futures notional (0 if no futures held).
    nifty_spot : float
        NIFTY 50 spot price.
    regime : str
        HMM regime.
    regime_confidence : float
        HMM confidence level.
    nifty_fut_price : float
        Current NIFTY futures price (for carry cost calc).
    """
    target_leverage = compute_regime_leverage(regime, regime_confidence)

    total_current = portfolio_value + current_futures_notional
    current_leverage = total_current / portfolio_value if portfolio_value > 0 else 1.0

    if nifty_spot <= 0:
        return FuturesOverlaySignal(
            regime=regime,
            target_leverage=target_leverage,
            current_leverage=round(current_leverage, 2),
        )

    # Desired futures notional = (target_leverage - 1) × portfolio_value
    desired_futures = (target_leverage - 1.0) * portfolio_value
    futures_delta = desired_futures - current_futures_notional

    # NIFTY lot size = 25, each lot = 25 × nifty_spot notional
    lot_value = 25 * nifty_spot
    lots_delta = int(futures_delta / lot_value) if lot_value > 0 else 0

    # Carry cost: futures premium over spot
    carry_cost = 0.0
    if nifty_fut_price > 0 and nifty_spot > 0:
        carry_cost = (nifty_fut_price - nifty_spot) / nifty_spot * 100

    # Margin estimate: ~12% of notional per lot
    margin_per_lot = lot_value * 0.12

    action = "HOLD"
    if lots_delta > 0:
        action = "BUY_FUT"
    elif lots_delta < 0:
        action = "SELL_FUT"

    return FuturesOverlaySignal(
        action=action,
        instrument="NIFTY",
        lots=abs(lots_delta),
        estimated_margin=round(abs(lots_delta) * margin_per_lot, 2),
        regime=regime,
        target_leverage=round(target_leverage, 2),
        current_leverage=round(current_leverage, 2),
        carry_cost_pct=round(carry_cost, 3),
    )


def check_rollover_needed(kite, current_fut_symbol: str, days_before: int = 3) -> Optional[str]:
    """Check if a futures position needs rolling to next month.

    Returns the next month's tradingsymbol if roll is needed, else None.
    """
    if not current_fut_symbol:
        return None

    try:
        instruments = kite.instruments("NFO")
        nifty_futs = [
            i for i in instruments
            if i.get("name", "").upper() == "NIFTY"
            and i.get("instrument_type", "").upper() == "FUT"
        ]
        nifty_futs.sort(key=lambda x: str(x.get("expiry", "")))

        # Find current and next month
        current = None
        next_month = None
        for fut in nifty_futs:
            if fut.get("tradingsymbol") == current_fut_symbol:
                current = fut
            elif current is not None and next_month is None:
                next_month = fut

        if current and next_month:
            expiry = current.get("expiry")
            if expiry:
                from datetime import date
                if isinstance(expiry, str):
                    expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
                days_to_expiry = (expiry - date.today()).days
                if days_to_expiry <= days_before:
                    logger.info(
                        "Futures rollover needed: %s expires in %d days → %s",
                        current_fut_symbol, days_to_expiry,
                        next_month.get("tradingsymbol"),
                    )
                    return next_month.get("tradingsymbol")

    except Exception as exc:
        logger.warning("Rollover check failed: %s", exc)

    return None
