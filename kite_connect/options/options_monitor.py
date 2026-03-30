"""
Options Lifecycle Monitor — Phase O-2.

Polls open short option positions and manages:
  - Profit target: Close at 50% of max profit
  - Roll before expiry: Buy-to-close at DTE <= 2 if ITM
  - Delta breach: Roll if short call delta > 0.40
  - Assignment avoidance: Close ITM options before expiry to avoid STT
  - Premium income tracking via options_journal

Scheduled as Job 13 (every 5 min during market hours).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def run_options_monitor_poll(kite) -> dict:
    """Poll all open options positions and manage lifecycle.

    Parameters
    ----------
    kite : KiteConnect
        Authenticated Kite session.

    Returns
    -------
    dict
        Summary of actions taken.
    """
    from kite_connect.options.options_executor import (
        get_open_positions, OptionsExecutor, close_position,
    )

    positions = get_open_positions()
    if not positions:
        return {"open_positions": 0, "actions": []}

    executor = OptionsExecutor(kite)
    actions = []

    for pos in positions:
        try:
            action = _manage_single_position(kite, executor, pos)
            if action:
                actions.append(action)
        except Exception as exc:
            logger.warning("Options monitor error for %s: %s", pos.get("tradingsymbol"), exc)

    return {
        "open_positions": len(positions),
        "actions_taken": len(actions),
        "actions": actions,
    }


def _manage_single_position(kite, executor, pos: dict) -> Optional[dict]:
    """Manage a single open options position.

    Decision tree:
    1. If DTE <= 2 and ITM → buy-to-close (avoid assignment STT)
    2. If profit >= 50% of max → buy-to-close (take profit)
    3. If short call delta > 0.40 → flag for roll
    4. Otherwise → hold
    """
    ts = pos.get("tradingsymbol", "")
    strategy = pos.get("strategy", "")
    side = pos.get("side", "")
    entry_premium = pos.get("entry_premium", 0)
    quantity = pos.get("quantity", 0)
    journal_id = pos.get("id")
    strike = pos.get("strike", 0)

    if not ts or not journal_id:
        return None

    # Fetch current premium
    try:
        ltp_data = kite.ltp([f"NFO:{ts}"])
        current_premium = ltp_data.get(f"NFO:{ts}", {}).get("last_price", 0)
    except Exception:
        logger.debug("Could not fetch LTP for %s", ts)
        return None

    if current_premium <= 0:
        return None

    # Estimate DTE from expiry string
    dte = _estimate_dte(pos.get("expiry", ""))

    # Fetch underlying price for ITM check
    underlying_symbol = pos.get("symbol", "")
    underlying_price = 0
    if underlying_symbol:
        try:
            key = f"NSE:{underlying_symbol}" if underlying_symbol != "NIFTY" else "NSE:NIFTY 50"
            udata = kite.ltp([key])
            underlying_price = udata.get(key, {}).get("last_price", 0)
        except Exception:
            pass

    # ── Decision 1: DTE <= 2 and ITM → close to avoid assignment STT ──
    if dte is not None and dte <= 2 and underlying_price > 0 and strike > 0:
        is_itm = False
        if strategy == "COVERED_CALL" and underlying_price > strike:
            is_itm = True
        elif strategy == "CSP" and underlying_price < strike:
            is_itm = True

        if is_itm:
            logger.info("Options monitor: Closing %s (ITM at DTE=%d)", ts, dte)
            result = executor.close_option_position(journal_id, ts, side, quantity)
            return {"action": "CLOSE_ITM_EXPIRY", "symbol": ts, "result": result}

    # ── Decision 2: Profit target — close at 50% of max profit ──
    if side == "SELL" and entry_premium > 0:
        profit_pct = (entry_premium - current_premium) / entry_premium
        try:
            from config import Config
            target = getattr(Config, "OPTIONS_PROFIT_TARGET_PCT", 0.50)
        except Exception:
            target = 0.50

        if profit_pct >= target:
            logger.info(
                "Options monitor: Taking profit on %s (%.0f%% of max)",
                ts, profit_pct * 100,
            )
            result = executor.close_option_position(journal_id, ts, side, quantity)
            return {"action": "PROFIT_TARGET", "symbol": ts, "profit_pct": round(profit_pct, 2), "result": result}

    # ── Decision 3: DTE <= 2 and OTM → let expire worthless ──
    if dte is not None and dte <= 2:
        is_otm = True
        if strategy == "COVERED_CALL" and underlying_price > strike:
            is_otm = False
        elif strategy == "CSP" and underlying_price < strike:
            is_otm = False

        if is_otm and side == "SELL":
            logger.info("Options monitor: %s OTM at DTE=%d — letting expire", ts, dte)
            return {"action": "LET_EXPIRE", "symbol": ts, "dte": dte}

    # ── Decision 4: Delta breach for covered calls ──
    if strategy == "COVERED_CALL" and underlying_price > 0 and strike > 0:
        # Rough delta estimate
        moneyness = (strike - underlying_price) / underlying_price
        est_delta = max(0.05, 0.50 - moneyness * 5)
        if est_delta > 0.40:
            logger.info("Options monitor: %s delta=%.2f > 0.40 — flagging for roll", ts, est_delta)
            return {"action": "FLAG_ROLL", "symbol": ts, "delta": round(est_delta, 2)}

    return None


def _estimate_dte(expiry_str: str) -> Optional[int]:
    """Estimate days-to-expiry from an expiry date string."""
    if not expiry_str:
        return None
    try:
        # Try various formats
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S"):
            try:
                exp = datetime.strptime(expiry_str.split(" ")[0] if " " in expiry_str else expiry_str, fmt)
                today = datetime.now(_IST).replace(hour=0, minute=0, second=0, microsecond=0)
                return max(0, (exp - today).days)
            except ValueError:
                continue
    except Exception:
        pass
    return None
