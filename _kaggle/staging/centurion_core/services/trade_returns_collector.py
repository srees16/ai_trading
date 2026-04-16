"""
Trade Returns Collector — T1-1 support module.

Collects realized trade returns from trade_monitor and persists them
for Monte Carlo bootstrap risk estimation.

Called by scheduler after each EOD reconciliation.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)
_RETURNS_PATH = os.path.join(_DATA_DIR, "recent_trade_returns.json")
_MAX_RETURNS = 1000  # Keep last 1000 trade returns


def collect_from_monitor() -> List[float]:
    """Extract closed trade returns from trade_monitor SQLite DB.

    Returns list of trade return fractions (e.g., 0.03 = +3%).
    """
    import sqlite3

    db_path = os.path.join(_DATA_DIR, "trade_monitor_state.sqlite3")
    if not os.path.exists(db_path):
        logger.debug("Trade monitor DB not found at %s", db_path)
        return []

    returns = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Try to read from closed_trades table if it exists
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            if "closed_trades" in tables:
                rows = conn.execute(
                    "SELECT entry_price, exit_price FROM closed_trades "
                    "WHERE entry_price > 0 AND exit_price > 0 "
                    "ORDER BY closed_at DESC LIMIT ?",
                    (_MAX_RETURNS,),
                ).fetchall()
                for r in rows:
                    ret = (r["exit_price"] - r["entry_price"]) / r["entry_price"]
                    returns.append(round(ret, 6))

            # Also try monitored_trades with CLOSED status
            if "monitored_trades" in tables:
                rows = conn.execute(
                    "SELECT state_json FROM monitored_trades"
                ).fetchall()
                for r in rows:
                    try:
                        state = json.loads(r["state_json"])
                        entry = float(state.get("entry_price", 0))
                        exit_p = float(state.get("exit_price", 0) or state.get("last_price", 0))
                        if entry > 0 and exit_p > 0 and state.get("status") in ("CLOSED", "SL_HIT", "TP_HIT"):
                            ret = (exit_p - entry) / entry
                            returns.append(round(ret, 6))
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

    except Exception as e:
        logger.warning("Failed to collect trade returns: %s", e)

    return returns


def persist_returns(returns: List[float]) -> str:
    """Persist trade returns to JSON file for MC consumption.

    Merges with existing returns (deduplicates by keeping most recent).
    """
    existing = []
    if os.path.exists(_RETURNS_PATH):
        try:
            with open(_RETURNS_PATH, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    # Append new returns, keep only last _MAX_RETURNS
    combined = existing + returns
    combined = combined[-_MAX_RETURNS:]

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_RETURNS_PATH, "w") as f:
        json.dump(combined, f)

    logger.info("Persisted %d trade returns (%d new) to %s",
                len(combined), len(returns), _RETURNS_PATH)
    return _RETURNS_PATH


def run_collection() -> int:
    """Full collection pipeline: extract → persist. Returns count."""
    returns = collect_from_monitor()
    if returns:
        persist_returns(returns)
    return len(returns)
