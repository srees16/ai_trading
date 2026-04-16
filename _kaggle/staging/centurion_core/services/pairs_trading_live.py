"""
Pairs Trading Live — Phase A-1.

Generates real-time mean-reversion signals for co-integrated stock pairs.
Uses the SpreadExecutor for synchronized 2-leg execution.

Workflow:
  1. Periodically compute z-score of spread for each pair
  2. Entry when |z| > ENTRY_Z  (buy underperformer, sell outperformer)
  3. Exit when |z| < EXIT_Z or stop at MAX_Z
  4. Feed signal into forecast_combiner as "pairs_arb" source
"""

from __future__ import annotations

import logging
import sqlite3
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# Pairs config defaults (overridden by Config)
DEFAULT_PAIRS = [
    ("HDFCBANK", "ICICIBANK"),
    ("TCS", "INFY"),
    ("RELIANCE", "ONGC"),
    ("SBIN", "PNB"),
    ("BHARTIARTL", "IDEA"),
]

ENTRY_Z = 2.0
EXIT_Z = 0.5
MAX_Z = 4.0       # Stop-loss z-score
LOOKBACK = 60     # trading days for spread calc


@dataclass
class PairsSignal:
    """Signal from pairs analysis."""
    leg1: str = ""
    leg2: str = ""
    z_score: float = 0.0
    spread: float = 0.0
    hedge_ratio: float = 1.0
    action: str = "HOLD"   # ENTER_LONG_LEG1, ENTER_SHORT_LEG1, EXIT, STOP, HOLD
    forecast: float = 0.0  # -20 to +20 for forecast_combiner


@dataclass
class PairsState:
    """Track open pair positions."""
    leg1: str = ""
    leg2: str = ""
    direction: str = ""    # "LONG_LEG1" or "SHORT_LEG1"
    entry_z: float = 0.0
    entry_time: str = ""
    leg1_qty: int = 0
    leg2_qty: int = 0


def _get_pairs_db_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "pairs_state.db"
    )


def _init_pairs_db():
    """Create pairs tracking table if it doesn't exist."""
    db_path = _get_pairs_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pairs_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            leg1 TEXT NOT NULL,
            leg2 TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_z REAL NOT NULL,
            entry_time TEXT NOT NULL,
            leg1_qty INTEGER DEFAULT 0,
            leg2_qty INTEGER DEFAULT 0,
            exit_time TEXT,
            exit_z REAL,
            pnl REAL DEFAULT 0.0,
            status TEXT DEFAULT 'OPEN'
        )
    """)
    conn.commit()
    conn.close()


def compute_hedge_ratio(prices_1: np.ndarray, prices_2: np.ndarray) -> float:
    """OLS hedge ratio: price_1 = beta × price_2 + alpha."""
    if len(prices_1) < 20 or len(prices_2) < 20:
        return 1.0
    x = prices_2[-LOOKBACK:]
    y = prices_1[-LOOKBACK:]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-12)
    return round(float(beta), 4)


def compute_z_score(
    prices_1: np.ndarray,
    prices_2: np.ndarray,
    hedge_ratio: float,
) -> Tuple[float, float]:
    """Compute z-score of spread = price_1 - beta × price_2.

    Returns (z_score, raw_spread).
    """
    spread = prices_1[-LOOKBACK:] - hedge_ratio * prices_2[-LOOKBACK:]
    mu = np.mean(spread)
    sigma = np.std(spread) + 1e-12
    current_spread = spread[-1]
    z = float((current_spread - mu) / sigma)
    return round(z, 3), round(float(current_spread), 2)


def generate_pairs_signal(
    leg1: str,
    leg2: str,
    prices_1: np.ndarray,
    prices_2: np.ndarray,
    has_open_position: bool = False,
    current_direction: str = "",
) -> PairsSignal:
    """Generate a trading signal for a single pair.

    Parameters
    ----------
    leg1, leg2 : str
        Stock symbols.
    prices_1, prices_2 : np.ndarray
        Close price arrays (at least LOOKBACK periods).
    has_open_position : bool
        Whether we already have an open position in this pair.
    current_direction : str
        "LONG_LEG1" or "SHORT_LEG1" if position is open.
    """
    try:
        from config import Config
        entry_z = getattr(Config, "PAIRS_ENTRY_Z", ENTRY_Z)
        exit_z = getattr(Config, "PAIRS_EXIT_Z", EXIT_Z)
        max_z = getattr(Config, "PAIRS_MAX_Z", MAX_Z)
    except Exception:
        entry_z, exit_z, max_z = ENTRY_Z, EXIT_Z, MAX_Z

    hedge_ratio = compute_hedge_ratio(prices_1, prices_2)
    z_score, spread = compute_z_score(prices_1, prices_2, hedge_ratio)

    signal = PairsSignal(
        leg1=leg1,
        leg2=leg2,
        z_score=z_score,
        spread=spread,
        hedge_ratio=hedge_ratio,
    )

    if has_open_position:
        # Check for exit or stop
        if abs(z_score) <= exit_z:
            signal.action = "EXIT"
            signal.forecast = 0.0
        elif abs(z_score) >= max_z:
            signal.action = "STOP"
            signal.forecast = 0.0
        else:
            signal.action = "HOLD"
            # Forecast proportional to expected reversion, decaying toward stop
            # At entry_z: full forecast; at max_z: forecast -> 0 (not peak)
            z_abs = abs(z_score)
            decay = max(0.0, (max_z - z_abs) / (max_z - entry_z)) if max_z > entry_z else 1.0
            raw_fc = decay * z_abs * 5.0
            if current_direction == "LONG_LEG1":
                signal.forecast = round(min(20.0, max(-20.0, -raw_fc)), 2)
            else:
                signal.forecast = round(min(20.0, max(-20.0, raw_fc)), 2)
    else:
        # Check for entry
        if z_score > entry_z:
            # Spread is wide → short leg1, long leg2 (expect reversion)
            signal.action = "ENTER_SHORT_LEG1"
            signal.forecast = round(min(20.0, max(-20.0, -z_score * 5.0)), 2)
        elif z_score < -entry_z:
            # Spread is narrow → long leg1, short leg2
            signal.action = "ENTER_LONG_LEG1"
            signal.forecast = round(min(20.0, max(-20.0, -z_score * 5.0)), 2)
        else:
            signal.action = "HOLD"
            signal.forecast = 0.0

    return signal


def scan_all_pairs(price_data: Dict[str, np.ndarray]) -> List[PairsSignal]:
    """Scan all configured pairs and return signals.

    Parameters
    ----------
    price_data : dict
        Mapping symbol -> np.ndarray of close prices.
    """
    try:
        from config import Config
        if not getattr(Config, "PAIRS_ENABLED", False):
            return []
        pairs = getattr(Config, "PAIRS_LIST", DEFAULT_PAIRS)
    except Exception:
        return []

    _init_pairs_db()

    # Load open positions from DB
    open_positions: Dict[str, PairsState] = {}
    try:
        conn = sqlite3.connect(_get_pairs_db_path())
        rows = conn.execute(
            "SELECT leg1, leg2, direction, entry_z FROM pairs_positions WHERE status='OPEN'"
        ).fetchall()
        conn.close()
        for row in rows:
            key = f"{row[0]}_{row[1]}"
            open_positions[key] = PairsState(
                leg1=row[0], leg2=row[1], direction=row[2], entry_z=row[3]
            )
    except Exception:
        pass

    signals: List[PairsSignal] = []
    for leg1, leg2 in pairs:
        if leg1 not in price_data or leg2 not in price_data:
            continue
        p1 = price_data[leg1]
        p2 = price_data[leg2]
        if len(p1) < LOOKBACK or len(p2) < LOOKBACK:
            continue

        key = f"{leg1}_{leg2}"
        has_open = key in open_positions
        direction = open_positions[key].direction if has_open else ""

        sig = generate_pairs_signal(leg1, leg2, p1, p2, has_open, direction)
        if sig.action != "HOLD":
            signals.append(sig)

    logger.info("Pairs scan: %d active signals from %d pairs", len(signals), len(pairs))
    return signals
