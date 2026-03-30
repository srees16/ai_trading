"""
Event Calendar — Phase A-2.

Aggregates upcoming market-moving events:
  - Quarterly earnings (from BSE/NSE filings)
  - RBI monetary policy dates
  - Index rebalancing dates (NIFTY 50 semi-annual)
  - Budget / Union Budget date
  - F&O expiry dates (monthly + weekly)

Events are used by event_strategy.py to generate pre-event signals.
"""

from __future__ import annotations

import logging
import sqlite3
import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class MarketEvent:
    """A single market-moving event."""
    event_type: str = ""       # EARNINGS, RBI_POLICY, REBALANCE, BUDGET, FNO_EXPIRY
    symbol: str = ""           # stock symbol or "NIFTY50" / "MARKET"
    event_date: str = ""       # YYYY-MM-DD
    description: str = ""
    impact: str = "MEDIUM"     # LOW, MEDIUM, HIGH
    days_until: int = 0


# -------------------------------------------------------------------
# Hard-coded RBI policy dates 2025-26 (updated annually)
# -------------------------------------------------------------------
RBI_POLICY_DATES_2025 = [
    "2025-04-09", "2025-06-06", "2025-08-08",
    "2025-10-08", "2025-12-05",
]

RBI_POLICY_DATES_2026 = [
    "2026-02-06", "2026-04-08", "2026-06-05",
    "2026-08-07", "2026-10-07", "2026-12-04",
]

# NIFTY 50 semi-annual rebalancing (approximate)
REBALANCE_DATES = [
    "2025-03-28", "2025-09-26",
    "2026-03-27", "2026-09-25",
]


def _get_db_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "event_calendar.db"
    )


def _init_db():
    """Create events table."""
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            symbol TEXT DEFAULT '',
            event_date TEXT NOT NULL,
            description TEXT DEFAULT '',
            impact TEXT DEFAULT 'MEDIUM',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)"
    )
    conn.commit()
    conn.close()


def seed_fixed_events():
    """Seed known fixed-date events (RBI, rebalance) into the DB."""
    _init_db()
    conn = sqlite3.connect(_get_db_path())

    existing = set()
    for row in conn.execute("SELECT event_type, event_date FROM events"):
        existing.add((row[0], row[1]))

    inserts = []
    for d in RBI_POLICY_DATES_2025 + RBI_POLICY_DATES_2026:
        if ("RBI_POLICY", d) not in existing:
            inserts.append(("RBI_POLICY", "MARKET", d, "RBI Monetary Policy Meeting", "HIGH"))

    for d in REBALANCE_DATES:
        if ("REBALANCE", d) not in existing:
            inserts.append(("REBALANCE", "NIFTY50", d, "NIFTY 50 Semi-Annual Rebalance", "HIGH"))

    if inserts:
        conn.executemany(
            "INSERT INTO events (event_type, symbol, event_date, description, impact) "
            "VALUES (?, ?, ?, ?, ?)",
            inserts,
        )
        conn.commit()
        logger.info("Seeded %d fixed events into calendar", len(inserts))

    conn.close()


def add_earnings_event(symbol: str, event_date: str, description: str = ""):
    """Add an earnings event for a specific stock."""
    _init_db()
    conn = sqlite3.connect(_get_db_path())
    # Avoid duplicates
    existing = conn.execute(
        "SELECT id FROM events WHERE event_type='EARNINGS' AND symbol=? AND event_date=?",
        (symbol, event_date),
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO events (event_type, symbol, event_date, description, impact) "
            "VALUES (?, ?, ?, ?, ?)",
            ("EARNINGS", symbol, event_date, description or f"{symbol} Q-earnings", "HIGH"),
        )
        conn.commit()
    conn.close()


def add_fno_expiry(event_date: str, expiry_type: str = "monthly"):
    """Add F&O expiry date."""
    _init_db()
    conn = sqlite3.connect(_get_db_path())
    existing = conn.execute(
        "SELECT id FROM events WHERE event_type='FNO_EXPIRY' AND event_date=?",
        (event_date,),
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO events (event_type, symbol, event_date, description, impact) "
            "VALUES (?, ?, ?, ?, ?)",
            ("FNO_EXPIRY", "MARKET", event_date,
             f"F&O {expiry_type} expiry",
             "HIGH" if expiry_type == "monthly" else "MEDIUM"),
        )
        conn.commit()
    conn.close()


def get_upcoming_events(
    days_ahead: int = 7,
    symbol: Optional[str] = None,
    event_types: Optional[List[str]] = None,
) -> List[MarketEvent]:
    """Fetch upcoming events within the next N days.

    Parameters
    ----------
    days_ahead : int
        Look-ahead window.
    symbol : str, optional
        Filter by symbol (None = all).
    event_types : list, optional
        Filter by event types.
    """
    _init_db()
    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    conn = sqlite3.connect(_get_db_path())
    query = "SELECT event_type, symbol, event_date, description, impact FROM events WHERE event_date BETWEEN ? AND ?"
    params: list = [today.isoformat(), end_date.isoformat()]

    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if event_types:
        placeholders = ",".join("?" * len(event_types))
        query += f" AND event_type IN ({placeholders})"
        params.extend(event_types)

    query += " ORDER BY event_date ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    events: List[MarketEvent] = []
    for row in rows:
        evt_date = datetime.strptime(row[2], "%Y-%m-%d").date()
        events.append(MarketEvent(
            event_type=row[0],
            symbol=row[1],
            event_date=row[2],
            description=row[3],
            impact=row[4],
            days_until=(evt_date - today).days,
        ))

    return events
