"""
Unified Options Order Executor — Phase O-1.

Bridges scan_opportunities() output from CoveredCallStrategy and
PutSellingStrategy into actual Kite Connect order placement.

Handles:
  - NFO instrument lookup (symbol → tradingsymbol mapping)
  - SELL LIMIT orders for covered calls and cash-secured puts
  - BUY orders for tail risk hedges (NIFTY puts)
  - SQLite options journal for P&L tracking
  - Order tagging for lifecycle management
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# ── Options journal DB ─────────────────────────────────────────

_JOURNAL_DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "options_journal.db")

_CREATE_JOURNAL_SQL = """
CREATE TABLE IF NOT EXISTS options_journal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    tradingsymbol   TEXT NOT NULL,
    strategy        TEXT NOT NULL,       -- COVERED_CALL, CSP, TAIL_HEDGE
    side            TEXT NOT NULL,       -- SELL or BUY
    quantity        INTEGER NOT NULL,
    entry_premium   REAL,
    exit_premium    REAL,
    pnl             REAL DEFAULT 0,
    status          TEXT DEFAULT 'OPEN', -- OPEN, CLOSED, EXPIRED, ASSIGNED
    order_id        TEXT,
    close_order_id  TEXT,
    opened_at       TEXT,
    closed_at       TEXT,
    underlying_at_open  REAL,
    strike          REAL,
    expiry          TEXT,
    lot_size        INTEGER DEFAULT 1,
    tag             TEXT
);
"""


def _init_journal():
    """Create options journal DB if it doesn't exist."""
    db_path = os.path.abspath(_JOURNAL_DB)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_JOURNAL_SQL)
    conn.commit()
    conn.close()


def _record_open(entry: dict):
    """Record an opened options position in the journal."""
    try:
        db_path = os.path.abspath(_JOURNAL_DB)
        conn = sqlite3.connect(db_path)
        conn.execute(
            """INSERT INTO options_journal
               (symbol, tradingsymbol, strategy, side, quantity, entry_premium,
                status, order_id, opened_at, underlying_at_open, strike, expiry, lot_size, tag)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entry["symbol"], entry["tradingsymbol"], entry["strategy"],
                entry["side"], entry["quantity"], entry.get("entry_premium"),
                "OPEN", entry.get("order_id"), datetime.now(_IST).isoformat(),
                entry.get("underlying_at_open"), entry.get("strike"),
                entry.get("expiry"), entry.get("lot_size", 1), entry.get("tag"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Options journal write failed: %s", exc)


def get_open_positions() -> List[dict]:
    """Return all OPEN options positions from the journal."""
    try:
        db_path = os.path.abspath(_JOURNAL_DB)
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM options_journal WHERE status = 'OPEN'"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Options journal read failed: %s", exc)
        return []


def close_position(journal_id: int, exit_premium: float, close_order_id: str = "",
                   status: str = "CLOSED"):
    """Mark a journal position as closed and record P&L."""
    try:
        db_path = os.path.abspath(_JOURNAL_DB)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT entry_premium, side, quantity FROM options_journal WHERE id = ?",
            (journal_id,),
        ).fetchone()
        if row:
            entry_prem, side, qty = row
            if side == "SELL":
                pnl = (entry_prem - exit_premium) * qty
            else:
                pnl = (exit_premium - entry_prem) * qty
            conn.execute(
                """UPDATE options_journal
                   SET exit_premium=?, pnl=?, close_order_id=?, status=?, closed_at=?
                   WHERE id=?""",
                (exit_premium, round(pnl, 2), close_order_id, status,
                 datetime.now(_IST).isoformat(), journal_id),
            )
            conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Options journal close failed: %s", exc)


# ── NFO Instrument Lookup ──────────────────────────────────────

_NFO_CACHE: Dict[str, dict] = {}
_NFO_CACHE_TS: Optional[datetime] = None


def _refresh_nfo_instruments(kite) -> Dict[str, dict]:
    """Fetch and cache NFO instruments from Kite."""
    global _NFO_CACHE, _NFO_CACHE_TS
    now = datetime.now(_IST)
    if _NFO_CACHE and _NFO_CACHE_TS and (now - _NFO_CACHE_TS).seconds < 3600:
        return _NFO_CACHE

    try:
        instruments = kite.instruments("NFO")
        cache = {}
        for inst in instruments:
            ts = inst.get("tradingsymbol", "")
            cache[ts] = {
                "instrument_token": inst.get("instrument_token"),
                "tradingsymbol": ts,
                "strike": inst.get("strike"),
                "expiry": str(inst.get("expiry", "")),
                "lot_size": inst.get("lot_size", 1),
                "instrument_type": inst.get("instrument_type", ""),
                "name": inst.get("name", ""),
            }
        _NFO_CACHE = cache
        _NFO_CACHE_TS = now
        logger.info("Refreshed NFO instrument cache: %d instruments", len(cache))
        return cache
    except Exception as exc:
        logger.error("NFO instrument fetch failed: %s", exc)
        return _NFO_CACHE


def _find_nfo_symbol(nfo_cache: dict, underlying: str, strike: float,
                     expiry: str, opt_type: str) -> Optional[str]:
    """Find the NFO tradingsymbol for a given strike/expiry/type."""
    for ts, info in nfo_cache.items():
        if (info.get("name", "").upper() == underlying.upper()
                and abs(info.get("strike", 0) - strike) < 0.01
                and info.get("instrument_type", "").upper() == opt_type.upper()
                and expiry in str(info.get("expiry", ""))):
            return ts
    return None


# ── Order Execution ────────────────────────────────────────────

class OptionsExecutor:
    """Unified options order placer for all options strategies."""

    def __init__(self, kite):
        self.kite = kite
        _init_journal()

    def execute_covered_calls(self, candidates: list, dry_run: bool = False) -> List[dict]:
        """Place SELL orders for covered call candidates.

        Parameters
        ----------
        candidates : list[CoveredCallCandidate]
            From CoveredCallStrategy.scan_opportunities().
        dry_run : bool
            If True, log but don't actually place orders.

        Returns
        -------
        list[dict]
            Order results for each candidate.
        """
        from kite_connect.trading.order_service import place_order

        nfo_cache = _refresh_nfo_instruments(self.kite)
        results = []

        for c in candidates:
            ts = _find_nfo_symbol(nfo_cache, c.underlying, c.strike, c.expiry_date, "CE")
            if not ts:
                logger.warning("NFO symbol not found for %s %s CE", c.underlying, c.strike)
                results.append({"symbol": c.underlying, "success": False, "error": "NFO symbol not found"})
                continue

            qty = c.lot_size * c.lots_available
            logger.info(
                "Covered call: SELL %s x%d @ %.2f (underlying=%s, strike=%.0f)",
                ts, qty, c.premium, c.underlying, c.strike,
            )

            if dry_run:
                results.append({"symbol": ts, "success": True, "dry_run": True, "qty": qty})
                continue

            result = place_order(
                self.kite, symbol=ts, exchange="NFO",
                transaction_type="SELL", quantity=qty,
                order_type="LIMIT", product="NRML",
                price=c.premium,
            )
            results.append(result)

            if result.get("success"):
                _record_open({
                    "symbol": c.underlying, "tradingsymbol": ts,
                    "strategy": "COVERED_CALL", "side": "SELL",
                    "quantity": qty, "entry_premium": c.premium,
                    "order_id": result.get("order_id"),
                    "underlying_at_open": c.spot_price,
                    "strike": c.strike, "expiry": c.expiry_date,
                    "lot_size": c.lot_size, "tag": "COVERED_CALL",
                })

        return results

    def execute_cash_secured_puts(self, candidates: list, dry_run: bool = False) -> List[dict]:
        """Place SELL orders for cash-secured put candidates.

        Parameters
        ----------
        candidates : list[PutCandidate]
            From PutSellingStrategy.scan_opportunities().
        """
        from kite_connect.trading.order_service import place_order

        nfo_cache = _refresh_nfo_instruments(self.kite)
        results = []

        for c in candidates:
            ts = _find_nfo_symbol(nfo_cache, c.underlying, c.strike, c.expiry_date, "PE")
            if not ts:
                logger.warning("NFO symbol not found for %s %s PE", c.underlying, c.strike)
                results.append({"symbol": c.underlying, "success": False, "error": "NFO symbol not found"})
                continue

            qty = c.lot_size
            logger.info(
                "CSP: SELL %s x%d @ %.2f (underlying=%s, strike=%.0f)",
                ts, qty, c.premium, c.underlying, c.strike,
            )

            if dry_run:
                results.append({"symbol": ts, "success": True, "dry_run": True, "qty": qty})
                continue

            result = place_order(
                self.kite, symbol=ts, exchange="NFO",
                transaction_type="SELL", quantity=qty,
                order_type="LIMIT", product="NRML",
                price=c.premium,
            )
            results.append(result)

            if result.get("success"):
                _record_open({
                    "symbol": c.underlying, "tradingsymbol": ts,
                    "strategy": "CSP", "side": "SELL",
                    "quantity": qty, "entry_premium": c.premium,
                    "order_id": result.get("order_id"),
                    "underlying_at_open": c.spot_price,
                    "strike": c.strike, "expiry": c.expiry_date,
                    "lot_size": c.lot_size, "tag": "CSP",
                })

        return results

    def execute_tail_hedge(self, recommendation, dry_run: bool = False) -> dict:
        """Place BUY order for tail risk hedge (NIFTY puts).

        Parameters
        ----------
        recommendation : HedgeRecommendation
            From TailRiskHedge.assess().
        """
        from kite_connect.trading.order_service import place_order

        if recommendation is None or recommendation.action != "BUY_HEDGE":
            return {"success": False, "error": "No hedge action"}

        nfo_cache = _refresh_nfo_instruments(self.kite)
        strike = recommendation.strike
        ts = _find_nfo_symbol(nfo_cache, "NIFTY", strike, "", "PE")

        if not ts:
            # Try to find nearest NIFTY PE
            nifty_puts = [
                (k, v) for k, v in nfo_cache.items()
                if v.get("name", "").upper() == "NIFTY"
                and v.get("instrument_type", "").upper() == "PE"
                and abs(v.get("strike", 0) - strike) < 200
            ]
            if nifty_puts:
                nifty_puts.sort(key=lambda x: abs(x[1]["strike"] - strike))
                ts = nifty_puts[0][0]

        if not ts:
            return {"success": False, "error": f"NIFTY {strike} PE not found in NFO"}

        qty = recommendation.lots * 25  # NIFTY lot size = 25
        logger.info("Tail hedge: BUY %s x%d (lots=%d)", ts, qty, recommendation.lots)

        if dry_run:
            return {"success": True, "dry_run": True, "symbol": ts, "qty": qty}

        result = place_order(
            self.kite, symbol=ts, exchange="NFO",
            transaction_type="BUY", quantity=qty,
            order_type="MARKET", product="NRML",
        )

        if result.get("success"):
            _record_open({
                "symbol": "NIFTY", "tradingsymbol": ts,
                "strategy": "TAIL_HEDGE", "side": "BUY",
                "quantity": qty,
                "entry_premium": recommendation.premium_per_lot,
                "order_id": result.get("order_id"),
                "strike": strike, "lot_size": 25,
                "tag": "TAIL_HEDGE",
            })

        return result

    def close_option_position(self, journal_id: int, tradingsymbol: str,
                              side: str, quantity: int) -> dict:
        """Close an open options position (buy-to-close or sell-to-close)."""
        from kite_connect.trading.order_service import place_order

        close_side = "BUY" if side == "SELL" else "SELL"

        # Get current premium
        try:
            ltp_data = self.kite.ltp([f"NFO:{tradingsymbol}"])
            current_premium = ltp_data.get(f"NFO:{tradingsymbol}", {}).get("last_price", 0)
        except Exception:
            current_premium = 0

        result = place_order(
            self.kite, symbol=tradingsymbol, exchange="NFO",
            transaction_type=close_side, quantity=quantity,
            order_type="MARKET", product="NRML",
        )

        if result.get("success"):
            close_position(journal_id, current_premium,
                           close_order_id=str(result.get("order_id", "")))
            logger.info("Closed option position: %s (journal_id=%d)", tradingsymbol, journal_id)

        return result
