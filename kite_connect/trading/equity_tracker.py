"""
Equity Tracker — Persistent equity-curve state for live trading.

Bridges Gap G3: tracks daily portfolio equity, computes SMA200,
DD tier, and bull/bear regime for the live Carver pipeline.

Stores state in SQLite so it survives restarts.  The backtest does
this in memory; this module persists it across trading days.

Usage::

    tracker = EquityTracker()
    tracker.record(portfolio_value=530_000.0)
    regime = tracker.get_regime()       # "BULL" | "BEAR" | "NEUTRAL"
    dd_tier = tracker.get_dd_tier()     # vol_target multiplier
    sma200 = tracker.get_sma200()       # float or None
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DB_DIR / "equity_tracker.sqlite3"

# SMA lookback must match backtest: 200 trading days
_SMA_LOOKBACK = 200
# Bull/bear thresholds must match full_pipeline_backtest.py
_BULL_THRESHOLD = 1.02   # equity > SMA200 × 1.02
_BEAR_THRESHOLD = 0.98   # equity < SMA200 × 0.98
# Bull confirmation: N consecutive days above threshold
_BULL_CONFIRM_DAYS = 5
# R22 infusion cooldown: trading days
_R22_INFUSION_COOLDOWN = 200


@dataclass
class RegimeState:
    """Current regime assessment."""
    regime: str            # "BULL", "BEAR", "NEUTRAL"
    sma200: Optional[float]
    equity: float
    peak_equity: float
    drawdown_pct: float
    vol_target_scale: float   # combined DD-tier × regime scale
    dd_tier_label: str        # "FULL", "MILD", "MODERATE", "SEVERE", "HALT"
    bull_streak: int          # consecutive days above SMA200+2%
    r22_infusion_due: bool    # True if bull confirmed + cooldown elapsed
    days_since_last_infusion: int


class EquityTracker:
    """Persistent equity-curve tracker backed by SQLite."""

    def __init__(self, db_path: Optional[str] = None,
                 initial_capital: float = 500_000.0):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initial_capital = initial_capital
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── DB setup ──────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS equity_history (
                trade_date  TEXT PRIMARY KEY,
                equity      REAL NOT NULL,
                peak_equity REAL NOT NULL,
                regime      TEXT DEFAULT 'NEUTRAL',
                bull_streak INTEGER DEFAULT 0,
                sma200      REAL,
                drawdown_pct REAL DEFAULT 0.0,
                notes       TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS infusion_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date  TEXT NOT NULL,
                amount      REAL NOT NULL,
                equity_before REAL NOT NULL,
                equity_after  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tracker_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.commit()
        # Seed initial capital if no history
        rows = conn.execute("SELECT COUNT(*) FROM equity_history").fetchone()[0]
        if rows == 0:
            self._set_meta("initial_capital", str(self._initial_capital))

    # ── Meta helpers ──────────────────────────────────────────

    def _set_meta(self, key: str, value: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO tracker_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def _get_meta(self, key: str, default: str = "") -> str:
        row = self._get_conn().execute(
            "SELECT value FROM tracker_meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    # ── Core API ──────────────────────────────────────────────

    def record(self, portfolio_value: float,
               trade_date: Optional[str] = None,
               notes: str = "") -> RegimeState:
        """Record today's portfolio value and return current regime state.

        Parameters
        ----------
        portfolio_value : float
            Total portfolio value (cash + holdings) from Kite.
        trade_date : str | None
            ISO date string.  Defaults to today.
        notes : str
            Optional annotation (e.g. "infused 50K").

        Returns
        -------
        RegimeState
        """
        trade_date = trade_date or date.today().isoformat()
        conn = self._get_conn()

        # Get previous peak
        prev_peak = self._get_peak_equity()
        peak_equity = max(prev_peak, portfolio_value)

        # Compute SMA200
        history = self._get_recent_equity(n=_SMA_LOOKBACK)
        history.append(portfolio_value)
        sma200 = sum(history) / len(history) if len(history) >= _SMA_LOOKBACK else None

        # Regime detection (matches backtest logic)
        prev_streak = self._get_last_bull_streak()
        regime, bull_streak = self._detect_regime(
            portfolio_value, sma200, prev_streak
        )

        # Drawdown
        dd_pct = (peak_equity - portfolio_value) / peak_equity if peak_equity > 0 else 0.0

        # DD tier vol scaling (matches full_pipeline_backtest.py P1c)
        dd_scale, dd_label = self._dd_tier(dd_pct)

        # Regime vol scaling
        regime_scale = self._regime_vol_scale(regime)
        combined_scale = min(dd_scale * regime_scale, 1.30)  # cap

        # R22 infusion check
        infusion_due, days_since = self._check_r22_infusion(
            regime, bull_streak, trade_date
        )

        # Persist
        conn.execute("""
            INSERT OR REPLACE INTO equity_history
            (trade_date, equity, peak_equity, regime, bull_streak, sma200, drawdown_pct, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_date, portfolio_value, peak_equity, regime,
              bull_streak, sma200, dd_pct, notes))
        conn.commit()

        return RegimeState(
            regime=regime,
            sma200=sma200,
            equity=portfolio_value,
            peak_equity=peak_equity,
            drawdown_pct=dd_pct,
            vol_target_scale=combined_scale,
            dd_tier_label=dd_label,
            bull_streak=bull_streak,
            r22_infusion_due=infusion_due,
            days_since_last_infusion=days_since,
        )

    def record_infusion(self, amount: float, trade_date: Optional[str] = None):
        """Log a capital infusion event."""
        trade_date = trade_date or date.today().isoformat()
        conn = self._get_conn()
        # Get current equity
        row = conn.execute(
            "SELECT equity FROM equity_history WHERE trade_date = ?",
            (trade_date,)
        ).fetchone()
        eq_before = row[0] if row else 0.0
        eq_after = eq_before + amount
        conn.execute(
            "INSERT INTO infusion_events (trade_date, amount, equity_before, equity_after) "
            "VALUES (?, ?, ?, ?)",
            (trade_date, amount, eq_before, eq_after),
        )
        conn.commit()
        logger.info("Infusion recorded: +₹%.0f on %s (%.0f → %.0f)",
                     amount, trade_date, eq_before, eq_after)

    def get_regime(self) -> RegimeState:
        """Get the most recent regime state without recording new data."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT trade_date, equity, peak_equity, regime, bull_streak, "
            "sma200, drawdown_pct FROM equity_history ORDER BY trade_date DESC LIMIT 1"
        ).fetchone()
        if not row:
            return RegimeState(
                regime="NEUTRAL", sma200=None,
                equity=self._initial_capital, peak_equity=self._initial_capital,
                drawdown_pct=0.0, vol_target_scale=1.0,
                dd_tier_label="FULL", bull_streak=0,
                r22_infusion_due=False, days_since_last_infusion=9999,
            )
        trade_date, equity, peak, regime, streak, sma, dd = row
        dd_scale, dd_label = self._dd_tier(dd)
        regime_scale = self._regime_vol_scale(regime)
        combined = min(dd_scale * regime_scale, 1.30)
        infusion_due, days_since = self._check_r22_infusion(regime, streak, trade_date)
        return RegimeState(
            regime=regime, sma200=sma, equity=equity,
            peak_equity=peak, drawdown_pct=dd,
            vol_target_scale=combined, dd_tier_label=dd_label,
            bull_streak=streak,
            r22_infusion_due=infusion_due,
            days_since_last_infusion=days_since,
        )

    def get_equity_history(self, n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return (date, equity) tuples, most recent last."""
        conn = self._get_conn()
        if n:
            rows = conn.execute(
                "SELECT trade_date, equity FROM equity_history "
                "ORDER BY trade_date DESC LIMIT ?", (n,)
            ).fetchall()
            return list(reversed(rows))
        rows = conn.execute(
            "SELECT trade_date, equity FROM equity_history ORDER BY trade_date"
        ).fetchall()
        return rows

    def get_total_infused(self) -> float:
        """Total fresh capital injected via R22 events."""
        row = self._get_conn().execute(
            "SELECT COALESCE(SUM(amount), 0) FROM infusion_events"
        ).fetchone()
        return row[0]

    # ── Private helpers ───────────────────────────────────────

    def _get_recent_equity(self, n: int = _SMA_LOOKBACK) -> List[float]:
        rows = self._get_conn().execute(
            "SELECT equity FROM equity_history ORDER BY trade_date DESC LIMIT ?",
            (n,)
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def _get_peak_equity(self) -> float:
        row = self._get_conn().execute(
            "SELECT MAX(peak_equity) FROM equity_history"
        ).fetchone()
        return row[0] if row and row[0] else self._initial_capital

    def _get_last_bull_streak(self) -> int:
        row = self._get_conn().execute(
            "SELECT bull_streak FROM equity_history ORDER BY trade_date DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else 0

    def _detect_regime(self, equity: float, sma200: Optional[float],
                       prev_streak: int) -> Tuple[str, int]:
        """Detect BULL/BEAR/NEUTRAL from equity vs SMA200."""
        if sma200 is None:
            return "NEUTRAL", 0
        if equity > sma200 * _BULL_THRESHOLD:
            streak = prev_streak + 1
            if streak >= _BULL_CONFIRM_DAYS:
                return "BULL", streak
            return "NEUTRAL", streak
        elif equity < sma200 * _BEAR_THRESHOLD:
            return "BEAR", 0
        else:
            return "NEUTRAL", 0

    @staticmethod
    def _dd_tier(dd_pct: float) -> Tuple[float, str]:
        """5-tier vol scaling matching full_pipeline_backtest.py P1c."""
        if dd_pct < 0.10:
            return 1.0, "FULL"     # vol = 0.50
        elif dd_pct < 0.20:
            return 0.90, "MILD"    # vol = 0.45
        elif dd_pct < 0.30:
            return 0.80, "MODERATE"  # vol = 0.40
        elif dd_pct < 0.35:
            return 0.60, "SEVERE"  # vol = 0.30
        else:
            return 0.0, "HALT"     # vol = 0.0 — go to cash

    @staticmethod
    def _regime_vol_scale(regime: str) -> float:
        """Regime-based vol multiplier matching backtest R21a scaling."""
        if regime == "BULL":
            return 1.25   # _R21A_REGIME_BOOST
        elif regime == "BEAR":
            return 0.55   # _R21A_REGIME_DEFEND
        return 1.0        # NEUTRAL

    def _check_r22_infusion(self, regime: str, bull_streak: int,
                            trade_date: str) -> Tuple[bool, int]:
        """Check if an R22 bull-crossover infusion is due."""
        conn = self._get_conn()
        # Count trading days since last infusion
        last_infusion = conn.execute(
            "SELECT trade_date FROM infusion_events ORDER BY trade_date DESC LIMIT 1"
        ).fetchone()
        if last_infusion:
            days_since = conn.execute(
                "SELECT COUNT(*) FROM equity_history WHERE trade_date > ?",
                (last_infusion[0],)
            ).fetchone()[0]
        else:
            days_since = 9999  # no infusion ever

        # Infusion due if: just confirmed bull AND cooldown elapsed
        infusion_due = (
            regime == "BULL"
            and bull_streak == _BULL_CONFIRM_DAYS  # exactly on confirmation day
            and days_since >= _R22_INFUSION_COOLDOWN
        )
        return infusion_due, days_since

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
