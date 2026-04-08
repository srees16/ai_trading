"""
Paper Trading Cloud Sync — Dual-write to Neon PostgreSQL.

After each local SQLite write (position, snapshot, signal, weekly checkpoint),
the PaperTrader calls into this module to persist the same data to Neon.

The Paper Dashboard UI reads from Neon first (cloud), falling back to
local SQLite when the cloud DB is unavailable.

All writes are best-effort: a Neon failure never blocks the trading loop.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

_cloud: Optional["PaperCloudSync"] = None


def get_paper_cloud() -> Optional["PaperCloudSync"]:
    """Return the singleton PaperCloudSync (or None if DB not configured).

    Performs a lightweight health check on the cached singleton to detect
    stale Neon connections (auto-suspend after idle timeout).
    """
    global _cloud
    if _cloud is not None:
        # Health check: verify the underlying engine is still usable
        try:
            with _cloud._db.get_session() as session:
                session.execute(text("SELECT 1"))
        except Exception:
            logger.info("Cloud sync connection stale — reinitialising")
            _cloud = None
    if _cloud is not None:
        return _cloud
    try:
        from database.connection import get_db_manager
        mgr = get_db_manager()
        if mgr is None:
            return None
        _cloud = PaperCloudSync(mgr)
        _cloud.ensure_tables()
        return _cloud
    except Exception as exc:
        logger.debug("Paper cloud sync unavailable: %s", exc)
        return None


class PaperCloudSync:
    """Best-effort sync of paper trading data to Neon PostgreSQL."""

    def __init__(self, db_manager):
        self._db = db_manager

    # ── Table creation ─────────────────────────────────────────

    def ensure_tables(self):
        """Create paper trading tables if they don't exist (idempotent)."""
        try:
            from database.models import Base
            engine = self._db.engine
            Base.metadata.create_all(
                engine,
                tables=[
                    Base.metadata.tables["paper_positions"],
                    Base.metadata.tables["paper_daily_snapshots"],
                    Base.metadata.tables["paper_signal_log"],
                    Base.metadata.tables["paper_weekly_checkpoints"],
                ],
            )
            logger.info("Paper trading cloud tables ensured.")
        except Exception as exc:
            logger.warning("Could not create paper cloud tables: %s", exc)

    # ── Write methods (called by PaperTrader) ──────────────────

    def sync_position(self, pos_data: dict) -> bool:
        """Upsert a paper position row."""
        try:
            from database.models import PaperPositionRecord
            with self._db.get_session() as session:
                # Try to find existing by symbol + opened_at
                existing = session.query(PaperPositionRecord).filter_by(
                    symbol=pos_data["symbol"],
                    opened_at=pos_data["opened_at"],
                ).first()
                if existing:
                    for k, v in pos_data.items():
                        if k != "id" and hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    row = PaperPositionRecord(
                        symbol=pos_data["symbol"],
                        side=pos_data["side"],
                        quantity=pos_data["quantity"],
                        entry_price=pos_data["entry_price"],
                        stop_loss=pos_data["stop_loss"],
                        target_price=pos_data["target_price"],
                        opened_at=pos_data["opened_at"],
                        closed_at=pos_data.get("closed_at", ""),
                        exit_price=pos_data.get("exit_price", 0),
                        exit_reason=pos_data.get("exit_reason", ""),
                        pnl=pos_data.get("pnl", 0),
                        pnl_pct=pos_data.get("pnl_pct", 0),
                        is_open=pos_data.get("is_open", True),
                    )
                    session.add(row)
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync position failed: %s", exc)
            return False

    def sync_snapshot(self, snap: dict) -> bool:
        """Upsert a daily snapshot row."""
        try:
            from database.models import PaperDailySnapshotRecord
            with self._db.get_session() as session:
                existing = session.query(PaperDailySnapshotRecord).filter_by(
                    date=snap["date"],
                ).first()
                if existing:
                    for k, v in snap.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(PaperDailySnapshotRecord(**snap))
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync snapshot failed: %s", exc)
            return False

    def sync_signals(self, date_str: str, signals: List[dict]) -> bool:
        """Batch-insert signal log rows (delete-then-insert for the date)."""
        if not signals:
            return True
        try:
            from database.models import PaperSignalLogRecord
            with self._db.get_session() as session:
                session.query(PaperSignalLogRecord).filter_by(date=date_str).delete()
                for sig in signals:
                    session.add(PaperSignalLogRecord(
                        date=date_str,
                        symbol=sig.get("symbol", ""),
                        forecast=sig.get("forecast", 0),
                        combined_forecast=sig.get("combined_forecast", 0),
                        action=sig.get("action", ""),
                        entry_price=sig.get("entry_price", 0),
                        stop_loss=sig.get("stop_loss", 0),
                        target_price=sig.get("target_price", 0),
                        quantity=sig.get("quantity", 0),
                        pipeline_sources=sig.get("pipeline_sources", ""),
                        was_traded=bool(sig.get("was_traded")),
                    ))
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync signals failed: %s", exc)
            return False

    def sync_weekly(self, ckpt: dict) -> bool:
        """Upsert a weekly checkpoint row."""
        try:
            from database.models import PaperWeeklyCheckpointRecord
            with self._db.get_session() as session:
                existing = session.query(PaperWeeklyCheckpointRecord).filter_by(
                    week_number=ckpt["week_number"],
                ).first()
                if existing:
                    for k, v in ckpt.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(PaperWeeklyCheckpointRecord(**ckpt))
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync weekly failed: %s", exc)
            return False

    def sync_stop_loss(self, symbol: str, opened_at: str, new_sl: float) -> bool:
        """Update trailing stop-loss on an open position."""
        try:
            from database.models import PaperPositionRecord
            with self._db.get_session() as session:
                pos = session.query(PaperPositionRecord).filter_by(
                    symbol=symbol, opened_at=opened_at, is_open=True,
                ).first()
                if pos:
                    pos.stop_loss = new_sl
                    session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync SL update failed: %s", exc)
            return False

    # ── Read methods (called by Paper Dashboard UI) ────────────

    def read_snapshots(self) -> pd.DataFrame:
        """Return all daily_snapshots as a DataFrame."""
        return self._read("SELECT * FROM paper_daily_snapshots ORDER BY date")

    def read_signals(self) -> pd.DataFrame:
        """Return all signal_log rows as a DataFrame."""
        return self._read("SELECT * FROM paper_signal_log ORDER BY date DESC, symbol")

    def read_positions(self) -> pd.DataFrame:
        """Return all paper_positions as a DataFrame."""
        return self._read("SELECT * FROM paper_positions ORDER BY opened_at DESC")

    def read_weekly(self) -> pd.DataFrame:
        """Return all weekly_checkpoints as a DataFrame."""
        return self._read("SELECT * FROM paper_weekly_checkpoints ORDER BY week_number")

    def _read(self, sql: str) -> pd.DataFrame:
        try:
            with self._db.get_session() as session:
                result = session.execute(text(sql))
                rows = result.fetchall()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows, columns=result.keys())
        except Exception as exc:
            logger.warning("Cloud read failed: %s", exc)
            return pd.DataFrame()
