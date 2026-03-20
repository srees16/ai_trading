"""
Background Scheduler for Centurion Core — IND Stocks Pipeline.

Runs screening and scoring pipelines at configurable times during
market hours without requiring the Streamlit UI to be open.

Usage::

    # Activate virtualenv first, then:
    python scheduler.py

    # Or, from the Streamlit shell:
    # Start-Process python -ArgumentList "scheduler.py" -WindowStyle Hidden

Requires: ``pip install apscheduler``

Results are written to a lightweight SQLite cache so the Streamlit UI
can read the latest signals without re-running.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger("centurion.scheduler")

# ── Constants ──────────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))
_DB_PATH = Path(__file__).parent / "data" / "scheduler_cache.sqlite3"

# ── Ensure project root is on sys.path ─────────────────────────
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ═══════════════════════════════════════════════════════════════
# Cache layer (SQLite — lightweight, no external DB dependency)
# ═══════════════════════════════════════════════════════════════

def _init_cache_db():
    """Create the scheduler cache table if it doesn't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type    TEXT NOT NULL,          -- 'pre_market' | 'intraday'
            timestamp   TEXT NOT NULL,
            universe_size  INTEGER DEFAULT 0,
            screened_count INTEGER DEFAULT 0,
            buy_signals    INTEGER DEFAULT 0,
            sell_signals   INTEGER DEFAULT 0,
            verdicts_json  TEXT,                -- JSON array of verdict summaries
            plans_json     TEXT,                -- JSON array of trade plan summaries
            status      TEXT DEFAULT 'success'
        )
    """)
    conn.commit()
    conn.close()


def _save_run(run_type: str, summary: dict):
    """Persist a pipeline run result to the cache."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(
        """INSERT INTO pipeline_runs
           (run_type, timestamp, universe_size, screened_count,
            buy_signals, sell_signals, verdicts_json, plans_json, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_type,
            datetime.now(_IST).isoformat(),
            summary.get("universe_size", 0),
            summary.get("screened_count", 0),
            summary.get("buy_signals", 0),
            summary.get("sell_signals", 0),
            json.dumps(summary.get("verdicts", []), default=str),
            json.dumps(summary.get("plans", []), default=str),
            summary.get("status", "success"),
        ),
    )
    conn.commit()
    conn.close()


def get_latest_run(run_type: Optional[str] = None) -> Optional[dict]:
    """Read the most recent pipeline run from cache.

    This is called by the Streamlit UI to display
    the latest scheduled scan results without re-running.
    """
    if not _DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    if run_type:
        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_type=? ORDER BY id DESC LIMIT 1",
            (run_type,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def get_run_history(limit: int = 50, offset: int = 0) -> list[dict]:
    """Return recent pipeline runs from the cache DB."""
    if not _DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════
# Pipeline runner (headless — no Streamlit, no Kite orders)
# ═══════════════════════════════════════════════════════════════

def run_pipeline(run_type: str = "pre_market"):
    """Execute the full screening + scoring pipeline headless.

    When STRONG_BUY signals are detected, auto-authenticates with Kite
    (using TOTP auto-fill if ``ZERODHA_TOTP_SECRET`` is configured) and
    places orders automatically via AutoExecutor.
    """
    logger.info("=== Pipeline run started: %s ===", run_type)

    try:
        from kite_connect.nse.nse_universe import get_nse_universe
        from kite_connect.nse.screener import NSEScreener, ScreenerConfig
        from services.integrated_scorer import IntegratedScorer

        # 1. Universe
        symbols = get_nse_universe()
        logger.info("Universe: %d symbols", len(symbols))

        # 2. Screen (index mode for blue-chip universe)
        cfg = ScreenerConfig(index_mode=True)
        screener = NSEScreener(config=cfg)
        screened_df = screener.screen(symbols)
        logger.info("Screened: %d passed", len(screened_df))

        if screened_df.empty:
            _save_run(run_type, {
                "universe_size": len(symbols),
                "screened_count": 0,
                "status": "no_stocks_passed",
            })
            return

        # 3. IntegratedScorer verdicts
        ns_tickers = [f"{s}.NS" for s in screened_df["symbol"].tolist()]
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365)

        scorer = IntegratedScorer()
        verdicts = scorer.evaluate(
            tickers=ns_tickers,
            market="IND",
            date_range=(str(start_dt), str(end_dt)),
        )

        buy_verdicts = [v for v in verdicts if v.classification in ("BUY", "STRONG_BUY")]
        sell_verdicts = [v for v in verdicts if v.classification in ("SELL", "STRONG_SELL")]

        verdict_summaries = [
            {"ticker": v.ticker, "score": round(v.final_score, 3),
             "classification": v.classification, "confidence": round(v.confidence, 2)}
            for v in verdicts
        ]

        summary = {
            "universe_size": len(symbols),
            "screened_count": len(screened_df),
            "buy_signals": len(buy_verdicts),
            "sell_signals": len(sell_verdicts),
            "verdicts": verdict_summaries,
            "status": "success",
        }
        _save_run(run_type, summary)

        # 4. Desktop notification if signals found
        if buy_verdicts or sell_verdicts:
            _notify_signals(buy_verdicts, sell_verdicts)

        # 5. Auto-authenticate Kite & place orders for STRONG_BUY signals
        strong_buy = [v for v in buy_verdicts if v.classification == "STRONG_BUY"]
        if strong_buy:
            _auto_place_orders(verdicts, screened_df)

        logger.info(
            "=== Pipeline complete: %d BUY, %d SELL signals ===",
            len(buy_verdicts), len(sell_verdicts),
        )

    except Exception as exc:
        logger.exception("Pipeline run failed: %s", exc)
        _save_run(run_type, {"status": f"error: {exc}"})


def _notify_signals(buy_verdicts: list, sell_verdicts: list):
    """Send desktop notification for discovered signals."""
    try:
        from notifications.manager import NotificationManager
        nm = NotificationManager()

        parts = []
        if buy_verdicts:
            syms = ", ".join(v.ticker.replace(".NS", "") for v in buy_verdicts[:5])
            parts.append(f"{len(buy_verdicts)} BUY: {syms}")
        if sell_verdicts:
            syms = ", ".join(v.ticker.replace(".NS", "") for v in sell_verdicts[:5])
            parts.append(f"{len(sell_verdicts)} SELL: {syms}")

        nm.send_notification(
            title="Centurion — Signals Detected",
            message=" | ".join(parts),
            duration=15,
        )
    except Exception as exc:
        logger.debug("Notification failed (non-fatal): %s", exc)


def _auto_place_orders(verdicts: list, screened_df):
    """Auto-authenticate Kite and place orders for BUY/STRONG_BUY verdicts.

    Called by the scheduler when STRONG_BUY signals are detected. Uses
    the auto-TOTP flow (pyotp) when ``ZERODHA_TOTP_SECRET`` is configured,
    making the entire pipeline zero-touch.
    """
    try:
        from kite_connect.auth.kite_session import create_kite_session
        logger.info("Auto-authenticating Kite for STRONG_BUY order placement…")
        kite = create_kite_session()
    except Exception as exc:
        logger.error("Kite auto-auth failed: %s — orders skipped", exc)
        try:
            from notifications.manager import NotificationManager
            NotificationManager().send_notification(
                "Centurion — Auth Failed",
                f"Could not auto-authenticate Kite: {exc}",
            )
        except Exception:
            pass
        return

    if kite is None:
        logger.warning("Kite session is None — orders skipped")
        return

    try:
        from kite_connect.trading.auto_executor import AutoExecutor
        from kite_connect.nse.screener import ScreenerConfig

        signal_dict = {
            v.ticker.replace(".NS", "").replace(".BO", ""): v.classification
            for v in verdicts
        }
        buy_symbols = [
            sym for sym, tag in signal_dict.items()
            if tag in ("BUY", "STRONG_BUY")
        ]

        if not buy_symbols:
            logger.info("No BUY symbols to execute")
            return

        executor = AutoExecutor(
            kite=kite,
            screener_cfg=ScreenerConfig(index_mode=True),
            auto_place=True,
        )
        report = executor.run(
            symbols=buy_symbols,
            signal_verdicts=signal_dict,
            pre_screened_df=screened_df,
        )
        logger.info(
            "Auto-orders: %d placed, %d failed, %d filtered",
            report.orders_placed, report.orders_failed,
            report.signal_filtered_count,
        )
    except Exception as exc:
        logger.exception("Auto-order placement failed: %s", exc)


# ═══════════════════════════════════════════════════════════════
# Scheduler setup
# ═══════════════════════════════════════════════════════════════

def start_scheduler():
    """Start the APScheduler background scheduler with IST-aware jobs.

    Jobs
    ----
    1. **pre_market_scan** — 9:20 AM IST, Mon-Fri
       Full pipeline run before market opens (NSE opens 9:15).
    2. **intraday_rescan** — every 2 hours (10:30, 12:30, 14:30) Mon-Fri
       Lighter re-scan for intraday momentum shifts.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error(
            "APScheduler not installed. Run: pip install apscheduler\n"
            "Falling back to single immediate run."
        )
        run_pipeline("manual")
        return

    _init_cache_db()

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # Job 1: Pre-market full scan at 9:20 AM IST, weekdays
    scheduler.add_job(
        run_pipeline,
        CronTrigger(hour=9, minute=20, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        args=["pre_market"],
        id="pre_market_scan",
        name="Pre-Market Full Scan",
        misfire_grace_time=600,
    )

    # Job 2: Intraday re-scan every 2 hours during 10:30–14:30 IST
    scheduler.add_job(
        run_pipeline,
        CronTrigger(hour="10,12,14", minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        args=["intraday"],
        id="intraday_rescan",
        name="Intraday Re-Scan",
        misfire_grace_time=600,
    )

    logger.info("Scheduler started — press Ctrl+C to stop")
    logger.info("  Pre-market scan : 09:20 IST, Mon-Fri")
    logger.info("  Intraday re-scan: 10:30, 12:30, 14:30 IST, Mon-Fri")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Centurion Core — Pipeline Scheduler")
    parser.add_argument(
        "--once", action="store_true",
        help="Run the pipeline once immediately (no scheduling)",
    )
    args = parser.parse_args()

    _init_cache_db()

    if args.once:
        run_pipeline("manual")
    else:
        start_scheduler()
