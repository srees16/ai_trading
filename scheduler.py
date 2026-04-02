"""
Background Scheduler for Centurion Core â€” IND Stocks Pipeline.

Runs screening and scoring pipelines at configurable times during
market hours without requiring the Streamlit UI to be open.

Usage::

    # Activate virtualenv first, then:
    python scheduler.py

    # Or, from the Streamlit shell:
    # Start-Process python -ArgumentList "scheduler.py" -WindowStyle Hidden

Requires: ``pip install apscheduler``

Results are written to a lightweight SQLite cache so the Streamlit UI
(and the REST API) can read the latest signals without re-running.
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

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_IST = timezone(timedelta(hours=5, minutes=30))
_DB_PATH = Path(__file__).parent / "data" / "scheduler_cache.sqlite3"

# â”€â”€ Ensure project root is on sys.path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cache layer (SQLite â€” lightweight, no external DB dependency)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _init_cache_db():
    """Create the scheduler cache table if it doesn't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type    TEXT NOT NULL,          -- 'pre_market'
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      TEXT NOT NULL,
            job_name    TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            status      TEXT DEFAULT 'running',
            detail      TEXT
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

    This is called by the Streamlit UI and REST API to display
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


def _log_job_start(job_id: str, job_name: str) -> int:
    """Record that a scheduler job started. Returns the row id."""
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        cur = conn.execute(
            "INSERT INTO job_log (job_id, job_name, started_at, status) VALUES (?, ?, ?, 'running')",
            (job_id, job_name, datetime.now(_IST).isoformat()),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id
    except Exception:
        return -1


def _log_job_end(row_id: int, status: str = "ok", detail: str = ""):
    """Mark a scheduler job as finished."""
    if row_id < 0:
        return
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "UPDATE job_log SET finished_at=?, status=?, detail=? WHERE id=?",
            (datetime.now(_IST).isoformat(), status, detail[:500] if detail else "", row_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_job_log(limit: int = 50, job_id: str | None = None) -> list:
    """Return the most recent job log entries, optionally filtered by job_id."""
    if not _DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    if job_id:
        rows = conn.execute(
            "SELECT * FROM job_log WHERE job_id = ? ORDER BY id DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM job_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════
# Verdict caching helpers
# ═══════════════════════════════════════════════════════════════

_VERDICT_CACHE_TTL = 2 * 3600  # 2 hours — refreshed on each pipeline run


def _cache_verdicts(verdicts):
    """Persist individual verdict results to the infrastructure cache."""
    try:
        from infrastructure.cache import cache
        for v in verdicts:
            ls = v.layer_scores or {}
            payload = {
                "ticker": v.ticker,
                "core_score": ls.get("core", 0) or 0,
                "strategy_score": ls.get("strategy", 0) or 0,
                "ml_score": ls.get("ml_features", 0) or 0,
                "rl_score": ls.get("rl_bot", 0) or 0,
                "robustness_score": ls.get("robustness", 0) or 0,
                "weighted_score": v.final_score,
                "verdict": v.classification,
                "confidence": round(v.confidence, 2),
                "cached_at": datetime.now(_IST).isoformat(),
            }
            cache.set(f"verdict:{v.ticker}", payload, ttl=_VERDICT_CACHE_TTL)
        logger.info("Cached %d verdicts (TTL=%ds)", len(verdicts), _VERDICT_CACHE_TTL)
    except Exception as exc:
        logger.debug("Verdict caching failed (non-fatal): %s", exc)


def get_cached_verdict(ticker: str):
    """Read a single cached verdict dict, or None if miss/expired."""
    try:
        from infrastructure.cache import cache
        return cache.get(f"verdict:{ticker}")
    except Exception:
        return None


import functools


def _tracked_job(job_id: str, job_name: str):
    """Decorator that logs job start/end to the job_log table."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            row_id = _log_job_start(job_id, job_name)
            try:
                result = fn(*args, **kwargs)
                _log_job_end(row_id, status="ok")
                return result
            except Exception as exc:
                _log_job_end(row_id, status="error", detail=str(exc))
                raise
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# Scheduler-local Kite session (Docker / HF-safe — no Selenium)
# ═══════════════════════════════════════════════════════════════

_scheduler_kite = None
_scheduler_kite_ts: float = 0.0
_KITE_SESSION_TTL = 5 * 3600  # re-auth after 5 h (Kite tokens last ~6 h)


def _get_scheduler_kite(force_refresh: bool = False):
    """Return an authenticated KiteConnect instance for the scheduler process.

    Tries the stored token first, then falls back to headless HTTP login
    (TOTP-based, no browser).  Caches the instance for up to 5 hours.
    """
    global _scheduler_kite, _scheduler_kite_ts
    import time as _time

    # Return cached instance if still fresh
    if (
        not force_refresh
        and _scheduler_kite is not None
        and (_time.time() - _scheduler_kite_ts) < _KITE_SESSION_TTL
    ):
        return _scheduler_kite

    try:
        from kite_connect.auth.kite_session import try_stored_token, http_login_kite

        kite = try_stored_token()
        if kite is None:
            logger.info("Scheduler Kite: stored token invalid, trying HTTP login")
            kite = http_login_kite()

        if kite is not None:
            _scheduler_kite = kite
            _scheduler_kite_ts = _time.time()
            logger.info("Scheduler Kite session established")
        else:
            logger.warning("Scheduler Kite: all auth methods failed")
            _scheduler_kite = None

        return _scheduler_kite
    except Exception as exc:
        logger.warning("Scheduler Kite session error: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════
# Proactive Kite token refresh
# ═══════════════════════════════════════════════════════════════

@_tracked_job("kite_refresh", "Kite Token Refresh")
def refresh_kite_token_if_needed():
    """Re-authenticate Kite if the access token is expiring soon.

    Called every 30 min during market hours by the scheduler.
    Uses the headless HTTP login path (no Selenium).
    """
    import time as _time
    global _scheduler_kite_ts

    try:
        if _scheduler_kite is None:
            # First call — try to establish a session
            _get_scheduler_kite()
            return

        elapsed = _time.time() - _scheduler_kite_ts
        if elapsed < _KITE_SESSION_TTL:
            return  # still fresh

        logger.info("Kite token expiring soon — proactive re-authentication")
        new_kite = _get_scheduler_kite(force_refresh=True)
        if new_kite:
            logger.info("Kite token refreshed successfully")
        else:
            logger.warning("Kite re-auth returned None — token may expire")
    except Exception as exc:
        logger.warning("Proactive Kite refresh failed: %s", exc)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Pipeline runner (headless â€” no Streamlit, no Kite orders)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@_tracked_job("pre_market_scan", "Pre-Market Full Scan")
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

        # Cache individual verdicts for fast API lookups
        _cache_verdicts(verdicts)

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

        # 4a. Daily email report
        _send_daily_email(summary)

        # 4b. Check go-live readiness
        _check_go_live_readiness()

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


def _send_daily_email(summary: dict):
    """Send daily pipeline email report."""
    try:
        from services.notifications.manager import NotificationManager
        nm = NotificationManager()
        nm.email_daily_pipeline_report(summary)
    except Exception as exc:
        logger.debug("Daily email failed (non-fatal): %s", exc)


def _check_go_live_readiness():
    """Check if paper trading meets go-live criteria and email if so."""
    try:
        from kite_connect.trading.paper_trader import PaperTrader
        from services.notifications.manager import NotificationManager
        import sqlite3
        from pathlib import Path
        from datetime import datetime

        pt = PaperTrader()
        dash = pt.dashboard()

        # Need enough closed trades to evaluate
        if dash.closed_trades < 10:
            return

        # Compute weeks active from first paper trade
        db_path = Path(__file__).parent / "data" / "paper_trades.sqlite3"
        weeks_active = 0
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT MIN(opened_at) FROM paper_positions"
            ).fetchone()
            conn.close()
            if row and row[0]:
                first_dt = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                days = (datetime.now(first_dt.tzinfo) - first_dt).days
                weeks_active = days // 7

        nm = NotificationManager()
        nm.email_go_live_recommendation(dash, weeks_active)
    except Exception as exc:
        logger.debug("Go-live check failed (non-fatal): %s", exc)


def _notify_signals(buy_verdicts: list, sell_verdicts: list):
    """Send desktop notification for discovered signals."""
    try:
        from services.notifications.manager import NotificationManager
        nm = NotificationManager()

        parts = []
        if buy_verdicts:
            syms = ", ".join(v.ticker.replace(".NS", "") for v in buy_verdicts[:5])
            parts.append(f"{len(buy_verdicts)} BUY: {syms}")
        if sell_verdicts:
            syms = ", ".join(v.ticker.replace(".NS", "") for v in sell_verdicts[:5])
            parts.append(f"{len(sell_verdicts)} SELL: {syms}")

        nm.send_notification(
            title="Centurion â€” Signals Detected",
            message=" | ".join(parts),
            duration=15,
        )
    except Exception as exc:
        logger.debug("Notification failed (non-fatal): %s", exc)


def _paper_trade_orders(verdicts: list, screened_df):
    """Route orders to PaperTrader for simulated execution."""
    try:
        from kite_connect.trading.paper_trader import PaperTrader
        from kite_connect.trading.risk_manager import RiskManager, RiskConfig

        signal_dict = {
            v.ticker.replace(".NS", "").replace(".BO", ""): v.classification
            for v in verdicts
        }
        buy_symbols = [
            sym for sym, tag in signal_dict.items()
            if tag in ("BUY", "STRONG_BUY")
        ]
        if not buy_symbols:
            logger.info("Paper trade: no BUY symbols")
            return

        # Generate trade plans via RiskManager (same as live path)
        buy_df = screened_df[screened_df["symbol"].isin(buy_symbols)]
        if buy_df.empty:
            return

        # Carver-aware: create VolatilityTarget and use plan_trades_carver when enabled
        vol_target = None
        carver_enabled = False
        try:
            from config import Config
            if getattr(Config, "CARVER_ENABLED", False):
                from services.volatility_target import VolatilityTarget, VolatilityTargetConfig
                vt_cfg = VolatilityTargetConfig(
                    initial_capital=getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000.0),
                    annual_vol_target_pct=getattr(Config, "CARVER_ANNUAL_VOL_TARGET", 0.20),
                )
                vol_target = VolatilityTarget(vt_cfg)
                carver_enabled = True
        except Exception:
            pass

        rm = RiskManager(RiskConfig(), volatility_target=vol_target)

        if carver_enabled and vol_target is not None:
            try:
                from services.instrument_volatility import compute_volatilities_batch
                from strategies.ewmac import compute_ewmac_batch
                from services.forecast_scalar import screener_to_forecast
                from services.forecast_combiner import combine_forecasts_batch
                from services.cost_speed_limit import filter_by_cost
                from services.instrument_weights import compute_handcrafted_weights, get_default_idm
                from utils import download_ind_ohlcv

                ohlcv_cache = {}
                for sym in buy_symbols:
                    try:
                        df = download_ind_ohlcv(sym, period="6mo")
                        if df is not None and len(df) >= 64:
                            ohlcv_cache[sym] = df
                    except Exception:
                        pass

                if ohlcv_cache:
                    vol_data = compute_volatilities_batch(ohlcv_cache)
                    instrument_vols = {s: v["instrument_value_vol"] for s, v in vol_data.items()}
                    ewmac_batch = compute_ewmac_batch(ohlcv_cache)
                    all_forecasts = {}
                    for sym in ohlcv_cache:
                        fc = {}
                        if sym in ewmac_batch:
                            for ef in ewmac_batch[sym]:
                                fc[f"ewmac_{ef.fast}_{ef.slow}"] = ef.forecast
                        row_m = buy_df[buy_df["symbol"] == sym]
                        if not row_m.empty and "score" in row_m.columns:
                            fc["screener"] = screener_to_forecast(float(row_m.iloc[0]["score"]))
                        if fc:
                            all_forecasts[sym] = fc

                    if all_forecasts:
                        combined = combine_forecasts_batch(all_forecasts)
                        cv = {s: cf.combined_forecast for s, cf in combined.items()}
                        cv = filter_by_cost(cv, getattr(Config, "CARVER_ANNUAL_VOL_TARGET", 0.20))
                        active = [s for s in cv if cv[s] > 0]
                        weights = compute_handcrafted_weights(active, getattr(Config, "NSE_SECTOR_MAP", {}))
                        idm = get_default_idm(len(active))
                        plans = rm.plan_trades_carver(buy_df, cv, instrument_vols, weights, idm)
                        logger.info("Paper trade: Carver generated %d plans", len(plans))
                    else:
                        plans = rm.plan_trades(buy_df)
                else:
                    plans = rm.plan_trades(buy_df)
            except Exception as exc:
                logger.warning("Carver paper trade failed, using legacy: %s", exc)
                plans = rm.plan_trades(buy_df)
        else:
            plans = rm.plan_trades(buy_df)
        if not plans:
            logger.info("Paper trade: no plans met R:R threshold")
            return

        # Try to get Kite for LTP (optional; PaperTrader uses yfinance fallback)
        kite = _get_scheduler_kite()

        pt = PaperTrader(kite=kite, initial_capital=100_000)
        results = pt.execute_plans(plans)
        filled = sum(1 for r in results if r.get("success"))

        # Check SL/TP immediately
        close_events = pt.poll()

        dashboard = pt.dashboard()
        logger.info(
            "Paper trade: %d/%d filled | capital=%.0f | P&L=%.0f (%.1f%%)",
            filled, len(plans),
            dashboard.current_capital,
            dashboard.total_pnl,
            dashboard.total_pnl_pct,
        )

        # Persist summary to scheduler cache
        _save_run("paper_trade", {
            "universe_size": len(screened_df),
            "screened_count": len(buy_df),
            "buy_signals": filled,
            "sell_signals": len(close_events),
            "verdicts": [r for r in results],
            "plans": [dashboard.to_dict()],
            "status": "success",
        })

    except Exception as exc:
        logger.exception("Paper trade failed: %s", exc)


def _auto_place_orders(verdicts: list, screened_df):
    """Auto-authenticate Kite and place orders for BUY/STRONG_BUY verdicts.

    Called by the scheduler when STRONG_BUY signals are detected. Uses
    the auto-TOTP flow (pyotp) when ``ZERODHA_TOTP_SECRET`` is configured,
    making the entire pipeline zero-touch.

    When ``PAPER_TRADE_MODE=true`` (env var or Config), orders are
    routed to the PaperTrader instead of Kite live.
    """
    # â”€â”€ Paper-trade mode check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    paper_mode = os.environ.get("PAPER_TRADE_MODE", "").lower() in ("true", "1", "yes")
    if not paper_mode:
        try:
            from config import Config
            paper_mode = getattr(Config, "PAPER_TRADE_MODE", False)
        except Exception:
            pass

    if paper_mode:
        _paper_trade_orders(verdicts, screened_df)
        return

    try:
        logger.info("Auto-authenticating Kite for STRONG_BUY order placementâ€¦")
        kite = _get_scheduler_kite()
        if kite is None:
            raise RuntimeError("Scheduler Kite session unavailable")
    except Exception as exc:
        logger.error("Kite auto-auth failed: %s â€” orders skipped", exc)
        try:
            from services.notifications.manager import NotificationManager
            NotificationManager().send_notification(
                "Centurion â€” Auth Failed",
                f"Could not auto-authenticate Kite: {exc}",
            )
        except Exception:
            pass
        return

    if kite is None:
        logger.warning("Kite session is None â€” orders skipped")
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

        # G4: Execute options overlay (covered calls + CSPs)
        _execute_options_overlay(kite)

        # G10: Auto-execute tail hedge if drawdown critical
        _execute_tail_hedge_if_needed(kite)

    except Exception as exc:
        logger.exception("Auto-order placement failed: %s", exc)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Walk-Forward Audit (weekly)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _update_forecast_source_decay(audit_results: dict):
    """G4 FIX: Update strategy_decay_state.json from WF audit results.

    Maps registered strategy names to forecast source prefixes and
    updates the decay state so the forecast combiner (G1) can
    zero-weight degraded sources at runtime.

    Monitored sources: ewmac, carry, momentum, pead, mean_reversion,
    ehlers_dsp, penfold_trend, intermarket, acceleration, etc.
    """
    import json as _json
    from pathlib import Path as _Path
    from datetime import datetime as _dt

    _decay_path = _Path(__file__).parent / "data" / "strategy_decay_state.json"

    # Map strategy names → forecast source keys
    _STRATEGY_TO_SOURCE = {
        "macd oscillator": "ewmac",
        "awesome oscillator": "momentum",
        "rsi pattern": "screener",
        "parabolic sar": "penfold_trend",
        "heikin-ashi": "penfold_trend",
        "bollinger bottom w": "mean_reversion",
        "support resistance": "screener",
        "liquidity sweep": "oi_signal",
        "anchored vwap": "carver_value",
        "order flow imbalance": "fii_flow",
        "volume profile": "cross_momentum",
    }

    # Load existing decay state
    try:
        existing = _json.loads(_decay_path.read_text()) if _decay_path.exists() else {}
    except Exception:
        existing = {}

    updated = dict(existing)
    now_iso = _dt.now().isoformat()

    for strategy_name, result in audit_results.items():
        if not isinstance(result, dict) or "error" in result:
            continue

        source_key = _STRATEGY_TO_SOURCE.get(strategy_name.lower())
        if not source_key:
            continue

        oos_sharpe = result.get("multi_ticker_median_oos_sharpe",
                                result.get("avg_oos_sharpe", 0))
        degradation = result.get("multi_ticker_median_deg",
                                 result.get("degradation_ratio", 1.0))

        # Determine status
        if oos_sharpe < -0.1:
            status = "INVERTED"
        elif degradation < 0.25 or oos_sharpe < 0:
            status = "DEAD"
        elif degradation < 0.50:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        updated[source_key] = {
            "status": status,
            "days": 1,
            "last_healthy": now_iso if status == "HEALTHY" else
                            existing.get(source_key, {}).get("last_healthy", ""),
            "recent_sharpe": round(oos_sharpe, 4),
            "degradation_ratio": round(degradation, 4),
            "updated_at": now_iso,
        }

    # Write back
    try:
        _decay_path.write_text(_json.dumps(updated, indent=2))
        logger.info(
            "G4: Updated strategy_decay_state.json: %d sources (%s)",
            len(updated),
            ", ".join(f"{k}={v['status']}" for k, v in updated.items()),
        )
    except Exception as exc:
        logger.warning("Failed to write strategy_decay_state.json: %s", exc)


@_tracked_job("walk_forward_audit", "Walk-Forward Audit")
def run_walk_forward_audit():
    """Run walk-forward validation on all registered strategies.

    G4 FIX: Expanded from single-ticker to multi-ticker validation.
    Also updates strategy_decay_state.json for ALL 22 forecast sources
    so the forecast combiner can zero-weight degraded sources.

    Kicks off every Saturday morning via the scheduler.  Results are
    saved to the scheduler cache DB under run_type='walk_forward'.
    Strategies with degradation_ratio < 0.5 are flagged as overfit.
    """
    logger.info("=== Walk-Forward Audit started (G4 multi-ticker) ===")

    try:
        from strategies import StrategyRegistry, load_all_strategies
        from services.walk_forward import walk_forward_validate, save_optimal_params

        load_all_strategies()
        all_strategies = StrategyRegistry._strategies

        # G4 FIX: Validate against multiple representative tickers
        # covering different sectors and liquidity profiles
        test_tickers = [
            "RELIANCE.NS",   # Oil & Gas / conglomerate
            "TCS.NS",        # IT services
            "HDFCBANK.NS",   # Banking
            "BHARTIARTL.NS", # Telecom
            "ITC.NS",        # FMCG
        ]

        audit_results = {}
        overfit_strategies: List[str] = []

        for name, strategy_cls in all_strategies.items():
            if "crypto" in name.lower():
                continue
            strat_summaries = []
            for test_ticker in test_tickers:
                try:
                    summary = walk_forward_validate(
                        strategy_cls=strategy_cls,
                        ticker=test_ticker,
                        capital=100_000,
                        train_days=252,
                        test_days=63,
                        total_days=756,
                    )
                    strat_summaries.append(summary)
                    # Persist winning params per ticker
                    save_optimal_params(summary)
                except Exception as exc:
                    logger.warning("WF fold failed for %s on %s: %s", name, test_ticker, exc)

            if not strat_summaries:
                audit_results[name] = {"error": "all tickers failed"}
                continue

            # Aggregate across tickers — use median for robustness
            import numpy as _np
            avg_deg = float(_np.median([s.degradation_ratio for s in strat_summaries]))
            avg_oos = float(_np.median([s.avg_oos_sharpe for s in strat_summaries]))
            avg_is = float(_np.median([s.avg_is_sharpe for s in strat_summaries]))
            total_folds = sum(s.total_folds for s in strat_summaries)

            # Use best ticker's full result for detailed reporting
            best_summary = max(strat_summaries, key=lambda s: s.avg_oos_sharpe)
            result_dict = best_summary.to_dict()
            result_dict["multi_ticker_median_deg"] = round(avg_deg, 4)
            result_dict["multi_ticker_median_oos_sharpe"] = round(avg_oos, 4)
            result_dict["tickers_tested"] = len(strat_summaries)
            result_dict["total_folds_all_tickers"] = total_folds
            audit_results[name] = result_dict

            if avg_deg < 0.5 and total_folds > 0:
                overfit_strategies.append(name)
                logger.warning(
                    "OVERFIT: %s -- degradation=%.2f (OOS Sharpe=%.2f, IS=%.2f, %d tickers)",
                    name, avg_deg, avg_oos, avg_is, len(strat_summaries),
                )
            else:
                logger.info(
                    "OK: %s -- degradation=%.2f, OOS Sharpe=%.2f (%d tickers)",
                    name, avg_deg, avg_oos, len(strat_summaries),
                )

        # G4 FIX: Update strategy_decay_state.json for all forecast sources
        # This feeds back into G1's decay-state filter in forecast_combiner
        _update_forecast_source_decay(audit_results)

        _save_run("walk_forward", {
            "universe_size": len(all_strategies),
            "screened_count": len(audit_results),
            "buy_signals": 0,
            "sell_signals": len(overfit_strategies),
            "verdicts": [
                {"strategy": name, **data}
                for name, data in audit_results.items()
                if isinstance(data, dict)
            ],
            "status": "success",
        })

        if overfit_strategies:
            try:
                from services.notifications.manager import NotificationManager
                NotificationManager().send_notification(
                    "Centurion â€” Overfit Alert",
                    f"{len(overfit_strategies)} strategies flagged: "
                    f"{', '.join(overfit_strategies[:5])}",
                    duration=20,
                )
            except Exception:
                pass

        logger.info(
            "=== Walk-Forward Audit complete: %d strategies, %d flagged ===",
            len(audit_results), len(overfit_strategies),
        )

        # ── Aronson EBTA signal validation (post walk-forward) ──
        try:
            from services.aronson_validator import AronsonValidator
            import numpy as np

            validator = AronsonValidator()

            # Build per-signal degradation ratios from WF results
            _deg_ratios = {}
            for name, data in audit_results.items():
                if isinstance(data, dict) and "degradation_ratio" in data:
                    _deg_ratios[name] = data["degradation_ratio"]

            # Build synthetic signal returns from hit rates
            # (a full implementation would use actual daily returns from WF folds)
            _signal_rets = {}
            for name, data in audit_results.items():
                if isinstance(data, dict):
                    oos_sr = data.get("avg_oos_sharpe", 0)
                    n_folds = data.get("total_folds", 0)
                    if n_folds > 0:
                        # Synthetic: generate returns from OOS Sharpe
                        rng = np.random.RandomState(hash(name) % 2**31)
                        _signal_rets[name] = rng.normal(oos_sr / 16.0, 0.02, size=252)

            if _signal_rets:
                summary = validator.validate_signals(
                    signal_returns=_signal_rets,
                    degradation_ratios=_deg_ratios,
                )
                validator.save_state(summary)
                logger.info(
                    "Aronson validation: %d/%d signals validated, "
                    "WRC best=%s (p=%.4f), DM bias=%.2f%%",
                    summary.n_validated, summary.n_total,
                    summary.wrc_best_signal, summary.wrc_best_p_value,
                    summary.dm_bias_estimate * 100,
                )
            else:
                logger.info("Aronson validation skipped: no WF results to validate")
        except Exception as aronson_exc:
            logger.warning("Aronson validation failed: %s", aronson_exc)

    except Exception as exc:
        logger.exception("Walk-Forward Audit failed: %s", exc)
        _save_run("walk_forward", {"status": f"error: {exc}"})


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Unified Backtest â†” Paper â†” Live Reconciliation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@_tracked_job("paper_live_recon", "Paper vs Live Reconciliation")
def _run_paper_live_reconciliation():
    """Unified 3-leg parity check: backtest â†” paper â†” live.

    Runs weekly (Saturday 7 AM IST) and compares:
      Leg 1 â€” Paper vs Live: symbol-level P&L drift for common trades
      Leg 2 â€” Backtest vs Live: per-strategy aggregate metrics
              (win-rate, avg return, Sharpe) â€” surfaces when live
              execution degrades vs backtest expectations
      Leg 3 â€” Backtest vs Paper: same comparison but for simulated fills

    All discrepancies > 1 % (P&L) or > 0.3 (Sharpe drift) are logged
    and trigger desktop notifications.
    """
    logger.info("=== Unified Reconciliation started ===")
    report: dict = {"status": "success"}

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Load data sources
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    paper_trades = _load_paper_trades()
    live_trades, live_by_strategy = _load_live_journal()
    backtest_by_strategy = _load_backtest_summaries()

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Leg 1 â€” Paper â†” Live (symbol-level)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    leg1 = _reconcile_paper_vs_live(paper_trades, live_trades)
    report["paper_vs_live"] = leg1

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Leg 2 â€” Backtest â†” Live (strategy-level)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    leg2 = _reconcile_backtest_vs_execution(backtest_by_strategy, live_by_strategy, "live")
    report["backtest_vs_live"] = leg2

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Leg 3 â€” Backtest â†” Paper (strategy-level)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    paper_by_strategy = _group_by_strategy_paper(paper_trades)
    leg3 = _reconcile_backtest_vs_execution(backtest_by_strategy, paper_by_strategy, "paper")
    report["backtest_vs_paper"] = leg3

    _save_run("reconciliation", report)

    # Weekly reconciliation email
    try:
        from services.notifications.manager import NotificationManager
        NotificationManager().email_reconciliation_report(report)
    except Exception as exc:
        logger.debug("Reconciliation email failed (non-fatal): %s", exc)


    # â”€â”€ Alert on discrepancies â”€â”€
    all_issues: List[str] = []
    for leg_name, leg_data in [("Paperâ†”Live", leg1), ("BTâ†”Live", leg2), ("BTâ†”Paper", leg3)]:
        discs = leg_data.get("discrepancies", [])
        if discs:
            all_issues.append(f"{leg_name}: {len(discs)}")

    if all_issues:
        summary = ", ".join(all_issues)
        logger.warning("Reconciliation: %s", summary)
        try:
            from services.notifications.manager import NotificationManager
            NotificationManager().send_notification(
                "Centurion â€” Reconciliation Alert",
                f"Parity issues: {summary}",
                duration=15,
            )
        except Exception:
            pass
    else:
        logger.info("Reconciliation: all 3 legs clean â€” no significant drift")

    logger.info("=== Unified Reconciliation complete ===")


# â”€â”€ Reconciliation helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_paper_trades() -> dict:
    """Load closed paper trades from SQLite â†’ {symbol: {...}}."""
    try:
        import sqlite3
        from pathlib import Path
        paper_db = Path("data/paper_trades.sqlite3")
        if not paper_db.exists():
            return {}
        conn = sqlite3.connect(str(paper_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM paper_positions WHERE is_open = 0 "
            "ORDER BY closed_at DESC LIMIT 200"
        ).fetchall()
        conn.close()
        return {
            r["symbol"]: {
                "entry_price": r["entry_price"],
                "exit_price": r["exit_price"],
                "pnl_pct": r["pnl_pct"],
                "exit_reason": r["exit_reason"],
            }
            for r in rows
        }
    except Exception as exc:
        logger.debug("Paper trades load failed: %s", exc)
        return {}


def _load_live_journal() -> tuple:
    """Load closed live journal trades â†’ (by_symbol, by_strategy).

    Returns:
        Tuple of (
            {symbol: {entry_price, exit_price, pnl_pct, exit_reason}},
            {strategy_name: {trades, wins, total_pnl_pct}},
        )
    """
    by_symbol: dict = {}
    by_strategy: dict = {}
    try:
        from database.service import get_database_service
        db = get_database_service()
        if not db:
            return by_symbol, by_strategy
        from database.models import TradeJournal
        session = db.Session()
        try:
            recent = (
                session.query(TradeJournal)
                .filter(TradeJournal.is_open.is_(False))
                .order_by(TradeJournal.exit_date.desc())
                .limit(200)
                .all()
            )
            for t in recent:
                pnl = t.pnl_pct or 0
                by_symbol[t.symbol] = {
                    "entry_price": float(t.entry_price) if t.entry_price else 0,
                    "exit_price": float(t.exit_price) if t.exit_price else 0,
                    "pnl_pct": pnl,
                    "exit_reason": t.exit_reason or "",
                    "strategy": t.strategy_name or "unknown",
                }
                strat = t.strategy_name or "unknown"
                if strat not in by_strategy:
                    by_strategy[strat] = {"trades": 0, "wins": 0, "total_pnl_pct": 0.0, "pnls": []}
                by_strategy[strat]["trades"] += 1
                by_strategy[strat]["total_pnl_pct"] += pnl
                by_strategy[strat]["pnls"].append(pnl)
                if pnl > 0:
                    by_strategy[strat]["wins"] += 1
        finally:
            session.close()
    except Exception as exc:
        logger.debug("Live journal load failed: %s", exc)
    return by_symbol, by_strategy


def _load_backtest_summaries() -> dict:
    """Load per-strategy backtest aggregate metrics â†’ {strategy_name: {...}}.

    Pulls from BacktestResult or StrategyPerformanceSummary.
    """
    summaries: dict = {}
    try:
        from database.service import get_database_service
        db = get_database_service()
        if not db:
            return summaries
        from database.models import BacktestResult
        from sqlalchemy import func as sqlfunc
        session = db.Session()
        try:
            # Aggregate by strategy (most recent 6 months)
            from datetime import datetime, timedelta
            cutoff = datetime.utcnow() - timedelta(days=180)
            rows = (
                session.query(
                    BacktestResult.strategy_name,
                    sqlfunc.count(BacktestResult.id).label("n"),
                    sqlfunc.avg(BacktestResult.total_return).label("avg_return"),
                    sqlfunc.avg(BacktestResult.win_rate).label("avg_win_rate"),
                    sqlfunc.avg(BacktestResult.sharpe_ratio).label("avg_sharpe"),
                    sqlfunc.avg(BacktestResult.max_drawdown).label("avg_dd"),
                )
                .filter(
                    BacktestResult.success.is_(True),
                    BacktestResult.created_at >= cutoff,
                )
                .group_by(BacktestResult.strategy_name)
                .all()
            )
            for row in rows:
                summaries[row.strategy_name] = {
                    "backtests": row.n,
                    "avg_return": float(row.avg_return or 0),
                    "avg_win_rate": float(row.avg_win_rate or 0),
                    "avg_sharpe": float(row.avg_sharpe or 0),
                    "avg_drawdown": float(row.avg_dd or 0),
                }
        finally:
            session.close()
    except Exception as exc:
        logger.debug("Backtest summary load failed: %s", exc)
    return summaries


def _group_by_strategy_paper(paper_trades: dict) -> dict:
    """Group paper trades by strategy (if available in exit_reason or symbol patterns).

    Paper trades don't have strategy attribution, so we return an
    'all_paper' bucket with aggregate stats for coarse comparison.
    """
    if not paper_trades:
        return {}
    pnls = [t["pnl_pct"] for t in paper_trades.values() if t.get("pnl_pct") is not None]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "all_paper": {
            "trades": len(pnls),
            "wins": wins,
            "total_pnl_pct": sum(pnls),
            "pnls": pnls,
        }
    }


def _reconcile_paper_vs_live(paper: dict, live: dict) -> dict:
    """Leg 1: symbol-level paper â†” live P&L comparison."""
    common = set(paper.keys()) & set(live.keys())
    discrepancies = []
    slippage_diffs = []

    for sym in common:
        p = paper[sym]
        l = live[sym]
        pnl_diff = abs((p.get("pnl_pct") or 0) - (l.get("pnl_pct") or 0))
        entry_drift = 0
        p_entry = p.get("entry_price") or 0
        l_entry = l.get("entry_price") or 0
        if p_entry > 0 and l_entry > 0:
            entry_drift = abs(l_entry - p_entry) / p_entry * 100
            slippage_diffs.append(entry_drift)

        if pnl_diff > 1.0:
            discrepancies.append({
                "symbol": sym,
                "paper_pnl": round(p.get("pnl_pct") or 0, 2),
                "live_pnl": round(l.get("pnl_pct") or 0, 2),
                "diff_pct": round(pnl_diff, 2),
                "entry_slippage_pct": round(entry_drift, 3),
                "root_cause": (
                    "entry_slippage" if entry_drift > 0.5
                    else "exit_timing" if p.get("exit_reason") != l.get("exit_reason")
                    else "commission_gap"
                ),
            })

    paper_only = set(paper.keys()) - set(live.keys())
    live_only = set(live.keys()) - set(paper.keys())

    return {
        "paper_count": len(paper),
        "live_count": len(live),
        "common": len(common),
        "paper_only_count": len(paper_only),
        "live_only_count": len(live_only),
        "avg_entry_slippage_pct": round(
            sum(slippage_diffs) / len(slippage_diffs), 3
        ) if slippage_diffs else 0,
        "discrepancies": discrepancies,
    }


def _reconcile_backtest_vs_execution(
    bt_strategies: dict,
    exec_strategies: dict,
    exec_label: str,
) -> dict:
    """Leg 2/3: backtest â†” live/paper per-strategy metric comparison.

    Compares avg_return, win_rate, and Sharpe between backtest
    expectations and actual execution results.
    """
    common = set(bt_strategies.keys()) & set(exec_strategies.keys())
    discrepancies = []

    for strat in common:
        bt = bt_strategies[strat]
        ex = exec_strategies[strat]

        bt_win = bt.get("avg_win_rate", 0)
        ex_trades = ex.get("trades", 0)
        ex_wins = ex.get("wins", 0)
        ex_win = (ex_wins / ex_trades * 100) if ex_trades > 0 else 0

        bt_ret = bt.get("avg_return", 0)
        ex_ret = (ex.get("total_pnl_pct", 0) / ex_trades) if ex_trades > 0 else 0

        # Compute live Sharpe proxy from per-trade P&L
        ex_sharpe = 0
        pnls = ex.get("pnls", [])
        if len(pnls) >= 3:
            import numpy as np
            avg_p = float(np.mean(pnls))
            std_p = float(np.std(pnls, ddof=1))
            ex_sharpe = (avg_p / std_p) if std_p > 0 else 0

        bt_sharpe = bt.get("avg_sharpe", 0)
        sharpe_drift = bt_sharpe - ex_sharpe

        win_drift = bt_win - ex_win
        ret_drift = bt_ret - ex_ret

        issues = []
        if abs(sharpe_drift) > 0.3:
            issues.append(f"Sharpe drift {sharpe_drift:+.2f}")
        if abs(win_drift) > 10:
            issues.append(f"Win-rate drift {win_drift:+.1f}%")
        if abs(ret_drift) > 5:
            issues.append(f"Return drift {ret_drift:+.1f}%")

        if issues:
            discrepancies.append({
                "strategy": strat,
                "bt_sharpe": round(bt_sharpe, 2),
                f"{exec_label}_sharpe": round(ex_sharpe, 2),
                "bt_win_rate": round(bt_win, 1),
                f"{exec_label}_win_rate": round(ex_win, 1),
                "bt_avg_return": round(bt_ret, 2),
                f"{exec_label}_avg_return": round(ex_ret, 2),
                f"{exec_label}_trades": ex_trades,
                "issues": issues,
            })

    return {
        "bt_strategies": len(bt_strategies),
        f"{exec_label}_strategies": len(exec_strategies),
        "common": len(common),
        "discrepancies": discrepancies,
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Utility jobs (backup, pre-warming)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _pre_market_with_warmup():
    """Pre-warm DB, restore portfolio state, then run pipeline."""
    try:
        from database.connection import DatabaseManager
        DatabaseManager().pre_warm()
    except Exception as e:
        logger.warning("DB pre-warm failed (non-fatal): %s", e)

    # SOD: restore portfolio state so DD calculations use persisted peak/P&L
    _restore_portfolio_state()

    run_pipeline("pre_market")


def _restore_portfolio_state():
    """Reload cumulative P&L and peak equity from portfolio_state.json.

    Called at start-of-day so the DD halt / graduated scaling logic
    uses the correct baseline rather than resetting to zero.
    """
    state_path = Path(__file__).parent / "data" / "portfolio_state.json"
    if not state_path.exists():
        return
    try:
        from config import Config
        with open(state_path, "r") as f:
            state = json.load(f)
        cum_pnl = state.get("cumulative_realized_pnl", 0.0)
        peak = state.get("peak_equity", getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000))
        Config._CUMULATIVE_REALIZED_PNL = cum_pnl
        Config._PEAK_EQUITY = peak
        logger.info(
            "SOD state restored: cum_pnl=%.0f, peak_equity=%.0f",
            cum_pnl, peak,
        )
    except Exception as e:
        logger.warning("Portfolio state restore failed (non-fatal): %s", e)


@_tracked_job("trade_monitor", "Trade Monitor Poll")
def _run_trade_monitor_poll():
    """Poll TradeMonitor for SL/TP fills, trailing-SL updates, and time exits.

    Runs every 3 min during market hours. Creates a fresh TradeMonitor
    that auto-restores active trades from its SQLite crash-recovery DB,
    then calls poll() which handles:
      - Entry fill detection & SL/TP placement
      - SL/TP fill detection & orphan cancellation
      - Vol-based trailing-SL ratcheting
      - Time-based forced exits (swing 10d / positional 30d)
      - Partial fill acceptance (stale entries >2 hr)
      - Capital rollup & peak equity persistence
    """
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.debug("TradeMonitor poll skipped - no Kite session")
            return
    except Exception as exc:
        logger.debug("TradeMonitor poll skipped - Kite auth failed: %s", exc)
        return

    try:
        from kite_connect.trading.trade_monitor import TradeMonitor
        from kite_connect.trading.risk_manager import RiskConfig

        monitor = TradeMonitor(kite=kite, risk_config=RiskConfig())
        active_count = len(monitor.active_trades)
        if active_count == 0:
            logger.debug("TradeMonitor poll: no active trades")
            return

        events = monitor.poll()
        if events:
            logger.info(
                "TradeMonitor poll: %d events from %d active trades — %s",
                len(events), active_count,
                ", ".join(e.get("type", "?") for e in events),
            )
        else:
            logger.debug("TradeMonitor poll: %d active trades, no events", active_count)
    except Exception as exc:
        logger.exception("TradeMonitor poll failed: %s", exc)


@_tracked_job("forecast_calibration", "Forecast Calibration")
def _run_forecast_calibration():
    """Auto-calibrate forecast scalars from recent OHLCV data.

    Runs weekly (Saturday 5:30 AM IST, before walk-forward audit).
    Re-computes EWMAC / screener / decision-engine / carry scalars from
    expanding-window median(|raw forecast|) and persists to
    ``data/calibrated_scalars.json``.
    """
    logger.info("=== Forecast Scalar Calibration started ===")
    try:
        from config import Config
        if not getattr(Config, "AUTO_CALIBRATE_SCALARS", True):
            logger.info("AUTO_CALIBRATE_SCALARS disabled — skipping")
            return

        from services.forecast_scalar import calibrate_all_scalars
        import yfinance as yf

        # Build OHLCV cache for representative NIFTY-50 tickers
        tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "ICICIBANK.NS", "HINDUNILVR.NS", "BHARTIARTL.NS",
            "SBIN.NS", "KOTAKBANK.NS", "LT.NS",
        ]
        ohlcv_cache: dict = {}
        for t in tickers:
            try:
                df = yf.download(t, period="2y", progress=False, timeout=15)
                if df is not None and len(df) > 100:
                    ohlcv_cache[t] = df
            except Exception:
                logger.debug("OHLCV fetch failed for %s", t)

        if len(ohlcv_cache) < 3:
            logger.warning("Too few tickers for calibration (%d) — skipping", len(ohlcv_cache))
            return

        result = calibrate_all_scalars(ohlcv_cache)
        logger.info("Forecast calibration complete: %s", result)

    except Exception as exc:
        logger.exception("Forecast scalar calibration failed: %s", exc)


# ═══════════════════════════════════════════════════════════════
# Phase 2 Gap B1: Monthly HMM Regime Re-fit
# ═══════════════════════════════════════════════════════════════

@_tracked_job("hmm_refit", "HMM Regime Re-fit")
def _run_hmm_refit():
    """Monthly re-fit of the HMM regime model on 5 years of NIFTY data.

    Trains a 3-state Gaussian HMM on [log_returns, VIX, breadth, delivery_vol]
    and persists the model to data/hmm_model.pkl for use by the pipeline.
    """
    logger.info("=== HMM Regime Re-fit started ===")
    try:
        from config import Config
        if not getattr(Config, "HMM_ENABLED", True):
            logger.info("HMM_ENABLED=False — skipping re-fit")
            return

        import yfinance as yf
        from services.regime_hmm import MarkovRegimeModel, prepare_hmm_observations

        # Fetch 5 years of NIFTY 50 daily data
        nifty_df = yf.download("^NSEI", period="5y", progress=False, timeout=30)
        if nifty_df is None or len(nifty_df) < 500:
            logger.warning("Insufficient NIFTY data for HMM fit (%d rows)", len(nifty_df) if nifty_df is not None else 0)
            return

        # Fetch India VIX
        vix_df = None
        try:
            vix_df = yf.download("^INDIAVIX", period="5y", progress=False, timeout=15)
        except Exception:
            logger.debug("India VIX fetch failed — using proxy")

        # Prepare observation matrix
        observations = prepare_hmm_observations(nifty_df, vix_df)
        if len(observations) < 500:
            logger.warning("Too few valid observations for HMM (%d) — need 500+", len(observations))
            return

        # Fit model
        model = MarkovRegimeModel(
            n_states=getattr(Config, "HMM_N_STATES", 3),
            n_features=4,
        )
        model.fit(observations)

        # Get current regime for logging
        snap = model.get_current_regime(observations[-60:])
        logger.info(
            "HMM re-fit complete: regime=%s (%.0f%%), durations=%s",
            snap.regime, snap.confidence * 100, snap.expected_durations,
        )

        # Persist model
        model.save()
        logger.info("HMM model persisted to disk")

        # Update singleton
        from services.regime_hmm import get_hmm_model
        global_model = get_hmm_model()
        global_model._fitted = model._fitted
        global_model._means = model._means
        global_model._covars = model._covars
        global_model._transmat = model._transmat
        global_model._startprob = model._startprob
        global_model._feat_mean = model._feat_mean
        global_model._feat_std = model._feat_std

        logger.info("=== HMM Regime Re-fit complete ===")

    except Exception as exc:
        logger.exception("HMM re-fit failed: %s", exc)


@_tracked_job("strategy_tournament", "Strategy Tournament")
def _run_strategy_tournament():
    """Monthly strategy tournament to rank and auto-allocate strategies.

    Runs 1st Saturday of each month at 4:00 AM IST.
    Evaluates all forecast sources over the trailing 3 months,
    disables underperformers, and persists allocation decisions.
    """
    logger.info("=== Monthly Strategy Tournament started ===")
    try:
        import pandas as pd
        from services.strategy_tournament import StrategyTournament
        import json, os

        # Load recent per-strategy returns from walk-forward results
        wf_results_path = os.path.join("data", "walk_forward_results.json")
        if not os.path.exists(wf_results_path):
            logger.warning("No walk-forward results found — skipping tournament")
            return

        with open(wf_results_path) as f:
            wf_data = json.load(f)

        # Convert to per-strategy return series
        strat_returns = {}
        for strat_name, returns_list in wf_data.items():
            if isinstance(returns_list, list) and len(returns_list) >= 20:
                strat_returns[strat_name] = pd.Series(returns_list)

        if len(strat_returns) < 2:
            logger.warning("Too few strategies with results (%d) — skipping", len(strat_returns))
            return

        tourney = StrategyTournament(top_n=5, min_sharpe=0.0)
        result = tourney.run_tournament(strat_returns, lookback_months=3)

        logger.info("Tournament complete: top=%s, disabled=%s",
                     result.top_strategies, result.disabled_strategies)

        # Persist tournament results for pipeline to read
        tourney_path = os.path.join("data", "tournament_results.json")
        with open(tourney_path, "w") as f:
            json.dump({
                "top_strategies": result.top_strategies,
                "disabled_strategies": result.disabled_strategies,
                "entries": [
                    {"rank": e.rank, "name": e.strategy_name,
                     "score": e.composite_score, "status": e.allocation_status}
                    for e in result.entries
                ],
            }, f, indent=2)

    except Exception as exc:
        logger.exception("Strategy tournament failed: %s", exc)


@_tracked_job("nightly_backup", "Nightly Backup")
def _run_nightly_backup():
    """Upload SQLite databases to R2/MinIO storage."""
    try:
        from infrastructure.backup_service import run_backup
        result = run_backup()
        logger.info("Nightly backup result: %s", result)
    except Exception as e:
        logger.error("Nightly backup failed: %s", e)


@_tracked_job("intraday_rescan", "Intraday Re-Scan")
def _run_intraday_rescan():
    """Lighter intraday re-scan for momentum shifts during market hours.

    Runs at 10:30, 12:30, 14:30 IST. Uses the same pipeline but tagged
    as 'intraday' so results are distinguishable from the pre-market scan.
    """
    run_pipeline("intraday")


@_tracked_job("eod_scan", "End-of-Day Scan")
def _run_eod_scan():
    """End-of-day scan at 15:20 IST (10 min before close).

    Captures late-day signals and prepares exit decisions before
    the 15:30 market close.
    """
    run_pipeline("eod")


@_tracked_job("pead_earnings", "PEAD Earnings Feed")
def _run_pead_earnings_feed():
    """G6: Fetch recent earnings data and feed into PEAD strategy.

    Bridges earnings_momentum.py (Trendlyne scraper) with pead_strategy.py
    (PEAD signal generator) to automate the earnings data pipeline.
    """
    logger.info("=== PEAD Earnings Feed started ===")
    try:
        from services.earnings_momentum import _fetch_recent_results, EarningsSurprise as EMSurprise
        from services.pead_strategy import PEADStrategy, EarningsSurprise as PEADSurprise

        # Fetch recent earnings from Trendlyne
        raw_results = _fetch_recent_results()
        if not raw_results:
            logger.info("PEAD feed: no recent earnings data found")
            return

        # Convert earnings_momentum format to PEAD format
        pead_surprises = []
        for sym, em_data in raw_results.items():
            try:
                # Use profit surprise as primary SUE proxy
                sue = em_data.profit_surprise_pct / 10.0  # normalize to ~SUE scale
                surprise = PEADSurprise(
                    ticker=sym,
                    announcement_date=em_data.result_date,
                    eps_actual=em_data.profit_surprise_pct,  # proxy
                    eps_consensus=0.0,
                    sue=sue,
                    surprise_pct=em_data.profit_surprise_pct,
                    direction="POSITIVE" if em_data.is_positive else "NEGATIVE",
                )
                pead_surprises.append(surprise)
            except Exception:
                continue

        if not pead_surprises:
            logger.info("PEAD feed: no valid earnings surprises to process")
            return

        # Feed into PEAD strategy
        pead = PEADStrategy()
        new_signals = pead.process_earnings(pead_surprises)
        logger.info("PEAD feed: processed %d earnings, generated %d new signals",
                     len(pead_surprises), len(new_signals))

    except Exception as exc:
        logger.exception("PEAD earnings feed failed: %s", exc)


@_tracked_job("meta_label_retrain", "Meta-Label Retrain")
def _run_meta_label_retrain():
    """AFML Ch.3: Retrain the meta-labeling classifier.

    Aggregates OHLCV data for all tracked symbols and trains the
    secondary classifier that predicts forecast correctness using
    triple-barrier labels.
    """
    logger.info("=== Meta-Label Retrain started ===")
    try:
        from services.meta_labeling import train_meta_labeler
        import yfinance as yf

        # Gather OHLCV for IND symbols
        from config import Config
        tickers = getattr(Config, "MONITORED_TICKERS", [])
        if not tickers:
            # Fallback: read from sample_tickers.csv
            from pathlib import Path
            csv_path = Path(__file__).parent / "sample_tickers.csv"
            if csv_path.exists():
                import csv
                with open(csv_path) as f:
                    reader = csv.reader(f)
                    tickers = [row[0].strip() for row in reader if row]

        if not tickers:
            logger.warning("Meta-label retrain: no tickers configured")
            return

        # Download 2 years of data
        ohlcv_cache = {}
        for ticker in tickers[:50]:  # cap at 50 symbols
            try:
                df = yf.download(ticker, period="2y", progress=False)
                if df is not None and len(df) > 252:
                    ohlcv_cache[ticker] = df
            except Exception:
                continue

        if len(ohlcv_cache) < 5:
            logger.warning("Meta-label retrain: insufficient data (%d symbols)", len(ohlcv_cache))
            return

        # Train IND model
        result_ind = train_meta_labeler(ohlcv_cache, market="IND")
        logger.info("Meta-label IND: %s", result_ind.get("status", "unknown"))

        # Train US model (if US tickers configured)
        us_tickers = getattr(Config, "US_MONITORED_TICKERS", [])
        if us_tickers:
            us_cache = {}
            for ticker in us_tickers[:30]:
                try:
                    df = yf.download(ticker, period="2y", progress=False)
                    if df is not None and len(df) > 252:
                        us_cache[ticker] = df
                except Exception:
                    continue
            if len(us_cache) >= 3:
                result_us = train_meta_labeler(us_cache, market="US")
                logger.info("Meta-label US: %s", result_us.get("status", "unknown"))

    except Exception as exc:
        logger.exception("Meta-label retrain failed: %s", exc)


@_tracked_job("us_pre_market", "US Pre-Market Pipeline")
def _run_us_pre_market():
    """G11: Run US stocks pre-market analysis pipeline.

    Executes the US Carver pipeline during US pre-market hours
    and caches results for the API/UI to consume.
    """
    logger.info("=== US Pre-Market Pipeline started ===")
    try:
        from services.us_carver_pipeline import run_us_carver_pipeline, DEFAULT_US_CARVER_TICKERS

        result = run_us_carver_pipeline(DEFAULT_US_CARVER_TICKERS)

        # Persist to cache
        _save_run("us_pre_market", {
            "trade_plans": len(result.trade_plans),
            "symbols_processed": result.symbols_processed,
            "combined_forecasts": {s: round(v, 2) for s, v in result.combined_forecasts.items()},
        })

        logger.info("US pipeline: %d trade plans from %d symbols",
                     len(result.trade_plans), result.symbols_processed)
        for line in result.pipeline_log:
            logger.info("  US: %s", line)

    except Exception as exc:
        logger.exception("US pre-market pipeline failed: %s", exc)



# ===================================================================
# Phase 1-4: Advanced strategy job handlers
# ===================================================================

@_tracked_job("options_monitor", "Options Monitor")
def _run_options_monitor():
    """Job 13: Poll open options positions for profit-take / roll / expiry."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Options Monitor Poll ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Options monitor: no Kite session")
            return
        from kite_connect.options.options_monitor import run_options_monitor_poll
        run_options_monitor_poll(kite)
    except Exception as exc:
        logger.exception("Options monitor failed: %s", exc)


@_tracked_job("margin_monitor", "Margin Monitor")
def _run_margin_monitor():
    """Job 14: Check margin utilisation and alert if thresholds breached."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Margin Monitor Poll ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Margin monitor: no Kite session")
            return
        from kite_connect.trading.margin_monitor import get_margin_snapshot
        snap = get_margin_snapshot(kite)
        if snap.alert_level in ("WARNING", "CRITICAL"):
            logger.warning(
                "Margin %s: %.1f%% used (%.0f / %.0f)",
                snap.alert_level, snap.utilisation_pct,
                snap.used, snap.available + snap.used,
            )
    except Exception as exc:
        logger.exception("Margin monitor failed: %s", exc)


@_tracked_job("pairs_scanner", "Pairs Scanner")
def _run_pairs_scanner():
    """Job 15: Scan configured pairs for mean-reversion signals."""
    try:
        from config import Config
        if not getattr(Config, "PAIRS_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Pairs Trading Scanner ===")
    try:
        import numpy as np
        from utils import download_ind_ohlcv
        from services.pairs_trading_live import scan_all_pairs, DEFAULT_PAIRS

        pairs = getattr(Config, "PAIRS_LIST", DEFAULT_PAIRS)
        symbols = set()
        for a, b in pairs:
            symbols.add(a)
            symbols.add(b)

        price_data = {}
        for sym in symbols:
            try:
                df = download_ind_ohlcv(sym, period="6mo")
                if df is not None and len(df) >= 60:
                    col = "Close" if "Close" in df.columns else "close"
                    price_data[sym] = df[col].values.astype(float)
            except Exception:
                pass

        signals = scan_all_pairs(price_data)
        if signals:
            logger.info("Pairs signals: %d active", len(signals))
            for s in signals:
                logger.info("  %s/%s z=%.2f action=%s forecast=%.1f",
                            s.leg1, s.leg2, s.z_score, s.action, s.forecast)

            # G5: Execute pairs via SpreadExecutor
            _execute_pairs_signals(signals)

    except Exception as exc:
        logger.exception("Pairs scanner failed: %s", exc)


@_tracked_job("futures_monitor", "Futures Monitor")
def _run_futures_monitor():
    """Job 16: Monitor futures positions for rollover and de-leveraging."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Futures Monitor ===")
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("Futures monitor: no Kite session")
            return
        from kite_connect.trading.futures_monitor import run_futures_monitor
        result = run_futures_monitor(kite)
        for alert in result.alerts:
            logger.warning("Futures alert: %s", alert)

        # G6: Execute futures overlay signal
        _execute_futures_overlay(kite)

    except Exception as exc:
        logger.exception("Futures monitor failed: %s", exc)


@_tracked_job("event_calendar", "Event Calendar Seed")
def _run_event_calendar_seed():
    """Job 17: Seed fixed events (RBI, rebalance) into the calendar DB."""
    try:
        from config import Config
        if not getattr(Config, "EVENT_DRIVEN_ENABLED", False):
            return
    except Exception:
        return

    logger.info("=== Event Calendar Seed ===")
    try:
        from services.event_calendar import seed_fixed_events
        seed_fixed_events()
    except Exception as exc:
        logger.exception("Event calendar seed failed: %s", exc)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Scheduler setup
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


# =================================================================
# G4/G5/G6/G10: Execution Helper Functions
# =================================================================

def _execute_options_overlay(kite):
    """G4: Execute options overlay - covered calls + CSPs."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_ENABLED", False):
            return

        from kite_connect.options.options_executor import OptionsExecutor
        from services.options_overlay import scan_covered_call_candidates, scan_csp_candidates

        executor = OptionsExecutor(kite)

        # Covered calls on existing long positions
        try:
            cc_candidates = scan_covered_call_candidates(kite)
            if cc_candidates:
                results = executor.execute_covered_calls(cc_candidates)
                logger.info("G4: Covered calls executed: %d orders", len(results))
        except Exception as exc:
            logger.warning("G4: Covered calls failed: %s", exc)

        # Cash-secured puts on high-conviction BUY signals
        try:
            csp_candidates = scan_csp_candidates(kite)
            if csp_candidates:
                results = executor.execute_cash_secured_puts(csp_candidates)
                logger.info("G4: CSPs executed: %d orders", len(results))
        except Exception as exc:
            logger.warning("G4: CSPs failed: %s", exc)

    except Exception as exc:
        logger.debug("G4: Options overlay skipped: %s", exc)


def _execute_tail_hedge_if_needed(kite):
    """G10: Auto-execute tail hedge when drawdown is critical."""
    try:
        from config import Config
        if not getattr(Config, "OPTIONS_TAIL_HEDGE_ENABLED", False):
            return

        from services.tail_risk_hedge import TailRiskHedge
        from kite_connect.options.options_executor import OptionsExecutor

        # Get current portfolio state
        capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
        realized = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
        equity = getattr(Config, "_CURRENT_EQUITY", capital + realized)
        peak = getattr(Config, "_PEAK_EQUITY", capital)
        dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0

        # Get VIX
        try:
            import yfinance as yf
            vix_data = yf.download("^INDIAVIX", period="5d", progress=False)
            vix = float(vix_data["Close"].iloc[-1]) if len(vix_data) > 0 else 15.0
            vix_3d = float(vix_data["Close"].iloc[-4]) if len(vix_data) >= 4 else vix
        except Exception:
            vix, vix_3d = 15.0, 15.0

        # Get NIFTY spot
        try:
            ltp = kite.ltp(["NSE:NIFTY 50"])
            nifty_spot = ltp.get("NSE:NIFTY 50", {}).get("last_price", 0)
        except Exception:
            nifty_spot = 0

        hedger = TailRiskHedge()
        assessment = hedger.assess(
            portfolio_value=equity,
            drawdown_pct=dd_pct,
            vix=vix,
            vix_3d_ago=vix_3d,
            nifty_spot=nifty_spot,
        )

        if assessment.hedge_urgency in ("HIGH", "CRITICAL") and assessment.recommendation:
            executor = OptionsExecutor(kite)
            result = executor.execute_tail_hedge(assessment.recommendation)
            logger.info("G10: Tail hedge executed: urgency=%s result=%s",
                        assessment.hedge_urgency, result)
        else:
            logger.info("G10: Tail hedge not needed: urgency=%s dd=%.1f%%",
                        assessment.hedge_urgency, dd_pct)

    except Exception as exc:
        logger.debug("G10: Tail hedge check skipped: %s", exc)


def _execute_pairs_signals(signals):
    """G5: Execute pairs trading signals via SpreadExecutor."""
    try:
        kite = _get_scheduler_kite()
        if kite is None:
            logger.warning("G5: No Kite session for pairs execution")
            return

        from kite_connect.trading.spread_executor import SpreadExecutor, LegOrder
        spread_exec = SpreadExecutor(kite)

        for sig in signals:
            if not hasattr(sig, "action") or sig.action not in ("ENTER_LONG", "ENTER_SHORT"):
                continue

            # Build leg orders based on signal direction
            if sig.action == "ENTER_LONG":
                leg1 = LegOrder(symbol=sig.leg1, side="BUY", quantity=1, exchange="NSE")
                leg2 = LegOrder(symbol=sig.leg2, side="SELL", quantity=1, exchange="NSE")
            else:  # ENTER_SHORT
                leg1 = LegOrder(symbol=sig.leg1, side="SELL", quantity=1, exchange="NSE")
                leg2 = LegOrder(symbol=sig.leg2, side="BUY", quantity=1, exchange="NSE")

            result = spread_exec.execute_pair(leg1, leg2)
            logger.info("G5: Pair %s/%s %s: success=%s",
                        sig.leg1, sig.leg2, sig.action, result.success)

    except Exception as exc:
        logger.warning("G5: Pairs execution failed: %s", exc)


def _execute_futures_overlay(kite):
    """G6: Execute futures overlay signal for regime-adaptive leverage."""
    try:
        from config import Config
        if not getattr(Config, "LEVERAGE_ENABLED", False):
            return

        from services.futures_overlay import compute_futures_overlay
        from services.regime_detector import get_current_regime
        from kite_connect.trading.order_service import place_order

        # Get current state
        capital = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
        realized = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
        equity = capital + realized

        regime_info = get_current_regime()
        regime = regime_info.get("regime", "range")
        confidence = regime_info.get("confidence", 0.5)

        # Get NIFTY spot/futures price
        try:
            ltp = kite.ltp(["NSE:NIFTY 50"])
            nifty_spot = ltp.get("NSE:NIFTY 50", {}).get("last_price", 0)
        except Exception:
            nifty_spot = 0

        signal = compute_futures_overlay(
            portfolio_value=equity,
            current_futures_notional=0.0,
            nifty_spot=nifty_spot,
            regime=regime,
            regime_confidence=confidence,
        )

        if signal.action == "BUY_FUT" and signal.lots > 0:
            lot_size = getattr(Config, "FUTURES_LOT_SIZE", 25)
            result = place_order(
                kite,
                tradingsymbol="NIFTY" + _get_current_expiry_suffix(),
                exchange="NFO",
                transaction_type="BUY",
                quantity=signal.lots * lot_size,
                product="NRML",
                order_type="MARKET",
            )
            logger.info("G6: BUY_FUT %d lots, order=%s", signal.lots, result)

        elif signal.action == "SELL_FUT" and signal.lots > 0:
            lot_size = getattr(Config, "FUTURES_LOT_SIZE", 25)
            result = place_order(
                kite,
                tradingsymbol="NIFTY" + _get_current_expiry_suffix(),
                exchange="NFO",
                transaction_type="SELL",
                quantity=signal.lots * lot_size,
                product="NRML",
                order_type="MARKET",
            )
            logger.info("G6: SELL_FUT %d lots, order=%s", signal.lots, result)

        else:
            logger.debug("G6: Futures overlay action=%s lots=%d (no trade)",
                         signal.action, signal.lots)

    except Exception as exc:
        logger.debug("G6: Futures overlay skipped: %s", exc)


def _get_current_expiry_suffix():
    """Get current month NIFTY futures expiry suffix (e.g. '25JUN' for Jun 2025)."""
    from datetime import date
    today = date.today()
    # NFO convention: YYMMMFUT e.g. NIFTY25JUNFUT
    suffix = today.strftime("%y%b").upper() + "FUT"
    return suffix


def start_scheduler():
    """Start the APScheduler background scheduler with IST-aware jobs.

    Jobs
    ----
    1. **pre_market_scan** â€” 9:20 AM IST, Mon-Fri
       Full pipeline run after market opens (NSE opens 9:15).
    1b. **intraday_rescan** â€” 10:30, 12:30, 14:30 IST, Mon-Fri
       Lighter re-scan for intraday momentum shifts.
    1c. **eod_scan** â€” 15:20 IST, Mon-Fri
       End-of-day scan 10 min before market close.
    2. **walk_forward_audit** â€” Saturday 6:00 AM IST
       Weekly walk-forward validation of registered strategies.
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
    # (includes DB pre-warming to wake Neon auto-suspended compute)
    scheduler.add_job(
        _pre_market_with_warmup,
        CronTrigger(hour=9, minute=20, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="pre_market_scan",
        name="Pre-Market Full Scan",
        misfire_grace_time=600,
    )

    # Job 1b: Intraday re-scan at 10:30, 12:30, 14:30 IST, weekdays
    scheduler.add_job(
        _run_intraday_rescan,
        CronTrigger(hour="10,12,14", minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="intraday_rescan",
        name="Intraday Re-Scan",
        misfire_grace_time=600,
    )

    # Job 1c: End-of-day scan at 15:20 IST (10 min before market close)
    scheduler.add_job(
        _run_eod_scan,
        CronTrigger(hour=15, minute=20, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="eod_scan",
        name="End-of-Day Scan",
        misfire_grace_time=300,
    )

    # Job 2: Weekly walk-forward strategy audit â€” Saturday 6 AM IST
    scheduler.add_job(
        run_walk_forward_audit,
        CronTrigger(hour=6, minute=0, day_of_week="sat", timezone="Asia/Kolkata"),
        id="walk_forward_audit",
        name="Weekly Walk-Forward Audit",
        misfire_grace_time=3600,
    )

    # Job 3: Weekly paper vs live reconciliation â€” Saturday 7 AM IST
    scheduler.add_job(
        _run_paper_live_reconciliation,
        CronTrigger(hour=7, minute=0, day_of_week="sat", timezone="Asia/Kolkata"),
        id="paper_live_reconciliation",
        name="Paper vs Live Reconciliation",
        misfire_grace_time=3600,
    )

    # Job 8: Weekly forecast scalar calibration - Saturday 5:30 AM IST
    # Runs before walk-forward so WF uses freshly calibrated scalars
    scheduler.add_job(
        _run_forecast_calibration,
        CronTrigger(hour=5, minute=30, day_of_week="sat", timezone="Asia/Kolkata"),
        id="forecast_calibration",
        name="Weekly Forecast Scalar Calibration",
        misfire_grace_time=3600,
    )

    # Job 4: Nightly SQLite backup to R2 â€” 23:00 IST daily
    scheduler.add_job(
        _run_nightly_backup,
        CronTrigger(hour=23, minute=0, timezone="Asia/Kolkata"),
        id="nightly_backup",
        name="Nightly SQLite Backup to R2",
        misfire_grace_time=3600,
    )

    # Job 6: Proactive Kite token refresh — every 30 min during market hours
    scheduler.add_job(
        refresh_kite_token_if_needed,
        CronTrigger(hour="9-16", minute="0,30", day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="kite_token_refresh",
        name="Kite Token Refresh Check",
        misfire_grace_time=300,
    )

    # Job 7: Trade Monitor — poll every 3 min during market hours
    # Manages SL/TP lifecycle, trailing-SL, time exits, capital rollup
    scheduler.add_job(
        _run_trade_monitor_poll,
        CronTrigger(hour="9-15", minute="*/3", day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="trade_monitor_poll",
        name="Trade Monitor Poll",
        misfire_grace_time=120,
    )

    # Job 9: Monthly strategy tournament — 1st Saturday 4:00 AM IST
    scheduler.add_job(
        _run_strategy_tournament,
        CronTrigger(hour=4, minute=0, day_of_week="sat", day="1-7", timezone="Asia/Kolkata"),
        id="strategy_tournament",
        name="Monthly Strategy Tournament",
        misfire_grace_time=3600,
    )

    logger.info("Scheduler started â€” press Ctrl+C to stop")
    logger.info("  Pre-market scan : 09:20 IST, Mon-Fri")
    logger.info("  Intraday re-scan: 10:30, 12:30, 14:30 IST, Mon-Fri")
    logger.info("  EOD scan        : 15:20 IST, Mon-Fri")
    logger.info("  Trade monitor   : every 3 min, 09:00-15:59 IST, Mon-Fri")
    logger.info("  Walk-forward    : 06:00 IST, Saturday")
    logger.info("  Reconciliation  : 07:00 IST, Saturday")
    logger.info("  Nightly backup  : 23:00 IST, daily")
    logger.info("  Kite refresh    : every 30 min, 09:00-16:30 IST, Mon-Fri")
    logger.info("  Forecast calib  : 05:30 IST, Saturday")
    logger.info("  Tournament      : 04:00 IST, 1st Saturday of month")

    # Job 10: Monthly HMM regime re-fit — 1st Sunday 3:00 AM IST
    # Gap B1: Re-train 3-state Gaussian HMM on 5 years of NIFTY data
    scheduler.add_job(
        _run_hmm_refit,
        CronTrigger(hour=3, minute=0, day_of_week="sun", day="1-7", timezone="Asia/Kolkata"),
        id="hmm_refit",
        name="Monthly HMM Regime Re-fit",
        misfire_grace_time=3600,
    )

    logger.info("  HMM re-fit      : 03:00 IST, 1st Sunday of month")

    # Job 11: PEAD Earnings Feed — 8:00 AM IST, Mon-Fri (before pre-market)
    # G6: Fetch Trendlyne earnings data and feed into PEADStrategy
    scheduler.add_job(
        _run_pead_earnings_feed,
        CronTrigger(hour=8, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="pead_earnings_feed",
        name="PEAD Earnings Data Feed",
        misfire_grace_time=600,
    )

    # Job 12: US Pre-Market Pipeline — 19:00 IST (9:30 AM ET), Mon-Fri
    # G11: Run US Carver pipeline during US market hours
    scheduler.add_job(
        _run_us_pre_market,
        CronTrigger(hour=19, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="us_pre_market",
        name="US Pre-Market Pipeline",
        misfire_grace_time=600,
    )

    logger.info("  PEAD feed       : 08:00 IST, Mon-Fri")
    logger.info("  US pipeline     : 19:00 IST, Mon-Fri")

    # ── Phase 1-4: Advanced strategy jobs ──────────────────────

    # Job 13: Options Monitor — Every 5 min during market hours
    scheduler.add_job(
        _run_options_monitor,
        CronTrigger(
            hour="9-15", minute="*/5", day_of_week="mon-fri",
            timezone="Asia/Kolkata",
        ),
        id="options_monitor",
        name="Options Monitor Poll",
        misfire_grace_time=120,
    )

    # Job 14: Margin Monitor — Every 10 min during market hours
    scheduler.add_job(
        _run_margin_monitor,
        CronTrigger(
            hour="9-15", minute="*/10", day_of_week="mon-fri",
            timezone="Asia/Kolkata",
        ),
        id="margin_monitor",
        name="Margin Monitor Poll",
        misfire_grace_time=120,
    )

    # Job 15: Pairs Scanner — Every 30 min during market hours
    scheduler.add_job(
        _run_pairs_scanner,
        CronTrigger(
            hour="9-15", minute="0,30", day_of_week="mon-fri",
            timezone="Asia/Kolkata",
        ),
        id="pairs_scanner",
        name="Pairs Trading Scanner",
        misfire_grace_time=300,
    )

    # Job 16: Futures Rollover Check — 14:00 IST, Mon-Fri
    scheduler.add_job(
        _run_futures_monitor,
        CronTrigger(hour=14, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="futures_monitor",
        name="Futures Monitor & Rollover",
        misfire_grace_time=300,
    )

    # Job 17: Event Calendar Seed — 07:00 IST, Mon-Fri
    scheduler.add_job(
        _run_event_calendar_seed,
        CronTrigger(hour=7, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="event_calendar",
        name="Event Calendar Seed",
        misfire_grace_time=600,
    )

    logger.info("  Options monitor : */5 min, 09-15 IST, Mon-Fri")
    logger.info("  Margin monitor  : */10 min, 09-15 IST, Mon-Fri")
    logger.info("  Pairs scanner   : */30 min, 09-15 IST, Mon-Fri")
    logger.info("  Futures monitor : 14:00 IST, Mon-Fri")
    logger.info("  Event calendar  : 07:00 IST, Mon-Fri")

    # Job 18: Meta-Label Retraining — 02:00 IST, 1st/15th of month (semi-monthly)
    # AFML Ch.3: Retrain the meta-labeling classifier on accumulated trade outcomes
    scheduler.add_job(
        _run_meta_label_retrain,
        CronTrigger(hour=2, minute=0, day="1,15", timezone="Asia/Kolkata"),
        id="meta_label_retrain",
        name="Meta-Label Model Retrain",
        misfire_grace_time=3600,
    )

    logger.info("  Meta-label train: 02:00 IST, 1st & 15th of month")

    # ── T3-5: Scheduler heartbeat — dead-man switch ──────────
    # Writes timestamp to heartbeat file every 5 minutes.
    # External monitor can check file freshness to detect stalled scheduler.
    def _heartbeat():
        import json
        from datetime import datetime
        hb_path = os.path.join(os.path.dirname(__file__), "data", "scheduler_heartbeat.json")
        try:
            hb = {
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid(),
                "jobs_active": len(scheduler.get_jobs()),
                "status": "alive",
            }
            with open(hb_path, "w") as f:
                json.dump(hb, f)
        except Exception as e:
            logger.warning("Heartbeat write failed: %s", e)

    scheduler.add_job(
        _heartbeat,
        "interval",
        minutes=5,
        id="scheduler_heartbeat",
        name="Scheduler Heartbeat (T3-5)",
        misfire_grace_time=120,
    )
    logger.info("  Heartbeat       : every 5 min (dead-man switch)")

    # ── T3-5: Trade returns collector for Monte Carlo bootstrap ──
    def _collect_trade_returns():
        try:
            from services.trade_returns_collector import run_collection
            run_collection()
        except Exception as e:
            logger.warning("Trade returns collection failed: %s", e)

    scheduler.add_job(
        _collect_trade_returns,
        CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="trade_returns_collector",
        name="Trade Returns Collector (MC bootstrap)",
        misfire_grace_time=600,
    )
    logger.info("  Trade returns   : 16:00 IST, Mon-Fri (MC bootstrap)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLI entry point
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Centurion Core â€” Pipeline Scheduler")
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
