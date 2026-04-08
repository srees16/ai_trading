#!/usr/bin/env python3
"""Cloud paper trading runner — executed by GitHub Actions cron.

Checks Neon for active paper trading state, runs the full CarverPipeline,
executes trades via PaperTrader, and syncs results to Neon.

Usage (GitHub Actions):
    python centurion_core/cloud_paper_runner.py

Required env vars:
    CENTURION_DATABASE_URL  — Neon PostgreSQL connection string
    CENTURION_PAPER_TRADE   — must be "true"

Optional:
    CENTURION_EMAIL_USER / CENTURION_EMAIL_PASS  — for daily reports
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

# Ensure centurion_core is on the path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("CENTURION_PAPER_TRADE", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cloud_paper_runner")


# ── Neon state helpers ────────────────────────────────────────────────────

def _get_neon_engine():
    """Create a SQLAlchemy engine for Neon."""
    import re
    from sqlalchemy import create_engine

    url = os.environ.get("CENTURION_DATABASE_URL", "")
    if not url:
        raise RuntimeError("CENTURION_DATABASE_URL not set")

    # Strip channel_binding (psycopg2 doesn't support it)
    url = re.sub(r"[&?]channel_binding=[^&]*", "", url)
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url += f"{sep}sslmode=require"

    return create_engine(url, pool_pre_ping=True, pool_size=2)


def _check_active() -> bool:
    """Check if paper trading is active and not expired in Neon."""
    from sqlalchemy import text

    engine = _get_neon_engine()
    with engine.connect() as conn:
        # Auto-create table if it doesn't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS paper_trading_state (
                id            INTEGER PRIMARY KEY DEFAULT 1,
                active        BOOLEAN NOT NULL DEFAULT FALSE,
                started_at    TIMESTAMPTZ,
                expires_at    TIMESTAMPTZ,
                stopped_at    TIMESTAMPTZ,
                last_run_at   TIMESTAMPTZ,
                total_runs    INTEGER DEFAULT 0,
                last_run_status VARCHAR(20) DEFAULT 'none',
                last_run_message TEXT DEFAULT '',
                updated_at    TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            INSERT INTO paper_trading_state (id, active)
            VALUES (1, FALSE)
            ON CONFLICT (id) DO NOTHING
        """))
        conn.commit()

        row = conn.execute(
            text("SELECT active, expires_at FROM paper_trading_state WHERE id = 1")
        ).fetchone()

        if not row or not row[0]:
            return False

        # Check expiry
        if row[1] and row[1] < datetime.now(row[1].tzinfo or None):
            conn.execute(text(
                "UPDATE paper_trading_state SET active = FALSE, stopped_at = NOW(), updated_at = NOW() WHERE id = 1"
            ))
            conn.commit()
            logger.info("Paper trading expired at %s — deactivated", row[1])
            return False

        return True


def _update_run_status(status: str, message: str):
    """Update the last run status in Neon."""
    from sqlalchemy import text

    try:
        engine = _get_neon_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE paper_trading_state
                SET last_run_at = NOW(),
                    total_runs = total_runs + 1,
                    last_run_status = :status,
                    last_run_message = :message,
                    updated_at = NOW()
                WHERE id = 1
            """), {"status": status, "message": message})
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to update run status in Neon: %s", exc)


# ── Pipeline execution ────────────────────────────────────────────────────

def _run_paper_pipeline():
    """Run the full screening + CarverPipeline + PaperTrader flow."""
    from kite_connect.nse.nse_universe import get_nse_universe
    from kite_connect.nse.screener import NSEScreener, ScreenerConfig
    from services.integrated_scorer import IntegratedScorer
    from kite_connect.trading.paper_trader import PaperTrader

    # 1. Universe
    symbols = get_nse_universe()
    logger.info("Universe: %d symbols", len(symbols))

    # 2. Screen
    cfg = ScreenerConfig(index_mode=True)
    screener = NSEScreener(config=cfg)
    screened_df = screener.screen(symbols)
    logger.info("Screened: %d passed", len(screened_df))

    if screened_df.empty:
        return "no_stocks_passed", "Screener returned 0 candidates"

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

    signal_dict = {
        v.ticker.replace(".NS", "").replace(".BO", ""): v.classification
        for v in verdicts
    }
    buy_symbols = [
        sym for sym, tag in signal_dict.items()
        if tag in ("BUY", "STRONG_BUY")
    ]
    if not buy_symbols:
        return "success", f"No BUY signals from {len(verdicts)} verdicts"

    buy_df = screened_df[screened_df["symbol"].isin(buy_symbols)]
    if buy_df.empty:
        return "success", "Buy symbols not in screened set"

    # 4. CarverPipeline
    plans = None
    pipe_result = None
    try:
        from services.carver_pipeline import CarverPipeline, PipelineConfig
        from utils import download_ind_ohlcv

        ohlcv_cache = {}
        for sym in buy_symbols:
            try:
                df = download_ind_ohlcv(sym, period="2y")
                if df is not None and len(df) >= 64:
                    ohlcv_cache[sym] = df
            except Exception:
                pass

        if ohlcv_cache:
            screener_scores = {}
            if "score" in screened_df.columns:
                for _, row in screened_df.iterrows():
                    sym = row.get("symbol", "")
                    if sym in ohlcv_cache:
                        screener_scores[sym] = float(row["score"])

            pipeline = CarverPipeline(PipelineConfig())
            pipe_result = pipeline.run(
                ohlcv_cache=ohlcv_cache,
                screener_scores=screener_scores,
            )
            plans = pipe_result.trade_plans
            logger.info("CarverPipeline: %d plans", len(plans))
    except Exception as exc:
        logger.warning("CarverPipeline failed, trying fallback: %s", exc)

    # 4b. Fallback: RiskManager
    if not plans:
        try:
            from kite_connect.trading.risk_manager import RiskManager, RiskConfig
            rm = RiskManager(RiskConfig())
            plans = rm.plan_trades(buy_df)
            logger.info("Fallback RiskManager: %d plans", len(plans))
        except Exception as exc:
            return "error", f"Both pipelines failed: {exc}"

    if not plans:
        return "success", "No plans met R:R threshold"

    # 5. Execute via PaperTrader
    pt = PaperTrader(kite=None, initial_capital=100_000)
    results = pt.execute_plans(plans)
    filled = sum(1 for r in results if r.get("success"))

    # 6. SL/TP poll
    close_events = pt.poll()

    # 7. Signal audit log
    try:
        today_str = date.today().isoformat()
        signal_entries = []
        traded_symbols = {r["symbol"] for r in results if r.get("success")}
        _indiv = getattr(pipe_result, "individual_forecasts", {}) if pipe_result else {}
        for plan in plans:
            _sym_fc = _indiv.get(plan.symbol, {})
            _active = sorted(k for k, v in _sym_fc.items() if v and abs(v) > 0.01)
            signal_entries.append({
                "symbol": plan.symbol,
                "forecast": getattr(plan, "score", 0),
                "combined_forecast": getattr(plan, "score", 0),
                "action": plan.side,
                "entry_price": plan.entry_price,
                "stop_loss": plan.stop_loss,
                "target_price": plan.target_price,
                "quantity": plan.quantity,
                "pipeline_sources": ",".join(_active) if _active else "CarverPipeline",
                "was_traded": plan.symbol in traded_symbols,
            })
        pt.log_signals(today_str, signal_entries)
    except Exception as exc:
        logger.debug("Signal logging failed (non-fatal): %s", exc)

    # 8. EOD snapshot
    try:
        pt.snapshot_daily()
    except Exception as exc:
        logger.debug("Snapshot failed (non-fatal): %s", exc)

    dashboard = pt.dashboard()
    msg = (
        f"{filled}/{len(plans)} filled | "
        f"capital={dashboard.current_capital:.0f} | "
        f"P&L={dashboard.total_pnl:.0f} ({dashboard.total_pnl_pct:.1f}%)"
    )
    logger.info("Paper trade: %s", msg)

    # 9. Daily email (best-effort)
    try:
        from services.notifications.manager import NotificationManager
        nm = NotificationManager()
        nm.email_daily_pipeline_report({
            "universe_size": len(symbols),
            "screened_count": len(screened_df),
            "buy_signals": len(buy_symbols),
            "sell_signals": len(close_events),
            "status": "success",
        })
    except Exception:
        pass

    return "success", msg


# ── Entrypoint ─────────────────────────────────────────────────────────

def main():
    logger.info("=== Cloud Paper Trading Runner ===")

    if not _check_active():
        logger.info("Paper trading is NOT active in Neon — skipping.")
        return

    logger.info("Paper trading is ACTIVE — running pipeline...")
    try:
        status, message = _run_paper_pipeline()
        _update_run_status(status, message[:500])
        logger.info("Run complete: [%s] %s", status, message)
    except Exception as exc:
        _update_run_status("error", str(exc)[:500])
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
