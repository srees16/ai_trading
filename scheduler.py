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


# ═══════════════════════════════════════════════════════════════
# Proactive Kite token refresh
# ═══════════════════════════════════════════════════════════════

def refresh_kite_token_if_needed():
    """Re-authenticate Kite if the access token is expiring soon.

    Called every 30 min during market hours by the scheduler.
    Uses the same TOTP auto-auth path as the pipeline.
    """
    try:
        from api.dependencies import is_kite_token_expiring_soon, get_kite_session, set_kite_session

        kite = get_kite_session()
        if kite is None:
            return  # not logged in — nothing to refresh

        if not is_kite_token_expiring_soon():
            return  # still fresh

        logger.info("Kite token expiring soon — proactive re-authentication")
        from kite_connect.auth.kite_session import create_kite_session
        new_kite = create_kite_session()
        if new_kite:
            set_kite_session(new_kite)
            logger.info("Kite token refreshed successfully")
        else:
            logger.warning("Kite re-auth returned None — token may expire")
    except Exception as exc:
        logger.warning("Proactive Kite refresh failed: %s", exc)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Pipeline runner (headless â€” no Streamlit, no Kite orders)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
        kite = None
        try:
            from kite_connect.auth.kite_session import create_kite_session
            kite = create_kite_session()
        except Exception:
            pass

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
        from kite_connect.auth.kite_session import create_kite_session
        logger.info("Auto-authenticating Kite for STRONG_BUY order placementâ€¦")
        kite = create_kite_session()
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
    except Exception as exc:
        logger.exception("Auto-order placement failed: %s", exc)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Walk-Forward Audit (weekly)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_walk_forward_audit():
    """Run walk-forward validation on all registered strategies.

    Kicks off every Saturday morning via the scheduler.  Results are
    saved to the scheduler cache DB under run_type='walk_forward'.
    Strategies with degradation_ratio < 0.5 are flagged as overfit.
    """
    logger.info("=== Walk-Forward Audit started ===")

    try:
        from strategies import StrategyRegistry, load_all_strategies
        from services.walk_forward import walk_forward_validate, save_optimal_params

        load_all_strategies()
        all_strategies = StrategyRegistry._strategies

        # Use a representative NIFTY-50 ticker for validation
        test_ticker = "RELIANCE.NS"

        audit_results = {}
        overfit_strategies: List[str] = []

        for name, strategy_cls in all_strategies.items():
            if "crypto" in name.lower():
                continue
            try:
                summary = walk_forward_validate(
                    strategy_cls=strategy_cls,
                    ticker=test_ticker,
                    capital=100_000,
                    train_days=252,
                    test_days=63,
                    total_days=756,
                )
                audit_results[name] = summary.to_dict()
                # Gap 5: Persist winning params for the live pipeline
                save_optimal_params(summary)
                if summary.degradation_ratio < 0.5 and summary.total_folds > 0:
                    overfit_strategies.append(name)
                    logger.warning(
                        "OVERFIT: %s â€” degradation=%.2f (OOS Sharpe=%.2f, IS=%.2f)",
                        name, summary.degradation_ratio,
                        summary.avg_oos_sharpe, summary.avg_is_sharpe,
                    )
                else:
                    logger.info(
                        "OK: %s â€” degradation=%.2f, OOS Sharpe=%.2f",
                        name, summary.degradation_ratio, summary.avg_oos_sharpe,
                    )
            except Exception as exc:
                audit_results[name] = {"error": str(exc)}
                logger.warning("WF audit failed for %s: %s", name, exc)

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

    except Exception as exc:
        logger.exception("Walk-Forward Audit failed: %s", exc)
        _save_run("walk_forward", {"status": f"error: {exc}"})


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Unified Backtest â†” Paper â†” Live Reconciliation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
        from kite_connect.auth.kite_session import create_kite_session
        kite = create_kite_session()
        if kite is None:
            logger.debug("TradeMonitor poll skipped — no Kite session")
            return
    except Exception as exc:
        logger.debug("TradeMonitor poll skipped — Kite auth failed: %s", exc)
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


def _run_nightly_backup():
    """Upload SQLite databases to R2/MinIO storage."""
    try:
        from infrastructure.backup_service import run_backup
        result = run_backup()
        logger.info("Nightly backup result: %s", result)
    except Exception as e:
        logger.error("Nightly backup failed: %s", e)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Scheduler setup
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def start_scheduler():
    """Start the APScheduler background scheduler with IST-aware jobs.

    Jobs
    ----
    1. **pre_market_scan** â€” 9:20 AM IST, Mon-Fri
       Full pipeline run before market opens (NSE opens 9:15).
    2. **intraday_rescan** â€” every 2 hours (10:30, 12:30, 14:30) Mon-Fri
       Lighter re-scan for intraday momentum shifts.
    3. **walk_forward_audit** â€” Saturday 6:00 AM IST
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

    # Job 2: Intraday re-scan every 2 hours during 10:30â€“14:30 IST
    scheduler.add_job(
        run_pipeline,
        CronTrigger(hour="10,12,14", minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        args=["intraday"],
        id="intraday_rescan",
        name="Intraday Re-Scan",
        misfire_grace_time=600,
    )

    # Job 3: Weekly walk-forward strategy audit â€” Saturday 6 AM IST
    scheduler.add_job(
        run_walk_forward_audit,
        CronTrigger(hour=6, minute=0, day_of_week="sat", timezone="Asia/Kolkata"),
        id="walk_forward_audit",
        name="Weekly Walk-Forward Audit",
        misfire_grace_time=3600,
    )

    # Job 4: Weekly paper vs live reconciliation â€” Saturday 7 AM IST
    scheduler.add_job(
        _run_paper_live_reconciliation,
        CronTrigger(hour=7, minute=0, day_of_week="sat", timezone="Asia/Kolkata"),
        id="paper_live_reconciliation",
        name="Paper vs Live Reconciliation",
        misfire_grace_time=3600,
    )

    # Job 5: Nightly SQLite backup to R2 â€” 23:00 IST daily
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

    logger.info("Scheduler started â€” press Ctrl+C to stop")
    logger.info("  Pre-market scan : 09:20 IST, Mon-Fri")
    logger.info("  Intraday re-scan: 10:30, 12:30, 14:30 IST, Mon-Fri")
    logger.info("  Trade monitor   : every 3 min, 09:00-15:59 IST, Mon-Fri")
    logger.info("  Walk-forward    : 06:00 IST, Saturday")
    logger.info("  Reconciliation  : 07:00 IST, Saturday")
    logger.info("  Nightly backup  : 23:00 IST, daily")
    logger.info("  Kite refresh    : every 30 min, 09:00-16:30 IST, Mon-Fri")

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
