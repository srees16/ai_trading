"""
Auto-Order Execution Engine for Zerodha Kite Connect.

Orchestrates the full pipeline:

1. Download NSE universe  â†’  :mod:`kite_connect.nse.nse_universe`
2. Screen & rank          â†’  :mod:`kite_connect.nse.screener`
3. Risk-manage & size     â†’  :mod:`kite_connect.trading.risk_manager`
4. Place orders via Kite  â†’  :mod:`kite_connect.trading.order_service`
5. Register with monitor  â†’  :mod:`kite_connect.trading.trade_monitor`

Signalâ†’Executor bridge: accepts analysis verdicts to filter
execution to only high-conviction BUY / STRONG_BUY signals.

Designed to be called from the Streamlit UI (sync) or from a
scheduled background job.
"""

from __future__ import annotations

import logging
import time
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from kite_connect.nse.nse_universe import get_nse_universe
from kite_connect.nse.screener import NSEScreener, ScreenerConfig
from kite_connect.trading.risk_manager import RiskManager, RiskConfig, TradePlan

logger = logging.getLogger(__name__)

# Allowed verdict tags for execution
# When Carver pipeline is active, HOLD verdicts also pass through (Carver does its own signal gating)
_BUY_TAGS = {"BUY", "STRONG_BUY"}
_CARVER_ALLOWED_TAGS = {"BUY", "STRONG_BUY", "HOLD"}  # P1 fix: Carver pipeline generates its own forecasts

# Rate-limiting: pause between Kite API calls (seconds)
_ORDER_DELAY_S = 0.15
_ORDER_TIMEOUT_S = 30  # Max seconds to wait for a single order placement


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Execution result
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@dataclass
class OrderResult:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    stop_loss: float
    target_price: float
    order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    # Gap E5: Execution quality tracking
    theoretical_price: float = 0.0   # price at signal generation
    actual_fill_price: float = 0.0   # actual fill from order book
    slippage_bps: float = 0.0        # abs(fill - theoretical) / theoretical × 10000

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_price": round(self.target_price, 2),
            "order_id": self.order_id or "",
            "sl_order_id": self.sl_order_id or "",
            "tp_order_id": self.tp_order_id or "",
            "success": self.success,
            "error": self.error or "",
            "theoretical_price": round(self.theoretical_price, 2),
            "actual_fill_price": round(self.actual_fill_price, 2),
            "slippage_bps": round(self.slippage_bps, 1),
        }


@dataclass
class ExecutionReport:
    """Full report returned after a screen-and-execute run."""
    timestamp: str = ""
    universe_size: int = 0
    screened_count: int = 0
    signal_filtered_count: int = 0
    plans_count: int = 0
    orders_placed: int = 0
    orders_failed: int = 0
    screened_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_plans: List[TradePlan] = field(default_factory=list)
    order_results: List[OrderResult] = field(default_factory=list)
    options_placed: int = 0
    options_failed: int = 0
    aronson_validated_signals: int = 0   # Aronson EBTA: count of statistically validated signals
    aronson_confidence_skipped: int = 0  # Aronson EBTA: trades skipped by confidence gate


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Executor
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class AutoExecutor:
    """
    End-to-end execution engine: screen â†’ signal-filter â†’ risk-check â†’ order.

    Parameters
    ----------
    kite : KiteConnect
        Authenticated Kite session (required for order placement,
        optional for screening if only yfinance data is needed).
    screener_cfg : ScreenerConfig | None
    risk_cfg : RiskConfig | None
    auto_place : bool
        If ``False`` (default), the engine produces plans but does
        **not** actually place orders.  Set to ``True`` to go live.
    """

    def __init__(
        self,
        kite=None,
        screener_cfg: ScreenerConfig | None = None,
        risk_cfg: RiskConfig | None = None,
        auto_place: bool = False,
        trade_monitor=None,
    ):
        self.kite = kite
        self.screener = NSEScreener(screener_cfg)
        self.auto_place = auto_place
        self._trade_monitor = trade_monitor

        # Carver framework: create VolatilityTarget and inject into RiskManager
        self._vol_target = None
        self._carver_enabled = False
        try:
            from config import Config
            if getattr(Config, "CARVER_ENABLED", False):
                from services.volatility_target import VolatilityTarget, VolatilityTargetConfig
                vt_cfg = VolatilityTargetConfig(
                    initial_capital=getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000.0),
                    annual_vol_target_pct=getattr(Config, "CARVER_ANNUAL_VOL_TARGET", 0.20),
                    max_leverage_factor=getattr(Config, "CARVER_MAX_LEVERAGE", 1.0),
                )
                self._vol_target = VolatilityTarget(vt_cfg)
                self._carver_enabled = True

                # Restore persisted capital state (peak equity + cumulative P&L)
                try:
                    import json as _json
                    import os as _os
                    _state_path = _os.path.join(
                        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                        "data", "portfolio_state.json",
                    )
                    if _os.path.exists(_state_path):
                        with open(_state_path) as _f:
                            _state = _json.load(_f)
                        cum_pnl = _state.get("cumulative_realized_pnl", 0.0)
                        if cum_pnl != 0:
                            self._vol_target.add_realized(cum_pnl)
                        Config._CUMULATIVE_REALIZED_PNL = cum_pnl
                        Config._PEAK_EQUITY = _state.get("peak_equity", vt_cfg.initial_capital)
                        logger.info("Restored portfolio state: cum_pnl=%.0f, peak=%.0f",
                                    cum_pnl, Config._PEAK_EQUITY)
                except Exception:
                    pass  # No persisted state yet

                logger.info("Carver framework enabled: vol_target=%.0f%%, capital=%.0f",
                            vt_cfg.annual_vol_target_pct * 100, self._vol_target.current_capital)
        except Exception as exc:
            logger.debug("Carver framework not available: %s", exc)

        self.risk_mgr = RiskManager(
            risk_cfg, kite=kite,
            volatility_target=self._vol_target,
        )

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run(
        self,
        symbols: Optional[List[str]] = None,
        progress_callback=None,
        signal_verdicts: Optional[Dict[str, str]] = None,
        pre_screened_df: Optional[pd.DataFrame] = None,
    ) -> ExecutionReport:
        """
        Execute the full pipeline.

        Parameters
        ----------
        symbols : list[str] | None
            If provided, screen only these symbols.
            If ``None``, download the full NSE universe first.
        progress_callback : callable | None
            ``callback(message: str)`` for UI progress updates.
        signal_verdicts : dict[str, str] | None
            Mapping of ``{symbol: decision_tag}`` from the analysis
            pipeline (e.g. ``{"RELIANCE": "STRONG_BUY", "TCS": "HOLD"}``).
            When provided, only symbols with BUY / STRONG_BUY tags
            are allowed through to execution (strict filter).
        pre_screened_df : pd.DataFrame | None
            Already-screened DataFrame from a prior pipeline run.
            When provided, the screening step is skipped entirely.

        Returns
        -------
        ExecutionReport
        """
        _cb = progress_callback or (lambda m: None)
        report = ExecutionReport(timestamp=datetime.now().isoformat())

        # â”€â”€ Fast-path: use pre-screened data (skip re-download) â”€
        if pre_screened_df is not None and not pre_screened_df.empty:
            screened_df = pre_screened_df
            report.universe_size = len(screened_df)
            report.screened_count = len(screened_df)
            _cb(f"Using pre-screened data: {len(screened_df)} stocks")
        else:
            # â”€â”€ 1.  Universe â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if symbols is None:
                _cb("Downloading NIFTY50 & NSE NEXT50")
                symbols = get_nse_universe(self.kite)
                # Auto-enable index mode for blue-chip universe
                if not self.screener.cfg.index_mode:
                    self.screener.cfg.index_mode = True
            report.universe_size = len(symbols)
            _cb(f"Universe: {len(symbols)} symbols")

            # â”€â”€ 2.  Screen â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            screened_df = self.screener.screen(symbols, progress_callback=_cb)
            report.screened_count = len(screened_df)

            if screened_df.empty:
                _cb("No stocks passed screening criteria")
                report.screened_df = screened_df
                return report

        report.screened_df = screened_df

        # â”€â”€ 2b. Signalâ†’Executor bridge: strict BUY-only filter â”€
        # If no verdicts provided, auto-generate them via IntegratedScorer
        if signal_verdicts is None and not screened_df.empty:
            signal_verdicts = self._auto_evaluate_verdicts(screened_df, _cb)

        if signal_verdicts:
            pre_filter = len(screened_df)
            # P1 fix: use expanded tags when Carver pipeline is active
            _active_tags = _CARVER_ALLOWED_TAGS if self._carver_enabled else _BUY_TAGS
            allowed = [
                sym for sym in screened_df["symbol"].tolist()
                if signal_verdicts.get(sym, "").upper() in _active_tags
            ]
            screened_df = screened_df[screened_df["symbol"].isin(allowed)]
            report.screened_df = screened_df
            report.signal_filtered_count = pre_filter - len(screened_df)
            _tag_label = "/".join(sorted(_active_tags))
            _cb(
                f"Signal filter: {len(allowed)} {_tag_label} passed, "
                f"{report.signal_filtered_count} rejected"
            )
            if screened_df.empty:
                _cb(f"No stocks passed signal filter -- skipping execution")
                return report

        _pre_ltp_closes = {}
        if not screened_df.empty:
            _pre_ltp_closes = dict(zip(screened_df["symbol"], screened_df["close"]))
        # â”€â”€ 3.  Enrich with live prices if Kite available â”€â”€â”€â”€â”€â”€
        if self.kite is not None and not screened_df.empty:
            screened_df = self._enrich_with_ltp(screened_df, _cb)
            report.screened_df = screened_df


        # G15: Stale verdict re-evaluation after LTP enrichment
        # If pre-scored verdicts were passed in and LTP drifted >5%, re-evaluate
        if signal_verdicts and _pre_ltp_closes and not screened_df.empty:
            drifted = []
            for _, row in screened_df.iterrows():
                sym = row["symbol"]
                old_close = _pre_ltp_closes.get(sym)
                new_close = row.get("close")
                if old_close and new_close and old_close > 0:
                    drift_pct = abs(new_close - old_close) / old_close
                    if drift_pct > 0.05:
                        drifted.append(sym)
            if drifted:
                _cb(f"G15: {len(drifted)} stocks drifted >5% -- re-evaluating verdicts")
                logger.info("Stale verdict re-eval for: %s", drifted)
                fresh = self._auto_evaluate_verdicts(
                    screened_df[screened_df["symbol"].isin(drifted)], _cb
                )
                revoked = []
                for sym in drifted:
                    new_verdict = fresh.get(sym, "HOLD").upper()
                    old_verdict = signal_verdicts.get(sym, "").upper()
                    if old_verdict in ("BUY", "STRONG_BUY") and new_verdict not in ("BUY", "STRONG_BUY"):
                        revoked.append(sym)
                        signal_verdicts[sym] = new_verdict
                if revoked:
                    screened_df = screened_df[~screened_df["symbol"].isin(revoked)]
                    report.screened_df = screened_df
                    _cb(f"G15: Revoked {len(revoked)} stale BUY verdicts: {revoked}")
                    logger.warning("Revoked stale verdicts: %s", revoked)
        # â”€â”€ 3b. Order book depth: filter illiquid stocks â”€â”€â”€â”€â”€â”€â”€
        if self.kite is not None and not screened_df.empty:
            screened_df = self._filter_by_spread(screened_df, _cb)
            report.screened_df = screened_df

        # ── 3b-vol. Tier 1 Gap 4: Volume filter — reject if order > 5% ADV ──
        if self.kite is not None and not screened_df.empty:
            screened_df = self._filter_by_volume(screened_df, _cb)
            report.screened_df = screened_df


        # -- 3c. P1 fix: Portfolio drawdown halt ---------------
        try:
            from services.portfolio_vol_monitor import assess_portfolio_risk
            from kite_connect.trading.order_service import get_holdings
            from config import Config
            if self.kite is not None:
                holdings = get_holdings(self.kite)
                held = [h for h in holdings if int(h.get("quantity", 0)) > 0]
                if held:
                    pos_values = {
                        h["tradingsymbol"]: float(h.get("last_price", 0)) * int(h.get("quantity", 0))
                        for h in held if h.get("last_price")
                    }
                    inst_vols = {s: 0.02 for s in pos_values}  # conservative 2% default
                    total_cap = getattr(Config, "CARVER_INITIAL_CAPITAL", 500_000)
                    peak_eq = getattr(Config, "_PEAK_EQUITY", None)
                    snap = assess_portfolio_risk(
                        pos_values, inst_vols,
                        target_annual_vol_pct=getattr(Config, "CARVER_ANNUAL_VOL_TARGET", 0.20),
                        total_capital=total_cap,
                        peak_equity=peak_eq,
                    )
                    # G13: Emergency liquidation on extreme drawdown (>20%)
                    if getattr(snap, 'emergency_liquidate', False) and held:
                        _cb(f'EMERGENCY LIQUIDATION: drawdown {snap.drawdown_pct:.1f}% - selling all positions')
                        logger.critical('Emergency liquidation triggered (DD=%.1f%%) - closing all positions', snap.drawdown_pct)
                        from kite_connect.trading.order_service import place_order
                        for h in held:
                            sym = h.get('tradingsymbol', '')
                            qty = int(h.get('quantity', 0))
                            if qty > 0 and sym:
                                try:
                                    place_order(self.kite, symbol=sym, exchange='NSE',
                                                transaction_type='SELL', quantity=qty,
                                                order_type='MARKET', product='CNC',
                                                tag='EMERGENCY_LIQUIDATE')
                                    logger.warning('Emergency SELL placed: %s x %d', sym, qty)
                                except Exception as liq_exc:
                                    logger.error('Emergency SELL failed for %s: %s', sym, liq_exc)
                        return report
                    if snap.risk_level.value == 'HALTED':
                        _cb(f'PORTFOLIO HALT: drawdown {snap.drawdown_pct:.1f}% exceeds limit - no new trades')
                        logger.warning('Portfolio DD halt triggered (%.1f%%) - blocking all new orders', snap.drawdown_pct)
                        return report
                    if snap.scale_factor < 1.0:
                        report._dd_scale = snap.scale_factor
                        _cb(f'Portfolio risk: scale factor {snap.scale_factor:.2f} applied (DD warning)')
        except Exception as exc:
            logger.debug('Portfolio DD check failed (non-fatal): %s', exc)

        # â”€â”€ 4.  Risk management / trade plans â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _cb("Generating trade plans with risk management â€¦")
        plans = self._generate_trade_plans(screened_df, _cb)

        # P1 fix: Apply portfolio DD scale_factor to position quantities
        dd_scale = getattr(report, '_dd_scale', 1.0)
        if dd_scale < 1.0 and plans:
            for plan in plans:
                plan.quantity = max(1, int(plan.quantity * dd_scale))
            _cb(f"DD scale {dd_scale:.2f} applied to {len(plans)} plans")

        report.trade_plans = plans
        report.plans_count = len(plans)

        if not plans:
            _cb("No trade plans met the R:R threshold")
            return report

        # â”€â”€ 4b. Portfolio correlation check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        plans = self._filter_correlated(plans, _cb)
        report.trade_plans = plans
        report.plans_count = len(plans)


        # ── 4c. Gap 6: Multi-timeframe entry confirmation ───────
        # Reject BUY entries where daily + weekly TradingView signals
        # disagree (both must be BUY or STRONG_BUY for swing/positional)
        if plans:
            plans = self._filter_by_mtf_consensus(plans, _cb)
            report.trade_plans = plans
            report.plans_count = len(plans)
        # â”€â”€ 5.  Order placement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.auto_place and self.kite is not None:
            _cb(f"Placing {len(plans)} orders via Kite â€¦")
            report.order_results = self._place_orders(plans, _cb)
            report.orders_placed = sum(1 for r in report.order_results if r.success)
            report.orders_failed = sum(1 for r in report.order_results if not r.success)
            _cb(
                f"Orders: {report.orders_placed} placed, "
                f"{report.orders_failed} failed"
            )
            # M2 fix: persist orders to database
            self._persist_orders(report.order_results, plans)

            # ── GAP-2: Execute options overlay (covered calls + CSP) via Kite NFO ──
            if hasattr(self, '_last_pipe_result') and self._last_pipe_result is not None:
                opt_placed, opt_failed = self._execute_options_overlay(
                    self._last_pipe_result, _cb
                )
                report.options_placed = opt_placed
                report.options_failed = opt_failed

                # ── FIX-05: Execute bear hedge orders when available ──
                hedge_placed, hedge_failed = self._execute_bear_hedges(
                    self._last_pipe_result, _cb
                )
                if hedge_placed > 0:
                    report.options_placed += hedge_placed
                    report.options_failed += hedge_failed
        else:
            _cb(
                f"Dry run â€” {len(plans)} plans generated "
                "(auto_place=False or no Kite session)"
            )

        return report

    # -- Carver-aware trade plan generation ------------------------------

    def _generate_trade_plans(
        self, screened_df: pd.DataFrame, _cb,
    ) -> List[TradePlan]:
        """Generate trade plans using Carver pipeline when enabled, else legacy.

        When CARVER_ENABLED=True:
          Delegates to the full CarverPipeline.run() which includes all 8+
          forecast sources, HMM regime blending, strategy decay, Markov
          signal filter, forecast capacity checks, and options overlay.

        Falls back to legacy plan_trades() if anything fails.
        """
        if not self._carver_enabled or self._vol_target is None:
            return self.risk_mgr.plan_trades(screened_df)

        try:
            _cb("Full Carver pipeline: running all forecast sources + HMM regime …")
            from services.carver_pipeline import CarverPipeline
            from config import Config
            from utils import download_ind_ohlcv

            symbols = screened_df["symbol"].tolist()

            # Step 1: Fetch OHLCV for pipeline (batch mode for large universes)
            ohlcv_cache = {}
            if len(symbols) > 30:
                try:
                    from utils import download_ohlcv_batch_parallel
                    ohlcv_cache = download_ohlcv_batch_parallel(
                        symbols, market="IND", period="6mo",
                    )
                    ohlcv_cache = {s: d for s, d in ohlcv_cache.items() if len(d) >= 64}
                    _cb(f"Batch OHLCV: {len(ohlcv_cache)}/{len(symbols)} tickers")
                except Exception as exc:
                    logger.warning("Batch OHLCV failed (%s), falling back to sequential", exc)
                    ohlcv_cache = {}

            if not ohlcv_cache:
                for sym in symbols:
                    try:
                        df = download_ind_ohlcv(sym, period="6mo")
                        if df is not None and len(df) >= 64:
                            ohlcv_cache[sym] = df
                    except Exception:
                        pass

            if not ohlcv_cache:
                logger.warning("Carver: no OHLCV data — falling back to legacy")
                return self.risk_mgr.plan_trades(screened_df)

            # Build screener scores for screener_to_forecast
            score_col = "score"
            screener_scores = {}
            for sym in ohlcv_cache:
                row_match = screened_df[screened_df["symbol"] == sym]
                if not row_match.empty and score_col in row_match.columns:
                    screener_scores[sym] = float(row_match.iloc[0][score_col])

            # Build decision engine scores if available
            decision_scores = {}
            try:
                from services.integrated_scorer import IntegratedScorer
                scorer = IntegratedScorer()
                for sym in list(ohlcv_cache.keys())[:20]:
                    try:
                        result = scorer.evaluate(sym, market="IND")
                        if hasattr(result, "final_score"):
                            decision_scores[sym] = result.final_score
                    except Exception:
                        pass
            except Exception:
                pass

            # Run the FULL Carver pipeline with all sources
            pipeline = CarverPipeline()
            pipe_result = pipeline.run(
                ohlcv_cache=ohlcv_cache,
                screener_scores=screener_scores,
                decision_engine_scores=decision_scores if decision_scores else None,
            )

            if not pipe_result.trade_plans:
                logger.info("Carver full pipeline: 0 trade plans — falling back to legacy")
                return self.risk_mgr.plan_trades(screened_df)

            # Convert PipelineResult.trade_plans to TradePlan objects
            plans = []
            for tp in pipe_result.trade_plans:
                try:
                    plan = TradePlan(
                        symbol=tp.symbol,
                        side=tp.side,
                        entry_price=tp.entry_price,
                        stop_loss=tp.stop_loss,
                        target_price=tp.target_price,
                        quantity=tp.quantity,
                        risk_amount=round((tp.entry_price - tp.stop_loss) * tp.quantity, 2),
                        reward_amount=round((tp.target_price - tp.entry_price) * tp.quantity, 2),
                        rr_ratio=round((tp.target_price - tp.entry_price) / max(tp.entry_price - tp.stop_loss, 0.01), 2),
                        score=screener_scores.get(tp.symbol, 0.5),
                    )
                    plans.append(plan)
                except Exception as exc:
                    logger.debug("TradePlan conversion failed for %s: %s", tp.symbol, exc)

            _cb(f"Full Carver pipeline: {len(plans)} trade plans from {pipe_result.symbols_processed} symbols")

            # Aronson EBTA: log validation stats from pipeline result
            _vstats = getattr(pipe_result, 'validation_stats', {})
            if _vstats:
                _skipped = _vstats.get('skipped_count', 0)
                if _skipped > 0:
                    _cb(f"Aronson: {_skipped} symbols skipped by confidence gate")

            # Store pipeline result for options overlay access
            self._last_pipe_result = pipe_result
            return plans

        except Exception as exc:
            logger.warning("Carver pipeline failed — falling back to legacy: %s", exc)
            _cb("Carver pipeline error — using legacy risk management")
            return self.risk_mgr.plan_trades(screened_df)

    # -- Gap 6: Multi-timeframe entry confirmation -----------------------

    def _filter_by_mtf_consensus(
        self, plans: List[TradePlan], _cb,
    ) -> List[TradePlan]:
        """Reject BUY plans where daily and weekly TradingView signals disagree.

        For swing/positional trades, both 1D and 1W timeframes must show
        BUY or STRONG_BUY.  If either shows SELL/STRONG_SELL, the plan is
        rejected.  NEUTRAL is permitted (no veto).
        """
        _BULLISH = {"BUY", "STRONG_BUY"}
        _BEARISH = {"SELL", "STRONG_SELL"}

        try:
            from services.technical_analysis.tradingview import fetch_tradingview_consensus
        except ImportError:
            logger.debug("tradingview_ta not available -- MTF gate skipped")
            return plans

        approved: List[TradePlan] = []
        rejected_count = 0

        for plan in plans:
            ticker = plan.symbol
            # Append .NS for Indian equities if not already present
            tv_ticker = f"{ticker}.NS" if not ticker.endswith((".NS", ".BO")) else ticker

            try:
                consensus = fetch_tradingview_consensus(
                    tv_ticker, timeframes=["1d", "1W"],
                )
                if not consensus.available:
                    # Gap B fix: TradingView unavailable — fall back to
                    # local technical consensus (RSI > 50 + Close > MA200)
                    if self._local_technical_consensus(ticker):
                        approved.append(plan)
                    else:
                        _cb(
                            f"  MTF fallback rejected {ticker} -- "
                            f"TradingView offline, local technicals bearish"
                        )
                        rejected_count += 1
                    continue

                daily = consensus.timeframes.get("1d")
                weekly = consensus.timeframes.get("1W")

                daily_rec = daily.recommendation if daily else "NEUTRAL"
                weekly_rec = weekly.recommendation if weekly else "NEUTRAL"

                # Veto: reject if EITHER timeframe is bearish
                if daily_rec in _BEARISH or weekly_rec in _BEARISH:
                    _cb(
                        f"  MTF rejected {ticker} -- "
                        f"1D={daily_rec}, 1W={weekly_rec}"
                    )
                    rejected_count += 1
                    continue

                # For swing trades: at least daily must be bullish
                if daily_rec not in _BULLISH:
                    _cb(
                        f"  MTF skipped {ticker} -- "
                        f"1D={daily_rec} (not bullish), 1W={weekly_rec}"
                    )
                    rejected_count += 1
                    continue

                approved.append(plan)

            except Exception as exc:
                logger.debug("MTF check failed for %s: %s -- using local fallback", ticker, exc)
                if self._local_technical_consensus(ticker):
                    approved.append(plan)
                else:
                    _cb(
                        f"  MTF fallback rejected {ticker} -- "
                        f"TradingView error, local technicals bearish"
                    )
                    rejected_count += 1

        if rejected_count:
            _cb(
                f"MTF gate: {len(approved)} passed, {rejected_count} rejected "
                f"(daily/weekly disagreement)"
            )

        return approved


    # -- Gap B fix: local technical fallback when TradingView offline ----

    @staticmethod
    def _local_technical_consensus(symbol: str) -> bool:
        """Quick local check: RSI > 50 AND Close > 200-day MA.

        Used as a fallback when TradingView is unavailable.  Returns True
        (bullish) if both conditions are met, False otherwise.
        """
        try:
            from utils import download_ind_ohlcv
            df = download_ind_ohlcv(f"{symbol}.NS", period="1y")
            if df is None or df.empty or len(df) < 200:
                return False
            close = df["Close"].squeeze()
            # RSI (14-period)
            delta = close.diff().dropna()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi = float((100 - 100 / (1 + rs)).iloc[-1])
            # MA200
            ma200 = float(close.rolling(200).mean().iloc[-1])
            last_close = float(close.iloc[-1])
            return rsi > 50 and last_close > ma200
        except Exception:
            return False  # conservative: reject if local check also fails

    # â”€â”€ Order persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _persist_orders(order_results: list, trade_plans: list) -> None:
        """Best-effort save of order records to the database."""
        try:
            from database.service import DatabaseService
            db = DatabaseService()
            db.save_orders(order_results, trade_plans)
        except Exception as exc:
            logger.warning("Order persistence failed (non-fatal): %s", exc)

    def _execute_options_overlay(self, pipe_result, _cb) -> tuple:
        """GAP-2: Execute options overlay orders via Kite NFO.

        Reads OptionOrder objects from pipe_result.options_overlay and places
        them as NRML LIMIT orders on the NFO exchange for covered calls and
        cash-secured puts.

        Returns (placed_count, failed_count).
        """
        placed = 0
        failed = 0
        try:
            overlay = getattr(pipe_result, 'options_overlay', None)
            if overlay is None:
                return (0, 0)

            all_orders = []
            if hasattr(overlay, 'covered_call_orders'):
                all_orders.extend(overlay.covered_call_orders or [])
            if hasattr(overlay, 'put_write_orders'):
                all_orders.extend(overlay.put_write_orders or [])

            if not all_orders:
                logger.info("Options overlay: no orders to execute")
                return (0, 0)

            from kite_connect.trading.order_service import place_order

            _cb(f"Options overlay: {len(all_orders)} NFO orders to place")

            for opt in all_orders:
                try:
                    bare = opt.symbol.replace('.NS', '').replace('.BO', '')
                    from datetime import datetime as _dt
                    exp = _dt.strptime(opt.expiry_date, "%Y-%m-%d")
                    exp_str = exp.strftime("%y%b").upper() + exp.strftime("%d")
                    strike_str = str(int(opt.strike))
                    nfo_symbol = f"{bare}{exp_str}{strike_str}{opt.option_type}"

                    qty = opt.lots * opt.lot_size
                    tx_type = opt.action

                    logger.info(
                        "Options overlay: %s %s x %d @ %.2f (%s)",
                        tx_type, nfo_symbol, qty, opt.premium, opt.strategy,
                    )

                    result = place_order(
                        kite=self.kite,
                        symbol=nfo_symbol,
                        exchange="NFO",
                        transaction_type=tx_type,
                        quantity=qty,
                        order_type="LIMIT",
                        product="NRML",
                        price=round(opt.premium, 2),
                        tag=f"OPT_{opt.strategy[:3]}",
                    )

                    if result and result.get("success", True):
                        placed += 1
                        _cb(f"  Options placed: {tx_type} {nfo_symbol} x {qty}")
                    else:
                        failed += 1
                        err = result.get("error", "unknown") if result else "no result"
                        _cb(f"  Options failed: {nfo_symbol}: {err}")

                except Exception as exc:
                    failed += 1
                    logger.warning("Options order failed for %s: %s",
                                   getattr(opt, 'symbol', '?'), exc)

            _cb(f"Options overlay: {placed} placed, {failed} failed")

        except Exception as exc:
            logger.warning("Options overlay execution failed: %s", exc)

        return (placed, failed)

    def _execute_bear_hedges(self, pipe_result, _cb) -> tuple:
        """FIX-05: Execute bear hedge protective put orders via Kite NFO.

        Reads BearHedgeResult candidates from pipe_result.hedge_result and
        places protective put orders.  Only runs in BEAR/CRISIS regime.

        Returns (placed_count, failed_count).
        """
        placed = 0
        failed = 0
        try:
            hedge_result = getattr(pipe_result, 'hedge_result', None)
            if hedge_result is None:
                return (0, 0)

            candidates = getattr(hedge_result, 'candidates', [])
            if not candidates:
                return (0, 0)

            from kite_connect.trading.order_service import place_order

            _cb(f"Bear hedge: {len(candidates)} protective put orders to place")

            for cand in candidates:
                try:
                    orders = getattr(cand, 'orders', [])
                    if not orders:
                        continue
                    for opt in orders:
                        bare = opt.symbol.replace('.NS', '').replace('.BO', '')
                        from datetime import datetime as _dt
                        exp = _dt.strptime(opt.expiry_date, "%Y-%m-%d")
                        exp_str = exp.strftime("%y%b").upper() + exp.strftime("%d")
                        strike_str = str(int(opt.strike))
                        nfo_symbol = f"{bare}{exp_str}{strike_str}{opt.option_type}"

                        qty = opt.lots * opt.lot_size
                        tx_type = opt.action

                        logger.info(
                            "Bear hedge: %s %s x %d @ %.2f",
                            tx_type, nfo_symbol, qty, opt.premium,
                        )

                        result = place_order(
                            kite=self.kite,
                            symbol=nfo_symbol,
                            exchange="NFO",
                            transaction_type=tx_type,
                            quantity=qty,
                            order_type="LIMIT",
                            product="NRML",
                            price=round(opt.premium, 2),
                            tag="HEDGE_PUT",
                        )

                        if result and result.get("success", True):
                            placed += 1
                            _cb(f"  Hedge placed: {tx_type} {nfo_symbol} x {qty}")
                        else:
                            failed += 1
                            err = result.get("error", "unknown") if result else "no result"
                            _cb(f"  Hedge failed: {nfo_symbol}: {err}")

                except Exception as exc:
                    failed += 1
                    logger.warning("Bear hedge order failed: %s", exc)

            if placed > 0:
                _cb(f"Bear hedge: {placed} placed, {failed} failed")

        except Exception as exc:
            logger.warning("Bear hedge execution failed: %s", exc)

        return (placed, failed)

#11) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _filter_by_spread(self, screened_df: pd.DataFrame, _cb) -> pd.DataFrame:
        """Remove stocks with bid-ask spread > 0.5%. Reduce position for 0.3-0.5%."""
        try:
            symbols = screened_df["symbol"].tolist()
            instrument_keys = [f"NSE:{s}" for s in symbols]
            quote_data = self.kite.quote(instrument_keys)

            remove_syms = set()
            reduce_syms = {}  # {symbol: scale_factor}
            for idx, row in screened_df.iterrows():
                key = f"NSE:{row['symbol']}"
                depth = quote_data.get(key, {}).get("depth", {})
                buy_depth = depth.get("buy", [])
                sell_depth = depth.get("sell", [])

                if buy_depth and sell_depth:
                    best_bid = buy_depth[0].get("price", 0)
                    best_ask = sell_depth[0].get("price", 0)
                    if best_bid > 0 and best_ask > 0:
                        spread_pct = (best_ask - best_bid) / best_bid
                        if spread_pct > 0.005:  # > 0.5% spread - too illiquid, remove
                            remove_syms.add(row["symbol"])
                            _cb(f"  Removed {row['symbol']} - spread {spread_pct:.2%} > 0.5%")
                        elif spread_pct > 0.003:  # 0.3-0.5% - reduce position by 50%
                            reduce_syms[row["symbol"]] = 0.5
                            _cb(f"  Reduce {row['symbol']} - spread {spread_pct:.2%} (position halved)")

            if remove_syms:
                screened_df = screened_df[~screened_df["symbol"].isin(remove_syms)]
                _cb(f"Depth filter removed {len(remove_syms)} illiquid stocks")

            # Tag reduced-position symbols for downstream sizing
            if reduce_syms:
                screened_df = screened_df.copy()
                screened_df["spread_scale"] = screened_df["symbol"].map(
                    lambda s: reduce_syms.get(s, 1.0)
                )
                _cb(f"Depth filter: {len(reduce_syms)} stocks position-reduced by 50%")
            elif "spread_scale" not in screened_df.columns:
                screened_df = screened_df.copy()
                screened_df["spread_scale"] = 1.0

        except Exception as exc:
            logger.warning("Depth filter failed (non-fatal): %s", exc)

        return screened_df


    # -- Tier 1 Gap 4: Volume filter -- reject order > 5% of 20-day ADV --

    def _filter_by_volume(self, screened_df: "pd.DataFrame", _cb) -> "pd.DataFrame":
        """Remove stocks where estimated order size would exceed 5% of 20-day ADV."""
        try:
            import yfinance as yf

            symbols = screened_df["symbol"].tolist()
            remove_syms = set()
            for sym in symbols:
                try:
                    hist = yf.download(
                        f"{sym}.NS", period="30d", progress=False, timeout=10,
                    )
                    if hist.empty or len(hist) < 5:
                        continue
                    adv = float(hist["Volume"].tail(20).mean())
                    if adv <= 0:
                        continue
                    row = screened_df[screened_df["symbol"] == sym].iloc[0]
                    order_qty = float(row.get("quantity", row.get("qty", 0)) or 0)
                    if order_qty > 0:
                        order_pct = order_qty / adv
                        if order_pct > 0.05:
                            remove_syms.add(sym)
                            _cb(f"  Volume filter: removed {sym} -- order {order_pct:.1%} of ADV")
                except Exception:
                    continue

            if remove_syms:
                screened_df = screened_df[~screened_df["symbol"].isin(remove_syms)]
                _cb(f"Volume filter removed {len(remove_syms)} over-concentrated stocks")
            else:
                _cb("Volume filter: all stocks within 5% ADV limit")

        except Exception as exc:
            logger.warning("Volume filter failed (non-fatal): %s", exc)

        return screened_df

    # â”€â”€ Portfolio correlation check (#6) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _filter_correlated(self, plans: List[TradePlan], _cb) -> List[TradePlan]:
        """Block trades if avg pairwise correlation with existing positions > 0.7."""
        if not plans or self.kite is None:
            return plans

        try:
            import yfinance as yf
            import numpy as np

            # Get existing held symbols
            from kite_connect.trading.order_service import get_holdings
            holdings = get_holdings(self.kite)
            held_syms = [h.get("tradingsymbol", "") for h in holdings
                         if int(h.get("quantity", 0)) > 0]

            if not held_syms:
                return plans  # No positions to correlate against

            # Combine held + proposed symbols
            proposed_syms = [p.symbol for p in plans]
            all_syms = list(set(held_syms + proposed_syms))

            # Download 60-day close prices (Bhavcopy â†’ yfinance)
            from utils import download_ind_ohlcv_batch
            ohlcv = download_ind_ohlcv_batch(all_syms, period="60d")
            if not ohlcv:
                return plans

            # Build returns matrix
            closes = pd.DataFrame({
                sym: df["Close"].squeeze() for sym, df in ohlcv.items()
            })
            returns = closes.pct_change().dropna()

            if returns.shape[1] < 2:
                return plans

            corr_matrix = returns.corr()

            approved: List[TradePlan] = []
            for plan in plans:
                sym_key = plan.symbol
                if sym_key not in corr_matrix.columns:
                    approved.append(plan)
                    continue

                # Check avg correlation with held positions
                held_keys = [s for s in held_syms if s in corr_matrix.columns]
                if not held_keys:
                    approved.append(plan)
                    continue

                corrs = [abs(corr_matrix.loc[sym_key, hk])
                         for hk in held_keys if hk != sym_key and hk in corr_matrix.index]

                if corrs:
                    avg_corr = float(np.mean(corrs))
                    if avg_corr > 0.7:
                        _cb(f"  Blocked {plan.symbol} â€” avg correlation {avg_corr:.2f} > 0.7 with portfolio")
                        continue

                approved.append(plan)

            blocked = len(plans) - len(approved)
            if blocked > 0:
                _cb(f"Correlation filter blocked {blocked} highly-correlated trades")
            return approved

        except Exception as exc:
            logger.warning("Correlation filter failed (non-fatal): %s", exc)
            return plans
    # â”€â”€ Live price enrichment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _enrich_with_ltp(
        self, screened_df: pd.DataFrame, _cb
    ) -> pd.DataFrame:
        """Replace stale 'close' prices with live LTP from Kite."""
        try:
            symbols = screened_df["symbol"].tolist()
            instrument_keys = [f"NSE:{s}" for s in symbols]
            ltp_data = self.kite.ltp(instrument_keys)
            updated = 0
            df = screened_df.copy()
            for idx, row in df.iterrows():
                key = f"NSE:{row['symbol']}"
                if key in ltp_data:
                    live_price = ltp_data[key].get("last_price")
                    if live_price and live_price > 0:
                        df.at[idx, "close"] = live_price
                        updated += 1
            _cb(f"Enriched {updated}/{len(symbols)} stocks with live prices")
            return df
        except Exception as exc:
            logger.warning("LTP enrichment failed, using screener close: %s", exc)
            _cb("Live price fetch failed â€” using screener close prices")
            return screened_df

    # â”€â”€ Auto-verdict via IntegratedScorer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _auto_evaluate_verdicts(
        screened_df: pd.DataFrame, _cb
    ) -> Dict[str, str]:
        """Run IntegratedScorer on screened stocks to generate BUY/SELL verdicts.

        This ensures every auto-placed order passes through fundamental,
        technical, macro, and robustness validation â€” not just the
        technical screener.
        """
        try:
            from services.integrated_scorer import IntegratedScorer
            from datetime import date, timedelta

            symbols = screened_df["symbol"].tolist()
            ns_tickers = [f"{s}.NS" for s in symbols]
            _cb(f"Running IntegratedScorer on {len(ns_tickers)} stocks â€¦")

            scorer = IntegratedScorer()
            end_dt = date.today()
            start_dt = end_dt - timedelta(days=365)
            verdicts = scorer.evaluate(
                tickers=ns_tickers,
                market="IND",
                date_range=(str(start_dt), str(end_dt)),
            )

            signal_dict: Dict[str, str] = {}
            buy_count = 0
            for v in verdicts:
                bare = v.ticker.replace(".NS", "").replace(".BO", "")
                signal_dict[bare] = v.classification
                if v.classification in _BUY_TAGS:
                    buy_count += 1

            _cb(
                f"Verdict: {buy_count} BUY/STRONG_BUY, "
                f"{len(signal_dict) - buy_count} HOLD/SELL out of {len(signal_dict)}"
            )
            return signal_dict
        except Exception as exc:
            logger.warning("Auto-verdict failed (non-fatal): %s", exc)
            _cb("IntegratedScorer unavailable â€” proceeding without verdict filter")
            return {}

    # â”€â”€ Market hours check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _is_nse_market_open() -> bool:
        """Check if NSE is within trading hours (9:15 AM â€“ 3:30 PM IST, weekdays)."""
        from datetime import timezone, timedelta
        _IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(_IST)
        if now.weekday() > 4:  # Saturday=5, Sunday=6
            return False
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    def _is_at_circuit_limit(self, symbol: str) -> bool:
        """Check if a stock is at its upper/lower circuit limit.

        Uses Kite OHLC data: if the daily move exceeds the circuit
        threshold (default 20%), or if the stock's last traded price
        equals the upper/lower circuit price, skip the order.
        """
        if self.kite is None:
            return False
        try:
            from config import Config
            key = f"NSE:{symbol}"
            data = self.kite.ohlc([key])
            ohlc = data.get(key, {}).get("ohlc", {})
            ltp = data.get(key, {}).get("last_price", 0)
            day_open = ohlc.get("open", 0)

            if day_open > 0 and ltp > 0:
                daily_move = abs(ltp - day_open) / day_open
                if daily_move >= Config.CIRCUIT_BREAKER_PCT:
                    logger.warning(
                        "%s: daily move %.1f%% >= %.0f%% circuit threshold â€” skipping",
                        symbol, daily_move * 100, Config.CIRCUIT_BREAKER_PCT * 100,
                    )
                    return True

            # Also check if lower_circuit_limit / upper_circuit_limit
            # are available in the instrument data
            lower = data.get(key, {}).get("lower_circuit_limit", 0)
            upper = data.get(key, {}).get("upper_circuit_limit", 0)
            if lower and ltp and ltp <= lower:
                return True
            if upper and ltp and ltp >= upper:
                return True

        except Exception as exc:
            logger.debug("Circuit-breaker check failed for %s: %s", symbol, exc)
        return False

    # â”€â”€ Earnings blackout detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _get_earnings_blackout_symbols(symbols: List[str]) -> set:
        """Return symbols that are near earnings announcements.

        Uses yfinance calendar data to detect upcoming/recent earnings.
        Returns a set of symbols in blackout (BUY suppressed).
        """
        blackout: set = set()
        try:
            import yfinance as yf
            from config import Config

            before = getattr(Config, "EARNINGS_BLACKOUT_DAYS_BEFORE", 2)
            after = getattr(Config, "EARNINGS_BLACKOUT_DAYS_AFTER", 1)
            today = datetime.now().date()

            for sym in symbols:
                try:
                    from utils import yf_nse_symbol
                    ticker = yf.Ticker(yf_nse_symbol(sym))
                    cal = ticker.calendar
                    if cal is None or (hasattr(cal, 'empty') and cal.empty):
                        continue
                    # yfinance calendar may be a dict or DataFrame
                    earnings_date = None
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if ed:
                            earnings_date = ed[0] if isinstance(ed, list) else ed
                    elif hasattr(cal, "loc"):
                        try:
                            ed = cal.loc["Earnings Date"]
                            earnings_date = ed.iloc[0] if hasattr(ed, 'iloc') else ed
                        except Exception:
                            pass

                    if earnings_date is not None:
                        if hasattr(earnings_date, 'date'):
                            earnings_date = earnings_date.date()
                        from datetime import timedelta
                        window_start = earnings_date - timedelta(days=before)
                        window_end = earnings_date + timedelta(days=after)
                        if window_start <= today <= window_end:
                            blackout.add(sym)
                            logger.info(
                                "%s: earnings on %s â€” blackout active",
                                sym, earnings_date,
                            )
                except Exception:
                    continue  # Skip if calendar unavailable
        except Exception as exc:
            logger.debug("Earnings blackout check failed: %s", exc)

        return blackout

    # â”€â”€ Order placement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _place_orders(
        self, plans: List[TradePlan], _cb
    ) -> List[OrderResult]:
        from kite_connect.trading.order_service import place_order, get_order_book
        from kite_connect.trading.trade_monitor import TradeMonitor, MonitoredTrade

        results: List[OrderResult] = []

        # ── T5-4: KILL SWITCH check — block all orders if activated ──
        try:
            from config import Config as _KillCfg
            if getattr(_KillCfg, 'KILL_SWITCH', False):
                _cb("⛔ KILL SWITCH is active — all order placement blocked")
                logger.critical("_place_orders: KILL_SWITCH=True, blocking %d plans", len(plans))
                for plan in plans:
                    results.append(OrderResult(
                        symbol=plan.symbol, side=plan.side,
                        quantity=plan.quantity, entry_price=plan.entry_price,
                        stop_loss=plan.stop_loss, target_price=plan.target_price,
                        success=False, error="KILL SWITCH active — orders blocked",
                    ))
                return results
        except Exception as ks_exc:
            logger.warning("Kill switch check error (non-blocking): %s", ks_exc)

        # ── T5-4: Daily loss limit pre-check ──
        try:
            from kite_connect.trading.risk_manager import RiskManager as _DLRisk
            from config import Config as _DLCfg
            capital = getattr(_DLCfg, 'CARVER_INITIAL_CAPITAL', 500_000)
            # Estimate today's realized P&L from positions
            daily_pnl = 0.0
            try:
                positions = self.kite.positions()
                day_positions = positions.get("day", []) if positions else []
                daily_pnl = sum(float(p.get("pnl", 0)) for p in day_positions)
            except Exception as pos_exc:
                # T6-6: Conservative fallback — assume loss at limit threshold
                # rather than permissively assuming pnl=0
                daily_pnl = -(capital * 0.03)
                logger.warning("Daily P&L fetch failed — conservative estimate ₹%.0f: %s", daily_pnl, pos_exc)
            if _DLRisk.check_daily_loss_limit(capital, daily_pnl):
                _cb(f"⛔ Daily loss limit breached (P&L: ₹{daily_pnl:,.0f}) — orders blocked")
                logger.critical("_place_orders: Daily loss limit hit, blocking %d plans", len(plans))
                for plan in plans:
                    results.append(OrderResult(
                        symbol=plan.symbol, side=plan.side,
                        quantity=plan.quantity, entry_price=plan.entry_price,
                        stop_loss=plan.stop_loss, target_price=plan.target_price,
                        success=False, error=f"Daily loss limit breached (P&L: ₹{daily_pnl:,.0f})",
                    ))
                return results
        except Exception as dl_exc:
            logger.warning("Daily loss limit check error (non-blocking): %s", dl_exc)

        # â”€â”€ L1 fix: session expiry fast-fail â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            self.kite.profile()
        except Exception as exc:
            _cb("Kite session expired â€” please re-authenticate via Fly Kite")
            logger.error("Kite session check failed: %s", exc)
            for plan in plans:
                results.append(OrderResult(
                    symbol=plan.symbol, side=plan.side,
                    quantity=plan.quantity, entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss, target_price=plan.target_price,
                    success=False, error="Kite session expired â€” re-authenticate",
                ))
            return results

        # â”€â”€ Market hours guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not self._is_nse_market_open():
            _cb("NSE market is closed â€” orders not placed")
            logger.warning("Order placement blocked: NSE market is closed")
            for plan in plans:
                results.append(OrderResult(
                    symbol=plan.symbol, side=plan.side,
                    quantity=plan.quantity, entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss, target_price=plan.target_price,
                    success=False, error="Market closed (NSE hours: 9:15 AM â€“ 3:30 PM IST)",
                ))
            return results

        # â”€â”€ Duplicate check: skip symbols with open BUY orders â”€â”€â”€â”€â”€
        existing_symbols: set = set()
        try:
            order_book = get_order_book(self.kite)
            for o in order_book:
                if (
                    o.get("status") in ("OPEN", "TRIGGER PENDING", "COMPLETE")
                    and o.get("transaction_type") in ("BUY", "SELL")
                ):
                    existing_symbols.add(o.get("tradingsymbol", ""))
        except Exception:
            pass  # proceed without dedup if order book fails

        # â”€â”€ Monitor for post-trade SL/TP lifecycle (reuse existing) â”€
        monitor = None
        if self._trade_monitor is not None:
            self._trade_monitor.kite = self.kite
            monitor = self._trade_monitor
        if monitor is None:
            monitor = TradeMonitor(self.kite)

        # â”€â”€ Pre-compute earnings blackout set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        earnings_blackout_syms = self._get_earnings_blackout_symbols(
            [p.symbol for p in plans]
        )

        for plan in plans:
            # Skip if order already exists for this symbol
            if plan.symbol in existing_symbols:
                _cb(f"  Skipped {plan.symbol} â€” open order already exists")
                results.append(OrderResult(
                    symbol=plan.symbol, side=plan.side,
                    quantity=plan.quantity, entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss, target_price=plan.target_price,
                    success=False, error="Duplicate â€” open order exists",
                ))
                continue

            # â”€â”€ Earnings blackout check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if plan.symbol in earnings_blackout_syms:
                _cb(f"  Skipped {plan.symbol} â€” earnings blackout period")
                results.append(OrderResult(
                    symbol=plan.symbol, side=plan.side,
                    quantity=plan.quantity, entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss, target_price=plan.target_price,
                    success=False,
                    error="Earnings blackout â€” BUY suppressed near results",
                ))
                continue

            # â”€â”€ S8: Circuit-breaker check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if self._is_at_circuit_limit(plan.symbol):
                _cb(f"  Skipped {plan.symbol} â€” at circuit limit")
                results.append(OrderResult(
                    symbol=plan.symbol, side=plan.side,
                    quantity=plan.quantity, entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss, target_price=plan.target_price,
                    success=False,
                    error="Circuit limit hit â€” stock frozen",
                ))
                continue

            _cb(f"  Placing {plan.side} {plan.symbol} Ã— {plan.quantity} â€¦")

            # G14: LTP re-check immediately before order placement
            # Refreshes entry_price to avoid stale limit prices
            try:
                ltp_key = f"NSE:{plan.symbol}"
                ltp_data = self.kite.ltp([ltp_key])
                fresh_ltp = ltp_data.get(ltp_key, {}).get("last_price", 0)
                if fresh_ltp > 0:
                    drift_pct = abs(fresh_ltp - plan.entry_price) / plan.entry_price
                    if drift_pct > 0.02:
                        logger.info(
                            "G14: %s price drifted %.1f%% (plan=%.2f ltp=%.2f), updating",
                            plan.symbol, drift_pct * 100, plan.entry_price, fresh_ltp,
                        )
                        plan.entry_price = round(fresh_ltp, 2)
            except Exception:
                pass  # proceed with original price if LTP fetch fails

            # G4 fix: All orders use CNC + TradeMonitor for SL/TP lifecycle.
            # Phase 2: SHORT trades use SELL-first with MIS/NRML product.
            # Phase 3: LONG trades can use NRML (F&O) when plan.product is set.
            is_short = getattr(plan, "direction", "LONG") == "SHORT"
            if True:
                if is_short:
                    order_type = "LIMIT"
                    product = getattr(plan, "product", "MIS")
                else:
                    order_type = "MARKET" if plan.side == "SELL" else "LIMIT"
                    product = getattr(plan, "product", "CNC")

                # Margin pre-check for leveraged (non-CNC) orders
                if product != "CNC":
                    try:
                        from kite_connect.trading.margin_monitor import check_margin_before_order
                        est_margin = plan.entry_price * plan.quantity * getattr(Config, "FUTURES_MARGIN_PCT", 0.12)
                        if not check_margin_before_order(self.kite, est_margin):
                            _cb(f"  Skipped {plan.symbol} \u2014 insufficient margin for {product}")
                            results.append(OrderResult(
                                symbol=plan.symbol, side=plan.side,
                                quantity=plan.quantity, entry_price=plan.entry_price,
                                stop_loss=plan.stop_loss, target_price=plan.target_price,
                                success=False, error=f"Margin check failed for {product} order",
                            ))
                            continue
                    except Exception as margin_exc:
                        logger.warning("Margin pre-check error (non-blocking): %s", margin_exc)

                exchange = "NFO" if product == "NRML" else "NSE"
                order_kwargs = dict(
                    kite=self.kite,
                    symbol=plan.symbol,
                    exchange=exchange,
                    transaction_type=plan.side,
                    quantity=plan.quantity,
                    order_type=order_type,
                    product=product,
                )
                if order_type == "LIMIT":
                    order_kwargs["price"] = plan.entry_price

                # T5-6: Route TWAP-tagged orders through algo executor
                _algo = getattr(plan, 'execution_algo', None)
                if _algo and _algo.upper() in ("TWAP", "VWAP"):
                    try:
                        from services.twap_vwap_executor import TWAPExecutor
                        _twap = TWAPExecutor(self.kite)
                        resp = _twap.execute(
                            symbol=plan.symbol,
                            exchange=exchange,
                            side=plan.side,
                            total_quantity=plan.quantity,
                            price=plan.entry_price,
                            product=product,
                            algo=_algo.upper(),
                        )
                        logger.info("TWAP execution for %s: %s", plan.symbol, resp)
                    except Exception as twap_exc:
                        logger.warning("TWAP execution failed for %s, falling back to direct: %s", plan.symbol, twap_exc)
                        _algo = None  # Fall through to direct order below

                if not _algo or _algo.upper() not in ("TWAP", "VWAP"):
                    # T5-5: Retry with exponential backoff (max 3 attempts)
                    resp = None
                    for _attempt in range(3):
                        try:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                                future = pool.submit(place_order, **order_kwargs)
                                resp = future.result(timeout=_ORDER_TIMEOUT_S)
                            if resp and resp.get("success"):
                                break  # Success — exit retry loop
                            if resp and "insufficient" in str(resp.get("error", "")).lower():
                                break  # Don't retry margin errors
                        except concurrent.futures.TimeoutError:
                            resp = {"success": False, "error": f"Order timed out after {_ORDER_TIMEOUT_S}s"}
                            logger.error("Order timeout for %s (attempt %d/3)", plan.symbol, _attempt + 1)
                        except Exception as order_exc:
                            resp = {"success": False, "error": str(order_exc)}
                            logger.error("Order error for %s (attempt %d/3): %s", plan.symbol, _attempt + 1, order_exc)
                        if _attempt < 2:
                            _backoff = (2 ** _attempt) * 0.5  # 0.5s, 1.0s
                            time.sleep(_backoff)
                    if resp is None:
                        resp = {"success": False, "error": "All 3 order attempts failed"}
                time.sleep(_ORDER_DELAY_S)

                result = OrderResult(
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=plan.quantity,
                    entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss,
                    target_price=plan.target_price,
                    order_id=resp.get("order_id"),
                    success=resp.get("success", False),
                    error=resp.get("error"),
                )

            # Register BUY orders with TradeMonitor â€” SL/TP will be placed
            # AFTER the entry order fills (polled by TradeMonitor).
            # SELL exits don't need SL/TP monitoring.
            if result.success and result.order_id:
                if plan.side in ("BUY", "SELL"):
                    monitor.register_trade(MonitoredTrade(
                        symbol=plan.symbol,
                        side=plan.side,
                        quantity=plan.quantity,
                        entry_price=plan.entry_price,
                        stop_loss=plan.stop_loss,
                        target_price=plan.target_price,
                        entry_order_id=result.order_id,
                        direction=getattr(plan, "direction", "LONG"),
                        product=getattr(plan, "product", "CNC"),
                    ))
                existing_symbols.add(plan.symbol)

                # â”€â”€ Persist to trade journal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                try:
                    self._journal_entry(plan, result)
                except Exception as jexc:
                    logger.debug("Trade journal write failed: %s", jexc)

                # Desktop notification on successful order
                self._notify_order(plan.symbol, plan.side, plan.quantity,
                                   plan.entry_price, result.order_id)
            elif not result.success:
                self._notify_order_failure(plan.symbol, plan.side,
                                           result.error or "Unknown error")

            results.append(result)

        # Store monitor reference for lifecycle management
        self._trade_monitor = monitor

        # ── P0 fix: Live capital rolling — update VolatilityTarget with real P&L ──
        if self._vol_target and results:
            self._update_capital_from_kite(_cb)

        return results

    # ── P0 fix: Live capital rolling from Kite portfolio ──────────────

    def _update_capital_from_kite(self, _cb=None):
        """Fetch real P&L from Kite positions and update VolatilityTarget.

        This closes the capital-rolling gap: position sizes now adapt
        to actual wins/losses instead of using stale initial capital.
        """
        if not self._vol_target or self.kite is None:
            return
        try:
            positions = self.kite.positions()
            net_positions = positions.get("net", [])

            realized_pnl = 0.0
            unrealized_pnl = 0.0
            for pos in net_positions:
                realized_pnl += float(pos.get("realised", 0))
                unrealized_pnl += float(pos.get("unrealised", 0))

            # Also aggregate closed-trade P&L from holdings
            try:
                from kite_connect.trading.order_service import get_holdings
                holdings = get_holdings(self.kite)
                for h in holdings:
                    pnl = float(h.get("pnl", 0))
                    if pnl != 0:
                        realized_pnl += pnl
            except Exception:
                pass

            self._vol_target.update_pnl(realized=realized_pnl, unrealized=unrealized_pnl)

            # Persist state for crash recovery
            self._persist_portfolio_state(realized_pnl)

            if _cb:
                _cb(f"Capital rolling: realized={realized_pnl:+,.0f}, "
                    f"unrealized={unrealized_pnl:+,.0f}, "
                    f"capital={self._vol_target.current_capital:,.0f}")
            logger.info(
                "Capital rolling updated: realized=%.0f, unrealized=%.0f, capital=%.0f",
                realized_pnl, unrealized_pnl, self._vol_target.current_capital,
            )
        except Exception as exc:
            logger.warning("Capital rolling update failed (non-fatal): %s", exc)

    def _persist_portfolio_state(self, realized_pnl: float):
        """Save portfolio state to disk for crash recovery."""
        try:
            import json as _json
            import os as _os

            state_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                "data", "portfolio_state.json",
            )
            _os.makedirs(_os.path.dirname(state_path), exist_ok=True)

            peak = max(
                getattr(self, '_peak_equity', self._vol_target.current_capital),
                self._vol_target.current_capital,
            )
            self._peak_equity = peak

            state = {
                "cumulative_realized_pnl": realized_pnl,
                "peak_equity": peak,
                "current_capital": self._vol_target.current_capital,
                "updated_at": datetime.now().isoformat(),
            }
            with open(state_path, "w") as f:
                _json.dump(state, f, indent=2)
        except Exception as exc:
            logger.debug("Portfolio state persistence failed: %s", exc)

    # â”€â”€ Trade journal persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _journal_entry(self, plan, result):
        """Write a trade journal row for a successfully placed order."""
        from datetime import datetime
        try:
            from database.service import get_database_service
            db = get_database_service()
            if not db:
                return
            from database.models import TradeJournal
            regime = None
            try:
                from services.regime_detector import detect_regime
                regime = detect_regime().name
            except Exception:
                pass

            entry = TradeJournal(
                symbol=plan.symbol,
                exchange="NSE",
                side=plan.side,
                trade_type="CNC",
                entry_price=plan.entry_price,
                entry_date=datetime.now(),
                quantity=plan.quantity,
                strategy_name=getattr(plan, "strategy_name", None),
                decision_score=getattr(plan, "score", None),
                screener_score=getattr(plan, "screener_score", None),
                regime_at_entry=regime,
                planned_sl=plan.stop_loss,
                planned_tp=plan.target_price,
                entry_order_id=result.order_id,
                mode="live",
                is_open=True,
            )
            session = db.Session()
            try:
                session.add(entry)
                session.commit()
            finally:
                session.close()
        except Exception as exc:
            logger.debug("Journal persistence skipped: %s", exc)

    # â”€â”€ Notification helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _notify_order(symbol: str, side: str, qty: int, price: float, order_id: str):
        try:
            from services.notifications.manager import NotificationManager
            NotificationManager().notify_order_placed(symbol, side, qty, price, order_id)
        except Exception:
            pass

    @staticmethod
    def _notify_order_failure(symbol: str, side: str, error: str):
        try:
            from services.notifications.manager import NotificationManager
            NotificationManager().notify_order_failed(symbol, side, error)
        except Exception:
            pass

    # â”€â”€ SELL pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run_sell_pipeline(
        self,
        sell_verdicts: list,
        progress_callback=None,
    ) -> List[OrderResult]:
        """Automated exit: match SELL/STRONG_SELL verdicts against holdings.

        G2 FIX: SELL signals are value-destructive (39% hit rate, -0.82
        Sharpe at 20D).  Only execute SELL exits when confidence >= 0.7
        or the verdict is STRONG_SELL.  Low-confidence SELL verdicts are
        demoted and skipped.

        Parameters
        ----------
        sell_verdicts : list[StockVerdict]
            Verdicts with classification SELL or STRONG_SELL.
        progress_callback : callable | None
            Progress reporter.

        Returns
        -------
        list[OrderResult]
            Results for each SELL order placed (or skipped).
        """
        _cb = progress_callback or (lambda m: None)

        if self.kite is None:
            _cb("Kite not authenticated -- cannot place SELL orders")
            return []

        # G2 FIX: Gate SELL verdicts -- only allow high-confidence exits
        # SELL signals have 39% hit rate and -0.82 Sharpe at 20D horizon.
        # Only STRONG_SELL (extreme conviction) or high-confidence SELL pass.
        qualified_sells = []
        skipped_sells = []
        for v in sell_verdicts:
            confidence = getattr(v, "confidence", 0.0)
            classification = getattr(v, "classification", "SELL")
            if classification == "STRONG_SELL" or confidence >= 0.7:
                qualified_sells.append(v)
            else:
                skipped_sells.append(v)

        if skipped_sells:
            skipped_tickers = [getattr(v, "ticker", "?") for v in skipped_sells]
            _cb(
                f"G2: Filtered {len(skipped_sells)} low-confidence SELL signals "
                f"(conf < 0.7): {skipped_tickers}"
            )
            logger.info(
                "G2: Skipped %d low-confidence SELL signals: %s",
                len(skipped_sells), skipped_tickers,
            )

        if not qualified_sells:
            _cb("No SELL verdicts passed G2 confidence gate -- no exits")
            return []

        # Fetch current holdings
        from kite_connect.trading.order_service import get_holdings
        holdings = get_holdings(self.kite)
        if not holdings:
            _cb("No holdings found -- nothing to exit")
            return []

        sell_syms = [
            v.ticker.replace(".NS", "").replace(".BO", "")
            for v in qualified_sells
        ]

        # Build SELL plans
        plans = self.risk_mgr.plan_exits(sell_syms, holdings)
        if not plans:
            _cb("No SELL verdicts match current holdings")
            return []

        _cb(f"Placing {len(plans)} SELL orders ...")
        return self._place_orders(plans, _cb)
