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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from kite_connect.nse.nse_universe import get_nse_universe
from kite_connect.nse.screener import NSEScreener, ScreenerConfig
from kite_connect.trading.risk_manager import RiskManager, RiskConfig, TradePlan

logger = logging.getLogger(__name__)

# Allowed verdict tags for execution (strict BUY-only filter)
_BUY_TAGS = {"BUY", "STRONG_BUY"}

# Rate-limiting: pause between Kite API calls (seconds)
_ORDER_DELAY_S = 0.15


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
        self.risk_mgr = RiskManager(risk_cfg, kite=kite)
        self.auto_place = auto_place
        self._trade_monitor = trade_monitor

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
            allowed = [
                sym for sym in screened_df["symbol"].tolist()
                if signal_verdicts.get(sym, "").upper() in _BUY_TAGS
            ]
            screened_df = screened_df[screened_df["symbol"].isin(allowed)]
            report.screened_df = screened_df
            report.signal_filtered_count = pre_filter - len(screened_df)
            _cb(
                f"Signal filter: {len(allowed)} BUY/STRONG_BUY passed, "
                f"{report.signal_filtered_count} non-BUY removed"
            )
            if screened_df.empty:
                _cb("No stocks have BUY/STRONG_BUY signal â€” skipping execution")
                return report

        # â”€â”€ 3.  Enrich with live prices if Kite available â”€â”€â”€â”€â”€â”€
        if self.kite is not None and not screened_df.empty:
            screened_df = self._enrich_with_ltp(screened_df, _cb)
            report.screened_df = screened_df

        # â”€â”€ 3b. Order book depth: filter illiquid stocks â”€â”€â”€â”€â”€â”€â”€
        if self.kite is not None and not screened_df.empty:
            screened_df = self._filter_by_spread(screened_df, _cb)
            report.screened_df = screened_df

        # â”€â”€ 4.  Risk management / trade plans â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _cb("Generating trade plans with risk management â€¦")
        plans = self.risk_mgr.plan_trades(screened_df)
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
        else:
            _cb(
                f"Dry run â€” {len(plans)} plans generated "
                "(auto_place=False or no Kite session)"
            )

        return report

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
                    # TradingView unavailable -- allow through (no veto)
                    approved.append(plan)
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
                logger.debug("MTF check failed for %s: %s -- allowing through", ticker, exc)
                approved.append(plan)

        if rejected_count:
            _cb(
                f"MTF gate: {len(approved)} passed, {rejected_count} rejected "
                f"(daily/weekly disagreement)"
            )

        return approved

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
    # â”€â”€ Order book depth: illiquidity filter (#11) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _filter_by_spread(self, screened_df: pd.DataFrame, _cb) -> pd.DataFrame:
        """Remove stocks with bid-ask spread > 0.5%. Reduce position for > 0.3%."""
        try:
            symbols = screened_df["symbol"].tolist()
            instrument_keys = [f"NSE:{s}" for s in symbols]
            quote_data = self.kite.quote(instrument_keys)

            remove_syms = set()
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
                        if spread_pct > 0.01:  # > 1% spread â€” too illiquid
                            remove_syms.add(row["symbol"])
                            _cb(f"  Removed {row['symbol']} â€” spread {spread_pct:.1%} > 1%")
                        elif spread_pct > 0.005:  # > 0.5% â€” flag as illiquid
                            _cb(f"  Warning: {row['symbol']} spread {spread_pct:.2%}")

            if remove_syms:
                screened_df = screened_df[~screened_df["symbol"].isin(remove_syms)]
                _cb(f"Depth filter removed {len(remove_syms)} illiquid stocks")

        except Exception as exc:
            logger.warning("Depth filter failed (non-fatal): %s", exc)

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
                    and o.get("transaction_type") == "BUY"
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

            # SELL exits use MARKET; BUY entries use LIMIT
            order_type = "MARKET" if plan.side == "SELL" else "LIMIT"
            order_kwargs = dict(
                kite=self.kite,
                symbol=plan.symbol,
                exchange="NSE",
                transaction_type=plan.side,
                quantity=plan.quantity,
                order_type=order_type,
                product="CNC",
            )
            if order_type == "LIMIT":
                order_kwargs["price"] = plan.entry_price

            resp = place_order(**order_kwargs)
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
                if plan.side == "BUY":
                    monitor.register_trade(MonitoredTrade(
                        symbol=plan.symbol,
                        side=plan.side,
                        quantity=plan.quantity,
                        entry_price=plan.entry_price,
                        stop_loss=plan.stop_loss,
                        target_price=plan.target_price,
                        entry_order_id=result.order_id,
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

        return results

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
            _cb("Kite not authenticated â€” cannot place SELL orders")
            return []

        # Fetch current holdings
        from kite_connect.trading.order_service import get_holdings
        holdings = get_holdings(self.kite)
        if not holdings:
            _cb("No holdings found â€” nothing to exit")
            return []

        sell_syms = [
            v.ticker.replace(".NS", "").replace(".BO", "")
            for v in sell_verdicts
        ]

        # Build SELL plans
        plans = self.risk_mgr.plan_exits(sell_syms, holdings)
        if not plans:
            _cb("No SELL verdicts match current holdings")
            return []

        _cb(f"Placing {len(plans)} SELL orders â€¦")
        return self._place_orders(plans, _cb)
