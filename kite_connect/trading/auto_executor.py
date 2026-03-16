"""
Auto-Order Execution Engine for Zerodha Kite Connect.

Orchestrates the full pipeline:

1. Download NSE universe  →  :mod:`kite_connect.nse.nse_universe`
2. Screen & rank          →  :mod:`kite_connect.nse.screener`
3. Risk-manage & size     →  :mod:`kite_connect.trading.risk_manager`
4. Place orders via Kite  →  :mod:`kite_connect.trading.order_service`
5. Register with monitor  →  :mod:`kite_connect.trading.trade_monitor`

Signal→Executor bridge: accepts analysis verdicts to filter
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


# ═══════════════════════════════════════════════════════════════
# Execution result
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Executor
# ═══════════════════════════════════════════════════════════════

class AutoExecutor:
    """
    End-to-end execution engine: screen → signal-filter → risk-check → order.

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
    ):
        self.kite = kite
        self.screener = NSEScreener(screener_cfg)
        self.risk_mgr = RiskManager(risk_cfg, kite=kite)
        self.auto_place = auto_place

    # ── Public API ─────────────────────────────────────────────

    def run(
        self,
        symbols: Optional[List[str]] = None,
        progress_callback=None,
        signal_verdicts: Optional[Dict[str, str]] = None,
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

        Returns
        -------
        ExecutionReport
        """
        _cb = progress_callback or (lambda m: None)
        report = ExecutionReport(timestamp=datetime.now().isoformat())

        # ── 1.  Universe ───────────────────────────────────────
        if symbols is None:
            _cb("Downloading NIFTY50 & NSE NEXT50")
            symbols = get_nse_universe(self.kite)
        report.universe_size = len(symbols)
        _cb(f"Universe: {len(symbols)} symbols")

        # ── 2.  Screen ─────────────────────────────────────────
        screened_df = self.screener.screen(symbols, progress_callback=_cb)
        report.screened_df = screened_df
        report.screened_count = len(screened_df)

        if screened_df.empty:
            _cb("No stocks passed screening criteria")
            return report

        # ── 2b. Signal→Executor bridge: strict BUY-only filter ─
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
                _cb("No stocks have BUY/STRONG_BUY signal — skipping execution")
                return report

        # ── 3.  Risk management / trade plans ──────────────────
        _cb("Generating trade plans with risk management …")
        plans = self.risk_mgr.plan_trades(screened_df)
        report.trade_plans = plans
        report.plans_count = len(plans)

        if not plans:
            _cb("No trade plans met the R:R threshold")
            return report

        # ── 4.  Order placement ────────────────────────────────
        if self.auto_place and self.kite is not None:
            _cb(f"Placing {len(plans)} orders via Kite …")
            report.order_results = self._place_orders(plans, _cb)
            report.orders_placed = sum(1 for r in report.order_results if r.success)
            report.orders_failed = sum(1 for r in report.order_results if not r.success)
            _cb(
                f"Orders: {report.orders_placed} placed, "
                f"{report.orders_failed} failed"
            )
        else:
            _cb(
                f"Dry run — {len(plans)} plans generated "
                "(auto_place=False or no Kite session)"
            )

        return report

    # ── Order placement ────────────────────────────────────────

    def _place_orders(
        self, plans: List[TradePlan], _cb
    ) -> List[OrderResult]:
        from kite_connect.trading.order_service import place_order, get_order_book
        from kite_connect.trading.trade_monitor import TradeMonitor, MonitoredTrade

        results: List[OrderResult] = []

        # ── Duplicate check: skip symbols with open BUY orders ─────
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

        # ── Monitor for post-trade SL/TP lifecycle ─────────────────
        monitor = TradeMonitor(self.kite)

        for plan in plans:
            # Skip if order already exists for this symbol
            if plan.symbol in existing_symbols:
                _cb(f"  Skipped {plan.symbol} — open order already exists")
                results.append(OrderResult(
                    symbol=plan.symbol, side=plan.side,
                    quantity=plan.quantity, entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss, target_price=plan.target_price,
                    success=False, error="Duplicate — open order exists",
                ))
                continue

            _cb(f"  Placing {plan.side} {plan.symbol} × {plan.quantity} …")

            # Entry order (LIMIT at entry price)
            resp = place_order(
                kite=self.kite,
                symbol=plan.symbol,
                exchange="NSE",
                transaction_type=plan.side,
                quantity=plan.quantity,
                order_type="LIMIT",
                product="CNC",
                price=plan.entry_price,
            )
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

            # Register with TradeMonitor — SL/TP will be placed
            # AFTER the entry order fills (polled by TradeMonitor).
            if result.success and result.order_id:
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

            results.append(result)

        # Store monitor in session state for lifecycle management
        try:
            import streamlit as st
            st.session_state["trade_monitor"] = monitor
        except Exception:
            pass  # non-Streamlit context (e.g. scheduled job)

        return results
