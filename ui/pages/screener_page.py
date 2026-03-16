"""
NSE Screener & Auto-Trade Page for Indian Stocks.

Streamlit page that:
1. Downloads the full NSE equity universe
2. Screens stocks through three stages (criteria → methodology → technicals)
3. Displays ranked results with trade plans
4. Optionally places orders via Kite Connect
"""

import logging
from typing import List

import pandas as pd
import streamlit as st

from ui.components import (
    render_header,
    render_footer,
    render_ind_navigation_buttons,
)

logger = logging.getLogger(__name__)


def render_screener_page():
    """Entry point for the NSE Screener & Auto-Trade page."""
    render_header()
    render_ind_navigation_buttons(
        current_page="screener",
        back_key_suffix="from_screener",
    )

    st.subheader("NSE Stock Screener")
    st.caption(
        "Full NSE universe → Price · Liquidity · Trend · Volatility filters → "
        "Pullback / Breakout / Sector strategies → RSI, Bollinger, S/R → "
        "Risk-managed trade plans → Kite order placement"
    )

    # ── Sidebar-style config in expanders ──────────────────────
    screen_cfg, risk_cfg, auto_place = _render_config()

    # ── Run button ─────────────────────────────────────────────
    label = "Place Order (s)" if auto_place else "Screen"
    run_clicked = st.button(label, type="primary", key="screener_run")

    # ── Execution ──────────────────────────────────────────────
    if run_clicked:
        _run_pipeline(screen_cfg, risk_cfg, auto_place)

    # ── Show cached results if available ───────────────────────
    _show_cached_results()

    render_footer()


# ═══════════════════════════════════════════════════════════════
# Configuration panel
# ═══════════════════════════════════════════════════════════════

def _render_config():
    from kite_connect.nse.screener import ScreenerConfig
    from kite_connect.trading.risk_manager import RiskConfig

    scfg = ScreenerConfig()
    rcfg = RiskConfig()

    col_screen, col_method, col_risk = st.columns(3)

    with col_screen:
        with st.expander("Screening Criteria", expanded=False):
            scfg.min_price = st.number_input(
                "Min Price (₹)", value=100.0, step=10.0, key="scr_price"
            )
            scfg.min_avg_volume = st.number_input(
                "Min Avg Volume", value=500_000, step=50_000, key="scr_vol"
            )
            scfg.min_beta = st.number_input(
                "Min Beta", value=1.0, step=0.1, format="%.1f", key="scr_beta"
            )
            scfg.max_workers = st.number_input(
                "Workers", value=8, min_value=1, max_value=16, key="scr_workers"
            )

    with col_method:
        with st.expander("Methodology Settings", expanded=False):
            scfg.pullback_pct = st.slider(
                "Pullback tolerance %", 1, 5, 2, key="scr_pb"
            ) / 100
            scfg.breakout_vol_mult = st.number_input(
                "Breakout volume multiplier", value=1.5, step=0.1,
                format="%.1f", key="scr_bv"
            )
            scfg.breakout_lookback = st.number_input(
                "Breakout lookback days", value=20, step=5, key="scr_bl"
            )

    with col_risk:
        with st.expander("Risk Management", expanded=False):
            rcfg.total_capital = st.number_input(
                "Total Capital (₹)", value=500_000.0, step=50_000.0,
                key="risk_cap"
            )
            rcfg.risk_per_trade_pct = st.slider(
                "Risk per trade %", 1, 5, 2, key="risk_pct"
            ) / 100
            rcfg.max_open_trades = st.number_input(
                "Max open trades", value=10, min_value=1, max_value=50,
                key="risk_max"
            )
            rcfg.min_rr_ratio = st.number_input(
                "Min R:R ratio", value=2.0, step=0.5, format="%.1f",
                key="risk_rr"
            )
            rcfg.sl_method = st.selectbox(
                "Stop-Loss method",
                ["tighter", "ma50", "swing_low"],
                key="risk_sl",
            )

    auto_place = st.checkbox(
        "Enable to place live orders (requires Kite auth)",
        value=False,
        key="screener_auto_place",
    )

    return scfg, rcfg, auto_place


# ═══════════════════════════════════════════════════════════════
# Pipeline execution
# ═══════════════════════════════════════════════════════════════

def _run_pipeline(screen_cfg, risk_cfg, auto_place: bool):
    from kite_connect.trading.auto_executor import AutoExecutor

    kite = st.session_state.get("kite")

    executor = AutoExecutor(
        kite=kite,
        screener_cfg=screen_cfg,
        risk_cfg=risk_cfg,
        auto_place=auto_place,
    )

    progress_area = st.empty()
    status_messages: List[str] = []

    def _progress(msg: str):
        status_messages.append(msg)
        progress_area.info("\n\n".join(status_messages[-5:]))

    with st.spinner("Running NSE screener pipeline …"):
        report = executor.run(progress_callback=_progress)

    progress_area.empty()

    # Persist to session for re-display
    st.session_state["screener_report"] = report
    st.session_state["screener_screened_df"] = report.screened_df
    st.session_state["screener_plans"] = report.trade_plans
    st.session_state["screener_orders"] = report.order_results

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Universe", report.universe_size)
    m2.metric("Passed Screen", report.screened_count)
    m3.metric("Trade Plans", report.plans_count)
    m4.metric("Orders Placed", report.orders_placed)

    st.success("Pipeline complete — see results below")


# ═══════════════════════════════════════════════════════════════
# Display results (from cache)
# ═══════════════════════════════════════════════════════════════

def _show_cached_results():
    screened_df = st.session_state.get("screener_screened_df")
    plans = st.session_state.get("screener_plans")
    orders = st.session_state.get("screener_orders")

    if screened_df is not None and not screened_df.empty:
        st.markdown("---")
        st.subheader("Screened Stocks (ranked by score)")

        # Colour-code score column
        def _highlight_score(val):
            if val >= 70:
                return "background-color: #22c55e22"
            elif val >= 50:
                return "background-color: #eab30822"
            return ""

        display_cols = [
            "symbol", "close", "score", "beta", "avg_volume",
            "ma_20", "ma_50", "ma_200",
            "rsi", "bb_upper", "bb_lower",
            "support", "resistance",
            "pullback", "breakout", "sector_leader", "sector_name",
            "strategies",
        ]
        cols_present = [c for c in display_cols if c in screened_df.columns]
        styled = screened_df[cols_present].style.map(
            _highlight_score, subset=["score"] if "score" in cols_present else []
        )
        st.dataframe(styled, width='stretch', height=400)

        # Allow CSV download
        csv = screened_df[cols_present].to_csv(index=False)
        st.download_button(
            "Download screened stocks CSV",
            csv,
            file_name="nse_screened_stocks.csv",
            mime="text/csv",
            key="dl_screened",
        )

        # Also push qualifying tickers into session for normal analysis flow
        qualifying = screened_df["symbol"].tolist()
        ns_tickers = [f"{s}.NS" for s in qualifying]
        st.session_state["screened_tickers_ns"] = ns_tickers

    if plans:
        st.markdown("---")
        st.subheader("Trade Plans (risk-managed)")
        plan_rows = [p.to_dict() for p in plans]
        plan_df = pd.DataFrame(plan_rows)
        st.dataframe(plan_df, width='stretch')

        st.download_button(
            "Download trade plans CSV",
            plan_df.to_csv(index=False),
            file_name="nse_trade_plans.csv",
            mime="text/csv",
            key="dl_plans",
        )

    if orders:
        st.markdown("---")
        st.subheader("Order Execution Results")
        order_rows = [o.to_dict() for o in orders]
        order_df = pd.DataFrame(order_rows)
        st.dataframe(order_df, width='stretch')

    # Buttons to continue with screened stocks
    if st.session_state.get("screened_tickers_ns"):
        st.markdown("---")
        btn_a, btn_v, _ = st.columns([1, 1, 2])
        with btn_a:
            if st.button(
                "Analyze",
                type="secondary",
                key="analyse_screened",
            ):
                tickers = st.session_state["screened_tickers_ns"]
                st.session_state.tickers = tickers
                st.session_state.analysis_tickers = tickers
                st.session_state.analysis_complete = False
                st.session_state.signals = []
                st.session_state.progress_messages = []
                st.session_state.analysis_run_id = (
                    st.session_state.get("analysis_run_id", 0) + 1
                )
                st.session_state.current_page = "main"
                st.rerun()
        with btn_v:
            if st.button(
                "Verdict",
                type="primary",
                key="verdict_screened",
            ):
                tickers = st.session_state["screened_tickers_ns"]
                st.session_state["verdict_from_screener"] = tickers
                st.session_state.current_page = "verdict"
                st.rerun()
