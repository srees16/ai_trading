"""
NSE Screener & Auto-Trade Page for Indian Stocks.

Streamlit page that:
1. Downloads the full NSE equity universe
2. Screens stocks through three stages (criteria → methodology → technicals)
3. Runs IntegratedScorer verdicts on screened stocks
4. Displays ranked results with trade plans
5. Optionally places orders via Kite Connect (with confirmation)
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

# Session-state prefix for verdict controls
_PFX = "verdict_"

# Colour map for verdict classification
_COLOUR_MAP = {
    "STRONG_BUY": "#00c853",
    "BUY": "#66bb6a",
    "HOLD": "#ffa726",
    "SELL": "#ef5350",
    "STRONG_SELL": "#b71c1c",
}


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
    run_clicked = st.button("Screen", type="primary", key="screener_run")

    # ── Execution ──────────────────────────────────────────────
    if run_clicked:
        _run_pipeline(screen_cfg, risk_cfg)

    # ── Show cached results if available ───────────────────────
    _show_cached_results(risk_cfg, auto_place)

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

    # Auth status warning (shown before checkbox so user knows upfront)
    if st.session_state.get("kite") is None:
        st.warning(
            "Kite session not authenticated - orders will be dry-run only. "
            "Visit **Fly Kite** to authenticate.",
            icon="⚠️",
        )

    auto_place = st.checkbox(
        "Enable to place live orders (requires Kite auth)",
        value=False,
        key="screener_auto_place",
    )

    return scfg, rcfg, auto_place


# ═══════════════════════════════════════════════════════════════
# Pipeline execution (screening only — no order placement here)
# ═══════════════════════════════════════════════════════════════

def _run_pipeline(screen_cfg, risk_cfg):
    from kite_connect.trading.auto_executor import AutoExecutor

    kite = st.session_state.get("kite")

    executor = AutoExecutor(
        kite=kite,
        screener_cfg=screen_cfg,
        risk_cfg=risk_cfg,
        auto_place=False,  # Never place orders at screening stage
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
    # Clear stale verdict results from previous runs
    st.session_state.pop(f"{_PFX}results", None)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Universe", report.universe_size)
    m2.metric("Passed Screen", report.screened_count)
    m3.metric("Trade Plans", report.plans_count)
    m4.metric("Orders Placed", report.orders_placed)

    st.success("Screening complete — see results below")


# ═══════════════════════════════════════════════════════════════
# Display results (from cache)
# ═══════════════════════════════════════════════════════════════

def _show_cached_results(risk_cfg, auto_place: bool):
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

        # Push qualifying tickers into session for analysis flow
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

    # ── Post-screen actions: Verdict + Order Placement ─────────
    if st.session_state.get("screened_tickers_ns"):
        st.markdown("---")
        _render_verdict_section(risk_cfg, auto_place)


# ═══════════════════════════════════════════════════════════════
# Integrated Verdict (merged from verdict_page.py)
# ═══════════════════════════════════════════════════════════════

def _render_verdict_section(risk_cfg, auto_place: bool):
    """Render IntegratedScorer controls and results for screened stocks."""
    st.subheader("Integrated Verdict")
    st.caption("Run the 5-layer IntegratedScorer on screened stocks to get BUY/SELL verdicts.")

    ns_tickers = st.session_state["screened_tickers_ns"]

    # Verdict configuration
    col1, col2 = st.columns([2, 1])

    with col1:
        skip_options = st.multiselect(
            "Skip layers",
            options=["core", "strategy", "ml_features", "robustness", "rag"],
            default=["rag"],
            help="RAG is skipped by default (LLM calls add latency)",
            key=f"{_PFX}skip_layers",
        )

    with col2:
        batch_size = st.selectbox(
            "Batch size",
            options=[20, 40, 60, 80, 100],
            index=0,
            help="Stocks are analysed in parallel batches.",
            key=f"{_PFX}batch_size",
        )

    with st.expander("Layer weights", expanded=False):
        w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns(5)
        w_core = w_col1.slider("Core", 0, 100, 30, key=f"{_PFX}w_core")
        w_strat = w_col2.slider("Strategy", 0, 100, 25, key=f"{_PFX}w_strat")
        w_ml = w_col3.slider("ML Features", 0, 100, 15, key=f"{_PFX}w_ml")
        w_robust = w_col4.slider("Robustness", 0, 100, 20, key=f"{_PFX}w_robust")
        w_rag = w_col5.slider("RAG", 0, 100, 0, key=f"{_PFX}w_rag")

    if st.button("Run Verdict", type="primary", key=f"{_PFX}run_btn"):
        total_w = w_core + w_strat + w_ml + w_robust + w_rag
        if total_w == 0:
            total_w = 1
        weights = {
            "core": w_core / total_w,
            "strategy": w_strat / total_w,
            "ml_features": w_ml / total_w,
            "robustness": w_robust / total_w,
            "rag": w_rag / total_w,
        }
        _run_batched_verdict(ns_tickers, weights, skip_options, batch_size)
        st.rerun()

    # Show verdict results if available
    verdicts = st.session_state.get(f"{_PFX}results")
    if verdicts:
        _render_verdict_results(verdicts, risk_cfg, auto_place)


def _run_batched_verdict(tickers, weights, skip_layers, batch_size):
    """Evaluate tickers in concurrent batches via IntegratedScorer."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from services.integrated_scorer import IntegratedScorer
    from datetime import date, timedelta

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=365)
    date_range = (str(start_dt), str(end_dt))

    total = len(tickers)
    num_batches = (total + batch_size - 1) // batch_size

    batches = [
        tickers[i * batch_size : min((i + 1) * batch_size, total)]
        for i in range(num_batches)
    ]

    def _evaluate_batch(batch_tickers):
        scorer = IntegratedScorer(weights=weights)
        return scorer.evaluate(
            tickers=batch_tickers,
            market="IND",
            date_range=date_range,
            skip_layers=skip_layers,
        )

    max_concurrent = min(num_batches, 3)
    all_verdicts = []

    with st.spinner(f"Analysing {total} stocks in {num_batches} parallel batches …"):
        with ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="batch") as pool:
            futures = {
                pool.submit(_evaluate_batch, batch): idx
                for idx, batch in enumerate(batches)
            }
            results_by_idx = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_by_idx[idx] = future.result()
                except Exception as exc:
                    logger.error("Batch %d failed: %s", idx, exc)
                    results_by_idx[idx] = []

        for idx in range(num_batches):
            all_verdicts.extend(results_by_idx.get(idx, []))

    st.session_state[f"{_PFX}results"] = all_verdicts


def _render_verdict_results(verdicts, risk_cfg, auto_place: bool):
    """Render verdict table, per-ticker breakdown, and order placement."""
    st.markdown("---")
    st.subheader("Verdicts")

    rows = []
    for v in verdicts:
        rows.append({
            "Ticker": v.ticker,
            "Score": f"{v.final_score:+.2f}",
            "Verdict": v.classification,
            "Confidence": f"{v.confidence:.0%}",
            "Core": _fmt_score(v.layer_scores.get("core")),
            "Strategy": _fmt_score(v.layer_scores.get("strategy")),
            "ML": _fmt_score(v.layer_scores.get("ml_features")),
            "Robustness": _fmt_score(v.layer_scores.get("robustness")),
            "RAG": _fmt_score(v.layer_scores.get("rag")),
        })

    df = pd.DataFrame(rows)

    def _colour_verdict(row):
        colour = _COLOUR_MAP.get(row["Verdict"], "#888")
        return [
            f"color: {colour}; font-weight: bold" if col == "Verdict" else ""
            for col in row.index
        ]

    styled = df.style.apply(_colour_verdict, axis=1)
    st.dataframe(styled, hide_index=True, width="stretch")

    # ── BUY signal execution section ─────────────────────────
    buy_verdicts = [
        v for v in verdicts
        if v.classification in ("BUY", "STRONG_BUY")
    ]
    if buy_verdicts:
        st.markdown("---")
        buy_syms = [v.ticker.replace(".NS", "").replace(".BO", "") for v in buy_verdicts]
        st.success(f"**{len(buy_verdicts)} BUY signals**: {', '.join(buy_syms)}")

        kite = st.session_state.get("kite")
        if kite is None:
            st.info("Kite session not authenticated — orders will be dry-run only. "
                    "Visit **Fly Kite** to authenticate.", icon="ℹ️")

        # Two-step confirmation for order placement
        if auto_place and kite is not None:
            st.warning(
                f"**{len(buy_verdicts)} live orders** will be placed on Zerodha. "
                "This uses real money. Review the trade plans above before proceeding.",
                icon="⚠️",
            )
            confirm = st.checkbox(
                f"I confirm: place {len(buy_verdicts)} BUY orders via Kite",
                value=False,
                key="confirm_place_orders",
            )
            if confirm:
                if st.button(
                    f"Place {len(buy_verdicts)} Orders",
                    type="primary",
                    key="execute_orders_btn",
                ):
                    _execute_buy_verdicts(verdicts, risk_cfg)
        else:
            if st.button(
                f"Dry-Run {len(buy_verdicts)} Orders",
                type="secondary",
                key="dryrun_orders_btn",
            ):
                _execute_buy_verdicts(verdicts, risk_cfg)

    # Per-ticker expandable breakdown
    for v in verdicts:
        with st.expander(
            f"**{v.ticker}** — "
            f":{v.classification.replace('_', ' ')}: "
            f"({v.final_score:+.2f})",
        ):
            radar_bytes = _build_radar(v)
            if radar_bytes:
                st.image(radar_bytes, width=400)

            for layer_name in ("core", "strategy", "ml_features", "robustness", "rag"):
                details = v.layer_details.get(layer_name, {})
                score = v.layer_scores.get(layer_name)
                header = f"**{layer_name.replace('_', ' ').title()}**"
                if score is not None:
                    header += f"  →  {score:+.4f}"
                else:
                    header += "  →  *skipped*"
                st.markdown(header)

                if details:
                    display = {
                        k: val for k, val in details.items()
                        if k != "per_strategy"
                    }
                    if display:
                        st.json(display)
                    per_strat = details.get("per_strategy")
                    if per_strat:
                        with st.expander("Per-strategy breakdown", expanded=False):
                            st.json(per_strat)

                st.markdown("---")


def _execute_buy_verdicts(verdicts, risk_cfg):
    """Place orders for BUY/STRONG_BUY verdicts via AutoExecutor."""
    kite = st.session_state.get("kite")
    signal_dict = {
        v.ticker.replace(".NS", "").replace(".BO", ""): v.classification
        for v in verdicts
    }
    buy_symbols = [
        sym for sym, tag in signal_dict.items()
        if tag in ("BUY", "STRONG_BUY")
    ]

    if not buy_symbols:
        st.warning("No BUY/STRONG_BUY signals to execute.")
        return

    try:
        from kite_connect.trading.auto_executor import AutoExecutor

        auto_place = kite is not None
        executor = AutoExecutor(
            kite=kite,
            risk_cfg=risk_cfg,
            auto_place=auto_place,
        )
        with st.spinner(f"Executing orders for {len(buy_symbols)} symbols…"):
            report = executor.run(
                symbols=buy_symbols,
                signal_verdicts=signal_dict,
            )
        st.success(
            f"Execution complete — "
            f"{report.orders_placed} orders placed, "
            f"{report.orders_failed} failed, "
            f"{report.signal_filtered_count} filtered by signal."
        )
        if report.order_results:
            order_rows = [o.to_dict() for o in report.order_results]
            st.dataframe(pd.DataFrame(order_rows), hide_index=True)
    except Exception as exc:
        logger.exception("Order execution failed")
        st.error(f"Order execution error: {exc}")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _fmt_score(val):
    if val is None:
        return "—"
    return f"{val:+.3f}"


def _build_radar(verdict) -> bytes | None:
    """Build a small radar chart for the verdict."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        labels = list(verdict.layer_scores.keys())
        values = [verdict.layer_scores.get(lbl) or 0 for lbl in labels]
        n = len(labels)
        if n < 3:
            return None

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25, color="steelblue")
        ax.plot(angles, values, color="steelblue", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([lbl.replace("_", "\n") for lbl in labels], fontsize=8)
        ax.set_ylim(-1, 1)
        ax.set_title(
            f"{verdict.ticker}  {verdict.classification}  ({verdict.final_score:+.2f})",
            fontsize=10, pad=15,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None
