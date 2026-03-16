"""
Integrated Verdict Page — Multi-Layer Stock Evaluation Dashboard.

Renders the IntegratedScorer output with colour-coded verdicts,
per-layer breakdowns, and a radar chart for each ticker.
"""

import logging
from datetime import date, timedelta

import streamlit as st

logger = logging.getLogger(__name__)

# Session-state key prefix
_PFX = "verdict_"


def render_verdict_page():
    """Main entry point called from app.py routing."""
    from ui.components import (
        render_page_header,
        render_footer,
        render_navigation_buttons,
        render_ind_navigation_buttons,
        render_stock_ticker_ribbon,
        render_vix_indicator,
    )

    market = st.session_state.get("current_market", "US")
    market_label = "Indian" if market == "IND" else "US"
    render_page_header(f"{market_label} Integrated Verdict")

    # Scrolling ribbon + VIX indicator
    render_stock_ticker_ribbon(market=market)
    render_vix_indicator(market=market)

    # Navigation buttons (same pattern as all other pages)
    if market == "IND":
        render_ind_navigation_buttons(current_page='verdict', back_key_suffix='from_verdict')
    else:
        render_navigation_buttons(current_page='verdict', back_key_suffix='from_verdict')

    st.markdown("---")

    _render_controls(market)

    if st.session_state.get(f"{_PFX}results"):
        _render_results(st.session_state[f"{_PFX}results"])

    render_footer()


# ─────────────────────────────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────────────────────────────

def _render_controls(market: str):
    # ── Default tickers based on market ──────────────────────
    # Priority: screener handoff → cached universe → fallback
    screener_tickers = st.session_state.pop("verdict_from_screener", None)
    if screener_tickers:
        default_tickers = screener_tickers
        st.info(
            f"Loaded {len(screener_tickers)} tickers from NSE Screener",
            icon="\u2705",
        )
    elif market == "IND":
        cached_key = f"{_PFX}nse_universe"
        if cached_key not in st.session_state:
            with st.spinner("Loading Nifty 50 + Nifty Next 50 …"):
                from kite_connect.nse.nse_universe import get_nse_default_tickers
                symbols = get_nse_default_tickers()
                # Store with .NS suffix for yfinance compatibility
                st.session_state[cached_key] = [
                    f"{s}.NS" for s in symbols
                ]
        default_tickers = st.session_state[cached_key]
    else:
        default_tickers = st.session_state.get(
            "tickers", ["AAPL", "MSFT", "GOOGL", "AMZN"],
        )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        tickers_input = st.text_input(
            f"Tickers — {len(default_tickers)} loaded (comma-separated)",
            value=", ".join(default_tickers),
            key=f"{_PFX}tickers_input",
        )

    with col2:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365)
        date_range_input = st.date_input(
            "Date range",
            value=(start_dt, end_dt),
            key=f"{_PFX}date_range",
        )

    with col3:
        skip_options = st.multiselect(
            "Skip layers",
            options=["core", "strategy", "ml_features", "robustness", "rag"],
            default=["rag"],
            help="RAG is skipped by default (LLM calls add latency with minimal scoring value)",
            key=f"{_PFX}skip_layers",
        )

    # Weight sliders in an expander
    with st.expander("Layer weights", expanded=False):
        w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns(5)
        w_core = w_col1.slider("Core", 0, 100, 30, key=f"{_PFX}w_core")
        w_strat = w_col2.slider("Strategy", 0, 100, 25, key=f"{_PFX}w_strat")
        w_ml = w_col3.slider("ML Features", 0, 100, 15, key=f"{_PFX}w_ml")
        w_robust = w_col4.slider("Robustness", 0, 100, 20, key=f"{_PFX}w_robust")
        w_rag = w_col5.slider("RAG", 0, 100, 0, key=f"{_PFX}w_rag")

    # Batch size control (relevant when ticker count is large)
    batch_size = 20
    if market == "IND":
        bs_col, _ = st.columns([1, 4])
        with bs_col:
            batch_size = st.selectbox(
                "Batch size",
                options=[20, 40, 60, 80, 100],
                index=0,
                help="Stocks are analysed in parallel batches to keep the app responsive.",
                key=f"{_PFX}batch_size",
            )

    if st.button("Run Integrated Analysis", type="primary", key=f"{_PFX}run_btn"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.warning("Enter at least one ticker.")
            return

        if isinstance(date_range_input, (list, tuple)) and len(date_range_input) == 2:
            dr = (str(date_range_input[0]), str(date_range_input[1]))
        else:
            dr = (str(start_dt), str(end_dt))

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

        _run_batched_analysis(tickers, market, dr, weights, skip_options, batch_size)
        st.rerun()


def _run_batched_analysis(tickers, market, date_range, weights, skip_layers, batch_size):
    """Evaluate tickers in concurrent batches."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from services.integrated_scorer import IntegratedScorer

    total = len(tickers)
    num_batches = (total + batch_size - 1) // batch_size

    # Split tickers into batches
    batches = [
        tickers[i * batch_size : min((i + 1) * batch_size, total)]
        for i in range(num_batches)
    ]

    def _evaluate_batch(batch_tickers):
        scorer = IntegratedScorer(weights=weights)
        return scorer.evaluate(
            tickers=batch_tickers,
            market=market,
            date_range=date_range,
            skip_layers=skip_layers,
        )

    # Run batches concurrently (cap workers to avoid overwhelming the system)
    max_concurrent = min(num_batches, 3)
    all_verdicts = []

    with st.spinner(f"Analysing {total} stocks in {num_batches} parallel batches …"):
        with ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="batch") as pool:
            futures = {
                pool.submit(_evaluate_batch, batch): idx
                for idx, batch in enumerate(batches)
            }
            # Collect results preserving original ordering
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


# ─────────────────────────────────────────────────────────────────────
# Results rendering
# ─────────────────────────────────────────────────────────────────────

_COLOUR_MAP = {
    "STRONG_BUY": "#00c853",
    "BUY": "#66bb6a",
    "HOLD": "#ffa726",
    "SELL": "#ef5350",
    "STRONG_SELL": "#b71c1c",
}


def _execute_buy_verdicts(verdicts):
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
        executor = AutoExecutor(kite=kite, auto_place=auto_place)
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
            import pandas as pd
            order_rows = [o.to_dict() for o in report.order_results]
            st.dataframe(pd.DataFrame(order_rows), hide_index=True)
    except Exception as exc:
        logger.exception("Order execution failed")
        st.error(f"Order execution error: {exc}")


def _render_results(verdicts):
    import pandas as pd

    st.subheader("Verdicts")

    # Summary table
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

    # Colour the Verdict column
    def _colour_verdict(row):
        colour = _COLOUR_MAP.get(row["Verdict"], "#888")
        return [
            f"color: {colour}; font-weight: bold" if col == "Verdict" else ""
            for col in row.index
        ]

    styled = df.style.apply(_colour_verdict, axis=1)
    st.dataframe(styled, hide_index=True, width="stretch")

    # ── Execute BUY/STRONG_BUY signals via Kite ──────────────
    buy_verdicts = [
        v for v in verdicts
        if v.classification in ("BUY", "STRONG_BUY")
    ]
    if buy_verdicts:
        st.markdown("---")
        buy_syms = [v.ticker.replace(".NS", "").replace(".BO", "") for v in buy_verdicts]
        st.success(
            f"**{len(buy_verdicts)} BUY signals**: {', '.join(buy_syms)}"
        )
        exec_col, warn_col = st.columns([1, 2])
        with exec_col:
            if st.button(
                f"Place Orders for {len(buy_verdicts)} BUY Signals",
                type="primary",
                key=f"{_PFX}execute_btn",
            ):
                _execute_buy_verdicts(verdicts)
        with warn_col:
            kite = st.session_state.get("kite")
            if kite is None:
                st.warning("Kite session not authenticated — orders will be dry-run only.", icon="\u26a0\ufe0f")

    # Per-ticker expandable breakdown
    for v in verdicts:
        colour = _COLOUR_MAP.get(v.classification, "#888")
        with st.expander(
            f"**{v.ticker}** — "
            f":{v.classification.replace('_', ' ')}: "
            f"({v.final_score:+.2f})",
        ):
            # Radar chart
            radar_bytes = _build_radar(v)
            if radar_bytes:
                st.image(radar_bytes, width=400)

            # Layer details
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
                    # Remove nested per_strategy dicts for cleaner display
                    display = {
                        k: v for k, v in details.items()
                        if k != "per_strategy"
                    }
                    if display:
                        st.json(display)
                    # Show per-strategy in a sub-expander if present
                    per_strat = details.get("per_strategy")
                    if per_strat:
                        with st.expander("Per-strategy breakdown", expanded=False):
                            st.json(per_strat)

                st.markdown("---")


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
        values = [verdict.layer_scores.get(l) or 0 for l in labels]
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
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
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
