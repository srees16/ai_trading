"""
Financial ML History Page for Centurion Capital LLC.

Provides a Streamlit UI to browse, filter, and drill-down into past
Financial ML chapter results stored in the PostgreSQL ``backtest_results``
table (``strategy_id LIKE 'fml_%'``).  Text/tabular output is read from
the ``metrics`` JSONB column; stored MinIO figures are also viewable.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import streamlit as st

from config import Config
from ui.components import render_header, render_footer
from ui.styles import Colors

logger = logging.getLogger(__name__)

DB_AVAILABLE = Config.is_database_configured()
MINIO_AVAILABLE = True

# Lazy heavy imports
pd = None  # type: ignore[assignment]
pio = None  # type: ignore[assignment]
_heavy_loaded = False

# Chapter key→name lookup built from the same registry used in the main
# FML page.  Duplicated here to avoid importing the entire page module.
_CHAPTER_MAP: Dict[str, str] = {
    "ch02": "Financial Data Structures",
    "ch03": "Labeling (Triple-Barrier)",
    "ch04": "Sample Weights",
    "ch05": "Fractional Differentiation",
    "ch06": "Ensemble Methods",
    "ch07": "Cross-Validation",
    "ch08": "Feature Importance",
    "ch09": "Hyper-Parameter Tuning",
    "ch10": "Bet Sizing",
    "ch11": "Dangers of Backtesting",
    "ch13": "Synthetic Backtesting",
    "ch14": "Backtest Statistics",
    "ch15": "Strategy Risk",
    "ch16": "ML Asset Allocation",
    "ch17": "Structural Breaks",
    "ch18": "Entropy Features",
    "ch19": "Microstructural Features",
    "ch20": "Multiprocessing",
    "ch21": "Brute Force & Quantum",
}


def _ensure_heavy():
    global pd, pio, _heavy_loaded
    if _heavy_loaded:
        return
    import pandas as _pd
    import plotly.io as _pio
    pd = _pd
    pio = _pio
    globals().update({"pd": _pd, "pio": _pio})
    _heavy_loaded = True


def _get_db_service():
    from database.service import get_database_service
    return get_database_service()


def _get_minio():
    from storage.minio_service import get_minio_service
    return get_minio_service()


def _get_backtest_repo_cls():
    from database.repositories import BacktestRepository
    return BacktestRepository


# ════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════

def render_fml_history_page():
    """Render the Financial ML history / saved results page."""
    _ensure_heavy()
    _user = st.session_state.get("username", "unknown")
    logger.info("[user=%s] Viewing FML History page", _user)

    render_header()

    # Title
    st.markdown(
        f"""
        <div style="text-align:center; padding:0.4rem 0 0.2rem;">
            <span style="font-size:1.5rem; font-weight:700;
                         color:{Colors.TEXT_PRIMARY};">
                Financial ML — Saved Results
            </span><br>
            <span style="font-size:0.85rem; color:{Colors.TEXT_MUTED};">
                Browse and query past chapter outputs stored in PostgreSQL &amp; MinIO
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Back button
    if st.button("← Back to Financial ML", key="fml_hist_back"):
        st.session_state["current_page"] = "finance_ml"
        st.rerun()

    st.markdown("---")

    if not DB_AVAILABLE:
        _render_db_unavailable()
        render_footer()
        return

    # Filters
    date_range = _render_date_filter()
    chapter_filter, ticker_filter = _render_extra_filters()

    # Tabs
    tab_results, tab_charts = st.tabs([
        "Saved Results",
        "Stored Charts (MinIO)",
    ])

    with tab_results:
        _render_fml_results(date_range, chapter_filter, ticker_filter)

    with tab_charts:
        _render_fml_minio_charts()

    render_footer()


# ════════════════════════════════════════════════════════════════════════
# Filter widgets
# ════════════════════════════════════════════════════════════════════════

def _render_db_unavailable():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(
            "**Database Not Configured**\n\n"
            "History requires a PostgreSQL database connection.\n\n"
            "Set the following environment variables to enable:\n"
            "- `CENTURION_DB_PASSWORD` or `CENTURION_DATABASE_URL`\n"
            "- `CENTURION_DB_ENABLED=true`\n\n"
            "Financial ML results will be stored automatically once configured."
        )


def _render_date_filter() -> Dict[str, Any]:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        period = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", "All Time", "Custom Range"],
            index=2,
            key="fml_hist_period",
        )

    days_map = {
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "All Time": 3650,
    }

    if period == "Custom Range":
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=30),
                key="fml_hist_start",
            )
        with col3:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                key="fml_hist_end",
            )
        return {
            "start_date": datetime.combine(start_date, datetime.min.time()),
            "end_date": datetime.combine(end_date, datetime.max.time()),
            "days": (end_date - start_date).days,
        }

    days = days_map[period]
    return {
        "start_date": datetime.utcnow() - timedelta(days=days),
        "end_date": datetime.utcnow(),
        "days": days,
    }


def _render_extra_filters():
    col1, col2 = st.columns(2)

    with col1:
        chapter_options = ["All Chapters"] + [
            f"{k} — {v}" for k, v in _CHAPTER_MAP.items()
        ]
        chapter_sel = st.selectbox(
            "Chapter",
            chapter_options,
            index=0,
            key="fml_hist_chapter",
        )

    with col2:
        ticker_input = st.text_input(
            "Filter by ticker (comma-separated, leave empty for all)",
            value="",
            key="fml_hist_ticker",
            placeholder="e.g. MSFT, GOOG",
        )

    # Parse chapter filter → strategy_id or None
    chapter_filter = None
    if chapter_sel != "All Chapters":
        ch_key = chapter_sel.split(" — ")[0].strip()
        chapter_filter = f"fml_{ch_key}"

    # Parse ticker filter
    ticker_filter = (
        [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if ticker_input
        else []
    )

    return chapter_filter, ticker_filter


# ════════════════════════════════════════════════════════════════════════
# Saved Results tab
# ════════════════════════════════════════════════════════════════════════

def _render_fml_results(
    date_range: Dict[str, Any],
    chapter_filter: str | None,
    ticker_filter: List[str],
):
    """Query and display FML results from PostgreSQL."""
    try:
        db_service = _get_db_service()
        BacktestRepository = _get_backtest_repo_cls()

        with db_service.session_scope() as session:
            from database.models import BacktestResult
            from sqlalchemy import desc as sa_desc

            cutoff = date_range["start_date"]
            query = session.query(BacktestResult).filter(
                BacktestResult.strategy_id.like("fml_%"),
                BacktestResult.created_at >= cutoff,
                BacktestResult.created_at <= date_range["end_date"],
            )

            if chapter_filter:
                query = query.filter(BacktestResult.strategy_id == chapter_filter)

            if ticker_filter:
                for tk in ticker_filter:
                    query = query.filter(BacktestResult.tickers.contains([tk]))

            results = (
                query.order_by(sa_desc(BacktestResult.created_at))
                .limit(200)
                .all()
            )

            if not results:
                st.info("No Financial ML results found for the selected filters.")
                return

            # Summary metrics
            _render_summary_metrics(results)

            # Results table
            st.markdown("##### Chapter Results")
            row_data = []
            for r in results:
                params = r.parameters or {}
                metrics = r.metrics or {}
                text_output = metrics.get("text_output", "")
                tables = metrics.get("tables", [])
                row_data.append({
                    "Date": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "N/A",
                    "Chapter": params.get("chapter", r.strategy_id),
                    "Name": params.get("chapter_name", r.strategy_name),
                    "Tickers": ", ".join(r.tickers) if r.tickers else "—",
                    "Output Lines": params.get("output_lines", len(text_output.splitlines()) if text_output else 0),
                    "Figures": params.get("figure_count", 0),
                    "Tables": len(tables),
                    "_idx": len(row_data),
                })

            df = pd.DataFrame(row_data)
            display_df = df.drop(columns=["_idx"])
            st.dataframe(display_df, width="stretch", hide_index=True)

            # Chapter distribution chart
            if len(results) > 1:
                with st.expander("Chapter Distribution", expanded=False):
                    ch_counts = {}
                    for rd in row_data:
                        ch = rd["Chapter"]
                        ch_counts[ch] = ch_counts.get(ch, 0) + 1
                    chart_df = pd.DataFrame(
                        list(ch_counts.items()), columns=["Chapter", "Runs"]
                    )
                    st.bar_chart(chart_df.set_index("Chapter"))

            # Drill-down
            st.markdown("")
            options_map = {
                f"{rd['Date']} | {rd['Chapter']} — {rd['Name']} | {rd['Tickers']}": rd["_idx"]
                for rd in row_data
            }
            selected = st.selectbox(
                "Select a result to view details",
                options=["— Select —"] + list(options_map.keys()),
                key="fml_hist_detail_select",
            )

            if selected != "— Select —":
                idx = options_map[selected]
                _render_result_detail(results[idx])

    except Exception as e:
        logger.error("Error loading FML history: %s", e)
        st.error(f"Failed to load Financial ML history: {e}")


def _render_summary_metrics(results: list):
    """Summary cards for the filtered result set."""
    total = len(results)
    chapters_seen = set()
    total_output_lines = 0
    total_figures = 0
    for r in results:
        params = r.parameters or {}
        chapters_seen.add(params.get("chapter", r.strategy_id))
        total_output_lines += params.get("output_lines", 0)
        total_figures += params.get("figure_count", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Runs", total)
    with c2:
        st.metric("Unique Chapters", len(chapters_seen))
    with c3:
        st.metric("Total Output Lines", total_output_lines)
    with c4:
        st.metric("Total Figures", total_figures)


def _render_result_detail(result):
    """Expand a single FML result showing full text output and tables."""
    st.markdown("---")
    params = result.parameters or {}
    metrics = result.metrics or {}
    ch_key = params.get("chapter", result.strategy_id)
    ch_name = params.get("chapter_name", result.strategy_name)

    st.markdown(f"##### {ch_key} — {ch_name}")
    st.caption(
        f"Run: {result.created_at.strftime('%Y-%m-%d %H:%M:%S') if result.created_at else '—'}  |  "
        f"Tickers: {', '.join(result.tickers) if result.tickers else '—'}"
    )

    # Text output
    text_output = metrics.get("text_output", "")
    if text_output:
        with st.expander("Text Output", expanded=True):
            st.code(text_output, language="text")
    else:
        st.info("No text output stored for this run.")

    # Tables
    tables = metrics.get("tables", [])
    if tables:
        with st.expander(f"Tables ({len(tables)})", expanded=True):
            for tbl in tables:
                title = tbl.get("title", "Untitled Table")
                data = tbl.get("data", [])
                st.markdown(f"**{title}**")
                if data:
                    tbl_df = pd.DataFrame(data)
                    st.dataframe(tbl_df, width="stretch", hide_index=True)
                else:
                    st.caption("(empty table)")
    else:
        st.info("No tables stored for this run.")

    # Raw parameters JSON
    with st.expander("Raw Parameters", expanded=False):
        st.json(params)


# ════════════════════════════════════════════════════════════════════════
# MinIO Charts tab
# ════════════════════════════════════════════════════════════════════════

def _render_fml_minio_charts():
    """Show stored FML figures from MinIO (runs with fml_ prefix)."""
    if not MINIO_AVAILABLE:
        st.info("MinIO storage is not available.")
        return

    try:
        minio_svc = _get_minio()
        if not minio_svc.is_available:
            st.info("MinIO service is not reachable.")
            return

        run_details = minio_svc.list_runs_detailed()
        if not run_details:
            st.info("No MinIO runs found.")
            return

        # Only show runs with the fml_ prefix
        fml_runs = [rd for rd in run_details if rd["run_id"].startswith("fml_")]
        if not fml_runs:
            st.info("No Financial ML chart runs found in MinIO.")
            return

        def _fmt_size(b: int) -> str:
            if b < 1024:
                return f"{b} B"
            if b < 1024 ** 2:
                return f"{b / 1024:.1f} KB"
            return f"{b / (1024 ** 2):.1f} MB"

        table_rows = []
        run_ids = []
        for rd in fml_runs:
            run_ids.append(rd["run_id"])
            table_rows.append({
                "Run ID": rd["run_id"],
                "Charts": rd["chart_count"],
                "Size": _fmt_size(rd["total_size"]),
                "Strategies": ", ".join(rd["strategies"]) if rd["strategies"] else "—",
                "Tickers": ", ".join(rd["tickers"]) if rd["tickers"] else "—",
                "Created At": (
                    rd["created_at"].strftime("%Y-%m-%d %H:%M:%S")
                    if rd["created_at"]
                    else "—"
                ),
            })

        df_runs = pd.DataFrame(table_rows).astype(str)
        st.dataframe(df_runs, width="stretch", hide_index=True)

        selected_run = st.selectbox(
            "Select a run to view charts",
            run_ids,
            key="fml_hist_minio_run",
        )

        if selected_run:
            images = minio_svc.get_backtest_images(run_id=selected_run)
            if not images:
                st.info("No chart images found for this run.")
                return

            st.caption(f"Found {len(images)} chart(s) for run **{selected_run}**")

            for img in images:
                title = img.get("chart_title") or img.get("object_name", "")
                chart_type = img.get("chart_type", "matplotlib")
                data = img.get("data", b"")

                if chart_type == "matplotlib" and data:
                    st.markdown(f"**{title}**")
                    st.image(data, width="stretch")
                elif chart_type == "plotly" and data:
                    st.markdown(f"**{title}**")
                    try:
                        fig_json = json.loads(data)
                        fig = pio.from_json(json.dumps(fig_json))
                        st.plotly_chart(fig, width="stretch")
                    except Exception:
                        url = img.get("presigned_url")
                        if url:
                            st.markdown(f"[Open chart]({url})")
                else:
                    st.warning(f"Unknown chart type: {chart_type}")

    except Exception as e:
        logger.error("Error rendering FML MinIO charts: %s", e)
        st.info("Could not load MinIO charts.")
