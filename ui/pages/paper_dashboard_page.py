"""
Paper Trading Dashboard — Real-time view of virtual portfolio performance.

Data source priority:
  1. Neon PostgreSQL (cloud) — always available from any machine
  2. Local SQLite (data/paper_trades.sqlite3) — fallback

Tabs:
  Tab 1 — Overview:   equity curve, KPI cards, daily snapshots table
  Tab 2 — Signals:    signal_log with date/symbol filters
  Tab 3 — Positions:  open + closed trades with P&L
  Tab 4 — Weekly:     weekly checkpoint stats, Sharpe trend
"""

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components import (
    render_header,
    render_footer,
    render_ind_navigation_buttons,
    render_ribbon_and_vix,
)

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "paper_trades.sqlite3"


# ── Data source abstraction ─────────────────────────────────────

def _get_cloud():
    """Return PaperCloudSync instance or None."""
    try:
        from database.paper_cloud import get_paper_cloud
        return get_paper_cloud()
    except Exception:
        return None


def _get_sqlite_conn():
    """Return a read-only sqlite3 connection (or None if DB missing)."""
    if not _DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _sqlite_df(sql: str, params=()) -> pd.DataFrame:
    """Run SQL against local SQLite. Empty DF if unavailable."""
    conn = _get_sqlite_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def _read_snapshots() -> pd.DataFrame:
    cloud = _get_cloud()
    if cloud:
        df = cloud.read_snapshots()
        if not df.empty:
            return df
    return _sqlite_df("SELECT * FROM daily_snapshots ORDER BY date")


def _read_signals() -> pd.DataFrame:
    cloud = _get_cloud()
    if cloud:
        df = cloud.read_signals()
        if not df.empty:
            return df
    return _sqlite_df("SELECT * FROM signal_log ORDER BY date DESC, symbol")


def _read_positions() -> pd.DataFrame:
    cloud = _get_cloud()
    if cloud:
        df = cloud.read_positions()
        if not df.empty:
            return df
    return _sqlite_df("SELECT * FROM paper_positions ORDER BY opened_at DESC")


def _read_weekly() -> pd.DataFrame:
    cloud = _get_cloud()
    if cloud:
        df = cloud.read_weekly()
        if not df.empty:
            return df
    return _sqlite_df("SELECT * FROM weekly_checkpoints ORDER BY week_number")


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

def render_paper_dashboard_page():
    """Entry point for the Paper Trading Dashboard."""
    render_header()
    render_ribbon_and_vix(market="IND")
    render_ind_navigation_buttons(
        current_page="paper_dashboard",
        back_key_suffix="from_paper",
    )

    st.subheader("Paper Trading Dashboard")

    cloud = _get_cloud()
    _source = "cloud (Neon)" if cloud else "local (SQLite)"
    if not _DB_PATH.exists() and not cloud:
        st.warning(
            "No paper trading database found. Start the scheduler with "
            "`CENTURION_PAPER_TRADE=true` to begin collecting data."
        )
        render_footer()
        return

    st.caption(f"Data source: **{_source}**")

    tab_overview, tab_signals, tab_positions, tab_weekly = st.tabs(
        ["Overview", "Signals", "Positions", "Weekly"]
    )

    with tab_overview:
        _render_overview()
    with tab_signals:
        _render_signals()
    with tab_positions:
        _render_positions()
    with tab_weekly:
        _render_weekly()

    render_footer()


# ═══════════════════════════════════════════════════════════════
# Tab 1 — Overview
# ═══════════════════════════════════════════════════════════════

def _render_overview():
    snapshots = _read_snapshots()
    if snapshots.empty:
        st.info("No daily snapshots yet. The scheduler's EOD job writes these at 15:35 IST.")
        return

    # ── KPI cards ──────────────────────────────────────
    latest = snapshots.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity", f"₹{latest['equity']:,.0f}")
    c2.metric("Day P&L", f"₹{latest['day_pnl']:,.0f}")
    c3.metric("Cumulative P&L", f"{latest['cumulative_pnl_pct']:+.1f}%")
    c4.metric("Max Drawdown", f"{latest['max_drawdown_pct']:.1f}%")
    c5.metric("Open Positions", int(latest['open_positions']))

    # ── Equity curve ──────────────────────────────────
    st.markdown("#### Equity Curve")
    chart_df = snapshots[["date", "equity"]].copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.set_index("date")
    st.line_chart(chart_df, use_container_width=True)

    # ── Daily drawdown ────────────────────────────────
    st.markdown("#### Drawdown %")
    dd_df = snapshots[["date", "max_drawdown_pct"]].copy()
    dd_df["date"] = pd.to_datetime(dd_df["date"])
    dd_df = dd_df.set_index("date")
    st.area_chart(dd_df, use_container_width=True, color="#ef5350")

    # ── Snapshots table ───────────────────────────────
    with st.expander("Daily Snapshots Table", expanded=False):
        display_cols = [
            "date", "equity", "cash", "open_positions", "closed_today",
            "day_pnl", "cumulative_pnl_pct", "max_drawdown_pct",
            "signals_generated", "signals_traded",
        ]
        st.dataframe(
            snapshots[display_cols].sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════
# Tab 2 — Signals
# ═══════════════════════════════════════════════════════════════

def _render_signals():
    signals = _read_signals()
    if signals.empty:
        st.info("No signals logged yet.")
        return

    # ── Filters ───────────────────────────────────────
    col_date, col_sym, col_traded = st.columns(3)
    dates = sorted(signals["date"].unique(), reverse=True)
    with col_date:
        sel_date = st.selectbox("Date", ["All"] + list(dates), key="sig_date")
    with col_sym:
        symbols = sorted(signals["symbol"].unique())
        sel_sym = st.selectbox("Symbol", ["All"] + list(symbols), key="sig_sym")
    with col_traded:
        sel_traded = st.selectbox("Traded?", ["All", "Yes", "No"], key="sig_traded")

    df = signals.copy()
    if sel_date != "All":
        df = df[df["date"] == sel_date]
    if sel_sym != "All":
        df = df[df["symbol"] == sel_sym]
    if sel_traded == "Yes":
        df = df[df["was_traded"] == 1]
    elif sel_traded == "No":
        df = df[df["was_traded"] == 0]

    st.caption(f"{len(df)} signals")

    display_cols = [
        "date", "symbol", "forecast", "combined_forecast", "action",
        "entry_price", "stop_loss", "target_price", "quantity",
        "pipeline_sources", "was_traded",
    ]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "forecast": st.column_config.NumberColumn(format="%.2f"),
            "combined_forecast": st.column_config.NumberColumn(format="%.2f"),
            "entry_price": st.column_config.NumberColumn(format="₹%.2f"),
            "stop_loss": st.column_config.NumberColumn(format="₹%.2f"),
            "target_price": st.column_config.NumberColumn(format="₹%.2f"),
        },
    )


# ═══════════════════════════════════════════════════════════════
# Tab 3 — Positions
# ═══════════════════════════════════════════════════════════════

def _render_positions():
    all_pos = _read_positions()
    if all_pos.empty:
        st.info("No paper trades yet.")
        return

    open_pos = all_pos[all_pos["is_open"] == 1]
    closed_pos = all_pos[all_pos["is_open"] == 0]

    # ── Open positions ────────────────────────────────
    st.markdown(f"#### Open Positions ({len(open_pos)})")
    if open_pos.empty:
        st.caption("No open positions.")
    else:
        open_cols = [
            "symbol", "side", "quantity", "entry_price",
            "stop_loss", "target_price", "opened_at",
        ]
        st.dataframe(
            open_pos[open_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "entry_price": st.column_config.NumberColumn(format="₹%.2f"),
                "stop_loss": st.column_config.NumberColumn(format="₹%.2f"),
                "target_price": st.column_config.NumberColumn(format="₹%.2f"),
            },
        )

    # ── Closed positions ──────────────────────────────
    st.markdown(f"#### Closed Trades ({len(closed_pos)})")
    if closed_pos.empty:
        st.caption("No closed trades yet.")
    else:
        closed_cols = [
            "symbol", "side", "quantity", "entry_price", "exit_price",
            "exit_reason", "pnl", "pnl_pct", "opened_at", "closed_at",
        ]
        st.dataframe(
            closed_pos[closed_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "entry_price": st.column_config.NumberColumn(format="₹%.2f"),
                "exit_price": st.column_config.NumberColumn(format="₹%.2f"),
                "pnl": st.column_config.NumberColumn(format="₹%.0f"),
                "pnl_pct": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        # ── P&L summary ──────────────────────────────
        wins = closed_pos[closed_pos["pnl"] > 0]
        losses = closed_pos[closed_pos["pnl"] <= 0]
        total_pnl = closed_pos["pnl"].sum()
        win_rate = len(wins) / len(closed_pos) if len(closed_pos) > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total P&L", f"₹{total_pnl:,.0f}")
        m2.metric("Win Rate", f"{win_rate:.0%}")
        m3.metric("Avg Win", f"{wins['pnl_pct'].mean():+.1f}%" if len(wins) else "—")
        m4.metric("Avg Loss", f"{losses['pnl_pct'].mean():+.1f}%" if len(losses) else "—")

        # Exit reason breakdown
        if "exit_reason" in closed_pos.columns:
            reason_counts = closed_pos["exit_reason"].value_counts()
            st.markdown("**Exit Reasons**")
            st.bar_chart(reason_counts)


# ═══════════════════════════════════════════════════════════════
# Tab 4 — Weekly
# ═══════════════════════════════════════════════════════════════

def _render_weekly():
    weeks = _read_weekly()
    if weeks.empty:
        st.info("No weekly checkpoints yet. These are written after each full trading week.")
        return

    # ── Weekly summary table ──────────────────────────
    display_cols = [
        "week_number", "week_start", "week_end",
        "start_equity", "end_equity", "week_return_pct",
        "trades_opened", "trades_closed", "win_rate",
        "sharpe_ratio", "max_dd_pct", "avg_holding_days",
    ]
    st.dataframe(
        weeks[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "start_equity": st.column_config.NumberColumn(format="₹%.0f"),
            "end_equity": st.column_config.NumberColumn(format="₹%.0f"),
            "week_return_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "win_rate": st.column_config.NumberColumn(format="%.0f%%"),
            "sharpe_ratio": st.column_config.NumberColumn(format="%.3f"),
            "max_dd_pct": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    # ── Weekly return chart ───────────────────────────
    st.markdown("#### Weekly Returns %")
    ret_df = weeks[["week_number", "week_return_pct"]].set_index("week_number")
    st.bar_chart(ret_df, use_container_width=True)

    # ── Sharpe trend ──────────────────────────────────
    st.markdown("#### Rolling Sharpe Ratio")
    sharpe_df = weeks[["week_number", "sharpe_ratio"]].set_index("week_number")
    st.line_chart(sharpe_df, use_container_width=True)

    # ── Equity progression ────────────────────────────
    st.markdown("#### Equity Progression")
    eq_df = weeks[["week_number", "start_equity", "end_equity"]].set_index("week_number")
    st.line_chart(eq_df, use_container_width=True)
