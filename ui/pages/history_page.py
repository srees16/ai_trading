"""
History Page Module for Centurion Capital LLC.

Allows users to review past analysis runs, signals, and backtest results
stored in the database, filtered by date/time.
"""

import base64
import json
import streamlit as st
import pandas as pd
import plotly.io as pio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID

from sqlalchemy import desc

from config import Config
from database.service import get_database_service
from database.repositories import (
    AnalysisRepository, SignalRepository, BacktestRepository
)
from database.models import NewsItem, StockSignal
from storage.minio_service import get_minio_service
from ui.components import render_page_header, render_footer, render_navigation_buttons, render_tickers_being_analyzed, get_decision_emoji

logger = logging.getLogger(__name__)

DB_AVAILABLE = Config.is_database_configured()
MINIO_AVAILABLE = True


def render_history_page():
    """Render the history page for reviewing past analysis results."""
    render_page_header(
        title="📋 History",
        subtitle="Review past analyses, signals, and backtest results"
    )

    # Show stocks being analyzed
    tickers = st.session_state.get('tickers', [])
    if tickers:
        render_tickers_being_analyzed(tickers, st.session_state.get('ticker_mode', 'Default Tickers'))

    st.markdown("---")

    # Navigation
    render_navigation_buttons(
        current_page='history',
        back_key_suffix="history"
    )

    st.markdown("")

    if not DB_AVAILABLE:
        _render_db_unavailable()
        render_footer()
        return

    # Date range filter
    date_range = _render_date_filter()

    # Tabs for different history views
    tab_runs, tab_signals, tab_backtests = st.tabs([
        "📊 Analysis Runs",
        "📈 Trading Signals",
        "🔬 Backtest Results",
    ])

    with tab_runs:
        _render_analysis_runs(date_range)

    with tab_signals:
        _render_signal_history(date_range)

    with tab_backtests:
        _render_backtest_history(date_range)

    render_footer()


def _render_db_unavailable():
    """Render message when database is not configured."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(
            "🗄️ **Database Not Configured**\n\n"
            "History requires a PostgreSQL database connection.\n\n"
            "Set the following environment variables to enable:\n"
            "- `CENTURION_DB_PASSWORD` or `CENTURION_DATABASE_URL`\n"
            "- `CENTURION_DB_ENABLED=true`\n\n"
            "Analysis results will be stored automatically once configured."
        )


def _render_date_filter() -> Dict[str, Any]:
    """
    Render date range selector.

    Returns:
        Dict with 'start_date', 'end_date', and 'days' keys
    """
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        period = st.selectbox(
            "📅 Time Period",
            ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", "Custom Range"],
            index=0,
            key="history_period"
        )

    days_map = {
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Last 90 days": 90,
    }

    if period == "Custom Range":
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=30),
                key="history_start"
            )
        with col3:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                key="history_end"
            )
        return {
            'start_date': datetime.combine(start_date, datetime.min.time()),
            'end_date': datetime.combine(end_date, datetime.max.time()),
            'days': (end_date - start_date).days
        }
    else:
        days = days_map[period]
        return {
            'start_date': datetime.utcnow() - timedelta(days=days),
            'end_date': datetime.utcnow(),
            'days': days
        }


# =================================================================
# Analysis Runs Tab
# =================================================================

def _render_analysis_runs(date_range: Dict[str, Any]):
    """Render past analysis runs with drill-down capability."""
    try:
        db_service = get_database_service()

        with db_service.session_scope() as session:
            repo = AnalysisRepository(session)
            runs = repo.get_recent_runs(limit=100, days=date_range['days'])

            if not runs:
                st.info("No analysis runs found for the selected period.")
                return

            # Summary metrics
            _render_run_summary(runs)

            st.markdown("##### Analysis Run History")

            # Build DataFrame
            run_data = []
            for run in runs:
                run_data.append({
                    'Run ID': str(run.id)[:8],
                    'Date & Time': run.created_at.strftime('%Y-%m-%d %H:%M:%S') if run.created_at else 'N/A',
                    'Type': run.run_type or 'N/A',
                    'Status': _format_status(run.status.value if run.status else 'N/A'),
                    'Tickers': ', '.join(run.tickers) if run.tickers else 'N/A',
                    'Signals': run.total_signals or 0,
                    'News Items': run.total_news_items or 0,
                    'Duration': f"{run.duration_seconds:.1f}s" if run.duration_seconds else 'N/A',
                    '_run_id': str(run.id),
                })

            df = pd.DataFrame(run_data)
            display_df = df.drop(columns=['_run_id'])

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            # Drill-down: select a run to see details
            st.markdown("")
            run_options = {f"{r['Run ID']} — {r['Date & Time']} — {r['Tickers']}": r['_run_id'] for r in run_data}

            if run_options:
                selected = st.selectbox(
                    "🔍 Select a run to view details",
                    options=["— Select —"] + list(run_options.keys()),
                    key="history_run_select"
                )

                if selected != "— Select —":
                    run_id = run_options[selected]
                    _render_run_details(session, run_id)

    except Exception as e:
        logger.error(f"Error loading analysis runs: {e}")
        st.error(f"Failed to load analysis history: {e}")


def _render_run_summary(runs: list):
    """Render summary metrics for analysis runs."""
    total = len(runs)
    completed = sum(1 for r in runs if r.status and r.status.value == 'completed')
    failed = sum(1 for r in runs if r.status and r.status.value == 'failed')
    total_signals = sum(r.total_signals or 0 for r in runs)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", total)
    with col2:
        st.metric("Completed", completed)
    with col3:
        st.metric("Failed", failed)
    with col4:
        st.metric("Total Signals", total_signals)


def _render_run_details(session, run_id: str):
    """Render detailed view for a specific analysis run."""
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        st.error("Invalid run ID")
        return

    st.markdown("---")
    st.markdown(f"##### 📋 Run Details — `{run_id[:8]}`")

    # Fetch signals for this run
    signal_repo = SignalRepository(session)
    signals = signal_repo.get_by_analysis_run(run_uuid)

    if signals:
        st.markdown(f"**Trading Signals** ({len(signals)} signals)")

        signal_data = []
        for s in signals:
            signal_data.append({
                'Ticker': s.ticker,
                'Decision': _format_decision(s.decision.value if s.decision else 'N/A'),
                'Score': f"{s.decision_score:.2f}" if s.decision_score else 'N/A',
                'Price': f"${float(s.current_price):.2f}" if s.current_price else 'N/A',
                'RSI': f"{s.rsi:.1f}" if s.rsi else 'N/A',
                'MACD': f"{s.macd:.4f}" if s.macd else 'N/A',
                'Reasoning': (s.reasoning[:80] + '...') if s.reasoning and len(s.reasoning) > 80 else (s.reasoning or 'N/A'),
                'Date': s.created_at.strftime('%Y-%m-%d %H:%M') if s.created_at else 'N/A',
            })

        df_signals = pd.DataFrame(signal_data)
        st.dataframe(df_signals, use_container_width=True, hide_index=True)
    else:
        st.info("No signals recorded for this run.")

    # Fetch news items for this run
    try:
        news_items = session.query(NewsItem).filter(
            NewsItem.analysis_run_id == run_uuid
        ).order_by(NewsItem.published_at.desc()).all()

        if news_items:
            with st.expander(f"📰 News Items ({len(news_items)})", expanded=False):
                news_data = []
                for n in news_items:
                    news_data.append({
                        'Ticker': n.ticker,
                        'Title': (n.title[:100] + '...') if n.title and len(n.title) > 100 else (n.title or 'N/A'),
                        'Source': n.source or 'N/A',
                        'Sentiment': n.sentiment_label.value if n.sentiment_label else 'N/A',
                        'Confidence': f"{n.sentiment_confidence:.1%}" if n.sentiment_confidence else 'N/A',
                        'Published': n.published_at.strftime('%Y-%m-%d %H:%M') if n.published_at else 'N/A',
                    })
                df_news = pd.DataFrame(news_data)
                st.dataframe(df_news, use_container_width=True, hide_index=True)
    except Exception as e:
        logger.debug(f"Could not load news items for run: {e}")


# =================================================================
# Trading Signals Tab
# =================================================================

def _render_signal_history(date_range: Dict[str, Any]):
    """Render historical trading signals filtered by ticker and date."""
    try:
        db_service = get_database_service()

        # Ticker filter
        ticker_input = st.text_input(
            "🔎 Filter by ticker (comma-separated, leave empty for all)",
            value="",
            key="history_signal_ticker",
            placeholder="e.g. AAPL, GOOGL, TSLA"
        )

        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()] if ticker_input else []

        with db_service.session_scope() as session:
            signal_repo = SignalRepository(session)

            all_signals = []
            if tickers:
                for ticker in tickers:
                    signals = signal_repo.get_by_ticker(
                        ticker,
                        limit=200,
                        days_back=date_range['days']
                    )
                    all_signals.extend(signals)
            else:
                # Get all recent signals
                cutoff = date_range['start_date']
                all_signals = session.query(StockSignal).filter(
                    StockSignal.created_at >= cutoff
                ).order_by(desc(StockSignal.created_at)).limit(500).all()

            if not all_signals:
                st.info("No trading signals found for the selected criteria.")
                return

            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Signals", len(all_signals))
            with col2:
                buy_count = sum(1 for s in all_signals if s.decision and s.decision.value in ('BUY', 'STRONG_BUY'))
                st.metric("Buy Signals", buy_count)
            with col3:
                sell_count = sum(1 for s in all_signals if s.decision and s.decision.value in ('SELL', 'STRONG_SELL'))
                st.metric("Sell Signals", sell_count)

            # Signal table
            st.markdown("##### Signal History")
            signal_data = []
            for s in all_signals:
                signal_data.append({
                    'Date': s.created_at.strftime('%Y-%m-%d %H:%M') if s.created_at else 'N/A',
                    'Ticker': s.ticker,
                    'Decision': _format_decision(s.decision.value if s.decision else 'N/A'),
                    'Score': f"{s.decision_score:.2f}" if s.decision_score is not None else 'N/A',
                    'Price': f"${float(s.current_price):.2f}" if s.current_price else 'N/A',
                    'Change %': f"{s.price_change_pct:.2f}%" if s.price_change_pct is not None else 'N/A',
                    'RSI': f"{s.rsi:.1f}" if s.rsi is not None else 'N/A',
                    'Z-Score': f"{s.altman_z_score:.2f}" if s.altman_z_score is not None else 'N/A',
                    'F-Score': str(s.piotroski_f_score) if s.piotroski_f_score is not None else 'N/A',
                })

            df = pd.DataFrame(signal_data)
            st.dataframe(df.astype(str), use_container_width=True, hide_index=True)

            # Decision distribution chart
            if len(all_signals) > 1:
                with st.expander("📊 Decision Distribution", expanded=False):
                    decision_counts = {}
                    for s in all_signals:
                        d = s.decision.value if s.decision else 'UNKNOWN'
                        decision_counts[d] = decision_counts.get(d, 0) + 1

                    chart_df = pd.DataFrame(
                        list(decision_counts.items()),
                        columns=['Decision', 'Count']
                    )
                    st.bar_chart(chart_df.set_index('Decision'))

    except Exception as e:
        logger.error(f"Error loading signal history: {e}")
        st.error(f"Failed to load signal history: {e}")


# =================================================================
# Backtest Results Tab
# =================================================================

def _render_backtest_history(date_range: Dict[str, Any]):
    """Render historical backtest results."""
    try:
        db_service = get_database_service()

        with db_service.session_scope() as session:
            repo = BacktestRepository(session)
            backtests = repo.get_recent_backtests(
                days=date_range['days'],
                limit=100
            )

            if not backtests:
                st.info("No backtest results found for the selected period.")
                return

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            successful = [b for b in backtests if b.success]
            with col1:
                st.metric("Total Backtests", len(backtests))
            with col2:
                st.metric("Successful", len(successful))
            with col3:
                avg_return = (
                    sum(b.total_return for b in successful if b.total_return is not None)
                    / max(1, len([b for b in successful if b.total_return is not None]))
                ) if successful else 0
                st.metric("Avg Return", f"{avg_return:.2f}%")
            with col4:
                avg_sharpe = (
                    sum(b.sharpe_ratio for b in successful if b.sharpe_ratio is not None)
                    / max(1, len([b for b in successful if b.sharpe_ratio is not None]))
                ) if successful else 0
                st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")

            # Backtest table
            st.markdown("##### Backtest Results")
            bt_data = []
            for b in backtests:
                bt_data.append({
                    'Date': b.created_at.strftime('%Y-%m-%d %H:%M') if b.created_at else 'N/A',
                    'Strategy': b.strategy_name or 'N/A',
                    'Tickers': ', '.join(b.tickers) if b.tickers else 'N/A',
                    'Status': '✅' if b.success else '❌',
                    'Return': f"{b.total_return:.2f}%" if b.total_return is not None else 'N/A',
                    'Sharpe': f"{b.sharpe_ratio:.2f}" if b.sharpe_ratio is not None else 'N/A',
                    'Max DD': f"{b.max_drawdown:.2f}%" if b.max_drawdown is not None else 'N/A',
                    'Win Rate': f"{b.win_rate:.1f}%" if b.win_rate is not None else 'N/A',
                    'Trades': b.total_trades if b.total_trades is not None else 'N/A',
                    'Period': (
                        f"{b.start_date.strftime('%Y-%m-%d')} to {b.end_date.strftime('%Y-%m-%d')}"
                        if b.start_date and b.end_date else 'N/A'
                    ),
                })

            df = pd.DataFrame(bt_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Strategy performance comparison
            if len(successful) > 1:
                with st.expander("📊 Strategy Performance Comparison", expanded=False):
                    _render_strategy_comparison(successful)

            # --- MinIO chart images ---
            if MINIO_AVAILABLE and get_minio_service:
                _render_minio_backtest_charts()

    except Exception as e:
        logger.error(f"Error loading backtest history: {e}")
        st.error(f"Failed to load backtest history: {e}")


def _render_minio_backtest_charts():
    """Render stored backtest chart images from MinIO object storage."""
    try:
        minio_svc = get_minio_service()
        if not minio_svc.is_available:
            return

        run_details = minio_svc.list_runs_detailed()
        if not run_details:
            return

        with st.expander("🖼️ Stored Backtest Charts", expanded=False):
            # --- Runs overview table ---
            def _fmt_size(b: int) -> str:
                if b < 1024:
                    return f"{b} B"
                if b < 1024 ** 2:
                    return f"{b / 1024:.1f} KB"
                return f"{b / (1024 ** 2):.1f} MB"

            table_rows = []
            run_ids = []
            for rd in run_details:
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
            st.dataframe(df_runs, use_container_width=True, hide_index=True)

            # --- Run selector & chart viewer ---
            selected_run = st.selectbox(
                "Select a run to view charts",
                run_ids,
                key="minio_run_select",
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
                        st.image(data, use_container_width=True)

                    elif chart_type == "plotly" and data:
                        st.markdown(f"**{title}**")
                        try:
                            fig_json = json.loads(data)
                            fig = pio.from_json(json.dumps(fig_json))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            # Fallback: show presigned URL link
                            url = img.get("presigned_url")
                            if url:
                                st.markdown(f"[Open chart]({url})")
                    else:
                        st.warning(f"Unknown chart type: {chart_type}")

    except Exception as e:
        logger.error(f"Error rendering MinIO charts: {e}")


def _render_strategy_comparison(backtests: list):
    """Render strategy comparison chart from backtest results."""
    strategy_metrics = {}
    for b in backtests:
        name = b.strategy_name or 'Unknown'
        if name not in strategy_metrics:
            strategy_metrics[name] = {
                'returns': [],
                'sharpes': [],
                'win_rates': [],
            }
        if b.total_return is not None:
            strategy_metrics[name]['returns'].append(b.total_return)
        if b.sharpe_ratio is not None:
            strategy_metrics[name]['sharpes'].append(b.sharpe_ratio)
        if b.win_rate is not None:
            strategy_metrics[name]['win_rates'].append(b.win_rate)

    comparison_data = []
    for name, metrics in strategy_metrics.items():
        comparison_data.append({
            'Strategy': name,
            'Avg Return %': sum(metrics['returns']) / len(metrics['returns']) if metrics['returns'] else 0,
            'Avg Sharpe': sum(metrics['sharpes']) / len(metrics['sharpes']) if metrics['sharpes'] else 0,
            'Avg Win Rate %': sum(metrics['win_rates']) / len(metrics['win_rates']) if metrics['win_rates'] else 0,
            'Runs': len(metrics['returns']),
        })

    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Bar chart of avg returns by strategy
    if len(comparison_data) > 1:
        chart_df = df_comp[['Strategy', 'Avg Return %']].set_index('Strategy')
        st.bar_chart(chart_df)


# =================================================================
# Helper Functions
# =================================================================

def _format_status(status: str) -> str:
    """Format status with emoji indicator."""
    status_map = {
        'completed': '✅ Completed',
        'running': '🔄 Running',
        'pending': '⏳ Pending',
        'failed': '❌ Failed',
    }
    return status_map.get(status.lower(), status)


def _format_decision(decision: str) -> str:
    """Format decision with emoji, reusing the shared emoji map."""
    emoji = get_decision_emoji(decision)
    label = decision.replace('_', ' ')
    return f"{emoji} {label}"
