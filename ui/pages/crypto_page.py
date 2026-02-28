"""
Crypto Trading Page for Centurion Capital LLC.

Dedicated page for cryptocurrency trading strategies powered by
the Binance public API.  Crypto strategies are isolated here so the
main backtesting page stays focused on equities.
"""

import base64
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import streamlit as st

from config import Config
from trading_strategies import list_strategies
from ui.components import (
    render_page_header,
    render_footer,
    render_no_data_warning,
    spinner_html,
)

logger = logging.getLogger(__name__)

DB_AVAILABLE = Config.is_database_configured()
MINIO_AVAILABLE = True

# Default crypto tickers for Binance
_DEFAULT_CRYPTO_TICKERS = "ETH, BTC, LTC"

# Lazy heavy imports
pd = None   # type: ignore[assignment]
go = None   # type: ignore[assignment]
render_backtest_signals_table = None  # type: ignore[assignment]
_heavy_loaded = False


def _ensure_heavy():
    global pd, go, _heavy_loaded
    if _heavy_loaded:
        return
    import pandas as _pd
    import plotly.graph_objects as _go
    pd = _pd
    go = _go
    globals()['pd'] = _pd
    globals()['go'] = _go
    from ui.tables import render_backtest_signals_table as _rbt
    globals()['render_backtest_signals_table'] = _rbt
    _heavy_loaded = True


def _get_db_service():
    from database.service import get_database_service
    return get_database_service()


def _get_minio():
    from storage.minio_service import get_minio_service
    return get_minio_service()


def _get_strategy(name):
    from trading_strategies import get_strategy
    return get_strategy(name)


# ====================================================================
# Main entry point
# ====================================================================
def render_crypto_page():
    """Render the cryptocurrency trading strategies page."""
    _ensure_heavy()
    logger.info("[user=%s] Viewing Crypto page",
                st.session_state.get('username', 'unknown'))
    render_page_header("₿ Crypto Strategies")

    st.markdown("---")

    # Get only crypto-category strategies
    all_strategies = list_strategies()
    crypto_strategies = [s for s in all_strategies if s.get("category") == "crypto"]

    if not crypto_strategies:
        st.warning("No crypto strategies are registered.")
        render_footer()
        return

    strategy_options = {s["name"]: s for s in crypto_strategies}

    # Initialise crypto-specific cache in session state
    if "crypto_cache" not in st.session_state:
        st.session_state.crypto_cache = {}

    # Layout: config column and results column
    outer_col1, config_col, results_col, outer_col2 = st.columns(
        [0.1, 5.5, 2.8, 0.1]
    )

    with config_col:
        _render_configuration_panel(crypto_strategies, strategy_options)

    with results_col:
        _render_results_panel()

    render_footer()


# ====================================================================
# Configuration panel
# ====================================================================
def _render_configuration_panel(
    strategies: list, strategy_options: Dict[str, Any]
):
    """Render strategy selection, parameters, and data settings."""
    st.subheader("⚙️ Configuration")

    sel_col, _ = st.columns([2, 1])
    with sel_col:
        selected_name = st.selectbox(
            "Select Crypto Strategy",
            options=sorted(strategy_options.keys()),
            help="Choose a cryptocurrency backtesting strategy",
        )

    if not selected_name:
        return

    logger.info("[user=%s] Selected crypto strategy: %s",
                st.session_state.get('username', 'unknown'), selected_name)
    strategy_info = strategy_options[selected_name]
    st.caption(strategy_info["description"])

    # Load cached result if available
    cache = st.session_state.get("crypto_cache", {})
    if selected_name in cache:
        st.session_state.crypto_result = cache[selected_name]
        st.session_state.crypto_selected = selected_name

    # Get strategy class and parameters
    strategy_cls = _get_strategy(strategy_info["id"])
    params = strategy_cls.get_parameters()

    params_col, data_col = st.columns([1, 1], gap="large")
    with params_col:
        st.subheader("📊 Parameters")
        param_values = _render_parameter_inputs(params)

    with data_col:
        st.subheader("🗂️ Data")
        _render_data_settings(param_values)

    if not param_values.get("tickers"):
        st.warning("⚠️ Enter at least 2 crypto ticker symbols (e.g. ETH, BTC, LTC).")

    run_btn = st.button(
        "Run Backtest",
        type="primary",
        disabled=len(param_values.get("tickers", [])) < 2,
        help="Run the crypto strategy with the parameters above",
        key="crypto_run_backtest",
    )

    if selected_name in cache:
        st.caption(f"📦 Cached result loaded for **{selected_name}**")

    if run_btn:
        logger.info("[user=%s] Clicked 'Run Backtest' for crypto strategy: %s",
                    st.session_state.get('username', 'unknown'), selected_name)
        _execute_backtest(strategy_cls, strategy_info, param_values, selected_name)


# ====================================================================
# Parameter inputs
# ====================================================================
def _render_parameter_inputs(params: Dict) -> Dict:
    """Render dynamic parameter inputs based on strategy parameters."""
    param_values: Dict[str, Any] = {}

    for param_name, param_config in params.items():
        param_type = param_config.get("type", "float")
        default = param_config.get("default")
        description = param_config.get("description", "")

        if param_type == "int":
            param_values[param_name] = st.number_input(
                param_name.replace("_", " ").title(),
                value=int(default) if default else 14,
                step=1,
                help=description,
            )
        elif param_type == "float":
            param_values[param_name] = st.number_input(
                param_name.replace("_", " ").title(),
                value=float(default) if default else 0.0,
                format="%.4f",
                help=description,
            )
        elif param_type == "bool":
            param_values[param_name] = st.checkbox(
                param_name.replace("_", " ").title(),
                value=bool(default) if default is not None else True,
                help=description,
            )
        elif param_type == "str":
            param_values[param_name] = st.text_input(
                param_name.replace("_", " ").title(),
                value=str(default) if default else "",
                help=description,
            )

    return param_values


# ====================================================================
# Data settings
# ====================================================================
def _render_data_settings(param_values: Dict):
    """Render ticker, period, and capital inputs for crypto."""
    ticker_input = st.text_input(
        "Crypto Ticker(s)",
        value=_DEFAULT_CRYPTO_TICKERS,
        help=(
            "Enter 2 or more crypto tickers separated by commas. "
            "Use base symbols like ETH, BTC, LTC — they are "
            "automatically mapped to Binance USDT pairs."
        ),
    )
    if ticker_input and ticker_input.strip():
        param_values["tickers"] = [
            t.strip().upper() for t in ticker_input.split(",") if t.strip()
        ]
    else:
        param_values["tickers"] = []

    # Period selection
    period = st.selectbox(
        "Data Period",
        options=["1y", "2y", "3y", "5y", "Custom"],
        index=1,
        help="Crypto markets trade 24/7 — longer periods give more data.",
    )

    if period == "Custom":
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=730),
                max_value=datetime.now(),
            )
        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now(),
            )
        if start_date >= end_date:
            st.warning("⚠️ Start date must be before end date.")
    else:
        end_date = datetime.now()
        period_days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        start_date = end_date - timedelta(days=period_days.get(period, 730))

    param_values["start_date"] = start_date.strftime("%Y-%m-%d")
    param_values["end_date"] = end_date.strftime("%Y-%m-%d")

    capital = st.number_input(
        "Initial Capital ($)",
        value=10000,
        min_value=100,
        step=1000,
    )
    param_values["capital"] = float(capital)


# ====================================================================
# Execute backtest
# ====================================================================
def _execute_backtest(
    strategy_cls, strategy_info: Dict, param_values: Dict, selected_name: str
):
    """Run the crypto backtest and persist results."""
    _crypto_sp = st.empty()
    _crypto_sp.markdown(spinner_html("Running crypto backtest — fetching data from Binance…"), unsafe_allow_html=True)
    try:
        strategy = strategy_cls()
        result = strategy.run(**param_values)
        _crypto_sp.empty()
        st.session_state.crypto_result = result
        st.session_state.crypto_selected = selected_name

        # Update cache
        if "crypto_cache" not in st.session_state:
            st.session_state.crypto_cache = {}
        st.session_state.crypto_cache[selected_name] = result

        if result.success:
            st.success("✅ Crypto backtest completed!")
            _save_backtest_to_database(
                strategy_info, param_values, result, selected_name
            )
            _save_charts_to_minio(result, selected_name)
        else:
            st.error(f"❌ Failed: {result.error_message}")

    except Exception as e:
        _crypto_sp.empty()
        logger.error(f"Error running crypto backtest: {e}")
        st.error(f"❌ Error: {str(e)}")


# ====================================================================
# Results panel
# ====================================================================
def _render_results_panel():
    """Render crypto backtest results."""
    st.subheader("📈 Results")

    cache = st.session_state.get("crypto_cache", {})

    if not cache:
        result = st.session_state.get("crypto_result")
        if result is None:
            st.info("👈 Run a backtest to see results.")
            return
        _render_single_result(result)
        return

    strategy_names = list(cache.keys())
    selected = st.session_state.get("crypto_selected", strategy_names[0])
    default_idx = (
        strategy_names.index(selected) if selected in strategy_names else 0
    )

    tabs = st.tabs(strategy_names)
    for tab, name in zip(tabs, strategy_names):
        with tab:
            _render_single_result(cache[name])


def _render_single_result(result):
    """Render a single strategy result."""
    if not result.success:
        st.error(f"❌ Backtest failed: {result.error_message}")
        return

    # Metrics
    if result.metrics:
        _render_metrics(result.metrics)

    st.markdown("---")

    # Charts
    if result.charts:
        _render_charts(result.charts)

    # Tables
    if result.tables:
        _render_tables(result.tables)

    # Signals
    if result.signals is not None:
        has = (
            (isinstance(result.signals, pd.DataFrame) and not result.signals.empty)
            or (isinstance(result.signals, list) and len(result.signals) > 0)
        )
        if has:
            st.markdown("---")
            st.markdown("#### Trading Signals")
            render_backtest_signals_table(result.signals)


# ====================================================================
# Rendering helpers (metrics, charts, tables)
# ====================================================================
def _render_metrics(metrics: Dict):
    """Render performance metrics."""
    st.markdown("#### Performance Metrics")

    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Separate per-ticker, aggregate, and flat metrics
    ticker_metrics: Dict[str, Dict] = {}
    aggregate_metrics: Dict = {}
    flat_metrics: Dict = {}

    for key, value in metrics.items():
        if isinstance(value, dict):
            if key == "aggregate":
                aggregate_metrics = value
            else:
                ticker_metrics[key] = value
        else:
            flat_metrics[key] = value

    if not ticker_metrics:
        _render_flat_metrics(flat_metrics)
        return

    if aggregate_metrics and len(ticker_metrics) > 1:
        st.markdown("##### 📊 Aggregate Summary")
        agg_cols = st.columns(min(4, len(aggregate_metrics)))
        for i, (key, val) in enumerate(aggregate_metrics.items()):
            with agg_cols[i % len(agg_cols)]:
                label = key.replace("_", " ").title()
                if isinstance(val, float):
                    st.metric(label, f"{val:.4f}")
                else:
                    st.metric(label, str(val))
        st.markdown("")

    display_keys = [
        ("total_return", "Total Return", "pct"),
        ("sharpe_ratio", "Sharpe Ratio", "dec"),
        ("sortino_ratio", "Sortino Ratio", "dec"),
        ("max_drawdown", "Max Drawdown", "pct"),
        ("total_trades", "Total Trades", "int"),
        ("final_value", "Final Value", "dollar"),
    ]

    if len(ticker_metrics) == 1:
        ticker, t_metrics = next(iter(ticker_metrics.items()))
        st.markdown(f"##### {ticker}")
        _render_metric_cards(t_metrics, display_keys)
    else:
        tabs = st.tabs(list(ticker_metrics.keys()))
        for tab, (ticker, t_metrics) in zip(tabs, ticker_metrics.items()):
            with tab:
                _render_metric_cards(t_metrics, display_keys)


def _render_metric_cards(t_metrics: Dict, display_keys: list):
    """Render metric cards for a single ticker."""
    if "error" in t_metrics:
        st.warning(f"Error: {t_metrics['error']}")
        return

    items = []
    for key, label, fmt in display_keys:
        if key in t_metrics and t_metrics[key] is not None:
            items.append((label, t_metrics[key], fmt))

    shown = {k for k, _, _ in display_keys}
    skip = {"ticker", "initial_capital", "calculation_error"}
    for key, val in t_metrics.items():
        if key not in shown and key not in skip and not isinstance(val, (dict, list)):
            items.append((key.replace("_", " ").title(), val, "auto"))

    if not items:
        st.info("No metrics available")
        return

    for row_start in range(0, len(items), 4):
        row = items[row_start : row_start + 4]
        cols = st.columns(len(row))
        for col, (label, val, fmt) in zip(cols, row):
            with col:
                st.metric(label, _fmt(val, fmt))


def _render_flat_metrics(flat_metrics: Dict):
    """Render flat (non-nested) metrics."""
    if not flat_metrics:
        st.info("No metrics available")
        return
    items = []
    for k, v in flat_metrics.items():
        label = k.replace("_", " ").title()
        if isinstance(v, (dict, list)):
            continue
        items.append((label, v))
    for row_start in range(0, len(items), 4):
        row = items[row_start : row_start + 4]
        cols = st.columns(len(row))
        for col, (name, val) in zip(cols, row):
            with col:
                if isinstance(val, float):
                    st.metric(name, f"{val:.4f}")
                else:
                    st.metric(name, str(val))


def _fmt(val, fmt: str) -> str:
    """Format a metric value."""
    if val is None:
        return "N/A"
    if fmt == "pct" and isinstance(val, (int, float)):
        return f"{val:.2f}%"
    if fmt == "dec" and isinstance(val, (int, float)):
        return f"{val:.4f}"
    if fmt == "int":
        return str(int(val)) if isinstance(val, (int, float)) else str(val)
    if fmt == "dollar" and isinstance(val, (int, float)):
        return f"${val:,.2f}"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _render_charts(charts: list):
    """Render charts."""
    st.markdown("#### Charts")
    for chart in charts:
        st.caption(chart.title)
        if chart.chart_type == "matplotlib":
            try:
                img_data = (
                    chart.data.split(",", 1)[1]
                    if chart.data.startswith("data:")
                    else chart.data
                )
                st.image(base64.b64decode(img_data), width="stretch")
            except Exception as e:
                st.warning(f"Could not display chart: {e}")
        elif chart.chart_type == "plotly":
            try:
                fig = go.Figure(json.loads(chart.data))
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.warning(f"Could not display chart: {e}")


def _render_tables(tables: list):
    """Render data tables."""
    st.markdown("---")
    st.markdown("#### Data Tables")
    for table in tables:
        st.caption(table.title)
        if table.data:
            st.dataframe(pd.DataFrame(table.data), width="stretch")


# ====================================================================
# Persistence helpers
# ====================================================================
def _save_backtest_to_database(
    strategy_info: Dict, param_values: Dict, result: Any, selected_name: str
):
    """Save crypto backtest results to database if available."""
    if not DB_AVAILABLE:
        return
    try:
        db_service = _get_db_service()
        tickers_list = param_values.get("tickers", [])
        backtest_data = {
            "strategy_name": selected_name,
            "strategy_id": strategy_info["id"],
            "tickers": tickers_list,
            "start_date": (
                datetime.strptime(param_values.get("start_date", ""), "%Y-%m-%d")
                if param_values.get("start_date")
                else None
            ),
            "end_date": (
                datetime.strptime(param_values.get("end_date", ""), "%Y-%m-%d")
                if param_values.get("end_date")
                else None
            ),
            "initial_capital": param_values.get("capital", 10000),
            "parameters": param_values,
        }
        if result.metrics:
            m = result.metrics
            backtest_data.update(
                {
                    "total_return": m.get("total_return"),
                    "sharpe_ratio": m.get("sharpe_ratio"),
                    "max_drawdown": m.get("max_drawdown"),
                    "total_trades": m.get("total_trades"),
                    "final_value": m.get("final_value"),
                }
            )
        if db_service.save_backtest_result(backtest_data):
            st.caption("🗄️ Results saved to database")
    except Exception as e:
        logger.error(f"Failed to save crypto backtest to database: {e}")


def _save_charts_to_minio(result: Any, strategy_name: str):
    """Save crypto backtest charts to MinIO."""
    if not MINIO_AVAILABLE:
        return
    try:
        minio_svc = _get_minio()
        if not minio_svc.is_available:
            return
        if not result.charts:
            return
        run_id = f"crypto_run_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved = minio_svc.save_backtest_charts(
            run_id=run_id,
            charts=result.charts,
            strategy_name=strategy_name,
        )
        if saved:
            st.toast(
                f"🪣 {len(saved)} chart(s) saved to object storage", icon="✅"
            )
    except Exception as e:
        logger.error(f"Failed to save crypto charts to MinIO: {e}")
