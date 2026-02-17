"""
Backtesting Page Module for Centurion Capital LLC.

Contains the strategy backtesting page rendering.
"""

import json
import base64
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

from config import Config
from trading_strategies import list_strategies, get_strategy
from ui.components import render_page_header, render_footer, render_navigation_buttons
from ui.tables import render_backtest_signals_table

logger = logging.getLogger(__name__)

# Database availability check
try:
    from database.service import get_database_service
    DB_AVAILABLE = Config.is_database_configured()
except ImportError:
    DB_AVAILABLE = False
    get_database_service = None


def render_backtesting_page():
    """Render the strategy backtesting page."""
    render_page_header(
        "🔬 Strategy Backtesting",
        description="Test and analyze trading strategies with historical data"
    )
    
    # Navigation buttons
    render_navigation_buttons(
        current_page='backtesting',
        back_key_suffix='from_backtest'
    )
    
    st.markdown("---")
    
    # Get available strategies
    strategies = list_strategies()
    strategy_options = {s['name']: s for s in strategies}
    
    # Layout: config column and results column
    outer_col1, config_col, results_col, outer_col2 = st.columns([0.3, 1.5, 3, 0.3])
    
    with config_col:
        _render_configuration_panel(strategies, strategy_options)
    
    with results_col:
        _render_results_panel()
    
    render_footer()


def _render_configuration_panel(strategies: list, strategy_options: Dict[str, Any]):
    """Render the strategy configuration panel."""
    st.subheader("⚙️ Configuration")
    
    # Strategy category filter
    categories = sorted(list(set(s['category'] for s in strategies)))
    selected_category = st.selectbox(
        "Strategy Category",
        options=["All"] + categories,
        help="Filter strategies by category"
    )
    
    # Filter strategies by category
    if selected_category != "All":
        filtered_strategies = {
            k: v for k, v in strategy_options.items()
            if v['category'] == selected_category
        }
    else:
        filtered_strategies = strategy_options
    
    # Strategy selection
    selected_name = st.selectbox(
        "Select Strategy",
        options=sorted(filtered_strategies.keys()),
        help="Choose a backtesting strategy"
    )
    
    if selected_name:
        strategy_info = filtered_strategies[selected_name]
        st.caption(strategy_info['description'])
        
        # Get strategy class and parameters
        strategy_cls = get_strategy(strategy_info['id'])
        params = strategy_cls.get_parameters()
        
        st.markdown("---")
        st.subheader("📊 Parameters")
        
        # Dynamic parameter inputs
        param_values = _render_parameter_inputs(params)
        
        st.markdown("---")
        st.subheader("�️ Data Settings")
        
        # Ticker and date inputs
        _render_data_settings(strategy_info, param_values)
        
        st.markdown("---")
        
        run_backtest = st.button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True
        )
        
        if run_backtest:
            _execute_backtest(strategy_cls, strategy_info, param_values, selected_name)


def _render_parameter_inputs(params: Dict) -> Dict:
    """
    Render dynamic parameter inputs based on strategy parameters.
    
    Args:
        params: Dictionary of parameter configurations
    
    Returns:
        Dictionary of parameter values
    """
    param_values = {}
    
    for param_name, param_config in params.items():
        param_type = param_config.get('type', 'float')
        default = param_config.get('default')
        description = param_config.get('description', '')
        
        if param_type == 'int':
            param_values[param_name] = st.number_input(
                param_name.replace('_', ' ').title(),
                value=int(default) if default else 14,
                step=1,
                help=description
            )
        elif param_type == 'float':
            param_values[param_name] = st.number_input(
                param_name.replace('_', ' ').title(),
                value=float(default) if default else 0.0,
                format="%.4f",
                help=description
            )
        elif param_type == 'str':
            param_values[param_name] = st.text_input(
                param_name.replace('_', ' ').title(),
                value=str(default) if default else '',
                help=description
            )
    
    return param_values


def _render_data_settings(strategy_info: Dict, param_values: Dict):
    """
    Render data settings (ticker, period, capital).
    
    Args:
        strategy_info: Strategy information dictionary
        param_values: Parameter values dictionary (modified in place)
    """
    # Ticker input
    if strategy_info['id'] == 'pairs_trading':
        t1, t2 = st.columns(2)
        with t1:
            ticker1 = st.text_input("Ticker 1", value="GLD")
        with t2:
            ticker2 = st.text_input("Ticker 2", value="SLV")
        param_values['tickers'] = (
            [ticker1.upper(), ticker2.upper()]
            if ticker1 and ticker2 else []
        )
    else:
        ticker_input = st.text_input(
            "Ticker Symbol(s)",
            value="AAPL",
            help="Enter one or more tickers separated by commas (e.g. AAPL, MSFT, GOOGL)"
        )
        if ticker_input:
            param_values['tickers'] = [
                t.strip().upper() for t in ticker_input.split(",") if t.strip()
            ]
        else:
            param_values['tickers'] = ["AAPL"]
    
    # Period selection
    period = st.selectbox(
        "Data Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    end_date = datetime.now()
    period_days = {
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825
    }
    start_date = end_date - timedelta(days=period_days.get(period, 365))
    
    param_values['start_date'] = start_date.strftime('%Y-%m-%d')
    param_values['end_date'] = end_date.strftime('%Y-%m-%d')
    
    # Capital input
    capital = st.number_input(
        "Initial Capital ($)",
        value=10000,
        min_value=1000,
        step=1000
    )
    param_values['capital'] = float(capital)


def _execute_backtest(strategy_cls, strategy_info: Dict, param_values: Dict, selected_name: str):
    """
    Execute the backtest and save results.
    
    Args:
        strategy_cls: Strategy class
        strategy_info: Strategy information dictionary
        param_values: Parameter values dictionary
        selected_name: Selected strategy name
    """
    with st.spinner("Running backtest..."):
        try:
            strategy = strategy_cls()
            result = strategy.run(**param_values)
            st.session_state.backtest_result = result
            st.session_state.selected_strategy = selected_name
            
            if result.success:
                st.success("✅ Backtest completed!")
                _save_backtest_to_database(
                    strategy_info, param_values, result, selected_name
                )
            else:
                st.error(f"❌ Failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            st.error(f"❌ Error: {str(e)}")


def _save_backtest_to_database(
    strategy_info: Dict,
    param_values: Dict,
    result: Any,
    selected_name: str
):
    """Save backtest results to database if available."""
    if not DB_AVAILABLE or not get_database_service:
        return
    
    try:
        db_service = get_database_service()
        tickers_list = param_values.get('tickers', [])
        
        backtest_data = {
            'strategy_name': selected_name,
            'strategy_id': strategy_info['id'],
            'tickers': tickers_list,
            'start_date': (
                datetime.strptime(param_values.get('start_date', ''), '%Y-%m-%d')
                if param_values.get('start_date') else None
            ),
            'end_date': (
                datetime.strptime(param_values.get('end_date', ''), '%Y-%m-%d')
                if param_values.get('end_date') else None
            ),
            'initial_capital': param_values.get('capital', 10000),
            'parameters': param_values,
        }
        
        # Extract metrics from result
        if result.metrics:
            metrics = result.metrics
            first_ticker = tickers_list[0] if tickers_list else ''
            
            if first_ticker and first_ticker in metrics:
                m = metrics[first_ticker]
            else:
                m = metrics
            
            backtest_data.update({
                'total_return': m.get('total_return'),
                'sharpe_ratio': m.get('sharpe_ratio'),
                'max_drawdown': m.get('max_drawdown'),
                'win_rate': m.get('win_rate'),
                'total_trades': m.get('total_trades'),
                'final_value': m.get('final_value'),
            })
        
        if db_service.save_backtest_result(backtest_data):
            st.caption("🗄️ Results saved to database")
            
    except Exception as e:
        logger.error(f"Failed to save backtest to database: {e}")


def _render_results_panel():
    """Render the backtest results panel."""
    st.subheader("📈 Results")
    
    result = st.session_state.backtest_result
    
    if result is None:
        st.info("👈 Configure a strategy and click **Run Backtest** to see results")
        return
    
    if not result.success:
        st.error(f"❌ Backtest failed: {result.error_message}")
        return
    
    # Display metrics
    if result.metrics:
        _render_performance_metrics(result.metrics)
    
    st.markdown("---")
    
    # Charts
    if result.charts:
        _render_charts(result.charts)
    
    # Tables
    if result.tables:
        _render_tables(result.tables)
    
    # Signals
    _render_signals(result.signals)


def _render_performance_metrics(metrics: Dict):
    """Render performance metrics cards for all tickers."""
    st.markdown("#### Performance Metrics")
    
    # Shrink metric value font so long numbers are fully visible
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Separate per-ticker metrics, aggregate, and flat (non-dict) metrics
    ticker_metrics: Dict[str, Dict] = {}
    aggregate_metrics: Dict = {}
    flat_metrics: Dict = {}
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            if key == 'aggregate':
                aggregate_metrics = value
            else:
                ticker_metrics[key] = value
        else:
            flat_metrics[key] = value
    
    # If no per-ticker breakdown, fall back to flat rendering
    if not ticker_metrics:
        _render_flat_metrics(flat_metrics)
        return
    
    # Show aggregate summary first (if multiple tickers)
    if aggregate_metrics and len(ticker_metrics) > 1:
        st.markdown("##### 📊 Aggregate Summary")
        agg_cols = st.columns(min(4, len(aggregate_metrics)))
        for i, (key, val) in enumerate(aggregate_metrics.items()):
            with agg_cols[i % len(agg_cols)]:
                label = key.replace('_', ' ').title()
                if isinstance(val, float):
                    if 'return' in key:
                        st.metric(label, f"{val:.2f}%")
                    else:
                        st.metric(label, f"{val:.4f}")
                else:
                    st.metric(label, str(val))
        st.markdown("")
    
    # Render per-ticker metrics
    display_keys = [
        ('total_return', 'Total Return', 'pct'),
        ('sharpe_ratio', 'Sharpe Ratio', 'dec'),
        ('sortino_ratio', 'Sortino Ratio', 'dec'),
        ('max_drawdown', 'Max Drawdown', 'pct'),
        ('total_trades', 'Total Trades', 'int'),
        ('final_value', 'Final Value', 'dollar'),
        ('mean_return', 'Mean Return', 'pct'),
        ('std_return', 'Std Return', 'pct'),
    ]
    
    if len(ticker_metrics) == 1:
        # Single ticker — show directly without tabs
        ticker, t_metrics = next(iter(ticker_metrics.items()))
        st.markdown(f"##### {ticker}")
        _render_ticker_metric_cards(t_metrics, display_keys)
    else:
        # Multiple tickers — use tabs
        tabs = st.tabs(list(ticker_metrics.keys()))
        for tab, (ticker, t_metrics) in zip(tabs, ticker_metrics.items()):
            with tab:
                _render_ticker_metric_cards(t_metrics, display_keys)


def _render_ticker_metric_cards(t_metrics: Dict, display_keys: list):
    """Render metric cards for a single ticker's metrics."""
    if 'error' in t_metrics:
        st.warning(f"Error: {t_metrics['error']}")
        return
    
    # Collect available metrics in display order
    items = []
    for key, label, fmt in display_keys:
        if key in t_metrics and t_metrics[key] is not None:
            items.append((label, t_metrics[key], fmt))
    
    # Also pick up any extra keys not in the predefined list
    shown_keys = {k for k, _, _ in display_keys}
    skip_keys = {'ticker', 'initial_capital', 'calculation_error'}
    for key, val in t_metrics.items():
        if key not in shown_keys and key not in skip_keys:
            items.append((key.replace('_', ' ').title(), val, 'auto'))
    
    if not items:
        st.info("No metrics available")
        return
    
    # Render in rows of 4
    for row_start in range(0, len(items), 4):
        row = items[row_start:row_start + 4]
        cols = st.columns(len(row))
        for col, (label, val, fmt) in zip(cols, row):
            with col:
                st.metric(label, _format_metric_value(val, fmt))


def _format_metric_value(val, fmt: str) -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if fmt == 'pct' and isinstance(val, (int, float)):
        return f"{val:.2f}%"
    if fmt == 'dec' and isinstance(val, (int, float)):
        return f"{val:.4f}"
    if fmt == 'int':
        return str(int(val)) if isinstance(val, (int, float)) else str(val)
    if fmt == 'dollar' and isinstance(val, (int, float)):
        return f"${val:,.2f}"
    # auto
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _render_flat_metrics(flat_metrics: Dict):
    """Fallback renderer for non-nested metrics dictionaries."""
    if not flat_metrics:
        st.info("No metrics available")
        return
    
    items = [(k.replace('_', ' ').title(), v) for k, v in flat_metrics.items()]
    for row_start in range(0, len(items), 4):
        row = items[row_start:row_start + 4]
        cols = st.columns(len(row))
        for col, (name, val) in zip(cols, row):
            with col:
                if isinstance(val, float):
                    if 'return' in name.lower() or 'drawdown' in name.lower():
                        st.metric(name, f"{val:.2f}%")
                    else:
                        st.metric(name, f"{val:.4f}")
                else:
                    st.metric(name, str(val))


def _render_charts(charts: list):
    """Render backtest charts."""
    st.markdown("#### Charts")
    
    for chart in charts:
        st.caption(chart.title)
        
        if chart.chart_type == 'matplotlib':
            try:
                img_data = (
                    chart.data.split(',', 1)[1]
                    if chart.data.startswith('data:')
                    else chart.data
                )
                st.image(base64.b64decode(img_data), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display chart: {e}")
                
        elif chart.chart_type == 'plotly':
            try:
                fig = go.Figure(json.loads(chart.data))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display chart: {e}")


def _render_tables(tables: list):
    """Render backtest data tables."""
    st.markdown("---")
    st.markdown("#### Data Tables")
    
    for table in tables:
        st.caption(table.title)
        if table.data:
            st.dataframe(pd.DataFrame(table.data), use_container_width=True)


def _render_signals(signals: Optional[Any]):
    """Render trading signals table."""
    has_signals = signals is not None and (
        (isinstance(signals, pd.DataFrame) and not signals.empty) or
        (isinstance(signals, list) and len(signals) > 0)
    )
    
    if has_signals:
        st.markdown("---")
        st.markdown("#### Trading Signals")
        render_backtest_signals_table(signals)
