"""
Backtesting Page Module for Centurion Capital LLC.

Contains the strategy backtesting page rendering.
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
from ui.components import render_page_header, render_footer, render_navigation_buttons, render_ind_navigation_buttons, render_stock_ticker_ribbon, render_vix_indicator, render_no_data_warning

logger = logging.getLogger(__name__)

DB_AVAILABLE = Config.is_database_configured()
MINIO_AVAILABLE = True


def _compact_caption(text: str) -> None:
    """Render a small, tight caption without extra vertical margin.

    Streamlit's ``st.caption`` adds top/bottom padding that can create
    large gaps when used immediately above buttons or other controls.
    This helper outputs the same text inside a zero-margin paragraph.
    """

    st.markdown(f"<p style='margin:0;padding:0'>{text}</p>", unsafe_allow_html=True)

# Lazy heavy imports — pandas alone takes ~70 s on some machines.
pd = None       # type: ignore[assignment]
go = None       # type: ignore[assignment]


def _ensure_heavy():
    """Import pandas, plotly, and ui.tables on first use."""
    global pd, go
    if pd is not None:
        return
    import pandas as _pd
    import plotly.graph_objects as _go
    pd = _pd
    go = _go
    globals()['pd'] = _pd
    globals()['go'] = _go
    from ui.tables import render_backtest_signals_table as _rbt
    globals()['render_backtest_signals_table'] = _rbt


# Forward-declare for Pylance
render_backtest_signals_table = None  # type: ignore[assignment]


def _get_db_service():
    """Lazy import of database.service (pulls in SQLAlchemy)."""
    from database.service import get_database_service
    return get_database_service()


def _get_minio():
    """Lazy import of storage.minio_service."""
    from storage.minio_service import get_minio_service
    return get_minio_service()


def _get_strategy(name):
    """Lazy import of get_strategy (pulls in the actual strategy class)."""
    from trading_strategies import get_strategy
    return get_strategy(name)

# ── Green "Run Backtest" button CSS ─────────────────────────────────
_STRATEGY_SPINNER_CSS = """
<style>
/* Green "Run Backtest" button */
[data-testid="stMainBlockContainer"] button[kind="primary"].bt-green-btn,
.bt-green-btn-scope button[kind="primary"] {
    background-color: #38a169 !important;
    border-color: #38a169 !important;
    color: #fff !important;
    white-space: nowrap !important;
    font-size: 0.82rem !important;
}
.bt-green-btn-scope button[kind="primary"]:hover {
    background-color: #2f855a !important;
    border-color: #2f855a !important;
}
</style>
"""

def _spinner_html(pct: int, label: str) -> str:
    from ui.components import spinner_html
    return spinner_html(f"{label} — {pct}%")


def render_backtesting_page():
    """Render the strategy backtesting page."""
    _ensure_heavy()  # lazy-load pandas, plotly, ui.tables
    logger.info("[user=%s] Viewing Backtesting page",
                st.session_state.get('username', 'unknown'))
    market = st.session_state.get('current_market', 'US')
    market_label = "Indian" if market == 'IND' else "US"
    render_page_header(
        f"{market_label} Backtest Strategy"
    )

    # Navigation buttons
    if market == 'IND':
        render_stock_ticker_ribbon(market="IND")
        render_vix_indicator(market="IND")
        render_ind_navigation_buttons(current_page='backtesting', back_key_suffix='from_backtest')
    else:
        render_stock_ticker_ribbon(market="US")
        render_vix_indicator(market="US")
        render_navigation_buttons(
            current_page='backtesting',
            back_key_suffix='from_backtest'
        )
    
    st.markdown("---")

    # Guard: if no analysis has been run yet, show a helpful warning
    if not st.session_state.get('analysis_complete', False):
        render_no_data_warning(page_name="backtesting")
        render_footer()
        return
    
    # Initialise cache in session state
    if 'backtest_cache' not in st.session_state:
        st.session_state.backtest_cache = {}  # {strategy_name: StrategyResult}
    
    # Get available strategies (exclude crypto — those live on the Crypto page)
    strategies = [s for s in list_strategies() if s.get('category') != 'crypto']
    strategy_options = {s['name']: s for s in strategies}
    
    # Ensure MinIO bucket exists before any chart saving
    _ensure_minio_bucket()

    # Pre-run all strategies once on first page visit
    _precompute_all_strategies(strategies)
    
    # Layout: config column and results column
    outer_col1, config_col, results_col, outer_col2 = st.columns([0.3, 1.5, 3, 0.3])
    
    with config_col:
        _render_configuration_panel(strategies, strategy_options)
    
    with results_col:
        _render_results_panel()
    
    render_footer()


def _ensure_minio_bucket():
    """Check (and create if missing) the MinIO 'centurion-backtests' bucket.

    Called once when the backtesting page loads so that all subsequent
    chart-save operations have a guaranteed destination bucket.
    """
    if not MINIO_AVAILABLE:
        return
    try:
        minio_svc = _get_minio()
        ready = minio_svc.ensure_bucket_ready()
        if ready:
            logger.info("MinIO bucket '%s' is ready",
                        minio_svc._config.bucket_name)
        else:
            logger.warning("MinIO bucket could not be verified/created — "
                           "charts will not be persisted this session")
    except Exception as e:
        logger.error("MinIO bucket pre-check failed: %s", e)


def _precompute_all_strategies(strategies: list):
    """
    Pre-compute backtests for all strategies using the analysis tickers.

    Results are stored in ``st.session_state.backtest_cache`` keyed by
    strategy name.  The cache is keyed exclusively by ``analysis_run_id``
    which only increments when the user clicks **Run Analysis** on the
    main page.  Navigating between pages never invalidates the cache.
    """
    # Only auto-precompute when the user has explicitly run an analysis
    if not st.session_state.get('analysis_complete', False):
        st.info(" Run an analysis on the main page first to auto-compute all strategies.")
        return

    # Use the snapshot of tickers captured at "Run Analysis" time.
    # This is immune to the main-page widgets overwriting
    # st.session_state.tickers during casual navigation.
    tickers = st.session_state.get('analysis_tickers',
                                   st.session_state.get('tickers', []))
    if not tickers:
        return

    # The main page increments this counter every time the user clicks
    # "Run Analysis", so it is the *sole* cache-buster.
    analysis_run = st.session_state.get('analysis_run_id', 0)
    cached_run   = st.session_state.get('backtest_cache_run_id', None)

    # Cache is still valid skip recomputation entirely
    if st.session_state.get('backtest_cache') and analysis_run == cached_run:
        return

    # ------------------------------------------------------------------
    # Cache miss — run all strategies
    # ------------------------------------------------------------------
    st.session_state.backtest_cache = {}
    st.session_state.backtest_result = None

    # Determine which strategies to pre-compute
    # Skip strategies that need more tickers than we have (e.g. pairs
    # trading, mean-reversion cointegration when only 1 ticker).
    num_tickers = len(tickers)
    eligible = [
        s for s in strategies
        if num_tickers >= s.get('min_tickers', 1)
    ]
    if not eligible:
        return

    # Default date range and capital
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    default_dates = {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'capital': 10000.0,
        'tickers': list(tickers),
    }

    spinner_slot = st.empty()
    st.markdown(_STRATEGY_SPINNER_CSS, unsafe_allow_html=True)
    total = len(eligible)
    succeeded = 0
    total_minio_saved = 0

    # Single run_id shared by all strategies in this batch
    minio_run_id = _build_minio_run_id(tickers, market=st.session_state.get('current_market', 'US'))

    for idx, s_info in enumerate(eligible):
        strategy_name = s_info['name']
        pct = int(idx / total * 100)
        logger.info("[user=%s] Pre-computing strategy %d/%d: %s",
                    st.session_state.get('username', 'unknown'),
                    idx + 1, total, strategy_name)
        spinner_slot.markdown(
            _spinner_html(pct, f"Running {strategy_name} ({idx + 1}/{total})…"),
            unsafe_allow_html=True,
        )

        try:
            strategy_cls = _get_strategy(s_info['id'])
            params = strategy_cls.get_parameters()

            param_values = {
                pname: pconf.get('default', 0)
                for pname, pconf in params.items()
            }
            param_values.update(default_dates)

            strategy = strategy_cls()
            result = strategy.run(**param_values)
            st.session_state.backtest_cache[strategy_name] = result

            if result.success:
                succeeded += 1
                logger.info("[user=%s] Strategy '%s' pre-compute succeeded",
                            st.session_state.get('username', 'unknown'), strategy_name)
                # Save charts to MinIO during pre-computation
                total_minio_saved += _save_charts_to_minio(
                    run_id=minio_run_id,
                    result=result,
                    strategy_name=strategy_name,
                )
        except Exception as e:
            logger.error(f"Pre-compute failed for {strategy_name}: {e}")

    spinner_slot.markdown(
        _spinner_html(100, " All strategies computed "),
        unsafe_allow_html=True,
    )
    import time as _time; _time.sleep(0.6)
    spinner_slot.empty()

    # Persist cache metadata
    st.session_state.backtest_cache_tickers = sorted(set(t.upper() for t in tickers))
    st.session_state.backtest_cache_run_id = analysis_run
    st.session_state.backtest_minio_run_id = minio_run_id

    # Set the first successful result as the active display
    if st.session_state.backtest_cache:
        first_name = next(iter(st.session_state.backtest_cache))
        st.session_state.backtest_result = st.session_state.backtest_cache[first_name]
        st.session_state.selected_strategy = first_name

    logger.info("[user=%s] Pre-computation complete: %d/%d strategies succeeded for %s",
                st.session_state.get('username', 'unknown'),
                succeeded, total, ', '.join(tickers))


def _render_configuration_panel(strategies: list, strategy_options: Dict[str, Any]):
    """Render the strategy configuration panel."""
    st.subheader(" Configuration")
    
    # Strategy category filter
    categories = sorted(list(set(s['category'] for s in strategies)))
    display_categories = {c: c.replace('_', ' ').title() for c in categories}
    selected_display = st.selectbox(
        "Strategy Category",
        options=["All"] + [display_categories[c] for c in categories],
        help="Filter strategies by category"
    )
    
    # Map display name back to raw category
    reverse_map = {v: k for k, v in display_categories.items()}
    selected_category = reverse_map.get(selected_display, selected_display)
    
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
        
        # Instantly load cached result when user switches strategy
        cache = st.session_state.get('backtest_cache', {})
        if selected_name in cache:
                st.session_state.backtest_result = cache[selected_name]
                st.session_state.selected_strategy = selected_name
        
        # Get strategy class and parameters
        strategy_cls = _get_strategy(strategy_info['id'])
        params = strategy_cls.get_parameters()
        
        st.markdown("---")
        st.subheader(" Parameters")
        
        # Dynamic parameter inputs
        param_values = _render_parameter_inputs(params)
        
        st.markdown("---")
        st.subheader(" Data Settings")
        
        # Ticker and date inputs
        _render_data_settings(strategy_info, param_values)
        
        st.markdown("---")
        
        # Warn if no tickers provided
        if not param_values.get('tickers'):
            st.warning(" Please enter at least one ticker symbol above.")
        
        # Show cache status (render with no extra vertical margin)
        if selected_name in cache:
            _compact_caption(f"Cached result loaded for **{selected_name}**")
        
        btn_col, _ = st.columns([1.3, 1.7])
        with btn_col:
            st.markdown('<div class="bt-green-btn-scope">', unsafe_allow_html=True)
            run_backtest = st.button(
                " Run Backtest",
                type="primary",
                disabled=not param_values.get('tickers'),
                help="Run with custom parameters (overrides cached result)",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if run_backtest:
            logger.info("[user=%s] Clicked 'Run Backtest' for strategy: %s",
                        st.session_state.get('username', 'unknown'), selected_name)
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
        # Pre-fill with user's tickers from the main page
        session_tickers = st.session_state.get('tickers', [])
        default_ticker_str = ", ".join(session_tickers) if session_tickers else ""
        ticker_input = st.text_input(
            "Ticker Symbol(s)",
            value=default_ticker_str,
            help="Enter one or more tickers separated by commas (e.g. AAPL, MSFT, GOOGL)"
        )
        if ticker_input and ticker_input.strip():
            param_values['tickers'] = [
                t.strip().upper() for t in ticker_input.split(",") if t.strip()
            ]
        else:
            param_values['tickers'] = []
    
    # Period selection
    period = st.selectbox(
        "Data Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "Custom"],
        index=3
    )
    
    if period == "Custom":
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        if start_date >= end_date:
            st.warning("Start date must be before end date.")
    else:
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
    _user = st.session_state.get('username', 'unknown')
    logger.info("[user=%s] Executing backtest: %s", _user, selected_name)

    st.markdown(_STRATEGY_SPINNER_CSS, unsafe_allow_html=True)
    manual_spinner = st.empty()
    manual_spinner.markdown(
        _spinner_html(0, f"Running {selected_name}…"),
        unsafe_allow_html=True,
    )

    try:
        strategy = strategy_cls()
        result = strategy.run(**param_values)
        st.session_state.backtest_result = result
        st.session_state.selected_strategy = selected_name

        # Update the cache so future switches are instant
        if 'backtest_cache' not in st.session_state:
            st.session_state.backtest_cache = {}
        st.session_state.backtest_cache[selected_name] = result

        manual_spinner.markdown(
            _spinner_html(100, f"{selected_name} complete "),
            unsafe_allow_html=True,
        )
        import time as _time; _time.sleep(0.6)
        manual_spinner.empty()

        if result.success:
            logger.info("[user=%s] Backtest completed successfully: %s",
                        _user, selected_name)
            _save_backtest_to_database(
                strategy_info, param_values, result, selected_name
            )
            minio_run_id = st.session_state.get('backtest_minio_run_id')
            if not minio_run_id:
                minio_run_id = _build_minio_run_id(
                    param_values.get('tickers', []),
                    market=st.session_state.get('current_market', 'US'),
                )
                st.session_state.backtest_minio_run_id = minio_run_id
            _save_charts_to_minio(
                run_id=minio_run_id,
                result=result,
                strategy_name=selected_name,
            )
        else:
            logger.warning("[user=%s] Backtest failed: %s — %s",
                           _user, selected_name, result.error_message)
            st.error(f"Failed: {result.error_message}")

    except Exception as e:
        manual_spinner.empty()
        logger.error(f"Error running backtest: {e}")
        st.error(f"Error: {str(e)}")


def _save_backtest_to_database(
    strategy_info: Dict,
    param_values: Dict,
    result: Any,
    selected_name: str
):
    """Save backtest results to database if available."""
    if not DB_AVAILABLE:
        return
    
    try:
        db_service = _get_db_service()
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
        
        if db_service.save_backtest_result(backtest_data, market=st.session_state.get('current_market', 'US')):
            _compact_caption("Results saved to database")
            
    except Exception as e:
        logger.error(f"Failed to save backtest to database: {e}")


def _build_minio_run_id(tickers: list, market: str = 'US') -> str:
    """
    Build a unique run_id with a short UUID + timestamp for MinIO storage.

    Args:
        tickers: List of ticker symbols (unused in ID but kept for API compat)
        market: 'US' or 'IND' — determines the prefix

    Returns:
        String like 'us_b080a824_20260218_163000' or 'in_b080a824_20260218_163000'
    """
    prefix = 'in' if market == 'IND' else 'us'
    short_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{short_id}_{ts}"


def _save_charts_to_minio(
    run_id: str,
    result: Any,
    strategy_name: str,
):
    """
    Save backtest chart images to MinIO object storage.

    Args:
        run_id: Shared run identifier (e.g. 'AAPL_TSLA_20260218_163000')
        result: StrategyResult containing charts
        strategy_name: Name of the strategy (used as subfolder)
    """
    if not MINIO_AVAILABLE:
        return 0

    try:
        minio_svc = _get_minio()
        if not minio_svc.is_available:
            return 0

        if not result.charts:
            return 0

        saved = minio_svc.save_backtest_charts(
            run_id=run_id,
            charts=result.charts,
            strategy_name=strategy_name,
        )

        count = len(saved) if saved else 0
        if count:
            logger.info("Saved %d chart(s) to MinIO for %s", count, strategy_name)
        else:
            logger.warning("No charts saved to MinIO for %s (charts=%d)",
                           strategy_name, len(result.charts))
        return count

    except Exception as e:
        logger.error(f"Failed to save charts to MinIO: {e}")
        st.warning(f"Could not save charts to MinIO: {e}")
        return 0


def _render_results_panel():
    """Render the backtest results panel with tabs for each strategy."""
    st.subheader("Results")

    cache = st.session_state.get('backtest_cache', {})

    # Fall back to single-result view when cache is empty
    if not cache:
        result = st.session_state.get('backtest_result')
        if result is None:
            st.info("Configure a strategy and click **Run Backtest** to see results")
            return
        _render_single_strategy_result(result)
        return

    # Determine default tab from currently selected strategy
    strategy_names = list(cache.keys())
    selected = st.session_state.get('selected_strategy', strategy_names[0])
    default_idx = strategy_names.index(selected) if selected in strategy_names else 0

    tabs = st.tabs(strategy_names)

    for tab, name in zip(tabs, strategy_names):
        with tab:
            result = cache[name]
            _render_single_strategy_result(result)


def _render_single_strategy_result(result):
    """Render metrics, charts, tables, and signals for a single strategy result."""
    if not result.success:
        st.error(f"Backtest failed: {result.error_message}")
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
        st.markdown("##### Aggregate Summary")
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
    
    # Separate scalar vs complex (dict/list) values
    scalar_items = []
    complex_items = []  # (label, value)
    
    # Collect available metrics in display order
    for key, label, fmt in display_keys:
        if key in t_metrics and t_metrics[key] is not None:
            val = t_metrics[key]
            if isinstance(val, (dict, list)):
                complex_items.append((label, val))
            else:
                scalar_items.append((label, val, fmt))
    
    # Also pick up any extra keys not in the predefined list
    shown_keys = {k for k, _, _ in display_keys}
    skip_keys = {'ticker', 'initial_capital', 'calculation_error'}
    for key, val in t_metrics.items():
        if key not in shown_keys and key not in skip_keys:
            label = key.replace('_', ' ').title()
            if isinstance(val, (dict, list)):
                complex_items.append((label, val))
            else:
                scalar_items.append((label, val, 'auto'))
    
    if not scalar_items and not complex_items:
        st.info("No metrics available")
        return
    
    # Render scalar metrics in rows of 4
    for row_start in range(0, len(scalar_items), 4):
        row = scalar_items[row_start:row_start + 4]
        cols = st.columns(len(row))
        for col, (label, val, fmt) in zip(cols, row):
            with col:
                st.metric(label, _format_metric_value(val, fmt))
    
    # Render complex (nested dict / list) values as expandable tables
    for label, val in complex_items:
        _render_complex_metric(label, val)


def _render_complex_metric(label: str, val):
    """Render a nested dict or list metric as an expandable section."""
    import pandas as pd
    with st.expander(f"{label}", expanded=False):
        if isinstance(val, list):
            if val and isinstance(val[0], dict):
                # List of dicts table
                df = pd.DataFrame(val)
                st.dataframe(df, width='stretch', hide_index=True)
            else:
                for item in val:
                    st.text(str(item))
        elif isinstance(val, dict):
            # Separate scalar sub-values from further nested ones
            scalar_pairs = []
            nested_pairs = []
            for k, v in val.items():
                if isinstance(v, (dict, list)):
                    nested_pairs.append((k, v))
                else:
                    scalar_pairs.append((k, v))
            # Show scalar pairs as a compact table
            if scalar_pairs:
                rows = []
                for k, v in scalar_pairs:
                    display_key = k.replace('_', ' ').title()
                    if isinstance(v, float):
                        if 'return' in k or 'drawdown' in k or 'pct' in k:
                            display_val = f"{v:.2f}%"
                        else:
                            display_val = f"{v:.4f}"
                    elif isinstance(v, bool):
                        display_val = "Yes" if v else "No"
                    else:
                        display_val = str(v)
                    rows.append({"Metric": display_key, "Value": display_val})
                df = pd.DataFrame(rows)
                st.dataframe(df, width='stretch', hide_index=True)
            # Recurse for nested items
            for k, v in nested_pairs:
                _render_complex_metric(k.replace('_', ' ').title(), v)
        else:
            st.text(str(val))


def _format_metric_value(val, fmt: str) -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return "Yes" if val else "No"
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
    
    scalar_items = []
    complex_items = []
    for k, v in flat_metrics.items():
        label = k.replace('_', ' ').title()
        if isinstance(v, (dict, list)):
            complex_items.append((label, v))
        else:
            scalar_items.append((label, v))
    
    # Render scalar metrics in rows of 4
    for row_start in range(0, len(scalar_items), 4):
        row = scalar_items[row_start:row_start + 4]
        cols = st.columns(len(row))
        for col, (name, val) in zip(cols, row):
            with col:
                if isinstance(val, bool):
                    st.metric(name, "Yes" if val else "No")
                elif isinstance(val, float):
                    if 'return' in name.lower() or 'drawdown' in name.lower():
                        st.metric(name, f"{val:.2f}%")
                    else:
                        st.metric(name, f"{val:.4f}")
                else:
                    st.metric(name, str(val))
    
    # Render complex values as expandable tables
    for label, val in complex_items:
        _render_complex_metric(label, val)


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
                st.image(base64.b64decode(img_data), width='stretch')
            except Exception as e:
                st.warning(f"Could not display chart: {e}")
                
        elif chart.chart_type == 'plotly':
            try:
                fig = go.Figure(json.loads(chart.data))
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.warning(f"Could not display chart: {e}")


def _render_tables(tables: list):
    """Render backtest data tables."""
    st.markdown("---")
    st.markdown("#### Data Tables")
    
    for table in tables:
        st.caption(table.title)
        if table.data:
            st.dataframe(pd.DataFrame(table.data), width='stretch')


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
