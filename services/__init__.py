"""
Services package for Centurion Capital LLC.

61 modules organised into logical domains.  Submodules are imported
directly where needed (``from services.carver_pipeline import CarverPipeline``).
The convenience re-exports below cover the 12 most-used entry points so
callers can also write ``from services import CarverPipeline``.

Module map
----------

**Core pipeline**
  carver_pipeline          – End-to-end Carver systematic trading pipeline
  us_carver_pipeline       – US-equity variant of the Carver pipeline
  full_pipeline_backtest   – 23-source production backtest harness

**Forecasting**
  forecast_scalar          – Normalise signals to Carver ±20 scale
  forecast_combiner        – Weighted combination with FDM (Carver Ch.8)
  integrated_scorer        – Multi-layer stock verdict engine
  sentiment_forecast       – FinBERT news sentiment → forecast

**Regime & risk**
  regime_detector          – Online 5-state regime detection
  regime_hmm              – Hamilton HMM 3-state probabilistic regime
  regime_performance       – Per-regime strategy attribution
  regime_strategy_mix      – Regime-conditional forecast weights
  risk_metrics             – Sortino, Calmar, Omega, CVaR, GEFR
  monte_carlo_risk         – Trade-level bootstrap for ruin probability
  tail_risk_hedge          – Portfolio-level NIFTY put hedging

**Position sizing & portfolio**
  position_sizer           – Volatility-targeted Carver Ch.11 sizing
  volatility_target        – Portfolio annual risk budget
  instrument_volatility    – 35-day EWMA vol estimation
  instrument_weights       – IDM from correlation matrix
  hrp_allocator            – Hierarchical risk parity (AFML Ch.16)
  portfolio_analyzer       – Real-time allocation & concentration
  portfolio_vol_monitor    – Portfolio vol deviation tracking

**Validation & statistics**
  aronson_validator        – EBTA statistical validation
  deflated_sharpe          – De Prado deflated Sharpe ratio
  mc_permutation_test      – Timothy Masters permutation test
  prediction_quality       – ROC, confusion matrices, bootstrap CIs
  signal_quality_evaluator – Regime-conditioned signal quality & CAGR
  walk_forward             – Rolling OOS walk-forward validation

**Market data**
  bhavcopy_fetcher         – NSE equity OHLCV from bhavcopy archives
  nse_fo_bhavcopy          – F&O bhavcopy for open interest
  delivery_volume          – NSE delivery % analysis
  fii_flow_signal          – FII/DII daily flow forecast
  corporate_actions        – Split/bonus/dividend adjustments
  survivorship_filter      – Delisted/dead stock detection

**Strategy modules**
  earnings_momentum        – Post-earnings momentum
  event_calendar           – Aggregate macro/micro events
  event_strategy           – Event-driven signals
  factor_momentum          – Dynamic factor weighting
  momentum_factor          – 12-minus-1 cross-sectional momentum
  sector_momentum          – Sector momentum scoring
  sector_rotation          – NIFTY sector ranking
  pead_strategy            – Post-earnings announcement drift
  pairs_trading_live       – Mean-reversion z-score pairs
  futures_overlay          – Regime-adaptive NIFTY/BANKNIFTY leverage
  options_overlay          – Systematic covered call / CSP
  iv_rank                  – Implied volatility rank percentile
  oi_signal                – F&O OI-based directional signal

**Execution & monitoring**
  cost_speed_limit         – Carver Ch.12 cost-aware trading gate
  vol_trailing_stop        – Volatility-aware adaptive stop loss
  strategy_decay           – Rolling Sharpe degradation monitor
  strategy_tournament      – Monthly cross-strategy competition
  vince_leverage_space     – Optimal-f & leverage-space optimisation
  vince_metrics            – HPR / TWR per-instrument tracking
  benchmark_tracker        – NIFTY 50 alpha comparison

**Infrastructure**
  analysis                 – Multi-ticker async analysis runner
  cache                    – TTL-aware in-process session cache
  session                  – Streamlit session state init
  meta_labeling            – AFML secondary classifier
  fundamental_freshness    – Intra-quarter fundamental proxies
  carver_calibration       – Expanding-window forecast calibration
  penfold_backtest         – 17-source Penfold trend backtest

**Sub-packages** (see their own ``__init__.py``)
  decision_engine/         – Rules engine for trade decisions
  layers/                  – Execution, market data, risk, portfolio layers
  metrics/                 – Technical indicator calculators
  notifications/           – Email / webhook / SMS alerts
  rl_bot/                  – Reinforcement-learning trade bot
  sentiment/               – Multi-source sentiment aggregation
  storage/                 – MinIO / local file storage
  technical_analysis/      – RSI, MACD, Bollinger, Supertrend aggregator
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Convenience re-exports (lazy – only resolved when accessed)
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    """Lazy import for frequently-used symbols."""
    _LAZY = {
        "CarverPipeline": ".carver_pipeline",
        "IntegratedScorer": ".integrated_scorer",
        "AronsonValidator": ".aronson_validator",
        "combine_forecasts": ".forecast_combiner",
        "compute_position_size": ".position_sizer",
        "compare_to_benchmark": ".benchmark_tracker",
        "run_full_backtest": ".full_pipeline_backtest",
        "run_full_evaluation": ".signal_quality_evaluator",
        "walk_forward_validate": ".walk_forward",
        "hrp_weights": ".hrp_allocator",
        "RiskMetrics": ".risk_metrics",
        "MarketRegime": ".regime_detector",
    }
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CarverPipeline",
    "IntegratedScorer",
    "AronsonValidator",
    "combine_forecasts",
    "compute_position_size",
    "compare_to_benchmark",
    "run_full_backtest",
    "run_full_evaluation",
    "walk_forward_validate",
    "hrp_weights",
    "RiskMetrics",
    "MarketRegime",
]
