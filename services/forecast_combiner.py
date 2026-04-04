"""
Forecast Combiner — Weighted combination of trading rules with FDM.

Implements Carver Chapter 8 'Combined Forecasts':

1. Each trading rule variation produces a forecast (avg abs ≈ 10, ±20).
2. Forecasts are combined via a weighted average.
3. The combined forecast is multiplied by a Forecast Diversification
   Multiplier (FDM) to restore the average absolute value to ≈ 10.
4. The final combined forecast is capped at ±20.

Forecast Weights (handcrafted, Carver Table 45):
  - For swing/positional equity with 3 EWMAC + 1 Carry + 1 Screener:
    EWMAC(16,64)  : 22%   (swing core)
    EWMAC(32,128) : 17%   (lower weight — highly correlated with neighbours)
    EWMAC(64,256) : 22%   (positional core)
    Carry         : 22%   (decorrelated fundamental signal)
    Screener      : 17%   (existing technical/methodology overlay)
  - Weights sum to 100%.

FDM Calculation:
  - From correlation matrix of forecast returns (Carver Table 18).
  - With 5 rules at avg correlation ~0.35, FDM ≈ 1.35.
  - Capped at 2.0 maximum.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from services.forecast_scalar import cap_forecast, TARGET_ABS_FORECAST

logger = logging.getLogger(__name__)

# Aronson EBTA validation weight multipliers (loaded lazily)
_aronson_weight_multipliers: Optional[Dict[str, float]] = None
_aronson_multipliers_loaded: bool = False

# Strategy decay state — lazy-loaded, maps source → status/sharpe
_decay_state: Optional[Dict[str, Dict]] = None
_decay_state_loaded: bool = False
_DECAY_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "strategy_decay_state.json"

# Statuses that should receive zero weight (actively losing money)
_DEAD_STATUSES = {"INVERTED", "DEAD"}

# Maximum FDM to prevent extreme positions (Carver recommendation)
MAX_FDM = 2.0


@dataclass
class ForecastWeight:
    """Weight for a single forecast source."""
    name: str
    weight: float  # 0.0 to 1.0


# R7 MODERATE CONCENTRATION: 8 proven signals.
# Diagnosis across 6 runs: 17 signals diluted forecast to ~4 (R4), 5 signals had
# too much variance causing -50% in 400 days (R5/R6). 8 signals balances:
# avg forecast ~6-7 (vs Carver target 10), sufficient diversification to avoid
# single-signal blowups, decorrelated styles (trend + momentum + adaptive + breakout).
DEFAULT_FORECAST_WEIGHTS: List[ForecastWeight] = [
    ForecastWeight("ewmac_8_32", 0.10),      # R14/R18: fast trend
    ForecastWeight("ewmac_16_64", 0.12),     # R14/R18: core swing trend
    ForecastWeight("ewmac_32_128", 0.00),    # zeroed — redundant
    ForecastWeight("ewmac_64_256", 0.10),    # R14/R18: positional trend
    ForecastWeight("carry", 0.00),           # zeroed — weak for equities
    ForecastWeight("screener", 0.07),        # R14/R18: RSI+MA mixed
    ForecastWeight("momentum", 0.16),        # R14/R18: primary trend alpha
    ForecastWeight("pead", 0.00),            # DEAD: 0% hit rate
    ForecastWeight("mean_reversion", 0.08),  # R14/R18: counter-trend diversifier
    ForecastWeight("fii_flow", 0.00),        # DEAD: 0% hit rate
    ForecastWeight("decision_engine", 0.00), # zeroed — circular dependency
    ForecastWeight("oi_signal", 0.00),       # HARMFUL: t-stat = -69.8
    ForecastWeight("cross_momentum", 0.00),  # zeroed — conflicts with long-only
    ForecastWeight("pairs_arb", 0.00),       # HARMFUL: t-stat = -190.7
    ForecastWeight("event_driven", 0.00),    # DEAD: 0% hit rate
    ForecastWeight("penfold_trend", 0.12),   # R14/R18: Turtle+ATR
    ForecastWeight("ehlers_dsp", 0.12),      # R14/R18: adaptive DSP
    ForecastWeight("intermarket", 0.00),     # zeroed — noisy
    ForecastWeight("acceleration", 0.06),    # R14/R18: trend rate-of-change
    ForecastWeight("carver_value", 0.00),    # R18: removed (R15-R17 experiment failed)
    ForecastWeight("skew_signal", 0.00),     # zeroed — weak signal
    ForecastWeight("sentiment", 0.00),       # DEAD: 0% hit rate
    ForecastWeight("breakout", 0.07),        # R14/R18: 20-day channel
    ForecastWeight("order_flow", 0.00),      # zeroed — microstructure noise
    # Total: 1.00 exact (24 sources, 10 active)
    # R18 = exact R14 weights (reverted from R15-R17 experiments)
    # Trend (92%): momentum(16%), ewmac_16_64(12%), penfold_trend(12%),
    #              ehlers_dsp(12%), ewmac_64_256(10%), ewmac_8_32(10%),
    #              breakout(7%), acceleration(6%), screener(7%)
    # Counter (8%): mean_reversion(8%)
]

# Rule-of-thumb correlations between forecast sources (Carver Appendix C):
#   Same rule, different speed:     ~0.7–0.9
#   Different rules, same style:    ~0.5
#   Different styles:               ~0.25
DEFAULT_CORRELATION_MATRIX = {
    ("ewmac_16_64", "ewmac_32_128"): 0.90,
    ("ewmac_16_64", "ewmac_64_256"): 0.60,
    ("ewmac_32_128", "ewmac_64_256"): 0.90,
    ("ewmac_16_64", "carry"): 0.25,
    ("ewmac_32_128", "carry"): 0.25,
    ("ewmac_64_256", "carry"): 0.25,
    ("ewmac_16_64", "screener"): 0.50,
    ("ewmac_32_128", "screener"): 0.50,
    ("ewmac_64_256", "screener"): 0.50,
    ("carry", "screener"): 0.30,
    # Momentum correlations (Phase 1)
    ("ewmac_16_64", "momentum"): 0.55,
    ("ewmac_32_128", "momentum"): 0.50,
    ("ewmac_64_256", "momentum"): 0.45,
    ("carry", "momentum"): 0.20,
    ("screener", "momentum"): 0.40,
    # PEAD correlations (Phase 1) — low corr with trend-following
    ("ewmac_16_64", "pead"): 0.15,
    ("ewmac_32_128", "pead"): 0.15,
    ("ewmac_64_256", "pead"): 0.15,
    ("carry", "pead"): 0.10,
    ("screener", "pead"): 0.20,
    ("momentum", "pead"): 0.25,
    # Mean-reversion correlations (Gap A2) — negatively correlated with trend
    ("ewmac_16_64", "mean_reversion"): -0.30,
    ("ewmac_32_128", "mean_reversion"): -0.25,
    ("ewmac_64_256", "mean_reversion"): -0.15,
    ("carry", "mean_reversion"): 0.10,
    ("screener", "mean_reversion"): 0.20,
    ("momentum", "mean_reversion"): -0.20,
    ("pead", "mean_reversion"): 0.05,
    # FII flow correlations (Gap A5) — same signal for all stocks
    ("ewmac_16_64", "fii_flow"): 0.30,
    ("ewmac_32_128", "fii_flow"): 0.25,
    ("ewmac_64_256", "fii_flow"): 0.20,
    ("carry", "fii_flow"): 0.15,
    ("screener", "fii_flow"): 0.20,
    ("momentum", "fii_flow"): 0.35,
    ("pead", "fii_flow"): 0.10,
    ("mean_reversion", "fii_flow"): 0.05,
    # OI signal removed (G8) — correlations kept for backward compat if re-enabled
    # Decision engine correlations (Gap B6)
    ("ewmac_16_64", "decision_engine"): 0.30,
    ("ewmac_32_128", "decision_engine"): 0.25,
    ("ewmac_64_256", "decision_engine"): 0.20,
    ("carry", "decision_engine"): 0.25,
    ("screener", "decision_engine"): 0.60,
    ("momentum", "decision_engine"): 0.35,
    ("pead", "decision_engine"): 0.15,
    ("mean_reversion", "decision_engine"): 0.20,
    ("fii_flow", "decision_engine"): 0.15,
    # Phase 4: Pairs arb correlations — highly uncorrelated with directional signals
    ("ewmac_16_64", "pairs_arb"): 0.05,
    ("ewmac_32_128", "pairs_arb"): 0.05,
    ("ewmac_64_256", "pairs_arb"): 0.05,
    ("carry", "pairs_arb"): 0.05,
    ("screener", "pairs_arb"): 0.10,
    ("momentum", "pairs_arb"): 0.00,
    ("pead", "pairs_arb"): 0.05,
    ("mean_reversion", "pairs_arb"): 0.30,
    ("fii_flow", "pairs_arb"): 0.00,
    ("decision_engine", "pairs_arb"): 0.10,
    # Phase 4: Event-driven correlations — episodic, low correlation with everything
    ("ewmac_16_64", "event_driven"): 0.10,
    ("ewmac_32_128", "event_driven"): 0.10,
    ("ewmac_64_256", "event_driven"): 0.10,
    ("carry", "event_driven"): 0.05,
    ("screener", "event_driven"): 0.15,
    ("momentum", "event_driven"): 0.10,
    ("pead", "event_driven"): 0.20,
    ("mean_reversion", "event_driven"): 0.05,
    ("fii_flow", "event_driven"): 0.10,
    ("decision_engine", "event_driven"): 0.15,
    ("pairs_arb", "event_driven"): 0.05,
    # G19: OI signal correlations — moderately correlated with momentum/trend
    ("ewmac_16_64", "oi_signal"): 0.30,
    ("ewmac_32_128", "oi_signal"): 0.25,
    ("ewmac_64_256", "oi_signal"): 0.20,
    ("carry", "oi_signal"): 0.10,
    ("screener", "oi_signal"): 0.25,
    ("momentum", "oi_signal"): 0.35,
    ("pead", "oi_signal"): 0.10,
    ("mean_reversion", "oi_signal"): 0.15,
    ("fii_flow", "oi_signal"): 0.20,
    ("decision_engine", "oi_signal"): 0.15,
    ("pairs_arb", "oi_signal"): 0.05,
    ("event_driven", "oi_signal"): 0.10,
    # EWMAC 8_32 correlations — fastest variation, high corr with 16_64
    ("ewmac_8_32", "ewmac_16_64"): 0.90,
    ("ewmac_8_32", "ewmac_32_128"): 0.60,
    ("ewmac_8_32", "ewmac_64_256"): 0.40,
    ("ewmac_8_32", "carry"): 0.20,
    ("ewmac_8_32", "screener"): 0.40,
    ("ewmac_8_32", "momentum"): 0.50,
    ("ewmac_8_32", "pead"): 0.10,
    ("ewmac_8_32", "mean_reversion"): -0.35,
    ("ewmac_8_32", "fii_flow"): 0.30,
    ("ewmac_8_32", "decision_engine"): 0.25,
    ("ewmac_8_32", "oi_signal"): 0.30,
    ("ewmac_8_32", "pairs_arb"): 0.05,
    ("ewmac_8_32", "event_driven"): 0.10,
    # Breakout correlations — 20-day high/low channel
    ("breakout", "ewmac_8_32"): 0.55,
    ("breakout", "ewmac_16_64"): 0.50,
    ("breakout", "ewmac_32_128"): 0.40,
    ("breakout", "ewmac_64_256"): 0.30,
    ("breakout", "carry"): 0.15,
    ("breakout", "screener"): 0.30,
    ("breakout", "momentum"): 0.45,
    ("breakout", "pead"): 0.10,
    ("breakout", "mean_reversion"): -0.25,
    ("breakout", "fii_flow"): 0.20,
    ("breakout", "decision_engine"): 0.20,
    ("breakout", "oi_signal"): 0.25,
    ("breakout", "pairs_arb"): 0.05,
    ("breakout", "event_driven"): 0.10,
    # Cross-sectional momentum — ranks stocks by relative performance
    ("cross_momentum", "ewmac_8_32"): 0.35,
    ("cross_momentum", "ewmac_16_64"): 0.40,
    ("cross_momentum", "ewmac_32_128"): 0.45,
    ("cross_momentum", "ewmac_64_256"): 0.50,
    ("cross_momentum", "carry"): 0.15,
    ("cross_momentum", "screener"): 0.25,
    ("cross_momentum", "momentum"): 0.60,
    ("cross_momentum", "pead"): 0.15,
    ("cross_momentum", "mean_reversion"): -0.30,
    ("cross_momentum", "fii_flow"): 0.20,
    ("cross_momentum", "decision_engine"): 0.20,
    ("cross_momentum", "oi_signal"): 0.20,
    ("cross_momentum", "breakout"): 0.45,
    ("cross_momentum", "pairs_arb"): 0.10,
    ("cross_momentum", "event_driven"): 0.10,
    ("ewmac_8_32", "breakout"): 0.55,
    # Penfold trend — composite: Turtle + ATR band + retracement + weekly Dow
    ("penfold_trend", "ewmac_8_32"): 0.50,
    ("penfold_trend", "ewmac_16_64"): 0.55,
    ("penfold_trend", "ewmac_32_128"): 0.50,
    ("penfold_trend", "ewmac_64_256"): 0.45,
    ("penfold_trend", "carry"): 0.10,
    ("penfold_trend", "screener"): 0.20,
    ("penfold_trend", "momentum"): 0.45,
    ("penfold_trend", "pead"): 0.10,
    ("penfold_trend", "mean_reversion"): -0.20,
    ("penfold_trend", "fii_flow"): 0.15,
    ("penfold_trend", "decision_engine"): 0.15,
    ("penfold_trend", "oi_signal"): 0.15,
    ("penfold_trend", "breakout"): 0.65,
    ("penfold_trend", "cross_momentum"): 0.50,
    ("penfold_trend", "pairs_arb"): 0.05,
    ("penfold_trend", "event_driven"): 0.05,
    # Ehlers DSP — adaptive filters, low corr with EWMAC (different approach)
    ("ehlers_dsp", "ewmac_8_32"): 0.35,
    ("ehlers_dsp", "ewmac_16_64"): 0.30,
    ("ehlers_dsp", "ewmac_32_128"): 0.25,
    ("ehlers_dsp", "ewmac_64_256"): 0.20,
    ("ehlers_dsp", "carry"): 0.10,
    ("ehlers_dsp", "screener"): 0.25,
    ("ehlers_dsp", "momentum"): 0.30,
    ("ehlers_dsp", "pead"): 0.10,
    ("ehlers_dsp", "mean_reversion"): -0.15,
    ("ehlers_dsp", "fii_flow"): 0.15,
    ("ehlers_dsp", "decision_engine"): 0.20,
    ("ehlers_dsp", "oi_signal"): 0.20,
    ("ehlers_dsp", "breakout"): 0.40,
    ("ehlers_dsp", "cross_momentum"): 0.25,
    ("ehlers_dsp", "pairs_arb"): 0.05,
    ("ehlers_dsp", "event_driven"): 0.10,
    ("ehlers_dsp", "penfold_trend"): 0.45,
    # Intermarket (Ruggiero) — macro-driven, low corr with stock-specific signals
    ("intermarket", "ewmac_8_32"): 0.15,
    ("intermarket", "ewmac_16_64"): 0.15,
    ("intermarket", "ewmac_32_128"): 0.15,
    ("intermarket", "ewmac_64_256"): 0.20,
    ("intermarket", "carry"): 0.20,
    ("intermarket", "screener"): 0.15,
    ("intermarket", "momentum"): 0.20,
    ("intermarket", "pead"): 0.05,
    ("intermarket", "mean_reversion"): 0.10,
    ("intermarket", "fii_flow"): 0.40,
    ("intermarket", "decision_engine"): 0.15,
    ("intermarket", "oi_signal"): 0.20,
    ("intermarket", "breakout"): 0.15,
    ("intermarket", "cross_momentum"): 0.15,
    ("intermarket", "pairs_arb"): 0.05,
    ("intermarket", "event_driven"): 0.15,
    ("intermarket", "penfold_trend"): 0.20,
    ("intermarket", "ehlers_dsp"): 0.15,
    # --- AFTS S23: Acceleration — derivative of EWMAC, high corr with trend ---
    ("acceleration", "ewmac_8_32"): 0.55,
    ("acceleration", "ewmac_16_64"): 0.50,
    ("acceleration", "ewmac_32_128"): 0.45,
    ("acceleration", "ewmac_64_256"): 0.35,
    ("acceleration", "carry"): 0.10,
    ("acceleration", "screener"): 0.25,
    ("acceleration", "momentum"): 0.40,
    ("acceleration", "pead"): 0.10,
    ("acceleration", "mean_reversion"): -0.20,
    ("acceleration", "fii_flow"): 0.15,
    ("acceleration", "decision_engine"): 0.15,
    ("acceleration", "oi_signal"): 0.20,
    ("acceleration", "breakout"): 0.40,
    ("acceleration", "cross_momentum"): 0.35,
    ("acceleration", "pairs_arb"): 0.05,
    ("acceleration", "event_driven"): 0.10,
    ("acceleration", "penfold_trend"): 0.45,
    ("acceleration", "ehlers_dsp"): 0.30,
    ("acceleration", "intermarket"): 0.15,
    # --- AFTS S22: Value — 5-year mean reversion, anti-correlated with trend ---
    ("carver_value", "ewmac_8_32"): -0.10,
    ("carver_value", "ewmac_16_64"): -0.05,
    ("carver_value", "ewmac_32_128"): 0.00,
    ("carver_value", "ewmac_64_256"): 0.10,
    ("carver_value", "carry"): 0.25,
    ("carver_value", "screener"): 0.15,
    ("carver_value", "momentum"): -0.30,
    ("carver_value", "pead"): 0.15,
    ("carver_value", "mean_reversion"): 0.55,
    ("carver_value", "fii_flow"): 0.05,
    ("carver_value", "decision_engine"): 0.10,
    ("carver_value", "oi_signal"): 0.05,
    ("carver_value", "breakout"): -0.15,
    ("carver_value", "cross_momentum"): -0.25,
    ("carver_value", "pairs_arb"): 0.20,
    ("carver_value", "event_driven"): 0.05,
    ("carver_value", "penfold_trend"): -0.10,
    ("carver_value", "ehlers_dsp"): -0.05,
    ("carver_value", "intermarket"): 0.10,
    ("carver_value", "acceleration"): -0.15,
    # --- AFTS S24: Skew — structural risk premium, low corr with everything ---
    ("skew_signal", "ewmac_8_32"): 0.10,
    ("skew_signal", "ewmac_16_64"): 0.10,
    ("skew_signal", "ewmac_32_128"): 0.10,
    ("skew_signal", "ewmac_64_256"): 0.10,
    ("skew_signal", "carry"): 0.15,
    ("skew_signal", "screener"): 0.05,
    ("skew_signal", "momentum"): 0.05,
    ("skew_signal", "pead"): 0.05,
    ("skew_signal", "mean_reversion"): 0.15,
    ("skew_signal", "fii_flow"): 0.05,
    ("skew_signal", "decision_engine"): 0.05,
    ("skew_signal", "oi_signal"): 0.10,
    ("skew_signal", "breakout"): 0.05,
    ("skew_signal", "cross_momentum"): 0.05,
    ("skew_signal", "pairs_arb"): 0.10,
    ("skew_signal", "event_driven"): 0.05,
    ("skew_signal", "penfold_trend"): 0.10,
    ("skew_signal", "ehlers_dsp"): 0.05,
    ("skew_signal", "intermarket"): 0.05,
    ("skew_signal", "acceleration"): 0.10,
    ("skew_signal", "carver_value"): 0.30,
    # --- Sentiment correlations ---
    # Sentiment is news-driven, low overlap with technical signals
    ("sentiment", "ewmac_8_32"): 0.10,
    ("sentiment", "ewmac_16_64"): 0.10,
    ("sentiment", "ewmac_32_128"): 0.10,
    ("sentiment", "ewmac_64_256"): 0.10,
    ("sentiment", "carry"): 0.05,
    ("sentiment", "screener"): 0.20,
    ("sentiment", "momentum"): 0.15,
    ("sentiment", "pead"): 0.30,       # High: both react to earnings/news events
    ("sentiment", "mean_reversion"): 0.10,
    ("sentiment", "fii_flow"): 0.20,   # Moderate: both reflect institutional views
    ("sentiment", "decision_engine"): 0.15,
    ("sentiment", "oi_signal"): 0.10,
    ("sentiment", "breakout"): 0.10,
    ("sentiment", "cross_momentum"): 0.10,
    ("sentiment", "pairs_arb"): 0.05,
    ("sentiment", "event_driven"): 0.35,  # High: both are event-driven
    ("sentiment", "penfold_trend"): 0.10,
    ("sentiment", "ehlers_dsp"): 0.05,
    ("sentiment", "intermarket"): 0.15,
    ("sentiment", "acceleration"): 0.10,
    ("sentiment", "carver_value"): 0.10,
    ("sentiment", "skew_signal"): 0.05,
}


@dataclass
class CombinedForecast:
    """Result of the forecast combination for one instrument."""
    symbol: str
    combined_forecast: float       # final capped forecast (-20 to +20)
    raw_combined: float            # before FDM and capping
    fdm: float                     # forecast diversification multiplier
    individual_forecasts: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    sources_available: int = 0
    sources_total: int = 0
    confidence_score: float = 1.0  # Aronson: fraction of validated signals agreeing on direction


def compute_rolling_correlations(
    forecast_history: Dict[str, list],
    lookback: int = 252,
    shrinkage: float = 0.3,
) -> Dict[tuple, float]:
    """Compute rolling pairwise correlations from forecast history.

    Uses shrinkage toward the static prior (DEFAULT_CORRELATION_MATRIX)
    to stabilize estimates when history is short.

    Parameters
    ----------
    forecast_history : dict[str, list[float]]
        {source_name: [daily_forecast_values]}.
    lookback : int
        Rolling window in trading days.
    shrinkage : float
        Blend factor: (1-shrinkage)*empirical + shrinkage*prior.

    Returns
    -------
    dict[tuple, float]
        Pairwise correlations {(source_a, source_b): corr}.
    """
    sources = sorted(forecast_history.keys())
    n = len(sources)
    if n < 2:
        return DEFAULT_CORRELATION_MATRIX

    # Build matrix of recent forecasts
    min_len = min(len(forecast_history[s]) for s in sources)
    usable = min(min_len, lookback)
    if usable < 30:
        return DEFAULT_CORRELATION_MATRIX

    data = np.column_stack([
        np.array(forecast_history[s][-usable:], dtype=float) for s in sources
    ])

    # Empirical correlation
    with np.errstate(divide='ignore', invalid='ignore'):
        empirical = np.corrcoef(data.T)
    if not np.all(np.isfinite(empirical)):
        return DEFAULT_CORRELATION_MATRIX

    # Shrink toward static prior
    result = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = (sources[i], sources[j])
            rev_key = (sources[j], sources[i])
            prior = DEFAULT_CORRELATION_MATRIX.get(
                key, DEFAULT_CORRELATION_MATRIX.get(rev_key, 0.0)
            )
            emp = float(empirical[i, j])
            blended = (1 - shrinkage) * emp + shrinkage * prior
            blended = max(-0.95, min(0.95, blended))
            result[key] = round(blended, 3)

    logger.info(
        "Dynamic correlations computed: %d pairs from %d-day window (shrinkage=%.1f)",
        len(result), usable, shrinkage,
    )
    return result


def compute_fdm(
    weights: Dict[str, float],
    correlations: Optional[Dict[tuple, float]] = None,
) -> float:
    """Compute the Forecast Diversification Multiplier.

    FDM = 1 / sqrt(w' × C × w)
    where w = weight vector, C = correlation matrix.

    This ensures the combined forecast has the same expected
    absolute value as individual forecasts (≈ 10).

    Parameters
    ----------
    weights : dict[str, float]
        Forecast weights (must sum to ~1.0).
    correlations : dict[tuple, float]
        Pairwise correlations {(name_a, name_b): corr}.

    Returns
    -------
    float
        FDM, capped at MAX_FDM.
    """
    correlations = correlations or DEFAULT_CORRELATION_MATRIX
    names = sorted(weights.keys())
    n = len(names)
    if n <= 1:
        return 1.0

    # Build weight vector and correlation matrix
    w = np.array([weights[name] for name in names])
    C = np.eye(n)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i != j:
                key = (ni, nj) if (ni, nj) in correlations else (nj, ni)
                C[i, j] = correlations.get(key, 0.0)

    # FDM = 1 / sqrt(w' C w)
    portfolio_var = float(w @ C @ w)
    if portfolio_var <= 0:
        return 1.0

    fdm = 1.0 / math.sqrt(portfolio_var)
    fdm = min(fdm, MAX_FDM)
    return round(fdm, 3)


def _load_decay_state() -> Dict[str, Dict]:
    """Lazy-load strategy decay state from data/strategy_decay_state.json.

    Returns a dict like {"ewmac": {"status": "INVERTED", "recent_sharpe": -0.306}, ...}.
    Empty dict on failure.
    """
    global _decay_state, _decay_state_loaded
    if _decay_state_loaded:
        return _decay_state or {}
    _decay_state_loaded = True
    try:
        if _DECAY_STATE_PATH.exists():
            _decay_state = json.loads(_DECAY_STATE_PATH.read_text())
            logger.info(
                "Loaded strategy decay state: %d entries (%s)",
                len(_decay_state),
                ", ".join(f"{k}={v.get('status')}" for k, v in _decay_state.items()),
            )
        else:
            _decay_state = {}
    except Exception as exc:
        logger.warning("Failed to load strategy decay state: %s", exc)
        _decay_state = {}
    return _decay_state


def apply_decay_state_filter(
    base_weights: List[ForecastWeight],
) -> List[ForecastWeight]:
    """G1 FIX (revised P1): Regime-conditional decay filtering.

    Instead of blanket-zeroing all EWMAC variations when the generic "ewmac"
    key is INVERTED, this now:
    1. Only zeroes sources whose EXACT name matches a decay key (e.g. "carry")
    2. For prefix matches (e.g. "ewmac" → "ewmac_8_32"), DOWNGRADES weight
       to 25% instead of zeroing — lets regime_strategy_mix handle the rest
    3. DEAD status still gets full zero (strategy is truly broken, not just
       regime-inappropriate)

    This fixes the ~10% CAGR loss from zeroing 25% of forecast weight when
    trend-following is merely in a range-bound regime (not actually broken).
    """
    decay = _load_decay_state()
    if not decay:
        return base_weights

    adjusted = []
    zeroed_sources: List[str] = []
    downgraded_sources: List[str] = []
    for fw in base_weights:
        source_status = None
        match_type = None  # "exact" or "prefix"
        for decay_key, decay_info in decay.items():
            if fw.name == decay_key:
                source_status = decay_info.get("status", "").upper()
                match_type = "exact"
                break
            elif fw.name.startswith(decay_key + "_"):
                source_status = decay_info.get("status", "").upper()
                match_type = "prefix"
                break

        if source_status == "DEAD":
            # Truly broken — full zero regardless of match type
            adjusted.append(ForecastWeight(fw.name, 0.0))
            zeroed_sources.append(fw.name)
        elif source_status == "INVERTED" and match_type == "exact":
            # Exact match INVERTED (e.g. "carry") — zero it
            adjusted.append(ForecastWeight(fw.name, 0.0))
            zeroed_sources.append(fw.name)
        elif source_status == "INVERTED" and match_type == "prefix":
            # Prefix match INVERTED (e.g. "ewmac" → "ewmac_8_32")
            # Downgrade to 25% weight — let regime_strategy_mix dynamically adjust
            adjusted.append(ForecastWeight(fw.name, fw.weight * 0.25))
            downgraded_sources.append(fw.name)
        else:
            adjusted.append(ForecastWeight(fw.name, fw.weight))

    # Renormalise non-zero weights to sum to 1.0
    total = sum(fw.weight for fw in adjusted)
    if total > 0:
        adjusted = [
            ForecastWeight(fw.name, fw.weight / total) if fw.weight > 0
            else fw
            for fw in adjusted
        ]

    if zeroed_sources:
        logger.warning(
            "G1: Zeroed %d inverted/dead forecast sources: %s",
            len(zeroed_sources), zeroed_sources,
        )
    if downgraded_sources:
        logger.info(
            "P1: Downgraded %d INVERTED prefix-matched sources to 25%% weight "
            "(regime_strategy_mix will further adjust): %s",
            len(downgraded_sources), downgraded_sources,
        )

    return adjusted


# ── P4: Regime-Specific Sharpe²-Weighted Forecast Allocation ──────────
# Historical signal quality by regime (from 13yr audit, April 2026):
#   BULL:     Sharpe 0.73 (10D) — trend-following strongest
#   SIDEWAYS: Sharpe 0.80 (5D)  — mean-reversion & adaptive strongest
#   BEAR:     Sharpe 0.10 (10D) — signals broken, near-random
#
# Sharpe² weighting: w_i ∝ sharpe_i² in that regime (Kelly-optimal allocation)
# This tilts forecast ensemble toward historically proven sources per regime.

REGIME_SHARPE_SCORES = {
    # {source: {regime: sharpe_estimate}} — from backtest signal quality audit
    "ewmac_8_32":      {"bull": 0.65, "sideways": 0.40, "bear": 0.05},
    "ewmac_16_64":     {"bull": 0.70, "sideways": 0.35, "bear": 0.08},
    "ewmac_32_128":    {"bull": 0.72, "sideways": 0.30, "bear": 0.10},
    "ewmac_64_256":    {"bull": 0.68, "sideways": 0.25, "bear": 0.12},
    "ehlers_dsp":      {"bull": 0.60, "sideways": 0.75, "bear": 0.15},
    "momentum":        {"bull": 0.55, "sideways": 0.30, "bear": 0.05},
    "intermarket":     {"bull": 0.50, "sideways": 0.45, "bear": 0.20},
    "penfold_trend":   {"bull": 0.65, "sideways": 0.35, "bear": 0.08},
    "cross_momentum":  {"bull": 0.50, "sideways": 0.25, "bear": 0.10},
    "screener":        {"bull": 0.40, "sideways": 0.50, "bear": 0.10},
    "acceleration":    {"bull": 0.55, "sideways": 0.30, "bear": 0.05},
    "breakout":        {"bull": 0.60, "sideways": 0.55, "bear": 0.08},
    "skew_signal":     {"bull": 0.30, "sideways": 0.40, "bear": 0.25},
    "decision_engine": {"bull": 0.35, "sideways": 0.45, "bear": 0.10},
    "carver_value":    {"bull": 0.20, "sideways": 0.55, "bear": 0.15},
    "carry":           {"bull": 0.25, "sideways": 0.30, "bear": 0.10},
    "order_flow":      {"bull": 0.30, "sideways": 0.35, "bear": 0.10},
}


def apply_regime_sharpe_weights(
    base_weights: List[ForecastWeight],
    regime: str = "",
    blend_factor: float = 0.5,
) -> List[ForecastWeight]:
    """P4: Tilt forecast weights toward historically strong sources for current regime.

    Uses Sharpe² weighting (Kelly-optimal) blended with base weights.
    blend_factor=0.5 means 50% base + 50% Sharpe²-optimised.

    Parameters
    ----------
    base_weights : list[ForecastWeight]
        Current weights (post-decay, pre-Aronson).
    regime : str
        Current market regime (bull/bear/sideways).
    blend_factor : float
        How aggressively to tilt (0.0=no change, 1.0=full Sharpe² weighting).

    Returns
    -------
    list[ForecastWeight]
        Regime-adjusted weights, renormalised to sum=1.0.
    """
    if not regime or blend_factor <= 0:
        return base_weights

    regime_key = regime.lower().replace("trending_", "").replace("range_bound", "sideways").replace("high_volatility", "bear").replace("crisis", "bear")
    if regime_key not in ("bull", "sideways", "bear"):
        regime_key = "sideways"  # default

    # Compute Sharpe² for each source in this regime
    sharpe_sq = {}
    for fw in base_weights:
        if fw.weight <= 0:
            sharpe_sq[fw.name] = 0.0
            continue
        scores = REGIME_SHARPE_SCORES.get(fw.name, {})
        sr = scores.get(regime_key, 0.30)  # default 0.30 Sharpe for unknown sources
        sharpe_sq[fw.name] = sr * sr  # Sharpe²

    total_sq = sum(sharpe_sq.values())
    if total_sq <= 0:
        return base_weights

    # Build Sharpe²-weighted allocation
    adjusted = []
    for fw in base_weights:
        if fw.weight <= 0:
            adjusted.append(ForecastWeight(fw.name, 0.0))
            continue
        optimal_w = sharpe_sq[fw.name] / total_sq
        blended = (1 - blend_factor) * fw.weight + blend_factor * optimal_w
        adjusted.append(ForecastWeight(fw.name, blended))

    # Renormalise to sum=1.0
    total = sum(fw.weight for fw in adjusted if fw.weight > 0)
    if total > 0:
        adjusted = [
            ForecastWeight(fw.name, fw.weight / total) if fw.weight > 0 else fw
            for fw in adjusted
        ]

    logger.info("P4: Regime-Sharpe² weights applied (regime=%s, blend=%.0f%%)", regime_key, blend_factor * 100)
    return adjusted


def get_aronson_adjusted_weights(
    base_weights: List[ForecastWeight],
) -> List[ForecastWeight]:
    """Apply Aronson EBTA validation multipliers to forecast weights.

    Loads persisted validation state (if available) and multiplies each
    source's weight by its validation multiplier.  Weights are then
    renormalised to sum to 1.0.

    This ensures that statistically validated signals receive their full
    weight, while unvalidated or weak signals are penalised.
    """
    global _aronson_weight_multipliers, _aronson_multipliers_loaded
    if not _aronson_multipliers_loaded:
        try:
            from services.aronson_validator import AronsonValidator
            _aronson_weight_multipliers = AronsonValidator.load_weight_multipliers()
        except Exception as exc:
            logger.debug("Aronson weight multipliers unavailable: %s", exc)
            _aronson_weight_multipliers = {}
        _aronson_multipliers_loaded = True

    if not _aronson_weight_multipliers:
        return base_weights

    adjusted = []
    for fw in base_weights:
        mult = _aronson_weight_multipliers.get(fw.name, 1.0)
        adjusted.append(ForecastWeight(fw.name, fw.weight * mult))

    total = sum(fw.weight for fw in adjusted)
    if total > 0:
        adjusted = [ForecastWeight(fw.name, fw.weight / total) for fw in adjusted]
    return adjusted


def combine_forecasts(
    symbol: str,
    forecasts: Dict[str, float],
    weights: Optional[List[ForecastWeight]] = None,
    correlations: Optional[Dict[tuple, float]] = None,
    vol_regime_multiplier: Optional[float] = None,
    regime: str = "",
    forecast_history: Optional[Dict[str, list]] = None,
) -> CombinedForecast:
    """Combine multiple forecast sources into a single combined forecast.

    Parameters
    ----------
    symbol : str
        Instrument ticker.
    forecasts : dict[str, float]
        ``{source_name: forecast_value}``.  Missing sources are excluded
        and weights are renormalised.
    weights : list[ForecastWeight] | None
        Forecast weights.  Default: handcrafted NSE weights.
    correlations : dict | None
        Pairwise correlations.
    vol_regime_multiplier : float | None
        AFTS S13 — ratio of (median_vol / current_vol).  If >1, current vol
        is below median → scale forecasts UP (calm markets = more signal).
        If <1, current vol is above median → scale DOWN (volatile markets =
        more noise).  Capped at [0.5, 1.5].  If None, no vol adjustment.
    regime : str
        P4: Current market regime for Sharpe²-weighted allocation.
    forecast_history : dict[str, list[float]] | None
        P6: Rolling daily forecast values per source for dynamic FDM.
        When provided (≥30 days), FDM uses empirical correlations
        (shrinkage-blended with static prior) instead of static defaults.

    Returns
    -------
    CombinedForecast
    """
    weights = weights or DEFAULT_FORECAST_WEIGHTS
    # G1 FIX: Zero-weight inverted/dead strategies BEFORE Aronson adjustment
    weights = apply_decay_state_filter(weights)
    # P4: Apply regime-specific Sharpe²-weighted allocation
    if regime:
        weights = apply_regime_sharpe_weights(weights, regime=regime, blend_factor=0.5)
    # Apply Aronson EBTA validation multipliers (penalise unvalidated signals)
    weights = get_aronson_adjusted_weights(weights)
    weight_map = {fw.name: fw.weight for fw in weights}
    total_sources = len(weight_map)

    # Filter to available forecasts and renormalise weights
    available = {k: v for k, v in forecasts.items() if k in weight_map}
    if not available:
        return CombinedForecast(
            symbol=symbol,
            combined_forecast=0.0,
            raw_combined=0.0,
            fdm=1.0,
            sources_available=0,
            sources_total=total_sources,
        )

    # Renormalise weights to sum to 1.0 for available sources only
    active_weights = {k: weight_map[k] for k in available}
    total_w = sum(active_weights.values())
    if total_w > 0:
        active_weights = {k: v / total_w for k, v in active_weights.items()}

    # Weighted average of forecasts
    raw_combined = sum(
        active_weights[k] * available[k] for k in available
    )

    # --- AFTS S13: Vol-regime forecast magnitude adjustment ---
    # In low-vol environments, scale forecasts UP (more signal content).
    # In high-vol environments, scale forecasts DOWN (more noise).
    if vol_regime_multiplier is not None:
        vrm = max(0.5, min(1.5, vol_regime_multiplier))
        raw_combined *= vrm

    # G20: Compute FDM per-symbol from AVAILABLE sources (Carver Ch.8).
    # When a source is missing, the reduced set has different correlations
    # and thus a different FDM. This avoids over-scaling thin-signal symbols.
    #
    # P6: Use rolling empirical correlations when forecast_history is available.
    # Shrinkage-blended with static prior (30% shrinkage) for stability.
    effective_correlations = correlations
    if forecast_history:
        rolling_corr = compute_rolling_correlations(
            forecast_history, lookback=252, shrinkage=0.3,
        )
        if rolling_corr is not DEFAULT_CORRELATION_MATRIX:
            effective_correlations = rolling_corr
            logger.debug("P6: Using rolling FDM correlations for %s", symbol)
    fdm = compute_fdm(active_weights, effective_correlations)

    # Apply FDM and cap
    combined = raw_combined * fdm
    combined = cap_forecast(combined)

    # R16: Correlation dampener REMOVED — pile-in IS correct during trends.
    # R15 showed dampening trend agreement hurt bull capture without crash protection.
    # Regime-based vol sizing (in backtest loop) handles exposure reduction instead.

    # Aronson: compute confidence score (fraction of validated signals agreeing)
    try:
        from services.aronson_validator import AronsonValidator
        _val_summary = AronsonValidator.load_state()
        _conf = AronsonValidator().compute_confidence_for_symbol(available, _val_summary)
    except Exception as exc:
        logger.debug("Aronson confidence unavailable, defaulting to 1.0: %s", exc)
        _conf = 1.0

    return CombinedForecast(
        symbol=symbol,
        combined_forecast=combined,
        raw_combined=raw_combined,
        fdm=fdm,
        individual_forecasts=dict(available),
        weights_used=active_weights,
        sources_available=len(available),
        sources_total=total_sources,
        confidence_score=_conf,
    )


def combine_forecasts_batch(
    all_forecasts: Dict[str, Dict[str, float]],
    weights: Optional[object] = None,
    correlations: Optional[Dict[tuple, float]] = None,
    vol_regime_multipliers: Optional[Dict[str, float]] = None,
) -> Dict[str, CombinedForecast]:
    """Combine forecasts for all symbols.

    Parameters
    ----------
    all_forecasts : dict[str, dict[str, float]]
        ``{symbol: {source_name: forecast_value}}``.
    weights : list[ForecastWeight] | dict[str, float] | None
        Forecast weights. Can be a list of ForecastWeight or a dict
        from HMM blending. Default: handcrafted NSE weights.
    vol_regime_multipliers : dict[str, float] | None
        AFTS S13 — per-symbol vol regime multiplier.

    Returns
    -------
    dict[str, CombinedForecast]
    """
    # Convert dict weights to ForecastWeight list
    fw_list = None
    if isinstance(weights, dict):
        fw_list = [ForecastWeight(name=k, weight=v) for k, v in weights.items()]
    elif isinstance(weights, list):
        fw_list = weights

    results = {}
    for sym, fc_dict in all_forecasts.items():
        vrm = None
        if vol_regime_multipliers:
            vrm = vol_regime_multipliers.get(sym)
        results[sym] = combine_forecasts(sym, fc_dict, fw_list, correlations, vrm)

    # Log summary
    avg_sources = (
        sum(cf.sources_available for cf in results.values()) / len(results)
        if results else 0
    )
    avg_forecast = (
        sum(abs(cf.combined_forecast) for cf in results.values()) / len(results)
        if results else 0
    )
    logger.info(
        "Forecasts combined for %d symbols: avg %.1f sources, avg abs forecast %.1f",
        len(results), avg_sources, avg_forecast,
    )
    return results


def apply_masters_quality_gate(
    combined_forecasts: Dict[str, CombinedForecast],
    ohlcv_dict: Dict[str, "pd.DataFrame"],
    forecast_history: Optional[Dict[str, Dict[str, "np.ndarray"]]] = None,
) -> Dict[str, CombinedForecast]:
    """Apply Masters prediction quality gating to combined forecasts.

    For each symbol, assesses the quality of recent forecast vs actual
    returns. Low-quality forecasts are dampened (multiplied by 0.3-1.0).

    Masters Ch. 9: "Scale each forecast by its assessed quality.
    This automatically de-weights unreliable signals."

    Args:
        combined_forecasts: {symbol: CombinedForecast} from combine_forecasts_batch
        ohlcv_dict: {symbol: OHLCV DataFrame} for computing actual returns
        forecast_history: optional {symbol: {date_idx: forecast}} for quality calc

    Returns:
        {symbol: CombinedForecast} with quality-gated forecasts
    """
    try:
        from strategies.masters_prediction import compute_prediction_quality
        import numpy as np
        import pandas as pd
    except ImportError:
        logger.debug("masters_prediction not available — skipping quality gate")
        return combined_forecasts

    gated = {}
    n_dampened = 0

    for sym, cf in combined_forecasts.items():
        df = ohlcv_dict.get(sym)
        if df is None or len(df) < 30:
            gated[sym] = cf
            continue

        try:
            close = df["Close"].values.astype(float)
            actual_returns = np.diff(close[-60:]) / np.maximum(
                np.abs(close[-61:-1]), 1e-10
            )

            # Use forecast sign as daily prediction proxy
            n_rets = len(actual_returns)
            if n_rets < 20:
                gated[sym] = cf
                continue

            # Build forecast array from current combined forecast direction
            forecast_arr = np.full(n_rets, cf.combined_forecast)

            quality = compute_prediction_quality(sym, forecast_arr, actual_returns)

            if quality.confidence_multiplier < 1.0:
                dampened_forecast = cf.combined_forecast * quality.confidence_multiplier
                dampened_forecast = max(-20.0, min(20.0, dampened_forecast))
                from dataclasses import replace as _dc_replace
                gated[sym] = _dc_replace(cf, combined_forecast=dampened_forecast)
                n_dampened += 1
            else:
                gated[sym] = cf
        except Exception as e:
            logger.debug("Quality gate skipped for %s: %s", sym, e)
            gated[sym] = cf

    if n_dampened > 0:
        logger.info(
            "Masters quality gate: %d/%d forecasts dampened",
            n_dampened, len(combined_forecasts),
        )

    return gated
