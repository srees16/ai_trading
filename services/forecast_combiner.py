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

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from services.forecast_scalar import cap_forecast, TARGET_ABS_FORECAST

logger = logging.getLogger(__name__)

# Maximum FDM to prevent extreme positions (Carver recommendation)
MAX_FDM = 2.0


@dataclass
class ForecastWeight:
    """Weight for a single forecast source."""
    name: str
    weight: float  # 0.0 to 1.0


# Default handcrafted weights for centurion_core NSE swing/positional
# Updated Phase 1: Added momentum and pead sources
# Updated Gap A2/A5/A6/B6: Added mean_reversion, fii_flow, oi_signal, decision_engine
DEFAULT_FORECAST_WEIGHTS: List[ForecastWeight] = [
    ForecastWeight("ewmac_16_64", 0.14),
    ForecastWeight("ewmac_32_128", 0.11),
    ForecastWeight("ewmac_64_256", 0.14),
    ForecastWeight("carry", 0.14),
    ForecastWeight("screener", 0.08),
    ForecastWeight("momentum", 0.12),
    ForecastWeight("pead", 0.05),
    ForecastWeight("mean_reversion", 0.08),
    ForecastWeight("fii_flow", 0.05),
    ForecastWeight("oi_signal", 0.04),
    ForecastWeight("decision_engine", 0.05),
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
    # OI signal correlations (Gap A6)
    ("ewmac_16_64", "oi_signal"): 0.25,
    ("ewmac_32_128", "oi_signal"): 0.20,
    ("ewmac_64_256", "oi_signal"): 0.15,
    ("carry", "oi_signal"): 0.10,
    ("screener", "oi_signal"): 0.15,
    ("momentum", "oi_signal"): 0.30,
    ("pead", "oi_signal"): 0.10,
    ("mean_reversion", "oi_signal"): -0.10,
    ("fii_flow", "oi_signal"): 0.25,
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
    ("oi_signal", "decision_engine"): 0.15,
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


def combine_forecasts(
    symbol: str,
    forecasts: Dict[str, float],
    weights: Optional[List[ForecastWeight]] = None,
    correlations: Optional[Dict[tuple, float]] = None,
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

    Returns
    -------
    CombinedForecast
    """
    weights = weights or DEFAULT_FORECAST_WEIGHTS
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

    # Compute FDM for the active set
    fdm = compute_fdm(active_weights, correlations)

    # Apply FDM and cap
    combined = raw_combined * fdm
    combined = cap_forecast(combined)

    return CombinedForecast(
        symbol=symbol,
        combined_forecast=combined,
        raw_combined=raw_combined,
        fdm=fdm,
        individual_forecasts=dict(available),
        weights_used=active_weights,
        sources_available=len(available),
        sources_total=total_sources,
    )


def combine_forecasts_batch(
    all_forecasts: Dict[str, Dict[str, float]],
    weights: Optional[any] = None,
    correlations: Optional[Dict[tuple, float]] = None,
) -> Dict[str, CombinedForecast]:
    """Combine forecasts for all symbols.

    Parameters
    ----------
    all_forecasts : dict[str, dict[str, float]]
        ``{symbol: {source_name: forecast_value}}``.
    weights : list[ForecastWeight] | dict[str, float] | None
        Forecast weights. Can be a list of ForecastWeight or a dict
        from HMM blending. Default: handcrafted NSE weights.

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
        results[sym] = combine_forecasts(sym, fc_dict, fw_list, correlations)

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
