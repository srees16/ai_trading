"""
Regime-Conditional Strategy Mixing — Phase 5.1.

Changes forecast combiner weights based on the current market regime.
In bull markets, overweight momentum. In bear/range, overweight
mean-reversion and carry. In crisis, go mostly to cash.

Integration:
  - Reads current regime from regime_detector.py
  - Provides dynamic weights to forecast_combiner.py
  - Takes precedence over factor_momentum weights when regime is extreme
  - Blends with factor momentum for moderate regimes
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Regime-specific strategy weight profiles
# Each profile defines ideal weights for each forecast source in that regime
REGIME_STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "TRENDING_BULL": {
        "ewmac_16_64": 0.16,
        "ewmac_32_128": 0.12,
        "ewmac_64_256": 0.12,
        "carry": 0.08,
        "screener": 0.06,
        "momentum": 0.20,
        "pead": 0.04,
        "mean_reversion": 0.03,
        "fii_flow": 0.06,
        "oi_signal": 0.05,
        "decision_engine": 0.08,
    },
    "TRENDING_BEAR": {
        "ewmac_16_64": 0.04,
        "ewmac_32_128": 0.04,
        "ewmac_64_256": 0.08,
        "carry": 0.16,
        "screener": 0.16,
        "momentum": 0.04,
        "pead": 0.12,
        "mean_reversion": 0.12,
        "fii_flow": 0.06,
        "oi_signal": 0.04,
        "decision_engine": 0.14,
    },
    "RANGE_BOUND": {
        "ewmac_16_64": 0.04,
        "ewmac_32_128": 0.04,
        "ewmac_64_256": 0.04,
        "carry": 0.18,
        "screener": 0.12,
        "momentum": 0.04,
        "pead": 0.08,
        "mean_reversion": 0.18,
        "fii_flow": 0.06,
        "oi_signal": 0.06,
        "decision_engine": 0.16,
    },
    "HIGH_VOLATILITY": {
        "ewmac_16_64": 0.10,
        "ewmac_32_128": 0.08,
        "ewmac_64_256": 0.08,
        "carry": 0.10,
        "screener": 0.10,
        "momentum": 0.08,
        "pead": 0.08,
        "mean_reversion": 0.10,
        "fii_flow": 0.08,
        "oi_signal": 0.06,
        "decision_engine": 0.14,
    },
    "CRISIS": {
        "ewmac_64_256": 0.15,
        "carry": 0.20,
        "screener": 0.20,
        "pead": 0.15,
        "mean_reversion": 0.15,
        "fii_flow": 0.05,
        "decision_engine": 0.10,
    },
}

# Default weights (used when regime is unknown)
DEFAULT_REGIME = "RANGE_BOUND"


def get_regime_weights(
    regime: str,
    blend_with_static: float = 0.3,
    static_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return forecast weights conditioned on current market regime.

    Parameters
    ----------
    regime : str
        Current regime from regime_detector (e.g., 'TRENDING_BULL').
    blend_with_static : float
        Blend ratio with static Carver weights (0 = fully regime, 1 = fully static).
    static_weights : dict | None
        Baseline weights. If None, uses DEFAULT_FORECAST_WEIGHTS.

    Returns
    -------
    dict[str, float]
        Normalized weights for the forecast combiner.
    """
    regime_key = regime.upper().replace(" ", "_")
    regime_w = REGIME_STRATEGY_WEIGHTS.get(regime_key)

    if regime_w is None:
        regime_w = REGIME_STRATEGY_WEIGHTS.get(DEFAULT_REGIME, {})
        logger.debug("Unknown regime '%s', using default (%s)", regime, DEFAULT_REGIME)

    if static_weights is None:
        static_weights = {
            "ewmac_16_64": 0.18, "ewmac_32_128": 0.14, "ewmac_64_256": 0.18,
            "carry": 0.18, "screener": 0.12, "momentum": 0.15, "pead": 0.05,
        }

    # Blend: regime_weight × (1 - blend) + static_weight × blend
    all_keys = set(regime_w.keys()) | set(static_weights.keys())
    blended = {}
    for k in all_keys:
        rw = regime_w.get(k, 0.0)
        sw = static_weights.get(k, 0.0)
        blended[k] = (1.0 - blend_with_static) * rw + blend_with_static * sw

    # Normalize to sum to 1.0
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended


def get_regime_aware_forecast_weights(
    regime_snapshot=None,
) -> Optional[List]:
    """Get ForecastWeight objects conditioned on current regime.

    Attempts to read current regime from regime_detector, then
    returns appropriate weights for the forecast combiner.

    Returns None if regime detection is unavailable (fallback to static).
    """
    from services.forecast_combiner import ForecastWeight

    if regime_snapshot is None:
        try:
            from services.regime_detector import get_current_regime
            regime_snapshot = get_current_regime()
        except Exception:
            return None

    if regime_snapshot is None:
        return None

    regime = regime_snapshot.regime.value if hasattr(regime_snapshot.regime, "value") else str(regime_snapshot.regime)
    confidence = getattr(regime_snapshot, "confidence", 0.5)

    # Higher confidence = more regime-specific weights
    # Lower confidence = more static weights
    blend = 1.0 - min(confidence, 0.9)  # at confidence=0.9, use 90% regime weights

    weights = get_regime_weights(regime, blend_with_static=blend)
    return [ForecastWeight(name=k, weight=v) for k, v in weights.items()]
