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
# Each profile defines ideal weights for each forecast source in that regime.
# Option A (Vince Enhancement Plan): Weights derived from strategy_combo_optimizer
# simulation across 3 regimes × 20 combos × 4 Vince modes. Key insights:
#   - PEAD and Pairs are genuinely regime-neutral (positive CAGR + high Sharpe everywhere)
#   - Trend/momentum DESTROYS capital in bear/sideways → near-zero weight there
#   - MeanReversion excels in sideways/bear but drags in strong bull
#   - Carry is mildly positive everywhere → small stable allocation
REGIME_STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "TRENDING_BULL": {
        "ewmac_8_32": 0.05,
        "ewmac_16_64": 0.07,
        "ewmac_32_128": 0.04,
        "ewmac_64_256": 0.03,
        "carry": 0.02,
        "screener": 0.03,
        "momentum": 0.10,
        "pead": 0.06,
        "mean_reversion": 0.01,
        "fii_flow": 0.03,
        "oi_signal": 0.02,
        "decision_engine": 0.02,
        "breakout": 0.00,
        "cross_momentum": 0.04,
        "pairs_arb": 0.01,
        "event_driven": 0.01,
        "penfold_trend": 0.12,
        "ehlers_dsp": 0.12,      # Ehlers: adaptive trend + cycle detection in bull
        "intermarket": 0.08,     # Ruggiero: intermarket confirmation in trending
        "acceleration": 0.07,    # S23: catches accelerating trends early
        "carver_value": 0.01,    # S22: contrarian — minimal in bull trend
        "skew_signal": 0.04,     # S24: skew premium strongest in calm bull
        "sentiment": 0.02,       # News sentiment: momentum amplifier in bull
    },
    "TRENDING_BEAR": {
        "ewmac_8_32": 0.01,
        "ewmac_16_64": 0.01,
        "ewmac_32_128": 0.01,
        "ewmac_64_256": 0.03,
        "carry": 0.04,
        "screener": 0.03,
        "momentum": 0.01,
        "pead": 0.15,            # +3%: strongest counter-cyclical alpha in bear
        "mean_reversion": 0.15,  # +3%: mean-reversion excels when trends exhaust
        "fii_flow": 0.03,
        "oi_signal": 0.02,
        "decision_engine": 0.03,
        "breakout": 0.00,
        "cross_momentum": 0.01,
        "pairs_arb": 0.12,
        "event_driven": 0.09,
        "penfold_trend": 0.02,
        "ehlers_dsp": 0.03,      # Ehlers: SNR filter reduces false signals in bear
        "intermarket": 0.08,     # Ruggiero: intermarket leads macro reversals
        "acceleration": 0.02,    # S23: detects trend deceleration -> early exit signal
        "carver_value": 0.05,    # +1%: value opportunities emerge in bear markets
        "skew_signal": 0.02,     # S24: skew regime shifts during bear
        "sentiment": 0.04,       # +2%: contrarian fear signal strongest in bear
    },  # sum = 1.00
    "RANGE_BOUND": {
        "ewmac_8_32": 0.01,
        "ewmac_16_64": 0.01,
        "ewmac_32_128": 0.02,
        "ewmac_64_256": 0.02,
        "carry": 0.05,
        "screener": 0.03,
        "momentum": 0.01,
        "pead": 0.13,            # +2%: event-driven alpha regime-neutral
        "mean_reversion": 0.14,  # +2%: core range-bound strategy
        "fii_flow": 0.03,
        "oi_signal": 0.03,
        "decision_engine": 0.04,
        "breakout": 0.00,
        "cross_momentum": 0.02,
        "pairs_arb": 0.12,
        "event_driven": 0.08,
        "penfold_trend": 0.03,
        "ehlers_dsp": 0.06,      # Ehlers: cycle detection excels in range-bound
        "intermarket": 0.05,     # Ruggiero: seasonal + intermarket for range timing
        "acceleration": 0.02,    # S23: low utility in range (no sustained trend)
        "carver_value": 0.05,    # S22: value excels in range-bound mean-reversion
        "skew_signal": 0.02,     # S24: moderate utility -- skew premium persists
        "sentiment": 0.03,       # +1%: sentiment helps catch range-bound reversals
    },  # sum = 1.00
    "HIGH_VOLATILITY": {
        "ewmac_8_32": 0.02,
        "ewmac_16_64": 0.02,
        "ewmac_32_128": 0.03,
        "ewmac_64_256": 0.03,
        "carry": 0.04,
        "screener": 0.03,
        "momentum": 0.01,
        "pead": 0.12,            # +2%: event alpha uncorrelated to vol spikes
        "mean_reversion": 0.10,  # +2%: mean-reversion profits from vol overshoot
        "fii_flow": 0.03,
        "oi_signal": 0.03,
        "decision_engine": 0.04,
        "breakout": 0.00,
        "cross_momentum": 0.02,
        "pairs_arb": 0.08,
        "event_driven": 0.07,
        "penfold_trend": 0.03,
        "ehlers_dsp": 0.07,      # Ehlers: adaptive indicators adjust to vol regime
        "intermarket": 0.11,     # Ruggiero: intermarket drives vol regime shifts
        "acceleration": 0.04,    # S23: catches vol-driven trend reversals
        "carver_value": 0.02,    # S22: value slightly useful in vol spikes
        "skew_signal": 0.03,     # S24: skew premium increases in high-vol
        "sentiment": 0.03,       # +1%: sentiment contrarian signal in high-vol
    },  # sum = 1.00
    "CRISIS": {
        "ewmac_8_32": 0.01,
        "ewmac_16_64": 0.01,
        "ewmac_32_128": 0.01,
        "ewmac_64_256": 0.03,
        "carry": 0.04,
        "screener": 0.02,
        "momentum": 0.01,
        "pead": 0.14,            # +2%: PEAD alpha strongest in panic (earnings surprises)
        "mean_reversion": 0.13,  # +3%: crisis overshoot -> mean-reversion bounces
        "fii_flow": 0.02,
        "oi_signal": 0.02,
        "decision_engine": 0.02,
        "breakout": 0.00,
        "cross_momentum": 0.01,
        "pairs_arb": 0.15,
        "event_driven": 0.12,
        "penfold_trend": 0.02,
        "ehlers_dsp": 0.03,      # Ehlers: minimal -- most signals unreliable in crisis
        "intermarket": 0.09,     # Ruggiero: intermarket is the strongest crisis signal
        "acceleration": 0.02,    # S23: detects crash deceleration (recovery signal)
        "carver_value": 0.05,    # S22: deep value emerges in crisis (buy fear)
        "skew_signal": 0.02,     # S24: skew premium highest post-crash
        "sentiment": 0.03,       # +1%: contrarian signal in extreme fear
    },  # sum = 1.00
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
