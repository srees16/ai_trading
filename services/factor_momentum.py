"""
Factor Momentum — Dynamic Strategy Weighting — Phase 1.4.

Replaces fixed Carver forecast weights with adaptive weights that
tilt towards recently-performing strategies.

Research basis:
  - Arnott et al. (2021): Factor momentum exists and is exploitable
  - Ehsani & Linnainmaa (2022): Strategy momentum in quantitative factors
  - Gupta & Kelly (2019): Momentum in time-series and cross-section of factors

Integration:
  - Plugs into forecast_combiner.py: dynamic weights replace DEFAULT_FORECAST_WEIGHTS
  - Rebalanced weekly (Saturday scheduler job)
  - Weights constrained: each strategy gets at least 5% and at most 40%

How it works:
  1. Track per-strategy daily returns (from paper trader or live journal)
  2. Compute rolling 3-month risk-adjusted return (Sharpe proxy) per strategy
  3. Reweight proportional to recent risk-adjusted performance
  4. Apply constraints and persist to disk
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "factor_momentum_weights.json"

# Constraints on dynamic weights
MIN_WEIGHT = 0.05   # no strategy below 5%
MAX_WEIGHT = 0.40   # no strategy above 40%

# Default strategy names that map to forecast combiner keys
DEFAULT_STRATEGIES = [
    "ewmac_16_64",
    "ewmac_32_128",
    "ewmac_64_256",
    "carry",
    "screener",
]


@dataclass
class StrategyPerformance:
    """Recent performance for a single strategy."""
    name: str
    sharpe_3m: float = 0.0       # 3-month rolling Sharpe
    return_3m_pct: float = 0.0   # 3-month cumulative return
    vol_3m_pct: float = 0.0      # 3-month annualized vol
    n_observations: int = 0
    weight: float = 0.0          # allocated weight (0..1)


@dataclass
class FactorMomentumResult:
    """Output of the dynamic weighting computation."""
    weights: Dict[str, float] = field(default_factory=dict)
    performances: List[StrategyPerformance] = field(default_factory=list)
    method: str = "factor_momentum"   # or "equal_weight" fallback
    computed_at: str = ""


class FactorMomentum:
    """Dynamic strategy weighting engine based on recent performance.

    Parameters
    ----------
    lookback : int
        Days of recent returns to evaluate (default 63 = ~3 months).
    min_obs : int
        Minimum observations needed per strategy before using dynamic weights.
    blend_with_static : float
        Blend ratio with static Carver weights (0 = fully dynamic, 1 = fully static).
        Default 0.5 = 50/50 blend for stability.
    """

    def __init__(
        self,
        lookback: int = 63,
        min_obs: int = 20,
        blend_with_static: float = 0.5,
    ):
        self.lookback = lookback
        self.min_obs = min_obs
        self.blend_with_static = max(0.0, min(1.0, blend_with_static))

    def compute_dynamic_weights(
        self,
        strategy_returns: Dict[str, "pd.Series"],
        static_weights: Optional[Dict[str, float]] = None,
    ) -> FactorMomentumResult:
        """Compute dynamic strategy weights from recent returns.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            {strategy_name: daily_return_series} for each active strategy.
        static_weights : dict[str, float] | None
            Baseline Carver weights. If None, uses equal weights.

        Returns
        -------
        FactorMomentumResult
        """
        if static_weights is None:
            n = len(strategy_returns) or len(DEFAULT_STRATEGIES)
            static_weights = {s: 1.0 / n for s in (strategy_returns.keys() or DEFAULT_STRATEGIES)}

        performances: List[StrategyPerformance] = []
        raw_sharpes: Dict[str, float] = {}
        has_enough_data = True

        for name, returns in strategy_returns.items():
            recent = returns.tail(self.lookback)
            n_obs = len(recent.dropna())

            if n_obs < self.min_obs:
                has_enough_data = False
                performances.append(StrategyPerformance(name=name, n_observations=n_obs))
                continue

            mean_ret = float(recent.mean())
            std_ret = float(recent.std())
            sharpe_3m = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)
            return_3m = float((1 + recent).prod() - 1) * 100
            vol_3m = std_ret * np.sqrt(252) * 100

            # Floor Sharpe at a small positive number to avoid division issues
            raw_sharpes[name] = max(sharpe_3m, 0.05)

            performances.append(StrategyPerformance(
                name=name,
                sharpe_3m=round(sharpe_3m, 3),
                return_3m_pct=round(return_3m, 2),
                vol_3m_pct=round(vol_3m, 2),
                n_observations=n_obs,
            ))

        # Fallback to equal/static weights if insufficient data
        if not raw_sharpes or not has_enough_data:
            logger.info("Factor momentum: insufficient data, using static weights")
            result = FactorMomentumResult(
                weights=static_weights,
                performances=performances,
                method="static_fallback",
                computed_at=datetime.utcnow().isoformat(),
            )
            self._persist_weights(result.weights)
            return result

        # Compute dynamic weights proportional to Sharpe
        total_sharpe = sum(raw_sharpes.values())
        dynamic_weights = {k: v / total_sharpe for k, v in raw_sharpes.items()}

        # Apply constraints
        dynamic_weights = self._apply_constraints(dynamic_weights)

        # Blend with static weights
        blended = {}
        all_keys = set(dynamic_weights.keys()) | set(static_weights.keys())
        for k in all_keys:
            dw = dynamic_weights.get(k, 0.0)
            sw = static_weights.get(k, 0.0)
            blended[k] = self.blend_with_static * sw + (1.0 - self.blend_with_static) * dw

        # Re-normalize after blending
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        # Apply constraints again after blending
        blended = self._apply_constraints(blended)

        # Update performance objects with weights
        for perf in performances:
            perf.weight = round(blended.get(perf.name, 0.0), 4)

        result = FactorMomentumResult(
            weights=blended,
            performances=performances,
            method="factor_momentum",
            computed_at=datetime.utcnow().isoformat(),
        )

        self._persist_weights(result.weights)
        logger.info(
            "Factor momentum weights: %s",
            {k: f"{v:.1%}" for k, v in blended.items()},
        )
        return result

    @staticmethod
    def _apply_constraints(weights: Dict[str, float]) -> Dict[str, float]:
        """Enforce min/max weight constraints and re-normalize."""
        constrained = {}
        for k, v in weights.items():
            constrained[k] = max(MIN_WEIGHT, min(MAX_WEIGHT, v))

        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}
        return constrained

    @staticmethod
    def _persist_weights(weights: Dict[str, float]) -> None:
        """Save weights to disk."""
        try:
            _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "weights": {k: round(v, 4) for k, v in weights.items()},
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(_WEIGHTS_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.debug("Factor momentum persist failed: %s", exc)


def load_dynamic_weights() -> Optional[Dict[str, float]]:
    """Load persisted factor momentum weights.

    Returns None if weights are stale (>14 days) or missing.
    """
    if not _WEIGHTS_PATH.exists():
        return None
    try:
        with open(_WEIGHTS_PATH, "r") as f:
            data = json.load(f)
        updated = datetime.fromisoformat(data.get("updated_at", "2000-01-01"))
        if (datetime.utcnow() - updated).days > 14:
            logger.info("Factor momentum weights stale (%s), using static", updated.isoformat())
            return None
        return data.get("weights")
    except Exception:
        return None


def get_forecast_weights() -> List:
    """Return ForecastWeight objects using dynamic weights if available.

    Falls back to DEFAULT_FORECAST_WEIGHTS from forecast_combiner if
    dynamic weights are unavailable or stale.
    """
    from services.forecast_combiner import ForecastWeight, DEFAULT_FORECAST_WEIGHTS

    dynamic = load_dynamic_weights()
    if dynamic is None:
        return DEFAULT_FORECAST_WEIGHTS

    return [ForecastWeight(name=k, weight=v) for k, v in dynamic.items()]
