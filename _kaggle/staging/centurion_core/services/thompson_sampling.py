"""
Thompson Sampling Bandit — T4-1 Adaptive Strategy Selection.

Replaces the monthly tournament heuristic with a continuous
Thompson Sampling multi-armed bandit for strategy weight allocation.

Each forecast source is an "arm". The bandit maintains Beta(α, β)
posteriors for each arm's "probability of generating positive alpha."
Weights are sampled from posteriors → sources that consistently
produce alpha get higher weights; decaying sources get less.

Research basis:
  - Agrawal & Goyal (2012): Analysis of Thompson Sampling
  - Russo et al. (2018): Tutorial on Thompson Sampling
  - Adapts to regime changes faster than monthly tournaments

Integration:
  - Called from carver_pipeline.py Step 4 as weight modifier
  - Updated daily with realized forecast→return outcomes
  - Persists state to data/bandit_state.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "bandit_state.json",
)


@dataclass
class BanditArm:
    """Beta-distributed arm for a forecast source."""
    name: str
    alpha: float = 1.0   # successes + 1 (prior)
    beta: float = 1.0    # failures + 1 (prior)
    total_pulls: int = 0
    cumulative_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Draw from Beta(alpha, beta) posterior."""
        rng = rng or np.random.default_rng()
        return float(rng.beta(self.alpha, self.beta))

    def update(self, reward: float) -> None:
        """Update posterior with observed reward.

        reward: 1.0 = forecast was profitable, 0.0 = not.
        Fractional rewards supported for partial success.
        """
        self.alpha += reward
        self.beta += (1.0 - reward)
        self.total_pulls += 1
        self.cumulative_reward += reward

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "total_pulls": self.total_pulls,
            "cumulative_reward": round(self.cumulative_reward, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BanditArm":
        return cls(
            name=d["name"],
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 1.0),
            total_pulls=d.get("total_pulls", 0),
            cumulative_reward=d.get("cumulative_reward", 0.0),
        )


class ThompsonSamplingBandit:
    """Multi-armed bandit for adaptive forecast source selection.

    Parameters
    ----------
    source_names : list[str]
        Names of all forecast sources (arms).
    decay_factor : float
        Exponential decay for old observations (0.99 = slow decay).
        Keeps the bandit adaptive to regime changes.
    min_weight : float
        Minimum weight floor per source (prevents complete elimination).
    """

    def __init__(
        self,
        source_names: List[str],
        decay_factor: float = 0.995,
        min_weight: float = 0.01,
    ):
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self._arms: Dict[str, BanditArm] = {}
        self._rng = np.random.default_rng()

        # Initialize arms
        for name in source_names:
            self._arms[name] = BanditArm(name=name)

        # Try to load persisted state
        self._load_state()

    def _load_state(self) -> None:
        """Load bandit state from disk."""
        try:
            if os.path.exists(_STATE_PATH):
                with open(_STATE_PATH, "r") as f:
                    data = json.load(f)
                for arm_data in data.get("arms", []):
                    name = arm_data["name"]
                    if name in self._arms:
                        self._arms[name] = BanditArm.from_dict(arm_data)
                logger.info("Thompson Sampling: loaded state with %d arms", len(data.get("arms", [])))
        except Exception as e:
            logger.debug("Thompson Sampling: state load failed: %s", e)

    def save_state(self) -> None:
        """Persist bandit state to disk."""
        try:
            os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
            data = {
                "arms": [arm.to_dict() for arm in self._arms.values()],
                "decay_factor": self.decay_factor,
            }
            with open(_STATE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Thompson Sampling: state save failed: %s", e)

    def sample_weights(self, base_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Sample weights from Thompson Sampling posteriors.

        Multiplies base weights by Thompson-sampled scores to create
        adaptive weights. Renormalizes to sum to 1.0.

        Parameters
        ----------
        base_weights : dict[str, float] | None
            Default static weights. If None, uses uniform.

        Returns
        -------
        dict[str, float]
            Adapted weights summing to 1.0.
        """
        samples = {}
        for name, arm in self._arms.items():
            samples[name] = arm.sample(self._rng)

        # Multiply base weights by Thompson samples
        if base_weights is None:
            base_weights = {name: 1.0 / len(self._arms) for name in self._arms}

        adapted = {}
        for name in self._arms:
            bw = base_weights.get(name, 0.0)
            ts = samples.get(name, 0.5)
            adapted[name] = max(self.min_weight, bw * ts)

        # Renormalize
        total = sum(adapted.values())
        if total > 0:
            adapted = {k: v / total for k, v in adapted.items()}

        return adapted

    def update_rewards(self, outcomes: Dict[str, float]) -> None:
        """Update arms with observed outcomes.

        Parameters
        ----------
        outcomes : dict[str, float]
            {source_name: reward} where reward ∈ [0, 1].
            1.0 = forecast was profitable, 0.0 = not.
            Can use 0.5 for breakeven.
        """
        # Apply decay to existing observations (keep bandit adaptive)
        for arm in self._arms.values():
            arm.alpha = max(1.0, arm.alpha * self.decay_factor)
            arm.beta = max(1.0, arm.beta * self.decay_factor)

        # Update with new observations
        for name, reward in outcomes.items():
            if name in self._arms:
                self._arms[name].update(max(0.0, min(1.0, reward)))

        self.save_state()

    def get_arm_stats(self) -> Dict[str, dict]:
        """Return summary statistics for all arms."""
        return {
            name: {
                "mean_reward": round(arm.mean_reward, 4),
                "alpha": round(arm.alpha, 2),
                "beta": round(arm.beta, 2),
                "pulls": arm.total_pulls,
            }
            for name, arm in self._arms.items()
        }


def compute_bandit_outcomes(
    forecasts: Dict[str, Dict[str, float]],
    returns: Dict[str, float],
) -> Dict[str, float]:
    """Compute reward signals for each forecast source.

    Parameters
    ----------
    forecasts : dict[source_name, dict[symbol, forecast_value]]
    returns : dict[symbol, realized_return]

    Returns
    -------
    dict[source_name, reward]
        Reward ∈ [0, 1] based on forecast-return agreement.
    """
    outcomes = {}
    for source_name, source_forecasts in forecasts.items():
        correct = 0
        total = 0
        for sym, fc in source_forecasts.items():
            if sym in returns and fc != 0:
                ret = returns[sym]
                # Forecast agrees with return direction
                if (fc > 0 and ret > 0) or (fc < 0 and ret < 0):
                    correct += 1
                total += 1
        if total > 0:
            outcomes[source_name] = correct / total
        else:
            outcomes[source_name] = 0.5  # No data → neutral

    return outcomes
