"""
Strategy Decay Detection & Auto-Deallocation — Phase 4.4.

Monitors rolling Sharpe ratio for each strategy and detects performance
degradation. Automatically reduces allocation for decaying strategies.

Decay levels:
  - HEALTHY: recent_sharpe > 0.5 × historical_sharpe → full allocation
  - DEGRADED: 0.25 × historical < recent < 0.5 × historical → halve alloc
  - DEAD: recent_sharpe < 0.25 × historical → zero allocation + alert
  - INVERTED: recent_sharpe < 0 → zero allocation + investigate

Integration:
  - Reads from walk-forward audit results
  - Updates factor_momentum weights
  - Triggers re-calibration via scheduler
  - Sends notifications on status changes
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

_DECAY_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "strategy_decay_state.json"


@dataclass
class DecayStatus:
    """Decay assessment for a single strategy."""
    strategy_name: str
    status: str = "HEALTHY"          # HEALTHY / DEGRADED / DEAD / INVERTED
    recent_sharpe: float = 0.0       # rolling 63-day Sharpe
    historical_sharpe: float = 0.0   # walk-forward average Sharpe
    decay_ratio: float = 1.0         # recent / historical
    allocation_multiplier: float = 1.0  # 1.0 = full, 0.5 = half, 0 = zero
    days_in_current_state: int = 0
    last_healthy_date: str = ""
    action_needed: str = ""
    equity_above_ma: bool = True        # FIX-09: equity curve filter (Penfold/Ruggiero)
    equity_ma_period: int = 63          # 63-day (~3 month) SMA


@dataclass
class StrategyDecayReport:
    """Overall decay monitoring report."""
    strategies: List[DecayStatus] = field(default_factory=list)
    healthy_count: int = 0
    degraded_count: int = 0
    dead_count: int = 0
    inverted_count: int = 0
    computed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "computed_at": self.computed_at,
            "summary": {
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "dead": self.dead_count,
                "inverted": self.inverted_count,
            },
            "strategies": [
                {
                    "name": s.strategy_name,
                    "status": s.status,
                    "recent_sharpe": round(s.recent_sharpe, 3),
                    "historical_sharpe": round(s.historical_sharpe, 3),
                    "decay_ratio": round(s.decay_ratio, 3),
                    "alloc_mult": s.allocation_multiplier,
                    "action": s.action_needed,
                }
                for s in self.strategies
            ],
        }


class StrategyDecayMonitor:
    """Monitor strategy health and auto-adjust allocations.

    Parameters
    ----------
    lookback : int
        Rolling window for recent Sharpe (default 63 = 3 months).
    healthy_threshold : float
        Fraction of historical Sharpe to be considered healthy.
    dead_threshold : float
        Fraction below which strategy is considered dead.
    """

    @staticmethod
    def equity_curve_filter(equity_series, ma_period: int = 63) -> bool:
        """FIX-09: Penfold/Ruggiero equity curve filter.

        Returns True if current equity is above its SMA(ma_period).
        When False, strategy should suppress new trades.
        """
        import numpy as np
        arr = np.asarray(equity_series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < ma_period:
            return True  # insufficient data, assume OK
        ma = float(np.mean(arr[-ma_period:]))
        return float(arr[-1]) >= ma

    def __init__(
        self,
        lookback: int = 63,
        healthy_threshold: float = 0.50,
        dead_threshold: float = 0.25,
    ):
        self.lookback = lookback
        self.healthy_thresh = healthy_threshold
        self.dead_thresh = dead_threshold
        self._previous_states = self._load_state()

    def check_all(
        self,
        strategy_returns: Dict[str, "pd.Series"],
        historical_sharpes: Dict[str, float],
    ) -> StrategyDecayReport:
        """Check decay status for all strategies.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Recent daily returns per strategy.
        historical_sharpes : dict[str, float]
            Long-term (walk-forward) Sharpe per strategy.
        """
        statuses: List[DecayStatus] = []

        for name, returns in strategy_returns.items():
            hist_sharpe = historical_sharpes.get(name, 0.0)
            status = self.check_single(name, returns, hist_sharpe)
            statuses.append(status)

        healthy = sum(1 for s in statuses if s.status == "HEALTHY")
        degraded = sum(1 for s in statuses if s.status == "DEGRADED")
        dead = sum(1 for s in statuses if s.status == "DEAD")
        inverted = sum(1 for s in statuses if s.status == "INVERTED")

        report = StrategyDecayReport(
            strategies=statuses,
            healthy_count=healthy,
            degraded_count=degraded,
            dead_count=dead,
            inverted_count=inverted,
            computed_at=datetime.utcnow().isoformat(),
        )

        self._save_state(statuses)
        return report

    def check_single(
        self,
        strategy_name: str,
        recent_returns: "pd.Series",
        historical_sharpe: float,
    ) -> DecayStatus:
        """Check decay for a single strategy."""
        n = len(recent_returns.tail(self.lookback).dropna())
        if n < 10:
            return DecayStatus(
                strategy_name=strategy_name,
                status="HEALTHY",
                allocation_multiplier=1.0,
                action_needed="Insufficient data",
            )

        recent = recent_returns.tail(self.lookback)
        recent_sharpe = float(recent.mean() / (recent.std() + 1e-10) * np.sqrt(252))

        # Determine status
        if historical_sharpe <= 0:
            hist_sharpe_safe = 0.5  # fallback
        else:
            hist_sharpe_safe = historical_sharpe

        decay_ratio = recent_sharpe / hist_sharpe_safe if hist_sharpe_safe > 0 else 0

        if recent_sharpe < 0:
            status = "INVERTED"
            alloc = 0.0
            action = "Zero allocation + investigate (strategy losing money)"
        elif decay_ratio < self.dead_thresh:
            status = "DEAD"
            alloc = 0.0
            action = "Zero allocation + schedule re-calibration"
        elif decay_ratio < self.healthy_thresh:
            status = "DEGRADED"
            alloc = 0.5
            action = "Halve allocation, monitor closely"
        else:
            status = "HEALTHY"
            alloc = 1.0
            action = ""

        # Track state duration
        prev = self._previous_states.get(strategy_name, {})
        prev_status = prev.get("status", "HEALTHY")
        if status == prev_status:
            days = prev.get("days", 0) + 1
        else:
            days = 1

        last_healthy = prev.get("last_healthy", "")
        if status == "HEALTHY":
            last_healthy = datetime.utcnow().strftime("%Y-%m-%d")

        return DecayStatus(
            strategy_name=strategy_name,
            status=status,
            recent_sharpe=round(recent_sharpe, 3),
            historical_sharpe=round(historical_sharpe, 3),
            decay_ratio=round(decay_ratio, 3),
            allocation_multiplier=alloc,
            days_in_current_state=days,
            last_healthy_date=last_healthy,
            action_needed=action,
        )

    def get_allocation_multipliers(
        self,
        strategy_returns: Dict[str, "pd.Series"],
        historical_sharpes: Dict[str, float],
    ) -> Dict[str, float]:
        """Return {strategy_name: allocation_multiplier} for use in forecast combiner."""
        report = self.check_all(strategy_returns, historical_sharpes)
        return {s.strategy_name: s.allocation_multiplier for s in report.strategies}

    def _save_state(self, statuses: List[DecayStatus]) -> None:
        try:
            data = {}
            for s in statuses:
                data[s.strategy_name] = {
                    "status": s.status,
                    "days": s.days_in_current_state,
                    "last_healthy": s.last_healthy_date,
                    "recent_sharpe": s.recent_sharpe,
                }
            _DECAY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_DECAY_STATE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.debug("Decay state save failed: %s", exc)

    def _load_state(self) -> dict:
        if not _DECAY_STATE_PATH.exists():
            return {}
        try:
            with open(_DECAY_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
