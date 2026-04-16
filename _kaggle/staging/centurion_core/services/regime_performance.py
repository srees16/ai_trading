"""
Regime-Conditional Performance Tracking — Phase 4.2.

Tracks strategy performance separately per market regime, surfacing
strategies that are regime-dependent (e.g., momentum works in bull,
fails in bear/range).

Integration:
  - Used in walk-forward summary reports
  - Fed into strategy_decay.py for auto-deallocation
  - Displayed on live dashboard for monitoring
  - Informs regime_detector → forecast_combiner weight adjustments
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class RegimeStats:
    """Performance statistics for a single regime."""
    regime: str
    n_days: int = 0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_daily_return_pct: float = 0.0
    win_rate: float = 0.0


@dataclass
class RegimePerformanceResult:
    """Strategy performance stratified by regime."""
    strategy_name: str
    regime_stats: Dict[str, RegimeStats] = field(default_factory=dict)
    overall_sharpe: float = 0.0
    weakest_regime: str = ""
    strongest_regime: str = ""
    regime_dependency_score: float = 0.0  # 0 = regime-neutral, 1 = fully regime-dependent

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "overall_sharpe": round(self.overall_sharpe, 3),
            "weakest_regime": self.weakest_regime,
            "strongest_regime": self.strongest_regime,
            "regime_dependency": round(self.regime_dependency_score, 3),
            "regimes": {
                k: {
                    "n_days": v.n_days,
                    "ann_return_pct": round(v.annualized_return_pct, 2),
                    "sharpe": round(v.sharpe, 3),
                    "max_dd_pct": round(v.max_drawdown_pct, 2),
                }
                for k, v in self.regime_stats.items()
            },
        }


class RegimePerformance:
    """Track strategy performance conditioned on market regime.

    Parameters
    ----------
    rf_annual : float
        Risk-free rate for Sharpe/Sortino calculation.
    """

    def __init__(self, rf_annual: float = 0.07):
        self.rf_annual = rf_annual

    def stratify_returns(
        self,
        strategy_name: str,
        daily_returns: "pd.Series",
        regime_labels: "pd.Series",
    ) -> RegimePerformanceResult:
        """Stratify strategy returns by regime and compute per-regime stats.

        Parameters
        ----------
        strategy_name : str
            Name of the strategy being evaluated.
        daily_returns : pd.Series
            Daily return series (date-indexed).
        regime_labels : pd.Series
            Regime label per date ('bull', 'bear', 'range', 'crisis').

        Returns
        -------
        RegimePerformanceResult
        """
        import pandas as pd

        # Align indices
        common = daily_returns.index.intersection(regime_labels.index)
        if len(common) < 20:
            logger.warning("Insufficient overlapping data for regime analysis")
            return RegimePerformanceResult(strategy_name=strategy_name)

        rets = daily_returns.loc[common]
        regimes = regime_labels.loc[common]

        regime_stats: Dict[str, RegimeStats] = {}
        rf_daily = (1 + self.rf_annual) ** (1 / TRADING_DAYS) - 1

        for regime in regimes.unique():
            mask = regimes == regime
            r = rets[mask]
            n = len(r)
            if n < 5:
                continue

            total_ret = float((1 + r).prod() - 1)
            years = n / TRADING_DAYS
            ann_ret = ((1 + total_ret) ** (1 / years) - 1) * 100 if years > 0 else 0

            excess = r - rf_daily
            sharpe = float(excess.mean() / (excess.std() + 1e-10) * np.sqrt(TRADING_DAYS))

            downside = r[r < 0]
            downside_std = float(downside.std()) if len(downside) > 1 else 1e-10
            sortino = float(excess.mean() / (downside_std + 1e-10) * np.sqrt(TRADING_DAYS))

            cum = (1 + r).cumprod()
            peak = cum.expanding().max()
            dd = (cum - peak) / peak
            max_dd = float(dd.min()) * 100

            win_rate = float((r > 0).sum() / n)

            regime_stats[regime] = RegimeStats(
                regime=regime,
                n_days=n,
                total_return_pct=round(total_ret * 100, 2),
                annualized_return_pct=round(ann_ret, 2),
                sharpe=round(sharpe, 3),
                sortino=round(sortino, 3),
                max_drawdown_pct=round(max_dd, 2),
                avg_daily_return_pct=round(float(r.mean()) * 100, 4),
                win_rate=round(win_rate, 3),
            )

        # Overall Sharpe
        excess_all = rets - rf_daily
        overall_sharpe = float(excess_all.mean() / (excess_all.std() + 1e-10) * np.sqrt(TRADING_DAYS))

        # Identify strongest/weakest regimes
        sharpes = {k: v.sharpe for k, v in regime_stats.items()}
        strongest = max(sharpes, key=sharpes.get) if sharpes else ""
        weakest = min(sharpes, key=sharpes.get) if sharpes else ""

        # Regime dependency score: std(sharpes) / mean(|sharpes|)
        if sharpes:
            sharpe_vals = list(sharpes.values())
            dep_score = float(np.std(sharpe_vals) / (np.mean(np.abs(sharpe_vals)) + 1e-10))
            dep_score = min(1.0, dep_score)
        else:
            dep_score = 0.0

        return RegimePerformanceResult(
            strategy_name=strategy_name,
            regime_stats=regime_stats,
            overall_sharpe=round(overall_sharpe, 3),
            weakest_regime=weakest,
            strongest_regime=strongest,
            regime_dependency_score=round(dep_score, 3),
        )

    def compare_strategies(
        self,
        strategy_returns: Dict[str, "pd.Series"],
        regime_labels: "pd.Series",
    ) -> Dict[str, RegimePerformanceResult]:
        """Compare multiple strategies across regimes.

        Returns dict keyed by strategy name.
        """
        results = {}
        for name, rets in strategy_returns.items():
            results[name] = self.stratify_returns(name, rets, regime_labels)
        return results
