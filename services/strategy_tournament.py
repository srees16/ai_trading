"""
Automated Strategy Tournament — Phase 8.1.

Monthly automated competition across all strategies:
1. Run each strategy on last 3 months of data (OOS)
2. Rank by Sortino, Calmar, Sharpe, MaxDD
3. Top strategies get allocation, bottom strategies get zero
4. Negative-Sharpe strategies auto-disabled

Integration:
  - Scheduled monthly via scheduler.py
  - Results feed into factor_momentum.py for weight updates
  - Integrates with strategy_decay.py for status tracking
  - Notifications on allocation changes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TournamentEntry:
    """Performance of one strategy in the tournament."""
    strategy_name: str
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    composite_score: float = 0.0
    rank: int = 0
    allocation_status: str = "ACTIVE"  # ACTIVE / REDUCED / DISABLED
    dsr_pvalue: float = 0.0            # FIX-08: Deflated Sharpe Ratio p-value
    dsr_significant: bool = False


@dataclass
class TournamentResult:
    """Results of a complete strategy tournament."""
    entries: List[TournamentEntry] = field(default_factory=list)
    top_strategies: List[str] = field(default_factory=list)
    disabled_strategies: List[str] = field(default_factory=list)
    period_months: int = 3
    computed_at: str = ""
    selection_bias_corrected_p: Optional[float] = None  # Best-of-N corrected p-value
    selection_bias_significant: Optional[bool] = None    # Is top strategy genuinely skilled?

    def to_dict(self) -> dict:
        d = {
            "computed_at": self.computed_at,
            "period_months": self.period_months,
            "top": self.top_strategies,
            "disabled": self.disabled_strategies,
            "rankings": [
                {
                    "rank": e.rank,
                    "name": e.strategy_name,
                    "sharpe": round(e.sharpe, 3),
                    "sortino": round(e.sortino, 3),
                    "calmar": round(e.calmar, 3),
                    "max_dd": round(e.max_drawdown_pct, 2),
                    "return": round(e.total_return_pct, 2),
                    "score": round(e.composite_score, 3),
                    "status": e.allocation_status,
                }
                for e in self.entries
            ],
        }
        if self.selection_bias_corrected_p is not None:
            d["selection_bias"] = {
                "corrected_p_value": round(self.selection_bias_corrected_p, 4),
                "significant": self.selection_bias_significant,
            }
        return d


class StrategyTournament:
    """Monthly automated strategy competition.

    Parameters
    ----------
    top_n : int
        Number of top strategies to include in active portfolio.
    min_sharpe : float
        Minimum Sharpe to keep a strategy active.
    weight_sharpe : float
        Weight for Sharpe in composite score.
    weight_sortino : float
        Weight for Sortino in composite score.
    weight_calmar : float
        Weight for Calmar in composite score.
    weight_dd : float
        Weight for (inverse) max drawdown in composite score.
    """

    def __init__(
        self,
        top_n: int = 5,
        min_sharpe: float = 0.0,
        weight_sharpe: float = 0.30,
        weight_sortino: float = 0.30,
        weight_calmar: float = 0.20,
        weight_dd: float = 0.20,
    ):
        self.top_n = top_n
        self.min_sharpe = min_sharpe
        self.w_sharpe = weight_sharpe
        self.w_sortino = weight_sortino
        self.w_calmar = weight_calmar
        self.w_dd = weight_dd

    def run_tournament(
        self,
        strategy_returns: Dict[str, "pd.Series"],
        lookback_months: int = 3,
        strategy_positions: Optional[Dict[str, "pd.Series"]] = None,
        raw_market_returns: Optional["pd.Series"] = None,
    ) -> TournamentResult:
        """Run the tournament on recent strategy returns.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            {strategy_name: daily_return_series} for each strategy.
        lookback_months : int
            Months of recent data to evaluate.
        strategy_positions : dict[str, pd.Series] | None
            {strategy_name: daily_position_vector}. If provided with
            raw_market_returns, runs best-of-N selection bias correction.
        raw_market_returns : pd.Series | None
            Raw market returns for MC permutation test.

        Returns
        -------
        TournamentResult
        """
        entries: List[TournamentEntry] = []
        lookback_days = lookback_months * 21

        for name, returns in strategy_returns.items():
            recent = returns.tail(lookback_days).dropna()
            if len(recent) < 20:
                continue

            entry = self._evaluate_strategy(name, recent)
            entries.append(entry)

        if not entries:
            return TournamentResult(computed_at=datetime.utcnow().isoformat())

        # Compute composite scores
        self._rank_entries(entries)

        # Determine allocation status
        top_strategies = []
        disabled = []

        for entry in entries:
            if entry.sharpe < self.min_sharpe:
                entry.allocation_status = "DISABLED"
                disabled.append(entry.strategy_name)
            elif entry.rank <= self.top_n:
                entry.allocation_status = "ACTIVE"
                top_strategies.append(entry.strategy_name)
            else:
                entry.allocation_status = "REDUCED"

        # Best-of-N selection bias correction via MC permutation
        _corrected_p = None
        _corrected_sig = None
        if strategy_positions and raw_market_returns is not None:
            _corrected_p = self._run_best_of_n(
                strategy_positions, raw_market_returns, entries,
            )
            if _corrected_p is not None:
                _corrected_sig = _corrected_p < 0.05

        return TournamentResult(
            entries=entries,
            top_strategies=top_strategies,
            disabled_strategies=disabled,
            period_months=lookback_months,
            computed_at=datetime.utcnow().isoformat(),
            selection_bias_corrected_p=_corrected_p,
            selection_bias_significant=_corrected_sig,
        )

    def _evaluate_strategy(
        self,
        name: str,
        returns: "pd.Series",
    ) -> TournamentEntry:
        """Compute all metrics for a single strategy."""
        n = len(returns)
        total_ret = float((1 + returns).prod() - 1) * 100

        mean_r = float(returns.mean())
        std_r = float(returns.std())
        sharpe = mean_r / (std_r + 1e-10) * np.sqrt(252)

        downside = returns[returns < 0]
        down_std = float(downside.std()) if len(downside) > 1 else 1e-10
        sortino = mean_r / (down_std + 1e-10) * np.sqrt(252)

        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = float(dd.min()) * 100

        ann_ret = (cum.iloc[-1] ** (252 / n) - 1) if n > 0 and cum.iloc[-1] > 0 else 0
        calmar = ann_ret / (abs(max_dd / 100) + 1e-10)

        win_rate = float((returns > 0).sum() / n) if n > 0 else 0

        return TournamentEntry(
            strategy_name=name,
            sharpe=round(float(sharpe), 3),
            sortino=round(float(sortino), 3),
            calmar=round(float(calmar), 3),
            max_drawdown_pct=round(max_dd, 2),
            total_return_pct=round(total_ret, 2),
            win_rate=round(win_rate, 3),
            n_trades=n,
        )

    def _rank_entries(self, entries: List[TournamentEntry]) -> None:
        """Compute composite score and rank all entries."""
        if not entries:
            return

        # Normalize each metric to [0, 1] range
        metrics = {
            "sharpe": [e.sharpe for e in entries],
            "sortino": [e.sortino for e in entries],
            "calmar": [e.calmar for e in entries],
            "dd": [-e.max_drawdown_pct for e in entries],  # less DD is better
        }

        normalized = {}
        for key, vals in metrics.items():
            mn, mx = min(vals), max(vals)
            rng = mx - mn
            if rng > 0:
                normalized[key] = [(v - mn) / rng for v in vals]
            else:
                normalized[key] = [0.5] * len(vals)

        # Composite score
        for i, entry in enumerate(entries):
            entry.composite_score = (
                self.w_sharpe * normalized["sharpe"][i]
                + self.w_sortino * normalized["sortino"][i]
                + self.w_calmar * normalized["calmar"][i]
                + self.w_dd * normalized["dd"][i]
            )

        # Sort by composite score descending
        entries.sort(key=lambda e: e.composite_score, reverse=True)

        # FIX-08: Compute Deflated Sharpe Ratio (de Prado AFML Ch.14)
        try:
            from services.deflated_sharpe import deflated_sharpe_ratio
            n_trials = len(entries)
            for entry in entries:
                n_obs = max(entry.n_trades * 5, 252)  # rough observation count
                entry.dsr_pvalue = deflated_sharpe_ratio(
                    observed_sr=entry.sharpe,
                    n_obs=n_obs,
                    n_trials=n_trials,
                )
                entry.dsr_significant = entry.dsr_pvalue >= 0.95
        except Exception as exc:
            logger.warning("DSR computation failed: %s", exc)

        for rank, entry in enumerate(entries, 1):
            entry.rank = rank

    def _run_best_of_n(
        self,
        strategy_positions: Optional[Dict[str, "pd.Series"]],
        raw_market_returns: Optional["pd.Series"],
        entries: List[TournamentEntry],
    ) -> Optional[float]:
        """Run best-of-N MC permutation test for selection bias correction.

        Returns the corrected p-value, or None if test can't be run.
        """
        if not strategy_positions or raw_market_returns is None:
            return None

        try:
            from services.mc_permutation_test import MCPermutationTest

            raw_ret = np.asarray(raw_market_returns.dropna().values, dtype=np.float64)
            if len(raw_ret) < 50:
                return None

            pos_vectors = []
            names = []
            for entry in entries:
                name = entry.strategy_name
                if name in strategy_positions:
                    pv = np.asarray(
                        strategy_positions[name].fillna(0).values,
                        dtype=np.float64,
                    )
                    m = min(len(pv), len(raw_ret))
                    if m >= 30:
                        pos_vectors.append(pv[-m:])
                        names.append(name)

            if len(pos_vectors) < 2:
                return None

            # Align all to same length
            min_len = min(len(pv) for pv in pos_vectors)
            min_len = min(min_len, len(raw_ret))
            aligned = [pv[-min_len:] for pv in pos_vectors]

            mc = MCPermutationTest(
                n_perms=2000,   # Fewer for tournament speed
                center_returns=True,
                normalize_time=True,
                seed=42,
            )
            result = mc.test_best_of_n(
                raw_ret[-min_len:], aligned, names,
            )

            logger.info(
                "Tournament best-of-%d: corrected p=%.4f, significant=%s",
                len(aligned), result.corrected_p_value, result.significant,
            )
            return result.corrected_p_value

        except Exception as exc:
            logger.warning("Best-of-N test failed: %s", exc)
            return None
