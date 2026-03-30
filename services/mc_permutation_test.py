"""
Monte Carlo Permutation Test Engine
====================================
Based on Timothy Masters — "Monte-Carlo Evaluation of Trading Systems"
(Testing and Tuning Market Trading Systems, Chapter 7)

Core principle: Permute *positions* across *raw returns* (Fisher-Yates shuffle).
This tests whether the mapping of positions to returns is intelligent,
NOT whether mean return ≠ 0 (that's bootstrap — anti-conservative with skew).

Key algorithms:
  1. Single-system permutation test (position shuffle)
  2. Best-of-N simultaneous permutation (selection bias correction)
  3. Sign-only nonparametric test (variable-duration trades)
  4. Skill vs luck decomposition (centered returns)
  5. Walk-forward factory permutation test

Centurion adaptation: swing/positional trades on IND + US equities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Configuration defaults (overridden by Config class) ─────────────

DEFAULT_N_PERMS = 5000
DEFAULT_SIGNIFICANCE = 0.05
DEFAULT_SEED = 42


# ── Result dataclasses ──────────────────────────────────────────────

@dataclass
class PermutationResult:
    """Result of a single-system MC permutation test."""
    real_metric: float = 0.0
    p_value: float = 1.0
    n_perms: int = 0
    mean_perm: float = 0.0
    std_perm: float = 0.0
    z_score: float = 0.0           # (real - mean_perm) / std_perm
    significant: bool = False       # p_value < significance_level
    significance_level: float = 0.05

    def to_dict(self) -> dict:
        return {
            "real_metric": round(self.real_metric, 6),
            "p_value": round(self.p_value, 4),
            "n_perms": self.n_perms,
            "mean_perm": round(self.mean_perm, 6),
            "std_perm": round(self.std_perm, 6),
            "z_score": round(self.z_score, 3),
            "significant": self.significant,
        }


@dataclass
class BestOfNResult:
    """Result of best-of-N simultaneous permutation test."""
    real_best_metric: float = 0.0
    corrected_p_value: float = 1.0  # Selection-bias-corrected
    naive_p_values: List[float] = field(default_factory=list)
    n_strategies: int = 0
    n_perms: int = 0
    significant: bool = False

    def to_dict(self) -> dict:
        return {
            "real_best_metric": round(self.real_best_metric, 6),
            "corrected_p_value": round(self.corrected_p_value, 4),
            "n_strategies": self.n_strategies,
            "n_perms": self.n_perms,
            "significant": self.significant,
            "naive_p_values": [round(p, 4) for p in self.naive_p_values],
        }


@dataclass
class SkillLuckResult:
    """Skill vs luck decomposition using centered returns."""
    total_return: float = 0.0
    luck_component: float = 0.0
    skill_component: float = 0.0
    skill_fraction: float = 0.0
    luck_fraction: float = 0.0
    p_value: float = 1.0           # Is skill statistically significant?

    def to_dict(self) -> dict:
        return {
            "total_return": round(self.total_return, 6),
            "luck_component": round(self.luck_component, 6),
            "skill_component": round(self.skill_component, 6),
            "skill_fraction": round(self.skill_fraction, 4),
            "luck_fraction": round(self.luck_fraction, 4),
            "p_value": round(self.p_value, 4),
        }


@dataclass
class SignOnlyResult:
    """Result of sign-only nonparametric test for variable-duration trades."""
    real_metric: float = 0.0
    p_value: float = 1.0
    n_perms: int = 0
    significant: bool = False

    def to_dict(self) -> dict:
        return {
            "real_metric": round(self.real_metric, 6),
            "p_value": round(self.p_value, 4),
            "n_perms": self.n_perms,
            "significant": self.significant,
        }


@dataclass
class WalkForwardPermResult:
    """Result of walk-forward factory permutation test."""
    real_oos_metric: float = 0.0
    p_value: float = 1.0
    n_perms: int = 0
    degradation_ratio: float = 0.0
    significant: bool = False

    def to_dict(self) -> dict:
        return {
            "real_oos_metric": round(self.real_oos_metric, 6),
            "p_value": round(self.p_value, 4),
            "n_perms": self.n_perms,
            "degradation_ratio": round(self.degradation_ratio, 4),
            "significant": self.significant,
        }


# ── Core Engine ─────────────────────────────────────────────────────

class MCPermutationTest:
    """Monte Carlo Permutation Test Engine.

    Implements Timothy Masters' position-shuffle approach:
    - Shuffle positions across raw returns (NOT bootstrap)
    - Tests: "Is the position-return pairing intelligent?"
    - Fisher-Yates shuffle for O(n) uniform permutation
    - Time-normalized metrics via sqrt(n) scaling
    """

    def __init__(
        self,
        n_perms: int = DEFAULT_N_PERMS,
        significance_level: float = DEFAULT_SIGNIFICANCE,
        center_returns: bool = True,
        normalize_time: bool = True,
        seed: Optional[int] = DEFAULT_SEED,
    ):
        self.n_perms = n_perms
        self.significance_level = significance_level
        self.center_returns = center_returns
        self.normalize_time = normalize_time
        self.rng = np.random.default_rng(seed)

    # ─── 1. Single-System Permutation Test ──────────────────────────

    def test_single_system(
        self,
        raw_returns: np.ndarray,
        position_vector: np.ndarray,
    ) -> PermutationResult:
        """Test whether a single system's position vector has genuine skill.

        Procedure (Timothy Masters pp.289-290):
          1. Compute real metric = sum(position * raw_returns) / sqrt(n)
          2. For each MC trial: shuffle position vector (Fisher-Yates),
             recompute metric
          3. p-value = fraction of permuted metrics >= real metric

        Parameters
        ----------
        raw_returns : np.ndarray
            Daily raw market returns (e.g. close-to-close pct changes).
        position_vector : np.ndarray
            Daily position vector from the strategy. Values in [-1, 1]
            where 1 = full long, -1 = full short, 0 = flat.

        Returns
        -------
        PermutationResult
        """
        raw_returns = np.asarray(raw_returns, dtype=np.float64).ravel()
        position_vector = np.asarray(position_vector, dtype=np.float64).ravel()

        n = min(len(raw_returns), len(position_vector))
        if n < 30:
            logger.warning("Too few observations (%d) for permutation test", n)
            return PermutationResult()

        returns = raw_returns[:n].copy()
        positions = position_vector[:n].copy()

        # Center returns to remove directional bias (Masters' recommendation)
        if self.center_returns:
            returns = returns - returns.mean()

        # Real metric: mean position-weighted return, time-normalized
        real_metric = self._compute_metric(returns, positions)

        # Monte Carlo permutation loop
        perm_metrics = np.empty(self.n_perms, dtype=np.float64)
        for i in range(self.n_perms):
            shuffled_pos = positions.copy()
            self.rng.shuffle(shuffled_pos)  # Fisher-Yates O(n)
            perm_metrics[i] = self._compute_metric(returns, shuffled_pos)

        p_value = float(np.mean(perm_metrics >= real_metric))
        mean_perm = float(np.mean(perm_metrics))
        std_perm = float(np.std(perm_metrics))
        z_score = (real_metric - mean_perm) / (std_perm + 1e-10)

        return PermutationResult(
            real_metric=float(real_metric),
            p_value=p_value,
            n_perms=self.n_perms,
            mean_perm=mean_perm,
            std_perm=std_perm,
            z_score=z_score,
            significant=p_value < self.significance_level,
            significance_level=self.significance_level,
        )

    # ─── 2. Best-of-N Simultaneous Permutation ─────────────────────

    def test_best_of_n(
        self,
        raw_returns: np.ndarray,
        position_vectors: List[np.ndarray],
        strategy_names: Optional[List[str]] = None,
    ) -> BestOfNResult:
        """Selection-bias-corrected test for the best of N strategies.

        Procedure (Masters pp.294-297):
          1. Compute real metric for ALL N strategies
          2. Real best = max of real metrics
          3. For each MC trial: generate ONE shuffle permutation,
             apply SAME shuffle to ALL N position vectors simultaneously,
             record max of shuffled metrics
          4. p-value = fraction of permuted max-metrics >= real best

        This corrects for "best-of-many" selection bias: if you test N
        strategies and pick the best, the naive p-value is anti-conservative
        by a factor of ~N.

        Parameters
        ----------
        raw_returns : np.ndarray
            Daily raw market returns.
        position_vectors : list[np.ndarray]
            List of position vectors, one per strategy.
        strategy_names : list[str] or None
            Optional names for logging.

        Returns
        -------
        BestOfNResult
        """
        raw_returns = np.asarray(raw_returns, dtype=np.float64).ravel()
        n_strategies = len(position_vectors)

        if n_strategies == 0:
            return BestOfNResult()

        # Ensure all vectors have same length as returns
        n = len(raw_returns)
        positions_matrix = np.zeros((n_strategies, n), dtype=np.float64)
        for i, pv in enumerate(position_vectors):
            pv = np.asarray(pv, dtype=np.float64).ravel()
            m = min(len(pv), n)
            positions_matrix[i, :m] = pv[:m]

        returns = raw_returns.copy()
        if self.center_returns:
            returns = returns - returns.mean()

        # Real metrics for all strategies
        real_metrics = np.array([
            self._compute_metric(returns, positions_matrix[i])
            for i in range(n_strategies)
        ])
        real_best = float(np.max(real_metrics))

        # Naive p-values (for comparison)
        naive_counts = np.zeros(n_strategies, dtype=int)

        # Simultaneous permutation loop
        perm_best_metrics = np.empty(self.n_perms, dtype=np.float64)
        for trial in range(self.n_perms):
            # Generate ONE shuffle index for this trial
            shuffle_idx = self.rng.permutation(n)

            # Apply SAME shuffle to ALL strategies simultaneously
            trial_metrics = np.empty(n_strategies, dtype=np.float64)
            for s in range(n_strategies):
                shuffled_pos = positions_matrix[s][shuffle_idx]
                trial_metrics[s] = self._compute_metric(returns, shuffled_pos)

            perm_best_metrics[trial] = float(np.max(trial_metrics))

            # Track naive per-strategy counts
            for s in range(n_strategies):
                if trial_metrics[s] >= real_metrics[s]:
                    naive_counts[s] += 1

        corrected_p = float(np.mean(perm_best_metrics >= real_best))
        naive_p_values = [float(c / self.n_perms) for c in naive_counts]

        logger.info(
            "Best-of-%d test: corrected p=%.4f (naive best p=%.4f)",
            n_strategies, corrected_p, min(naive_p_values) if naive_p_values else 1.0,
        )

        return BestOfNResult(
            real_best_metric=real_best,
            corrected_p_value=corrected_p,
            naive_p_values=naive_p_values,
            n_strategies=n_strategies,
            n_perms=self.n_perms,
            significant=corrected_p < self.significance_level,
        )

    # ─── 3. Sign-Only Nonparametric Test ────────────────────────────

    def test_sign_only(
        self,
        trade_returns: np.ndarray,
    ) -> SignOnlyResult:
        """Sign-only test for variable-duration swing/positional trades.

        For swing trades with different holding periods, we can't use
        bar-level position shuffling. Instead:
          - Each completed trade has a return (can be +/-)
          - Under H0: each trade's sign is equally likely +/-
          - Permute signs (flip each with 50% probability)
          - Metric: sum of trade returns (sign-preserved vs sign-shuffled)

        This is the nonparametric sign test from Masters' thin position
        vector extension.

        Parameters
        ----------
        trade_returns : np.ndarray
            Array of individual completed trade returns (each a float).
            Positive = profitable trade, negative = losing trade.

        Returns
        -------
        SignOnlyResult
        """
        trade_returns = np.asarray(trade_returns, dtype=np.float64).ravel()
        n_trades = len(trade_returns)

        if n_trades < 10:
            logger.warning("Too few trades (%d) for sign-only test", n_trades)
            return SignOnlyResult()

        # Use absolute values; sign indicates direction
        abs_returns = np.abs(trade_returns)
        real_signs = np.sign(trade_returns)
        real_metric = float(np.sum(abs_returns * real_signs))

        # Permute signs randomly
        perm_metrics = np.empty(self.n_perms, dtype=np.float64)
        for i in range(self.n_perms):
            random_signs = self.rng.choice([-1.0, 1.0], size=n_trades)
            perm_metrics[i] = float(np.sum(abs_returns * random_signs))

        p_value = float(np.mean(perm_metrics >= real_metric))

        return SignOnlyResult(
            real_metric=real_metric,
            p_value=p_value,
            n_perms=self.n_perms,
            significant=p_value < self.significance_level,
        )

    # ─── 4. Skill vs Luck Decomposition ────────────────────────────

    def partition_skill_luck(
        self,
        raw_returns: np.ndarray,
        position_vector: np.ndarray,
    ) -> SkillLuckResult:
        """Decompose total strategy return into skill and luck components.

        Procedure (Masters pp.298-302):
          1. Center returns: r_centered = r - mean(r)
          2. Luck = sum(positions * mean(r))  (directional bias × positions)
          3. Skill = sum(positions * r_centered)  (timing ability)
          4. Verify: total ≈ luck + skill

        For a centered return series, luck → 0 and skill → total.
        The decomposition reveals how much of the return is from
        being "net long in a rising market" vs "timing entries/exits".

        Parameters
        ----------
        raw_returns : np.ndarray
            Daily raw market returns.
        position_vector : np.ndarray
            Daily position vector from the strategy.

        Returns
        -------
        SkillLuckResult
        """
        raw_returns = np.asarray(raw_returns, dtype=np.float64).ravel()
        position_vector = np.asarray(position_vector, dtype=np.float64).ravel()

        n = min(len(raw_returns), len(position_vector))
        if n < 10:
            return SkillLuckResult()

        returns = raw_returns[:n]
        positions = position_vector[:n]

        mean_r = float(returns.mean())
        total = float(np.sum(positions * returns))

        # Luck: return from directional exposure × market drift
        luck = float(np.sum(positions)) * mean_r

        # Skill: return from timing (centered returns × positions)
        centered = returns - mean_r
        skill = float(np.sum(positions * centered))

        abs_total = abs(total) + 1e-10

        # Test skill significance via permutation
        perm_skills = np.empty(self.n_perms, dtype=np.float64)
        for i in range(self.n_perms):
            shuffled_pos = positions.copy()
            self.rng.shuffle(shuffled_pos)
            perm_skills[i] = float(np.sum(shuffled_pos * centered))

        p_value = float(np.mean(perm_skills >= skill))

        return SkillLuckResult(
            total_return=total,
            luck_component=luck,
            skill_component=skill,
            skill_fraction=skill / abs_total,
            luck_fraction=luck / abs_total,
            p_value=p_value,
        )

    # ─── 5. Walk-Forward Factory Permutation Test ───────────────────

    def test_walk_forward_factory(
        self,
        raw_returns: np.ndarray,
        factory_fn,
        train_days: int = 252,
        test_days: int = 63,
    ) -> WalkForwardPermResult:
        """Permutation test for a walk-forward model factory.

        Procedure (Masters pp.291-293):
          1. Run the real WF: train → optimize → test → accumulate OOS
          2. For each MC trial: shuffle returns, run entire WF on shuffled
          3. p-value = fraction of shuffled OOS >= real OOS

        This tests the ENTIRE model building process including
        the optimization step — catches overfitting in the factory.

        Parameters
        ----------
        raw_returns : np.ndarray
            Full return series (must be long enough for multiple folds).
        factory_fn : callable
            factory_fn(train_returns) → position_vector for test period.
            Must return np.ndarray of positions for test_days length.
        train_days : int
            In-sample window size.
        test_days : int
            Out-of-sample window size.

        Returns
        -------
        WalkForwardPermResult
        """
        raw_returns = np.asarray(raw_returns, dtype=np.float64).ravel()
        n = len(raw_returns)
        min_required = train_days + test_days
        if n < min_required:
            logger.warning(
                "Not enough data (%d bars) for WF permutation (need %d)",
                n, min_required,
            )
            return WalkForwardPermResult()

        def _run_wf(returns_series: np.ndarray) -> Tuple[float, float]:
            """Run walk-forward and return (OOS metric, IS metric)."""
            oos_total = 0.0
            is_total = 0.0
            start = 0
            n_folds = 0
            while start + train_days + test_days <= len(returns_series):
                train = returns_series[start: start + train_days]
                test = returns_series[start + train_days: start + train_days + test_days]

                try:
                    positions = factory_fn(train)
                    if positions is None or len(positions) == 0:
                        start += test_days
                        continue
                    positions = np.asarray(positions, dtype=np.float64)
                    m = min(len(positions), len(test))
                    oos_total += float(np.sum(positions[:m] * test[:m]))

                    # IS metric for degradation ratio
                    is_pos = factory_fn(train)
                    if is_pos is not None and len(is_pos) > 0:
                        is_pos = np.asarray(is_pos, dtype=np.float64)
                        is_m = min(len(is_pos), len(train))
                        is_total += float(np.sum(is_pos[:is_m] * train[:is_m]))
                except Exception:
                    pass

                n_folds += 1
                start += test_days

            return oos_total, is_total

        real_oos, real_is = _run_wf(raw_returns)
        deg_ratio = real_oos / (real_is + 1e-10) if real_is != 0 else 0.0

        # Permutation loop: shuffle entire return series, then run WF
        perm_oos = np.empty(self.n_perms, dtype=np.float64)
        for i in range(self.n_perms):
            shuffled = raw_returns.copy()
            self.rng.shuffle(shuffled)
            perm_oos[i], _ = _run_wf(shuffled)

        p_value = float(np.mean(perm_oos >= real_oos))

        return WalkForwardPermResult(
            real_oos_metric=real_oos,
            p_value=p_value,
            n_perms=self.n_perms,
            degradation_ratio=deg_ratio,
            significant=p_value < self.significance_level,
        )

    # ─── Internal helpers ───────────────────────────────────────────

    def _compute_metric(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
    ) -> float:
        """Compute position-weighted return metric.

        Metric = sum(positions * returns) / sqrt(n)  if normalize_time
        else   = mean(positions * returns) * sqrt(252)  (annualized Sharpe-like)
        """
        weighted = positions * returns
        if self.normalize_time:
            # Masters' sqrt(n) normalization for comparable p-values
            n = len(returns)
            return float(np.sum(weighted) / np.sqrt(n))
        else:
            std = float(np.std(weighted))
            if std < 1e-10:
                return 0.0
            return float(np.mean(weighted) / std * np.sqrt(252))
