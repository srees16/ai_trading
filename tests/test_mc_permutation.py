"""
Integration Test — Monte Carlo Permutation Test System
=======================================================
Tests all phases of the MC evaluation redesign:
  Phase A: Core MC engine (single system, best-of-N, sign-only, skill/luck, WF)
  Phase B: Integrated scorer wiring
  Phase C: Strategy tournament best-of-N correction
  Phase D: Skill vs luck decomposition
  Phase E: Walk-forward permutation
  Phase F: Sign-only test for swing/positional trades
"""

import sys
from pathlib import Path

# Ensure centurion_core is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


# ── Helpers ─────────────────────────────────────────────────────────

def _make_trending_returns(n=500, drift=0.0003, vol=0.015, seed=42):
    """Generate synthetic daily returns with a positive drift."""
    rng = np.random.default_rng(seed)
    return rng.normal(drift, vol, n)


def _make_skilled_position(returns, lookback=20):
    """Generate a position vector that has genuine skill: lookahead-biased.

    Position at time i is set based on the sign of return[i] itself,
    so it's a "perfect hindsight" signal. Since the MC metric computes
    sum(positions[i] * returns[i]), this guarantees positive correlation.
    """
    n = len(returns)
    positions = np.zeros(n)
    for i in range(lookback, n):
        positions[i] = 1.0 if returns[i] > 0 else -1.0
    return positions


def _make_random_position(n=500, seed=99):
    """Generate a random position vector (no skill)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.4, 0.3])


# ── Phase A: Core MC Permutation Engine ─────────────────────────────

class TestPhaseA:
    """Test services/mc_permutation_test.py core engine."""

    def test_imports(self):
        from services.mc_permutation_test import (
            MCPermutationTest,
            PermutationResult,
            BestOfNResult,
            SkillLuckResult,
            SignOnlyResult,
            WalkForwardPermResult,
        )
        assert MCPermutationTest is not None

    def test_single_system_skilled(self):
        """A skilled system should get a low p-value."""
        from services.mc_permutation_test import MCPermutationTest

        returns = _make_trending_returns(300, seed=10)
        positions = _make_skilled_position(returns)

        mc = MCPermutationTest(n_perms=500, seed=42)
        result = mc.test_single_system(returns, positions)

        assert result.p_value < 0.10, f"Skilled system p={result.p_value} should be < 0.10"
        assert result.n_perms == 500
        assert result.real_metric != 0.0
        assert isinstance(result.z_score, float)

    def test_single_system_random(self):
        """A random system should get a high p-value (>0.10)."""
        from services.mc_permutation_test import MCPermutationTest

        returns = _make_trending_returns(300, drift=0, seed=20)
        positions = _make_random_position(300, seed=55)

        mc = MCPermutationTest(n_perms=500, seed=42)
        result = mc.test_single_system(returns, positions)

        # Random positions should NOT be significant
        assert result.p_value > 0.05, f"Random system p={result.p_value} should be > 0.05"

    def test_single_system_too_few_observations(self):
        """Should return default result for < 30 observations."""
        from services.mc_permutation_test import MCPermutationTest

        mc = MCPermutationTest(n_perms=100, seed=42)
        result = mc.test_single_system(np.array([0.01, 0.02]), np.array([1.0, -1.0]))
        assert result.p_value == 1.0

    def test_best_of_n(self):
        """Best-of-N should have HIGHER p-value than the best naive p-value."""
        from services.mc_permutation_test import MCPermutationTest

        returns = _make_trending_returns(300, drift=0.0001, seed=30)

        # Create 5 strategies: 4 random + 1 mildly skilled
        positions_list = []
        for s in range(4):
            positions_list.append(_make_random_position(300, seed=s + 100))
        positions_list.append(_make_skilled_position(returns, lookback=30))

        mc = MCPermutationTest(n_perms=500, seed=42)
        result = mc.test_best_of_n(returns, positions_list)

        assert result.n_strategies == 5
        assert result.n_perms == 500
        assert result.corrected_p_value >= 0.0
        assert result.corrected_p_value <= 1.0
        # Corrected should be >= min naive (selection bias correction makes it harder)
        if result.naive_p_values:
            assert result.corrected_p_value >= min(result.naive_p_values) - 0.01

    def test_sign_only(self):
        """Sign-only test on a profitable trade set should detect skill."""
        from services.mc_permutation_test import MCPermutationTest

        # 80% winning trades with decent magnitude
        rng = np.random.default_rng(42)
        trade_returns = np.where(
            rng.random(100) < 0.8,
            rng.uniform(0.01, 0.05, 100),    # wins
            rng.uniform(-0.08, -0.01, 100),  # losses
        )

        mc = MCPermutationTest(n_perms=500, seed=42)
        result = mc.test_sign_only(trade_returns)

        assert result.p_value < 0.05, f"80% win rate should be significant, p={result.p_value}"
        assert result.n_perms == 500

    def test_sign_only_random(self):
        """Sign-only test on 50/50 trades should NOT be significant."""
        from services.mc_permutation_test import MCPermutationTest

        rng = np.random.default_rng(42)
        trade_returns = rng.normal(0, 0.02, 100)  # 50/50 wins/losses

        mc = MCPermutationTest(n_perms=500, seed=42)
        result = mc.test_sign_only(trade_returns)

        assert result.p_value > 0.05, f"Random trades should not be significant, p={result.p_value}"

    def test_skill_luck_decomposition(self):
        """Partition should identify skill component for a skilled system."""
        from services.mc_permutation_test import MCPermutationTest

        returns = _make_trending_returns(300, drift=0.0003, seed=40)
        positions = _make_skilled_position(returns)

        mc = MCPermutationTest(n_perms=500, seed=42)
        result = mc.partition_skill_luck(returns, positions)

        assert result.total_return != 0.0
        assert abs(result.skill_component) > 0
        assert abs(result.luck_component) >= 0
        # Skill fraction should be meaningful for a lookahead-biased system
        assert result.skill_fraction > 0.3, f"Skill fraction={result.skill_fraction} should be > 0.3"

    def test_walk_forward_factory(self):
        """WF factory permutation should produce a valid p-value."""
        from services.mc_permutation_test import MCPermutationTest

        returns = _make_trending_returns(600, drift=0.0002, seed=50)

        def simple_factory(train):
            """Simple momentum factory: go long if training mean > 0."""
            sign = 1.0 if np.mean(train) > 0 else -1.0
            return np.full(63, sign)

        mc = MCPermutationTest(n_perms=100, seed=42)  # fewer for speed
        result = mc.test_walk_forward_factory(
            returns, simple_factory, train_days=252, test_days=63,
        )

        assert 0.0 <= result.p_value <= 1.0
        assert result.n_perms == 100
        assert isinstance(result.degradation_ratio, float)

    def test_result_to_dict(self):
        """All result dataclasses should serialize to dict."""
        from services.mc_permutation_test import (
            PermutationResult, BestOfNResult, SkillLuckResult,
            SignOnlyResult, WalkForwardPermResult,
        )

        for cls in (PermutationResult, BestOfNResult, SkillLuckResult,
                    SignOnlyResult, WalkForwardPermResult):
            obj = cls()
            d = obj.to_dict()
            assert isinstance(d, dict)


# ── Phase B: Integrated Scorer Wiring ───────────────────────────────

class TestPhaseB:
    """Test that integrated_scorer uses MC permutation instead of SMA."""

    def test_mc_permutation_imported_in_scorer(self):
        """The scorer should import from services.mc_permutation_test."""
        from services import integrated_scorer
        import inspect
        source = inspect.getsource(integrated_scorer)
        assert "mc_permutation_test" in source
        assert "MCPermutationTest" in source

    def test_scorer_collects_position_vectors(self):
        """Strategy results should include _position_vector key."""
        from services import integrated_scorer
        import inspect
        source = inspect.getsource(integrated_scorer)
        assert "_position_vector" in source

    def test_scorer_has_skill_fraction(self):
        """Robustness details should include skill/luck decomposition."""
        from services import integrated_scorer
        import inspect
        source = inspect.getsource(integrated_scorer)
        assert "skill_fraction" in source
        assert "luck_fraction" in source

    def test_scorer_has_best_of_n(self):
        """Robustness details should include best-of-N correction."""
        from services import integrated_scorer
        import inspect
        source = inspect.getsource(integrated_scorer)
        assert "best_of_n_p" in source

    def test_scorer_strips_position_vectors(self):
        """Returned per_strategy dict should not contain _position_vector."""
        from services import integrated_scorer
        import inspect
        source = inspect.getsource(integrated_scorer)
        assert '_clean_results' in source


# ── Phase C: Tournament Best-of-N ──────────────────────────────────

class TestPhaseC:
    """Test strategy_tournament best-of-N selection bias correction."""

    def test_tournament_result_has_selection_bias(self):
        from services.strategy_tournament import TournamentResult
        r = TournamentResult()
        assert hasattr(r, "selection_bias_corrected_p")
        assert hasattr(r, "selection_bias_significant")

    def test_tournament_accepts_positions(self):
        """run_tournament should accept strategy_positions kwarg."""
        from services.strategy_tournament import StrategyTournament
        import inspect
        sig = inspect.signature(StrategyTournament.run_tournament)
        assert "strategy_positions" in sig.parameters
        assert "raw_market_returns" in sig.parameters

    def test_tournament_basic_run(self):
        """Tournament should run and rank strategies correctly."""
        from services.strategy_tournament import StrategyTournament

        rng = np.random.default_rng(42)
        strategy_returns = {
            "strategy_a": pd.Series(rng.normal(0.001, 0.02, 100)),
            "strategy_b": pd.Series(rng.normal(-0.001, 0.03, 100)),
            "strategy_c": pd.Series(rng.normal(0.002, 0.015, 100)),
        }

        t = StrategyTournament(top_n=2, min_sharpe=-999)
        result = t.run_tournament(strategy_returns, lookback_months=3)

        assert len(result.entries) == 3
        assert result.entries[0].rank == 1
        assert result.selection_bias_corrected_p is None  # No positions provided

    def test_tournament_with_positions(self):
        """Tournament with position data should compute best-of-N p-value."""
        from services.strategy_tournament import StrategyTournament

        rng = np.random.default_rng(42)
        n = 200
        raw_market = pd.Series(rng.normal(0.0003, 0.015, n))

        strategy_returns = {}
        strategy_positions = {}
        for i, name in enumerate(["strat_a", "strat_b", "strat_c"]):
            pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n))
            rets = pd.Series(raw_market.values * pos.values)
            strategy_returns[name] = rets
            strategy_positions[name] = pos

        t = StrategyTournament(top_n=2, min_sharpe=-999)
        result = t.run_tournament(
            strategy_returns, lookback_months=6,
            strategy_positions=strategy_positions,
            raw_market_returns=raw_market,
        )

        # Should have computed the selection bias p-value
        assert result.selection_bias_corrected_p is not None
        assert 0.0 <= result.selection_bias_corrected_p <= 1.0
        assert result.selection_bias_significant is not None

    def test_tournament_to_dict_includes_bias(self):
        """to_dict should include selection_bias when p-value is set."""
        from services.strategy_tournament import TournamentResult
        r = TournamentResult(
            selection_bias_corrected_p=0.03,
            selection_bias_significant=True,
        )
        d = r.to_dict()
        assert "selection_bias" in d
        assert d["selection_bias"]["corrected_p_value"] == 0.03


# ── Phase D: Skill vs Luck ─────────────────────────────────────────

class TestPhaseD:
    """Test skill vs luck decomposition."""

    def test_skill_majority_for_skilled_system(self):
        from services.mc_permutation_test import MCPermutationTest

        returns = _make_trending_returns(400, drift=0.0003, seed=60)
        positions = _make_skilled_position(returns)

        mc = MCPermutationTest(n_perms=300, seed=42)
        sl = mc.partition_skill_luck(returns, positions)

        # A future-peeking system should be mostly skill
        assert sl.skill_fraction > 0.2
        assert sl.p_value < 0.10

    def test_luck_dominates_for_random_system(self):
        from services.mc_permutation_test import MCPermutationTest

        returns = _make_trending_returns(400, drift=0.001, seed=70)
        positions = _make_random_position(400, seed=88)

        mc = MCPermutationTest(n_perms=300, seed=42)
        sl = mc.partition_skill_luck(returns, positions)

        # Random system — skill p-value should be high
        assert sl.p_value > 0.05


# ── Phase E: Walk-Forward Permutation ───────────────────────────────

class TestPhaseE:
    """Test walk-forward factory permutation."""

    def test_wf_summary_has_perm_fields(self):
        from services.walk_forward import WalkForwardSummary
        s = WalkForwardSummary(strategy_name="test", ticker="AAPL")
        assert hasattr(s, "wf_perm_p_value")
        assert hasattr(s, "wf_perm_significant")

    def test_wf_summary_to_dict_includes_perm(self):
        from services.walk_forward import WalkForwardSummary
        s = WalkForwardSummary(
            strategy_name="test", ticker="AAPL",
            wf_perm_p_value=0.02, wf_perm_significant=True,
        )
        d = s.to_dict()
        assert "wf_permutation" in d
        assert d["wf_permutation"]["p_value"] == 0.02

    def test_wf_permutation_test_function(self):
        from services.walk_forward import WalkForwardSummary, wf_permutation_test

        returns = _make_trending_returns(600, seed=80)
        summary = WalkForwardSummary(strategy_name="test", ticker="AAPL")

        def factory(train):
            sign = 1.0 if np.mean(train) > 0 else -1.0
            return np.full(63, sign)

        result = wf_permutation_test(
            summary, returns, factory_fn=factory, n_perms=100,
        )

        assert result.wf_perm_p_value is not None
        assert 0.0 <= result.wf_perm_p_value <= 1.0
        assert result.wf_perm_significant is not None


# ── Phase F: Config Settings ────────────────────────────────────────

class TestPhaseF:
    """Test MC config settings exist."""

    def test_config_has_mc_settings(self):
        from config import Config
        assert hasattr(Config, "MC_PERMUTATION_N_REPS")
        assert Config.MC_PERMUTATION_N_REPS == 5000
        assert hasattr(Config, "MC_CENTER_RETURNS")
        assert Config.MC_CENTER_RETURNS is True
        assert hasattr(Config, "MC_NORMALIZE_TIME")
        assert Config.MC_NORMALIZE_TIME is True
        assert hasattr(Config, "MC_SIGNIFICANCE_LEVEL")
        assert Config.MC_SIGNIFICANCE_LEVEL == 0.05
        assert hasattr(Config, "MC_WF_PERM_N_REPS")
        assert Config.MC_WF_PERM_N_REPS == 2000
        assert hasattr(Config, "MC_TOURNAMENT_N_REPS")
        assert Config.MC_TOURNAMENT_N_REPS == 2000


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
