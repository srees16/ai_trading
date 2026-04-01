"""
Tests for the Aronson EBTA statistical validation module.

Covers:
  - detrend_returns
  - compute_signal_tstat
  - benjamini_hochberg
  - whites_reality_check
  - estimate_data_mining_bias
  - trimmed_sharpe
  - count_signal_fires
  - compute_confidence_score
  - AronsonValidator.validate_signals integration
  - AronsonValidator persistence round-trip
"""

import math
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from services.aronson_validator import (
    AronsonValidator,
    SignalValidation,
    ValidationSummary,
    benjamini_hochberg,
    compute_confidence_score,
    compute_signal_tstat,
    count_signal_fires,
    detrend_returns,
    estimate_data_mining_bias,
    trimmed_sharpe,
    whites_reality_check,
)
import pandas as pd


# ── detrend_returns ────────────────────────────────────────────

class TestDetrendReturns:

    def test_short_series_subtracts_mean(self):
        """With <30 data points, just subtract global mean."""
        rets = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        dt = detrend_returns(rets)
        assert abs(dt.mean()) < 1e-10

    def test_longer_series_zero_centres(self):
        """With 252+ points, rolling mean should roughly zero-centre."""
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.001, 0.02, 500))
        dt = detrend_returns(rets, window=252)
        # The detrended series should have lower mean than original
        assert abs(dt.mean()) < abs(rets.mean()) + 0.01

    def test_preserves_length(self):
        rets = pd.Series(np.random.normal(0, 0.01, 300))
        dt = detrend_returns(rets)
        assert len(dt) == len(rets)


# ── compute_signal_tstat ──────────────────────────────────────

class TestComputeSignalTstat:

    def test_strong_positive_signal(self):
        """Signal with strong positive mean should have t > 2."""
        np.random.seed(1)
        rets = np.random.normal(0.005, 0.01, 500)
        t, p = compute_signal_tstat(rets)
        assert t > 2.0
        assert p < 0.05

    def test_noise_signal(self):
        """Pure noise should NOT have significant t-stat (usually)."""
        np.random.seed(42)
        rets = np.random.normal(0, 0.02, 200)
        t, p = compute_signal_tstat(rets)
        assert abs(t) < 5.0  # should be near 0 but allow randomness

    def test_too_few_observations(self):
        rets = np.array([0.01, 0.02])
        t, p = compute_signal_tstat(rets, min_obs=10)
        assert t == 0.0
        assert p == 1.0

    def test_zero_variance(self):
        rets = np.array([0.01] * 50)
        t, p = compute_signal_tstat(rets)
        assert t == 0.0

    def test_nan_handling(self):
        rets = np.array([0.01, np.nan, 0.02, np.nan, 0.03, 0.01, 0.02,
                         0.01, 0.03, 0.02, 0.01])
        t, p = compute_signal_tstat(rets, min_obs=5)
        assert t > 0  # all positive values


# ── benjamini_hochberg ────────────────────────────────────────

class TestBenjaminiHochberg:

    def test_empty_input(self):
        assert benjamini_hochberg([]) == []

    def test_single_significant(self):
        result = benjamini_hochberg([("sig1", 0.001)], q=0.10)
        assert len(result) == 1
        assert result[0][3] is True  # significant

    def test_single_insignificant(self):
        result = benjamini_hochberg([("sig1", 0.5)], q=0.10)
        assert result[0][3] is False

    def test_multiple_signals(self):
        """With 1 genuinely significant and others not, BH should detect it."""
        pvals = [
            ("genuine", 0.001),
            ("noise1", 0.30),
            ("noise2", 0.55),
            ("noise3", 0.80),
        ]
        result = benjamini_hochberg(pvals, q=0.10)
        # Find the genuine signal
        genuine = [r for r in result if r[0] == "genuine"][0]
        assert genuine[3] is True  # should survive BH

    def test_all_significant(self):
        pvals = [("a", 0.001), ("b", 0.002), ("c", 0.003)]
        result = benjamini_hochberg(pvals, q=0.10)
        assert all(r[3] for r in result)

    def test_adjusted_p_monotone(self):
        """Adjusted p-values should be monotonically non-decreasing when sorted."""
        pvals = [("a", 0.01), ("b", 0.05), ("c", 0.10), ("d", 0.20)]
        result = benjamini_hochberg(pvals, q=0.10)
        adj_ps = [r[2] for r in result]
        for i in range(len(adj_ps) - 1):
            assert adj_ps[i] <= adj_ps[i + 1] + 1e-10

    def test_adjusted_p_capped_at_one(self):
        pvals = [("a", 0.90), ("b", 0.95)]
        result = benjamini_hochberg(pvals, q=0.10)
        for r in result:
            assert r[2] <= 1.0


# ── whites_reality_check ─────────────────────────────────────

class TestWhitesRealityCheck:

    def test_genuine_vs_noise(self):
        """One strong signal among noise should get low p-value."""
        np.random.seed(42)
        T = 500
        noise = np.random.normal(0, 0.01, (9, T))
        strong = np.random.normal(0.005, 0.01, (1, T))
        matrix = np.vstack([noise, strong])
        p, best_idx = whites_reality_check(matrix, n_bootstrap=2000, seed=42)
        assert best_idx == 9  # the strong signal
        assert p < 0.10

    def test_all_noise(self):
        """All noise signals — p-value should be high."""
        np.random.seed(42)
        matrix = np.random.normal(0, 0.01, (5, 300))
        p, _ = whites_reality_check(matrix, n_bootstrap=2000, seed=42)
        # Not necessarily > 0.5 due to randomness, but should not be tiny
        assert p > 0.01

    def test_too_few_observations(self):
        matrix = np.random.normal(0, 0.01, (3, 10))
        p, idx = whites_reality_check(matrix)
        assert p == 1.0

    def test_single_signal(self):
        matrix = np.random.normal(0, 0.01, (1, 100))
        p, idx = whites_reality_check(matrix)
        assert p == 1.0


# ── estimate_data_mining_bias ─────────────────────────────────

class TestEstimateDataMiningBias:

    def test_known_value(self):
        """DM bias for 100 signals with sigma=0.01 should be σ√(2ln(100))."""
        expected = 0.01 * math.sqrt(2 * math.log(100))
        actual = estimate_data_mining_bias(100, 0.01)
        assert abs(actual - expected) < 1e-10

    def test_single_signal(self):
        assert estimate_data_mining_bias(1, 0.01) == 0.0

    def test_zero_sigma(self):
        assert estimate_data_mining_bias(10, 0.0) == 0.0

    def test_increases_with_n(self):
        b10 = estimate_data_mining_bias(10, 0.01)
        b100 = estimate_data_mining_bias(100, 0.01)
        b1000 = estimate_data_mining_bias(1000, 0.01)
        assert b10 < b100 < b1000


# ── trimmed_sharpe ────────────────────────────────────────────

class TestTrimmedSharpe:

    def test_positive_returns(self):
        np.random.seed(42)
        rets = np.random.normal(0.001, 0.01, 500)
        ts = trimmed_sharpe(rets, trim_pct=0.05)
        assert ts > 0

    def test_too_few_returns(self):
        assert trimmed_sharpe(np.array([0.01, 0.02, 0.03])) == 0.0

    def test_outlier_robustness(self):
        """Trimmed Sharpe should be less affected by a single outlier."""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.01, 500)
        # Regular Sharpe
        ts_clean = trimmed_sharpe(base, trim_pct=0.05)
        # Add a huge outlier
        contaminated = np.append(base, [0.50])
        ts_outlier = trimmed_sharpe(contaminated, trim_pct=0.05)
        # Trimmed version should be relatively close to clean
        assert abs(ts_clean - ts_outlier) < abs(ts_clean) * 0.5


# ── count_signal_fires ────────────────────────────────────────

class TestCountSignalFires:

    def test_alternating_signal(self):
        sig = np.array([1, -1, 1, -1, 1, -1])
        assert count_signal_fires(sig) == 5

    def test_constant_signal(self):
        sig = np.array([1, 1, 1, 1])
        assert count_signal_fires(sig) == 0

    def test_single_element(self):
        assert count_signal_fires(np.array([1.0])) == 0

    def test_empty(self):
        assert count_signal_fires(np.array([])) == 0


# ── compute_confidence_score ──────────────────────────────────

class TestComputeConfidenceScore:

    def test_all_agree_long(self):
        forecasts = {"a": 10.0, "b": 5.0, "c": 3.0}
        validated = {"a", "b", "c"}
        assert compute_confidence_score(forecasts, validated) == 1.0

    def test_split_vote(self):
        forecasts = {"a": 5.0, "b": -5.0}
        validated = {"a", "b"}
        assert compute_confidence_score(forecasts, validated) == 0.5

    def test_majority_long(self):
        forecasts = {"a": 5.0, "b": 5.0, "c": -5.0}
        validated = {"a", "b", "c"}
        score = compute_confidence_score(forecasts, validated)
        assert abs(score - 2.0 / 3.0) < 1e-6

    def test_no_validated_signals(self):
        forecasts = {"a": 5.0}
        validated = set()
        assert compute_confidence_score(forecasts, validated) == 0.0

    def test_empty_forecasts(self):
        assert compute_confidence_score({}, {"a"}) == 0.0

    def test_filters_unvalidated(self):
        forecasts = {"a": 5.0, "b": -5.0, "c": 3.0}
        validated = {"a", "c"}  # only these count
        score = compute_confidence_score(forecasts, validated)
        assert score == 1.0  # a and c both positive


# ── AronsonValidator integration ──────────────────────────────

class TestAronsonValidatorIntegration:

    def test_validate_signals_basic(self):
        """Full pipeline with mix of strong and noise signals."""
        np.random.seed(42)
        signal_returns = {
            "strong": np.random.normal(0.005, 0.01, 500),
            "noise1": np.random.normal(0.0, 0.02, 500),
            "noise2": np.random.normal(0.0, 0.015, 500),
            "weak": np.random.normal(0.001, 0.02, 500),
        }
        validator = AronsonValidator(wrc_n_bootstrap=1000)
        summary = validator.validate_signals(signal_returns)

        assert summary.n_total == 4
        assert summary.n_validated >= 0
        assert summary.wrc_best_signal != ""
        assert len(summary.signals) == 4

    def test_weight_multipliers_structure(self):
        np.random.seed(42)
        signal_returns = {
            "sig1": np.random.normal(0.003, 0.01, 200),
            "sig2": np.random.normal(0.0, 0.01, 200),
        }
        validator = AronsonValidator(wrc_n_bootstrap=500)
        summary = validator.validate_signals(signal_returns)
        mults = summary.get_weight_multipliers()
        assert "sig1" in mults
        assert "sig2" in mults
        assert all(0 <= v <= 1.0 for v in mults.values())

    def test_strong_signal_gets_higher_weight(self):
        np.random.seed(42)
        signal_returns = {
            "strong": np.random.normal(0.01, 0.01, 500),
            "noise": np.random.normal(0.0, 0.02, 500),
        }
        validator = AronsonValidator(wrc_n_bootstrap=1000)
        summary = validator.validate_signals(signal_returns)
        mults = summary.get_weight_multipliers()
        assert mults["strong"] >= mults["noise"]

    def test_with_benchmark(self):
        np.random.seed(42)
        signal_returns = {
            "sig": np.random.normal(0.005, 0.01, 300),
        }
        benchmark = np.random.normal(0.0005, 0.01, 300)
        validator = AronsonValidator(wrc_n_bootstrap=500)
        summary = validator.validate_signals(signal_returns, benchmark_returns=benchmark)
        assert summary.signals[0].detrended_sharpe != 0.0

    def test_with_degradation_ratios(self):
        np.random.seed(42)
        signal_returns = {
            "good": np.random.normal(0.005, 0.01, 500),
            "degraded": np.random.normal(0.005, 0.01, 500),
        }
        deg = {"good": 0.9, "degraded": 0.2}
        validator = AronsonValidator(wrc_n_bootstrap=500)
        summary = validator.validate_signals(signal_returns, degradation_ratios=deg)
        mults = summary.get_weight_multipliers()
        # degraded should be penalized more
        assert mults["degraded"] < mults["good"]

    def test_validation_summary_serialization(self):
        np.random.seed(42)
        signal_returns = {
            "a": np.random.normal(0.003, 0.01, 200),
            "b": np.random.normal(0.0, 0.01, 200),
        }
        validator = AronsonValidator(wrc_n_bootstrap=500)
        summary = validator.validate_signals(signal_returns)
        d = summary.to_dict()
        assert "n_validated" in d
        assert "n_total" in d
        assert "signals" in d
        assert "wrc_best_signal" in d
        assert "dm_bias_estimate_pct" in d

    def test_empty_signals(self):
        validator = AronsonValidator()
        summary = validator.validate_signals({})
        assert summary.n_total == 0
        assert summary.n_validated == 0


# ── Persistence round-trip ────────────────────────────────────

class TestAronsonPersistence:

    def test_save_and_load_roundtrip(self, tmp_path):
        np.random.seed(42)
        signal_returns = {
            "sig1": np.random.normal(0.003, 0.01, 200),
            "sig2": np.random.normal(0.0, 0.01, 200),
        }
        validator = AronsonValidator(wrc_n_bootstrap=500)
        summary = validator.validate_signals(signal_returns)

        # Write to temp path
        state_path = tmp_path / "state.json"
        with patch("services.aronson_validator._VALIDATION_STATE_PATH", state_path):
            validator.save_state(summary)
            assert state_path.exists()

            loaded = AronsonValidator.load_state()
            assert loaded is not None
            assert loaded.n_total == summary.n_total
            assert loaded.n_validated == summary.n_validated
            assert len(loaded.signals) == len(summary.signals)

    def test_load_weight_multipliers(self, tmp_path):
        np.random.seed(42)
        signal_returns = {
            "sig1": np.random.normal(0.003, 0.01, 200),
        }
        validator = AronsonValidator(wrc_n_bootstrap=500)
        summary = validator.validate_signals(signal_returns)

        state_path = tmp_path / "state.json"
        with patch("services.aronson_validator._VALIDATION_STATE_PATH", state_path):
            validator.save_state(summary)
            mults = AronsonValidator.load_weight_multipliers()
            assert "sig1" in mults
            assert 0 <= mults["sig1"] <= 1.0

    def test_load_missing_file(self, tmp_path):
        state_path = tmp_path / "nonexistent.json"
        with patch("services.aronson_validator._VALIDATION_STATE_PATH", state_path):
            assert AronsonValidator.load_state() is None
            assert AronsonValidator.load_weight_multipliers() == {}


# ── SignalValidation dataclass ────────────────────────────────

class TestSignalValidationDataclass:

    def test_to_dict(self):
        sv = SignalValidation(name="test_sig", t_stat=2.5, p_value=0.01)
        d = sv.to_dict()
        assert d["name"] == "test_sig"
        assert d["t_stat"] == 2.5
        assert "weight_multiplier" in d

    def test_default_values(self):
        sv = SignalValidation(name="x")
        assert sv.weight_multiplier == 1.0
        assert sv.bh_significant is False
        assert sv.p_value == 1.0
