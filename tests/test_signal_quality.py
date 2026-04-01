"""
Tests for Signal Quality Evaluator — regime segmentation, signal metrics,
CAGR estimation, stress testing, and documentation generation.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.signal_quality_evaluator import (
    # Regime segmentation
    classify_regimes,
    classify_regimes_ohlcv,
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_SIDEWAYS,
    _compute_adx_from_close,
    # Signal quality
    SignalRecord,
    compute_signal_quality,
    SignalQualityMetrics,
    # Regime performance
    compute_regime_performance,
    RegimePerformance,
    # Backtest
    PortfolioBacktestResult,
    # CAGR
    estimate_cagr,
    CAGREstimate,
    # Stress tests
    run_stress_tests,
    StressTestResult,
    _stress_result,
    # Doc generators
    _generate_signal_quality_doc,
    _generate_regime_performance_doc,
    _generate_cagr_doc,
    _generate_insights_doc,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_trending_up(n: int = 600, start: float = 100.0) -> pd.Series:
    """Create an uptrending close series."""
    np.random.seed(42)
    drift = 0.0005  # daily upward drift
    noise = np.random.normal(0, 0.01, n)
    log_returns = drift + noise
    prices = start * np.exp(np.cumsum(log_returns))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="Close")


def _make_trending_down(n: int = 600, start: float = 100.0) -> pd.Series:
    """Create a downtrending close series."""
    np.random.seed(42)
    drift = -0.0005
    noise = np.random.normal(0, 0.01, n)
    log_returns = drift + noise
    prices = start * np.exp(np.cumsum(log_returns))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="Close")


def _make_sideways(n: int = 600, start: float = 100.0) -> pd.Series:
    """Create a mean-reverting sideways series."""
    np.random.seed(42)
    prices = [start]
    for _ in range(n - 1):
        mr = -0.01 * (prices[-1] - start)  # mean reversion force
        prices.append(prices[-1] * (1 + mr + np.random.normal(0, 0.005)))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="Close")


def _make_ohlcv(close: pd.Series) -> pd.DataFrame:
    """Create OHLCV DataFrame from close series."""
    np.random.seed(42)
    spread = close * 0.01  # 1% daily range
    return pd.DataFrame({
        "Open": close + np.random.uniform(-0.5, 0.5, len(close)) * spread,
        "High": close + abs(np.random.normal(0, 1, len(close))) * spread,
        "Low": close - abs(np.random.normal(0, 1, len(close))) * spread,
        "Close": close,
        "Volume": np.random.randint(100000, 1000000, len(close)),
    }, index=close.index)


def _make_signals(n: int = 200) -> list:
    """Create synthetic signal records for testing."""
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    regimes = np.random.choice([REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS], n)
    directions = np.random.choice(["BUY", "SELL"], n)
    forecasts = np.random.uniform(-15, 15, n)
    signals = []
    for i in range(n):
        s = SignalRecord(
            ticker="TEST.NS",
            date=dates[i],
            direction=directions[i],
            forecast=forecasts[i],
            confidence=abs(forecasts[i]) / 20,
            regime=regimes[i],
            fwd_5d=np.random.normal(0.002, 0.02),
            fwd_10d=np.random.normal(0.003, 0.03),
            fwd_20d=np.random.normal(0.005, 0.04),
        )
        signals.append(s)
    return signals


# ═══════════════════════════════════════════════════════════════
# 1. Regime Segmentation
# ═══════════════════════════════════════════════════════════════

class TestRegimeClassification:

    def test_classify_regimes_basic(self):
        close = _make_trending_up(600)
        df = classify_regimes(close)
        assert len(df) == len(close)
        assert "regime" in df.columns
        assert "trend_score" in df.columns
        assert "adx" in df.columns
        assert all(r in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS] for r in df["regime"])

    def test_uptrend_classified_as_bull(self):
        # Use a stronger drift to ensure ADX > 20
        np.random.seed(42)
        n = 600
        drift = 0.001  # stronger daily drift
        noise = np.random.normal(0, 0.008, n)
        prices = 100.0 * np.exp(np.cumsum(drift + noise))
        close = pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="B"))
        df = classify_regimes(close)
        tail = df.iloc[-100:]
        bull_pct = (tail["regime"] == REGIME_BULL).mean()
        assert bull_pct > 0.2, f"Expected >20% BULL in strong uptrend, got {bull_pct:.1%}"

    def test_downtrend_classified_as_bear(self):
        close = _make_trending_down(600)
        df = classify_regimes(close)
        tail = df.iloc[-100:]
        bear_pct = (tail["regime"] == REGIME_BEAR).mean()
        assert bear_pct > 0.3, f"Expected >30% BEAR in downtrend, got {bear_pct:.1%}"

    def test_sideways_classified(self):
        close = _make_sideways(600)
        df = classify_regimes(close)
        tail = df.iloc[-100:]
        sideways_pct = (tail["regime"] == REGIME_SIDEWAYS).mean()
        assert sideways_pct > 0.3, f"Expected >30% SIDEWAYS, got {sideways_pct:.1%}"

    def test_classify_insufficient_data(self):
        close = pd.Series([100] * 50, index=pd.date_range("2020-01-01", periods=50))
        with pytest.raises(ValueError, match="Need at least"):
            classify_regimes(close)

    def test_classify_regimes_ohlcv(self):
        close = _make_trending_up(600)
        ohlcv = _make_ohlcv(close)
        df = classify_regimes_ohlcv(ohlcv)
        assert len(df) == len(close)
        assert "regime" in df.columns

    def test_adx_computation(self):
        close = _make_trending_up(300)
        adx = _compute_adx_from_close(close)
        assert len(adx) == len(close)
        assert adx.iloc[-1] > 0  # ADX should be positive


# ═══════════════════════════════════════════════════════════════
# 2. Signal Quality Metrics
# ═══════════════════════════════════════════════════════════════

class TestSignalQuality:

    def test_compute_all_signals(self):
        signals = _make_signals(200)
        m = compute_signal_quality(signals, regime="ALL", direction="ALL", horizon="20D")
        assert m.n_signals > 0
        assert 0 <= m.hit_rate <= 100
        assert m.profit_factor >= 0

    def test_compute_by_regime(self):
        signals = _make_signals(200)
        m_bull = compute_signal_quality(signals, regime=REGIME_BULL, direction="ALL", horizon="20D")
        m_bear = compute_signal_quality(signals, regime=REGIME_BEAR, direction="ALL", horizon="20D")
        assert m_bull.regime == REGIME_BULL
        assert m_bear.regime == REGIME_BEAR
        assert m_bull.n_signals + m_bear.n_signals <= len(signals)

    def test_compute_by_direction(self):
        signals = _make_signals(200)
        m_buy = compute_signal_quality(signals, direction="BUY", horizon="10D")
        m_sell = compute_signal_quality(signals, direction="SELL", horizon="10D")
        assert m_buy.direction == "BUY"
        assert m_sell.direction == "SELL"

    def test_different_horizons(self):
        signals = _make_signals(200)
        m5 = compute_signal_quality(signals, horizon="5D")
        m10 = compute_signal_quality(signals, horizon="10D")
        m20 = compute_signal_quality(signals, horizon="20D")
        assert m5.horizon == "5D"
        assert m10.horizon == "10D"
        assert m20.horizon == "20D"

    def test_empty_signals(self):
        m = compute_signal_quality([], regime="ALL", direction="ALL", horizon="20D")
        assert m.n_signals == 0
        assert m.hit_rate == 0

    def test_perfect_signals(self):
        """All signals are profitable."""
        signals = []
        dates = pd.date_range("2021-01-01", periods=50, freq="B")
        for i in range(50):
            signals.append(SignalRecord(
                ticker="TEST.NS", date=dates[i], direction="BUY",
                forecast=10.0, confidence=0.5, regime=REGIME_BULL,
                fwd_5d=0.02, fwd_10d=0.03, fwd_20d=0.05,
            ))
        m = compute_signal_quality(signals, horizon="20D")
        assert m.hit_rate == 100.0
        assert m.avg_return > 0
        assert m.false_signal_rate == 0.0

    def test_metric_to_row(self):
        m = SignalQualityMetrics(
            regime="BULL", direction="BUY", horizon="20D",
            n_signals=100, hit_rate=55.0, avg_return=0.02,
        )
        row = m.to_row()
        assert "Regime" in row
        assert row["N"] == 100


# ═══════════════════════════════════════════════════════════════
# 3. Regime Performance
# ═══════════════════════════════════════════════════════════════

class TestRegimePerformance:

    def test_compute_regime_performance(self):
        signals = _make_signals(300)
        close = _make_trending_up(500)
        regime_df = classify_regimes(close)
        results = compute_regime_performance(signals, regime_df)
        assert len(results) == 3
        assert all(isinstance(r, RegimePerformance) for r in results)

    def test_regime_performance_fields(self):
        signals = _make_signals(300)
        close = _make_trending_up(500)
        regime_df = classify_regimes(close)
        results = compute_regime_performance(signals, regime_df)
        for rp in results:
            assert rp.regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]
            assert rp.total_signals >= 0
            assert rp.regime_days >= 0

    def test_regime_performance_sums(self):
        signals = _make_signals(300)
        close = _make_trending_up(500)
        regime_df = classify_regimes(close)
        results = compute_regime_performance(signals, regime_df)
        total = sum(rp.total_signals for rp in results)
        assert total == len(signals)


# ═══════════════════════════════════════════════════════════════
# 4. CAGR Estimation
# ═══════════════════════════════════════════════════════════════

class TestCAGREstimation:

    def _make_bt_result(self, ann_ret: float = 0.30) -> PortfolioBacktestResult:
        """Create a synthetic backtest result with ann_ret daily drift."""
        n_days = 504  # ~2 years
        daily_drift = (1 + ann_ret) ** (1 / 252) - 1
        np.random.seed(42)
        daily_rets = daily_drift + np.random.normal(0, 0.015, n_days)

        equity = [500000.0]
        for r in daily_rets:
            equity.append(equity[-1] * (1 + r))

        return PortfolioBacktestResult(
            daily_equity=equity,
            daily_returns=daily_rets.tolist(),
            daily_regimes=[REGIME_BULL] * n_days,
            total_return_pct=round((equity[-1] / equity[0] - 1) * 100, 2),
            annual_return_pct=round(ann_ret * 100, 2),
            sharpe=round(daily_drift / 0.015 * np.sqrt(252), 3),
            max_drawdown_pct=15.0,
            n_trades=500,
        )

    def test_cagr_basic(self):
        bt = self._make_bt_result(0.30)
        est = estimate_cagr(bt)
        assert est.ideal_cagr > 0
        assert est.realistic_cagr > 0
        assert est.conservative_cagr >= 0
        assert est.n_years > 0

    def test_cagr_hierarchy(self):
        """Ideal ≥ Realistic ≥ Conservative."""
        bt = self._make_bt_result(0.40)
        est = estimate_cagr(bt)
        assert est.ideal_cagr >= est.realistic_cagr, \
            f"Ideal ({est.ideal_cagr}) < Realistic ({est.realistic_cagr})"
        assert est.realistic_cagr >= est.conservative_cagr, \
            f"Realistic ({est.realistic_cagr}) < Conservative ({est.conservative_cagr})"

    def test_cagr_bootstrap_ci(self):
        bt = self._make_bt_result(0.30)
        est = estimate_cagr(bt)
        assert est.cagr_ci_90[0] < est.cagr_ci_90[1], "CI lower should be < upper"
        assert len(est.bootstrap_cagrs) > 0

    def test_cagr_overfitting_haircut(self):
        bt = self._make_bt_result(0.30)
        est = estimate_cagr(bt, n_strategies=50)
        assert est.overfitting_haircut_pct > 0, "More strategies should increase haircut"

    def test_cagr_insufficient_data(self):
        bt = PortfolioBacktestResult(
            daily_equity=[100000, 100100],
            daily_returns=[0.001],
        )
        est = estimate_cagr(bt)
        assert est.ideal_cagr == 0  # too short

    def test_cagr_max_dd_conservative(self):
        bt = self._make_bt_result(0.30)
        est = estimate_cagr(bt)
        assert est.conservative_max_dd >= est.realistic_max_dd


# ═══════════════════════════════════════════════════════════════
# 5. Stress Tests
# ═══════════════════════════════════════════════════════════════

class TestStressTesting:

    def test_stress_result_basic(self):
        signals = _make_signals(200)
        r = _stress_result("Test Scenario", signals)
        assert r.scenario == "Test Scenario"
        assert r.n_signals == 200
        assert 0 <= r.hit_rate <= 100

    def test_stress_result_empty(self):
        r = _stress_result("Empty", [])
        assert r.n_signals == 0
        assert r.hit_rate == 0

    def test_run_stress_tests(self):
        signals = _make_signals(300)
        close = _make_trending_up(500)
        regime_df = classify_regimes(close)
        results = run_stress_tests(signals, regime_df)
        assert len(results) >= 3  # At least 3 scenarios
        assert all(isinstance(r, StressTestResult) for r in results)

    def test_stress_scenarios_named(self):
        signals = _make_signals(300)
        close = _make_trending_up(500)
        regime_df = classify_regimes(close)
        results = run_stress_tests(signals, regime_df)
        scenarios = {r.scenario for r in results}
        assert any("High Volatility" in s for s in scenarios)
        assert any("Low Confidence" in s for s in scenarios)


# ═══════════════════════════════════════════════════════════════
# 6. Documentation Generators
# ═══════════════════════════════════════════════════════════════

class TestDocGeneration:

    def test_signal_quality_doc(self):
        signals = _make_signals(200)
        metrics = []
        for h in ["5D", "10D", "20D"]:
            for r in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, "ALL"]:
                for d in ["BUY", "SELL", "ALL"]:
                    metrics.append(compute_signal_quality(signals, r, d, h))
        doc = _generate_signal_quality_doc(metrics, signals)
        assert "# Signal Quality by Regime" in doc
        assert "Hit Rate" in doc
        assert "BULL" in doc

    def test_regime_performance_doc(self):
        signals = _make_signals(200)
        close = _make_trending_up(500)
        regime_df = classify_regimes(close)
        perf = compute_regime_performance(signals, regime_df)
        doc = _generate_regime_performance_doc(perf, regime_df)
        assert "# Regime Performance Breakdown" in doc
        assert "Key Findings" in doc

    def test_cagr_doc(self):
        bt = PortfolioBacktestResult(
            daily_equity=[500000, 550000],
            daily_returns=[0.001] * 252,
            annual_return_pct=28.5,
            sharpe=1.5,
            max_drawdown_pct=15.0,
            n_trades=100,
            avg_positions=5.0,
            total_costs=5000,
            max_dd_duration_days=30,
            regime_returns={"BULL": 35.0},
            regime_sharpes={"BULL": 1.8},
            regime_drawdowns={"BULL": 10.0},
        )
        cagr = CAGREstimate(
            ideal_cagr=35.0, realistic_cagr=28.0, conservative_cagr=20.0,
            ideal_sharpe=1.8, realistic_sharpe=1.5, conservative_sharpe=1.1,
            ideal_max_dd=15.0, realistic_max_dd=18.0, conservative_max_dd=23.0,
            cagr_ci_90=(12.0, 45.0), n_years=2.0, overfitting_haircut_pct=25.0,
        )
        doc = _generate_cagr_doc(cagr, bt, "IND")
        assert "# CAGR Estimation" in doc
        assert "Ideal" in doc
        assert "Conservative" in doc

    def test_insights_doc(self):
        signals = _make_signals(200)
        metrics = [compute_signal_quality(signals, "ALL", "ALL", "20D")]
        close = _make_trending_up(500)
        regime_df = classify_regimes(close)
        perf = compute_regime_performance(signals, regime_df)
        stress = [_stress_result("Test", signals)]
        cagr = CAGREstimate(
            ideal_cagr=30.0, realistic_cagr=25.0, conservative_cagr=18.0,
            cagr_ci_90=(10.0, 40.0), overfitting_haircut_pct=30.0,
        )
        bt = PortfolioBacktestResult(max_drawdown_pct=20.0)
        doc = _generate_insights_doc(metrics, perf, stress, cagr, bt, signals)
        assert "# Signal Insights" in doc
        assert "Recommendations" in doc


# ═══════════════════════════════════════════════════════════════
# 7. Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestSignalRecordDataclass:

    def test_signal_record_creation(self):
        s = SignalRecord(
            ticker="RELIANCE.NS",
            date=pd.Timestamp("2024-01-15"),
            direction="BUY",
            forecast=12.5,
            confidence=0.625,
            regime=REGIME_BULL,
        )
        assert s.ticker == "RELIANCE.NS"
        assert s.direction == "BUY"
        assert np.isnan(s.fwd_5d)  # Not yet filled

    def test_signal_record_with_returns(self):
        s = SignalRecord(
            ticker="TCS.NS",
            date=pd.Timestamp("2024-01-15"),
            direction="SELL",
            forecast=-8.0,
            confidence=0.4,
            regime=REGIME_BEAR,
            fwd_5d=0.01,
            fwd_10d=0.025,
            fwd_20d=0.04,
        )
        assert s.fwd_20d == 0.04


class TestEdgeCases:

    def test_all_winning_signals(self):
        signals = []
        dates = pd.date_range("2021-01-01", periods=100, freq="B")
        for i in range(100):
            signals.append(SignalRecord(
                ticker="WIN.NS", date=dates[i], direction="BUY",
                forecast=15.0, confidence=0.75, regime=REGIME_BULL,
                fwd_20d=0.05,
            ))
        m = compute_signal_quality(signals, horizon="20D")
        assert m.hit_rate == 100.0
        assert m.profit_factor == float('inf')
        assert m.false_signal_rate == 0.0

    def test_all_losing_signals(self):
        signals = []
        dates = pd.date_range("2021-01-01", periods=100, freq="B")
        for i in range(100):
            signals.append(SignalRecord(
                ticker="LOSE.NS", date=dates[i], direction="BUY",
                forecast=5.0, confidence=0.25, regime=REGIME_BEAR,
                fwd_20d=-0.03,
            ))
        m = compute_signal_quality(signals, horizon="20D")
        assert m.hit_rate == 0.0
        assert m.profit_factor == 0.0
        assert m.false_signal_rate == 100.0

    def test_nan_forward_returns_handled(self):
        signals = [
            SignalRecord(
                ticker="NAN.NS", date=pd.Timestamp("2021-01-01"),
                direction="BUY", forecast=10.0, confidence=0.5,
                regime=REGIME_BULL, fwd_5d=np.nan, fwd_10d=np.nan, fwd_20d=np.nan,
            )
        ]
        m = compute_signal_quality(signals, horizon="20D")
        assert m.n_signals == 0  # NaN returns excluded


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
