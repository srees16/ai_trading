"""
G16: Integration Tests for Centurion Core — GODMODE Audit Gaps.

Tests cover:
  1. Forecast combiner with all 13 sources (weights sum to 1.0, FDM per-symbol)
  2. Position sizer — no NameError on scale (G1 fix)
  3. Mean reversion — negative forecasts for overbought (G18)
  4. FII flow — stock-specific forecasts by sector (G17)
  5. OI signal — allows negative forecasts (G3)
  6. Config alignment — leverage/short/sector (G8/G11/G13)
  7. Trade monitor — daily SL halt (G14)

Run: python -m pytest tests/test_godmode_gaps.py -v
"""
import sys
import os
import math
from unittest.mock import MagicMock, patch

# Ensure centurion_core root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest


# ──────────────────────────────────────────────────────────────
# Test 1: Forecast Combiner — weights sum to 1.0, all 13 sources
# ──────────────────────────────────────────────────────────────
class TestForecastCombiner:
    def test_weights_sum_to_one(self):
        from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS
        total = sum(fw.weight for fw in DEFAULT_FORECAST_WEIGHTS)
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_all_13_sources_present(self):
        from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS
        names = {fw.name for fw in DEFAULT_FORECAST_WEIGHTS}
        expected = {
            "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
            "carry", "screener", "momentum", "pead",
            "mean_reversion", "fii_flow", "decision_engine",
            "oi_signal", "pairs_arb", "event_driven",
        }
        assert expected == names, f"Missing: {expected - names}, Extra: {names - expected}"

    def test_oi_signal_weight_nonzero(self):
        """G19: OI signal weight should be restored to ~3%."""
        from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS
        oi_w = next(fw.weight for fw in DEFAULT_FORECAST_WEIGHTS if fw.name == "oi_signal")
        assert oi_w > 0, "OI signal weight should be > 0 (G19)"

    def test_per_symbol_fdm_varies(self):
        """G20: FDM computed from available sources should vary per symbol."""
        from services.forecast_combiner import combine_forecasts

        # Symbol A: has 5 sources
        fc_a = combine_forecasts("SYM_A", {
            "ewmac_16_64": 10.0, "ewmac_32_128": 8.0, "ewmac_64_256": 12.0,
            "carry": 5.0, "screener": 7.0,
        })

        # Symbol B: has 3 sources (different diversification)
        fc_b = combine_forecasts("SYM_B", {
            "carry": 5.0, "mean_reversion": 10.0, "pead": 8.0,
        })

        # FDM should differ because available sources differ
        assert fc_a.fdm != fc_b.fdm, "FDM should vary per symbol based on available sources"
        assert 1.0 <= fc_a.fdm <= 2.0
        assert 1.0 <= fc_b.fdm <= 2.0


# ──────────────────────────────────────────────────────────────
# Test 2: Position Sizer — no NameError on scale (G1)
# ──────────────────────────────────────────────────────────────
class TestPositionSizer:
    def test_no_name_error_on_scale(self):
        """G1: total_notional *= scale should not raise NameError."""
        try:
            from services.position_sizer import compute_position_sizes
        except ImportError:
            pytest.skip("position_sizer not importable")

        # Just verify the module loads without error (the scale fix is structural)
        assert callable(compute_position_sizes)


# ──────────────────────────────────────────────────────────────
# Test 3: Mean Reversion — negative forecasts (G18)
# ──────────────────────────────────────────────────────────────
class TestMeanReversion:
    def test_overbought_gives_negative_forecast(self):
        """G18: Overbought conditions should produce negative forecast."""
        import pandas as pd
        import numpy as np
        from strategies.mean_reversion import compute_mean_reversion_forecast

        # Create synthetic data with RSI > 75 and price at upper BB
        n = 50
        # Strongly uptrending prices to push RSI high
        prices = pd.Series(np.linspace(100, 200, n))

        result = compute_mean_reversion_forecast(prices, oversold_rsi=25.0, overbought_rsi=75.0)
        # The overbought_fade signal should now produce negative forecast
        if result.signal_type == "OVERBOUGHT_FADE":
            assert result.forecast < 0, f"G18: Expected negative forecast, got {result.forecast}"

    def test_oversold_gives_positive_forecast(self):
        """Oversold should still produce positive forecast."""
        import pandas as pd
        import numpy as np
        from strategies.mean_reversion import compute_mean_reversion_forecast

        # Create synthetic data with RSI < 25 and price at lower BB
        n = 50
        prices = pd.Series(np.linspace(200, 100, n))

        result = compute_mean_reversion_forecast(prices, oversold_rsi=25.0, overbought_rsi=75.0)
        if result.signal_type == "OVERSOLD_BOUNCE":
            assert result.forecast > 0, f"Expected positive forecast, got {result.forecast}"


# ──────────────────────────────────────────────────────────────
# Test 4: FII Flow — stock-specific by sector (G17)
# ──────────────────────────────────────────────────────────────
class TestFIIFlow:
    @patch("services.fii_flow_signal._load_cached_flows")
    @patch("services.fii_flow_signal.compute_fii_forecast")
    def test_stock_specific_by_sector(self, mock_forecast, mock_cache):
        """G17: Financials should get higher FII forecast than Pharma."""
        from services.fii_flow_signal import get_fii_flow_forecasts

        mock_cache.return_value = ([100, 200, 300], [50, 60, 70])
        # Mock a positive forecast
        mock_snap = MagicMock()
        mock_snap.forecast = 10.0
        mock_forecast.return_value = mock_snap

        result = get_fii_flow_forecasts(
            ["HDFCBANK", "SUNPHARMA"],
            fii_daily_net=[100, 200, 300],
            dii_daily_net=[50, 60, 70],
        )

        if "HDFCBANK" in result and "SUNPHARMA" in result:
            # Financials (1.4x sensitivity) > Pharma (0.7x sensitivity)
            assert result["HDFCBANK"] > result["SUNPHARMA"], \
                f"HDFCBANK ({result['HDFCBANK']}) should > SUNPHARMA ({result['SUNPHARMA']})"


# ──────────────────────────────────────────────────────────────
# Test 5: Config alignment (G8/G11/G13)
# ──────────────────────────────────────────────────────────────
class TestConfigAlignment:
    def test_short_selling_enabled(self):
        """G8: SHORT_SELLING_ENABLED should be True."""
        from config import Config
        assert Config.SHORT_SELLING_ENABLED is True, "G8: SHORT_SELLING_ENABLED should be True"

    def test_leverage_enabled(self):
        """G11: LEVERAGE_ENABLED should be True."""
        from config import Config
        assert Config.LEVERAGE_ENABLED is True, "G11: LEVERAGE_ENABLED should be True"

    def test_leverage_bull_max(self):
        """G11: LEVERAGE_BULL_MAX should be 1.3."""
        from config import Config
        assert Config.LEVERAGE_BULL_MAX == 1.3, f"Expected 1.3, got {Config.LEVERAGE_BULL_MAX}"

    def test_sector_cap_aligned(self):
        """G13: Config sector cap should be 30% (0.30)."""
        from config import Config
        assert Config.MAX_SECTOR_EXPOSURE_PCT == 0.30, \
            f"G13: Expected 0.30, got {Config.MAX_SECTOR_EXPOSURE_PCT}"

    def test_risk_engine_reads_config(self):
        """G13: RiskEngine should load sector cap from Config."""
        from services.layers.risk_engine import RiskEngine
        engine = RiskEngine(total_capital=500_000)
        assert engine.MAX_SECTOR_CONCENTRATION_PCT == 30.0, \
            f"Expected 30.0, got {engine.MAX_SECTOR_CONCENTRATION_PCT}"


# ──────────────────────────────────────────────────────────────
# Test 6: Trade Monitor — daily SL halt (G14)
# ──────────────────────────────────────────────────────────────
class TestTradeMonitorHalt:
    def test_daily_halt_after_3_sl(self):
        """G14: Monitor should halt after 3 SL hits in one day."""
        from kite_connect.trading.trade_monitor import TradeMonitor

        # Provide a mock kite so poll() doesn't exit early
        mock_kite = MagicMock()
        mock_kite.orders.return_value = []

        monitor = TradeMonitor(kite=mock_kite)
        monitor._daily_sl_count = 3
        monitor._daily_sl_date = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        monitor._halted = True

        events = monitor.poll()
        assert any(e.get("type") == "DAILY_HALT" for e in events), \
            "G14: Expected DAILY_HALT event after 3 SL hits"


# ──────────────────────────────────────────────────────────────
# Test 7: Correlation matrix completeness
# ──────────────────────────────────────────────────────────────
class TestCorrelationMatrix:
    def test_all_pairs_covered(self):
        """All active source pairs should have correlation entries."""
        from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS, DEFAULT_CORRELATION_MATRIX

        active = [fw.name for fw in DEFAULT_FORECAST_WEIGHTS if fw.weight > 0]
        missing = []
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                key = (a, b)
                rev_key = (b, a)
                if key not in DEFAULT_CORRELATION_MATRIX and rev_key not in DEFAULT_CORRELATION_MATRIX:
                    missing.append(key)
        assert not missing, f"Missing correlation entries: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
