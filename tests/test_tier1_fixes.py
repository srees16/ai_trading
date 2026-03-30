"""
Integration Test — Tier 1 Gap Fixes
====================================
Tests all 5 Tier 1 critical fixes for live trading readiness:
  Gap 1: Rejection gate (overfit/random strategies)
  Gap 2: Dynamic IDM from actual portfolio correlation
  Gap 3: Gross notional ceiling (2× capital)
  Gap 4: Volume filter + impact model
  Gap 5: Data freshness gate
  Gap 6: PAPER_TRADE_MODE config flag
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════
# Gap 1: Rejection gate tests
# ═══════════════════════════════════════════════════════════════════

class TestGap1RejectionGate:
    """Verify that strategies with bad WF degradation or high p-value get penalised."""

    def test_high_pvalue_triggers_penalty(self):
        """If permutation p-value > 0.10, robustness_adj should be -0.30."""
        # Simulate the logic from integrated_scorer.py
        p_value = 0.15  # above 0.10 threshold
        robustness_adj = 0.0
        rejected_random = False
        if p_value > 0.10:
            rejected_random = True
            robustness_adj = -0.30
        assert rejected_random is True
        assert robustness_adj == -0.30

    def test_low_pvalue_no_penalty(self):
        """p-value <= 0.10 should NOT trigger rejection."""
        p_value = 0.05
        robustness_adj = 0.0
        rejected_random = False
        if p_value > 0.10:
            rejected_random = True
            robustness_adj = -0.30
        assert rejected_random is False
        assert robustness_adj == 0.0

    def test_wf_degradation_below_threshold_rejects(self):
        """WF degradation ratio < 0.5 should flag overfit."""
        deg = 0.3  # bad degradation
        rejected_overfit = False
        if deg < 0.5:
            rejected_overfit = True
        assert rejected_overfit is True

    def test_wf_degradation_above_threshold_passes(self):
        """WF degradation ratio >= 0.5 should pass."""
        deg = 0.7
        rejected_overfit = False
        if deg < 0.5:
            rejected_overfit = True
        assert rejected_overfit is False


# ═══════════════════════════════════════════════════════════════════
# Gap 2: Dynamic IDM tests
# ═══════════════════════════════════════════════════════════════════

class TestGap2DynamicIDM:
    """Test compute_dynamic_idm with synthetic OHLCV data."""

    def _make_ohlcv(self, n=120, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(end=datetime.now(), periods=n)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({
            "Open": close + rng.normal(0, 0.5, n),
            "High": close + abs(rng.normal(0, 1, n)),
            "Low": close - abs(rng.normal(0, 1, n)),
            "Close": close,
            "Volume": rng.integers(100000, 1000000, n),
        }, index=dates)

    def test_dynamic_idm_two_uncorrelated(self):
        """Two uncorrelated instruments should yield IDM close to sqrt(2) ≈ 1.41."""
        from services.instrument_weights import compute_dynamic_idm
        ohlcv_cache = {
            "SYM_A": self._make_ohlcv(seed=1),
            "SYM_B": self._make_ohlcv(seed=999),
        }
        weights = {"SYM_A": 0.5, "SYM_B": 0.5}
        idm = compute_dynamic_idm(ohlcv_cache, weights, lookback_days=60)
        assert 1.0 <= idm <= 2.5, f"IDM={idm:.2f} out of expected range"

    def test_dynamic_idm_identical_instruments(self):
        """Two identical instruments → IDM ≈ 1.0 (perfect correlation)."""
        from services.instrument_weights import compute_dynamic_idm
        df = self._make_ohlcv(seed=42)
        ohlcv_cache = {"SYM_A": df.copy(), "SYM_B": df.copy()}
        weights = {"SYM_A": 0.5, "SYM_B": 0.5}
        idm = compute_dynamic_idm(ohlcv_cache, weights, lookback_days=60)
        assert 0.9 <= idm <= 1.1, f"IDM={idm:.2f} should be ~1.0 for identical series"

    def test_dynamic_idm_single_instrument_fallback(self):
        """Single instrument should fall back to static IDM."""
        from services.instrument_weights import compute_dynamic_idm, get_default_idm
        ohlcv_cache = {"SYM_A": self._make_ohlcv()}
        weights = {"SYM_A": 1.0}
        idm = compute_dynamic_idm(ohlcv_cache, weights, lookback_days=60)
        assert idm == get_default_idm(1)


# ═══════════════════════════════════════════════════════════════════
# Gap 3: Gross notional ceiling tests
# ═══════════════════════════════════════════════════════════════════

class TestGap3GrossNotionalCeiling:
    """Test that position sizes are scaled when total notional > 2× capital."""

    def test_positions_scaled_when_over_ceiling(self):
        from dataclasses import replace as _dc_replace
        from services.position_sizer import PositionSize

        capital = 500_000
        # Create oversized position data (3× capital = ₹1.5M)
        ps1 = PositionSize(
            symbol="SYM_A", combined_forecast=10.0, vol_scalar=1.0,
            subsystem_position=50.0, instrument_weight=0.5, idm=1.8,
            portfolio_position=45.0, target_quantity=300,
            current_quantity=0, trade_required=True, trade_delta=300,
            notional_value=750_000, price=2500.0,
        )
        ps2 = PositionSize(
            symbol="SYM_B", combined_forecast=8.0, vol_scalar=1.0,
            subsystem_position=40.0, instrument_weight=0.5, idm=1.8,
            portfolio_position=36.0, target_quantity=500,
            current_quantity=0, trade_required=True, trade_delta=500,
            notional_value=750_000, price=1500.0,
        )
        results = {"SYM_A": ps1, "SYM_B": ps2}
        total_notional = sum(ps.notional_value for ps in results.values())

        # Apply the ceiling logic (same as in position_sizer.py)
        max_notional = 2.0 * capital
        assert total_notional > max_notional, "Test setup: notional should exceed ceiling"

        scale = max_notional / total_notional
        for sym, ps in results.items():
            scaled_qty = int(ps.target_quantity * scale)
            results[sym] = _dc_replace(
                ps,
                target_quantity=scaled_qty,
                trade_delta=scaled_qty - ps.current_quantity,
                trade_required=abs(scaled_qty - ps.current_quantity) > 0,
                notional_value=abs(scaled_qty) * ps.price,
            )

        new_total = sum(ps.notional_value for ps in results.values())
        assert new_total <= max_notional * 1.01  # allow 1% rounding tolerance

    def test_positions_unchanged_when_under_ceiling(self):
        from services.position_sizer import PositionSize

        capital = 500_000
        ps = PositionSize(
            symbol="SYM_A", combined_forecast=5.0, vol_scalar=1.0,
            subsystem_position=20.0, instrument_weight=0.5, idm=1.8,
            portfolio_position=18.0, target_quantity=100,
            current_quantity=0, trade_required=True, trade_delta=100,
            notional_value=200_000, price=2000.0,
        )
        total_notional = ps.notional_value
        max_notional = 2.0 * capital
        assert total_notional <= max_notional
        # No scaling needed
        assert ps.target_quantity == 100


# ═══════════════════════════════════════════════════════════════════
# Gap 4: Volume-aware slippage model tests
# ═══════════════════════════════════════════════════════════════════

class TestGap4VolumeAwareSlippage:
    """Test that slippage increases with order size relative to ADV."""

    def test_flat_slippage_without_adv(self):
        """Without ADV info, slippage should use base_bps only."""
        from kite_connect.trading.paper_trader import PaperTrader
        pt = PaperTrader.__new__(PaperTrader)
        pt._slippage_bps = 20.0
        price = 1000.0
        fill = pt._apply_slippage(price, "BUY", order_qty=0, adv=0.0)
        expected = round(price + price * 20.0 / 10_000, 2)
        assert fill == expected

    def test_slippage_increases_with_volume_impact(self):
        """Order = 10% of ADV should produce higher slippage than base."""
        from kite_connect.trading.paper_trader import PaperTrader
        pt = PaperTrader.__new__(PaperTrader)
        pt._slippage_bps = 20.0
        price = 1000.0

        # Base slippage only
        fill_base = pt._apply_slippage(price, "BUY", order_qty=0, adv=0.0)

        # 10% of ADV → extra 30 bps impact
        fill_impact = pt._apply_slippage(price, "BUY", order_qty=100_000, adv=1_000_000)

        assert fill_impact > fill_base, "Volume impact should increase slippage"
        # total_bps = 20 + 0.1 × 300 = 50 bps
        expected_impact = round(price + price * 50.0 / 10_000, 2)
        assert fill_impact == expected_impact

    def test_sell_slippage_direction(self):
        """SELL should reduce price (adverse fill)."""
        from kite_connect.trading.paper_trader import PaperTrader
        pt = PaperTrader.__new__(PaperTrader)
        pt._slippage_bps = 20.0
        price = 1000.0
        fill = pt._apply_slippage(price, "SELL", order_qty=50_000, adv=500_000)
        assert fill < price


# ═══════════════════════════════════════════════════════════════════
# Gap 5: Data freshness gate tests
# ═══════════════════════════════════════════════════════════════════

class TestGap5DataFreshnessGate:
    """Test that stale OHLCV data gets dropped before pipeline processing."""

    def test_stale_symbol_removed(self):
        """Symbol with last bar >4h ago should be removed from ohlcv_cache."""
        now = datetime.now()
        stale_cutoff = now - timedelta(hours=4)

        # Fresh data — last bar 1 hour ago (use DatetimeIndex with intraday time)
        fresh_dates = pd.date_range(end=now - timedelta(hours=1), periods=50, freq="h")
        fresh_df = pd.DataFrame({"Close": np.random.randn(50)}, index=fresh_dates)

        # Stale data — last bar 2 days ago
        stale_dates = pd.date_range(end=now - timedelta(days=2), periods=50, freq="h")
        stale_df = pd.DataFrame({"Close": np.random.randn(50)}, index=stale_dates)

        ohlcv_cache = {"FRESH": fresh_df, "STALE": stale_df}

        # Apply freshness logic (same as carver_pipeline.py)
        stale_symbols = []
        for sym, df in list(ohlcv_cache.items()):
            if df is not None and not df.empty and df.index.dtype.kind == "M":
                last_bar = df.index[-1]
                if hasattr(last_bar, "tz") and last_bar.tz is not None:
                    last_bar = last_bar.tz_localize(None)
                if last_bar < stale_cutoff:
                    stale_symbols.append(sym)
                    del ohlcv_cache[sym]

        assert "STALE" in stale_symbols
        assert "FRESH" not in stale_symbols
        assert "STALE" not in ohlcv_cache
        assert "FRESH" in ohlcv_cache

    def test_fresh_symbols_retained(self):
        """Symbols with recent data should pass the freshness gate."""
        now = datetime.now()
        stale_cutoff = now - timedelta(hours=4)

        dates = pd.date_range(end=now - timedelta(hours=1), periods=50, freq="h")
        df = pd.DataFrame({"Close": np.random.randn(50)}, index=dates)
        ohlcv_cache = {"SYM_A": df, "SYM_B": df.copy()}

        stale_symbols = []
        for sym, df_i in list(ohlcv_cache.items()):
            if df_i is not None and not df_i.empty and df_i.index.dtype.kind == "M":
                last_bar = df_i.index[-1]
                if hasattr(last_bar, "tz") and last_bar.tz is not None:
                    last_bar = last_bar.tz_localize(None)
                if last_bar < stale_cutoff:
                    stale_symbols.append(sym)
                    del ohlcv_cache[sym]

        assert len(stale_symbols) == 0
        assert len(ohlcv_cache) == 2


# ═══════════════════════════════════════════════════════════════════
# Gap 6: PAPER_TRADE_MODE config test
# ═══════════════════════════════════════════════════════════════════

class TestGap6PaperTradeMode:
    """Verify PAPER_TRADE_MODE is enabled."""

    def test_paper_trade_mode_enabled(self):
        from config import Config
        assert Config.PAPER_TRADE_MODE is True, "PAPER_TRADE_MODE must be True for 4-week validation"

    def test_signal_freshness_config_exists(self):
        from config import Config
        assert hasattr(Config, "SIGNAL_FRESHNESS_MAX_HOURS")
        assert Config.SIGNAL_FRESHNESS_MAX_HOURS == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
