"""
Carver Live Forecasts — Generate the 10-source v27 forecast for live trading.

Bridges Gap G1: produces the SAME forecasts as full_pipeline_backtest.py
but from a single day's OHLCV snapshot (suitable for daily rebalancing).

Each function takes ``Dict[str, DataFrame]`` (symbol → OHLCV) and returns
``Dict[str, float]`` (symbol → forecast in [-20, +20]).

Usage::

    from kite_connect.trading.carver_live_forecasts import generate_all_forecasts
    forecasts = generate_all_forecasts(ohlcv_cache)
    # forecasts["RELIANCE.NS"] = {"carver_value": 8.3, "ehlers_dsp": -2.1, ...}
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum bars required for each forecast source
_MIN_BARS = {
    "ewmac_16_64": 270,
    "ewmac_64_256": 270,
    "screener": 50,
    "momentum": 283,
    "mean_reversion": 30,
    "penfold_trend": 60,
    "ehlers_dsp": 50,
    "acceleration": 280,
    "carver_value": 756,
    "breakout": 22,
}


def generate_all_forecasts(
    ohlcv_cache: Dict[str, pd.DataFrame],
    active_sources: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Generate forecasts for all symbols across the 10 active v27 sources.

    Parameters
    ----------
    ohlcv_cache : dict[str, DataFrame]
        ``{symbol: DataFrame}`` with columns ``Open, High, Low, Close, Volume``.
        Must contain enough bars for each source (756 bars for carver_value).
    active_sources : dict[str, float] | None
        ``{source_name: weight}`` filter.  Only sources with weight > 0 are
        computed.  Defaults to v27 champion weights from forecast_combiner.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{symbol: {source_name: forecast_value}}``.
    """
    if active_sources is None:
        active_sources = _get_v27_weights()

    result: Dict[str, Dict[str, float]] = {
        sym: {} for sym in ohlcv_cache
    }

    # Only compute sources with weight > 0
    active = {k for k, v in active_sources.items() if v > 0.005}

    # ── EWMAC (inline, same as backtest lines 687-702) ──
    if "ewmac_16_64" in active or "ewmac_64_256" in active:
        _compute_ewmac(ohlcv_cache, result, active)

    # ── Momentum ──
    if "momentum" in active:
        _compute_momentum(ohlcv_cache, result)

    # ── Mean Reversion ──
    if "mean_reversion" in active:
        _compute_mean_reversion(ohlcv_cache, result)

    # ── Penfold Trend ──
    if "penfold_trend" in active:
        _compute_penfold(ohlcv_cache, result)

    # ── Ehlers DSP ──
    if "ehlers_dsp" in active:
        _compute_ehlers(ohlcv_cache, result)

    # ── Acceleration ──
    if "acceleration" in active:
        _compute_acceleration(ohlcv_cache, result)

    # ── Carver Value ──
    if "carver_value" in active:
        _compute_carver_value(ohlcv_cache, result)

    # ── Breakout (inline, same as backtest lines 740-754) ──
    if "breakout" in active:
        _compute_breakout(ohlcv_cache, result)

    # ── Screener (inline, same as backtest lines 930-951) ──
    if "screener" in active:
        _compute_screener(ohlcv_cache, result)

    return result


# ══════════════════════════════════════════════════════════════
# Individual forecast generators
# ══════════════════════════════════════════════════════════════

def _compute_ewmac(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
    active: set,
):
    """EWMA crossover forecasts (Carver Ch. 7)."""
    try:
        from services.forecast_scalar import ewmac_to_forecast
        from services.instrument_volatility import daily_price_volatility
    except ImportError:
        logger.warning("EWMAC: missing forecast_scalar or instrument_volatility")
        return

    pairs = []
    if "ewmac_16_64" in active:
        pairs.append(("ewmac_16_64", 16, 64))
    if "ewmac_64_256" in active:
        pairs.append(("ewmac_64_256", 64, 256))

    for sym, df in ohlcv.items():
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if len(close) < _MIN_BARS.get("ewmac_64_256", 270):
            continue
        dpv = daily_price_volatility(close)
        if dpv is None or dpv <= 0:
            continue

        for name, fast, slow in pairs:
            try:
                fast_ewma = close.ewm(span=fast, adjust=False).mean()
                slow_ewma = close.ewm(span=slow, adjust=False).mean()
                raw = float(fast_ewma.iloc[-1] - slow_ewma.iloc[-1])
                fc = ewmac_to_forecast(raw, dpv, fast, slow)
                if np.isfinite(fc):
                    result[sym][name] = float(fc)
            except Exception:
                pass


def _compute_momentum(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
):
    """12-minus-1 month cross-sectional momentum."""
    try:
        from services.momentum_factor import compute_momentum_forecasts
        fc_map = compute_momentum_forecasts(ohlcv)
        for sym, fc in fc_map.items():
            if sym in result and np.isfinite(fc):
                result[sym]["momentum"] = float(fc)
    except Exception as e:
        logger.warning("Momentum forecast failed: %s", e)


def _compute_mean_reversion(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
):
    """RSI + Bollinger Band mean reversion."""
    try:
        from strategies.mean_reversion import compute_mean_reversion_batch
        fc_map = compute_mean_reversion_batch(ohlcv)
        for sym, fc in fc_map.items():
            if sym in result and np.isfinite(fc):
                result[sym]["mean_reversion"] = float(fc)
    except Exception as e:
        logger.warning("Mean reversion forecast failed: %s", e)


def _compute_penfold(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
):
    """Penfold trend-following channel."""
    try:
        from strategies.penfold_trend import compute_penfold_forecast_batch
        fc_map = compute_penfold_forecast_batch(ohlcv)
        for sym, fc in fc_map.items():
            if sym in result and np.isfinite(fc):
                result[sym]["penfold_trend"] = float(fc)
    except Exception as e:
        logger.warning("Penfold trend forecast failed: %s", e)


def _compute_ehlers(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
):
    """Ehlers DSP adaptive filter."""
    try:
        from strategies.ehlers_dsp import compute_ehlers_forecast_batch
        fc_map = compute_ehlers_forecast_batch(ohlcv)
        for sym, fc in fc_map.items():
            if sym in result and np.isfinite(fc):
                result[sym]["ehlers_dsp"] = float(fc)
    except Exception as e:
        logger.warning("Ehlers DSP forecast failed: %s", e)


def _compute_acceleration(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
):
    """EWMAC acceleration (second derivative of trend)."""
    try:
        from strategies.acceleration import compute_acceleration_batch
        fc_map = compute_acceleration_batch(ohlcv)
        for sym, fc in fc_map.items():
            if sym in result and np.isfinite(fc):
                result[sym]["acceleration"] = float(fc)
    except Exception as e:
        logger.warning("Acceleration forecast failed: %s", e)


def _compute_carver_value(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
):
    """Carver value (mean reversion over 5-year horizon)."""
    try:
        from strategies.carver_value import compute_value_batch
        fc_map = compute_value_batch(ohlcv)
        for sym, fc in fc_map.items():
            if sym in result and np.isfinite(fc):
                result[sym]["carver_value"] = float(fc)
    except Exception as e:
        logger.warning("Carver value forecast failed: %s", e)


def _compute_breakout(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
):
    """20-day channel breakout (inline, matches backtest lines 740-754)."""
    for sym, df in ohlcv.items():
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if len(close) < 22:
            continue
        try:
            price_now = float(close.iloc[-1])
            lookback = close.iloc[-21:-1]
            high_20 = float(lookback.max())
            low_20 = float(lookback.min())
            rng = high_20 - low_20
            if rng < 1e-10:
                continue
            fc = ((price_now - low_20) / rng - 0.5) * 20.0
            fc = max(-20.0, min(20.0, fc))
            if np.isfinite(fc):
                result[sym]["breakout"] = fc
        except Exception:
            pass


def _compute_screener(
    ohlcv: Dict[str, pd.DataFrame],
    result: Dict[str, Dict[str, float]],
):
    """RSI + 50-day MA slope (inline, matches backtest lines 930-951)."""
    for sym, df in ohlcv.items():
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if len(close) < 50:
            continue
        try:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).ewm(span=14).mean()
            loss = (-delta).where(delta < 0, 0.0).ewm(span=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            ma50 = close.rolling(50).mean()
            ma_slope = float(
                (ma50.iloc[-1] - ma50.iloc[-5]) / (ma50.iloc[-5] + 1e-10)
            ) * 100
            fc = ((rsi - 50) / 5.0) + ma_slope * 2.0
            fc = max(-20.0, min(20.0, fc))
            if np.isfinite(fc):
                result[sym]["screener"] = fc
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _get_v27_weights() -> Dict[str, float]:
    """Load v27 champion weights from forecast_combiner."""
    try:
        from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS
        return {fw.name: fw.weight for fw in DEFAULT_FORECAST_WEIGHTS}
    except ImportError:
        # Fallback: hardcoded v27 champion
        return {
            "carver_value": 0.1878, "ehlers_dsp": 0.1779,
            "momentum": 0.1587, "acceleration": 0.1563,
            "ewmac_64_256": 0.1151, "screener": 0.1015,
            "mean_reversion": 0.0552, "penfold_trend": 0.0180,
            "breakout": 0.0159, "ewmac_16_64": 0.0137,
        }
