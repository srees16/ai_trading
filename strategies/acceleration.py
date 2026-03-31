"""
Acceleration Forecast — Rate of change of trend following forecasts.

Implements Carver AFTS Strategy 23.  Instead of measuring the *direction*
and *speed* of a trend (which EWMAC does), acceleration measures the
*rate of change* of the trend strength.  This catches trend turns faster
than EWMAC alone:

  - When a trend is *accelerating* (getting stronger) → larger position
  - When a trend is *decelerating* (losing strength) → cut position early
  - When a trend is *reversing* → flip sides before EWMAC signals it

Formula:
    raw_accel = EWMAC_forecast(t) − EWMAC_forecast(t − N)
    scaled    = raw_accel × scalar  (calibrated to avg|forecast| ≈ 10)
    forecast  = cap(scaled, ±20)

Book result (Table 107):
    SR 1.29 when combined with trend + carry (vs 1.27 for trend + carry alone)
    Alpha: 24.0% (vs 22.3%)
    Catches trend reversals on avg 3–5 days earlier than EWMAC alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.forecast_scalar import cap_forecast, TARGET_ABS_FORECAST
from services.instrument_volatility import daily_price_volatility

logger = logging.getLogger(__name__)

# Default acceleration look-back pairs: (EWMAC speed, accel horizon)
# For each EWMAC variation, we measure how the forecast changed over N days.
DEFAULT_ACCEL_VARIATIONS: List[Tuple[Tuple[int, int], int]] = [
    ((8, 32), 10),    # Fast EWMAC accel over 10 days
    ((16, 64), 20),   # Core swing accel over 20 days
    ((32, 128), 40),  # Positional accel over 40 days
]

# Pre-calibrated scalars — avg|raw_accel / daily_vol_price| → ≈ 10
# Acceleration signals are noisier than EWMAC; scalars are higher.
ACCEL_SCALARS: Dict[Tuple[int, int], float] = {
    (8, 32): 8.0,
    (16, 64): 5.5,
    (32, 128): 4.0,
}


@dataclass
class AccelerationForecast:
    """Output of a single acceleration variation for one instrument."""
    symbol: str
    ewmac_fast: int
    ewmac_slow: int
    accel_horizon: int
    forecast: float            # −20 to +20
    raw_acceleration: float    # unscaled rate of change
    current_ewmac: float       # current EWMAC forecast value
    prior_ewmac: float         # EWMAC forecast N days ago


def _compute_ewmac_series(
    close: pd.Series,
    fast: int,
    slow: int,
) -> Optional[pd.Series]:
    """Compute the full EWMAC forecast time series (vol-adjusted crossover)."""
    if close is None or len(close) < slow + 10:
        return None

    fast_ewma = close.ewm(span=fast, min_periods=fast).mean()
    slow_ewma = close.ewm(span=slow, min_periods=slow).mean()
    raw_crossover = fast_ewma - slow_ewma

    # Vol-adjust the crossover by daily price volatility
    daily_vol = close.pct_change().rolling(35, min_periods=10).std()
    daily_vol_price = close * daily_vol
    daily_vol_price = daily_vol_price.replace(0, np.nan)

    vol_adjusted = raw_crossover / daily_vol_price
    return vol_adjusted.dropna()


def compute_acceleration(
    close: pd.Series,
    ewmac_fast: int,
    ewmac_slow: int,
    accel_horizon: int,
) -> Optional[AccelerationForecast]:
    """Compute a single acceleration forecast.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices, chronological.
    ewmac_fast, ewmac_slow : int
        EWMAC look-back pair whose derivative we measure.
    accel_horizon : int
        Number of days over which we measure the change in forecast.

    Returns
    -------
    AccelerationForecast or None if insufficient data.
    """
    min_required = ewmac_slow + accel_horizon + 20
    if close is None or len(close) < min_required:
        return None

    ewmac_series = _compute_ewmac_series(close, ewmac_fast, ewmac_slow)
    if ewmac_series is None or len(ewmac_series) < accel_horizon + 5:
        return None

    current_ewmac = float(ewmac_series.iloc[-1])
    prior_ewmac = float(ewmac_series.iloc[-accel_horizon])
    raw_acceleration = current_ewmac - prior_ewmac

    # Scale to target avg|forecast| ≈ 10
    scalar = ACCEL_SCALARS.get((ewmac_fast, ewmac_slow), 5.5)
    scaled = raw_acceleration * scalar
    forecast = cap_forecast(scaled)

    return AccelerationForecast(
        symbol="",
        ewmac_fast=ewmac_fast,
        ewmac_slow=ewmac_slow,
        accel_horizon=accel_horizon,
        forecast=forecast,
        raw_acceleration=raw_acceleration,
        current_ewmac=current_ewmac,
        prior_ewmac=prior_ewmac,
    )


def compute_acceleration_all_variations(
    close: pd.Series,
    variations: Optional[List[Tuple[Tuple[int, int], int]]] = None,
) -> List[AccelerationForecast]:
    """Compute acceleration for all default variations on one instrument."""
    variations = variations or DEFAULT_ACCEL_VARIATIONS
    results = []
    for (fast, slow), horizon in variations:
        fc = compute_acceleration(close, fast, slow, horizon)
        if fc is not None:
            results.append(fc)
    return results


def compute_acceleration_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    variations: Optional[List[Tuple[Tuple[int, int], int]]] = None,
) -> Dict[str, float]:
    """Compute combined acceleration forecast for every symbol.

    Returns a dict of {symbol: combined_acceleration_forecast}.
    The combined value is the equal-weighted average of all variations
    that produce a valid signal, capped at ±20.
    """
    variations = variations or DEFAULT_ACCEL_VARIATIONS
    results: Dict[str, float] = {}

    for sym, df in ohlcv_cache.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        forecasts = compute_acceleration_all_variations(close, variations)

        if not forecasts:
            continue

        for fc in forecasts:
            fc.symbol = sym

        # Equal-weight average across variations, then cap
        avg_forecast = sum(f.forecast for f in forecasts) / len(forecasts)
        results[sym] = cap_forecast(avg_forecast)

    logger.info(
        "Acceleration computed for %d / %d symbols",
        len(results), len(ohlcv_cache),
    )
    return results
