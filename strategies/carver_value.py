"""
Value Forecast — 5-Year Slow Mean Reversion (Carver AFTS Strategy 22).

Concept: Instruments that have underperformed relative to their
long-term average are expected to revert — "buy low, sell high".

For equities, this is a simplified P/E proxy: since earnings are
relatively stable, price changes drive most of the P/E movement.
So a value strategy ≈ buying stocks whose prices have fallen
relative to their 5-year average.

Formula:
    relative_price = close / SMA(close, lookback)
    raw_value      = −(relative_price − 1.0)  # Negative when above avg → sell
    vol_adjusted   = raw_value / annual_vol
    forecast       = scale_and_cap(vol_adjusted × scalar, ±20)

Properties:
    - Very low turnover (holding period: months to years)
    - Standalone SR ≈ 0.13 (weak alone)
    - Correlation with trend ≈ −0.20 (excellent diversifier)
    - Equity value factor: historically delivers ~3% annual premium
    - WARNING: Can underperform for years at a stretch (2017–2020)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from services.forecast_scalar import cap_forecast, TARGET_ABS_FORECAST
from services.instrument_volatility import daily_price_volatility

logger = logging.getLogger(__name__)

# 5 years of trading days (≈ 252/yr × 5)
DEFAULT_VALUE_LOOKBACK = 1260

# Minimum data requirement — need at least 3 years for reasonable signal
MIN_DATA_DAYS = 756

# Pre-calibrated scalar: maps typical raw_value/vol to avg|forecast| ≈ 10
# For equities, relative_price std ≈ 0.15–0.30, vol ≈ 0.20–0.40
# So vol_adjusted std ≈ 0.50, median_abs ≈ 0.40 → scalar ≈ 10/0.40 = 25
SCALAR_VALUE = 25.0


@dataclass
class ValueForecast:
    """Output of the value signal for one instrument."""
    symbol: str
    forecast: float           # −20 to +20
    relative_price: float     # current / SMA(lookback)
    raw_value: float          # −(relative_price − 1)
    lookback_days: int        # actual lookback used


def compute_value_forecast(
    close: pd.Series,
    lookback: int = DEFAULT_VALUE_LOOKBACK,
) -> Optional[ValueForecast]:
    """Compute value (slow mean reversion) forecast for one instrument.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices, chronological.
    lookback : int
        SMA window in trading days (default 1260 ≈ 5 years).

    Returns
    -------
    ValueForecast or None if insufficient data.
    """
    if close is None or len(close) < MIN_DATA_DAYS:
        return None

    # Use available data, but cap at requested lookback
    effective_lookback = min(lookback, len(close))
    sma = close.rolling(window=effective_lookback, min_periods=MIN_DATA_DAYS).mean()
    current_sma = float(sma.iloc[-1])

    if np.isnan(current_sma) or current_sma <= 0:
        return None

    current_price = float(close.iloc[-1])
    relative_price = current_price / current_sma

    # Negative when above average (=overvalued → sell signal)
    # Positive when below average (=undervalued → buy signal)
    raw_value = -(relative_price - 1.0)

    # Vol-adjust: divide by annualized volatility
    annual_vol = daily_price_volatility(close) * np.sqrt(252)
    if annual_vol <= 0.01:
        annual_vol = 0.25  # fallback 25% vol

    vol_adjusted = raw_value / annual_vol
    scaled = vol_adjusted * SCALAR_VALUE
    forecast = cap_forecast(scaled)

    return ValueForecast(
        symbol="",
        forecast=forecast,
        relative_price=relative_price,
        raw_value=raw_value,
        lookback_days=effective_lookback,
    )


def compute_value_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    lookback: int = DEFAULT_VALUE_LOOKBACK,
) -> Dict[str, float]:
    """Compute value forecast for every symbol in the cache.

    Returns
    -------
    dict[str, float]
        {symbol: forecast} where forecast ∈ [−20, +20].
    """
    results: Dict[str, float] = {}

    for sym, df in ohlcv_cache.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue

        close = df["Close"].dropna()
        fc = compute_value_forecast(close, lookback)
        if fc is not None:
            fc.symbol = sym
            results[sym] = fc.forecast

    logger.info(
        "Value forecast computed for %d / %d symbols",
        len(results), len(ohlcv_cache),
    )
    return results
