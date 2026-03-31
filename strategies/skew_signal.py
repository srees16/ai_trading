"""
Skew Signal Forecast — Realized Return Skew as a Trading Signal.

Implements Carver AFTS Strategy 24.  Instruments whose returns exhibit
negative skew carry a *risk premium* — investors dislike negative skew
(sudden large losses) and are willing to accept lower expected returns
to avoid it.  We can earn this premium by going *long* instruments
with negative skew and *short* (or underweight) those with positive skew.

Formula:
    rolling_skew  = skew(daily_returns, window=252)
    raw_signal    = −rolling_skew  (negative skew → positive forecast)
    vol_adjusted  = raw_signal / abs_avg_skew_history
    forecast      = scale_and_cap(vol_adjusted × scalar, ±20)

Properties:
    - Very low turnover (rebalances weekly at most)
    - Standalone SR modest (~0.10–0.15)
    - Correlation with trend ≈ 0.10–0.20 (low → good diversifier)
    - Correlation with value ≈ 0.30 (both are "risk premium" plays)
    - Captures the structural equity skew premium
    - NOTE: In IND (long-only), used to SIZE positions — stocks with
      more negative skew get slightly larger positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from services.forecast_scalar import cap_forecast

logger = logging.getLogger(__name__)

# Rolling window for skew calculation (≈1 year)
DEFAULT_SKEW_WINDOW = 252

# Minimum data for reliable skew estimate
MIN_DATA_DAYS = 126  # 6 months

# Pre-calibrated scalar: typical |rolling_skew| ≈ 0.3–0.8
# vol_adjusted std ≈ 1.0, median_abs ≈ 0.80 → scalar = 10/0.80 ≈ 12.5
SCALAR_SKEW = 12.5


@dataclass
class SkewForecast:
    """Output of the skew signal for one instrument."""
    symbol: str
    forecast: float           # −20 to +20
    rolling_skew: float       # realized skew of returns
    raw_signal: float         # −rolling_skew (sign flipped)
    window: int               # lookback window used


def compute_skew_forecast(
    close: pd.Series,
    window: int = DEFAULT_SKEW_WINDOW,
) -> Optional[SkewForecast]:
    """Compute skew-based forecast for one instrument.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices, chronological.
    window : int
        Rolling window for skew calculation in trading days.

    Returns
    -------
    SkewForecast or None if insufficient data.
    """
    if close is None or len(close) < window:
        return None

    returns = close.pct_change().dropna()
    if len(returns) < MIN_DATA_DAYS:
        return None

    # Rolling skew of daily returns
    rolling_skew_series = returns.rolling(window=window, min_periods=MIN_DATA_DAYS).skew()
    current_skew = float(rolling_skew_series.iloc[-1])

    if np.isnan(current_skew):
        return None

    # Negative skew → long (risk premium capture)
    raw_signal = -current_skew

    # Normalize by the typical magnitude of skew across history
    # This makes the signal self-calibrating across instruments
    abs_skew_history = rolling_skew_series.abs().dropna()
    if len(abs_skew_history) < 30:
        normalizer = 1.0
    else:
        normalizer = float(abs_skew_history.expanding(min_periods=30).median().iloc[-1])
        normalizer = max(normalizer, 0.1)

    vol_adjusted = raw_signal / normalizer
    scaled = vol_adjusted * SCALAR_SKEW
    forecast = cap_forecast(scaled)

    return SkewForecast(
        symbol="",
        forecast=forecast,
        rolling_skew=current_skew,
        raw_signal=raw_signal,
        window=window,
    )


def compute_skew_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    window: int = DEFAULT_SKEW_WINDOW,
) -> Dict[str, float]:
    """Compute skew forecast for every symbol in the cache.

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
        fc = compute_skew_forecast(close, window)
        if fc is not None:
            fc.symbol = sym
            results[sym] = fc.forecast

    logger.info(
        "Skew forecast computed for %d / %d symbols",
        len(results), len(ohlcv_cache),
    )
    return results
