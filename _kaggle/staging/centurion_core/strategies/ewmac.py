"""
EWMAC Trading Rule — Exponentially Weighted Moving Average Crossover.

Implements Carver's primary momentum rule (Appendix B).  For each
look-back pair (fast, slow), the raw signal is:

    raw_crossover = EWMA(close, fast) - EWMA(close, slow)

This is then volatility-adjusted and scaled to produce a forecast
with an expected average absolute value of ≈ 10, capped at ±20.

Recommended variations for swing / positional trading:
  - EWMAC(16, 64)   — captures 3–10 day swings        (swing core)
  - EWMAC(32, 128)  — captures 2–4 week trends         (positional core)
  - EWMAC(64, 256)  — captures 1–3 month macro trends  (positional slow)

Faster variations (2/8, 4/16, 8/32) are excluded because NSE
equity CNC trades have too-high costs for sub-weekly turnover.

AFTS Enhancements:
  - S12 (Adjusted Trend): For slow filters (64,256), cap raw forecast at
    ±15 then multiply by 1.25.  Large slow forecasts are usually driven by
    low-vol environments, not genuinely strong trends, so this cap-and-scale
    improves the forecast distribution and raises SR by ~0.02.
  - S17 (Normalised Trend): Maps the EWMAC forecast to its percentile rank
    within its own history.  Produces a uniform forecast distribution — more
    robust across regimes and avoids extreme forecasts in trending markets.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.forecast_scalar import ewmac_to_forecast, cap_forecast
from services.instrument_volatility import daily_price_volatility

logger = logging.getLogger(__name__)

# Default variations suitable for swing/positional equity trading
DEFAULT_VARIATIONS: List[Tuple[int, int]] = [
    (8, 32),    # fast swing: ~1 week trends (catches regime changes early)
    (16, 64),   # swing: ~1–2 week trends
    (32, 128),  # positional: ~1–2 month trends
    (64, 256),  # slow positional: ~3–6 month trends
]

# --- S12: Adjusted Trend parameters ---
# Variations considered "slow" — adjusted trend capping applied
ADJUSTED_SLOW_VARIATIONS = {(64, 256)}
ADJUSTED_FORECAST_CAP = 15.0   # cap raw forecast at ±15 before scaling
ADJUSTED_SCALE_FACTOR = 1.25   # then multiply by 1.25

# --- S17: Normalised Trend parameters ---
NORMALISED_LOOKBACK = 2520  # 10 years of history for percentile ranking
NORMALISED_MIN_HISTORY = 504  # minimum 2 years for meaningful ranking


@dataclass
class EWMACForecast:
    """Output of a single EWMAC variation for one instrument."""
    symbol: str
    fast: int
    slow: int
    forecast: float          # -20 to +20
    raw_crossover: float     # unscaled EWMA difference
    fast_ewma: float         # current fast EWMA value
    slow_ewma: float         # current slow EWMA value


def compute_ewmac(
    close: pd.Series,
    fast: int,
    slow: int,
    adjusted_trend: bool = True,
    normalised: bool = False,
    forecast_history: Optional[deque] = None,
) -> Optional[EWMACForecast]:
    """Compute a single EWMAC forecast for one close series.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices, chronological.
    fast, slow : int
        Look-back periods for the fast and slow EWMAs.
    adjusted_trend : bool
        If True and (fast,slow) is in ADJUSTED_SLOW_VARIATIONS, apply
        S12 capping: cap at ±15 then ×1.25.
    normalised : bool
        If True, return a normalised (percentile-ranked) forecast (S17)
        instead of the raw scaled forecast.
    forecast_history : deque | None
        If normalised=True, supply a deque of recent forecast values to
        compute the percentile rank.  The caller should maintain this
        across calls (one per variation-symbol pair).

    Returns
    -------
    EWMACForecast or None if insufficient data.
    """
    min_required = slow + 10  # need enough data for slow EWMA to stabilise
    if close is None or len(close) < min_required:
        return None

    fast_ewma = close.ewm(span=fast, min_periods=fast).mean()
    slow_ewma = close.ewm(span=slow, min_periods=slow).mean()

    raw_crossover = float(fast_ewma.iloc[-1] - slow_ewma.iloc[-1])

    # Daily price volatility (in price points, not percentage)
    price = float(close.iloc[-1])
    daily_vol_pct = daily_price_volatility(close)
    daily_vol_price = price * daily_vol_pct if daily_vol_pct > 0 else 1.0

    forecast = ewmac_to_forecast(raw_crossover, daily_vol_price, fast, slow)

    # --- S12: Adjusted Trend for slow variations ---
    if adjusted_trend and (fast, slow) in ADJUSTED_SLOW_VARIATIONS:
        capped = max(-ADJUSTED_FORECAST_CAP, min(ADJUSTED_FORECAST_CAP, forecast))
        forecast = capped * ADJUSTED_SCALE_FACTOR
        forecast = cap_forecast(forecast)  # final ±20 cap

    # --- S17: Normalised Trend (percentile ranking) ---
    if normalised and forecast_history is not None:
        forecast_history.append(forecast)
        if len(forecast_history) >= NORMALISED_MIN_HISTORY:
            arr = np.array(forecast_history)
            # Percentile rank: what fraction of history is <= current?
            rank = float(np.searchsorted(np.sort(arr), forecast)) / len(arr)
            # Map [0,1] → [−20, +20] maintaining sign
            forecast = (rank - 0.5) * 40.0  # range: -20 to +20
            forecast = cap_forecast(forecast)

    return EWMACForecast(
        symbol="",  # caller fills this
        fast=fast,
        slow=slow,
        forecast=forecast,
        raw_crossover=raw_crossover,
        fast_ewma=float(fast_ewma.iloc[-1]),
        slow_ewma=float(slow_ewma.iloc[-1]),
    )


def compute_ewmac_all_variations(
    close: pd.Series,
    variations: Optional[List[Tuple[int, int]]] = None,
) -> List[EWMACForecast]:
    """Compute EWMAC forecasts for all variations on one instrument.

    Returns
    -------
    list[EWMACForecast]
        One per variation that has sufficient data.
    """
    variations = variations or DEFAULT_VARIATIONS
    results = []
    for fast, slow in variations:
        fc = compute_ewmac(close, fast, slow)
        if fc is not None:
            results.append(fc)
    return results


def compute_ewmac_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    variations: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, List[EWMACForecast]]:
    """Compute EWMAC for every symbol in the cache.

    Parameters
    ----------
    ohlcv_cache : dict[str, DataFrame]
        ``{symbol: df}`` with a ``"Close"`` column.

    Returns
    -------
    dict[str, list[EWMACForecast]]
        ``{symbol: [forecast_per_variation]}``
    """
    variations = variations or DEFAULT_VARIATIONS
    results: Dict[str, List[EWMACForecast]] = {}

    for sym, df in ohlcv_cache.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        forecasts = compute_ewmac_all_variations(close, variations)
        for fc in forecasts:
            fc.symbol = sym
        if forecasts:
            results[sym] = forecasts

    logger.info("EWMAC computed for %d / %d symbols", len(results), len(ohlcv_cache))
    return results
