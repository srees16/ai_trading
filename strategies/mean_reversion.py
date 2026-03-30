"""
Mean-Reversion Forecast — Gap A2.

Counter-trend signal based on RSI extremes + Bollinger Band reversals.
Designed for swing trading (5–15 day holds) on NSE equities.

Works best in RANGE_BOUND regimes where trend-following fails.
Generates Carver-scale forecasts (avg |f| ≈ 10, capped ±20).

Research basis:
  - Bollinger (2001): Bollinger Bands capture mean-reversion edges
  - Jegadeesh (1990): Short-term reversal (1-month) premium 1–2% monthly
  - Indian evidence: Range-bound NIFTY periods (40% of calendar days)

Integration:
  - Added as forecast source #8 in Carver combiner
  - Weight: 8–12% (higher in RANGE_BOUND regime)
  - Long-only: only BUY signals generated (negative = no signal)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

FORECAST_CAP = 20.0


@dataclass
class MeanReversionSignal:
    """Mean-reversion signal for a single stock."""
    ticker: str
    rsi: float
    bb_percentile: float       # 0 = at lower band, 1 = at upper band
    forecast: float
    signal_type: str           # "OVERSOLD_BOUNCE", "OVERBOUGHT_FADE", "NONE"


def compute_rsi(close: "pd.Series", period: int = 14) -> "pd.Series":
    """Compute RSI using Wilder's smoothing."""
    import pandas as pd
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(
    close: "pd.Series", period: int = 20, n_std: float = 2.0
) -> tuple:
    """Return (lower_band, middle_band, upper_band)."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    lower = middle - n_std * std
    upper = middle + n_std * std
    return lower, middle, upper


def compute_mean_reversion_forecast(
    close: "pd.Series",
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    oversold_rsi: float = 25.0,
    overbought_rsi: float = 75.0,
) -> MeanReversionSignal:
    """Compute mean-reversion forecast for one stock.

    BUY signal: RSI < oversold AND price below lower Bollinger Band
    Scaled to Carver convention (avg |f| ≈ 10, cap ±20).

    For long-only NSE: only generates positive forecasts (oversold bounce).
    Overbought signals return 0 (no signal) instead of negative.

    Parameters
    ----------
    close : pd.Series
        Close prices with at least ``bb_period + 10`` bars.
    rsi_period, bb_period, bb_std : int/float
        RSI and Bollinger Band parameters.
    oversold_rsi, overbought_rsi : float
        RSI thresholds for signal generation.

    Returns
    -------
    MeanReversionSignal
    """
    ticker = getattr(close, 'name', 'UNKNOWN')

    if close is None or len(close) < bb_period + 10:
        return MeanReversionSignal(
            ticker=str(ticker), rsi=50.0, bb_percentile=0.5,
            forecast=0.0, signal_type="NONE",
        )

    rsi_series = compute_rsi(close, rsi_period)
    lower, middle, upper = compute_bollinger_bands(close, bb_period, bb_std)

    current_rsi = float(rsi_series.iloc[-1])
    current_price = float(close.iloc[-1])
    current_lower = float(lower.iloc[-1])
    current_upper = float(upper.iloc[-1])
    current_middle = float(middle.iloc[-1])

    # Bollinger percentile: 0 = at lower, 0.5 = at middle, 1 = at upper
    bb_range = current_upper - current_lower
    if bb_range > 0:
        bb_pctile = (current_price - current_lower) / bb_range
    else:
        bb_pctile = 0.5

    bb_pctile = max(0.0, min(1.0, bb_pctile))

    forecast = 0.0
    signal_type = "NONE"

    # OVERSOLD BOUNCE: RSI < threshold AND price near/below lower BB
    if current_rsi < oversold_rsi and bb_pctile < 0.15:
        # Stronger signal for more extreme oversold
        rsi_strength = (oversold_rsi - current_rsi) / oversold_rsi  # 0 to ~1
        bb_strength = max(0, 0.15 - bb_pctile) / 0.15               # 0 to 1
        raw = (rsi_strength * 0.6 + bb_strength * 0.4) * FORECAST_CAP
        forecast = min(FORECAST_CAP, max(0.0, raw))
        signal_type = "OVERSOLD_BOUNCE"

    elif current_rsi < 35 and bb_pctile < 0.25:
        # Moderate oversold — weaker signal
        rsi_strength = (35 - current_rsi) / 35
        bb_strength = max(0, 0.25 - bb_pctile) / 0.25
        raw = (rsi_strength * 0.6 + bb_strength * 0.4) * FORECAST_CAP * 0.5
        forecast = min(FORECAST_CAP, max(0.0, raw))
        signal_type = "OVERSOLD_BOUNCE"

    # For long-only: overbought = reduce forecast to 0 (no shorting)
    elif current_rsi > overbought_rsi and bb_pctile > 0.85:
        forecast = 0.0
        signal_type = "OVERBOUGHT_FADE"

    return MeanReversionSignal(
        ticker=str(ticker),
        rsi=round(current_rsi, 2),
        bb_percentile=round(bb_pctile, 4),
        forecast=round(forecast, 2),
        signal_type=signal_type,
    )


def compute_mean_reversion_batch(
    ohlcv_cache: Dict[str, "pd.DataFrame"],
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> Dict[str, float]:
    """Compute mean-reversion forecasts for all symbols.

    Returns {symbol: forecast} for integration with Carver combiner.
    """
    forecasts: Dict[str, float] = {}

    for sym, df in ohlcv_cache.items():
        if df is None or df.empty:
            continue
        try:
            close = df["Close"]
            if hasattr(close, "squeeze"):
                close = close.squeeze()
            close = close.dropna()
            close.name = sym

            signal = compute_mean_reversion_forecast(
                close, rsi_period=rsi_period, bb_period=bb_period, bb_std=bb_std,
            )
            if signal.forecast != 0.0:
                forecasts[sym] = signal.forecast
        except Exception as exc:
            logger.debug("Mean-reversion calc failed for %s: %s", sym, exc)

    if forecasts:
        logger.info(
            "Mean-reversion: %d/%d signals generated (avg forecast=%.1f)",
            len(forecasts), len(ohlcv_cache),
            np.mean(list(forecasts.values())),
        )

    return forecasts
