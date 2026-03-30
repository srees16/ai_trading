"""
F&O Open Interest Signal — Gap A6.

Uses NSE F&O open interest data to gauge directional conviction.
OI changes combined with price movements reveal institutional positioning.

Signal logic:
  - Rising OI + Rising price → Long buildup (bullish)
  - Rising OI + Falling price → Short buildup (bearish)
  - Falling OI + Rising price → Short covering (mildly bullish)
  - Falling OI + Falling price → Long unwinding (bearish)

Also provides IV rank for options overlay strategy selection.

Integration:
  - Generates per-stock conviction modifier for F&O stocks only
  - Weight: 5% of combined forecast
  - Only active for ~180 F&O-eligible NSE stocks
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

# NSE F&O lot sizes for major stocks (subset — extend as needed)
FNO_LOT_SIZES: Dict[str, int] = {
    "RELIANCE": 250, "TCS": 150, "INFY": 300, "HDFCBANK": 550,
    "ICICIBANK": 700, "SBIN": 750, "KOTAKBANK": 400, "AXISBANK": 600,
    "BAJFINANCE": 125, "BHARTIARTL": 950, "ITC": 1600, "LT": 150,
    "HCLTECH": 350, "WIPRO": 1500, "MARUTI": 100, "TATAMOTORS": 575,
    "TATASTEEL": 550, "SUNPHARMA": 350, "NTPC": 2800, "POWERGRID": 2700,
    "ONGC": 1925, "HINDALCO": 1075, "INDUSINDBK": 400, "M&M": 350,
    "JSWSTEEL": 675, "ADANIENT": 250, "ADANIPORTS": 625, "TITAN": 175,
    "DIVISLAB": 100, "ULTRACEMCO": 50, "DRREDDY": 125, "TECHM": 350,
    "NESTLEIND": 25, "COALINDIA": 700, "APOLLOHOSP": 125, "HDFCLIFE": 500,
    "CIPLA": 325, "GRASIM": 250, "SBILIFE": 375, "EICHERMOT": 75,
    "BAJAJ-AUTO": 125, "BPCL": 1800, "TATACONSUM": 450, "HEROMOTOCO": 75,
    "ASIANPAINT": 200, "BRITANNIA": 100, "VEDL": 1550, "HINDPETRO": 1350,
}


@dataclass
class OISignal:
    """Open Interest based signal for one stock."""
    ticker: str
    oi_change_pct: float       # % change in OI
    price_change_pct: float    # % change in price
    buildup_type: str          # LONG_BUILDUP, SHORT_BUILDUP, SHORT_COVERING, LONG_UNWINDING
    forecast: float            # Carver-scale forecast
    iv_rank: float = 50.0      # IV rank (0-100), 50 = median
    is_fno: bool = True


def classify_oi_buildup(
    oi_change_pct: float,
    price_change_pct: float,
) -> tuple:
    """Classify OI buildup type and assign conviction.

    Returns (buildup_type, conviction_multiplier)
    """
    if oi_change_pct > 2.0 and price_change_pct > 0.5:
        return "LONG_BUILDUP", 1.0
    elif oi_change_pct > 2.0 and price_change_pct < -0.5:
        return "SHORT_BUILDUP", -1.0
    elif oi_change_pct < -2.0 and price_change_pct > 0.5:
        return "SHORT_COVERING", 0.5
    elif oi_change_pct < -2.0 and price_change_pct < -0.5:
        return "LONG_UNWINDING", -0.5
    else:
        return "NEUTRAL", 0.0


def compute_oi_forecast(
    oi_change_pct: float,
    price_change_pct: float,
    volume_ratio: float = 1.0,
) -> float:
    """Compute OI-based forecast.

    Parameters
    ----------
    oi_change_pct : float
        % change in open interest (e.g., 5.0 = 5% increase).
    price_change_pct : float
        % change in stock price.
    volume_ratio : float
        Today's volume / 20-day avg volume (amplifier).

    Returns
    -------
    float
        Carver-scale forecast (0 to 20 for long-only).
    """
    buildup_type, conviction = classify_oi_buildup(oi_change_pct, price_change_pct)

    if conviction <= 0:
        return 0.0  # Long-only: no signal for bearish setups

    # Scale by OI magnitude
    oi_strength = min(1.0, abs(oi_change_pct) / 10.0)  # 10% OI change = max strength

    # Volume confirmation
    vol_boost = min(1.5, max(0.5, volume_ratio))

    raw = conviction * oi_strength * vol_boost * FORECAST_CAP
    return round(max(0.0, min(FORECAST_CAP, raw)), 2)


def compute_iv_rank(
    current_iv: float,
    iv_history: list[float],
) -> float:
    """Compute IV rank (percentile of current IV vs 1-year history).

    IV Rank = (current_IV - min_IV) / (max_IV - min_IV) × 100

    Parameters
    ----------
    current_iv : float
        Current implied volatility.
    iv_history : list[float]
        Last 252 days of IV values.

    Returns
    -------
    float
        IV rank (0-100).
    """
    if not iv_history or len(iv_history) < 30:
        return 50.0

    min_iv = min(iv_history)
    max_iv = max(iv_history)

    if max_iv <= min_iv:
        return 50.0

    rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
    return round(max(0.0, min(100.0, rank)), 1)


def compute_oi_signals_batch(
    oi_data: Dict[str, Dict],
) -> Dict[str, float]:
    """Compute OI-based forecasts for all F&O stocks.

    Parameters
    ----------
    oi_data : dict
        {symbol: {"oi_change_pct": float, "price_change_pct": float,
                   "volume_ratio": float}}

    Returns
    -------
    dict[str, float]
        {symbol: forecast} for Carver combiner.
    """
    forecasts: Dict[str, float] = {}

    for sym, data in oi_data.items():
        if sym not in FNO_LOT_SIZES:
            continue  # Only F&O stocks

        oi_pct = data.get("oi_change_pct", 0.0)
        price_pct = data.get("price_change_pct", 0.0)
        vol_ratio = data.get("volume_ratio", 1.0)

        forecast = compute_oi_forecast(oi_pct, price_pct, vol_ratio)
        if forecast > 0:
            forecasts[sym] = forecast

    if forecasts:
        logger.info(
            "OI signals: %d/%d F&O stocks with signals (avg forecast=%.1f)",
            len(forecasts), len(oi_data),
            np.mean(list(forecasts.values())),
        )

    return forecasts


def is_fno_eligible(ticker: str) -> bool:
    """Check if a ticker is F&O eligible on NSE."""
    return ticker in FNO_LOT_SIZES


def get_lot_size(ticker: str) -> int:
    """Get F&O lot size for a ticker."""
    return FNO_LOT_SIZES.get(ticker, 0)
