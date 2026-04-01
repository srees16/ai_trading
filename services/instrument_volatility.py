"""
Instrument Volatility — Carver-framework volatility estimation.

Computes daily price volatility using a 35-day exponentially weighted
moving average (EWMA) of absolute daily percentage returns, as
recommended in *Systematic Trading* (Robert Carver, Appendix D).

Key outputs:
  - ``price_volatility_pct``    — daily % standard deviation of returns
  - ``annual_volatility_pct``   — annualised (×16) percentage volatility
  - ``instrument_value_vol``    — daily cash volatility per 1 share
                                  (= price × price_vol_pct)

These feed into the Carver position-sizing formula:
  ``volatility_scalar = daily_cash_vol_target / instrument_value_vol``
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Carver recommends a 35-day EWMA for daily vol (Appendix D).
# Gap D5 FIX: Use halflife=20 for proper 20-day half-life decay.
# span=35 gives effective lookback of ~10 days; halflife=20 gives ~20 days.
DEFAULT_VOL_HALFLIFE = 20
ANNUALISATION_FACTOR = 15.874507866387544  # sqrt(252 trading days)


def daily_price_volatility(
    close: pd.Series,
    *,
    lookback: int = DEFAULT_VOL_HALFLIFE,
) -> float:
    """Return the latest daily percentage price volatility (as a decimal).

    Uses an exponentially weighted moving standard deviation of daily
    percentage returns with a half-life of ``lookback`` days.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices (DatetimeIndex, chronological).
    lookback : int
        EWMA halflife (default 20 business days ≈ 4 weeks).

    Returns
    -------
    float
        Latest daily percentage volatility (e.g. 0.018 = 1.8 % daily).
        Returns 0.0 if insufficient data.
    """
    if close is None or len(close) < max(5, lookback // 2):
        return 0.0

    pct_returns = close.pct_change().dropna()
    if pct_returns.empty:
        return 0.0

    ewm_std = pct_returns.ewm(halflife=lookback, min_periods=max(5, lookback // 2)).std()
    latest = ewm_std.iloc[-1]
    return float(latest) if np.isfinite(latest) else 0.0


def annual_price_volatility(
    close: pd.Series,
    *,
    lookback: int = DEFAULT_VOL_HALFLIFE,
) -> float:
    """Annualised percentage volatility = daily_vol × 16."""
    return daily_price_volatility(close, lookback=lookback) * ANNUALISATION_FACTOR


def instrument_value_volatility(
    close: pd.Series,
    *,
    lookback: int = DEFAULT_VOL_HALFLIFE,
) -> float:
    """Daily cash volatility per 1 share = price × daily_price_vol.

    This is the denominator in Carver's volatility scalar formula:
        ``vol_scalar = daily_cash_vol_target / instrument_value_vol``
    """
    if close is None or close.empty:
        return 0.0
    price = float(close.iloc[-1])
    daily_vol = daily_price_volatility(close, lookback=lookback)
    return price * daily_vol


def compute_volatilities_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    *,
    lookback: int = DEFAULT_VOL_HALFLIFE,
) -> Dict[str, dict]:
    """Compute volatility metrics for a batch of instruments.

    Parameters
    ----------
    ohlcv_cache : dict[str, DataFrame]
        ``{symbol: DataFrame}`` with at least a ``"Close"`` column.

    Returns
    -------
    dict[str, dict]
        ``{symbol: {"daily_vol": float, "annual_vol": float,
                     "instr_value_vol": float, "price": float}}``
    """
    results: Dict[str, dict] = {}
    for sym, df in ohlcv_cache.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if close.empty:
            continue
        dv = daily_price_volatility(close, lookback=lookback)
        price = float(close.iloc[-1])
        results[sym] = {
            "daily_vol": dv,
            "annual_vol": dv * ANNUALISATION_FACTOR,
            "instr_value_vol": price * dv,
            "price": price,
        }
    return results
