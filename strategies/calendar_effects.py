"""
Calendar Effects Forecast — Seasonality anomalies for Indian equity markets.

Exploits well-documented calendar anomalies in NSE:
  - Month-end liquidity surge (last 3 trading days of month)
  - F&O expiry day (last Thursday of month) — gamma unwinding
  - Budget-day (Feb 1) — elevated vol + mean-reverting post-announcement
  - Diwali rally (Oct-Nov seasonality)
  - January effect (FPI rebalancing)
  - Tax-loss selling pressure (Mar 20-31)

Each anomaly produces a forecast in [-20, +20], capped by Carver convention.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Calendar event amplitudes (calibrated to avg|forecast| ≈ 10)
_MONTH_END_BOOST = 6.0       # last 3 trading days: mild positive bias
_EXPIRY_DAMpen = -4.0        # expiry day: gamma unwind → negative bias
_BUDGET_PRE = 3.0            # day before budget: pre-announcement drift
_BUDGET_POST_REVERSAL = -5.0 # 2 days after budget: mean reversion
_DIWALI_BOOST = 8.0          # Oct 15 - Nov 15: seasonal rally
_JAN_EFFECT = 5.0            # first 2 weeks of Jan: FPI rebalance inflow
_TAX_LOSS_DRAG = -7.0        # Mar 20-31: tax-loss selling pressure


def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of the given month (F&O expiry)."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    # Walk back to Thursday (weekday=3)
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)


def compute_calendar_forecast(as_of_date: date) -> float:
    """Compute calendar effect forecast for a given date.

    Returns a single market-wide forecast value in [-20, +20].
    """
    fc = 0.0
    month = as_of_date.month
    day = as_of_date.day
    weekday = as_of_date.weekday()  # 0=Mon, 6=Sun

    # Month-end: last 3 trading days (approximate: day >= 27)
    if day >= 27:
        fc += _MONTH_END_BOOST

    # F&O expiry (last Thursday)
    expiry = _last_thursday_of_month(as_of_date.year, month)
    days_to_expiry = (expiry - as_of_date).days
    if days_to_expiry == 0:
        fc += _EXPIRY_DAMpen  # expiry day itself
    elif days_to_expiry == 1:
        fc += _EXPIRY_DAMpen * 0.5  # day before expiry

    # Budget day (Feb 1)
    if month == 2:
        if day == 1:
            fc += _BUDGET_PRE  # pre-announcement optimism
        elif day in (2, 3):
            fc += _BUDGET_POST_REVERSAL  # post-budget reversal

    # Diwali rally (Oct 15 - Nov 15)
    if (month == 10 and day >= 15) or (month == 11 and day <= 15):
        fc += _DIWALI_BOOST

    # January effect (first 2 weeks)
    if month == 1 and day <= 14:
        fc += _JAN_EFFECT

    # Tax-loss selling (Mar 20-31)
    if month == 3 and day >= 20:
        fc += _TAX_LOSS_DRAG

    return max(-20.0, min(20.0, fc))


def compute_calendar_forecast_batch(
    ohlcv_slice: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute calendar effect forecast for all symbols.

    Calendar effects are market-wide, so the same forecast applies to all.
    """
    result: Dict[str, float] = {}
    if not ohlcv_slice:
        return result

    # Infer current date from the last row of any symbol's data
    any_df = next(iter(ohlcv_slice.values()))
    try:
        last_idx = any_df.index[-1]
        if hasattr(last_idx, 'date'):
            as_of = last_idx.date() if callable(last_idx.date) else last_idx
        else:
            as_of = date.today()
    except Exception:
        as_of = date.today()

    fc = compute_calendar_forecast(as_of)
    if abs(fc) < 0.5:
        return result  # no significant calendar effect today

    for sym in ohlcv_slice:
        result[sym] = fc

    return result
