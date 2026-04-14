"""
Fundamental Momentum Forecast — EPS/revenue revision-based signals.

Uses the *change in fundamental metrics* rather than levels:
  - Rising EPS estimates → positive forecast (analysts upgrading)
  - Falling EPS estimates → negative forecast (analysts downgrading)
  - Revenue surprise momentum → persistence of beats/misses

In backtest, uses price-to-52w-high as a proxy for fundamental momentum
(stocks nearer 52w high tend to have improving fundamentals).
In live mode, will use Screener.in or TipRanks API for actual EPS revisions.

Academic basis:
  Chan, Jegadeesh & Lakonishok (1996): Momentum of analyst estimate revisions
  predicts returns with Sharpe ~0.6 annualized.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOOKBACK_52W = 252         # 1 year = 252 trading days
_LOOKBACK_MOMENTUM = 63     # 3-month fundamental momentum
_FORECAST_SCALAR = 15.0     # calibrated to avg|forecast| ≈ 10


def compute_fundamental_momentum_batch(
    ohlcv_slice: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute fundamental momentum proxy for all symbols.

    Backtest proxy: distance from 52-week high + 3-month price momentum.
    Stocks near 52w high with positive momentum → improving fundamentals.
    """
    result: Dict[str, float] = {}

    for sym, df in ohlcv_slice.items():
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        close = close.dropna()
        if len(close) < _LOOKBACK_52W:
            continue

        current = float(close.iloc[-1])
        high_52w = float(close.iloc[-_LOOKBACK_52W:].max())
        if high_52w <= 0:
            continue

        # Distance from 52w high: 1.0 = at high, 0.5 = 50% down
        dist_ratio = current / high_52w

        # 3-month momentum
        if len(close) >= _LOOKBACK_MOMENTUM:
            mom_3m = float(close.iloc[-1] / close.iloc[-_LOOKBACK_MOMENTUM] - 1)
        else:
            mom_3m = 0.0

        # Combine: 60% proximity to 52w high + 40% momentum
        # Centered at 0: dist_ratio 0.8 → near high → positive; 0.5 → far → negative
        raw = ((dist_ratio - 0.75) * 3.0) * 0.6 + mom_3m * 0.4
        fc = raw * _FORECAST_SCALAR
        fc = max(-20.0, min(20.0, fc))

        if abs(fc) > 0.5:
            result[sym] = fc

    return result
