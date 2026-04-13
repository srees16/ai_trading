"""
Dispersion Trading Forecast — NIFTY vol vs constituent vol spread.

When implied (or realized) vol of the index is LOW relative to the
average vol of its constituents, it indicates:
  - High pairwise correlation → systemic move imminent
  - Underpriced portfolio hedging → buy options / reduce equity

When dispersion is HIGH (index vol << component vol):
  - Low correlation → stock-picking environment
  - Positive for long equity with diversification

Formula:
  dispersion = avg_constituent_vol / index_vol
  When dispersion < 1.0 → correlated sell-off likely → negative forecast
  When dispersion > 2.0 → stock-picker market → positive forecast
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VOL_LOOKBACK = 20         # 1-month realized vol
_MIN_CONSTITUENTS = 10     # need enough stocks for dispersion calc
_FORECAST_SCALAR = 8.0     # calibrated to avg|forecast| ≈ 10


def compute_dispersion_forecast_batch(
    ohlcv_slice: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute vol dispersion forecast for all symbols.

    Uses cross-sectional realized vol of constituents vs portfolio vol
    as a proxy for index dispersion trading signal.
    """
    result: Dict[str, float] = {}

    # Collect per-stock realized vol (20d annualized)
    stock_vols = {}
    stock_rets = {}
    for sym, df in ohlcv_slice.items():
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        close = close.dropna()
        if len(close) < _VOL_LOOKBACK + 5:
            continue
        rets = close.pct_change().dropna().iloc[-_VOL_LOOKBACK:]
        if len(rets) < _VOL_LOOKBACK - 2:
            continue
        vol = float(rets.std()) * 16.0  # annualized
        if vol > 0 and np.isfinite(vol):
            stock_vols[sym] = vol
            stock_rets[sym] = rets

    if len(stock_vols) < _MIN_CONSTITUENTS:
        return result

    # Average constituent vol
    avg_constituent_vol = float(np.mean(list(stock_vols.values())))

    # Portfolio vol (equal-weight proxy)
    rets_df = pd.DataFrame(stock_rets)
    if len(rets_df) < _VOL_LOOKBACK - 2:
        return result
    # Equal-weight portfolio return
    portfolio_rets = rets_df.mean(axis=1)
    portfolio_vol = float(portfolio_rets.std()) * 16.0

    if portfolio_vol <= 0:
        return result

    # Dispersion ratio
    dispersion = avg_constituent_vol / portfolio_vol

    # Map dispersion → forecast
    # dispersion ≈ 1.0 → correlated (bad) → negative
    # dispersion ≈ 2.0 → dispersed (good for stock-picking) → positive
    # Center at 1.5 (typical value)
    raw = (dispersion - 1.5) * _FORECAST_SCALAR
    fc = max(-20.0, min(20.0, raw))

    # Market-wide signal: apply to all symbols
    for sym in stock_vols:
        result[sym] = fc

    return result
