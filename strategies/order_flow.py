"""
Order Flow Forecast Source — T1-6 / Market Microstructure.

Promotes volume microstructure signals (OBV, CVD proxy, MFI) from
backtest-only registry into a live Carver forecast source.

Generates a normalized forecast in [-20, +20] range:
  - OBV rising + price rising = strong buy confirmation
  - OBV divergence (OBV up, price down) = potential reversal
  - MFI > 80 = overbought, MFI < 20 = oversold
  - Delivery % > 60% = institutional conviction

Weight in combiner: 2% (24th source).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORECAST_CAP = 20.0


def compute_order_flow_forecast(
    ohlcv: pd.DataFrame,
    delivery_pct: Optional[float] = None,
    lookback: int = 20,
) -> float:
    """Compute order flow forecast from OHLCV + delivery data.

    Parameters
    ----------
    ohlcv : DataFrame
        Must have Open, High, Low, Close, Volume columns.
    delivery_pct : float | None
        NSE delivery percentage (0-100). >60% = institutional conviction.
    lookback : int
        Signal lookback period.

    Returns
    -------
    float
        Forecast in [-20, +20].
    """
    if ohlcv is None or len(ohlcv) < lookback + 5:
        return 0.0

    try:
        close = ohlcv["Close"].values
        high = ohlcv["High"].values
        low = ohlcv["Low"].values
        volume = ohlcv["Volume"].values.astype(float)

        n = len(close)
        if n < lookback + 5:
            return 0.0

        # 1. On-Balance Volume (OBV) trend
        obv = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # OBV slope (z-score of recent vs longer lookback)
        obv_recent = obv[-lookback:]
        obv_slope = (obv_recent[-1] - obv_recent[0]) / max(1, lookback)
        obv_std = np.std(np.diff(obv[-lookback * 2:])) if n >= lookback * 2 else max(abs(obv_slope), 1)
        obv_z = obv_slope / max(obv_std, 1e-8)
        obv_signal = np.clip(obv_z * 5, -10, 10)  # Scale to ±10

        # 2. Money Flow Index (MFI) — RSI on volume-weighted price
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        pos_flow = np.zeros(n)
        neg_flow = np.zeros(n)
        for i in range(1, n):
            if typical_price[i] > typical_price[i - 1]:
                pos_flow[i] = money_flow[i]
            else:
                neg_flow[i] = money_flow[i]

        period = min(14, lookback)
        pos_sum = np.sum(pos_flow[-period:])
        neg_sum = np.sum(neg_flow[-period:])
        if neg_sum > 0:
            mfi = 100 - (100 / (1 + pos_sum / neg_sum))
        else:
            mfi = 100.0

        # MFI signal: overbought >80 → sell, oversold <20 → buy
        if mfi > 80:
            mfi_signal = -(mfi - 80) / 20 * 5  # -5 at MFI=100
        elif mfi < 20:
            mfi_signal = (20 - mfi) / 20 * 5   # +5 at MFI=0
        else:
            mfi_signal = (mfi - 50) / 30 * 3    # Mild directional

        # 3. Cumulative Volume Delta proxy (CVD)
        # Approximate: up bars = buy volume, down bars = sell volume
        cvd = np.zeros(n)
        for i in range(n):
            bar_range = high[i] - low[i]
            if bar_range > 0:
                buy_ratio = (close[i] - low[i]) / bar_range
            else:
                buy_ratio = 0.5
            cvd[i] = volume[i] * (2 * buy_ratio - 1)

        cvd_cum = np.cumsum(cvd)
        cvd_recent = cvd_cum[-lookback:]
        cvd_trend = (cvd_recent[-1] - cvd_recent[0])
        cvd_std = np.std(cvd[-lookback * 2:]) * math.sqrt(lookback) if n >= lookback * 2 else max(abs(cvd_trend), 1)
        cvd_z = cvd_trend / max(cvd_std, 1e-8)
        cvd_signal = np.clip(cvd_z * 4, -8, 8)

        # 4. Delivery percentage bonus (NSE-specific)
        delivery_bonus = 0.0
        if delivery_pct is not None:
            if delivery_pct > 70:
                delivery_bonus = 3.0  # Strong institutional conviction
            elif delivery_pct > 60:
                delivery_bonus = 1.5
            elif delivery_pct < 30:
                delivery_bonus = -2.0  # Speculative / day-trader driven

        # Combine: OBV 40%, CVD 30%, MFI 20%, Delivery 10%
        raw = obv_signal * 0.40 + cvd_signal * 0.30 + mfi_signal * 0.20 + delivery_bonus * 0.10
        # Scale to forecast range
        forecast = np.clip(raw * 1.5, -FORECAST_CAP, FORECAST_CAP)

        return round(float(forecast), 2)

    except Exception as e:
        logger.debug("Order flow forecast failed: %s", e)
        return 0.0


def compute_order_flow_forecasts_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    delivery_data: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute order flow forecasts for all symbols.

    Parameters
    ----------
    ohlcv_cache : dict[str, DataFrame]
    delivery_data : dict[str, float] | None
        {symbol: delivery_pct}.

    Returns
    -------
    dict[str, float]
        {symbol: forecast}.
    """
    results = {}
    delivery_data = delivery_data or {}
    for sym, df in ohlcv_cache.items():
        results[sym] = compute_order_flow_forecast(
            df, delivery_pct=delivery_data.get(sym)
        )
    return results
