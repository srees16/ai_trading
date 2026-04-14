"""
Insider Activity Forecast — SAST filings and bulk/block deal signals.

Uses information from NSE SAST (Substantial Acquisition of Shares and
Takeovers) filings and bulk deal disclosures:
  - Insider buying → positive forecast (insiders are informed buyers)
  - Insider selling → negative forecast (but weaker — many sell for liquidity)
  - Bulk deals by institutional investors → directional signal

In backtest: uses volume surge + price action as a proxy for insider activity
(abnormal volume without news = likely informed trading).
In live mode: will scrape NSE SAST filings from nse-bhavcopy data.

Academic basis:
  Lakonishok & Lee (2001): Insider buying predicts returns (Sharpe ~0.4).
  Only incremental when NOT already captured by momentum signals.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VOL_LOOKBACK = 20          # 20-day volume average
_VOL_SURGE_THRESHOLD = 2.0  # 2× average volume = significant
_PRICE_THRESHOLD = 0.01     # > 1% price move with volume surge
_FORECAST_SCALAR = 10.0     # calibrated to avg|forecast| ≈ 10


def compute_insider_activity_batch(
    ohlcv_slice: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute insider activity proxy forecast for all symbols.

    Backtest proxy: abnormal volume with directional price move.
    Volume surge (>2× avg) + positive price → insider buying proxy.
    Volume surge + negative price → insider selling proxy.
    """
    result: Dict[str, float] = {}

    for sym, df in ohlcv_slice.items():
        if "Volume" not in df.columns:
            continue
        close = df["Close"]
        volume = df["Volume"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if hasattr(volume, "squeeze"):
            volume = volume.squeeze()
        close = close.dropna()
        volume = volume.dropna()
        if len(close) < _VOL_LOOKBACK + 5 or len(volume) < _VOL_LOOKBACK + 5:
            continue

        # Current volume vs 20-day average
        avg_vol = float(volume.iloc[-_VOL_LOOKBACK - 1:-1].mean())
        if avg_vol <= 0:
            continue
        current_vol = float(volume.iloc[-1])
        vol_ratio = current_vol / avg_vol

        if vol_ratio < _VOL_SURGE_THRESHOLD:
            continue  # no volume surge → no insider signal

        # Price change on the surge day
        price_change = float(close.iloc[-1] / close.iloc[-2] - 1)

        if abs(price_change) < _PRICE_THRESHOLD:
            continue  # volume surge without price move → ambiguous

        # Directional forecast: +ve price with vol surge → buying
        # -ve price with vol surge → selling (weaker signal)
        if price_change > 0:
            raw = vol_ratio * price_change * _FORECAST_SCALAR * 2.0
        else:
            raw = vol_ratio * price_change * _FORECAST_SCALAR * 1.0  # weaker for sells

        fc = max(-20.0, min(20.0, raw))
        if abs(fc) > 0.5:
            result[sym] = fc

    return result
