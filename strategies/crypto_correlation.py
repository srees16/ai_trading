"""
Crypto Correlation Forecast — Bitcoin as risk-on/risk-off indicator.

Uses Bitcoin (BTC-USD) as a macro risk-appetite proxy:
  - BTC rallying → risk-on → positive equity forecast
  - BTC crashing → risk-off → negative equity forecast
  - Correlation between BTC and equity returns → regime indicator

Research basis:
  - BTC-equity correlation increased significantly post-2020
  - BTC leads equity drawdowns by 1-3 days during risk-off events
  - Rolling 30d correlation ≥ 0.5 → correlated regime → BTC signal valid
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BTC_TICKERS = ["BTC-USD", "BTC-INR"]
_LOOKBACK_MOMENTUM = 14    # 2-week BTC momentum
_LOOKBACK_CORR = 30        # 30d rolling correlation threshold
_CORR_THRESHOLD = 0.3      # min correlation to use BTC signal
_FORECAST_SCALAR = 12.0    # calibrated to avg|forecast| ≈ 10


def _get_btc_series(ohlcv_slice: Dict[str, pd.DataFrame]) -> Optional[pd.Series]:
    """Try to find BTC data in the OHLCV slice."""
    for bt in _BTC_TICKERS:
        if bt in ohlcv_slice:
            df = ohlcv_slice[bt]
            close = df["Close"]
            if hasattr(close, "squeeze"):
                close = close.squeeze()
            close = close.dropna()
            if len(close) >= _LOOKBACK_CORR + 10:
                return close
    return None


def compute_crypto_correlation_batch(
    ohlcv_slice: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute crypto-based risk forecast for equity symbols.

    Uses BTC momentum as a market-wide risk-on/off signal, weighted by
    rolling stock-BTC correlation. Stocks with low BTC correlation get
    weaker signals (crypto not informative for them).
    """
    result: Dict[str, float] = {}

    btc_close = _get_btc_series(ohlcv_slice)
    if btc_close is None:
        return result

    # BTC momentum: 14-day return
    btc_ret = float(btc_close.iloc[-1] / btc_close.iloc[-_LOOKBACK_MOMENTUM] - 1)
    btc_daily_rets = btc_close.pct_change().dropna().iloc[-_LOOKBACK_CORR:]

    for sym, df in ohlcv_slice.items():
        if sym in _BTC_TICKERS:
            continue
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        close = close.dropna()
        if len(close) < _LOOKBACK_CORR + 10:
            continue

        stk_daily_rets = close.pct_change().dropna().iloc[-_LOOKBACK_CORR:]

        # Align by index for correlation
        aligned = pd.concat([btc_daily_rets, stk_daily_rets], axis=1, join='inner')
        if len(aligned) < 15:
            continue

        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if not np.isfinite(corr):
            continue

        # Only use BTC signal when correlation is meaningful
        if abs(corr) < _CORR_THRESHOLD:
            continue

        # BTC momentum → equity risk forecast, weighted by correlation
        raw = btc_ret * corr * _FORECAST_SCALAR
        fc = max(-20.0, min(20.0, raw))
        result[sym] = fc

    return result
