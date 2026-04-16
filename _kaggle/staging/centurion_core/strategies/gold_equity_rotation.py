"""
Gold-Equity Rotation Forecast — Crisis hedge via gold/equity relative strength.

Uses GOLDBEES (or gold price proxy) relative to NIFTY50 to generate
rotation signals:
  - Gold outperforming equity → risk-off → reduce equity exposure (negative fc)
  - Equity outperforming gold → risk-on → increase equity exposure (positive fc)
  - Rate of change in gold/equity ratio → momentum of risk regime shift

Academic basis:
  Baur & Lucey (2010): Gold as safe-haven in extreme equity market conditions.
  Gold-equity correlation turns negative in crises but ~0 normally.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Gold proxy tickers (tried in order)
_GOLD_TICKERS = ["GOLDBEES.NS", "GOLDIETF.NS"]
_LOOKBACK_FAST = 20   # 1-month relative strength
_LOOKBACK_SLOW = 63   # 3-month relative strength
_FORECAST_SCALAR = 15.0  # calibrated to avg|forecast| ≈ 10


def compute_gold_equity_rotation_batch(
    ohlcv_slice: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute gold-equity rotation forecast for equity symbols.

    If GOLDBEES is in the OHLCV slice, compare its momentum to each stock.
    Stocks that are strong relative to gold get positive forecast (risk-on).
    Stocks weak relative to gold get negative forecast (risk-off → avoid).
    """
    result: Dict[str, float] = {}

    # Find gold ticker in OHLCV data
    gold_df = None
    gold_sym = None
    for gt in _GOLD_TICKERS:
        if gt in ohlcv_slice:
            gold_df = ohlcv_slice[gt]
            gold_sym = gt
            break

    if gold_df is None or len(gold_df) < _LOOKBACK_SLOW + 5:
        return result

    gold_close = gold_df["Close"]
    if hasattr(gold_close, "squeeze"):
        gold_close = gold_close.squeeze()
    gold_close = gold_close.dropna()
    if len(gold_close) < _LOOKBACK_SLOW + 5:
        return result

    # Gold momentum (fast and slow)
    gold_ret_fast = float(gold_close.iloc[-1] / gold_close.iloc[-_LOOKBACK_FAST] - 1)
    gold_ret_slow = float(gold_close.iloc[-1] / gold_close.iloc[-_LOOKBACK_SLOW] - 1)

    for sym, df in ohlcv_slice.items():
        if sym == gold_sym:
            continue
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        close = close.dropna()
        if len(close) < _LOOKBACK_SLOW + 5:
            continue

        # Stock momentum
        stk_ret_fast = float(close.iloc[-1] / close.iloc[-_LOOKBACK_FAST] - 1)
        stk_ret_slow = float(close.iloc[-1] / close.iloc[-_LOOKBACK_SLOW] - 1)

        # Relative strength: stock vs gold (positive = stock winning)
        rel_fast = stk_ret_fast - gold_ret_fast
        rel_slow = stk_ret_slow - gold_ret_slow

        # Blend 60% fast + 40% slow
        raw = rel_fast * 0.6 + rel_slow * 0.4
        fc = raw * _FORECAST_SCALAR
        fc = max(-20.0, min(20.0, fc))
        result[sym] = fc

    return result
