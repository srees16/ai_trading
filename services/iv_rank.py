"""
IV Rank / Percentile Service — Gap A6.

Computes implied volatility rank and percentile for NSE F&O stocks
to determine optimal timing for options selling strategies.

IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV)
IV Percentile = % of days in past year where IV was below current IV

High IV Rank (> 50th percentile) → good time to sell options (premium-rich)
Low IV Rank (< 30th percentile) → bad time to sell options (premium-poor)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IVRankResult:
    """IV rank and percentile for a single instrument."""
    symbol: str
    current_iv: float          # Current implied volatility (annualized %)
    iv_rank: float             # 0-100: position within 52-week range
    iv_percentile: float       # 0-100: % of days below current IV
    high_52w: float            # 52-week high IV
    low_52w: float             # 52-week low IV
    is_premium_rich: bool      # True if IV rank > 50
    recommendation: str        # "SELL_OPTIONS" / "NEUTRAL" / "AVOID_SELLING"


def compute_iv_rank(
    current_iv: float,
    historical_ivs: list,
) -> IVRankResult:
    """Compute IV rank and percentile from historical IV data.

    Parameters
    ----------
    current_iv : float
        Current implied volatility (annualized, as decimal e.g. 0.25 = 25%).
    historical_ivs : list[float]
        Daily IV values for the past 252 trading days.

    Returns
    -------
    IVRankResult
    """
    if not historical_ivs or len(historical_ivs) < 20:
        return IVRankResult(
            symbol="", current_iv=current_iv,
            iv_rank=50.0, iv_percentile=50.0,
            high_52w=current_iv, low_52w=current_iv,
            is_premium_rich=False, recommendation="NEUTRAL",
        )

    arr = np.array(historical_ivs)
    high_52w = float(np.max(arr))
    low_52w = float(np.min(arr))

    # IV Rank
    iv_range = high_52w - low_52w
    if iv_range > 0:
        iv_rank = (current_iv - low_52w) / iv_range * 100
    else:
        iv_rank = 50.0
    iv_rank = max(0.0, min(100.0, iv_rank))

    # IV Percentile
    below_current = np.sum(arr < current_iv)
    iv_percentile = float(below_current / len(arr) * 100)

    # Recommendation
    is_rich = iv_rank > 50
    if iv_rank > 70:
        rec = "SELL_OPTIONS"
    elif iv_rank > 40:
        rec = "NEUTRAL"
    else:
        rec = "AVOID_SELLING"

    return IVRankResult(
        symbol="",
        current_iv=round(current_iv, 4),
        iv_rank=round(iv_rank, 1),
        iv_percentile=round(iv_percentile, 1),
        high_52w=round(high_52w, 4),
        low_52w=round(low_52w, 4),
        is_premium_rich=is_rich,
        recommendation=rec,
    )


def compute_iv_from_close(
    close_prices: "np.ndarray | list",
    window: int = 20,
) -> float:
    """Estimate realized/implied volatility from close prices.

    Uses standard 20-day realized vol annualized as IV proxy
    when actual option IV is not available.
    """
    arr = np.array(close_prices, dtype=float)
    if len(arr) < window + 1:
        return 0.20  # default 20%
    log_returns = np.diff(np.log(arr))
    recent_vol = float(np.std(log_returns[-window:])) * np.sqrt(252)
    return max(0.05, recent_vol)


def compute_historical_ivs(
    close_prices: "np.ndarray | list",
    window: int = 20,
) -> list:
    """Compute rolling realized vol as IV proxy for 252 trading days."""
    arr = np.array(close_prices, dtype=float)
    if len(arr) < window + 252:
        if len(arr) < window + 20:
            return []
        # Use what we have
        pass

    log_returns = np.diff(np.log(arr))
    ivs = []
    for i in range(window, len(log_returns)):
        vol = float(np.std(log_returns[i - window:i])) * np.sqrt(252)
        ivs.append(vol)
    return ivs


def compute_iv_ranks_batch(
    ohlcv_cache: "Dict[str, 'pd.DataFrame']",
) -> Dict[str, IVRankResult]:
    """Compute IV rank for all symbols in the OHLCV cache.

    Parameters
    ----------
    ohlcv_cache : dict[str, pd.DataFrame]

    Returns
    -------
    dict[str, IVRankResult]
    """
    results: Dict[str, IVRankResult] = {}
    for sym, df in ohlcv_cache.items():
        try:
            col = "Close" if "Close" in df.columns else "close"
            close = df[col].dropna().values
            if len(close) < 50:
                continue
            current_iv = compute_iv_from_close(close)
            historical = compute_historical_ivs(close)
            result = compute_iv_rank(current_iv, historical)
            result.symbol = sym
            results[sym] = result
        except Exception:
            continue
    return results
