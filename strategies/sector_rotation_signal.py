"""
Sector Rotation Forecast Signal — converts sector momentum tiers to per-stock forecasts.

Uses the existing `services.sector_rotation` infrastructure which ranks 12 NIFTY
sectoral indices by dual-timeframe (1M + 3M) momentum into TOP / MID / BOTTOM tiers.

Forecast mapping:
  - Stocks in TOP sector:    +8  (bullish momentum, overweight)
  - Stocks in MID sector:     0  (neutral, no signal)
  - Stocks in BOTTOM sector: -8  (bearish momentum, underweight)

This is a slow, macro-level signal that complements stock-level trend/momentum
signals by capturing sector-level rotation effects (e.g., cyclical → defensive shifts).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Forecast amplitudes per sector tier (Carver convention: target avg|fc| ≈ 10)
_TOP_FORECAST = 8.0
_MID_FORECAST = 0.0
_BOTTOM_FORECAST = -8.0


def _build_stock_to_sector_map(sector_indices: Dict[str, list]) -> Dict[str, str]:
    """Build reverse map: stock_symbol → sector_name."""
    result: Dict[str, str] = {}
    for sector, members in sector_indices.items():
        for sym in members:
            # Store both bare and .NS variants for flexible lookup
            result[sym] = sector
            result[f"{sym}.NS"] = sector
    return result


def compute_sector_rotation_forecast_batch(
    ohlcv_slice: Dict[str, pd.DataFrame],
    sector_indices: Optional[Dict[str, list]] = None,
) -> Dict[str, float]:
    """Compute sector rotation forecast for all symbols in the slice.

    Parameters
    ----------
    ohlcv_slice : dict[str, DataFrame]
        Current OHLCV data per symbol (used to compute constituent-level sector returns).
    sector_indices : dict[str, list] | None
        Sector → constituent list. If None, loads from INDEX_CONSTITUENTS.

    Returns
    -------
    dict[str, float]
        Symbol → forecast value in [-20, +20].
    """
    if not ohlcv_slice:
        return {}

    # Load sector constituents if not provided
    if sector_indices is None:
        try:
            from kite_connect.core.config import INDEX_CONSTITUENTS
            # Filter to sectoral indices only (skip NIFTY50, NIFTYBANK etc.)
            _sectoral_prefixes = (
                "NIFTYAUTO", "NIFTYFMCG", "NIFTYMETAL", "NIFTYPHARMA",
                "NIFTYREALTY", "NIFTYPSUBANK", "NIFTYFINSERV", "NIFTYMEDIA",
                "NIFTYCONSUMERDURABLES", "NIFTYOILGAS", "NIFTYIT", "NIFTYENERGY",
            )
            sector_indices = {
                k: v for k, v in INDEX_CONSTITUENTS.items()
                if k in _sectoral_prefixes
            }
        except ImportError:
            logger.debug("INDEX_CONSTITUENTS not available for sector rotation signal")
            return {}

    if not sector_indices:
        return {}

    # Build stock → sector reverse map
    _stock_sector = _build_stock_to_sector_map(sector_indices)

    # Compute sector returns from constituent OHLCV (backtest-compatible, no yfinance call)
    import numpy as np
    sector_returns_1m: Dict[str, list] = {s: [] for s in sector_indices}
    sector_returns_3m: Dict[str, list] = {s: [] for s in sector_indices}

    for sym, df in ohlcv_slice.items():
        bare = sym.replace('.NS', '').replace('.BO', '')
        sector = _stock_sector.get(bare) or _stock_sector.get(sym)
        if sector is None:
            continue
        if "Close" not in df.columns:
            continue
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        close = close.dropna()
        if len(close) >= 22:
            ret_1m = float(close.iloc[-1] / close.iloc[-22] - 1)
            sector_returns_1m[sector].append(ret_1m)
        if len(close) >= 63:
            ret_3m = float(close.iloc[-1] / close.iloc[-63] - 1)
            sector_returns_3m[sector].append(ret_3m)

    # Compute sector combined scores: 0.6 × 1M + 0.4 × 3M
    sector_scores: Dict[str, float] = {}
    for sector in sector_indices:
        r1m = sector_returns_1m.get(sector, [])
        r3m = sector_returns_3m.get(sector, [])
        if r1m:
            avg_1m = float(np.mean(r1m))
            avg_3m = float(np.mean(r3m)) if r3m else avg_1m
            sector_scores[sector] = 0.6 * avg_1m + 0.4 * avg_3m

    if not sector_scores:
        return {}

    # Rank sectors and assign tiers
    sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_sectors)
    top_n = max(1, n // 3)

    sector_tier: Dict[str, str] = {}
    for rank, (sector, _) in enumerate(sorted_sectors):
        if rank < top_n:
            sector_tier[sector] = "TOP"
        elif rank >= n - top_n:
            sector_tier[sector] = "BOTTOM"
        else:
            sector_tier[sector] = "MID"

    # Map tier → forecast for each stock
    result: Dict[str, float] = {}
    for sym in ohlcv_slice:
        bare = sym.replace('.NS', '').replace('.BO', '')
        sector = _stock_sector.get(bare) or _stock_sector.get(sym)
        if sector is None:
            continue
        tier = sector_tier.get(sector, "MID")
        if tier == "TOP":
            result[sym] = _TOP_FORECAST
        elif tier == "BOTTOM":
            result[sym] = _BOTTOM_FORECAST
        # MID → 0.0, omitted (no signal)

    return result
