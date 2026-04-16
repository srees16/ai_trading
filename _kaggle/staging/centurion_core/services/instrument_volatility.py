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


# ── GARCH(1,1) forward-looking conditional volatility ──────────────────

# Minimum observations for GARCH fitting (need enough for convergence)
_GARCH_MIN_OBS = 120
# Re-fit GARCH every N calls per symbol to avoid excessive compute
_GARCH_REFIT_INTERVAL = 5
# Cache: {symbol: (call_count, last_cond_vol)}
_garch_cache: Dict[str, tuple] = {}


def garch_daily_volatility(
    close: pd.Series,
    *,
    sym: str = "",
    fallback_lookback: int = DEFAULT_VOL_HALFLIFE,
) -> float:
    """Return 1-step-ahead conditional daily volatility from GARCH(1,1).

    Fits GARCH(1,1) on percentage returns scaled to 100 (arch convention).
    Falls back to EWMA vol if fitting fails or insufficient data.

    The result is cached per symbol to avoid re-fitting every call.
    Re-fit occurs every _GARCH_REFIT_INTERVAL calls per symbol.
    """
    global _garch_cache

    # Check cache — return cached value if not due for re-fit
    if sym and sym in _garch_cache:
        call_count, cached_vol = _garch_cache[sym]
        if call_count % _GARCH_REFIT_INTERVAL != 0:
            _garch_cache[sym] = (call_count + 1, cached_vol)
            return cached_vol

    if close is None or len(close) < _GARCH_MIN_OBS:
        return daily_price_volatility(close, lookback=fallback_lookback)

    pct_returns = close.pct_change().dropna()
    if len(pct_returns) < _GARCH_MIN_OBS:
        return daily_price_volatility(close, lookback=fallback_lookback)

    try:
        from arch import arch_model
        # Scale returns to percentage (arch convention for numerical stability)
        scaled_returns = pct_returns * 100.0

        am = arch_model(
            scaled_returns,
            vol="Garch",
            p=1,
            q=1,
            mean="Zero",         # Zero-mean — we don't need return forecast
            rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)

        # 1-step-ahead forecast: conditional variance
        forecast = res.forecast(horizon=1)
        cond_var = forecast.variance.iloc[-1, 0]  # h_{t+1}

        if np.isfinite(cond_var) and cond_var > 0:
            # Convert back from percentage to decimal
            cond_vol = np.sqrt(cond_var) / 100.0
            # Sanity bounds: 0.1% to 15% daily vol
            cond_vol = max(0.001, min(0.15, cond_vol))
            if sym:
                prev_count = _garch_cache.get(sym, (0, 0))[0]
                _garch_cache[sym] = (prev_count + 1, cond_vol)
            return cond_vol
    except Exception:
        pass  # GARCH fit failure — fall back to EWMA

    ewma_vol = daily_price_volatility(close, lookback=fallback_lookback)
    if sym:
        prev_count = _garch_cache.get(sym, (0, 0))[0]
        _garch_cache[sym] = (prev_count + 1, ewma_vol)
    return ewma_vol


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
