"""
Carry Trading Rule for Equities — Carver Ch. 7 / Appendix B.

For equities the carry signal is:
    carry = dividend_yield - funding_cost

Where:
  - dividend_yield: trailing 12-month dividend / current price
  - funding_cost: risk-free rate (RBI repo rate for IND stocks)

The raw carry is then volatility-adjusted and scaled to a forecast:
    forecast = (carry / annual_vol) × SCALAR_CARRY

This is a slow, fundamental forecast that naturally complements
momentum-based EWMAC rules.  Carry tends to be weakly correlated
with momentum (Carver: ~0.25), providing valuable diversification.

For NSE stocks:
  - Dividend yield sourced from yfinance ``info['dividendYield']``
  - Funding cost ≈ RBI repo rate (currently ~6.5 %)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from services.forecast_scalar import carry_to_forecast
from services.instrument_volatility import annual_price_volatility

logger = logging.getLogger(__name__)

# RBI repo rate as of March 2026 — update periodically
DEFAULT_FUNDING_COST = 0.065  # 6.5 %


@dataclass
class CarryForecast:
    """Output of the carry rule for one equity."""
    symbol: str
    forecast: float              # -20 to +20
    dividend_yield: float        # decimal (e.g. 0.02 = 2%)
    funding_cost: float          # decimal
    net_carry: float             # div_yield - funding
    annual_vol: float            # annualised price vol


def compute_carry(
    symbol: str,
    close: pd.Series,
    dividend_yield: Optional[float] = None,
    funding_cost: float = DEFAULT_FUNDING_COST,
) -> Optional[CarryForecast]:
    """Compute equity carry forecast for one stock.

    Parameters
    ----------
    symbol : str
        Ticker (plain or with .NS suffix).
    close : pd.Series
        Daily closing prices (for volatility estimation).
    dividend_yield : float | None
        Trailing 12-month dividend yield as decimal.
        If None, attempts to fetch from yfinance.
    funding_cost : float
        Annualised funding cost (risk-free rate).

    Returns
    -------
    CarryForecast or None if data insufficient.
    """
    if close is None or len(close) < 30:
        return None

    # Fetch dividend yield if not provided
    if dividend_yield is None:
        dividend_yield = _fetch_dividend_yield(symbol)
    if dividend_yield is None:
        dividend_yield = 0.0  # assume zero if unavailable

    net_carry = dividend_yield - funding_cost
    annual_vol = annual_price_volatility(close)

    if annual_vol <= 0:
        return None

    forecast = carry_to_forecast(net_carry, annual_vol)

    return CarryForecast(
        symbol=symbol,
        forecast=forecast,
        dividend_yield=dividend_yield,
        funding_cost=funding_cost,
        net_carry=net_carry,
        annual_vol=annual_vol,
    )


def compute_carry_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    dividend_yields: Optional[Dict[str, float]] = None,
    funding_cost: float = DEFAULT_FUNDING_COST,
) -> Dict[str, CarryForecast]:
    """Compute carry forecasts for all symbols.

    Parameters
    ----------
    ohlcv_cache : dict[str, DataFrame]
        ``{symbol: df}`` with ``"Close"`` column.
    dividend_yields : dict[str, float] | None
        Pre-fetched yields; keys = plain symbol.
    funding_cost : float
        Risk-free rate.

    Returns
    -------
    dict[str, CarryForecast]
    """
    dividend_yields = dividend_yields or {}
    results: Dict[str, CarryForecast] = {}

    for sym, df in ohlcv_cache.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        dy = dividend_yields.get(sym)
        fc = compute_carry(sym, close, dividend_yield=dy, funding_cost=funding_cost)
        if fc is not None:
            results[sym] = fc

    logger.info("Carry computed for %d / %d symbols", len(results), len(ohlcv_cache))
    return results


def _fetch_dividend_yield(symbol: str) -> Optional[float]:
    """Fetch trailing dividend yield from yfinance."""
    try:
        import yfinance as yf
        ns = symbol if any(c in symbol for c in '.-=^') else f"{symbol}.NS"
        info = yf.Ticker(ns).info
        dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        if dy is not None and dy > 0:
            return float(dy)
    except Exception:
        pass
    return None


def fetch_dividend_yields_batch(symbols: list[str]) -> Dict[str, float]:
    """Batch-fetch dividend yields (best-effort)."""
    results: Dict[str, float] = {}
    try:
        import yfinance as yf
        for sym in symbols:
            ns = sym if any(c in sym for c in '.-=^') else f"{sym}.NS"
            try:
                info = yf.Ticker(ns).info
                dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
                if dy is not None and dy > 0:
                    results[sym] = float(dy)
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Batch dividend yield fetch failed: %s", exc)
    logger.info("Fetched dividend yields for %d / %d symbols", len(results), len(symbols))
    return results
