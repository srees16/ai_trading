"""
Options Overlay Strategy — Gap A1.

Systematic covered call + cash-secured put writing on NSE F&O stocks.
This is the single largest alpha source for achieving 60%+ CAGR.

Strategies:
  1. **Covered Calls** — Sell 1-month OTM calls on portfolio holdings
     - Entry: IV rank > 50, stock in portfolio, forecast weakening
     - Strike: 30-delta OTM call (≈ 2σ above current price)
     - Roll: At 50% max profit or 14 DTE remaining
     - Expected: 2–4% monthly premium on deployed capital

  2. **Cash-Secured Puts** — Sell OTM puts on stocks with BUY forecast
     - Entry: IV rank > 40, symbol has positive Carver forecast
     - Strike: 25-delta OTM put (≈ 1.5σ below current price)
     - Roll: At 50% max profit or 14 DTE
     - If assigned: stock enters portfolio at effective discount

Research basis:
  - Whaley (2002): CBOE BXM (Buy-Write Index) outperforms S&P with 25% less vol
  - NSE NIFTY covered call writing: ~10-12% annual premium at-the-money
  - Theta decay accelerates last 30 days → optimal 30-45 DTE entry

Integration:
  - Reads portfolio holdings from Carver pipeline
  - Uses IV rank from oi_signal.py
  - Generates option orders to complement equity positions
  - Premium collected added to capital rolling in volatility_target.py
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Black-Scholes Pricing
# ═══════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    from scipy.stats import norm
    return float(norm.cdf(x))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes call option price with dividend yield.

    Parameters
    ----------
    S : Current stock price
    K : Strike price
    T : Time to expiry in years
    r : Risk-free rate (annual)
    sigma : Implied volatility (annual)
    q : Continuous dividend yield (annual, e.g. 0.02 for 2%)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0, S - K)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes put option price with dividend yield."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0, K - S)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def compute_delta_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Call option delta with dividend yield."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * _norm_cdf(d1)


def compute_delta_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Put option delta with dividend yield."""
    return compute_delta_call(S, K, T, r, sigma, q) - math.exp(-q * T)


def find_strike_by_delta(
    S: float, T: float, r: float, sigma: float,
    target_delta: float, option_type: str = "CALL",
    precision: float = 0.5,
) -> float:
    """Find strike price that matches target delta.

    Uses bisection search. For CALL: target_delta should be 0.2-0.4.
    For PUT: target_delta should be -0.4 to -0.2.
    """
    if option_type == "CALL":
        lo, hi = S * 0.8, S * 1.5
        for _ in range(50):
            mid = (lo + hi) / 2
            d = compute_delta_call(S, mid, T, r, sigma)
            if abs(d - target_delta) < 0.01:
                break
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        return round(mid / precision) * precision
    else:
        lo, hi = S * 0.5, S * 1.2
        for _ in range(50):
            mid = (lo + hi) / 2
            d = compute_delta_put(S, mid, T, r, sigma)
            if abs(d - target_delta) < 0.01:
                break
            if d < target_delta:
                hi = mid
            else:
                lo = mid
        return round(mid / precision) * precision


# ═══════════════════════════════════════════════════════════════
# Option Order Data Classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class OptionOrder:
    """An option order to be executed."""
    symbol: str
    option_type: str           # "CE" or "PE"
    strike: float
    expiry_date: str           # "2026-04-24"
    action: str                # "SELL" (writing) or "BUY" (closing)
    lots: int                  # number of lots
    lot_size: int
    premium: float             # expected premium per share
    total_premium: float       # premium × lots × lot_size
    delta: float
    iv: float
    strategy: str              # "COVERED_CALL" or "CASH_SECURED_PUT"
    underlying_price: float


@dataclass
class OverlayResult:
    """Result of the options overlay scan."""
    covered_call_orders: List[OptionOrder] = field(default_factory=list)
    put_write_orders: List[OptionOrder] = field(default_factory=list)
    total_premium_expected: float = 0.0
    monthly_yield_pct: float = 0.0
    annualized_yield_pct: float = 0.0
    log: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Options Overlay Engine
# ═══════════════════════════════════════════════════════════════

class OptionsOverlay:
    """Systematic options overlay for premium collection.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate (RBI repo rate ≈ 6.5%).
    target_call_delta : float
        Delta for covered call strike selection.
    target_put_delta : float
        Delta for cash-secured put strike selection.
    min_iv_rank_call : float
        Minimum IV rank to write covered calls.
    min_iv_rank_put : float
        Minimum IV rank to write cash-secured puts.
    min_premium_pct : float
        Minimum premium yield per trade (% of underlying).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.065,
        target_call_delta: float = 0.30,
        target_put_delta: float = -0.25,
        min_iv_rank_call: float = 50.0,
        min_iv_rank_put: float = 40.0,
        min_premium_pct: float = 0.015,
        max_overlay_pct: float = 0.60,
    ):
        self.r = risk_free_rate
        self.target_call_delta = target_call_delta
        self.target_put_delta = target_put_delta
        self.min_iv_rank_call = min_iv_rank_call
        self.min_iv_rank_put = min_iv_rank_put
        self.min_premium_pct = min_premium_pct
        self.max_overlay_pct = max_overlay_pct

    def scan_covered_calls(
        self,
        holdings: Dict[str, Dict],
        iv_data: Dict[str, Dict],
        forecasts: Optional[Dict[str, float]] = None,
    ) -> List[OptionOrder]:
        """Scan portfolio holdings for covered call writing opportunities.

        Parameters
        ----------
        holdings : dict
            {symbol: {"quantity": int, "avg_price": float, "current_price": float}}
        iv_data : dict
            {symbol: {"iv": float, "iv_rank": float}}
        forecasts : dict | None
            {symbol: current_forecast} — write calls when forecast weakening

        Returns
        -------
        list[OptionOrder]
        """
        from services.oi_signal import FNO_LOT_SIZES

        orders: List[OptionOrder] = []
        days_to_expiry = 30  # Target 1-month expiry
        T = days_to_expiry / 365.0

        for sym, holding in holdings.items():
            lot_size = FNO_LOT_SIZES.get(sym, 0)
            if lot_size == 0:
                continue  # Not F&O eligible

            qty = holding.get("quantity", 0)
            if qty < lot_size:
                continue  # Need at least 1 lot for covered call

            price = holding.get("current_price", 0)
            if price <= 0:
                continue

            iv_info = iv_data.get(sym, {})
            iv = iv_info.get("iv", 0.25)
            iv_rank = iv_info.get("iv_rank", 50.0)

            if iv_rank < self.min_iv_rank_call:
                continue

            # Check if forecast is weakening (optional filter)
            if forecasts and sym in forecasts:
                if forecasts[sym] > 15.0:
                    continue  # Don't cap upside on very strong BUY signals

            # Find 30-delta call strike
            strike = find_strike_by_delta(price, T, self.r, iv, self.target_call_delta, "CALL")
            premium = black_scholes_call(price, strike, T, self.r, iv)

            if premium / price < self.min_premium_pct:
                continue  # Premium too low

            lots = qty // lot_size
            total_premium = premium * lots * lot_size
            delta = compute_delta_call(price, strike, T, self.r, iv)

            expiry_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime("%Y-%m-%d")

            orders.append(OptionOrder(
                symbol=sym,
                option_type="CE",
                strike=strike,
                expiry_date=expiry_date,
                action="SELL",
                lots=lots,
                lot_size=lot_size,
                premium=round(premium, 2),
                total_premium=round(total_premium, 2),
                delta=round(delta, 3),
                iv=round(iv, 4),
                strategy="COVERED_CALL",
                underlying_price=price,
            ))

        return orders

    def scan_cash_secured_puts(
        self,
        candidates: Dict[str, Dict],
        iv_data: Dict[str, Dict],
        available_capital: float,
    ) -> List[OptionOrder]:
        """Scan for cash-secured put writing opportunities.

        Parameters
        ----------
        candidates : dict
            {symbol: {"current_price": float, "forecast": float}}
            — stocks with positive Carver forecast (BUY candidates)
        iv_data : dict
            {symbol: {"iv": float, "iv_rank": float}}
        available_capital : float
            Capital available for margin/assignment.

        Returns
        -------
        list[OptionOrder]
        """
        from services.oi_signal import FNO_LOT_SIZES
        from config import Config

        orders: List[OptionOrder] = []
        days_to_expiry = 30
        T = days_to_expiry / 365.0
        capital_used = 0.0
        max_capital = available_capital * self.max_overlay_pct

        # Sector concentration limit: max 2 puts per sector
        _MAX_PUTS_PER_SECTOR = 2
        sector_map = Config.NSE_SECTOR_MAP if hasattr(Config, 'NSE_SECTOR_MAP') else {}
        sector_put_count: Dict[str, int] = {}

        # Sort by forecast strength (strongest first)
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1].get("forecast", 0),
            reverse=True,
        )

        for sym, info in sorted_candidates:
            lot_size = FNO_LOT_SIZES.get(sym, 0)
            if lot_size == 0:
                continue

            # Sector concentration check
            sector = sector_map.get(sym, "Unknown")
            if sector_put_count.get(sector, 0) >= _MAX_PUTS_PER_SECTOR:
                continue

            price = info.get("current_price", 0)
            forecast = info.get("forecast", 0)
            if price <= 0 or forecast <= 5.0:
                continue  # Need minimum positive conviction

            iv_info = iv_data.get(sym, {})
            iv = iv_info.get("iv", 0.25)
            iv_rank = iv_info.get("iv_rank", 50.0)

            if iv_rank < self.min_iv_rank_put:
                continue

            # Find 25-delta put strike
            strike = find_strike_by_delta(price, T, self.r, iv, self.target_put_delta, "PUT")
            premium = black_scholes_put(price, strike, T, self.r, iv)

            if premium / price < self.min_premium_pct:
                continue

            # Capital needed: strike × lot_size × SPAN margin
            # Dynamic SPAN margin: higher IV → higher margin requirement
            span_margin_pct = min(0.25, max(0.10, 0.12 + iv * 0.30))  # 10-25% based on IV
            margin_required = strike * lot_size * span_margin_pct
            if capital_used + margin_required > max_capital:
                continue

            delta = compute_delta_put(price, strike, T, self.r, iv)
            expiry_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime("%Y-%m-%d")

            orders.append(OptionOrder(
                symbol=sym,
                option_type="PE",
                strike=strike,
                expiry_date=expiry_date,
                action="SELL",
                lots=1,
                lot_size=lot_size,
                premium=round(premium, 2),
                total_premium=round(premium * lot_size, 2),
                delta=round(delta, 3),
                iv=round(iv, 4),
                strategy="CASH_SECURED_PUT",
                underlying_price=price,
            ))
            capital_used += margin_required
            sector_put_count[sector] = sector_put_count.get(sector, 0) + 1

        return orders

    def run_overlay(
        self,
        holdings: Dict[str, Dict],
        candidates: Dict[str, Dict],
        iv_data: Dict[str, Dict],
        available_capital: float,
        forecasts: Optional[Dict[str, float]] = None,
    ) -> OverlayResult:
        """Run full options overlay scan.

        Parameters
        ----------
        holdings : dict
            Current portfolio holdings {symbol: {quantity, avg_price, current_price}}
        candidates : dict
            BUY candidates {symbol: {current_price, forecast}}
        iv_data : dict
            IV data {symbol: {iv, iv_rank}}
        available_capital : float
            Total available capital
        forecasts : dict
            Current forecasts for call writing filter

        Returns
        -------
        OverlayResult
        """
        result = OverlayResult()

        # Covered calls on existing holdings
        cc_orders = self.scan_covered_calls(holdings, iv_data, forecasts)
        result.covered_call_orders = cc_orders

        # Cash-secured puts on buy candidates
        csp_orders = self.scan_cash_secured_puts(candidates, iv_data, available_capital)
        result.put_write_orders = csp_orders

        # Total premium
        cc_premium = sum(o.total_premium for o in cc_orders)
        csp_premium = sum(o.total_premium for o in csp_orders)
        total = cc_premium + csp_premium
        result.total_premium_expected = round(total, 2)

        if available_capital > 0:
            result.monthly_yield_pct = round(total / available_capital * 100, 2)
            result.annualized_yield_pct = round(result.monthly_yield_pct * 12, 2)

        result.log.append(f"Covered calls: {len(cc_orders)} orders, ₹{cc_premium:,.0f} premium")
        result.log.append(f"Cash-secured puts: {len(csp_orders)} orders, ₹{csp_premium:,.0f} premium")
        result.log.append(f"Total monthly yield: {result.monthly_yield_pct:.2f}% ({result.annualized_yield_pct:.1f}% annualized)")

        for line in result.log:
            logger.info("Options overlay: %s", line)

        return result
