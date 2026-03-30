"""
Cash-Secured Put Selling Strategy — Phase 2.2.

Systematically sells OTM puts on stocks the system wants to buy,
generating income while waiting for entry prices.

Research basis:
  - Volatility Risk Premium: Implied vol > realized vol ~85% of the time
  - Carr & Wu (2009): Variance risk premium is large and persistent
  - Win-win: Either collect premium or acquire stock at desired price

Integration:
  - Takes input from Carver pipeline BUY signals
  - Only sells puts on stocks with positive momentum forecast
  - Cash-secured only (no naked puts)
  - VIX-conditional sizing from Phase 2.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PutCandidate:
    """A cash-secured put opportunity."""
    underlying: str
    spot_price: float
    strike: float               # desired entry price
    premium: float
    premium_pct: float          # premium / strike
    delta: float
    days_to_expiry: int
    lot_size: int
    cash_required: float        # strike × lot_size (margin required)
    expiry_date: str
    forecast_score: float = 0.0  # Carver forecast for the underlying
    annualized_yield_pct: float = 0.0


@dataclass
class PutSellingResult:
    """Output of the put selling scan."""
    candidates: List[PutCandidate] = field(default_factory=list)
    total_premium_potential: float = 0.0
    total_cash_reserved: float = 0.0
    vix_level: float = 0.0
    vix_action: str = "NORMAL"
    computed_at: str = ""


class PutSellingStrategy:
    """Cash-secured put selling for systematic entries.

    Parameters
    ----------
    delta_target : float
        Target put delta (OTM level). -0.20 to -0.30 typical.
    max_concurrent : int
        Maximum number of concurrent put positions.
    min_forecast : float
        Minimum Carver combined forecast to consider selling puts.
    max_cash_pct : float
        Maximum % of available capital to allocate to put selling.
    """

    def __init__(
        self,
        delta_target: float = -0.25,
        max_concurrent: int = 3,
        min_forecast: float = 5.0,
        max_cash_pct: float = 0.30,
    ):
        self.delta_target = delta_target
        self.max_concurrent = max_concurrent
        self.min_forecast = min_forecast
        self.max_cash_pct = max_cash_pct

    def scan_opportunities(
        self,
        buy_signals: Dict[str, float],
        option_chains: Dict[str, list],
        available_capital: float,
        vix: float = 15.0,
    ) -> PutSellingResult:
        """Find put selling opportunities for BUY-signaled stocks.

        Parameters
        ----------
        buy_signals : dict
            {symbol: carver_combined_forecast} for BUY candidates.
        option_chains : dict
            {symbol: [list of option contracts]}.
        available_capital : float
            Cash available for securing puts.
        vix : float
            Current India VIX.
        """
        from kite_connect.options.covered_call_strategy import CoveredCallStrategy
        vix_action = CoveredCallStrategy._vix_sizing_action(vix)

        if vix_action == "STOP":
            logger.info("VIX=%.1f — stopping put selling", vix)
            return PutSellingResult(
                vix_level=vix,
                vix_action=vix_action,
                computed_at=datetime.utcnow().isoformat(),
            )

        max_cash = available_capital * self.max_cash_pct
        candidates: List[PutCandidate] = []
        cash_used = 0.0

        # Sort by forecast strength (highest conviction first)
        sorted_signals = sorted(buy_signals.items(), key=lambda x: x[1], reverse=True)

        for symbol, forecast in sorted_signals:
            if forecast < self.min_forecast:
                continue
            if symbol not in option_chains:
                continue
            if len(candidates) >= self.max_concurrent:
                break

            chain = option_chains[symbol]
            best = self._find_optimal_put(chain, forecast)

            if best is not None:
                if cash_used + best.cash_required > max_cash:
                    continue
                candidates.append(best)
                cash_used += best.cash_required

        total_premium = sum(c.premium * c.lot_size for c in candidates)

        if vix_action == "REDUCE":
            total_premium *= 0.5

        return PutSellingResult(
            candidates=candidates,
            total_premium_potential=round(total_premium, 2),
            total_cash_reserved=round(cash_used, 2),
            vix_level=vix,
            vix_action=vix_action,
            computed_at=datetime.utcnow().isoformat(),
        )

    def _find_optimal_put(
        self,
        chain: list,
        forecast: float,
    ) -> Optional[PutCandidate]:
        """Select optimal put strike from option chain."""
        best: Optional[PutCandidate] = None
        best_score = -1.0

        for opt in chain:
            opt_type = opt.get("type", "").upper()
            if opt_type != "PE" and opt_type != "PUT":
                continue

            strike = opt.get("strike", 0)
            spot = opt.get("spot", 0) or opt.get("underlying_price", strike * 1.05)
            premium = opt.get("ltp", 0) or opt.get("premium", 0)
            dte = opt.get("dte", 0)
            delta = opt.get("delta", 0)
            lot_size = opt.get("lot_size", 1)

            if dte <= 0 or dte > 14:  # max 2-week puts
                continue
            if strike >= spot:
                continue  # must be OTM

            premium_pct = premium / strike if strike > 0 else 0
            if premium_pct < 0.002:  # below 0.2% not worth it
                continue

            # Estimate delta if not provided
            if delta >= 0:
                moneyness = (spot - strike) / spot
                delta = -max(0.05, 0.50 - moneyness * 5)

            delta_dist = abs(abs(delta) - abs(self.delta_target))
            score = premium_pct * (1.0 - delta_dist) * min(forecast / 10.0, 2.0)

            if score > best_score:
                cash_required = strike * lot_size
                ann_yield = (premium_pct / dte * 365 * 100) if dte > 0 else 0
                best = PutCandidate(
                    underlying=opt.get("symbol", ""),
                    spot_price=spot,
                    strike=strike,
                    premium=premium,
                    premium_pct=round(premium_pct * 100, 3),
                    delta=round(delta, 3),
                    days_to_expiry=dte,
                    lot_size=lot_size,
                    cash_required=cash_required,
                    expiry_date=opt.get("expiry", ""),
                    forecast_score=forecast,
                    annualized_yield_pct=round(ann_yield, 1),
                )
                best_score = score

        return best

    def manage_assignment(
        self,
        assigned_symbol: str,
        strike: float,
        lot_size: int,
        premium_collected: float,
    ) -> dict:
        """Handle put assignment — stock acquired at desired price.

        Returns action dict for the executor.
        """
        effective_entry = strike - premium_collected / lot_size
        logger.info(
            "Put assigned: %s at strike=%.2f, effective_entry=%.2f",
            assigned_symbol, strike, effective_entry,
        )
        return {
            "action": "ACCEPT_ASSIGNMENT",
            "symbol": assigned_symbol,
            "quantity": lot_size,
            "entry_price": effective_entry,
            "note": f"Put assigned at {strike}, net entry after premium: {effective_entry:.2f}",
        }
