"""
Systematic Covered Call Writing Strategy — Phase 2.1.

Generates income by selling OTM calls on existing long equity positions.
Targets 1-3% monthly premium income on the portfolio.

Research basis:
  - CBOE BuyWrite Index (BXM): consistent outperformance in range-bound markets
  - Feldman & Roy (2005): Covered calls reduce portfolio volatility by ~30%
  - India: NIFTY weekly options have rich premium due to retail participation

NSE specifics:
  - Weekly options: NIFTY, BANKNIFTY, and ~50 F&O stocks
  - Thursday expiry
  - Lot size varies by underlying (e.g., RELIANCE = 250, TCS = 150)
  - STT on options exercise is higher than on closing before expiry

Integration:
  - Called after Carver pipeline generates long positions
  - Uses VIX for conditional sizing (Phase 2.3)
  - Premium income tracked in paper trader / live journal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CoveredCallCandidate:
    """A single covered call opportunity."""
    underlying: str
    spot_price: float
    strike: float
    premium: float
    premium_pct: float      # premium / spot_price
    delta: float
    days_to_expiry: int
    lot_size: int
    lots_available: int      # based on equity holding
    expiry_date: str
    annualized_yield_pct: float = 0.0


@dataclass
class CoveredCallResult:
    """Output of the covered call scan."""
    candidates: List[CoveredCallCandidate] = field(default_factory=list)
    total_premium_potential: float = 0.0
    portfolio_yield_monthly_pct: float = 0.0
    vix_level: float = 0.0
    vix_action: str = "NORMAL"  # SELL_MORE / NORMAL / REDUCE / STOP
    computed_at: str = ""


class CoveredCallStrategy:
    """Systematic covered call writer for NSE F&O stocks.

    Parameters
    ----------
    delta_target : float
        Target call delta (OTM level). Lower = further OTM = less assignment risk.
    min_premium_pct : float
        Minimum premium as % of spot to make the trade worthwhile.
    max_days_to_expiry : int
        Maximum DTE for options to consider (weekly = 7).
    """

    def __init__(
        self,
        delta_target: float = 0.25,
        min_premium_pct: float = 0.003,
        max_days_to_expiry: int = 7,
    ):
        self.delta_target = delta_target
        self.min_premium_pct = min_premium_pct
        self.max_dte = max_days_to_expiry

    def scan_opportunities(
        self,
        holdings: Dict[str, dict],
        option_chains: Dict[str, list],
        vix: float = 15.0,
    ) -> CoveredCallResult:
        """Scan current holdings for covered call opportunities.

        Parameters
        ----------
        holdings : dict
            {symbol: {qty, avg_price, lot_size}}.
        option_chains : dict
            {symbol: [list of option contracts]}.
        vix : float
            Current India VIX level.
        """
        vix_action = self._vix_sizing_action(vix)
        candidates: List[CoveredCallCandidate] = []

        if vix_action == "STOP":
            logger.info("VIX=%.1f — STOP selling options", vix)
            return CoveredCallResult(
                vix_level=vix,
                vix_action=vix_action,
                computed_at=datetime.utcnow().isoformat(),
            )

        for symbol, info in holdings.items():
            qty = info.get("qty", 0)
            lot_size = info.get("lot_size", 1)
            avg_price = info.get("avg_price", 0)

            if qty < lot_size or symbol not in option_chains:
                continue

            lots_available = qty // lot_size
            chain = option_chains[symbol]
            best = self._find_optimal_strike(chain, avg_price, lots_available, lot_size)

            if best is not None:
                candidates.append(best)

        total_premium = sum(c.premium * c.lots_available * c.lot_size for c in candidates)
        portfolio_value = sum(
            info.get("qty", 0) * info.get("avg_price", 0) for info in holdings.values()
        )
        monthly_yield = (total_premium / portfolio_value * 100) if portfolio_value > 0 else 0

        # Apply VIX-conditional sizing
        if vix_action == "REDUCE":
            total_premium *= 0.5
            monthly_yield *= 0.5

        return CoveredCallResult(
            candidates=candidates,
            total_premium_potential=round(total_premium, 2),
            portfolio_yield_monthly_pct=round(monthly_yield, 2),
            vix_level=vix,
            vix_action=vix_action,
            computed_at=datetime.utcnow().isoformat(),
        )

    def _find_optimal_strike(
        self,
        chain: list,
        spot_price: float,
        lots_available: int,
        lot_size: int,
    ) -> Optional[CoveredCallCandidate]:
        """Select the optimal call strike from the option chain."""
        best: Optional[CoveredCallCandidate] = None
        best_score = -1.0

        for opt in chain:
            opt_type = opt.get("type", "").upper()
            if opt_type != "CE" and opt_type != "CALL":
                continue

            strike = opt.get("strike", 0)
            premium = opt.get("ltp", 0) or opt.get("premium", 0)
            dte = opt.get("dte", 0)
            delta = opt.get("delta", 0)

            if dte > self.max_dte or dte <= 0:
                continue
            if strike <= spot_price:
                continue  # must be OTM

            premium_pct = premium / spot_price if spot_price > 0 else 0
            if premium_pct < self.min_premium_pct:
                continue

            # Estimate delta if not provided
            if delta <= 0:
                moneyness = (strike - spot_price) / spot_price
                delta = max(0.05, 0.50 - moneyness * 5)

            # Score: premium weighted by proximity to target delta
            delta_dist = abs(delta - self.delta_target)
            score = premium_pct * (1.0 - delta_dist)

            if score > best_score:
                ann_yield = (premium_pct / dte * 365 * 100) if dte > 0 else 0
                best = CoveredCallCandidate(
                    underlying=opt.get("symbol", ""),
                    spot_price=spot_price,
                    strike=strike,
                    premium=premium,
                    premium_pct=round(premium_pct * 100, 3),
                    delta=round(delta, 3),
                    days_to_expiry=dte,
                    lot_size=lot_size,
                    lots_available=lots_available,
                    expiry_date=opt.get("expiry", ""),
                    annualized_yield_pct=round(ann_yield, 1),
                )
                best_score = score

        return best

    @staticmethod
    def _vix_sizing_action(vix: float) -> str:
        """VIX-conditional options sizing (Phase 2.3).

        Returns sizing action based on India VIX level.
        """
        if vix < 15:
            return "SELL_MORE"   # Low vol, safe to sell
        elif vix < 20:
            return "NORMAL"
        elif vix < 25:
            return "REDUCE"      # Elevated risk
        else:
            return "STOP"        # Crisis mode

    def should_roll(
        self,
        spot_price: float,
        strike: float,
        dte: int,
    ) -> str:
        """Determine if a short call needs rolling.

        Returns 'HOLD', 'ROLL_UP', 'CLOSE', or 'LET_EXPIRE'.
        """
        if dte <= 0:
            if spot_price < strike:
                return "LET_EXPIRE"
            return "CLOSE"

        # If stock > 90% of strike, roll up to avoid assignment
        if spot_price >= strike * 0.90:
            return "ROLL_UP"

        return "HOLD"
