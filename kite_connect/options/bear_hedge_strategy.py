"""
Options-Based Bear Hedging Strategy — Phase 2.5.

Since IND equities cannot be shorted directly via equity delivery (CNC),
this module implements downside protection using options:

Strategies:
  1. **Protective Put**: Buy OTM puts on held positions → capped downside
  2. **Covered Call**: Sell OTM calls on held positions → premium income, cap upside
  3. **Collar**: Buy OTM put + sell OTM call → zero/low-cost protection

This replaces direct short selling for Indian equities. Activated when:
  - HMM regime = BEAR or CRISIS
  - VIX is in the sweet spot (18-35) — don't buy expensive puts in panic
  - Portfolio has open long positions to hedge

Integration:
  - Called from carver_pipeline.py Step 9b (replaces SHORT trade plans)
  - Uses Kite option chain for strike selection
  - Premium budget controlled by OPTIONS_HEDGE_MAX_PORTFOLIO_PCT (5%)
  - Hedges journaled via options_executor for lifecycle tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HedgeCandidate:
    """A single hedge opportunity."""
    underlying: str
    spot_price: float
    hedge_type: str            # PROTECTIVE_PUT | COVERED_CALL | COLLAR
    # Put leg (for protective_put / collar)
    put_strike: float = 0.0
    put_premium: float = 0.0
    put_delta: float = 0.0
    put_dte: int = 0
    # Call leg (for covered_call / collar)
    call_strike: float = 0.0
    call_premium: float = 0.0
    call_delta: float = 0.0
    call_dte: int = 0
    # Common
    lot_size: int = 1
    lots: int = 1
    net_premium: float = 0.0   # negative = cost, positive = income
    max_loss_pct: float = 0.0  # worst-case loss as % of spot
    protection_pct: float = 0.0  # protected downside %
    expiry_date: str = ""


@dataclass
class BearHedgeResult:
    """Output of the bear hedge scan."""
    candidates: List[HedgeCandidate] = field(default_factory=list)
    total_premium_cost: float = 0.0
    total_protection_value: float = 0.0
    portfolio_hedge_pct: float = 0.0  # % of portfolio hedged
    vix_level: float = 0.0
    regime: str = ""
    computed_at: str = ""
    skipped_reason: str = ""


class BearHedgeStrategy:
    """Options-based bear hedging for IND equities.

    Parameters
    ----------
    strategy : str
        Hedging approach: "protective_put", "covered_call", or "collar".
    max_premium_pct : float
        Max premium budget as fraction of portfolio value.
    put_delta_target : float
        Target delta for protective puts (e.g., -0.25 = ~10% OTM).
    call_delta_target : float
        Target delta for covered calls (e.g., 0.25 = ~10% OTM).
    min_dte : int
        Minimum days to expiry for hedge options.
    max_dte : int
        Maximum days to expiry for hedge options.
    """

    def __init__(
        self,
        strategy: str = "protective_put",
        max_premium_pct: float = 0.05,
        put_delta_target: float = 0.25,
        call_delta_target: float = 0.25,
        min_dte: int = 15,
        max_dte: int = 45,
    ):
        self.strategy = strategy
        self.max_premium_pct = max_premium_pct
        self.put_delta_target = put_delta_target
        self.call_delta_target = call_delta_target
        self.min_dte = min_dte
        self.max_dte = max_dte

    def scan_hedge_opportunities(
        self,
        holdings: Dict[str, dict],
        option_chains: Dict[str, list],
        vix: float,
        regime: str,
        portfolio_value: float,
    ) -> BearHedgeResult:
        """Scan current holdings for hedging opportunities.

        Parameters
        ----------
        holdings : dict
            {symbol: {qty, avg_price, lot_size, ltp}}.
        option_chains : dict
            {symbol: [option contracts from chain]}.
        vix : float
            Current India VIX.
        regime : str
            HMM regime: TRENDING_BULL, TRENDING_BEAR, etc.
        portfolio_value : float
            Total portfolio value for budget calculation.
        """
        result = BearHedgeResult(
            vix_level=vix,
            regime=regime,
            computed_at=datetime.utcnow().isoformat(),
        )

        # Gate checks
        try:
            from config import Config
            min_vix = getattr(Config, "OPTIONS_HEDGE_MIN_VIX", 18.0)
            max_vix = getattr(Config, "OPTIONS_HEDGE_MAX_VIX", 35.0)
            required_regime = getattr(Config, "OPTIONS_HEDGE_REGIME_REQUIRED", "bear")
        except Exception:
            min_vix, max_vix, required_regime = 18.0, 35.0, "bear"

        regime_lower = regime.lower().replace("trending_", "")
        if regime_lower not in (required_regime, "crisis", "high_volatility"):
            result.skipped_reason = f"Regime '{regime}' does not require hedging"
            return result

        if vix > max_vix:
            result.skipped_reason = f"VIX {vix:.1f} > {max_vix} — puts too expensive"
            return result

        # Premium budget
        premium_budget = portfolio_value * self.max_premium_pct
        premium_spent = 0.0

        for symbol, info in holdings.items():
            qty = info.get("qty", 0)
            lot_size = info.get("lot_size", 1)
            spot = info.get("ltp", 0) or info.get("avg_price", 0)

            if qty < lot_size or spot <= 0 or symbol not in option_chains:
                continue

            lots = qty // lot_size
            chain = option_chains[symbol]

            candidate = None
            if self.strategy == "protective_put":
                candidate = self._find_protective_put(symbol, spot, chain, lots, lot_size)
            elif self.strategy == "covered_call":
                candidate = self._find_covered_call(symbol, spot, chain, lots, lot_size)
            elif self.strategy == "collar":
                candidate = self._find_collar(symbol, spot, chain, lots, lot_size)

            if candidate is None:
                continue

            # Budget check
            cost = abs(candidate.net_premium) * candidate.lots * candidate.lot_size
            if cost > 0 and (premium_spent + cost) > premium_budget:
                logger.info("Hedge budget exhausted: spent ₹%.0f / ₹%.0f", premium_spent, premium_budget)
                break

            premium_spent += cost
            result.candidates.append(candidate)

        result.total_premium_cost = round(premium_spent, 2)
        if portfolio_value > 0:
            hedged_value = sum(
                c.spot_price * c.lots * c.lot_size for c in result.candidates
            )
            result.portfolio_hedge_pct = round(hedged_value / portfolio_value * 100, 1)
            result.total_protection_value = round(hedged_value, 2)

        return result

    def _find_protective_put(
        self, symbol: str, spot: float, chain: list, lots: int, lot_size: int,
    ) -> Optional[HedgeCandidate]:
        """Find optimal OTM put for downside protection."""
        best: Optional[HedgeCandidate] = None
        best_score = -1.0

        for opt in chain:
            opt_type = (opt.get("type") or opt.get("instrument_type", "")).upper()
            if opt_type not in ("PE", "PUT"):
                continue

            strike = opt.get("strike", 0)
            premium = opt.get("ltp", 0) or opt.get("premium", 0)
            dte = opt.get("dte", 0)
            delta = abs(opt.get("delta", 0))

            if dte < self.min_dte or dte > self.max_dte:
                continue
            if strike >= spot:
                continue  # must be OTM put
            if premium <= 0:
                continue

            # Score: prefer delta near target, reasonable premium
            delta_diff = abs(delta - self.put_delta_target)
            premium_pct = premium / spot
            protection = (spot - strike) / spot  # how far OTM

            score = (1.0 - delta_diff) * 0.5 + (1.0 - min(premium_pct / 0.03, 1.0)) * 0.3 + protection * 0.2

            if score > best_score:
                best_score = score
                best = HedgeCandidate(
                    underlying=symbol,
                    spot_price=spot,
                    hedge_type="PROTECTIVE_PUT",
                    put_strike=strike,
                    put_premium=premium,
                    put_delta=delta,
                    put_dte=dte,
                    lot_size=lot_size,
                    lots=lots,
                    net_premium=-premium,  # cost
                    max_loss_pct=round((spot - strike + premium) / spot * 100, 2),
                    protection_pct=round((spot - strike) / spot * 100, 1),
                    expiry_date=str(opt.get("expiry", "")),
                )

        return best

    def _find_covered_call(
        self, symbol: str, spot: float, chain: list, lots: int, lot_size: int,
    ) -> Optional[HedgeCandidate]:
        """Find optimal OTM call to sell for premium income."""
        best: Optional[HedgeCandidate] = None
        best_score = -1.0

        for opt in chain:
            opt_type = (opt.get("type") or opt.get("instrument_type", "")).upper()
            if opt_type not in ("CE", "CALL"):
                continue

            strike = opt.get("strike", 0)
            premium = opt.get("ltp", 0) or opt.get("premium", 0)
            dte = opt.get("dte", 0)
            delta = abs(opt.get("delta", 0))

            if dte < self.min_dte or dte > self.max_dte:
                continue
            if strike <= spot:
                continue  # must be OTM call
            if premium <= 0:
                continue

            delta_diff = abs(delta - self.call_delta_target)
            premium_pct = premium / spot
            score = (1.0 - delta_diff) * 0.5 + premium_pct * 10 * 0.5

            if score > best_score:
                best_score = score
                best = HedgeCandidate(
                    underlying=symbol,
                    spot_price=spot,
                    hedge_type="COVERED_CALL",
                    call_strike=strike,
                    call_premium=premium,
                    call_delta=delta,
                    call_dte=dte,
                    lot_size=lot_size,
                    lots=lots,
                    net_premium=premium,  # income
                    max_loss_pct=0.0,     # no additional loss from selling call
                    protection_pct=round(premium / spot * 100, 1),  # premium cushion
                    expiry_date=str(opt.get("expiry", "")),
                )

        return best

    def _find_collar(
        self, symbol: str, spot: float, chain: list, lots: int, lot_size: int,
    ) -> Optional[HedgeCandidate]:
        """Find put+call collar for zero/low-cost protection."""
        put_cand = self._find_protective_put(symbol, spot, chain, lots, lot_size)
        call_cand = self._find_covered_call(symbol, spot, chain, lots, lot_size)

        if put_cand is None or call_cand is None:
            return put_cand or call_cand

        # Collar = buy put + sell call
        net_premium = call_cand.call_premium - put_cand.put_premium
        return HedgeCandidate(
            underlying=symbol,
            spot_price=spot,
            hedge_type="COLLAR",
            put_strike=put_cand.put_strike,
            put_premium=put_cand.put_premium,
            put_delta=put_cand.put_delta,
            put_dte=put_cand.put_dte,
            call_strike=call_cand.call_strike,
            call_premium=call_cand.call_premium,
            call_delta=call_cand.call_delta,
            call_dte=call_cand.call_dte,
            lot_size=lot_size,
            lots=lots,
            net_premium=round(net_premium, 2),
            max_loss_pct=round((spot - put_cand.put_strike + abs(net_premium)) / spot * 100, 2),
            protection_pct=put_cand.protection_pct,
            expiry_date=put_cand.expiry_date,
        )
