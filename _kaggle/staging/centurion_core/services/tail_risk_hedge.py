"""
Tail Risk Hedging via Portfolio Puts — Phase 3.4.

Provides portfolio-level crash protection by purchasing NIFTY put
options when conditions warrant hedging.

When to hedge:
  - VIX rising rapidly (3-day change > +30%)
  - Portfolio at critical drawdown (>10%)
  - Regime detector signals bear market transition
  - Pre-event hedging (elections, budget, RBI)

Cost: ~2-4% annual portfolio drag from put premiums
Payoff: Protection against 20%+ market crashes (occurs ~1 per 5 years)

Integration:
  - Called from portfolio_vol_monitor when drawdown > CRITICAL
  - Hedges with NIFTY monthly puts at ~5% OTM
  - Hedge ratio: 50-100% of portfolio notional
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HedgeRecommendation:
    """Recommendation for a tail risk hedge."""
    action: str               # BUY_HEDGE / HOLD / CLOSE_HEDGE / NONE
    instrument: str = ""      # e.g., "NIFTY 24000 PE"
    strike: float = 0.0
    lots: int = 0
    premium_per_lot: float = 0.0
    total_cost: float = 0.0
    hedge_ratio_pct: float = 0.0  # % of portfolio hedged
    reason: str = ""
    vix: float = 0.0
    drawdown_pct: float = 0.0


@dataclass
class TailRiskAssessment:
    """Assessment of current tail risk conditions."""
    vix_level: float = 0.0
    vix_3d_change_pct: float = 0.0
    portfolio_drawdown_pct: float = 0.0
    hedge_urgency: str = "LOW"   # LOW / MEDIUM / HIGH / CRITICAL
    recommendation: Optional[HedgeRecommendation] = None
    computed_at: str = ""


class TailRiskHedge:
    """Portfolio-level crash protection engine.

    Parameters
    ----------
    otm_pct : float
        How far out-of-the-money for hedge puts (default 0.05 = 5%).
    max_hedge_cost_pct : float
        Maximum annual portfolio % to spend on hedges.
    hedge_ratio : float
        Fraction of portfolio to hedge (0.5 = 50%).
    nifty_lot_size : int
        NIFTY futures/options lot size.
    """

    def __init__(
        self,
        otm_pct: float = 0.05,
        max_hedge_cost_pct: float = 0.04,
        hedge_ratio: float = 0.50,
        nifty_lot_size: int = 25,
    ):
        self.otm_pct = otm_pct
        self.max_hedge_cost = max_hedge_cost_pct
        self.hedge_ratio = hedge_ratio
        self.nifty_lot_size = nifty_lot_size

    def assess(
        self,
        portfolio_value: float,
        drawdown_pct: float,
        vix: float,
        vix_3d_ago: float = 0.0,
        nifty_spot: float = 0.0,
        existing_hedge: bool = False,
    ) -> TailRiskAssessment:
        """Assess tail risk and generate hedging recommendation.

        Parameters
        ----------
        portfolio_value : float
            Current portfolio notional value.
        drawdown_pct : float
            Current portfolio drawdown (0-100).
        vix : float
            Current India VIX level.
        vix_3d_ago : float
            VIX level 3 trading days ago (for spike detection).
        nifty_spot : float
            Current NIFTY 50 spot price.
        existing_hedge : bool
            Whether a hedge is already in place.
        """
        vix_change = ((vix - vix_3d_ago) / vix_3d_ago * 100) if vix_3d_ago > 0 else 0

        # Determine urgency
        urgency = "LOW"
        reason_parts = []

        if drawdown_pct >= 15:
            urgency = "CRITICAL"
            reason_parts.append(f"DD={drawdown_pct:.1f}%")
        elif drawdown_pct >= 10:
            urgency = "HIGH"
            reason_parts.append(f"DD={drawdown_pct:.1f}%")

        if vix_change > 30:
            urgency = max(urgency, "HIGH", key=["LOW", "MEDIUM", "HIGH", "CRITICAL"].index)
            reason_parts.append(f"VIX spike +{vix_change:.0f}%")

        if vix > 25:
            urgency = max(urgency, "MEDIUM", key=["LOW", "MEDIUM", "HIGH", "CRITICAL"].index)
            reason_parts.append(f"VIX={vix:.1f}")

        # Generate recommendation
        recommendation = None
        if urgency in ("HIGH", "CRITICAL") and not existing_hedge and nifty_spot > 0:
            strike = round(nifty_spot * (1 - self.otm_pct) / 50) * 50  # round to nearest 50
            notional_to_hedge = portfolio_value * self.hedge_ratio
            lots = max(1, int(notional_to_hedge / (nifty_spot * self.nifty_lot_size)))

            # Estimate premium (rough: ~1-3% of spot for 5% OTM monthly put)
            est_premium = nifty_spot * 0.015  # ~1.5% of spot per lot
            total_cost = est_premium * lots * self.nifty_lot_size
            hedge_ratio_pct = (lots * self.nifty_lot_size * nifty_spot) / portfolio_value * 100

            recommendation = HedgeRecommendation(
                action="BUY_HEDGE",
                instrument=f"NIFTY {strike} PE",
                strike=strike,
                lots=lots,
                premium_per_lot=round(est_premium, 2),
                total_cost=round(total_cost, 2),
                hedge_ratio_pct=round(hedge_ratio_pct, 1),
                reason="; ".join(reason_parts),
                vix=vix,
                drawdown_pct=drawdown_pct,
            )
        elif existing_hedge and urgency == "LOW":
            recommendation = HedgeRecommendation(
                action="CLOSE_HEDGE",
                reason="Conditions normalized",
                vix=vix,
                drawdown_pct=drawdown_pct,
            )
        elif existing_hedge:
            recommendation = HedgeRecommendation(
                action="HOLD",
                reason="Hedge active, conditions still elevated",
                vix=vix,
                drawdown_pct=drawdown_pct,
            )
        else:
            recommendation = HedgeRecommendation(
                action="NONE",
                reason="No hedge needed",
                vix=vix,
                drawdown_pct=drawdown_pct,
            )

        return TailRiskAssessment(
            vix_level=vix,
            vix_3d_change_pct=round(vix_change, 1),
            portfolio_drawdown_pct=drawdown_pct,
            hedge_urgency=urgency,
            recommendation=recommendation,
            computed_at=datetime.utcnow().isoformat(),
        )
