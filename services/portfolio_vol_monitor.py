"""
Portfolio Volatility Monitor — Real-time portfolio risk dashboard.

Tracks:
  1. Portfolio-level daily volatility (from position weights × instrument vols × correlations)
  2. Deviation from target volatility  
  3. Concentration risk (HHI)
  4. Drawdown from peak equity

Alarm levels:
  - NORMAL:  portfolio vol within ±20% of target
  - WARNING: portfolio vol 20-50% above target
  - CRITICAL: portfolio vol >50% above target → scale positions

Implements Carver Chapter 11 risk monitoring without portfolio optimisation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    HALTED = "HALTED"


@dataclass
class PortfolioRiskSnapshot:
    """Point-in-time portfolio risk assessment."""
    timestamp: str = ""

    # Volatility
    portfolio_daily_vol: float = 0.0       # portfolio daily vol (₹)
    portfolio_annual_vol_pct: float = 0.0  # annualised % vol
    target_annual_vol_pct: float = 0.20    # target
    vol_ratio: float = 0.0                 # actual / target

    # Concentration
    hhi: float = 0.0                       # Herfindahl-Hirschman Index (0-1)
    largest_position_pct: float = 0.0      # largest single position weight

    # Drawdown
    peak_equity: float = 0.0
    current_equity: float = 0.0
    drawdown_pct: float = 0.0

    # Status
    risk_level: RiskLevel = RiskLevel.NORMAL
    scale_factor: float = 1.0              # position scale-down factor
    alerts: List[str] = field(default_factory=list)


ANNUALISATION_FACTOR = 16.0  # sqrt(252) ≈ 16


def compute_portfolio_volatility(
    position_values: Dict[str, float],
    instrument_daily_vols: Dict[str, float],
    correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    avg_correlation: float = 0.40,
) -> float:
    """Compute portfolio daily volatility in ₹.

    portfolio_vol = sqrt( Σᵢ Σⱼ  wᵢ·σᵢ · wⱼ·σⱼ · ρᵢⱼ )

    where wᵢ = position_value / total_value, σᵢ = daily vol in ₹.

    Parameters
    ----------
    position_values : dict[str, float]
        {symbol: notional_value_in_rupees}.
    instrument_daily_vols : dict[str, float]
        {symbol: daily_price_volatility_fraction}.
    correlation_matrix : dict | None
        {(sym_a, sym_b): correlation}.
    avg_correlation : float
        Default off-diagonal correlation.

    Returns
    -------
    float
        Portfolio daily volatility in ₹.
    """
    symbols = sorted(position_values.keys())
    n = len(symbols)
    if n == 0:
        return 0.0

    # Position dollar-vols
    pos_vols = np.array([
        position_values.get(s, 0) * instrument_daily_vols.get(s, 0.02)
        for s in symbols
    ])

    # Correlation matrix
    C = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                if correlation_matrix:
                    key = (symbols[i], symbols[j])
                    rev = (symbols[j], symbols[i])
                    C[i, j] = correlation_matrix.get(key, correlation_matrix.get(rev, avg_correlation))
                else:
                    C[i, j] = avg_correlation

    port_var = float(pos_vols @ C @ pos_vols)
    return math.sqrt(max(0, port_var))


def assess_portfolio_risk(
    position_values: Dict[str, float],
    instrument_daily_vols: Dict[str, float],
    target_annual_vol_pct: float = 0.20,
    total_capital: float = 500_000.0,
    peak_equity: Optional[float] = None,
    correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None,
) -> PortfolioRiskSnapshot:
    """Full portfolio risk assessment.

    Parameters
    ----------
    position_values : dict[str, float]
        {symbol: current_notional_value}.
    instrument_daily_vols : dict[str, float]
        {symbol: daily_price_vol_fraction}.
    target_annual_vol_pct : float
        Target annual volatility (fraction).
    total_capital : float
        Current total capital.
    peak_equity : float | None
        Historical peak equity for drawdown calc.
    correlation_matrix : dict | None
        Pairwise correlations.

    Returns
    -------
    PortfolioRiskSnapshot
    """
    from datetime import datetime, timezone

    snap = PortfolioRiskSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        target_annual_vol_pct=target_annual_vol_pct,
    )

    total_value = sum(position_values.values())
    if total_value <= 0 or total_capital <= 0:
        return snap

    # Portfolio daily vol
    daily_vol = compute_portfolio_volatility(
        position_values, instrument_daily_vols, correlation_matrix
    )
    snap.portfolio_daily_vol = round(daily_vol, 2)
    snap.portfolio_annual_vol_pct = round(
        (daily_vol / total_capital) * ANNUALISATION_FACTOR, 4
    )

    # Vol ratio
    if target_annual_vol_pct > 0:
        snap.vol_ratio = round(snap.portfolio_annual_vol_pct / target_annual_vol_pct, 3)

    # Concentration (HHI)
    weights = [v / total_value for v in position_values.values()]
    snap.hhi = round(sum(w * w for w in weights), 4)
    snap.largest_position_pct = round(max(weights) * 100, 1) if weights else 0.0

    # Drawdown
    current_equity = total_capital
    peak = peak_equity or current_equity
    peak = max(peak, current_equity)
    snap.peak_equity = peak
    snap.current_equity = current_equity
    if peak > 0:
        snap.drawdown_pct = round((peak - current_equity) / peak * 100, 2)

    # Risk level classification
    alerts = []
    scale = 1.0

    if snap.vol_ratio > 1.5:
        snap.risk_level = RiskLevel.CRITICAL
        scale = target_annual_vol_pct / snap.portfolio_annual_vol_pct if snap.portfolio_annual_vol_pct > 0 else 0.5
        scale = max(0.3, min(scale, 1.0))
        alerts.append(f"CRITICAL: Portfolio vol {snap.portfolio_annual_vol_pct:.1%} is {snap.vol_ratio:.1f}× target — scaling positions to {scale:.0%}")
    elif snap.vol_ratio > 1.2:
        snap.risk_level = RiskLevel.WARNING
        scale = 0.8
        alerts.append(f"WARNING: Portfolio vol {snap.portfolio_annual_vol_pct:.1%} is {snap.vol_ratio:.1f}× target")
    else:
        snap.risk_level = RiskLevel.NORMAL

    if snap.hhi > 0.25:
        alerts.append(f"Concentration risk: HHI={snap.hhi:.2f} (>0.25), largest={snap.largest_position_pct:.0f}%")

    if snap.drawdown_pct > 15.0:
        snap.risk_level = RiskLevel.HALTED
        scale = 0.0
        alerts.append(f"HALTED: Drawdown {snap.drawdown_pct:.1f}% exceeds 15% limit")
    elif snap.drawdown_pct > 10.0:
        scale = min(scale, 0.5)
        alerts.append(f"Drawdown {snap.drawdown_pct:.1f}% — reducing position sizes to 50%")

    snap.scale_factor = round(scale, 2)
    snap.alerts = alerts

    for alert in alerts:
        logger.warning("Portfolio risk: %s", alert)

    return snap
