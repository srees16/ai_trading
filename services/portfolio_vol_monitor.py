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
    emergency_liquidate: bool = False       # G13: trigger full portfolio liquidation
    alerts: List[str] = field(default_factory=list)


ANNUALISATION_FACTOR = 16.0  # sqrt(252) ≈ 16

# Module-level EWMA correlation cache for decay smoothing
_prev_corr_matrix: Optional[np.ndarray] = None
_prev_corr_symbols: Optional[list] = None
_CORR_EWMA_ALPHA = 0.06  # ~16-day half-life: alpha ≈ 1 - exp(-ln2/16)


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

    # EWMA smoothing: blend current C with previous C for decay
    global _prev_corr_matrix, _prev_corr_symbols
    if (_prev_corr_matrix is not None
            and _prev_corr_symbols is not None
            and _prev_corr_symbols == symbols
            and _prev_corr_matrix.shape == C.shape):
        C = _CORR_EWMA_ALPHA * C + (1 - _CORR_EWMA_ALPHA) * _prev_corr_matrix
    _prev_corr_matrix = C.copy()
    _prev_corr_symbols = list(symbols)

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

    # Per-instrument vol cap: no single instrument's vol contribution > 40% of target
    max_instrument_vol_frac = 0.40 * target_annual_vol_pct
    capped_vols = {}
    for sym, vol in instrument_daily_vols.items():
        annual_vol = vol * ANNUALISATION_FACTOR
        if annual_vol > max_instrument_vol_frac and sym in position_values:
            cap_ratio = max_instrument_vol_frac / annual_vol
            capped_vols[sym] = vol * cap_ratio
            alerts_pre = getattr(snap, '_pre_alerts', [])
        else:
            capped_vols[sym] = vol

    # Portfolio daily vol
    daily_vol = compute_portfolio_volatility(
        position_values, capped_vols, correlation_matrix
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

    # BUG-2 FIX: Compute vol_scale and dd_scale independently, then take min().
    # Previously the elif chain could OVERWRITE a lower vol_scale with a higher dd_scale.
    vol_scale = 1.0
    dd_scale = 1.0

    if snap.vol_ratio > 1.5:
        snap.risk_level = RiskLevel.CRITICAL
        vol_scale = target_annual_vol_pct / snap.portfolio_annual_vol_pct if snap.portfolio_annual_vol_pct > 0 else 0.5
        vol_scale = max(0.3, min(vol_scale, 1.0))
        alerts.append(f"CRITICAL: Portfolio vol {snap.portfolio_annual_vol_pct:.1%} is {snap.vol_ratio:.1f}× target — scaling positions to {vol_scale:.0%}")
    elif snap.vol_ratio > 1.2:
        snap.risk_level = RiskLevel.WARNING
        vol_scale = 0.8
        alerts.append(f"WARNING: Portfolio vol {snap.portfolio_annual_vol_pct:.1%} is {snap.vol_ratio:.1f}× target")
    else:
        snap.risk_level = RiskLevel.NORMAL

    if snap.hhi > 0.25:
        alerts.append(f"Concentration risk: HHI={snap.hhi:.2f} (>0.25), largest={snap.largest_position_pct:.0f}%")

    # ── Drawdown tiers — read from Config for hot-reload ──
    try:
        from config import Config
        dd_halt = getattr(Config, "PORTFOLIO_DRAWDOWN_HALT", 0.30) * 100
        dd_critical = getattr(Config, "PORTFOLIO_DRAWDOWN_CRITICAL", 0.25) * 100
        dd_warning = getattr(Config, "PORTFOLIO_DRAWDOWN_WARNING", 0.15) * 100
    except Exception:
        dd_halt, dd_critical, dd_warning = 30.0, 25.0, 15.0

    # ── Smooth quadratic DD scaling curve ──────────────────────
    # Replaces step-function cliff effects with continuous curve:
    #   scale = max(0, 1 - (dd / dd_halt)²)
    # This gives smooth degradation: at dd_warning ~85%, at dd_critical ~30%,
    # at dd_halt = 0%. No sudden cliffs that cause whipsaw.
    if snap.drawdown_pct > dd_halt:
        snap.risk_level = RiskLevel.HALTED
        dd_scale = 0.0
        snap.emergency_liquidate = True
        alerts.append(f"EMERGENCY: Drawdown {snap.drawdown_pct:.1f}% > {dd_halt:.0f}% — LIQUIDATING ALL POSITIONS")
    elif snap.drawdown_pct > dd_critical:
        snap.risk_level = RiskLevel.HALTED
        dd_ratio = snap.drawdown_pct / dd_halt
        dd_scale = max(0.0, 1.0 - dd_ratio * dd_ratio)
        alerts.append(f"HALTED: Drawdown {snap.drawdown_pct:.1f}% > {dd_critical:.0f}% — scale {dd_scale:.0%}")
    elif snap.drawdown_pct > dd_warning:
        snap.risk_level = RiskLevel.CRITICAL
        dd_ratio = snap.drawdown_pct / dd_halt
        dd_scale = max(0.0, 1.0 - dd_ratio * dd_ratio)
        alerts.append(f"CRITICAL: Drawdown {snap.drawdown_pct:.1f}% > {dd_warning:.0f}% — scale {dd_scale:.0%}")
    elif snap.drawdown_pct > (dd_warning * 0.5):
        # Gentle ramp: below warning but non-trivial DD
        dd_ratio = snap.drawdown_pct / dd_halt
        dd_scale = max(0.0, 1.0 - dd_ratio * dd_ratio)
        alerts.append(f"Drawdown {snap.drawdown_pct:.1f}% — smooth scale {dd_scale:.0%}")

    # Always take the MORE conservative (lower) of vol_scale and dd_scale
    snap.scale_factor = round(min(vol_scale, dd_scale), 2)
    snap.alerts = alerts

    for alert in alerts:
        logger.warning("Portfolio risk: %s", alert)

    return snap
