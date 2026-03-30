"""
Portfolio Correlation Risk — Phase 3.2.

Sizes positions considering inter-stock correlation to prevent
concentrated sector bets (e.g., 6 bank stocks = 1 correlated bet).

Integration:
  - Post-sizing constraint in the Carver pipeline (after position_sizer)
  - Computes true portfolio vol from correlation matrix
  - Scales down positions if portfolio vol exceeds target
  - Reports diversification ratio for monitoring

Research basis:
  - Markowitz (1952): Portfolio Selection
  - Carver Ch. 11: Instrument diversification multiplier
  - Indian market: NIFTY sectors highly correlated in crises (~0.7+)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationRiskResult:
    """Portfolio correlation risk assessment."""
    portfolio_vol_ann: float = 0.0        # true portfolio vol (annualized)
    target_vol_ann: float = 0.25          # target from Carver config
    vol_ratio: float = 1.0               # portfolio_vol / target_vol
    diversification_ratio: float = 1.0    # DR > 1 = diversification helps
    max_sector_weight: float = 0.0
    concentration_warning: bool = False
    scale_factor: float = 1.0            # multiply all positions by this
    correlation_matrix_size: int = 0
    highest_pair_corr: float = 0.0
    highest_pair: tuple = ("", "")

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


class PortfolioCorrelationRisk:
    """Correlation-aware position sizing constraint.

    Parameters
    ----------
    max_portfolio_vol : float
        Maximum annualized portfolio vol (default 0.25 = 25%).
    max_pair_corr : float
        Maximum pairwise correlation before triggering scaling.
    max_sector_weight : float
        Maximum weight any single sector can have.
    lookback_days : int
        Days of return history for correlation estimation.
    """

    def __init__(
        self,
        max_portfolio_vol: float = 0.25,
        max_pair_corr: float = 0.70,
        max_sector_weight: float = 0.35,
        lookback_days: int = 126,
    ):
        self.max_portfolio_vol = max_portfolio_vol
        self.max_pair_corr = max_pair_corr
        self.max_sector_weight = max_sector_weight
        self.lookback = lookback_days

    def assess(
        self,
        position_weights: Dict[str, float],
        returns_data: "pd.DataFrame",
        sector_map: Optional[Dict[str, str]] = None,
    ) -> CorrelationRiskResult:
        """Assess portfolio correlation risk.

        Parameters
        ----------
        position_weights : dict[str, float]
            {symbol: weight} where weights sum to ~1.
        returns_data : pd.DataFrame
            Daily returns DataFrame with symbol columns.
        sector_map : dict | None
            {symbol: sector_name} for sector concentration check.
        """
        import pandas as pd

        symbols = [s for s in position_weights if s in returns_data.columns]
        if len(symbols) < 2:
            return CorrelationRiskResult(
                portfolio_vol_ann=0.0,
                scale_factor=1.0,
                diversification_ratio=1.0,
            )

        # Get recent returns
        recent = returns_data[symbols].tail(self.lookback).dropna(how="all")
        if len(recent) < 20:
            return CorrelationRiskResult(scale_factor=1.0)

        # Compute correlation and covariance matrices
        cov_matrix = recent.cov() * 252  # annualized
        corr_matrix = recent.corr()

        # Weight vector (normalize to sum to 1)
        w = np.array([position_weights.get(s, 0) for s in symbols])
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        # Portfolio volatility: sqrt(w' Σ w)
        cov_np = cov_matrix.values
        port_var = float(w @ cov_np @ w)
        port_vol = np.sqrt(max(port_var, 0))

        # Individual volatilities
        individual_vols = np.sqrt(np.diag(cov_np))

        # Diversification ratio: sum(w_i * σ_i) / σ_portfolio
        weighted_vol_sum = float(np.sum(w * individual_vols))
        div_ratio = weighted_vol_sum / port_vol if port_vol > 0 else 1.0

        # Find highest pairwise correlation
        corr_np = corr_matrix.values
        np.fill_diagonal(corr_np, 0)
        max_idx = np.unravel_index(np.argmax(np.abs(corr_np)), corr_np.shape)
        highest_corr = float(corr_np[max_idx])
        highest_pair = (symbols[max_idx[0]], symbols[max_idx[1]])

        # Sector concentration check
        max_sec_weight = 0.0
        concentration_warning = False
        if sector_map:
            sector_weights: Dict[str, float] = {}
            for s, wt in zip(symbols, w):
                sec = sector_map.get(s, "OTHER")
                sector_weights[sec] = sector_weights.get(sec, 0) + wt
            max_sec_weight = max(sector_weights.values()) if sector_weights else 0
            concentration_warning = max_sec_weight > self.max_sector_weight

        # Compute scale factor if portfolio vol exceeds target
        vol_ratio = port_vol / self.max_portfolio_vol if self.max_portfolio_vol > 0 else 1.0
        scale = 1.0
        if vol_ratio > 1.0:
            scale = 1.0 / vol_ratio
            logger.warning(
                "Portfolio vol %.1f%% exceeds target %.1f%% — scaling by %.2f",
                port_vol * 100, self.max_portfolio_vol * 100, scale,
            )

        return CorrelationRiskResult(
            portfolio_vol_ann=round(port_vol, 4),
            target_vol_ann=self.max_portfolio_vol,
            vol_ratio=round(vol_ratio, 3),
            diversification_ratio=round(div_ratio, 3),
            max_sector_weight=round(max_sec_weight, 3),
            concentration_warning=concentration_warning,
            scale_factor=round(scale, 4),
            correlation_matrix_size=len(symbols),
            highest_pair_corr=round(highest_corr, 3),
            highest_pair=highest_pair,
        )

    def adjust_position_sizes(
        self,
        position_sizes: Dict[str, float],
        assessment: CorrelationRiskResult,
    ) -> Dict[str, float]:
        """Apply correlation-based scaling to position sizes.

        Parameters
        ----------
        position_sizes : dict[str, float]
            {symbol: target_quantity or notional_value}.
        assessment : CorrelationRiskResult
            Result from assess().

        Returns
        -------
        dict[str, float]
            Scaled position sizes.
        """
        scale = assessment.scale_factor
        if scale >= 1.0:
            return position_sizes

        return {sym: qty * scale for sym, qty in position_sizes.items()}
