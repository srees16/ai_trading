"""
Fama-French Factor Decomposition — Advanced Analytics.

Implements 3-factor and 5-factor Fama-French models for:
1. Portfolio attribution — what factors drive returns
2. Alpha estimation — isolate true alpha from factor exposures
3. Risk decomposition — factor vs idiosyncratic risk
4. Factor timing — regime-conditional factor tilts

Fama-French 3-Factor: R_i - R_f = alpha + beta_MKT*(R_m-R_f) + beta_SMB*SMB + beta_HML*HML + eps
Fama-French 5-Factor: + beta_RMW*RMW + beta_CMA*CMA

For Indian markets:
  - MKT: Nifty 50 excess return
  - SMB: Small-cap vs large-cap returns (NSE SmallCap 250 vs Nifty 50)
  - HML: Value vs growth (Nifty Value 20 vs Nifty Growth Sectors 15)
  - RMW: Robust vs weak profitability (from CMIE data or proxy)
  - CMA: Conservative vs aggressive investment (low vs high capex)

For US markets:
  - Direct from Ken French's data library

Research:
  - Fama & French (1993): "Common Risk Factors in Returns on Stocks and Bonds"
  - Fama & French (2015): "A Five-Factor Asset Pricing Model"
  - Agarwalla, Jacob & Varma (2013): "Four-Factor Model for Indian Equities"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Factor loadings for a single asset or portfolio."""
    symbol: str = ""
    alpha: float = 0.0           # Annualized alpha (intercept × 252)
    alpha_tstat: float = 0.0
    beta_mkt: float = 0.0       # Market beta
    beta_smb: float = 0.0       # Size factor exposure
    beta_hml: float = 0.0       # Value factor exposure
    beta_rmw: float = 0.0       # Profitability factor exposure (5-factor)
    beta_cma: float = 0.0       # Investment factor exposure (5-factor)
    r_squared: float = 0.0      # Model R²
    adj_r_squared: float = 0.0
    residual_vol: float = 0.0   # Idiosyncratic volatility (annualized)
    n_obs: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "alpha_annual": round(self.alpha, 4),
            "alpha_tstat": round(self.alpha_tstat, 2),
            "beta_mkt": round(self.beta_mkt, 3),
            "beta_smb": round(self.beta_smb, 3),
            "beta_hml": round(self.beta_hml, 3),
            "beta_rmw": round(self.beta_rmw, 3),
            "beta_cma": round(self.beta_cma, 3),
            "r_squared": round(self.r_squared, 4),
            "residual_vol": round(self.residual_vol, 4),
        }


@dataclass
class RiskDecomposition:
    """Factor vs idiosyncratic risk breakdown."""
    factor_risk_pct: float = 0.0    # % of variance from factors
    idio_risk_pct: float = 0.0      # % of variance idiosyncratic
    top_factor: str = ""            # Dominant factor
    top_factor_contribution: float = 0.0


@dataclass
class FactorAnalysisResult:
    """Complete Fama-French analysis result."""
    exposures: Dict[str, FactorExposure] = field(default_factory=dict)
    portfolio_exposure: Optional[FactorExposure] = None
    risk_decomposition: Optional[RiskDecomposition] = None
    factor_returns: Dict[str, float] = field(default_factory=dict)  # Recent factor returns
    log: List[str] = field(default_factory=list)


class FamaFrenchDecomposition:
    """Fama-French factor analysis for Indian and US equities.

    Parameters
    ----------
    n_factors : int
        3 or 5 factor model (default 3).
    market : str
        "IND" or "US" (default "IND").
    min_observations : int
        Minimum days of data required (default 120).
    risk_free_rate : float
        Annual risk-free rate (default 0.065 for India, 0.045 for US).
    """

    def __init__(
        self,
        n_factors: int = 3,
        market: str = "IND",
        min_observations: int = 120,
        risk_free_rate: Optional[float] = None,
    ):
        self.n_factors = n_factors
        self.market = market.upper()
        self.min_observations = min_observations
        self.risk_free_rate = risk_free_rate or (0.065 if self.market == "IND" else 0.045)

    def _construct_factors_ind(
        self,
        nifty50_returns: pd.Series,
        smallcap_returns: pd.Series,
        value_returns: pd.Series,
        growth_returns: pd.Series,
        profitability_returns: Optional[pd.Series] = None,
        investment_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Construct Indian Fama-French factor returns.

        Uses NSE index proxies:
          - MKT: Nifty 50 - Rf
          - SMB: SmallCap 250 - Nifty 50
          - HML: Value 20 - Growth Sectors 15
        """
        rf_daily = self.risk_free_rate / 252.0

        factors = pd.DataFrame(index=nifty50_returns.index)
        factors["MKT"] = nifty50_returns - rf_daily
        factors["SMB"] = smallcap_returns - nifty50_returns
        factors["HML"] = value_returns - growth_returns

        if self.n_factors == 5 and profitability_returns is not None:
            factors["RMW"] = profitability_returns
            if investment_returns is not None:
                factors["CMA"] = investment_returns
            else:
                factors["CMA"] = 0.0
        elif self.n_factors == 5:
            factors["RMW"] = 0.0
            factors["CMA"] = 0.0

        return factors.dropna()

    def _run_regression(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """OLS regression: y = X*beta + eps.

        Returns (coefficients, R², adj_R², t_stats).
        """
        n, k = X.shape

        # Add constant (alpha)
        X_c = np.column_stack([np.ones(n), X])

        try:
            # OLS: beta = (X'X)^-1 X'y
            XtX_inv = np.linalg.inv(X_c.T @ X_c)
            beta = XtX_inv @ (X_c.T @ y)
        except np.linalg.LinAlgError:
            return np.zeros(k + 1), 0.0, 0.0, np.zeros(k + 1)

        # Residuals
        residuals = y - X_c @ beta
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        r_sq = 1.0 - ss_res / max(ss_tot, 1e-10) if ss_tot > 0 else 0.0
        adj_r_sq = 1.0 - (1.0 - r_sq) * (n - 1) / max(n - k - 2, 1)

        # Standard errors and t-stats
        sigma2 = ss_res / max(n - k - 1, 1)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        t_stats = beta / np.maximum(se, 1e-10)

        return beta, r_sq, adj_r_sq, t_stats

    def analyze_asset(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
        symbol: str = "",
    ) -> Optional[FactorExposure]:
        """Run FF regression on a single asset.

        Parameters
        ----------
        asset_returns : pd.Series
            Daily returns for the asset.
        factor_returns : pd.DataFrame
            DataFrame with columns MKT, SMB, HML (and optionally RMW, CMA).
        symbol : str
            Asset identifier.
        """
        # Align dates
        common = asset_returns.index.intersection(factor_returns.index)
        if len(common) < self.min_observations:
            return None

        y = asset_returns.loc[common].values
        rf_daily = self.risk_free_rate / 252.0
        y_excess = y - rf_daily

        factor_cols = ["MKT", "SMB", "HML"]
        if self.n_factors == 5:
            factor_cols.extend(["RMW", "CMA"])

        X = factor_returns.loc[common, factor_cols].values

        beta, r_sq, adj_r_sq, t_stats = self._run_regression(y_excess, X)

        alpha_daily = beta[0]
        alpha_annual = alpha_daily * 252

        residuals = y_excess - np.column_stack([np.ones(len(y_excess)), X]) @ beta
        residual_vol = float(np.std(residuals)) * math.sqrt(252)

        exposure = FactorExposure(
            symbol=symbol,
            alpha=round(alpha_annual, 6),
            alpha_tstat=round(float(t_stats[0]), 3),
            beta_mkt=round(float(beta[1]), 4),
            beta_smb=round(float(beta[2]), 4),
            beta_hml=round(float(beta[3]), 4),
            beta_rmw=round(float(beta[4]), 4) if self.n_factors == 5 else 0.0,
            beta_cma=round(float(beta[5]), 4) if self.n_factors == 5 and len(beta) > 5 else 0.0,
            r_squared=round(r_sq, 4),
            adj_r_squared=round(adj_r_sq, 4),
            residual_vol=round(residual_vol, 4),
            n_obs=len(common),
        )

        return exposure

    def analyze_portfolio(
        self,
        asset_returns: Dict[str, pd.Series],
        portfolio_weights: Dict[str, float],
        factor_returns: pd.DataFrame,
    ) -> FactorAnalysisResult:
        """Analyze full portfolio: per-asset + aggregated.

        Parameters
        ----------
        asset_returns : dict
            {symbol: pd.Series} of daily returns.
        portfolio_weights : dict
            {symbol: weight} — current portfolio weights.
        factor_returns : pd.DataFrame
            Factor return series (MKT, SMB, HML, etc.).

        Returns
        -------
        FactorAnalysisResult with per-asset and portfolio exposures.
        """
        result = FactorAnalysisResult()

        # Per-asset analysis
        for symbol, ret_series in asset_returns.items():
            exp = self.analyze_asset(ret_series, factor_returns, symbol)
            if exp is not None:
                result.exposures[symbol] = exp

        # Aggregate portfolio exposure (weighted average)
        if result.exposures:
            total_weight = sum(
                portfolio_weights.get(s, 0.0)
                for s in result.exposures
            )
            if total_weight > 0:
                port = FactorExposure(symbol="PORTFOLIO")
                for sym, exp in result.exposures.items():
                    w = portfolio_weights.get(sym, 0.0) / total_weight
                    port.alpha += exp.alpha * w
                    port.beta_mkt += exp.beta_mkt * w
                    port.beta_smb += exp.beta_smb * w
                    port.beta_hml += exp.beta_hml * w
                    port.beta_rmw += exp.beta_rmw * w
                    port.beta_cma += exp.beta_cma * w

                # Round
                for attr in ["alpha", "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"]:
                    setattr(port, attr, round(getattr(port, attr), 4))

                result.portfolio_exposure = port

                # Risk decomposition
                factors = [
                    ("MKT", port.beta_mkt),
                    ("SMB", port.beta_smb),
                    ("HML", port.beta_hml),
                ]
                if self.n_factors == 5:
                    factors.extend([("RMW", port.beta_rmw), ("CMA", port.beta_cma)])

                total_factor_var = sum(b ** 2 for _, b in factors)
                avg_r_sq = np.mean([e.r_squared for e in result.exposures.values()])

                if factors:
                    top_factor_name, top_factor_beta = max(factors, key=lambda x: abs(x[1]))
                else:
                    top_factor_name, top_factor_beta = "MKT", 0.0

                result.risk_decomposition = RiskDecomposition(
                    factor_risk_pct=round(avg_r_sq * 100, 1),
                    idio_risk_pct=round((1 - avg_r_sq) * 100, 1),
                    top_factor=top_factor_name,
                    top_factor_contribution=round(abs(top_factor_beta), 4),
                )

        # Recent factor returns (last 21 days annualized)
        if len(factor_returns) >= 21:
            recent = factor_returns.iloc[-21:]
            for col in factor_returns.columns:
                result.factor_returns[col] = round(float(recent[col].mean()) * 252, 4)

        n_sig_alpha = sum(
            1 for e in result.exposures.values()
            if abs(e.alpha_tstat) > 1.96
        )
        result.log.append(
            f"FF{self.n_factors} ({self.market}): {len(result.exposures)} assets analyzed, "
            f"{n_sig_alpha} with significant alpha (|t|>1.96)"
        )
        if result.portfolio_exposure:
            result.log.append(
                f"Portfolio: α={result.portfolio_exposure.alpha:.2%}, "
                f"β_mkt={result.portfolio_exposure.beta_mkt:.2f}, "
                f"β_smb={result.portfolio_exposure.beta_smb:.2f}, "
                f"β_hml={result.portfolio_exposure.beta_hml:.2f}"
            )

        for line in result.log:
            logger.info("FF: %s", line)

        return result
