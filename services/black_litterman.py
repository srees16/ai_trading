"""
Black-Litterman Portfolio Optimization — T4-2.

Uses Carver forecast signals as "investor views" input to the
Black-Litterman (1992) model to produce posterior expected returns
and a mean-variance optimal portfolio.

Integration points:
  - Carver pipeline Step 4: replaces/blends with Carver weights
  - US pipeline: same integration available
  - Can serve as HRP alternative (switches based on regime)

Research basis:
  - Black & Litterman (1992): "Global Portfolio Optimization"
  - He & Litterman (1999): "The Intuition Behind Black-Litterman"
  - Meucci (2010): "The Black-Litterman Approach: Original Model and Extensions"
  
Parameters:
  - tau: 0.05 (uncertainty in prior, lower = trust market more)
  - risk_aversion: 2.5 (default market risk aversion)
  - forecast_confidence: maps Carver source FDM to view confidence
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BLResult:
    """Result of Black-Litterman optimization."""
    posterior_returns: Dict[str, float] = field(default_factory=dict)
    posterior_weights: Dict[str, float] = field(default_factory=dict)
    prior_returns: Dict[str, float] = field(default_factory=dict)
    view_adjustment: Dict[str, float] = field(default_factory=dict)
    risk_contribution: Dict[str, float] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimization using Carver forecasts as views.

    Parameters
    ----------
    tau : float
        Scaling factor for uncertainty in the prior (0.01-0.10). Default 0.05.
    risk_aversion : float
        Market risk aversion coefficient delta (default 2.5).
    max_weight : float
        Maximum single-asset weight (default 0.20 = 20%).
    min_weight : float
        Minimum single-asset weight (default 0.0 = long-only).
    use_shrinkage : bool
        Apply Ledoit-Wolf shrinkage to covariance (default True).
    """

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        max_weight: float = 0.20,
        min_weight: float = 0.0,
        use_shrinkage: bool = True,
    ):
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.use_shrinkage = use_shrinkage

    def _ledoit_wolf_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """Compute Ledoit-Wolf shrinkage estimate of covariance matrix."""
        n, p = returns.shape
        sample_cov = np.cov(returns, rowvar=False, ddof=1)
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)

        # Compute optimal shrinkage intensity
        delta = sample_cov - target
        sum_sq = np.sum(delta ** 2)

        # Estimate shrinkage intensity
        X_centered = returns - returns.mean(axis=0)
        sum_sq_diag = 0.0
        for i in range(n):
            xi = X_centered[i:i+1, :]
            yi = xi.T @ xi
            sum_sq_diag += np.sum((yi - sample_cov) ** 2)
        sum_sq_diag /= n ** 2

        shrinkage_intensity = max(0.0, min(1.0, sum_sq_diag / max(sum_sq, 1e-10)))
        logger.debug("Ledoit-Wolf shrinkage intensity: %.4f", shrinkage_intensity)

        return shrinkage_intensity * target + (1 - shrinkage_intensity) * sample_cov

    def _equilibrium_returns(
        self,
        cov: np.ndarray,
        market_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute implied equilibrium returns: pi = delta * Sigma * w_mkt."""
        return self.risk_aversion * cov @ market_weights

    def _build_view_matrices(
        self,
        n_assets: int,
        asset_list: List[str],
        forecasts: Dict[str, float],
        forecast_confidences: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build P (pick matrix), Q (view returns), Omega (view uncertainty).

        Each Carver forecast is converted to an absolute view:
          - forecast > 0 → bullish view (expect +X% annualized return)
          - forecast < 0 → bearish view
          - |forecast| scaled: ±20 → ±30% pa expected return
        """
        views = []
        for i, sym in enumerate(asset_list):
            fc = forecasts.get(sym, 0.0)
            if abs(fc) < 2.0:  # Ignore weak forecasts
                continue
            # Convert Carver forecast [-20, +20] to annual return expectation
            # ±20 → ±30% pa; linear scaling
            expected_return = fc / 20.0 * 0.30
            confidence = forecast_confidences.get(sym, 0.5)
            views.append((i, expected_return, confidence))

        n_views = len(views)
        if n_views == 0:
            return np.zeros((0, n_assets)), np.zeros(0), np.zeros((0, 0))

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)

        for k, (idx, ret, conf) in enumerate(views):
            P[k, idx] = 1.0
            Q[k] = ret
            # Higher confidence → lower omega (less uncertainty)
            # conf ∈ [0, 1]: omega = tau * sigma_ii * (1/conf - 1)
            omega_diag[k] = self.tau * max(0.01, 1.0 / max(conf, 0.1) - 1.0)

        Omega = np.diag(omega_diag)
        return P, Q, Omega

    def optimize(
        self,
        symbols: List[str],
        return_matrix: np.ndarray,
        forecasts: Dict[str, float],
        forecast_confidences: Optional[Dict[str, float]] = None,
        market_weights: Optional[Dict[str, float]] = None,
    ) -> BLResult:
        """Run Black-Litterman optimization.

        Parameters
        ----------
        symbols : list
            Asset symbols corresponding to columns of return_matrix.
        return_matrix : ndarray
            (T x N) matrix of asset returns.
        forecasts : dict
            {symbol: carver_forecast} where forecast ∈ [-20, +20].
        forecast_confidences : dict, optional
            {symbol: confidence} where confidence ∈ [0, 1]. Default 0.5.
        market_weights : dict, optional
            {symbol: weight} — market cap weights. Default equal weight.

        Returns
        -------
        BLResult with posterior_weights, posterior_returns, etc.
        """
        result = BLResult()
        n_assets = len(symbols)

        if n_assets < 2 or return_matrix.shape[0] < 30:
            result.log.append(f"Insufficient data: {n_assets} assets, {return_matrix.shape[0]} obs")
            # Fall back to equal weight
            w = 1.0 / max(n_assets, 1)
            result.posterior_weights = {s: w for s in symbols}
            return result

        forecast_confidences = forecast_confidences or {}
        default_conf = {s: 0.5 for s in symbols}
        default_conf.update(forecast_confidences)
        forecast_confidences = default_conf

        # Step 1: Covariance estimation
        if self.use_shrinkage and return_matrix.shape[0] > return_matrix.shape[1]:
            Sigma = self._ledoit_wolf_shrinkage(return_matrix)
        else:
            Sigma = np.cov(return_matrix, rowvar=False, ddof=1)

        # Ensure positive definiteness
        eigenvalues = np.linalg.eigvalsh(Sigma)
        if eigenvalues.min() < 0:
            Sigma += (-eigenvalues.min() + 1e-6) * np.eye(n_assets)

        # Step 2: Market equilibrium weights (prior)
        if market_weights:
            w_mkt = np.array([market_weights.get(s, 1.0 / n_assets) for s in symbols])
        else:
            w_mkt = np.ones(n_assets) / n_assets
        w_mkt = w_mkt / w_mkt.sum()

        # Step 3: Implied equilibrium returns
        pi = self._equilibrium_returns(Sigma, w_mkt)
        for i, s in enumerate(symbols):
            result.prior_returns[s] = round(float(pi[i]), 6)

        # Step 4: Build view matrices from Carver forecasts
        P, Q, Omega = self._build_view_matrices(
            n_assets, symbols, forecasts, forecast_confidences
        )

        if P.shape[0] == 0:
            # No views → return prior (market equilibrium)
            result.posterior_weights = {s: round(float(w_mkt[i]), 6) for i, s in enumerate(symbols)}
            result.posterior_returns = dict(result.prior_returns)
            result.log.append("No confident views — using market equilibrium")
            return result

        # Step 5: Black-Litterman posterior returns
        # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]
        tau_Sigma = self.tau * Sigma
        try:
            tau_Sigma_inv = np.linalg.inv(tau_Sigma)
            Omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            result.log.append("Matrix inversion failed — falling back to prior")
            result.posterior_weights = {s: round(float(w_mkt[i]), 6) for i, s in enumerate(symbols)}
            result.posterior_returns = dict(result.prior_returns)
            return result

        M_inv = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        posterior_mu = M_inv @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)

        for i, s in enumerate(symbols):
            result.posterior_returns[s] = round(float(posterior_mu[i]), 6)
            result.view_adjustment[s] = round(float(posterior_mu[i] - pi[i]), 6)

        # Step 6: Mean-variance optimal weights from posterior
        # w* = (delta * Sigma)^-1 * E[R]
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            w_star = (1.0 / self.risk_aversion) * Sigma_inv @ posterior_mu
        except np.linalg.LinAlgError:
            w_star = w_mkt

        # Normalize and apply constraints
        # Long-only constraint: clip negative weights to min_weight
        w_star = np.clip(w_star, self.min_weight, self.max_weight)
        w_sum = w_star.sum()
        if w_sum > 0:
            w_star = w_star / w_sum
        else:
            w_star = np.ones(n_assets) / n_assets

        for i, s in enumerate(symbols):
            result.posterior_weights[s] = round(float(w_star[i]), 6)

        # Step 7: Risk contribution
        portfolio_var = w_star @ Sigma @ w_star
        if portfolio_var > 0:
            marginal_risk = Sigma @ w_star
            for i, s in enumerate(symbols):
                result.risk_contribution[s] = round(
                    float(w_star[i] * marginal_risk[i] / portfolio_var), 4
                )

        n_views = P.shape[0]
        result.log.append(f"BL optimization: {n_assets} assets, {n_views} views, tau={self.tau}")
        result.log.append(f"Portfolio vol: {math.sqrt(portfolio_var * 252):.1%}")

        for line in result.log:
            logger.info("BL: %s", line)

        return result


def blend_bl_with_carver(
    carver_weights: Dict[str, float],
    bl_weights: Dict[str, float],
    bl_blend: float = 0.30,
) -> Dict[str, float]:
    """Blend standard Carver weights with BL posterior weights.

    Parameters
    ----------
    carver_weights : dict
        {symbol: weight} from standard Carver vol-target sizing.
    bl_weights : dict
        {symbol: weight} from BL optimizer.
    bl_blend : float
        How much BL contributes (0.0-1.0). Default 0.30 = 30% BL, 70% Carver.

    Returns
    -------
    Blended {symbol: weight}.
    """
    all_symbols = set(carver_weights) | set(bl_weights)
    blended = {}
    for s in all_symbols:
        cw = carver_weights.get(s, 0.0)
        bw = bl_weights.get(s, 0.0)
        blended[s] = round((1.0 - bl_blend) * cw + bl_blend * bw, 6)

    # Normalize
    total = sum(abs(v) for v in blended.values())
    if total > 0:
        blended = {s: round(v / total, 6) for s, v in blended.items()}

    return blended
