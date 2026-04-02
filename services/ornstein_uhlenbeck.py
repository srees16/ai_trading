"""
Ornstein-Uhlenbeck Mean Reversion Model — T4-7.

Replaces z-score heuristic in pairs/mean reversion strategies with
a proper Ornstein-Uhlenbeck (O-U) stochastic process model.

The O-U process: dX_t = theta * (mu - X_t) * dt + sigma * dW_t

Key parameters estimated from data:
  - theta: mean reversion speed (higher = faster reversion)
  - mu: long-run mean
  - sigma: volatility of the process

Entry/exit rules derived from O-U:
  - Enter when spread > mu + sigma/sqrt(2*theta) (expected profit > cost)
  - Exit at mu (or profit target)
  - Stop at mu + 2*sigma/sqrt(2*theta)

Half-life = ln(2) / theta — tells how long mean reversion takes

Research basis:
  - Uhlenbeck & Ornstein (1930): Original process
  - Elliott, Van der Hoek & Malcolm (2005): Pairs trading with O-U
  - Leung & Li (2016): Optimal Mean Reversion Trading
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
class OUParams:
    """Estimated Ornstein-Uhlenbeck parameters."""
    theta: float = 0.0       # Mean reversion speed
    mu: float = 0.0          # Long-run mean
    sigma: float = 0.0       # Process volatility
    half_life: float = 0.0   # Days for 50% reversion
    r_squared: float = 0.0   # Fit quality
    n_obs: int = 0


@dataclass
class OUSignal:
    """Trading signal from O-U model."""
    symbol: str
    spread: float = 0.0
    z_score: float = 0.0
    ou_score: float = 0.0     # Normalized O-U deviation
    half_life: float = 0.0
    signal: str = "HOLD"      # "ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"
    forecast: float = 0.0     # [-20, +20] Carver-format forecast
    entry_threshold: float = 0.0
    stop_threshold: float = 0.0
    params: Optional[OUParams] = None


@dataclass
class OUResult:
    """Batch result of O-U analysis."""
    signals: List[OUSignal] = field(default_factory=list)
    valid_pairs: int = 0
    entry_signals: int = 0
    log: List[str] = field(default_factory=list)


class OrnsteinUhlenbeckModel:
    """Ornstein-Uhlenbeck mean reversion model for pairs/spread trading.

    Parameters
    ----------
    min_half_life : float
        Minimum half-life in days to consider (filter out noise). Default 5.
    max_half_life : float
        Maximum half-life in days (filter out slow mean reverters). Default 60.
    estimation_window : int
        Number of days for parameter estimation (default 252).
    entry_sigma_mult : float
        Entry threshold in sigma units (default 1.0).
    exit_sigma_mult : float
        Exit threshold in sigma units (default 0.0 = at mu).
    stop_sigma_mult : float
        Stop-loss threshold in sigma units (default 2.5).
    min_r_squared : float
        Minimum R² for O-U fit to be valid (default 0.05).
    """

    def __init__(
        self,
        min_half_life: float = 5.0,
        max_half_life: float = 60.0,
        estimation_window: int = 252,
        entry_sigma_mult: float = 1.0,
        exit_sigma_mult: float = 0.0,
        stop_sigma_mult: float = 2.5,
        min_r_squared: float = 0.05,
    ):
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.estimation_window = estimation_window
        self.entry_sigma_mult = entry_sigma_mult
        self.exit_sigma_mult = exit_sigma_mult
        self.stop_sigma_mult = stop_sigma_mult
        self.min_r_squared = min_r_squared

    def estimate_params(self, spread: pd.Series) -> Optional[OUParams]:
        """Estimate O-U parameters from spread time series using OLS.

        Uses the discretized O-U model:
          X_{t+1} - X_t = theta*(mu - X_t)*dt + sigma*sqrt(dt)*epsilon

        Estimate via regression:
          dX = a + b*X_t + error
          where theta = -b, mu = -a/b, sigma from residuals
        """
        spread = spread.dropna()
        if len(spread) < 50:
            return None

        # Use last estimation_window observations
        s = spread.values[-self.estimation_window:]
        n = len(s)
        if n < 50:
            return None

        # OLS regression: dX_t = a + b * X_t + epsilon
        dx = np.diff(s)
        x = s[:-1]

        if np.std(x) < 1e-10:
            return None

        # OLS: y = a + b*x
        x_mean = np.mean(x)
        dx_mean = np.mean(dx)
        cov_xdx = np.mean(x * dx) - x_mean * dx_mean
        var_x = np.var(x)

        if var_x < 1e-10:
            return None

        b = cov_xdx / var_x
        a = dx_mean - b * x_mean

        # O-U parameters (daily dt=1)
        theta = -b
        if theta <= 0:
            # No mean reversion detected
            return None

        mu = -a / b if abs(b) > 1e-10 else np.mean(s)

        # Sigma from residuals
        residuals = dx - (a + b * x)
        sigma = float(np.std(residuals)) * math.sqrt(252)  # Annualize

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((dx - dx_mean) ** 2)
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-10) if ss_tot > 0 else 0.0

        # Half-life
        half_life = math.log(2) / theta if theta > 0 else float("inf")

        return OUParams(
            theta=round(float(theta), 6),
            mu=round(float(mu), 4),
            sigma=round(float(sigma), 6),
            half_life=round(float(half_life), 1),
            r_squared=round(float(max(0, r_sq)), 4),
            n_obs=n,
        )

    def generate_signal(
        self, spread: pd.Series, params: OUParams
    ) -> OUSignal:
        """Generate trading signal from O-U model.

        Entry: spread deviates > entry_sigma from mu
        Exit: spread returns to mu ± exit_sigma
        Stop: spread exceeds stop_sigma from mu
        """
        current = float(spread.iloc[-1])

        # O-U equilibrium standard deviation: sigma_eq = sigma / sqrt(2*theta)
        if params.theta <= 0:
            sigma_eq = params.sigma
        else:
            sigma_eq = params.sigma / math.sqrt(2 * params.theta)

        if sigma_eq <= 0:
            sigma_eq = max(float(spread.std()), 1e-6)

        # Deviation from equilibrium
        deviation = current - params.mu
        ou_score = deviation / sigma_eq if sigma_eq > 0 else 0.0

        # Thresholds
        entry_upper = params.mu + self.entry_sigma_mult * sigma_eq
        entry_lower = params.mu - self.entry_sigma_mult * sigma_eq
        exit_upper = params.mu + self.exit_sigma_mult * sigma_eq
        exit_lower = params.mu - self.exit_sigma_mult * sigma_eq
        stop_upper = params.mu + self.stop_sigma_mult * sigma_eq
        stop_lower = params.mu - self.stop_sigma_mult * sigma_eq

        signal = "HOLD"
        forecast = 0.0

        if current > entry_upper:
            # Spread is high → expect reversion down → short spread
            signal = "ENTER_SHORT"
            # Scale forecast by how far past entry
            intensity = min(abs(ou_score) / self.entry_sigma_mult, 3.0) / 3.0
            forecast = -20.0 * intensity
        elif current < entry_lower:
            # Spread is low → expect reversion up → long spread
            signal = "ENTER_LONG"
            intensity = min(abs(ou_score) / self.entry_sigma_mult, 3.0) / 3.0
            forecast = 20.0 * intensity
        elif abs(deviation) < self.exit_sigma_mult * sigma_eq:
            signal = "EXIT"

        # Override: if at stop, signal exit
        if current > stop_upper or current < stop_lower:
            signal = "EXIT"
            forecast = 0.0

        return OUSignal(
            symbol="",
            spread=round(current, 4),
            z_score=round(ou_score, 3),
            ou_score=round(ou_score, 3),
            half_life=params.half_life,
            signal=signal,
            forecast=round(forecast, 2),
            entry_threshold=round(self.entry_sigma_mult * sigma_eq, 4),
            stop_threshold=round(self.stop_sigma_mult * sigma_eq, 4),
            params=params,
        )

    def analyze_batch(
        self,
        spreads: Dict[str, pd.Series],
    ) -> OUResult:
        """Analyze multiple spread series for O-U mean reversion signals.

        Parameters
        ----------
        spreads : dict
            {pair_name: pd.Series of spread values}

        Returns
        -------
        OUResult with signals sorted by absolute ou_score (strongest first).
        """
        result = OUResult()

        for name, spread in spreads.items():
            if spread is None or len(spread) < 60:
                continue

            params = self.estimate_params(spread)
            if params is None:
                continue

            # Quality filters
            if params.r_squared < self.min_r_squared:
                continue
            if params.half_life < self.min_half_life:
                continue
            if params.half_life > self.max_half_life:
                continue

            result.valid_pairs += 1

            sig = self.generate_signal(spread, params)
            sig.symbol = name

            if sig.signal in ("ENTER_LONG", "ENTER_SHORT"):
                result.entry_signals += 1

            result.signals.append(sig)

        # Sort by absolute O-U score (strongest signals first)
        result.signals.sort(key=lambda s: abs(s.ou_score), reverse=True)

        result.log.append(
            f"O-U analysis: {len(spreads)} pairs → {result.valid_pairs} valid "
            f"(HL {self.min_half_life}-{self.max_half_life}d) → "
            f"{result.entry_signals} entry signals"
        )

        for line in result.log:
            logger.info("O-U MR: %s", line)

        return result
