"""
Trade-Level Bootstrap Monte Carlo Simulation — Phase 3.1.

Proper risk estimation for trading systems. NOT forecasting price
direction, but estimating the risk distribution of the actual trading
system based on its historical trade results.

Answers:
  - P(ruin): Probability of 50% drawdown → guides position sizing
  - CVaR (5%): Average loss in worst 5% of scenarios → guides stop levels
  - Optimal Kelly: Data-driven Kelly fraction (not hardcoded 0.5)
  - Confidence interval on CAGR: e.g., "25-55% with 90% confidence"

Research basis:
  - Efron & Tibshirani (1993): Bootstrap Methods
  - Politis & Romano (1994): Stationary Bootstrap (block resampling)
  - Vince (2009): Optimal-f from trade distributions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation output."""
    # CAGR distribution
    median_cagr_pct: float = 0.0
    p5_cagr_pct: float = 0.0       # 5th percentile (pessimistic)
    p25_cagr_pct: float = 0.0
    p75_cagr_pct: float = 0.0
    p95_cagr_pct: float = 0.0      # 95th percentile (optimistic)

    # Drawdown risk
    median_max_dd_pct: float = 0.0
    p95_max_dd_pct: float = 0.0    # 95th percentile worst drawdown
    probability_of_ruin_pct: float = 0.0  # P(equity drops 50%+)

    # Kelly and risk metrics
    optimal_kelly: float = 0.0
    cvar_5pct: float = 0.0

    # Vince geometric mean metrics
    geometric_mean: float = 1.0       # G = TWR^(1/N) at half-Kelly
    vince_optimal_f: float = 0.0      # optimal f from exhaustive search
    fundamental_eq_A: float = 1.0     # arithmetic mean HPR
    fundamental_eq_SD: float = 0.0    # std of returns

    # Portfolio statistics
    n_simulations: int = 0
    n_trades_per_sim: int = 0
    input_trades: int = 0

    # Confidence intervals
    ci_90_cagr: tuple = (0.0, 0.0)  # 5th-95th

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()
                if not k.startswith("_")}


class TradeBootstrapMonteCarlo:
    """Bootstrap Monte Carlo for trading system risk estimation.

    Parameters
    ----------
    n_simulations : int
        Number of simulation paths (default 10,000).
    n_trades_per_sim : int
        Trades per simulation path (default 500 = ~2 years daily).
    ruin_threshold : float
        Equity fraction defining "ruin" (default 0.50 = 50% loss).
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        n_trades_per_sim: int = 500,
        ruin_threshold: float = 0.50,
    ):
        self.n_sims = n_simulations
        self.n_trades = n_trades_per_sim
        self.ruin_threshold = ruin_threshold

    def simulate(self, trade_returns: List[float]) -> MonteCarloResult:
        """Run bootstrap Monte Carlo from actual trade returns.

        Parameters
        ----------
        trade_returns : list[float]
            Historical trade returns as fractions (e.g., 0.02 = +2%).
            Must have at least 20 trades.

        Returns
        -------
        MonteCarloResult
        """
        trades = np.array(trade_returns, dtype=np.float64)
        n_input = len(trades)

        if n_input < 20:
            logger.warning("Monte Carlo needs ≥20 trades, got %d", n_input)
            return MonteCarloResult(input_trades=n_input)

        # Bootstrap: resample with replacement
        sampled = np.random.choice(trades, size=(self.n_sims, self.n_trades), replace=True)

        # Compute equity curves: cumulative product of (1 + r)
        equity_curves = np.cumprod(1.0 + sampled, axis=1)

        # Final equity (relative to starting capital of 1.0)
        final_equity = equity_curves[:, -1]

        # CAGR: annualize assuming ~252 trades per year
        trades_per_year = 252
        years = self.n_trades / trades_per_year
        cagrs = (final_equity ** (1.0 / years) - 1.0) * 100

        # Max drawdown per simulation
        max_dds = self._compute_max_drawdowns(equity_curves)

        # Probability of ruin
        min_equity = equity_curves.min(axis=1)
        ruin_count = (min_equity < (1.0 - self.ruin_threshold)).sum()
        p_ruin = ruin_count / self.n_sims * 100

        # Kelly fraction from trade distribution
        optimal_kelly = self._compute_optimal_kelly(trades)

        # CVaR from trade returns
        cvar_5 = self._compute_cvar(trades, alpha=0.05)

        # Vince geometric mean metrics
        gm, opt_f, fund_A, fund_SD = self._compute_vince_metrics(trades)

        return MonteCarloResult(
            median_cagr_pct=round(float(np.median(cagrs)), 2),
            p5_cagr_pct=round(float(np.percentile(cagrs, 5)), 2),
            p25_cagr_pct=round(float(np.percentile(cagrs, 25)), 2),
            p75_cagr_pct=round(float(np.percentile(cagrs, 75)), 2),
            p95_cagr_pct=round(float(np.percentile(cagrs, 95)), 2),
            median_max_dd_pct=round(float(np.median(max_dds)) * 100, 2),
            p95_max_dd_pct=round(float(np.percentile(max_dds, 95)) * 100, 2),
            probability_of_ruin_pct=round(p_ruin, 2),
            optimal_kelly=round(optimal_kelly, 4),
            cvar_5pct=round(cvar_5 * 100, 4),
            geometric_mean=round(gm, 6),
            vince_optimal_f=round(opt_f, 4),
            fundamental_eq_A=round(fund_A, 6),
            fundamental_eq_SD=round(fund_SD, 6),
            n_simulations=self.n_sims,
            n_trades_per_sim=self.n_trades,
            input_trades=n_input,
            ci_90_cagr=(
                round(float(np.percentile(cagrs, 5)), 2),
                round(float(np.percentile(cagrs, 95)), 2),
            ),
        )

    def block_bootstrap(
        self,
        trade_returns: List[float],
        block_size: int = 10,
    ) -> MonteCarloResult:
        """Block bootstrap preserving serial correlation (streaks).

        Resamples blocks of consecutive trades instead of individual
        trades. Captures winning/losing streak effects.
        """
        trades = np.array(trade_returns, dtype=np.float64)
        n = len(trades)
        if n < max(20, block_size * 2):
            return self.simulate(trade_returns)

        n_blocks = (self.n_trades + block_size - 1) // block_size
        equity_curves = np.ones((self.n_sims, self.n_trades))

        for sim in range(self.n_sims):
            path = []
            for _ in range(n_blocks):
                start = np.random.randint(0, max(1, n - block_size))
                block = trades[start:start + block_size].tolist()
                path.extend(block)
            path = path[:self.n_trades]
            equity_curves[sim] = np.cumprod(1.0 + np.array(path))

        final_equity = equity_curves[:, -1]
        years = self.n_trades / 252
        cagrs = (final_equity ** (1.0 / years) - 1.0) * 100
        max_dds = self._compute_max_drawdowns(equity_curves)
        min_eq = equity_curves.min(axis=1)
        p_ruin = (min_eq < (1.0 - self.ruin_threshold)).sum() / self.n_sims * 100

        gm, opt_f, fund_A, fund_SD = self._compute_vince_metrics(trades)

        return MonteCarloResult(
            median_cagr_pct=round(float(np.median(cagrs)), 2),
            p5_cagr_pct=round(float(np.percentile(cagrs, 5)), 2),
            p25_cagr_pct=round(float(np.percentile(cagrs, 25)), 2),
            p75_cagr_pct=round(float(np.percentile(cagrs, 75)), 2),
            p95_cagr_pct=round(float(np.percentile(cagrs, 95)), 2),
            median_max_dd_pct=round(float(np.median(max_dds)) * 100, 2),
            p95_max_dd_pct=round(float(np.percentile(max_dds, 95)) * 100, 2),
            probability_of_ruin_pct=round(p_ruin, 2),
            optimal_kelly=round(self._compute_optimal_kelly(trades), 4),
            cvar_5pct=round(self._compute_cvar(trades, 0.05) * 100, 4),
            geometric_mean=round(gm, 6),
            vince_optimal_f=round(opt_f, 4),
            fundamental_eq_A=round(fund_A, 6),
            fundamental_eq_SD=round(fund_SD, 6),
            n_simulations=self.n_sims,
            n_trades_per_sim=self.n_trades,
            input_trades=len(trades),
            ci_90_cagr=(
                round(float(np.percentile(cagrs, 5)), 2),
                round(float(np.percentile(cagrs, 95)), 2),
            ),
        )

    @staticmethod
    def _compute_max_drawdowns(equity_curves: np.ndarray) -> np.ndarray:
        """Compute max drawdown for each simulation path."""
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (equity_curves - running_max) / running_max
        return -drawdowns.min(axis=1)  # positive values

    @staticmethod
    def _compute_optimal_kelly(trades: np.ndarray) -> float:
        """Compute optimal Kelly fraction from trade distribution.

        Kelly = p - q/R where p = win_rate, q = 1-p, R = avg_win/avg_loss
        Returns half-Kelly for conservative sizing.
        """
        wins = trades[trades > 0]
        losses = trades[trades < 0]

        if len(wins) < 5 or len(losses) < 5:
            return 0.02  # conservative default

        p = len(wins) / len(trades)
        q = 1.0 - p
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean(np.abs(losses)))
        R = avg_win / avg_loss if avg_loss > 0 else 2.0

        kelly = p - q / R
        kelly = max(kelly, 0.0)

        # Return half-Kelly (conservative)
        return kelly * 0.5

    @staticmethod
    def _compute_cvar(trades: np.ndarray, alpha: float = 0.05) -> float:
        """Compute Conditional VaR (Expected Shortfall) at alpha level."""
        sorted_trades = np.sort(trades)
        cutoff = int(len(sorted_trades) * alpha)
        if cutoff < 1:
            cutoff = 1
        return float(np.mean(sorted_trades[:cutoff]))

    @staticmethod
    def _compute_vince_metrics(trades: np.ndarray):
        """Compute Vince geometric mean, optimal f, and Fundamental Equation.

        Returns (geometric_mean, optimal_f, A, SD).
        """
        import math

        if len(trades) < 10:
            A = 1.0 + float(np.mean(trades)) if len(trades) > 0 else 1.0
            SD = float(np.std(trades, ddof=0)) if len(trades) > 0 else 0.0
            return 1.0, 0.0, A, SD

        biggest_loss = abs(float(np.min(trades)))
        best_f, best_gm = 0.01, 0.0

        if biggest_loss > 1e-9:
            for step in range(1, 501):
                f = step / 500
                hprs = 1.0 + f * (trades / biggest_loss)
                if np.any(hprs <= 0):
                    break
                twr = float(np.prod(hprs))
                gm = twr ** (1.0 / len(trades))
                if gm > best_gm:
                    best_gm = gm
                    best_f = f

        A = 1.0 + float(np.mean(trades))
        SD = float(np.std(trades, ddof=0))
        return best_gm, best_f, A, SD
