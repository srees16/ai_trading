"""
Risk-Adjusted Return Metrics — Phase 0 Gap Fix.

Provides all missing risk metrics identified in the audit:
  - Sortino Ratio (downside-only volatility)
  - Calmar Ratio (return / max drawdown)
  - Omega Ratio (probability-weighted gain/loss)
  - CVaR / Expected Shortfall (tail risk)
  - Ulcer Index (drawdown severity + duration)
  - Recovery Factor (net profit / max drawdown)
  - Gain-to-Pain Ratio (sum gains / sum losses)

Usage:
    from services.risk_metrics import RiskMetrics
    metrics = RiskMetrics.compute_all(daily_returns)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
ANNUALISATION = np.sqrt(TRADING_DAYS)
RISK_FREE_RATE_IND = 0.07  # India 10-year G-Sec
RISK_FREE_RATE_US = 0.04   # US 10-year Treasury


@dataclass
class RiskMetricsResult:
    """Complete set of risk-adjusted return metrics."""
    n_days: int = 0
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0

    # Volatility
    annual_volatility_pct: float = 0.0
    downside_volatility_pct: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    gain_to_pain_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0

    # Tail risk
    var_95_daily: float = 0.0      # Value at Risk (95% confidence)
    cvar_95_daily: float = 0.0     # Conditional VaR (Expected Shortfall)
    var_99_daily: float = 0.0
    cvar_99_daily: float = 0.0

    # Trade statistics (optional — populated when trade_returns given)
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for API/dashboard display."""
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


class RiskMetrics:
    """Static methods for computing risk-adjusted return metrics."""

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free: float = RISK_FREE_RATE_IND,
    ) -> float:
        """Annualized Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
        daily_rf = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
        excess = returns - daily_rf
        return float(excess.mean() / excess.std() * ANNUALISATION)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free: float = RISK_FREE_RATE_IND,
    ) -> float:
        """Annualized Sortino ratio (downside deviation only).

        Unlike Sharpe, this only penalizes volatility that causes losses,
        not upside volatility. Better for asymmetric return distributions.
        """
        if returns.empty:
            return 0.0
        daily_rf = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
        excess = returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) < 2:
            return float(excess.mean() / (excess.std() + 1e-10) * ANNUALISATION)
        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std <= 0:
            return 0.0
        return float(excess.mean() / downside_std * ANNUALISATION)

    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """Calmar ratio = annualized return / max drawdown.

        Measures quality of profit relative to worst pain.
        Higher is better; Calmar > 1.0 is good for swing strategies.
        """
        if returns.empty:
            return 0.0
        n_years = len(returns) / TRADING_DAYS
        if n_years <= 0:
            return 0.0
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        max_dd = abs(RiskMetrics.max_drawdown(returns))
        if max_dd <= 0:
            return 0.0
        return float(annual_return / max_dd)

    @staticmethod
    def omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0,
    ) -> float:
        """Omega ratio = sum(gains above threshold) / sum(losses below threshold).

        Omega > 1 means the return distribution is favorable.
        Superior to Sharpe as it considers the entire distribution,
        not just the first two moments.
        """
        if returns.empty:
            return 0.0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        total_gains = gains.sum()
        total_losses = losses.sum()
        if total_losses <= 0:
            return float('inf') if total_gains > 0 else 0.0
        return float(total_gains / total_losses)

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Maximum drawdown as a negative fraction (e.g., -0.15 = 15% DD)."""
        if returns.empty:
            return 0.0
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        drawdowns = (equity - peak) / peak
        return float(drawdowns.min())

    @staticmethod
    def max_drawdown_duration(returns: pd.Series) -> int:
        """Number of days in the longest drawdown period."""
        if returns.empty:
            return 0
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        in_dd = equity < peak

        max_dur = 0
        current_dur = 0
        for is_dd in in_dd:
            if is_dd:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
            else:
                current_dur = 0
        return max_dur

    @staticmethod
    def cvar(
        returns: pd.Series,
        alpha: float = 0.05,
    ) -> float:
        """Conditional Value-at-Risk (Expected Shortfall).

        Average loss in the worst α% of scenarios.
        More informative than VaR because it measures
        how bad the BAD scenarios actually are.
        """
        if returns.empty:
            return 0.0
        var_threshold = returns.quantile(alpha)
        tail = returns[returns <= var_threshold]
        if tail.empty:
            return float(var_threshold)
        return float(tail.mean())

    @staticmethod
    def var(
        returns: pd.Series,
        alpha: float = 0.05,
    ) -> float:
        """Value-at-Risk at confidence level (1-α)."""
        if returns.empty:
            return 0.0
        return float(returns.quantile(alpha))

    @staticmethod
    def ulcer_index(returns: pd.Series) -> float:
        """Ulcer Index — measures drawdown severity and duration.

        UI = sqrt(mean(drawdown_pct²))

        A low UI means shallow, brief drawdowns.
        Invented by Peter Martin (1987) as a better risk measure
        than standard deviation for trend-following systems.
        """
        if returns.empty:
            return 0.0
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        dd_pct = ((equity - peak) / peak * 100)
        return float(np.sqrt(np.mean(dd_pct ** 2)))

    @staticmethod
    def recovery_factor(returns: pd.Series) -> float:
        """Recovery factor = net profit / max drawdown."""
        if returns.empty:
            return 0.0
        net_profit = (1 + returns).prod() - 1
        max_dd = abs(RiskMetrics.max_drawdown(returns))
        if max_dd <= 0:
            return 0.0
        return float(net_profit / max_dd)

    @staticmethod
    def gain_to_pain_ratio(returns: pd.Series) -> float:
        """Sum of all returns / sum of absolute negative returns."""
        if returns.empty:
            return 0.0
        total = returns.sum()
        pain = abs(returns[returns < 0].sum())
        if pain <= 0:
            return float('inf') if total > 0 else 0.0
        return float(total / pain)

    @staticmethod
    def profit_factor(trade_returns: pd.Series) -> float:
        """Gross profit / gross loss (from individual trade returns)."""
        if trade_returns.empty:
            return 0.0
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        if gross_loss <= 0:
            return float('inf') if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    @classmethod
    def compute_all(
        cls,
        daily_returns: pd.Series,
        trade_returns: Optional[pd.Series] = None,
        risk_free: float = RISK_FREE_RATE_IND,
    ) -> RiskMetricsResult:
        """Compute all risk metrics from a daily return series.

        Parameters
        ----------
        daily_returns : pd.Series
            Daily portfolio returns (fractional, e.g., 0.01 = 1%).
        trade_returns : pd.Series | None
            Individual trade returns (for win rate, profit factor, etc.).
        risk_free : float
            Annual risk-free rate.

        Returns
        -------
        RiskMetricsResult
        """
        r = daily_returns.dropna()
        result = RiskMetricsResult(n_days=len(r))

        if r.empty:
            return result

        # Returns
        total_return = (1 + r).prod() - 1
        n_years = len(r) / TRADING_DAYS
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        result.total_return_pct = round(total_return * 100, 2)
        result.annual_return_pct = round(annual_return * 100, 2)

        # Volatility
        result.annual_volatility_pct = round(float(r.std() * ANNUALISATION * 100), 2)
        downside = r[r < 0]
        result.downside_volatility_pct = round(
            float(np.sqrt(np.mean(downside ** 2)) * ANNUALISATION * 100) if len(downside) > 0 else 0.0, 2
        )

        # Risk-adjusted
        result.sharpe_ratio = round(cls.sharpe_ratio(r, risk_free), 4)
        result.sortino_ratio = round(cls.sortino_ratio(r, risk_free), 4)
        result.calmar_ratio = round(cls.calmar_ratio(r), 4)
        result.omega_ratio = round(cls.omega_ratio(r), 4)
        result.gain_to_pain_ratio = round(cls.gain_to_pain_ratio(r), 4)

        # Drawdown
        result.max_drawdown_pct = round(abs(cls.max_drawdown(r)) * 100, 2)
        result.max_drawdown_duration_days = cls.max_drawdown_duration(r)
        result.recovery_factor = round(cls.recovery_factor(r), 4)
        result.ulcer_index = round(cls.ulcer_index(r), 4)

        # Tail risk
        result.var_95_daily = round(cls.var(r, 0.05) * 100, 4)
        result.cvar_95_daily = round(cls.cvar(r, 0.05) * 100, 4)
        result.var_99_daily = round(cls.var(r, 0.01) * 100, 4)
        result.cvar_99_daily = round(cls.cvar(r, 0.01) * 100, 4)

        # Trade-level stats (if provided)
        if trade_returns is not None and not trade_returns.empty:
            tr = trade_returns.dropna()
            wins = tr[tr > 0]
            losses = tr[tr < 0]
            result.win_rate = round(len(wins) / len(tr) * 100, 2) if len(tr) > 0 else 0.0
            result.avg_win_pct = round(float(wins.mean() * 100), 2) if len(wins) > 0 else 0.0
            result.avg_loss_pct = round(float(losses.mean() * 100), 2) if len(losses) > 0 else 0.0
            result.profit_factor = round(cls.profit_factor(tr), 4)
            result.expectancy = round(
                float(tr.mean() * 100), 4
            )

        return result
