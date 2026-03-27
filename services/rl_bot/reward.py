"""
RL Trading Bot — Reward Functions.

Multiple reward strategies that can be swapped via config:
  1. PnL-based     — raw portfolio value change
  2. Risk-adjusted — Sharpe-ratio proxy (return / rolling vol)
  3. Hybrid        — PnL + penalties for drawdown + excessive trading

Reuses Config.TRANSACTION_COST_IND / _US for cost penalties.
"""

import numpy as np

from config import Config


def compute_reward(
    portfolio_value: float,
    prev_portfolio_value: float,
    *,
    action: int,
    prev_action: int,
    peak_value: float,
    rolling_returns: np.ndarray,
    is_indian: bool = True,
    reward_type: str = "hybrid",
) -> float:
    """Compute step reward.

    Args:
        portfolio_value: Current total portfolio value.
        prev_portfolio_value: Previous step's portfolio value.
        action: Current action (0=Hold, 1=Buy, 2=Sell).
        prev_action: Previous action.
        peak_value: All-time high portfolio value.
        rolling_returns: Last N step returns for Sharpe proxy.
        is_indian: Use IND or US transaction cost.
        reward_type: "pnl", "sharpe", or "hybrid".

    Returns:
        Scalar reward (float).
    """
    if reward_type == "pnl":
        return _reward_pnl(portfolio_value, prev_portfolio_value)
    elif reward_type == "sharpe":
        return _reward_sharpe(portfolio_value, prev_portfolio_value, rolling_returns)
    else:
        return _reward_hybrid(
            portfolio_value, prev_portfolio_value,
            action, prev_action, peak_value,
            rolling_returns, is_indian,
        )


# ── Reward strategies ───────────────────────────────────────────────

def _reward_pnl(pv: float, prev_pv: float) -> float:
    """Raw percentage PnL."""
    if prev_pv <= 0:
        return 0.0
    return (pv - prev_pv) / prev_pv


def _reward_sharpe(pv: float, prev_pv: float, rolling_returns: np.ndarray) -> float:
    """Sharpe-ratio proxy: recent return / rolling std."""
    ret = (pv - prev_pv) / prev_pv if prev_pv > 0 else 0.0
    if len(rolling_returns) < 5:
        return ret
    std = np.std(rolling_returns)
    if std < 1e-8:
        return ret
    return ret / std


def _reward_hybrid(
    pv: float,
    prev_pv: float,
    action: int,
    prev_action: int,
    peak_value: float,
    rolling_returns: np.ndarray,
    is_indian: bool,
) -> float:
    """Hybrid reward optimised for swing/positional trading.

    Components:
      +  PnL change (normalised)
      -  Drawdown penalty (scaled by depth)
      -  Transaction cost penalty (on trades)
      +  Sharpe bonus (risk-adjusted performance)
      +  Swing hold bonus (reward patient holds during uptrend)
      -  Whipsaw penalty (penalise rapid buy-sell-buy)
    """
    if prev_pv <= 0:
        return 0.0

    # Base: percentage return
    ret = (pv - prev_pv) / prev_pv

    # Drawdown penalty — penalise being far below peak
    drawdown = (pv - peak_value) / peak_value if peak_value > 0 else 0
    dd_penalty = 0.0
    if drawdown < -0.05:
        dd_penalty = drawdown * 0.5   # Scales with drawdown depth

    # Transaction cost penalty — penalise excessive trading
    trade_penalty = 0.0
    if action != prev_action and action != 0:  # 0 = Hold
        cost = Config.TRANSACTION_COST_IND if is_indian else Config.TRANSACTION_COST_US
        trade_penalty = -cost

    # Sharpe bonus
    sharpe_bonus = 0.0
    if len(rolling_returns) >= 10:
        std = np.std(rolling_returns)
        if std > 1e-8:
            recent_sharpe = np.mean(rolling_returns) / std
            sharpe_bonus = np.clip(recent_sharpe * 0.01, -0.02, 0.02)

    # Gap 7: Swing hold bonus — reward staying in profitable positions
    # Encourages the agent to hold through multi-day trends (swing trade)
    hold_bonus = 0.0
    if action == 0 and prev_action != 2:  # Holding (not just after a sell)
        if ret > 0:
            # Profitable hold — small bonus encourages patience
            hold_bonus = min(ret * 0.3, 0.005)

    # Gap 7: Whipsaw penalty — penalise rapid reversal (buy → sell next bar)
    whipsaw_penalty = 0.0
    if (action == 2 and prev_action == 1) or (action == 1 and prev_action == 2):
        whipsaw_penalty = -0.003  # Fixed penalty for flip-flopping

    reward = ret + dd_penalty + trade_penalty + sharpe_bonus + hold_bonus + whipsaw_penalty

    return float(reward)
