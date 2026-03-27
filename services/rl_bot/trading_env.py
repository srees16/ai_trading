"""
RL Trading Bot — Custom Gymnasium Trading Environment.

A Gym-compatible environment for training RL agents on swing/positional
trading.  Reuses Centurion's existing data services, indicators, and
regime detection.

Actions:  {0: Hold, 1: Buy, 2: Sell}
State:    Feature vector from feature_builder (technical + quant +
          fundamental + regime + portfolio state)
Reward:   Configurable via reward.py (PnL / Sharpe / Hybrid)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from services.rl_bot.feature_builder import (
    FEATURE_DIM,
    build_features_from_ohlcv,
)
from services.rl_bot.reward import compute_reward

logger = logging.getLogger(__name__)

# Actions
HOLD = 0
BUY = 1
SELL = 2


class TradingEnv(gym.Env):
    """Custom Gymnasium trading environment for swing/positional trades.

    Simulates day-by-day trading on a single ticker's OHLCV data.
    Designed for DQN / PPO / A2C via Stable-Baselines3.

    Args:
        ohlcv: DataFrame with columns [Open, High, Low, Close, Volume].
        initial_capital: Starting cash.
        transaction_cost: Round-trip cost as fraction (e.g. 0.0013).
        slippage_bps: Slippage in basis points.
        reward_type: "pnl", "sharpe", or "hybrid".
        max_position: Max fraction of capital per trade (0..1).
        lookback: Number of bars for feature window.
        metrics_series: Optional list of StockMetrics per bar (from MetricsCalculator).
        regime: Optional RegimeSnapshot (from RegimeDetector).
        is_indian: Whether this is an Indian ticker (affects costs).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        *,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.0013,
        slippage_bps: float = 20.0,
        reward_type: str = "hybrid",
        max_position: float = 1.0,
        lookback: int = 60,
        metrics_series: Optional[list] = None,
        regime: Optional[object] = None,
        is_indian: bool = True,
    ):
        super().__init__()

        # Normalise columns
        if isinstance(ohlcv.columns, pd.MultiIndex):
            ohlcv = ohlcv.copy()
            ohlcv.columns = [c[0] if isinstance(c, tuple) else c for c in ohlcv.columns]
        self._ohlcv = ohlcv.reset_index(drop=True)

        self._initial_capital = initial_capital
        self._transaction_cost = transaction_cost
        self._slippage_bps = slippage_bps
        self._reward_type = reward_type
        self._max_position = max_position
        self._lookback = lookback
        self._metrics_series = metrics_series
        self._regime = regime
        self._is_indian = is_indian

        # Gym spaces
        self.action_space = spaces.Discrete(3)  # Hold / Buy / Sell
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(FEATURE_DIM,),
            dtype=np.float32,
        )

        # Episode state (initialised in reset())
        self._step_idx = 0
        self._cash = initial_capital
        self._shares = 0
        self._entry_price = 0.0
        self._peak_value = initial_capital
        self._prev_action = HOLD
        self._days_held = 0
        self._rolling_returns: list = []
        self._trade_log: list = []
        self._portfolio_values: list = []

    @property
    def n_steps(self) -> int:
        return len(self._ohlcv)

    # ── Gym API ─────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._step_idx = self._lookback  # start after lookback
        self._cash = self._initial_capital
        self._shares = 0
        self._entry_price = 0.0
        self._peak_value = self._initial_capital
        self._prev_action = HOLD
        self._days_held = 0
        self._rolling_returns = []
        self._trade_log = []
        self._portfolio_values = [self._initial_capital]

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one trading step.

        Returns:
            obs, reward, terminated, truncated, info
        """
        prev_pv = self._portfolio_value()
        current_price = self._current_price()

        # ── Execute action ──────────────────────────────────
        if action == BUY and self._shares == 0:
            # Buy with available cash (respecting max_position)
            available = self._cash * self._max_position
            cost_per_share = current_price * (1 + self._slippage_bps / 10_000)
            shares_to_buy = int(available / cost_per_share)
            if shares_to_buy > 0:
                total_cost = shares_to_buy * cost_per_share
                commission = total_cost * self._transaction_cost
                self._cash -= (total_cost + commission)
                self._shares = shares_to_buy
                self._entry_price = current_price
                self._days_held = 0
                self._trade_log.append({
                    "step": self._step_idx,
                    "action": "BUY",
                    "price": current_price,
                    "shares": shares_to_buy,
                    "commission": commission,
                })

        elif action == SELL and self._shares > 0:
            # Sell all shares
            sell_price = current_price * (1 - self._slippage_bps / 10_000)
            proceeds = self._shares * sell_price
            commission = proceeds * self._transaction_cost
            self._cash += (proceeds - commission)
            pnl = (sell_price - self._entry_price) * self._shares
            self._trade_log.append({
                "step": self._step_idx,
                "action": "SELL",
                "price": current_price,
                "shares": self._shares,
                "pnl": pnl,
                "commission": commission,
                "days_held": self._days_held,
            })
            self._shares = 0
            self._entry_price = 0.0
            self._days_held = 0

        # Hold (or invalid Buy/Sell) → no trade
        if self._shares > 0:
            self._days_held += 1

        # ── Update portfolio tracking ───────────────────────
        pv = self._portfolio_value()
        self._peak_value = max(self._peak_value, pv)
        self._portfolio_values.append(pv)

        # Rolling returns for Sharpe calculation
        ret = (pv - prev_pv) / prev_pv if prev_pv > 0 else 0
        self._rolling_returns.append(ret)
        if len(self._rolling_returns) > 60:
            self._rolling_returns = self._rolling_returns[-60:]

        # ── Reward ──────────────────────────────────────────
        reward = compute_reward(
            pv, prev_pv,
            action=action,
            prev_action=self._prev_action,
            peak_value=self._peak_value,
            rolling_returns=np.array(self._rolling_returns),
            is_indian=self._is_indian,
            reward_type=self._reward_type,
        )

        self._prev_action = action

        # ── Advance step ────────────────────────────────────
        self._step_idx += 1
        terminated = self._step_idx >= len(self._ohlcv) - 1
        truncated = False

        # Emergency stop: portfolio wiped out
        if pv < self._initial_capital * 0.5:
            terminated = True
            reward -= 0.5  # Large penalty for catastrophic loss

        obs = self._get_observation() if not terminated else np.zeros(FEATURE_DIM, dtype=np.float32)
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        pv = self._portfolio_value()
        ret = (pv / self._initial_capital - 1) * 100
        pos = "LONG" if self._shares > 0 else "FLAT"
        print(
            f"Step {self._step_idx:4d} | "
            f"PV={pv:12,.0f} ({ret:+.1f}%) | "
            f"{pos} {self._shares} shares | "
            f"Trades={len(self._trade_log)}"
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _current_price(self) -> float:
        idx = min(self._step_idx, len(self._ohlcv) - 1)
        return float(self._ohlcv.iloc[idx]["Close"])

    def _portfolio_value(self) -> float:
        return self._cash + self._shares * self._current_price()

    def _get_observation(self) -> np.ndarray:
        metrics = None
        if self._metrics_series and self._step_idx < len(self._metrics_series):
            metrics = self._metrics_series[self._step_idx]

        ps = {
            "cash": self._cash,
            "holdings_value": self._shares * self._current_price(),
            "portfolio_value": self._portfolio_value(),
            "entry_price": self._entry_price,
            "current_price": self._current_price(),
            "initial_capital": self._initial_capital,
            "peak_value": self._peak_value,
            "days_held": self._days_held,
        }

        return build_features_from_ohlcv(
            self._ohlcv,
            self._step_idx,
            lookback=self._lookback,
            metrics=metrics,
            regime=self._regime,
            portfolio_state=ps,
        )

    def _get_info(self) -> Dict[str, Any]:
        pv = self._portfolio_value()
        return {
            "portfolio_value": pv,
            "cash": self._cash,
            "shares": self._shares,
            "total_return_pct": (pv / self._initial_capital - 1) * 100,
            "max_drawdown_pct": (pv / self._peak_value - 1) * 100 if self._peak_value > 0 else 0,
            "n_trades": len(self._trade_log),
            "step": self._step_idx,
        }

    def get_trade_log(self) -> list:
        """Return full trade log for backtesting analysis."""
        return list(self._trade_log)

    def get_portfolio_series(self) -> np.ndarray:
        """Return portfolio value time series."""
        return np.array(self._portfolio_values)
