"""
RL Trading Bot — Training Pipeline.

Supports three algorithms via Stable-Baselines3:
  - DQN  (baseline — discrete actions, experience replay)
  - PPO  (recommended — stable, clipped surrogate)
  - A2C  (lightweight — synchronous advantage actor-critic)

Walk-forward training:
  1. Split data into rolling train/test windows
  2. Train agent on train window
  3. Evaluate on test window (out-of-sample)
  4. Roll forward and repeat

Reuses:
  - strategies.data_service.DataService  for OHLCV fetching
  - config.Config                        for cost / period params
  - services.regime_detector             for regime context
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from services.rl_bot.trading_env import TradingEnv

logger = logging.getLogger(__name__)

# Model save directory
MODEL_DIR = Path("data") / "rl_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    algorithm: str = "PPO"                # DQN | PPO | A2C
    total_timesteps: int = 50_000         # Training steps per fold
    learning_rate: float = 3e-4
    batch_size: int = 64                  # DQN batch size
    gamma: float = 0.99                   # Discount factor
    buffer_size: int = 50_000             # DQN replay buffer
    exploration_fraction: float = 0.2     # DQN epsilon schedule
    n_steps: int = 2048                   # PPO/A2C steps per update
    ent_coef: float = 0.01               # Entropy bonus (exploration)
    reward_type: str = "hybrid"           # pnl | sharpe | hybrid
    initial_capital: float = 100_000
    lookback: int = 60
    # Walk-forward
    train_days: int = 252                 # 1 year train
    test_days: int = 63                   # 1 quarter test
    total_folds: int = 4                  # Number of walk-forward folds
    # Multi-stock
    tickers: List[str] = field(default_factory=lambda: ["RELIANCE.NS"])


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_return_pct: float = 0.0
    test_sharpe: float = 0.0
    test_max_drawdown_pct: float = 0.0
    test_n_trades: int = 0
    test_win_rate: float = 0.0


@dataclass
class TrainResult:
    """Aggregate training result."""
    ticker: str
    algorithm: str
    folds: List[FoldResult] = field(default_factory=list)
    avg_test_return: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_test_drawdown: float = 0.0
    model_path: str = ""


def train_rl_agent(
    ticker: str,
    cfg: Optional[TrainConfig] = None,
) -> TrainResult:
    """Train an RL agent with walk-forward validation.

    Args:
        ticker: Stock ticker (e.g. 'RELIANCE.NS', 'AAPL').
        cfg: Training configuration. Uses defaults if None.

    Returns:
        TrainResult with fold metrics and saved model path.
    """
    cfg = cfg or TrainConfig()
    result = TrainResult(ticker=ticker, algorithm=cfg.algorithm)

    # ── Fetch OHLCV data ────────────────────────────────────
    ohlcv = _fetch_data(ticker, cfg)
    if ohlcv is None or len(ohlcv) < cfg.train_days + cfg.test_days + cfg.lookback:
        logger.error("Insufficient data for %s (%d bars needed, %d available)",
                     ticker, cfg.train_days + cfg.test_days + cfg.lookback,
                     len(ohlcv) if ohlcv is not None else 0)
        return result

    # ── Fetch regime context ────────────────────────────────
    regime = _fetch_regime()

    # ── Determine Indian vs US ──────────────────────────────
    is_indian = ticker.upper().endswith((".NS", ".BO"))
    tx_cost = Config.TRANSACTION_COST_IND if is_indian else Config.TRANSACTION_COST_US
    slippage = Config.SLIPPAGE_MODEL_IND_BPS if is_indian else Config.SLIPPAGE_MODEL_US_BPS

    # ── Walk-forward folds ──────────────────────────────────
    total_bars = len(ohlcv)
    fold_size = cfg.train_days + cfg.test_days
    max_folds = min(cfg.total_folds, (total_bars - cfg.lookback) // cfg.test_days - 1)

    if max_folds < 1:
        logger.warning("Not enough data for walk-forward on %s", ticker)
        max_folds = 1

    model = None

    for fold_idx in range(max_folds):
        start_offset = total_bars - fold_size - (max_folds - 1 - fold_idx) * cfg.test_days
        train_start = max(0, start_offset)
        train_end = train_start + cfg.train_days
        test_start = train_end
        test_end = min(test_start + cfg.test_days, total_bars)

        train_df = ohlcv.iloc[train_start:train_end].reset_index(drop=True)
        test_df = ohlcv.iloc[test_start:test_end].reset_index(drop=True)

        if len(train_df) < cfg.lookback + 50 or len(test_df) < 10:
            continue

        logger.info(
            "%s fold %d: train [%d:%d] (%d bars), test [%d:%d] (%d bars)",
            ticker, fold_idx, train_start, train_end, len(train_df),
            test_start, test_end, len(test_df),
        )

        # ── Create train environment ────────────────────────
        train_env = TradingEnv(
            train_df,
            initial_capital=cfg.initial_capital,
            transaction_cost=tx_cost,
            slippage_bps=slippage,
            reward_type=cfg.reward_type,
            lookback=cfg.lookback,
            regime=regime,
            is_indian=is_indian,
        )

        # ── Build or retrain model ──────────────────────────
        model = _build_model(cfg, train_env, existing_model=model)
        model.learn(
            total_timesteps=cfg.total_timesteps,
            progress_bar=False,
        )

        # ── Evaluate on test window ─────────────────────────
        test_env = TradingEnv(
            test_df,
            initial_capital=cfg.initial_capital,
            transaction_cost=tx_cost,
            slippage_bps=slippage,
            reward_type=cfg.reward_type,
            lookback=min(cfg.lookback, len(test_df) - 2),
            regime=regime,
            is_indian=is_indian,
        )

        fold_metrics = _evaluate_on_env(model, test_env)

        # Date labels from index
        t_start_str = str(ohlcv.index[train_start]) if hasattr(ohlcv.index[0], "strftime") else str(train_start)
        t_end_str = str(ohlcv.index[train_end - 1]) if hasattr(ohlcv.index[0], "strftime") else str(train_end)
        te_start_str = str(ohlcv.index[test_start]) if hasattr(ohlcv.index[0], "strftime") else str(test_start)
        te_end_str = str(ohlcv.index[test_end - 1]) if hasattr(ohlcv.index[0], "strftime") else str(test_end)

        fold_result = FoldResult(
            fold=fold_idx,
            train_start=t_start_str,
            train_end=t_end_str,
            test_start=te_start_str,
            test_end=te_end_str,
            test_return_pct=fold_metrics["total_return_pct"],
            test_sharpe=fold_metrics["sharpe"],
            test_max_drawdown_pct=fold_metrics["max_drawdown_pct"],
            test_n_trades=fold_metrics["n_trades"],
            test_win_rate=fold_metrics["win_rate"],
        )
        result.folds.append(fold_result)

        logger.info(
            "%s fold %d: return=%.1f%%, sharpe=%.2f, dd=%.1f%%, trades=%d, win=%.0f%%",
            ticker, fold_idx, fold_result.test_return_pct,
            fold_result.test_sharpe, fold_result.test_max_drawdown_pct,
            fold_result.test_n_trades, fold_result.test_win_rate * 100,
        )

    # ── Aggregate metrics ───────────────────────────────────
    if result.folds:
        result.avg_test_return = np.mean([f.test_return_pct for f in result.folds])
        result.avg_test_sharpe = np.mean([f.test_sharpe for f in result.folds])
        result.avg_test_drawdown = np.mean([f.test_max_drawdown_pct for f in result.folds])

    # ── Save final model ────────────────────────────────────
    if model is not None:
        safe_ticker = ticker.replace(".", "_").replace(":", "_")
        model_path = str(MODEL_DIR / f"{safe_ticker}_{cfg.algorithm.lower()}")
        model.save(model_path)
        result.model_path = model_path
        logger.info("Model saved: %s", model_path)

    return result


# ── Multi-ticker training ───────────────────────────────────────────

def train_multi_ticker(
    tickers: List[str],
    cfg: Optional[TrainConfig] = None,
) -> Dict[str, TrainResult]:
    """Train RL agents for multiple tickers.

    Returns:
        dict mapping ticker → TrainResult.
    """
    cfg = cfg or TrainConfig()
    results = {}
    for ticker in tickers:
        logger.info("Training RL agent for %s ...", ticker)
        results[ticker] = train_rl_agent(ticker, cfg)
    return results


# ── Internal helpers ────────────────────────────────────────────────

def _fetch_data(ticker: str, cfg: TrainConfig) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data using existing DataService."""
    try:
        from strategies.data_service import DataService
        ds = DataService()
        total_days = cfg.train_days * cfg.total_folds + cfg.test_days * cfg.total_folds + cfg.lookback + 100
        end = datetime.now()
        start = end - timedelta(days=int(total_days * 1.5))  # extra margin for weekends
        df = ds.get_ohlcv(
            ticker,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        if df is not None and not df.empty:
            # Flatten MultiIndex if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df
    except Exception as e:
        logger.warning("DataService failed for %s: %s — falling back to yfinance", ticker, e)

    # Fallback: direct yfinance
    try:
        import yfinance as yf
        total_days = cfg.train_days * cfg.total_folds + cfg.test_days * cfg.total_folds + cfg.lookback + 100
        end = datetime.now()
        start = end - timedelta(days=int(total_days * 1.5))
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if not df.empty:
            return df
    except Exception as e:
        logger.error("yfinance fallback failed for %s: %s", ticker, e)

    return None


def _fetch_regime():
    """Fetch current regime from existing RegimeDetector."""
    try:
        from services.regime_detector import regime_detector
        return regime_detector.detect()
    except Exception:
        return None


def _build_model(cfg: TrainConfig, env: TradingEnv, existing_model=None):
    """Build or update SB3 model.

    If existing_model is provided, re-assigns the new env (for walk-forward
    incremental training).
    """
    from stable_baselines3 import DQN, PPO, A2C

    algo_map = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
    algo_cls = algo_map.get(cfg.algorithm.upper(), PPO)

    if existing_model is not None:
        existing_model.set_env(env)
        return existing_model

    common_kwargs = {
        "env": env,
        "learning_rate": cfg.learning_rate,
        "gamma": cfg.gamma,
        "verbose": 0,
        "device": "auto",
    }

    if cfg.algorithm.upper() == "DQN":
        return algo_cls(
            "MlpPolicy",
            **common_kwargs,
            batch_size=cfg.batch_size,
            buffer_size=cfg.buffer_size,
            exploration_fraction=cfg.exploration_fraction,
            exploration_final_eps=0.05,
            target_update_interval=500,
        )
    else:
        # PPO or A2C
        return algo_cls(
            "MlpPolicy",
            **common_kwargs,
            n_steps=min(cfg.n_steps, env.n_steps - env._lookback - 1),
            ent_coef=cfg.ent_coef,
        )


def _evaluate_on_env(model, env: TradingEnv) -> Dict:
    """Run a trained model through an environment and compute metrics."""
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    pv_series = env.get_portfolio_series()
    trades = env.get_trade_log()

    # Total return
    total_return = (pv_series[-1] / pv_series[0] - 1) * 100 if len(pv_series) > 1 else 0

    # Sharpe ratio (annualised)
    if len(pv_series) > 2:
        daily_returns = np.diff(pv_series) / pv_series[:-1]
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0
    else:
        sharpe = 0

    # Max drawdown
    peak = np.maximum.accumulate(pv_series)
    drawdowns = (pv_series - peak) / peak * 100
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Win rate
    sell_trades = [t for t in trades if t.get("action") == "SELL"]
    wins = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
    win_rate = wins / len(sell_trades) if sell_trades else 0

    return {
        "total_return_pct": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "n_trades": len(trades),
        "win_rate": float(win_rate),
        "portfolio_series": pv_series,
        "trade_log": trades,
    }
