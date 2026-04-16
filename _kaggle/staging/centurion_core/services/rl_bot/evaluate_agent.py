"""
RL Trading Bot — Evaluation & Backtesting.

Run a trained agent end-to-end on held-out data and produce a full
performance report.  Reuses existing helpers:
  - strategies.data_service.DataService   for data
  - utils.calculate_max_drawdown          for drawdown
  - config.Config                         for costs/thresholds

Key output metrics:
  CAGR, Sharpe, Sortino, max drawdown, win rate, profit factor,
  average holding period, daily signal (Buy/Sell/Hold + confidence).
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config
from services.rl_bot.trading_env import TradingEnv

logger = logging.getLogger(__name__)

REPORT_DIR = Path("data") / "rl_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EvalMetrics:
    """Full evaluation performance metrics."""
    ticker: str = ""
    algorithm: str = ""
    period_start: str = ""
    period_end: str = ""
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    buy_and_hold_return_pct: float = 0.0
    excess_return_pct: float = 0.0
    final_portfolio_value: float = 0.0


@dataclass
class RLSignal:
    """Daily signal emitted by the RL agent."""
    ticker: str
    date: str
    action: str        # BUY | SELL | HOLD
    confidence: float  # 0.0 – 1.0
    portfolio_value: float = 0.0
    position: str = "FLAT"  # FLAT | LONG


def evaluate_agent(
    ticker: str,
    model_path: str,
    algorithm: str = "PPO",
    *,
    eval_days: int = 252,
    initial_capital: float = 100_000,
    lookback: int = 60,
) -> Tuple[EvalMetrics, List[RLSignal], List[dict]]:
    """Run an RL agent on recent data and compute full metrics.

    Args:
        ticker: Stock ticker.
        model_path: Path to saved SB3 model (without extension).
        algorithm: DQN | PPO | A2C.
        eval_days: Number of trading days to evaluate on.
        initial_capital: Starting capital.
        lookback: Feature lookback window.

    Returns:
        (EvalMetrics, list of daily RLSignals, list of trade records)
    """
    from stable_baselines3 import DQN, PPO, A2C

    algo_map = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
    algo_cls = algo_map.get(algorithm.upper(), PPO)

    # Load model
    model = algo_cls.load(model_path)

    # Fetch data
    ohlcv = _fetch_eval_data(ticker, eval_days + lookback + 50)
    if ohlcv is None or len(ohlcv) < lookback + eval_days:
        logger.error("Insufficient eval data for %s", ticker)
        return EvalMetrics(ticker=ticker), [], []

    # Take last eval_days + lookback
    ohlcv = ohlcv.iloc[-(eval_days + lookback):].reset_index(drop=True)

    # Detect market
    is_indian = ticker.upper().endswith((".NS", ".BO"))
    tx_cost = Config.TRANSACTION_COST_IND if is_indian else Config.TRANSACTION_COST_US
    slippage = Config.SLIPPAGE_MODEL_IND_BPS if is_indian else Config.SLIPPAGE_MODEL_US_BPS

    # Attempt to get regime
    regime = None
    try:
        from services.regime_detector import regime_detector
        regime = regime_detector.detect()
    except Exception:
        pass

    # Build environment
    env = TradingEnv(
        ohlcv,
        initial_capital=initial_capital,
        transaction_cost=tx_cost,
        slippage_bps=slippage,
        reward_type="hybrid",
        lookback=lookback,
        regime=regime,
        is_indian=is_indian,
    )

    # ── Run episode ─────────────────────────────────────────
    obs, info = env.reset()
    done = False
    action_history = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        action_history.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # ── Collect raw data ────────────────────────────────────
    pv_series = np.array(env.get_portfolio_series())
    trade_log = env.get_trade_log()

    # ── Compute metrics ─────────────────────────────────────
    metrics = _compute_metrics(
        ticker, algorithm, ohlcv, pv_series, trade_log,
        initial_capital, lookback,
    )

    # ── Build daily signals ─────────────────────────────────
    signals = _build_daily_signals(
        ticker, ohlcv, action_history, pv_series, lookback,
    )

    # ── Save report ─────────────────────────────────────────
    _save_report(metrics, signals, trade_log, ticker, algorithm)

    return metrics, signals, trade_log


def get_latest_signal(
    ticker: str,
    model_path: str,
    algorithm: str = "PPO",
    lookback: int = 60,
) -> RLSignal:
    """Get the RL agent's signal for the most recent bar.

    This is the primary API used by the signal integrator
    to query the RL bot for a live trading decision.
    """
    from stable_baselines3 import DQN, PPO, A2C

    algo_map = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
    model = algo_map.get(algorithm.upper(), PPO).load(model_path)

    ohlcv = _fetch_eval_data(ticker, lookback + 30)
    if ohlcv is None or len(ohlcv) < lookback + 5:
        return RLSignal(
            ticker=ticker,
            date=datetime.now().strftime("%Y-%m-%d"),
            action="HOLD",
            confidence=0.0,
        )

    is_indian = ticker.upper().endswith((".NS", ".BO"))
    tx_cost = Config.TRANSACTION_COST_IND if is_indian else Config.TRANSACTION_COST_US
    slippage = Config.SLIPPAGE_MODEL_IND_BPS if is_indian else Config.SLIPPAGE_MODEL_US_BPS

    env = TradingEnv(
        ohlcv,
        initial_capital=100_000,
        transaction_cost=tx_cost,
        slippage_bps=slippage,
        lookback=lookback,
        is_indian=is_indian,
    )

    obs, _ = env.reset()
    # Step through to the last bar
    for _ in range(len(ohlcv) - lookback - 2):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    # Final prediction on latest bar
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    action_labels = {0: "HOLD", 1: "BUY", 2: "SELL"}

    # Confidence: use action probability from policy if available
    confidence = _get_action_confidence(model, obs, action)

    pv = env._portfolio_value()
    pos = "LONG" if env._shares > 0 else "FLAT"

    return RLSignal(
        ticker=ticker,
        date=datetime.now().strftime("%Y-%m-%d"),
        action=action_labels.get(action, "HOLD"),
        confidence=confidence,
        portfolio_value=pv,
        position=pos,
    )


# ── Internal helpers ────────────────────────────────────────────────

def _fetch_eval_data(ticker: str, bars_needed: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for evaluation."""
    try:
        from strategies.data_service import DataService
        ds = DataService()
        end = datetime.now()
        start = end - timedelta(days=int(bars_needed * 1.5))
        df = ds.get_ohlcv(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df
    except Exception as e:
        logger.warning("DataService failed: %s — falling back to yfinance", e)

    try:
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=int(bars_needed * 1.5))
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if not df.empty:
            return df
    except Exception as e:
        logger.error("yfinance fallback failed: %s", e)

    return None


def _compute_metrics(
    ticker: str,
    algorithm: str,
    ohlcv: pd.DataFrame,
    pv_series: np.ndarray,
    trade_log: List[dict],
    initial_capital: float,
    lookback: int,
) -> EvalMetrics:
    """Compute comprehensive performance metrics."""
    m = EvalMetrics(ticker=ticker, algorithm=algorithm)

    if len(pv_series) < 2:
        return m

    # Date range
    if hasattr(ohlcv.index[0], "strftime"):
        m.period_start = str(ohlcv.index[lookback])
        m.period_end = str(ohlcv.index[-1])

    m.final_portfolio_value = float(pv_series[-1])

    # Total return
    m.total_return_pct = (pv_series[-1] / pv_series[0] - 1) * 100

    # CAGR
    n_days = len(pv_series) - 1
    if n_days > 0:
        years = n_days / 252
        ratio = pv_series[-1] / pv_series[0]
        m.cagr_pct = (ratio ** (1 / years) - 1) * 100 if years > 0 and ratio > 0 else 0

    # Daily returns
    daily_returns = np.diff(pv_series) / pv_series[:-1]

    # Sharpe (annualised)
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    m.sharpe_ratio = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0

    # Sortino (annualised)
    neg_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 0
    m.sortino_ratio = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 1e-8 else 0

    # Max drawdown
    peak = np.maximum.accumulate(pv_series)
    drawdowns = (pv_series - peak) / peak * 100
    m.max_drawdown_pct = float(np.min(drawdowns))

    # Trade-level metrics
    sell_trades = [t for t in trade_log if t.get("action") == "SELL"]
    m.total_trades = len(sell_trades)

    if sell_trades:
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
        m.win_rate = len(wins) / len(sell_trades)

        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        days_held = [t.get("days_held", 0) for t in sell_trades if "days_held" in t]
        m.avg_holding_days = np.mean(days_held) if days_held else 0

        if wins:
            win_pcts = [t["pnl"] / (t.get("price", 1) * t.get("shares", 1)) * 100 for t in wins]
            m.avg_win_pct = float(np.mean(win_pcts))
        if losses:
            loss_pcts = [t["pnl"] / (t.get("price", 1) * t.get("shares", 1)) * 100 for t in losses]
            m.avg_loss_pct = float(np.mean(loss_pcts))

    # Buy & hold benchmark
    prices = ohlcv["Close"].values
    if len(prices) > lookback + 1:
        m.buy_and_hold_return_pct = (prices[-1] / prices[lookback] - 1) * 100
    m.excess_return_pct = m.total_return_pct - m.buy_and_hold_return_pct

    return m


def _build_daily_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    action_history: List[int],
    pv_series: np.ndarray,
    lookback: int,
) -> List[RLSignal]:
    """Convert action history to daily signal objects."""
    signals = []
    action_labels = {0: "HOLD", 1: "BUY", 2: "SELL"}

    for i, action in enumerate(action_history):
        bar_idx = lookback + i + 1  # actions start after first obs
        if bar_idx >= len(ohlcv):
            break

        date_str = str(ohlcv.index[bar_idx]) if hasattr(ohlcv.index[0], "strftime") else str(bar_idx)
        pv_idx = min(i + 1, len(pv_series) - 1)

        signals.append(RLSignal(
            ticker=ticker,
            date=date_str,
            action=action_labels.get(action, "HOLD"),
            confidence=0.0,  # filled by action probs when available
            portfolio_value=float(pv_series[pv_idx]),
        ))

    return signals


def _get_action_confidence(model, obs: np.ndarray, chosen_action: int) -> float:
    """Extract action confidence from model's policy distribution."""
    try:
        import torch
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(model.device)
        with torch.no_grad():
            # PPO / A2C have action distribution
            if hasattr(model.policy, "get_distribution"):
                dist = model.policy.get_distribution(
                    model.policy.extract_features(obs_tensor, model.policy.features_extractor)
                )
                probs = dist.distribution.probs.cpu().numpy().flatten()
                return float(probs[chosen_action])
            # DQN: use softmax over Q-values
            elif hasattr(model.policy, "q_net"):
                q_values = model.policy.q_net(
                    model.policy.extract_features(obs_tensor, model.policy.features_extractor)
                )
                q_np = q_values.cpu().numpy().flatten()
                # Temperature softmax
                exp_q = np.exp(q_np - np.max(q_np))
                probs = exp_q / exp_q.sum()
                return float(probs[chosen_action])
    except Exception as e:
        logger.debug("Could not extract confidence: %s", e)

    return 0.5  # Default moderate confidence


def _save_report(
    metrics: EvalMetrics,
    signals: List[RLSignal],
    trade_log: List[dict],
    ticker: str,
    algorithm: str,
):
    """Save evaluation report to JSON."""
    safe_ticker = ticker.replace(".", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"{safe_ticker}_{algorithm.lower()}_{timestamp}.json"

    report = {
        "metrics": asdict(metrics),
        "signals_count": len(signals),
        "last_5_signals": [
            {"date": s.date, "action": s.action, "confidence": s.confidence}
            for s in signals[-5:]
        ],
        "trade_log": trade_log,
    }

    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Evaluation report saved: %s", path)
