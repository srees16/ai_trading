"""
RL Trading Bot Module.

Gymnasium-based reinforcement learning agent for swing/positional
trading signals.  Supports DQN, PPO, and A2C via Stable-Baselines3.

Quick start:
    from services.rl_bot import train_rl_agent, evaluate_agent, get_latest_signal

    # Train
    result = train_rl_agent("RELIANCE.NS")
    print(result.avg_test_return, result.model_path)

    # Evaluate
    metrics, signals, trades = evaluate_agent(
        "RELIANCE.NS", result.model_path, algorithm="PPO"
    )

    # Live signal
    signal = get_latest_signal("RELIANCE.NS", result.model_path)
    print(signal.action, signal.confidence)
"""

from services.rl_bot.evaluate_agent import (
    EvalMetrics,
    RLSignal,
    evaluate_agent,
    get_latest_signal,
)
from services.rl_bot.rl_signal_integrator import (
    get_rl_layer_score,
    run_rl_layer,
)
from services.rl_bot.train_rl_agent import (
    TrainConfig,
    TrainResult,
    train_multi_ticker,
    train_rl_agent,
)

__all__ = [
    "train_rl_agent",
    "train_multi_ticker",
    "TrainConfig",
    "TrainResult",
    "evaluate_agent",
    "get_latest_signal",
    "EvalMetrics",
    "RLSignal",
    "get_rl_layer_score",
    "run_rl_layer",
]
