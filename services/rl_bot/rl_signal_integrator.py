"""
RL Trading Bot — Signal Integrator.

Plugs RL predictions into the existing Centurion evaluation pipeline
as an optional scoring layer in IntegratedScorer.

Integration approach:
  - RL acts as a standalone layer alongside 'core' and 'strategy'.
  - Default weight: 15% (redistributed from core & strategy).
  - When no trained model exists for a ticker, the layer is silently
    skipped and weights re-normalise.

Reuses:
  - services.integrated_scorer.IntegratedScorer weights system
  - evaluate_agent.get_latest_signal  for live predictions
  - config.Config                     for model paths and thresholds
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config

logger = logging.getLogger(__name__)

# Default RL layer weight when integrated
RL_LAYER_WEIGHT = 0.15

# Mapping RL action → numeric score (-1..+1 scale, matching Centurion scoring)
ACTION_SCORE_MAP = {
    "STRONG_BUY": 0.8,
    "BUY": 0.5,
    "HOLD": 0.0,
    "SELL": -0.5,
    "STRONG_SELL": -0.8,
}

# Weights when RL layer is active
WEIGHTS_WITH_RL = {
    "core": 0.40,
    "strategy": 0.45,
    "rl_bot": 0.15,
}

WEIGHTS_WITH_RL_AND_ML = {
    "core": 0.35,
    "strategy": 0.35,
    "ml_features": 0.15,
    "rl_bot": 0.15,
}

MODEL_DIR = Path("data") / "rl_models"


def get_rl_layer_score(
    ticker: str,
    algorithm: str = "PPO",
    lookback: int = 60,
) -> Optional[Dict[str, Any]]:
    """Compute RL layer score for a single ticker.

    Returns a dict compatible with IntegratedScorer's layer result format:
        {"score": float, "details": dict}

    Returns None if no model is available for the ticker.
    """
    model_path = _find_model(ticker, algorithm)
    if model_path is None:
        logger.debug("No RL model found for %s — skipping RL layer", ticker)
        return None

    try:
        from services.rl_bot.evaluate_agent import get_latest_signal

        signal = get_latest_signal(
            ticker,
            model_path=model_path,
            algorithm=algorithm,
            lookback=lookback,
        )

        # Map action to score
        raw_score = ACTION_SCORE_MAP.get(signal.action, 0.0)

        # Scale by confidence (0..1) → dampens low-confidence signals
        score = raw_score * signal.confidence

        details = {
            "action": signal.action,
            "confidence": round(signal.confidence, 3),
            "raw_score": raw_score,
            "weighted_score": round(score, 4),
            "position": signal.position,
            "model_path": model_path,
            "algorithm": algorithm,
        }

        logger.info(
            "RL layer for %s: %s (conf=%.2f, score=%.3f)",
            ticker, signal.action, signal.confidence, score,
        )

        return {"score": score, "details": details}

    except Exception as e:
        logger.warning("RL layer failed for %s: %s", ticker, e)
        return None


def run_rl_layer(
    ticker: str,
    market: str = "US",
) -> Dict[str, Any]:
    """Run RL layer — callable from IntegratedScorer's ThreadPoolExecutor.

    This function signature matches _run_layer_core / _run_layer_strategy
    so it can be submitted as a parallel future.
    """
    algorithm = getattr(Config, "RL_ALGORITHM", "PPO")
    lookback = getattr(Config, "RL_LOOKBACK", 60)

    result = get_rl_layer_score(ticker, algorithm, lookback)

    if result is None:
        return {"score": None, "details": {"status": "no_model"}}

    return result


def get_rl_weights(include_ml: bool = False) -> Dict[str, float]:
    """Return weight dict with RL layer included.

    Called by IntegratedScorer when RL is enabled.
    """
    if include_ml:
        return dict(WEIGHTS_WITH_RL_AND_ML)
    return dict(WEIGHTS_WITH_RL)


# ── Helper ──────────────────────────────────────────────────────────

def _find_model(ticker: str, algorithm: str) -> Optional[str]:
    """Locate a saved model for the given ticker + algorithm.

    Convention: models are saved as <safe_ticker>_<algo>.zip
    """
    safe_ticker = ticker.replace(".", "_").replace(":", "_")
    candidate = MODEL_DIR / f"{safe_ticker}_{algorithm.lower()}"

    # SB3 saves as .zip
    if (candidate.with_suffix(".zip")).exists():
        return str(candidate)

    # Legacy: no extension
    if candidate.exists():
        return str(candidate)

    # Try other algorithms as fallback
    for algo in ("ppo", "dqn", "a2c"):
        alt = MODEL_DIR / f"{safe_ticker}_{algo}"
        if alt.with_suffix(".zip").exists():
            logger.info("Using fallback model %s for %s (requested %s)",
                        algo.upper(), ticker, algorithm)
            return str(alt)

    return None
