"""
Masters Prediction & Classification Quality — From Timothy Masters'
"Assessing and Improving Prediction and Classification".

Implements:
  1. Walk-Forward Validation — expanding-window OOS testing (Masters Ch. 2-3)
  2. Directional Accuracy — fraction of correct sign predictions (Masters Ch. 4)
  3. Mean Absolute Calibration Error — forecast magnitude accuracy (Masters Ch. 5)
  4. Monte Carlo Permutation Test — is the signal statistically significant? (Masters Ch. 6-7)
  5. Information Coefficient (IC) — rank correlation of forecast vs actual (Masters Ch. 8)
  6. Forecast Quality Gate — suppress low-quality forecasts in real-time (Masters Ch. 9)
  7. R-Squared of Equity Curve — smoothness proxy for prediction quality
  8. Brier Score — for classification tasks (BUY/SELL/HOLD)

Integration:
  - IND: Quality-gates forecasts before they enter forecast_combiner → only high-quality
         signals reach the Kite auto-order pipeline
  - US: Provides prediction_quality metrics in API response → UI displays
         confidence indicators for manual trading decisions
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class PredictionQuality:
    """Quality assessment of a forecast signal."""
    symbol: str
    directional_accuracy: float    # % of correct sign predictions (0-1)
    mean_abs_error: float          # MAE of forecast vs actual returns
    information_coefficient: float  # Rank correlation forecast vs actual
    r_squared: float               # R² of equity curve (smoothness)
    brier_score: float             # Classification accuracy (0=perfect, 1=worst)
    monte_carlo_p_value: float     # p-value from permutation test (< 0.05 = significant)
    is_significant: bool           # True if MC p-value < threshold
    quality_score: float           # Combined 0-1 quality score
    confidence_multiplier: float   # Dampening factor for forecast (0.3-1.0)


@dataclass
class WalkForwardResult:
    """Walk-forward validation result for a strategy."""
    symbol: str
    n_windows: int               # Number of OOS windows
    oos_sharpe: float            # Sharpe ratio on OOS data
    oos_cagr: float              # CAGR on OOS data
    oos_directional_accuracy: float  # OOS directional accuracy
    is_degradation: float        # IS Sharpe - OOS Sharpe (positive = overfit)
    walk_forward_efficiency: float  # OOS Sharpe / IS Sharpe (> 0.5 = robust)


# ═══════════════════════════════════════════════════════════════
# Directional Accuracy (Masters Ch. 4)
# ═══════════════════════════════════════════════════════════════

def compute_directional_accuracy(
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
) -> float:
    """
    Fraction of correct sign predictions.

    Masters Ch. 4: "The most basic measure of prediction quality
    is whether the predicted direction matches the actual direction."

    > 0.55 = useful signal
    > 0.60 = strong signal
    < 0.50 = worse than random

    Args:
        forecasts: array of forecast values (sign matters)
        actual_returns: array of actual returns

    Returns:
        directional accuracy (0 to 1)
    """
    n = min(len(forecasts), len(actual_returns))
    if n < 10:
        return 0.5

    correct = 0
    for i in range(n):
        if forecasts[i] * actual_returns[i] > 0:
            correct += 1

    return round(correct / n, 4)


# ═══════════════════════════════════════════════════════════════
# Information Coefficient (Masters Ch. 8)
# ═══════════════════════════════════════════════════════════════

def compute_information_coefficient(
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
) -> float:
    """
    Spearman rank correlation between forecasts and actual returns.

    Masters Ch. 8: "The Information Coefficient quantifies the
    value added by a forecast beyond what random guessing provides."

    IC > 0.05 = signal has value
    IC > 0.10 = strong predictive power
    IC < 0.0  = anti-predictive (reverse the signal)

    Returns:
        IC value (-1 to +1)
    """
    n = min(len(forecasts), len(actual_returns))
    if n < 10:
        return 0.0

    f = forecasts[:n]
    a = actual_returns[:n]

    # Rank-based (Spearman)
    f_ranks = np.argsort(np.argsort(f)).astype(float)
    a_ranks = np.argsort(np.argsort(a)).astype(float)

    f_mean = np.mean(f_ranks)
    a_mean = np.mean(a_ranks)

    num = np.sum((f_ranks - f_mean) * (a_ranks - a_mean))
    den = math.sqrt(np.sum((f_ranks - f_mean) ** 2) * np.sum((a_ranks - a_mean) ** 2))

    if den < 1e-10:
        return 0.0

    return round(float(num / den), 4)


# ═══════════════════════════════════════════════════════════════
# Monte Carlo Permutation Test (Masters Ch. 6-7)
# ═══════════════════════════════════════════════════════════════

def monte_carlo_permutation_test(
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
    n_permutations: int = 2000,
    metric: str = "directional_accuracy",
) -> float:
    """
    Test if the forecast signal is statistically significant
    via permutation test.

    Masters Ch. 6: "We randomly shuffle the forecast-actual pairs
    and compute the same metric. The p-value is the fraction of
    random shuffles that beat the actual performance."

    Returns:
        p-value (< 0.05 = signal is significant)
    """
    n = min(len(forecasts), len(actual_returns))
    if n < 20:
        return 1.0

    f = forecasts[:n]
    a = actual_returns[:n]

    # Compute actual metric
    if metric == "directional_accuracy":
        actual_metric = compute_directional_accuracy(f, a)
    elif metric == "ic":
        actual_metric = abs(compute_information_coefficient(f, a))
    else:
        actual_metric = compute_directional_accuracy(f, a)

    # Permutation test
    rng = np.random.default_rng(42)
    n_better = 0

    for _ in range(n_permutations):
        shuffled_a = rng.permutation(a)
        if metric == "directional_accuracy":
            perm_metric = compute_directional_accuracy(f, shuffled_a)
        elif metric == "ic":
            perm_metric = abs(compute_information_coefficient(f, shuffled_a))
        else:
            perm_metric = compute_directional_accuracy(f, shuffled_a)

        if perm_metric >= actual_metric:
            n_better += 1

    return round((n_better + 1) / (n_permutations + 1), 4)


# ═══════════════════════════════════════════════════════════════
# Brier Score (Masters Ch. 5: Classification)
# ═══════════════════════════════════════════════════════════════

def compute_brier_score(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
) -> float:
    """
    Brier score for classification quality.

    Masters Ch. 5: "The Brier score is the mean squared error of
    probabilistic predictions against binary outcomes."

    0.0 = perfect calibration
    0.25 = no skill (always predict 50%)
    > 0.25 = worse than random

    Args:
        predicted_probs: predicted probabilities [0, 1]
        actual_outcomes: binary outcomes (0 or 1)

    Returns:
        Brier score (0 to 1)
    """
    n = min(len(predicted_probs), len(actual_outcomes))
    if n < 5:
        return 0.25

    return round(float(np.mean((predicted_probs[:n] - actual_outcomes[:n]) ** 2)), 4)


# ═══════════════════════════════════════════════════════════════
# R-Squared of Equity Curve
# ═══════════════════════════════════════════════════════════════

def compute_equity_r_squared(pnl_series: np.ndarray) -> float:
    """
    R² of the cumulative equity curve vs a straight line.

    Higher R² = smoother equity curve = more reliable prediction.
    R² > 0.80 = very smooth (strong system)
    R² > 0.60 = acceptable
    R² < 0.40 = choppy (unreliable predictions)
    """
    n = len(pnl_series)
    if n < 10:
        return 0.0

    equity = np.cumsum(pnl_series)
    x = np.arange(n, dtype=float)

    # Linear regression: equity = a × x + b
    x_mean = np.mean(x)
    eq_mean = np.mean(equity)

    ss_xy = np.sum((x - x_mean) * (equity - eq_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-10:
        return 0.0

    slope = ss_xy / ss_xx
    intercept = eq_mean - slope * x_mean

    fitted = slope * x + intercept
    ss_res = np.sum((equity - fitted) ** 2)
    ss_tot = np.sum((equity - eq_mean) ** 2)

    if ss_tot < 1e-10:
        return 0.0

    r2 = 1.0 - ss_res / ss_tot
    return round(max(0.0, r2), 4)


# ═══════════════════════════════════════════════════════════════
# Walk-Forward Validation (Masters Ch. 2-3)
# ═══════════════════════════════════════════════════════════════

def walk_forward_validate(
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
    n_splits: int = 5,
    train_ratio: float = 0.6,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward validation.

    Masters Ch. 2: "Walk-forward analysis is the gold standard for
    evaluating trading predictions. It prevents lookahead bias and
    measures true out-of-sample performance."

    Args:
        forecasts: full series of forecasts
        actual_returns: full series of actual returns
        n_splits: number of walk-forward windows
        train_ratio: initial train fraction

    Returns:
        WalkForwardResult with OOS metrics
    """
    n = min(len(forecasts), len(actual_returns))
    symbol = "PORTFOLIO"

    if n < 50:
        return WalkForwardResult(
            symbol=symbol, n_windows=0,
            oos_sharpe=0.0, oos_cagr=0.0,
            oos_directional_accuracy=0.5,
            is_degradation=0.0, walk_forward_efficiency=0.0,
        )

    initial_train = int(n * train_ratio)
    step = max(1, (n - initial_train) // n_splits)

    oos_returns_all = []
    is_sharpes = []
    oos_sharpes = []
    oos_da_all = []

    for i in range(n_splits):
        train_end = initial_train + i * step
        test_end = min(train_end + step, n)

        if train_end >= n or test_end <= train_end:
            break

        # In-sample metrics
        is_f = forecasts[:train_end]
        is_a = actual_returns[:train_end]
        # Signed returns: forecast_sign × actual
        is_pnl = np.sign(is_f) * is_a
        is_sharpe = float(np.mean(is_pnl) / max(np.std(is_pnl), 1e-10) * math.sqrt(252))

        # Out-of-sample
        oos_f = forecasts[train_end:test_end]
        oos_a = actual_returns[train_end:test_end]
        oos_pnl = np.sign(oos_f) * oos_a
        oos_sharpe = float(np.mean(oos_pnl) / max(np.std(oos_pnl), 1e-10) * math.sqrt(252))

        oos_da = compute_directional_accuracy(oos_f, oos_a)

        is_sharpes.append(is_sharpe)
        oos_sharpes.append(oos_sharpe)
        oos_returns_all.extend(oos_pnl.tolist())
        oos_da_all.append(oos_da)

    if not oos_sharpes:
        return WalkForwardResult(
            symbol=symbol, n_windows=0,
            oos_sharpe=0.0, oos_cagr=0.0,
            oos_directional_accuracy=0.5,
            is_degradation=0.0, walk_forward_efficiency=0.0,
        )

    avg_is = float(np.mean(is_sharpes))
    avg_oos = float(np.mean(oos_sharpes))
    oos_da = float(np.mean(oos_da_all))

    # CAGR from OOS returns
    oos_arr = np.array(oos_returns_all)
    total_ret = float(np.sum(oos_arr))
    n_days = len(oos_arr)
    oos_cagr = ((1 + total_ret) ** (252 / max(n_days, 1)) - 1) if n_days > 0 else 0.0

    # Walk-forward efficiency = OOS/IS (> 0.5 good, > 0.8 excellent)
    wfe = avg_oos / avg_is if abs(avg_is) > 0.01 else 0.0

    return WalkForwardResult(
        symbol=symbol,
        n_windows=len(oos_sharpes),
        oos_sharpe=round(avg_oos, 3),
        oos_cagr=round(oos_cagr, 4),
        oos_directional_accuracy=round(oos_da, 4),
        is_degradation=round(avg_is - avg_oos, 3),
        walk_forward_efficiency=round(max(0.0, wfe), 3),
    )


# ═══════════════════════════════════════════════════════════════
# Forecast Quality Gate (Masters Ch. 9)
# ═══════════════════════════════════════════════════════════════

def compute_prediction_quality(
    symbol: str,
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
    significance_level: float = 0.05,
    mc_permutations: int = 2000,
) -> PredictionQuality:
    """
    Comprehensive prediction quality assessment for a single signal.

    Masters Ch. 9: "Before committing capital, assess the quality
    of every prediction signal. Poor signals dilute good ones."

    Computes directional accuracy, IC, MC significance, Brier score,
    equity R², and a combined quality_score.

    The confidence_multiplier is used to dampen low-quality forecasts:
      quality > 0.7: multiplier = 1.0 (full confidence)
      quality 0.4-0.7: multiplier = 0.6 (dampen)
      quality < 0.4: multiplier = 0.3 (heavily suppress)

    Returns:
        PredictionQuality with all metrics
    """
    n = min(len(forecasts), len(actual_returns))

    if n < 20:
        return PredictionQuality(
            symbol=symbol, directional_accuracy=0.5,
            mean_abs_error=0.0, information_coefficient=0.0,
            r_squared=0.0, brier_score=0.25,
            monte_carlo_p_value=1.0, is_significant=False,
            quality_score=0.0, confidence_multiplier=0.3,
        )

    f = forecasts[:n]
    a = actual_returns[:n]

    # Directional accuracy
    da = compute_directional_accuracy(f, a)

    # Information coefficient
    ic = compute_information_coefficient(f, a)

    # MAE -- forecast magnitude vs actual (both normalized)
    f_norm = f / max(np.std(f), 1e-10)
    a_norm = a / max(np.std(a), 1e-10)
    mae = float(np.mean(np.abs(f_norm - a_norm)))

    # Brier score (convert to classification: positive forecast → 1)
    pred_probs = 1.0 / (1.0 + np.exp(-f))  # sigmoid(forecast)
    outcomes = (a > 0).astype(float)
    brier = compute_brier_score(pred_probs, outcomes)

    # Equity curve R²
    pnl = np.sign(f) * a
    r2 = compute_equity_r_squared(pnl)

    # Monte Carlo permutation test
    p_val = monte_carlo_permutation_test(f, a, mc_permutations)
    is_sig = p_val < significance_level

    # Combined quality score (0-1)
    # Weighted: DA (30%), IC (25%), R² (20%), MC significance (15%), Brier (10%)
    da_score = max(0.0, (da - 0.5) / 0.15)  # 0.5→0, 0.65→1
    ic_score = max(0.0, min(1.0, abs(ic) / 0.10))  # 0→0, 0.10→1
    r2_score = max(0.0, min(1.0, r2 / 0.60))  # 0→0, 0.60→1
    mc_score = 1.0 if is_sig else 0.0
    brier_score_norm = max(0.0, 1.0 - brier / 0.25)  # 0.25→0, 0→1

    quality = (0.30 * min(1.0, da_score)
               + 0.25 * ic_score
               + 0.20 * r2_score
               + 0.15 * mc_score
               + 0.10 * brier_score_norm)
    quality = round(max(0.0, min(1.0, quality)), 4)

    # Confidence multiplier
    if quality > 0.70:
        confidence = 1.0
    elif quality > 0.40:
        confidence = 0.6
    else:
        confidence = 0.3

    logger.info(
        "Quality %s: DA=%.3f IC=%.3f R²=%.3f MC_p=%.3f brier=%.3f → q=%.2f conf=%.1f",
        symbol, da, ic, r2, p_val, brier, quality, confidence,
    )

    return PredictionQuality(
        symbol=symbol,
        directional_accuracy=da,
        mean_abs_error=round(mae, 4),
        information_coefficient=ic,
        r_squared=r2,
        brier_score=brier,
        monte_carlo_p_value=p_val,
        is_significant=is_sig,
        quality_score=quality,
        confidence_multiplier=confidence,
    )


# ═══════════════════════════════════════════════════════════════
# Batch Processing for Pipeline Integration
# ═══════════════════════════════════════════════════════════════

def compute_prediction_quality_batch(
    forecast_dict: Dict[str, np.ndarray],
    return_dict: Dict[str, np.ndarray],
    significance_level: float = 0.05,
    mc_permutations: int = 1000,
) -> Dict[str, PredictionQuality]:
    """
    Compute prediction quality for multiple symbols.

    Uses fewer MC permutations (1000) in batch mode for speed.

    Args:
        forecast_dict: {symbol: array of daily forecasts}
        return_dict: {symbol: array of daily actual returns}

    Returns:
        {symbol: PredictionQuality}
    """
    results = {}

    for symbol in forecast_dict:
        if symbol not in return_dict:
            continue

        results[symbol] = compute_prediction_quality(
            symbol,
            forecast_dict[symbol],
            return_dict[symbol],
            significance_level,
            mc_permutations,
        )

    return results


def gate_forecasts_by_quality(
    raw_forecasts: Dict[str, float],
    quality_scores: Dict[str, PredictionQuality],
) -> Dict[str, float]:
    """
    Apply quality gate to raw forecasts.

    Masters Ch. 9: "Scale each forecast by its assessed quality.
    This automatically de-weights unreliable signals."

    Multiplies each forecast by its confidence_multiplier:
      high-quality: 1.0× (unchanged)
      medium-quality: 0.6× (dampened)
      low-quality: 0.3× (heavily suppressed)

    Returns:
        {symbol: quality-gated forecast}
    """
    gated = {}

    for symbol, forecast in raw_forecasts.items():
        quality = quality_scores.get(symbol)
        if quality:
            gated[symbol] = round(forecast * quality.confidence_multiplier, 4)
            if quality.confidence_multiplier < 1.0:
                logger.info(
                    "Quality gate %s: forecast %.1f → %.1f (conf=%.1f, q=%.2f)",
                    symbol, forecast,
                    forecast * quality.confidence_multiplier,
                    quality.confidence_multiplier,
                    quality.quality_score,
                )
        else:
            # No quality data → conservative dampening
            gated[symbol] = round(forecast * 0.5, 4)

    return gated
