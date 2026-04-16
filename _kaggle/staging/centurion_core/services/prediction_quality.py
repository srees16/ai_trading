"""
Prediction Quality Assessment — From Timothy Masters'
"Assessing and Improving Prediction and Classification".

Implements rigorous statistical methods for evaluating and improving
the quality of centurion_core's 19 forecast sources:

  1. ROC Curves — Receiver Operating Characteristic for buy/sell
  2. Confusion Matrix — TP/FP/TN/FN for directional accuracy
  3. Expected Gain/Loss — E[G] from confusion matrix × trade returns
  4. Permutation Test — shuffle signals to verify real edge exists
  5. Bootstrap CI — confidence intervals for Sharpe, CAGR
  6. Per-Signal Quality — individual metrics for each forecast source
  7. Ensemble Optimizer — data-driven weight rebalancing
  8. Information Metrics — mutual information for feature selection

Integration:
  - IND: auto-reweight forecast_combiner based on OOS quality → Kite
  - US: full quality report in API response → display on UI
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConfusionMatrix:
    """Trade-level confusion matrix for a single forecast source."""
    source_name: str
    true_positives: int = 0    # Predicted BUY, actual profit
    false_positives: int = 0   # Predicted BUY, actual loss
    true_negatives: int = 0    # Predicted HOLD/SELL, actual drop
    false_negatives: int = 0   # Predicted HOLD/SELL, actual rise

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / max(1, total)

    @property
    def precision(self) -> float:
        """Of all BUY signals, what fraction were profitable."""
        return self.true_positives / max(1, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Of all profitable moves, what fraction did we catch."""
        return self.true_positives / max(1, self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2.0 * p * r / max(1e-10, p + r)

    @property
    def hit_rate(self) -> float:
        """Alias for precision — trade-level win rate."""
        return self.precision


@dataclass
class ROCPoint:
    """Single point on the ROC curve."""
    threshold: float
    true_positive_rate: float   # Sensitivity / Recall
    false_positive_rate: float  # 1 - Specificity


@dataclass
class SignalQuality:
    """Complete quality assessment for a single forecast source."""
    source_name: str
    confusion: ConfusionMatrix
    roc_auc: float                     # Area under ROC curve (0.5 = random, 1.0 = perfect)
    sharpe_ratio: float                # Source-specific Sharpe (annualized)
    profit_factor: float               # Gross profit / gross loss
    expected_gain: float               # E[G] from confusion matrix × returns
    permutation_p_value: float         # p-value from permutation test (< 0.05 = real edge)
    bootstrap_sharpe_ci: Tuple[float, float]  # 95% CI for Sharpe
    optimal_threshold: float           # ROC-optimized buy threshold
    is_significant: bool               # permutation_p_value < 0.05


@dataclass
class PredictionQualityReport:
    """Full prediction quality report for all forecast sources."""
    signal_qualities: Dict[str, SignalQuality]
    recommended_weights: Dict[str, float]    # Data-driven optimal weights
    overall_ensemble_auc: float
    overall_sharpe: float
    overall_sharpe_ci: Tuple[float, float]


# ═══════════════════════════════════════════════════════════════
# Confusion Matrix Computation (Masters Ch. 2)
# ═══════════════════════════════════════════════════════════════

def compute_confusion_matrix(
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
    threshold: float = 0.0,
    source_name: str = "unknown",
) -> ConfusionMatrix:
    """
    Compute trade-level confusion matrix.

    Masters Ch. 2: "The confusion matrix is the most fundamental
    tool for assessing classification performance."

    Args:
        forecasts: array of forecast values (positive = BUY signal)
        actual_returns: array of forward returns (positive = profitable)
        threshold: forecast value above which we classify as BUY
        source_name: name of the forecast source

    Returns:
        ConfusionMatrix with TP/FP/TN/FN counts
    """
    n = min(len(forecasts), len(actual_returns))
    cm = ConfusionMatrix(source_name=source_name)

    for i in range(n):
        predicted_buy = forecasts[i] > threshold
        actual_profit = actual_returns[i] > 0

        if predicted_buy and actual_profit:
            cm.true_positives += 1
        elif predicted_buy and not actual_profit:
            cm.false_positives += 1
        elif not predicted_buy and not actual_profit:
            cm.true_negatives += 1
        else:
            cm.false_negatives += 1

    return cm


# ═══════════════════════════════════════════════════════════════
# ROC Curve (Masters Ch. 2)
# ═══════════════════════════════════════════════════════════════

def compute_roc_curve(
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[List[ROCPoint], float]:
    """
    Compute ROC curve and AUC for a forecast source.

    Masters Ch. 2: "ROC curves are the most powerful tool for
    assessing the quality of a binary classifier."

    Args:
        forecasts: forecast values
        actual_returns: forward returns (binary: profit/loss)
        n_thresholds: number of threshold points to evaluate

    Returns:
        (list of ROCPoints, AUC)
    """
    actual_binary = (actual_returns > 0).astype(int)
    n_pos = np.sum(actual_binary)
    n_neg = len(actual_binary) - n_pos

    if n_pos == 0 or n_neg == 0:
        return [], 0.5

    thresholds = np.linspace(np.min(forecasts) - 0.01,
                             np.max(forecasts) + 0.01, n_thresholds)

    roc_points = []
    for t in thresholds:
        predicted_pos = forecasts > t
        tp = np.sum(predicted_pos & (actual_binary == 1))
        fp = np.sum(predicted_pos & (actual_binary == 0))

        tpr = tp / max(1, n_pos)
        fpr = fp / max(1, n_neg)
        roc_points.append(ROCPoint(threshold=float(t),
                                   true_positive_rate=float(tpr),
                                   false_positive_rate=float(fpr)))

    # Sort by FPR for AUC computation
    roc_points.sort(key=lambda p: p.false_positive_rate)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(roc_points)):
        dx = roc_points[i].false_positive_rate - roc_points[i - 1].false_positive_rate
        avg_y = (roc_points[i].true_positive_rate + roc_points[i - 1].true_positive_rate) / 2.0
        auc += dx * avg_y

    return roc_points, float(auc)


def find_optimal_threshold(
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
) -> float:
    """
    Find the threshold that maximizes Expected Gain.

    Masters Ch. 2: "Maximizing the threshold for maximum total gain
    is generally the most appropriate criterion for traders."

    E[G] = TP × avg_win - FP × avg_loss
    """
    n_thresholds = 100
    thresholds = np.linspace(np.min(forecasts), np.max(forecasts), n_thresholds)
    best_gain = -np.inf
    best_threshold = 0.0

    for t in thresholds:
        predicted_buy = forecasts > t
        actual_profit = actual_returns > 0

        tp = np.sum(predicted_buy & actual_profit)
        fp = np.sum(predicted_buy & ~actual_profit)

        expected_gain = tp * avg_win - fp * avg_loss

        if expected_gain > best_gain:
            best_gain = expected_gain
            best_threshold = float(t)

    return best_threshold


# ═══════════════════════════════════════════════════════════════
# Permutation Test (Masters Ch. 5)
# ═══════════════════════════════════════════════════════════════

def permutation_test(
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
    n_permutations: int = 1000,
    metric: str = "sharpe",
) -> float:
    """
    Test whether a forecast source has a statistically significant
    edge vs random chance.

    Masters Ch. 5: "Permutation training and testing provides a
    rigorous, distribution-free test of predictive ability."

    Shuffles the alignment between forecasts and returns, computes
    the metric for each shuffled version, and returns the p-value.

    p < 0.05: forecast has real predictive edge
    p > 0.05: forecast may be random/overfit

    Args:
        forecasts: forecast values
        actual_returns: actual forward returns
        n_permutations: number of random shuffles
        metric: "sharpe" or "profit_factor"

    Returns:
        p-value (fraction of shuffled trials that beat actual)
    """
    actual_metric = _compute_metric(forecasts, actual_returns, metric)

    count_better = 0
    rng = np.random.default_rng(42)

    for _ in range(n_permutations):
        shuffled = rng.permutation(actual_returns)
        perm_metric = _compute_metric(forecasts, shuffled, metric)
        if perm_metric >= actual_metric:
            count_better += 1

    return count_better / n_permutations


def _compute_metric(
    forecasts: np.ndarray,
    returns: np.ndarray,
    metric: str,
) -> float:
    """Compute a scalar metric for forecast quality."""
    # Forecast-weighted returns: long when forecast>0, sized by forecast
    position = np.sign(forecasts)
    strategy_returns = position * returns

    if metric == "sharpe":
        if len(strategy_returns) < 2:
            return 0.0
        mean_r = np.mean(strategy_returns)
        std_r = np.std(strategy_returns, ddof=1)
        if std_r == 0:
            return 0.0
        return float(mean_r / std_r * math.sqrt(252))

    elif metric == "profit_factor":
        gains = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        gross_gain = np.sum(gains) if len(gains) > 0 else 0.0
        gross_loss = -np.sum(losses) if len(losses) > 0 else 1e-10
        return float(gross_gain / max(gross_loss, 1e-10))

    return 0.0


# ═══════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals (Masters Ch. 3)
# ═══════════════════════════════════════════════════════════════

def bootstrap_confidence_interval(
    returns: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    metric: str = "sharpe",
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a performance metric.

    Masters Ch. 3: "Bootstrap estimation is a versatile tool for
    computing confidence intervals without distributional assumptions."

    Args:
        returns: array of daily strategy returns
        n_bootstrap: number of bootstrap resamples
        confidence: confidence level (default 95%)
        metric: "sharpe", "cagr", or "max_dd"

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(42)
    metrics = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        sample = rng.choice(returns, size=len(returns), replace=True)
        metrics[b] = _bootstrap_metric(sample, metric)

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(metrics, 100 * alpha))
    upper = float(np.percentile(metrics, 100 * (1 - alpha)))

    return (lower, upper)


def _bootstrap_metric(returns: np.ndarray, metric: str) -> float:
    """Compute a single metric from a bootstrap sample."""
    if metric == "sharpe":
        if len(returns) < 2:
            return 0.0
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r == 0:
            return 0.0
        return float(mean_r / std_r * math.sqrt(252))

    elif metric == "cagr":
        cum = np.cumprod(1.0 + returns)
        if len(cum) == 0 or cum[-1] <= 0:
            return 0.0
        years = len(returns) / 252.0
        if years <= 0:
            return 0.0
        return float((cum[-1]) ** (1.0 / years) - 1.0) * 100.0

    elif metric == "max_dd":
        cum = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return float(np.min(dd)) * 100.0

    return 0.0


# ═══════════════════════════════════════════════════════════════
# Per-Signal Quality Assessment
# ═══════════════════════════════════════════════════════════════

def assess_signal_quality(
    source_name: str,
    forecasts: np.ndarray,
    actual_returns: np.ndarray,
    n_permutations: int = 500,
) -> SignalQuality:
    """
    Complete quality assessment for a single forecast source.

    Masters: "Each prediction source should be individually assessed
    before combining into an ensemble."

    Returns SignalQuality with ROC, confusion matrix, permutation test,
    bootstrap CI, and optimal threshold.
    """
    # ROC + AUC
    roc_points, auc = compute_roc_curve(forecasts, actual_returns)

    # Optimal threshold via Expected Gain
    avg_win = float(np.mean(actual_returns[actual_returns > 0])) if np.any(actual_returns > 0) else 0.0
    avg_loss = float(np.abs(np.mean(actual_returns[actual_returns < 0]))) if np.any(actual_returns < 0) else 0.0
    optimal_thresh = find_optimal_threshold(forecasts, actual_returns, avg_win, max(avg_loss, 1e-10))

    # Confusion matrix at optimal threshold
    cm = compute_confusion_matrix(forecasts, actual_returns, threshold=optimal_thresh, source_name=source_name)

    # Strategy returns for Sharpe and PF
    position = np.sign(forecasts)
    strategy_returns = position * actual_returns

    sharpe = 0.0
    if len(strategy_returns) > 1:
        std = np.std(strategy_returns, ddof=1)
        if std > 0:
            sharpe = float(np.mean(strategy_returns) / std * math.sqrt(252))

    # Profit factor
    gains = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    gross_gain = float(np.sum(gains)) if len(gains) > 0 else 0.0
    gross_loss = float(-np.sum(losses)) if len(losses) > 0 else 1e-10
    pf = gross_gain / max(gross_loss, 1e-10)

    # Expected gain
    eg = cm.true_positives * avg_win - cm.false_positives * avg_loss

    # Permutation test
    p_value = permutation_test(forecasts, actual_returns, n_permutations)

    # Bootstrap CI for Sharpe
    ci = bootstrap_confidence_interval(strategy_returns, n_bootstrap=1000, metric="sharpe")

    return SignalQuality(
        source_name=source_name,
        confusion=cm,
        roc_auc=auc,
        sharpe_ratio=sharpe,
        profit_factor=pf,
        expected_gain=eg,
        permutation_p_value=p_value,
        bootstrap_sharpe_ci=ci,
        optimal_threshold=optimal_thresh,
        is_significant=(p_value < 0.05),
    )


# ═══════════════════════════════════════════════════════════════
# Ensemble Weight Optimizer (Masters Ch. 6)
# ═══════════════════════════════════════════════════════════════

def optimize_ensemble_weights(
    signal_qualities: Dict[str, SignalQuality],
    min_weight: float = 0.01,
    max_weight: float = 0.25,
) -> Dict[str, float]:
    """
    Data-driven weight optimization based on quality metrics.

    Masters Ch. 6: "Constrained linear combinations of predictions
    can be optimized to maximize expected performance."

    Weighting formula:
      raw_w = AUC_excess × Sharpe × (1 if significant else 0.1)
      normalized to sum = 1.0, clipped to [min_weight, max_weight]

    Sources that fail permutation test (p > 0.05) get 90% penalty.
    """
    raw_weights = {}
    for name, sq in signal_qualities.items():
        auc_excess = max(0.0, sq.roc_auc - 0.5)  # Excess over random
        sharpe_factor = max(0.01, sq.sharpe_ratio)
        significance = 1.0 if sq.is_significant else 0.1

        raw_weights[name] = auc_excess * sharpe_factor * significance

    total = sum(raw_weights.values())
    if total <= 0:
        # Fallback: equal weights
        n = len(signal_qualities)
        return {name: 1.0 / max(1, n) for name in signal_qualities}

    # Normalize and clip
    weights = {}
    for name, rw in raw_weights.items():
        w = rw / total
        w = max(min_weight, min(max_weight, w))
        weights[name] = w

    # Re-normalize after clipping
    total_clipped = sum(weights.values())
    weights = {name: w / total_clipped for name, w in weights.items()}

    return weights


# ═══════════════════════════════════════════════════════════════
# Mutual Information (Masters Ch. 9)
# ═══════════════════════════════════════════════════════════════

def mutual_information(
    x: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute mutual information between predictor x and target y.

    Masters Ch. 9: "Mutual information quantifies the amount of
    information that one variable provides about another."

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Higher MI = more predictive power (but beware noise).
    """
    # Discretize into bins
    x_bins = np.digitize(x, np.linspace(np.min(x), np.max(x), n_bins))
    y_bins = np.digitize(y, np.linspace(np.min(y), np.max(y), n_bins))

    # Joint histogram
    n = len(x)
    joint = np.zeros((n_bins + 1, n_bins + 1))
    for i in range(n):
        joint[x_bins[i], y_bins[i]] += 1
    joint /= n

    # Marginals
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)

    mi = 0.0
    for ix in range(n_bins + 1):
        for iy in range(n_bins + 1):
            if joint[ix, iy] > 0 and px[ix] > 0 and py[iy] > 0:
                mi += joint[ix, iy] * math.log2(
                    joint[ix, iy] / (px[ix] * py[iy])
                )

    return float(mi)


# ═══════════════════════════════════════════════════════════════
# Full Quality Report
# ═══════════════════════════════════════════════════════════════

def generate_prediction_quality_report(
    forecast_history: Dict[str, np.ndarray],
    actual_returns: np.ndarray,
    n_permutations: int = 500,
) -> PredictionQualityReport:
    """
    Generate comprehensive prediction quality report for all sources.

    Args:
        forecast_history: {source_name: array of daily forecasts}
        actual_returns: array of actual daily forward returns

    Returns:
        PredictionQualityReport with per-signal and ensemble metrics
    """
    qualities = {}
    for name, forecasts in forecast_history.items():
        n = min(len(forecasts), len(actual_returns))
        if n < 30:
            logger.warning("Insufficient data for %s (%d bars), skipping", name, n)
            continue

        sq = assess_signal_quality(name, forecasts[:n], actual_returns[:n], n_permutations)
        qualities[name] = sq

        logger.info(
            "Signal %s: AUC=%.3f Sharpe=%.2f PF=%.2f p=%.3f %s thresh=%.2f",
            name, sq.roc_auc, sq.sharpe_ratio, sq.profit_factor,
            sq.permutation_p_value,
            "SIGNIFICANT" if sq.is_significant else "NOT SIG",
            sq.optimal_threshold,
        )

    # Optimize ensemble weights
    rec_weights = optimize_ensemble_weights(qualities) if qualities else {}

    # Overall ensemble metrics
    if qualities and len(actual_returns) > 30:
        # Combined forecast = weighted sum
        combined = np.zeros(len(actual_returns))
        for name, sq in qualities.items():
            if name in forecast_history:
                w = rec_weights.get(name, 0.0)
                n = min(len(forecast_history[name]), len(combined))
                combined[:n] += w * forecast_history[name][:n]

        _, ensemble_auc = compute_roc_curve(combined, actual_returns)
        position = np.sign(combined)
        ens_returns = position * actual_returns
        ens_std = np.std(ens_returns, ddof=1) if len(ens_returns) > 1 else 1.0
        ens_sharpe = float(np.mean(ens_returns) / max(ens_std, 1e-10) * math.sqrt(252))
        ens_ci = bootstrap_confidence_interval(ens_returns, metric="sharpe")
    else:
        ensemble_auc = 0.5
        ens_sharpe = 0.0
        ens_ci = (0.0, 0.0)

    return PredictionQualityReport(
        signal_qualities=qualities,
        recommended_weights=rec_weights,
        overall_ensemble_auc=ensemble_auc,
        overall_sharpe=ens_sharpe,
        overall_sharpe_ci=ens_ci,
    )
