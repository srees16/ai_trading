"""
Aronson Evidence-Based Technical Analysis — Statistical Validation Module.

Implements the core statistical tests from David Aronson's EBTA framework:
  1. Detrended (zero-centred) returns for unbiased signal evaluation
  2. Per-signal t-statistic gating (t >= 2.0 ↔ p < 0.05)
  3. Benjamini-Hochberg FDR control across multiple signals
  4. White's Reality Check (bootstrap null for best-of-N)
  5. Data-mining bias estimation: σ√(2·ln(N))
  6. Trimmed (winsorized) performance metrics
  7. Signal fire count & minimum sample size ramp
  8. Composite confidence score

Reference: Aronson, D.R. (2006). Evidence-Based Technical Analysis.
           Chapters 1, 5, 6, 8, 9.

Usage:
    from services.aronson_validator import (
        AronsonValidator, SignalValidation, ValidationSummary,
    )
    validator = AronsonValidator()
    summary = validator.validate_signals(signal_returns_dict, benchmark_returns)
"""

import logging
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ── Persistence ──────────────────────────────────────────────
_VALIDATION_STATE_PATH = Path("data") / "signal_validation_state.json"


# ══════════════════════════════════════════════════════════════
#  Data Classes
# ══════════════════════════════════════════════════════════════

@dataclass
class SignalValidation:
    """Validation result for a single forecast source."""
    name: str
    t_stat: float = 0.0
    p_value: float = 1.0
    bh_adjusted_p: float = 1.0
    bh_significant: bool = False       # Survives BH FDR at q-level
    n_fires: int = 0
    sample_size_ok: bool = True        # n_fires >= 30
    sample_ramp: float = 1.0           # min(1.0, n_fires/30)
    mean_return: float = 0.0
    trimmed_mean_return: float = 0.0
    trimmed_sharpe: float = 0.0
    detrended_sharpe: float = 0.0
    weight_multiplier: float = 1.0     # Combined penalty: t-stat × BH × sample-ramp × degradation

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "bh_adjusted_p": round(self.bh_adjusted_p, 6),
            "bh_significant": self.bh_significant,
            "n_fires": self.n_fires,
            "sample_size_ok": self.sample_size_ok,
            "sample_ramp": round(self.sample_ramp, 3),
            "trimmed_sharpe": round(self.trimmed_sharpe, 4),
            "detrended_sharpe": round(self.detrended_sharpe, 4),
            "weight_multiplier": round(self.weight_multiplier, 4),
        }


@dataclass
class ValidationSummary:
    """Aggregated validation results for all signals."""
    signals: List[SignalValidation] = field(default_factory=list)
    wrc_best_p_value: float = 1.0          # White's Reality Check p for best signal
    wrc_best_signal: str = ""
    dm_bias_estimate: float = 0.0
    n_validated: int = 0
    n_total: int = 0
    confidence_threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "n_validated": self.n_validated,
            "n_total": self.n_total,
            "wrc_best_signal": self.wrc_best_signal,
            "wrc_best_p_value": round(self.wrc_best_p_value, 6),
            "dm_bias_estimate_pct": round(self.dm_bias_estimate * 100, 2),
            "signals": [s.to_dict() for s in self.signals],
        }

    def get_weight_multipliers(self) -> Dict[str, float]:
        """Return {signal_name: weight_multiplier} for forecast_combiner."""
        return {s.name: s.weight_multiplier for s in self.signals}

    def get_validated_set(self) -> set:
        """Return set of signal names that passed validation."""
        return {s.name for s in self.signals if s.bh_significant and s.sample_size_ok}


# ══════════════════════════════════════════════════════════════
#  Core Statistical Functions
# ══════════════════════════════════════════════════════════════

def detrend_returns(returns: pd.Series, window: int = 252) -> pd.Series:
    """Zero-centre returns by subtracting the expanding (or rolling) mean.

    Aronson Ch 1: Detrending eliminates the confounding of signal timing
    skill with the market's secular trend.  Uses expanding mean for short
    histories, rolling 252-day mean once enough data is available.
    """
    if len(returns) < 30:
        return returns - returns.mean()
    rolling_mean = returns.rolling(window=min(window, len(returns)), min_periods=30).mean()
    rolling_mean = rolling_mean.ffill().bfill()
    return returns - rolling_mean


def compute_signal_tstat(
    signal_returns: np.ndarray,
    min_obs: int = 10,
) -> Tuple[float, float]:
    """Compute t-statistic and p-value for a signal's mean excess return.

    H0: mean excess return = 0 (signal has no predictive power).
    Aronson Ch 5: t = mean / (std / sqrt(N)).

    Returns (t_stat, p_value).  Two-sided test.
    """
    arr = np.asarray(signal_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < min_obs:
        return 0.0, 1.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return 0.0, 1.0
    t = mean / (std / math.sqrt(n))
    p = float(2.0 * sp_stats.t.sf(abs(t), df=n - 1))  # two-sided
    return t, p


def benjamini_hochberg(
    pvalues: List[Tuple[str, float]],
    q: float = 0.10,
) -> List[Tuple[str, float, float, bool]]:
    """Benjamini-Hochberg FDR procedure.

    Aronson Ch 6: Controls the expected proportion of false discoveries
    among rejected hypotheses at level q.

    Parameters
    ----------
    pvalues : list of (name, raw_p_value)
    q : FDR level (default 0.10)

    Returns
    -------
    list of (name, raw_p, adjusted_p, significant)
    """
    if not pvalues:
        return []
    # Sort ascending by p-value
    sorted_pv = sorted(pvalues, key=lambda x: x[1])
    m = len(sorted_pv)
    results = []
    prev_adj = 0.0
    adj_ps = [0.0] * m

    # Compute adjusted p-values (step-up from largest to smallest)
    for i in range(m - 1, -1, -1):
        name, raw_p = sorted_pv[i]
        rank = i + 1
        adj = raw_p * m / rank
        if i < m - 1:
            adj = min(adj, adj_ps[i + 1])
        adj = min(adj, 1.0)
        adj_ps[i] = adj

    for i, (name, raw_p) in enumerate(sorted_pv):
        results.append((name, raw_p, adj_ps[i], adj_ps[i] <= q))

    # Re-sort by original order would require tracking — return sorted by p
    return results


def whites_reality_check(
    signal_returns_matrix: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Tuple[float, int]:
    """White's Reality Check for data snooping.

    Aronson Ch 6: Constructs the null distribution of the maximum
    performance statistic across N signals using bootstrap resampling.

    Parameters
    ----------
    signal_returns_matrix : np.ndarray, shape (N_signals, T_days)
        Each row is a signal's daily returns.
    n_bootstrap : int
        Number of bootstrap replications.

    Returns
    -------
    (corrected_p_value, best_signal_index)
    """
    rng = np.random.RandomState(seed)
    n_signals, T = signal_returns_matrix.shape
    if T < 30 or n_signals < 2:
        return 1.0, 0

    # Observed: mean return of each signal
    observed_means = signal_returns_matrix.mean(axis=1)
    best_idx = int(np.argmax(observed_means))
    observed_best = observed_means[best_idx]

    # Centre each signal's returns (impose H0: all signals have mean 0)
    centred = signal_returns_matrix - observed_means[:, np.newaxis]

    # Bootstrap: resample time indices with replacement
    best_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, T, size=T)
        boot_means = centred[:, idx].mean(axis=1)
        best_stats[b] = boot_means.max()

    # p-value: fraction of bootstrap max-stats >= observed max-stat
    p_value = float(np.mean(best_stats >= observed_best))
    return p_value, best_idx


def estimate_data_mining_bias(
    n_signals: int,
    sigma_best: float,
) -> float:
    """Estimate data-mining bias for best-of-N selection.

    Aronson Ch 6: bias ≈ σ · √(2 · ln(N))
    where σ is the std of the best signal's metric distribution.

    Returns estimated bias (same units as sigma_best).
    """
    if n_signals <= 1 or sigma_best <= 0:
        return 0.0
    return sigma_best * math.sqrt(2.0 * math.log(n_signals))


def trimmed_sharpe(
    returns: np.ndarray,
    trim_pct: float = 0.05,
    annualisation: float = 252.0,
) -> float:
    """Sharpe ratio computed on winsorized returns (top/bottom trim_pct removed).

    Aronson Ch 5: Trimmed mean is more robust to heavy tails.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 10:
        return 0.0
    k = max(1, int(n * trim_pct))
    sorted_ret = np.sort(arr)
    trimmed = sorted_ret[k: n - k]
    if len(trimmed) < 5:
        return 0.0
    mean_t = float(np.mean(trimmed))
    std_t = float(np.std(trimmed, ddof=1))
    if std_t < 1e-12:
        return 0.0
    return (mean_t / std_t) * math.sqrt(annualisation)


def count_signal_fires(
    signal_series: np.ndarray,
    threshold: float = 0.0,
) -> int:
    """Count the number of times a signal transitions (changes sign or crosses threshold).

    Aronson Ch 5: Minimum ~30 independent fires needed for reliable statistics.
    """
    arr = np.asarray(signal_series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0
    signs = np.sign(arr - threshold)
    changes = np.diff(signs)
    return int(np.count_nonzero(changes))


def compute_confidence_score(
    forecasts: Dict[str, float],
    validated_set: set,
) -> float:
    """Fraction of statistically validated signals that agree on direction.

    Aronson Ch 9: Combining multiple uncorrelated, validated signals
    provides stronger evidence than any single signal.

    Returns confidence in [0, 1].  Direction = sign of combined forecast.
    """
    if not forecasts or not validated_set:
        return 0.0

    validated_forecasts = {k: v for k, v in forecasts.items()
                          if k in validated_set and abs(v) > 0.5}
    n = len(validated_forecasts)
    if n == 0:
        return 0.0

    # Determine majority direction
    n_long = sum(1 for v in validated_forecasts.values() if v > 0)
    n_short = n - n_long
    agreement = max(n_long, n_short) / n
    return agreement


# ══════════════════════════════════════════════════════════════
#  Main Validator Class
# ══════════════════════════════════════════════════════════════

class AronsonValidator:
    """Orchestrates all Aronson EBTA validation checks on a set of signals."""

    def __init__(
        self,
        fdr_q: float = 0.10,
        min_tstat: float = 2.0,
        min_fires: int = 30,
        trim_pct: float = 0.05,
        wrc_n_bootstrap: int = 5000,
        tstat_penalty: float = 0.5,     # Weight multiplier for t < min_tstat
        bh_fail_penalty: float = 0.25,  # Weight multiplier for BH-insignificant
    ):
        self.fdr_q = fdr_q
        self.min_tstat = min_tstat
        self.min_fires = min_fires
        self.trim_pct = trim_pct
        self.wrc_n_bootstrap = wrc_n_bootstrap
        self.tstat_penalty = tstat_penalty
        self.bh_fail_penalty = bh_fail_penalty

    def validate_signals(
        self,
        signal_returns: Dict[str, np.ndarray],
        benchmark_returns: Optional[np.ndarray] = None,
        signal_series: Optional[Dict[str, np.ndarray]] = None,
        degradation_ratios: Optional[Dict[str, float]] = None,
    ) -> ValidationSummary:
        """Run full Aronson validation pipeline on all signals.

        Parameters
        ----------
        signal_returns : dict
            {signal_name: array of daily returns when signal is active}
        benchmark_returns : array, optional
            Market benchmark returns for detrending.
        signal_series : dict, optional
            {signal_name: full signal output series} for fire counting.
        degradation_ratios : dict, optional
            {signal_name: OOS/IS Sharpe ratio from walk-forward}

        Returns
        -------
        ValidationSummary with per-signal stats and aggregate results.
        """
        summary = ValidationSummary()
        names = sorted(signal_returns.keys())
        summary.n_total = len(names)

        if not names:
            return summary

        # 1. Per-signal t-statistic & basic metrics
        pvalues_for_bh = []
        validations = {}

        for name in names:
            rets = np.asarray(signal_returns[name], dtype=float)
            rets = rets[np.isfinite(rets)]

            sv = SignalValidation(name=name)

            # Detrend if benchmark provided
            if benchmark_returns is not None:
                bm = np.asarray(benchmark_returns, dtype=float)
                bm = bm[np.isfinite(bm)]
                min_len = min(len(rets), len(bm))
                if min_len > 10:
                    rets_dt = rets[:min_len] - bm[:min_len]
                    sv.detrended_sharpe = trimmed_sharpe(rets_dt, self.trim_pct)
                else:
                    rets_dt = rets
            else:
                rets_dt = rets

            # t-stat on detrended returns
            sv.t_stat, sv.p_value = compute_signal_tstat(rets_dt)
            sv.mean_return = float(np.mean(rets)) if len(rets) > 0 else 0.0

            # Trimmed Sharpe on raw returns
            sv.trimmed_sharpe = trimmed_sharpe(rets, self.trim_pct)
            sv.trimmed_mean_return = float(
                np.mean(np.sort(rets)[max(1, int(len(rets) * self.trim_pct)):
                                       len(rets) - max(1, int(len(rets) * self.trim_pct))])
            ) if len(rets) > 10 else sv.mean_return

            # Signal fire count
            if signal_series and name in signal_series:
                sv.n_fires = count_signal_fires(signal_series[name])
            else:
                sv.n_fires = max(10, len(rets) // 5)  # rough estimate

            sv.sample_size_ok = sv.n_fires >= self.min_fires
            sv.sample_ramp = min(1.0, sv.n_fires / self.min_fires)

            pvalues_for_bh.append((name, sv.p_value))
            validations[name] = sv

        # 2. BH FDR correction
        bh_results = benjamini_hochberg(pvalues_for_bh, self.fdr_q)
        for name, raw_p, adj_p, sig in bh_results:
            validations[name].bh_adjusted_p = adj_p
            validations[name].bh_significant = sig

        # 3. White's Reality Check
        if len(names) >= 2:
            T = min(len(signal_returns[n]) for n in names)
            if T >= 30:
                matrix = np.array([
                    np.asarray(signal_returns[n][:T], dtype=float)
                    for n in names
                ])
                wrc_p, wrc_idx = whites_reality_check(matrix, self.wrc_n_bootstrap)
                summary.wrc_best_p_value = wrc_p
                summary.wrc_best_signal = names[wrc_idx]

        # 4. Data-mining bias
        if len(names) >= 2:
            best_name = max(validations, key=lambda n: validations[n].trimmed_sharpe)
            best_rets = np.asarray(signal_returns[best_name], dtype=float)
            best_std = float(np.std(best_rets[np.isfinite(best_rets)], ddof=1))
            summary.dm_bias_estimate = estimate_data_mining_bias(len(names), best_std)

        # 5. Compute weight multipliers
        for name, sv in validations.items():
            mult = 1.0

            # t-stat penalty
            if abs(sv.t_stat) < self.min_tstat:
                mult *= self.tstat_penalty

            # BH penalty
            if not sv.bh_significant:
                mult *= self.bh_fail_penalty

            # Sample ramp
            mult *= sv.sample_ramp

            # Degradation penalty
            if degradation_ratios and name in degradation_ratios:
                deg = degradation_ratios[name]
                if deg < 0.5:
                    mult *= max(0.1, deg)

            sv.weight_multiplier = round(mult, 4)

        summary.signals = sorted(validations.values(), key=lambda s: -s.weight_multiplier)
        summary.n_validated = sum(1 for s in summary.signals
                                  if s.bh_significant and s.sample_size_ok)
        return summary

    def compute_confidence_for_symbol(
        self,
        forecasts: Dict[str, float],
        validation_summary: Optional[ValidationSummary] = None,
    ) -> float:
        """Compute composite confidence score for a single symbol's forecasts.

        Parameters
        ----------
        forecasts : dict
            {signal_name: forecast_value} for this symbol.
        validation_summary : ValidationSummary, optional
            If provided, only count validated signals.

        Returns
        -------
        float in [0, 1].
        """
        if validation_summary is not None:
            validated = validation_summary.get_validated_set()
        else:
            validated = set(forecasts.keys())  # trust all if no validation
        return compute_confidence_score(forecasts, validated)

    # ── Persistence ──────────────────────────────────────────

    def save_state(self, summary: ValidationSummary) -> None:
        """Persist validation results to JSON."""
        try:
            _VALIDATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_VALIDATION_STATE_PATH, "w") as f:
                json.dump(summary.to_dict(), f, indent=2)
            logger.info("Saved Aronson validation state → %s", _VALIDATION_STATE_PATH)
        except Exception as exc:
            logger.warning("Failed to save validation state: %s", exc)

    @staticmethod
    def load_state() -> Optional[ValidationSummary]:
        """Load persisted validation results."""
        try:
            if not _VALIDATION_STATE_PATH.exists():
                return None
            with open(_VALIDATION_STATE_PATH) as f:
                data = json.load(f)
            summary = ValidationSummary(
                n_validated=data.get("n_validated", 0),
                n_total=data.get("n_total", 0),
                wrc_best_signal=data.get("wrc_best_signal", ""),
                wrc_best_p_value=data.get("wrc_best_p_value", 1.0),
                dm_bias_estimate=data.get("dm_bias_estimate_pct", 0.0) / 100.0,
            )
            for sd in data.get("signals", []):
                sv = SignalValidation(
                    name=sd["name"],
                    t_stat=sd.get("t_stat", 0),
                    bh_adjusted_p=sd.get("bh_adjusted_p", 1.0),
                    bh_significant=sd.get("bh_significant", False),
                    n_fires=sd.get("n_fires", 0),
                    sample_size_ok=sd.get("sample_size_ok", True),
                    sample_ramp=sd.get("sample_ramp", 1.0),
                    trimmed_sharpe=sd.get("trimmed_sharpe", 0.0),
                    detrended_sharpe=sd.get("detrended_sharpe", 0.0),
                    weight_multiplier=sd.get("weight_multiplier", 1.0),
                )
                summary.signals.append(sv)
            return summary
        except Exception as exc:
            logger.warning("Failed to load validation state: %s", exc)
            return None

    @staticmethod
    def load_weight_multipliers() -> Dict[str, float]:
        """Quick-load just the weight multipliers from persisted state."""
        summary = AronsonValidator.load_state()
        if summary is None:
            return {}
        return summary.get_weight_multipliers()
