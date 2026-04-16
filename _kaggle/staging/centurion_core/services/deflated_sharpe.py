"""
Deflated Sharpe Ratio (DSR) — de Prado AFML Ch.14.

Adjusts the Sharpe ratio for multiple testing when selecting among
strategies or parameter sets.

Key concepts:
  - Probabilistic Sharpe Ratio (PSR): P(true SR > SR*)
  - Deflated Sharpe Ratio (DSR): PSR with SR* = expected max SR under null
  - Minimum Backtest Length (MinBTL): required n for reliable SR estimate

Usage:
    from services.deflated_sharpe import deflated_sharpe_ratio, min_backtest_length
    dsr_pvalue = deflated_sharpe_ratio(observed_sr, n_obs, n_trials)
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark_sr).

    de Prado (2018) Eq. 14.1:
        PSR = Φ[(SR - SR*) × √(n-1) / √(1 - γ₃·SR + (γ₄-1)/4 · SR²)]

    Parameters
    ----------
    observed_sr : annualized Sharpe ratio
    benchmark_sr : threshold SR (SR*)
    n_obs : number of observations
    skewness : sample skewness (γ₃)
    kurtosis : sample kurtosis (γ₄, NOT excess kurtosis; normal = 3.0)

    Returns
    -------
    p-value in [0, 1]. Higher = more confident SR exceeds benchmark.
    """
    if n_obs <= 2:
        return 0.0

    num = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1)
    denom_sq = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2
    if denom_sq <= 0:
        return 0.0

    z = num / math.sqrt(denom_sq)
    return float(sp_stats.norm.cdf(z))


def expected_max_sr(
    n_trials: int,
    mean_sr: float = 0.0,
    std_sr: float = 1.0,
) -> float:
    """Expected maximum Sharpe ratio under the null hypothesis.

    de Prado (2018) Eq. 14.4:
        E[max(SR)] ≈ std_sr × [(1 - γ) × Φ⁻¹(1 - 1/N) + γ × Φ⁻¹(1 - 1/(N×e))]

    where γ ≈ 0.5772 (Euler-Mascheroni constant).

    Simplified approximation for large N:
        E[max(SR)] ≈ std_sr × √(2 × ln(N))

    Parameters
    ----------
    n_trials : number of strategies/parameter sets tested
    mean_sr : mean SR under null (typically 0)
    std_sr : std of SR estimates under null

    Returns
    -------
    Expected maximum SR.
    """
    if n_trials <= 1:
        return mean_sr

    gamma = 0.5772156649  # Euler-Mascheroni constant

    z1 = sp_stats.norm.ppf(1.0 - 1.0 / n_trials) if n_trials > 1 else 0.0
    z2 = sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)) if n_trials > 1 else 0.0

    e_max = (1.0 - gamma) * z1 + gamma * z2
    return mean_sr + std_sr * e_max


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_std: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio: PSR with SR* = E[max(SR)] under null.

    Combines the expected maximum SR from multiple testing (Eq 14.4) with
    the Probabilistic Sharpe Ratio (Eq 14.1) to produce a corrected p-value.

    Parameters
    ----------
    observed_sr : annualized Sharpe of the selected strategy
    n_obs : number of return observations
    n_trials : number of strategies/parameter sets tested
    skewness : sample skewness of returns
    kurtosis : sample kurtosis of returns (normal = 3.0)
    sr_std : standard deviation of SR estimates under null

    Returns
    -------
    DSR p-value in [0, 1]. Values > 0.95 indicate the SR is likely genuine.
    Values < 0.05 indicate the SR is likely due to multiple testing luck.
    """
    sr_star = expected_max_sr(n_trials, mean_sr=0.0, std_sr=sr_std)
    return probabilistic_sharpe_ratio(observed_sr, sr_star, n_obs, skewness, kurtosis)


def min_backtest_length(
    target_sr: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> int:
    """Minimum Backtest Length to trust a given SR at a confidence level.

    de Prado (2018) Eq. 14.2:
        MinBTL = 1 + [1 - γ₃·SR + (γ₄-1)/4 · SR²] × (z_α / SR)²

    Parameters
    ----------
    target_sr : annualized Sharpe ratio to verify
    skewness : expected skewness
    kurtosis : expected kurtosis
    confidence : confidence level (default 0.95)

    Returns
    -------
    Minimum number of observations (e.g., trading days).
    """
    if abs(target_sr) < 1e-6:
        return 99999  # infinite for SR=0

    z_alpha = sp_stats.norm.ppf(confidence)
    bracket = 1.0 - skewness * target_sr + ((kurtosis - 1.0) / 4.0) * target_sr ** 2
    n = 1.0 + bracket * (z_alpha / target_sr) ** 2
    return max(10, int(math.ceil(n)))


def compute_dsr_for_strategies(
    sharpe_ratios: dict,
    n_obs: int,
    returns_stats: Optional[dict] = None,
) -> dict:
    """Compute DSR p-values for a set of strategies.

    Parameters
    ----------
    sharpe_ratios : {strategy_name: annualized_sharpe}
    n_obs : common observation count
    returns_stats : optional {strategy_name: {"skewness": float, "kurtosis": float}}

    Returns
    -------
    {strategy_name: {"sharpe": float, "dsr_pvalue": float, "dsr_significant": bool}}
    """
    n_trials = len(sharpe_ratios)
    if n_trials == 0:
        return {}

    results = {}
    for name, sr in sharpe_ratios.items():
        skew = 0.0
        kurt = 3.0
        if returns_stats and name in returns_stats:
            skew = returns_stats[name].get("skewness", 0.0)
            kurt = returns_stats[name].get("kurtosis", 3.0)

        dsr_p = deflated_sharpe_ratio(
            observed_sr=sr,
            n_obs=n_obs,
            n_trials=n_trials,
            skewness=skew,
            kurtosis=kurt,
        )

        results[name] = {
            "sharpe": round(sr, 4),
            "dsr_pvalue": round(dsr_p, 4),
            "dsr_significant": dsr_p >= 0.95,
            "min_btl": min_backtest_length(sr, skew, kurt),
        }

    return results
