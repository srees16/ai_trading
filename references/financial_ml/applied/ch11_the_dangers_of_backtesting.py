"""
Chapter 11: The Dangers of Backtesting
Advances in Financial Machine Learning – Marcos Lopez de Prado

This chapter is primarily theoretical, warning against backtesting pitfalls
such as selection bias, multiple testing, and backtest overfitting.

Snippet 11.1 is Marcos' Second Law of Backtesting (a quote).

The quantitative tools discussed in the chapter and its references
(Bailey & Lopez de Prado [2014b], Bailey et al. [2017a]) are implemented
here as utility functions:

  - expected_max_sharpe_ratio :  E[max{SR}] from N independent trials
  - deflated_sharpe_ratio     :  Deflated Sharpe Ratio (DSR)
  - prob_backtest_overfitting  :  PBO via Combinatorially Symmetric
                                  Cross-Validation (CSCV)
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_returns

# ======================================================================
# Snippet 11.1 – Marcos' Second Law of Backtesting
# ======================================================================
SECOND_LAW = (
    "Backtesting while researching is like drinking and driving. "
    "Do not research under the influence of a backtest."
    "\n  — Marcos Lopez de Prado, "
    "Advances in Financial Machine Learning (2018)"
)

# ======================================================================
# Expected Maximum Sharpe Ratio  (Bailey & Lopez de Prado [2014b])
# ======================================================================

def expected_max_sharpe_ratio(num_trials, mean_sr=0.0, std_sr=1.0):
    """Approximate E[max{SR_n}] for *num_trials* independent strategies.

    Uses the Euler–Mascheroni approximation for the expected maximum of
    N iid standard-normal draws, then rescales by mean/std of SR:

        E[max(Z)] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e))

    Parameters
    ----------
    num_trials : int   – number of independent strategy trials (N).
    mean_sr    : float – assumed mean of the Sharpe-ratio distribution.
    std_sr     : float – assumed std of the Sharpe-ratio distribution.

    Returns
    -------
    float – E[max{SR}] under the given distribution.
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if num_trials == 1:
        return mean_sr

    euler_mascheroni = 0.5772156649015329  # γ
    # Quantiles of the standard normal
    z1 = norm.ppf(1.0 - 1.0 / num_trials)
    z2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
    e_max_z = (1 - euler_mascheroni) * z1 + euler_mascheroni * z2
    return mean_sr + std_sr * e_max_z


# ======================================================================
# Estimated variance of the Sharpe Ratio (Lo [2002] / Bailey & LdP)
# ======================================================================

def sharpe_ratio_variance(sr_hat, n_obs, skew=0.0, kurtosis=3.0):
    """Estimated variance of the Sharpe-ratio estimator.

    Var[SR] ≈ (1/(T-1)) * [1 - γ₃*SR + ((γ₄ - 1)/4)*SR²]

    where γ₃ = skewness, γ₄ = kurtosis of returns.

    Parameters
    ----------
    sr_hat   : float – observed (annualised) Sharpe ratio.
    n_obs    : int   – number of return observations (T).
    skew     : float – skewness of returns.
    kurtosis : float – kurtosis of returns (excess kurtosis + 3).

    Returns
    -------
    float – estimated variance of the SR estimator.
    """
    return (1.0 / (n_obs - 1)) * (
        1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat ** 2
    )


# ======================================================================
# Deflated Sharpe Ratio  (Bailey & Lopez de Prado [2014b])
# ======================================================================

def deflated_sharpe_ratio(sr_hat, sr_benchmark, n_obs,
                          skew=0.0, kurtosis=3.0):
    """Compute the Deflated Sharpe Ratio (DSR).

    DSR = Φ[ (SR̂ - SR₀) / √Var[SR̂] ]

    where SR₀ is the benchmark (e.g. expected max SR from N trials).

    A DSR close to 1.0 means the observed SR is very likely genuine;
    close to 0.5 means it is no better than luck; below 0.5 means
    the strategy likely underperforms the null.

    Parameters
    ----------
    sr_hat        : float – observed Sharpe ratio.
    sr_benchmark  : float – benchmark SR, e.g. expected_max_sharpe_ratio().
    n_obs         : int   – number of return observations.
    skew          : float – skewness of strategy returns.
    kurtosis      : float – kurtosis of strategy returns (not excess).

    Returns
    -------
    float – probability that the observed SR exceeds the benchmark.
    """
    sr_var = sharpe_ratio_variance(sr_hat, n_obs, skew, kurtosis)
    if sr_var <= 0:
        return 0.0
    dsr = norm.cdf((sr_hat - sr_benchmark) / np.sqrt(sr_var))
    return dsr


# ======================================================================
# Probability of Backtest Overfitting  (Bailey et al. [2017a])
# via Combinatorially Symmetric Cross-Validation (CSCV)
# ======================================================================

def prob_backtest_overfitting(pnl_matrix, n_partitions=10,
                              metric='sharpe', max_combos=5000):
    """Estimate the Probability of Backtest Overfitting (PBO) via CSCV.

    Parameters
    ----------
    pnl_matrix   : DataFrame – (T x N) matrix of PnL or returns from N trials.
    n_partitions : int – even number S of row-partitions (default 10).
    metric       : str – 'sharpe' or 'total_return'.
    max_combos   : int – cap on combinations to evaluate (for speed).

    Returns
    -------
    dict with keys:
        'pbo'     – float, estimated probability of backtest overfitting.
        'logits'  – array of rank-logit values across combinations.
        'phi_oos' – array of OOS relative-rank values.
    """
    if n_partitions % 2 != 0:
        raise ValueError("n_partitions must be even")

    M = pnl_matrix.values if isinstance(pnl_matrix, pd.DataFrame) else pnl_matrix
    T, N = M.shape

    # Step 1: partition rows into S disjoint blocks
    S = n_partitions
    block_size = T // S
    blocks = [M[i * block_size:(i + 1) * block_size, :] for i in range(S)]

    # Step 2: form all C(S, S/2) combinations for training set
    half = S // 2
    all_combos = list(combinations(range(S), half))

    # Subsample if too many combinations
    rng = np.random.default_rng(42)
    if len(all_combos) > max_combos:
        idx = rng.choice(len(all_combos), size=max_combos, replace=False)
        all_combos = [all_combos[i] for i in sorted(idx)]

    logits = []
    phi_oos_list = []

    for combo in all_combos:
        complement = tuple(i for i in range(S) if i not in combo)

        # Build training and testing sets
        J = np.concatenate([blocks[i] for i in combo], axis=0)
        J_bar = np.concatenate([blocks[i] for i in complement], axis=0)

        # Compute performance metric for each strategy
        if metric == 'sharpe':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu_is = J.mean(axis=0)
                sd_is = J.std(axis=0, ddof=1)
                sd_is[sd_is == 0] = 1e-10
                R = mu_is / sd_is

                mu_oos = J_bar.mean(axis=0)
                sd_oos = J_bar.std(axis=0, ddof=1)
                sd_oos[sd_oos == 0] = 1e-10
                R_bar = mu_oos / sd_oos
        else:  # total_return
            R = J.sum(axis=0)
            R_bar = J_bar.sum(axis=0)

        # Select best in-sample strategy
        n_star = np.argmax(R)

        # Relative rank of n* in OOS performance
        oos_perf = R_bar[n_star]
        rank = np.sum(R_bar <= oos_perf) / N
        # Clamp to avoid log(0)
        rank = np.clip(rank, 1e-6, 1.0 - 1e-6)
        phi_oos_list.append(rank)

        # Logit transformation
        lam = np.log(rank / (1.0 - rank))
        logits.append(lam)

    logits = np.array(logits)
    phi_oos = np.array(phi_oos_list)

    # PBO = fraction of logits below zero
    pbo = np.mean(logits < 0)

    return {
        'pbo': pbo,
        'logits': logits,
        'phi_oos': phi_oos,
    }


# ======================================================================
# Helper: compute annualised Sharpe ratio from a return series
# ======================================================================

def sharpe_ratio(returns, periods_per_year=252):
    """Annualised Sharpe ratio (assuming zero risk-free rate)."""
    mu = returns.mean() * periods_per_year
    sigma = returns.std(ddof=1) * np.sqrt(periods_per_year)
    if sigma == 0:
        return 0.0
    return mu / sigma


# =====================================================================
# MAIN – Demonstration
# =====================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Chapter 11 – The Dangers of Backtesting")
    print("  (Advances in Financial Machine Learning)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Snippet 11.1 – Marcos' Second Law of Backtesting
    # ------------------------------------------------------------------
    print("\n--- Snippet 11.1: Marcos' Second Law of Backtesting ---")
    print(f'  "{SECOND_LAW}"')

    # ------------------------------------------------------------------
    # Generate synthetic multi-strategy returns to simulate multiple
    # backtest trials (the scenario the chapter warns about).
    # ------------------------------------------------------------------
    N_STRATEGIES = 50      # number of backtested strategies
    T_OBS = 1000           # daily observations per strategy
    rng = np.random.default_rng(0)

    # Most strategies have near-zero true alpha; a few have mild signal
    # that gets drowned out by selection bias.
    true_alphas = rng.normal(0.0, 0.0001, N_STRATEGIES)
    strategy_returns = pd.DataFrame(
        {f"Strat_{i}": generate_returns(n=T_OBS, n_assets=1, seed=i).squeeze()
                        + true_alphas[i]
         for i in range(N_STRATEGIES)}
    )

    # ------------------------------------------------------------------
    # A. Compute observed Sharpe ratios for each strategy
    # ------------------------------------------------------------------
    print("\n--- Observed Sharpe Ratios for 50 Simulated Strategies ---")
    sr_values = strategy_returns.apply(sharpe_ratio)
    best_idx = sr_values.idxmax()
    best_sr = sr_values[best_idx]
    print(f"  SR range : [{sr_values.min():.4f}, {sr_values.max():.4f}]")
    print(f"  SR mean  : {sr_values.mean():.4f}")
    print(f"  SR std   : {sr_values.std():.4f}")
    print(f"  Best strategy: {best_idx}  (SR = {best_sr:.4f})")

    # ------------------------------------------------------------------
    # B. Expected Maximum Sharpe Ratio from N trials
    # ------------------------------------------------------------------
    print("\n--- Expected Maximum Sharpe Ratio ---")
    sr_mean = sr_values.mean()
    sr_std = sr_values.std()
    e_max_sr = expected_max_sharpe_ratio(N_STRATEGIES, sr_mean, sr_std)
    print(f"  E[max(SR)] given {N_STRATEGIES} trials: {e_max_sr:.4f}")
    print(f"  Observed max SR              : {best_sr:.4f}")
    print(f"  Difference                   : {best_sr - e_max_sr:+.4f}")
    if best_sr < e_max_sr:
        print("  --> Best SR does NOT exceed the expected maximum from luck alone.")
    else:
        print("  --> Best SR exceeds E[max(SR)] — but check DSR below.")

    # Show how E[max(SR)] grows with the number of trials
    print("\n  E[max(SR)] vs number of trials (mean=0, std=1):")
    print(f"  {'N':>6s}  {'E[max(SR)]':>10s}")
    for n in [1, 5, 10, 25, 50, 100, 200, 500, 1000]:
        print(f"  {n:6d}  {expected_max_sharpe_ratio(n):10.4f}")

    # ------------------------------------------------------------------
    # C. Deflated Sharpe Ratio for the best strategy
    # ------------------------------------------------------------------
    print("\n--- Deflated Sharpe Ratio (DSR) ---")
    best_rets = strategy_returns[best_idx]
    obs_skew = float(best_rets.skew())
    obs_kurt = float(best_rets.kurtosis()) + 3.0  # convert excess -> raw

    dsr = deflated_sharpe_ratio(
        sr_hat=best_sr,
        sr_benchmark=e_max_sr,
        n_obs=T_OBS,
        skew=obs_skew,
        kurtosis=obs_kurt,
    )
    print(f"  Observed SR  : {best_sr:.4f}")
    print(f"  SR benchmark : {e_max_sr:.4f}  (E[max(SR)] from {N_STRATEGIES} trials)")
    print(f"  Skewness     : {obs_skew:.4f}")
    print(f"  Kurtosis     : {obs_kurt:.4f}")
    print(f"  DSR          : {dsr:.4f}")
    if dsr >= 0.95:
        print("  --> High DSR: strong evidence the strategy is genuine.")
    elif dsr >= 0.5:
        print("  --> Moderate DSR: some evidence, but not conclusive.")
    else:
        print("  --> Low DSR: likely a false discovery from multiple testing.")

    # ------------------------------------------------------------------
    # D. Probability of Backtest Overfitting (PBO) via CSCV
    # ------------------------------------------------------------------
    print("\n--- Probability of Backtest Overfitting (PBO via CSCV) ---")
    pbo_result = prob_backtest_overfitting(
        strategy_returns, n_partitions=10, metric='sharpe'
    )
    print(f"  PBO              : {pbo_result['pbo']:.4f}")
    print(f"  Logit mean       : {pbo_result['logits'].mean():.4f}")
    print(f"  Logit std        : {pbo_result['logits'].std():.4f}")
    print(f"  Combos evaluated : {len(pbo_result['logits'])}")

    if pbo_result['pbo'] > 0.5:
        print("  --> HIGH risk of backtest overfitting (PBO > 0.50).")
    else:
        print("  --> Moderate/low PBO — but still exercise caution.")

    # ------------------------------------------------------------------
    # E. Compare individual DSR values across all strategies
    # ------------------------------------------------------------------
    print("\n--- DSR for Top-10 Strategies (by observed SR) ---")
    top10 = sr_values.nlargest(10)
    print(f"  {'Strategy':>12s}  {'SR obs':>8s}  {'DSR':>8s}  {'Verdict':>12s}")
    for strat, sr_obs in top10.items():
        rets_i = strategy_returns[strat]
        sk_i = float(rets_i.skew())
        ku_i = float(rets_i.kurtosis()) + 3.0
        dsr_i = deflated_sharpe_ratio(sr_obs, e_max_sr, T_OBS, sk_i, ku_i)
        verdict = "Genuine" if dsr_i >= 0.95 else ("Marginal" if dsr_i >= 0.5 else "Suspect")
        print(f"  {strat:>12s}  {sr_obs:8.4f}  {dsr_i:8.4f}  {verdict:>12s}")

    print("\n" + "=" * 70)
    print("Chapter 11 demo complete.")
