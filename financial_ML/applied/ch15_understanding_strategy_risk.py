"""
Chapter 15: Understanding Strategy Risk
From "Advances in Financial Machine Learning" by Marcos Lopez de Prado

Snippets 15.1–15.5
- Symmetric payoff matrix and binomial distribution for strategy outcomes
- Implied precision from strategy parameters (SR → betting frequency)
- Strategy failure probability (ruin probability)
- Solve for implied betting frequency n
- Strategy risk profiling (compute various risk metrics)
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_returns


# ============================================================================
# Snippet 15.1 – Targeting a Sharpe ratio as a function of the number of bets
# ============================================================================
def target_sharpe_ratio_sim(p=0.55, n_sim=1_000_000, seed=42):
    """
    Simulate IID symmetric bets with precision p.
    Each bet pays +1 (with prob p) or -1 (with prob 1-p).
    Returns (mean, std, SR_per_bet).
    """
    rng = np.random.default_rng(seed)
    rnd = rng.binomial(n=1, p=p, size=n_sim)
    out = np.where(rnd == 1, 1.0, -1.0)
    mean_ = np.mean(out)
    std_ = np.std(out)
    sr_per_bet = mean_ / std_ if std_ > 0 else 0.0
    return mean_, std_, sr_per_bet


def sharpe_ratio_symmetric(p, n):
    """
    Annualized Sharpe ratio for symmetric payouts.
    θ[p, n] = (2p - 1) / (2 * sqrt(p(1-p))) * sqrt(n)
    """
    if p <= 0 or p >= 1:
        return 0.0
    return (2.0 * p - 1.0) / (2.0 * (p * (1.0 - p)) ** 0.5) * n ** 0.5


def implied_precision_symmetric(theta, n):
    """
    Given a target Sharpe ratio θ and betting frequency n,
    compute the required precision p (symmetric payouts).
    p = 0.5 * (1 + sqrt(1 - n / (θ² + n)))
    """
    denom = theta ** 2 + n
    if denom == 0:
        return np.nan
    val = 1.0 - n / denom
    if val < 0:
        return np.nan
    return 0.5 * (1.0 + val ** 0.5)


# ============================================================================
# Snippet 15.2 – Verify with SymPy (symbolic check of variance formula)
# ============================================================================
def sympy_verify_variance():
    """
    Use SymPy to verify that V[X] = (u - d)^2 * p * (1-p)
    for a two-outcome random variable with payouts u, d.
    """
    try:
        from sympy import symbols, factor
        p, u, d = symbols('p u d')
        m2 = p * u ** 2 + (1 - p) * d ** 2       # E[X^2]
        m1 = p * u + (1 - p) * d                  # E[X]
        v = m2 - m1 ** 2                           # Var[X]
        factored = factor(v)
        return str(factored)
    except ImportError:
        return "SymPy not installed – expected result: -p*(p - 1)*(d - u)**2"


# ============================================================================
# Snippet 15.3 – Computing the implied precision (asymmetric payouts)
# ============================================================================
def binSR(sl, pt, freq, p):
    """
    Compute the annualized Sharpe ratio given:
      sl   : stop-loss threshold (negative, e.g. -0.01)
      pt   : profit-taking threshold (positive, e.g. 0.005)
      freq : number of bets per year
      p    : precision rate (probability of profit-taking exit)
    """
    rng = pt - sl  # total range (positive)
    mean_bet = rng * p + sl
    vol_bet = rng * (p * (1.0 - p)) ** 0.5
    if vol_bet == 0:
        return 0.0
    return mean_bet / vol_bet * freq ** 0.5


def binHR(sl, pt, freq, tSR):
    """
    Snippet 15.3 – Implied precision.
    Given a trading rule characterized by {sl, pt, freq},
    what is the minimum precision p required to achieve a Sharpe ratio tSR?

    Parameters
    ----------
    sl   : stop-loss threshold (negative, e.g. -0.01)
    pt   : profit-taking threshold (positive, e.g. 0.005)
    freq : number of bets per year
    tSR  : target annual Sharpe ratio

    Returns
    -------
    p : minimum precision rate required, or NaN if not achievable
    """
    a = (freq + tSR ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - tSR ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return np.nan
    p = (-b + discriminant ** 0.5) / (2.0 * a)
    if not 0 <= p <= 1:
        return np.nan
    return p


# ============================================================================
# Snippet 15.4 – Computing the implied betting frequency
# ============================================================================
def binFreq(sl, pt, p, tSR):
    """
    Snippet 15.4 – Implied betting frequency.
    Given a trading rule characterized by {sl, pt, p},
    what is the number of bets/year needed to achieve Sharpe ratio tSR?

    Parameters
    ----------
    sl   : stop-loss threshold
    pt   : profit-taking threshold
    p    : precision rate
    tSR  : target annual Sharpe ratio

    Returns
    -------
    freq : number of bets per year, or None if extraneous solution
    """
    mean_bet = (pt - sl) * p + sl
    if mean_bet == 0:
        return None
    freq = (tSR * (pt - sl)) ** 2 * p * (1.0 - p) / mean_bet ** 2
    # Check for extraneous solution
    if not np.isclose(binSR(sl, pt, freq, p), tSR):
        return None
    return freq


# ============================================================================
# Snippet 15.5 – Calculating the strategy risk in practice
# ============================================================================
def mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs):
    """
    Generate random draws from a mixture of two Gaussians.
    """
    n1 = int(nObs * prob1)
    ret1 = np.random.normal(mu1, sigma1, size=n1)
    ret2 = np.random.normal(mu2, sigma2, size=int(nObs) - n1)
    ret = np.append(ret1, ret2, axis=0)
    np.random.shuffle(ret)
    return ret


def probFailure(ret, freq, tSR):
    """
    Derive the probability that the strategy may fail.
    Returns P[p < p_θ*], i.e. the probability that precision
    is below the threshold needed for the target Sharpe ratio.
    """
    rPos = ret[ret > 0].mean()
    rNeg = ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    thresP = binHR(rNeg, rPos, freq, tSR)
    if np.isnan(thresP):
        return np.nan
    # Normal approximation to the bootstrap distribution of p
    risk = ss.norm.cdf(thresP, p, (p * (1.0 - p)) ** 0.5)
    return risk


# ============================================================================
# Strategy risk profiling – extended analysis
# ============================================================================
def strategy_risk_profile(sl, pt, freq, p, tSR):
    """
    Compute a comprehensive risk profile for a strategy.
    """
    achieved_sr = binSR(sl, pt, freq, p)
    implied_p = binHR(sl, pt, freq, tSR)
    implied_n = binFreq(sl, pt, p, tSR)

    mean_bet = (pt - sl) * p + sl
    annual_return = mean_bet * freq
    annual_vol = (pt - sl) * (p * (1.0 - p)) ** 0.5 * freq ** 0.5

    profile = {
        "Stop-Loss (sl)": sl,
        "Profit-Taking (pt)": pt,
        "Betting Frequency (n)": freq,
        "Precision (p)": p,
        "Target SR": tSR,
        "Achieved SR": round(achieved_sr, 4),
        "Implied Precision (for target SR)": round(implied_p, 4) if not np.isnan(implied_p) else "N/A",
        "Implied Frequency (for target SR)": round(implied_n, 2) if implied_n is not None else "N/A",
        "Expected Annual Return": round(annual_return, 6),
        "Annual Volatility": round(annual_vol, 6),
        "Mean Bet P&L": round(mean_bet, 6),
    }
    return profile


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("CHAPTER 15: Understanding Strategy Risk")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Snippet 15.1 – Symmetric payoff simulation
    # ------------------------------------------------------------------
    print("\n--- Snippet 15.1: Symmetric Payoff Simulation ---")
    p_val = 0.55
    mean_, std_, sr_per_bet = target_sharpe_ratio_sim(p=p_val, n_sim=1_000_000)
    print(f"  p = {p_val}")
    print(f"  Simulated: mean={mean_:.4f}, std={std_:.4f}, SR/bet={sr_per_bet:.4f}")

    # Analytical t-value of p
    t_val = (2 * p_val - 1) / (2 * (p_val * (1 - p_val)) ** 0.5)
    print(f"  Analytical t-value (per bet) = {t_val:.4f}")

    # SR for various betting frequencies
    print("\n  Sharpe ratio vs. betting frequency (p=0.55):")
    for n in [52, 100, 260, 396, 1000]:
        sr = sharpe_ratio_symmetric(p_val, n)
        print(f"    n={n:>5d}  =>  SR = {sr:.4f}")

    n_for_sr2 = int(np.ceil((2.0 / t_val) ** 2))
    print(f"  Bets needed for SR=2.0 with p=0.55: n = {n_for_sr2}")

    # ------------------------------------------------------------------
    # Snippet 15.2 – SymPy verification
    # ------------------------------------------------------------------
    print("\n--- Snippet 15.2: SymPy Variance Verification ---")
    result = sympy_verify_variance()
    print(f"  Factored V[X] = {result}")

    # ------------------------------------------------------------------
    # Implied precision (symmetric)
    # ------------------------------------------------------------------
    print("\n--- Implied Precision (Symmetric) ---")
    for n in [52, 100, 260, 520]:
        p_imp = implied_precision_symmetric(theta=2.0, n=n)
        print(f"  n={n:>4d}, target SR=2.0  =>  required p = {p_imp:.4f}")

    # ------------------------------------------------------------------
    # Snippet 15.3 – Implied precision (asymmetric payouts)
    # ------------------------------------------------------------------
    print("\n--- Snippet 15.3: Implied Precision (Asymmetric Payouts) ---")
    # Example from book: n=260, sl=-0.01, pt=0.005, p=0.7 => SR=1.173
    sr_check = binSR(sl=-0.01, pt=0.005, freq=260, p=0.7)
    print(f"  binSR(sl=-0.01, pt=0.005, freq=260, p=0.7) = {sr_check:.4f}")

    # Implied precision for SR=2
    p_req = binHR(sl=-0.01, pt=0.005, freq=260, tSR=2.0)
    print(f"  binHR(sl=-0.01, pt=0.005, freq=260, tSR=2.0) = {p_req:.4f}")

    # Various stop-loss / profit-taking combinations
    print("\n  Implied precision for different SL/PT (freq=260, tSR=1.5):")
    for sl, pt in [(-0.01, 0.01), (-0.02, 0.01), (-0.01, 0.005), (-0.05, 0.10)]:
        p_imp = binHR(sl, pt, freq=260, tSR=1.5)
        sr_at_p = binSR(sl, pt, 260, p_imp) if not np.isnan(p_imp) else np.nan
        tag = f"{p_imp:.4f}" if not np.isnan(p_imp) else "N/A"
        print(f"    sl={sl:>6.3f}, pt={pt:>5.3f}  =>  p = {tag}")

    # ------------------------------------------------------------------
    # Snippet 15.4 – Implied betting frequency
    # ------------------------------------------------------------------
    print("\n--- Snippet 15.4: Implied Betting Frequency ---")
    combos = [
        (-0.01, 0.01, 0.55, 2.0),
        (-0.01, 0.005, 0.72, 2.0),
        (-0.02, 0.02, 0.60, 1.5),
        (-0.05, 0.10, 0.55, 1.5),
    ]
    for sl, pt, p, tSR in combos:
        freq = binFreq(sl, pt, p, tSR)
        tag = f"{freq:.1f}" if freq is not None else "N/A (extraneous)"
        print(f"  sl={sl:>6.3f}, pt={pt:>5.3f}, p={p:.2f}, tSR={tSR:.1f}"
              f"  =>  freq = {tag}")

    # ------------------------------------------------------------------
    # Snippet 15.5 – Strategy failure probability
    # ------------------------------------------------------------------
    print("\n--- Snippet 15.5: Strategy Failure Probability ---")
    # Parameters from the book
    mu1, mu2, sigma1, sigma2, prob1, nObs = 0.05, -0.1, 0.05, 0.1, 0.75, 2600
    tSR, freq = 2.0, 260

    ret = mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs)
    probF = probFailure(ret, freq, tSR)
    print(f"  Mixture params: mu+={mu1}, mu-={mu2}, σ+={sigma1}, σ-={sigma2}")
    print(f"  prob(positive)={prob1}, nObs={nObs}")
    print(f"  Target SR={tSR}, freq={freq}")
    print(f"  Prob strategy will fail: {probF:.6f}")

    # Test with different target Sharpe ratios
    print("\n  Failure probability for different target SRs:")
    for tsr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pf = probFailure(ret, freq, tsr)
        tag = f"{pf:.6f}" if not np.isnan(pf) else "N/A"
        print(f"    tSR={tsr:.1f}  =>  P[failure] = {tag}")

    # ------------------------------------------------------------------
    # Strategy risk profiling
    # ------------------------------------------------------------------
    print("\n--- Strategy Risk Profiling ---")
    scenarios = [
        {"sl": -0.02, "pt": 0.02, "freq": 260, "p": 0.55, "tSR": 2.0},
        {"sl": -0.02, "pt": 0.02, "freq": 52,  "p": 0.55, "tSR": 2.0},
        {"sl": -0.01, "pt": 0.005,"freq": 260, "p": 0.70, "tSR": 2.0},
        {"sl": -0.05, "pt": 0.10, "freq": 52,  "p": 0.60, "tSR": 1.5},
    ]
    for i, s in enumerate(scenarios, 1):
        print(f"\n  Scenario {i}:")
        profile = strategy_risk_profile(**s)
        for k, v in profile.items():
            print(f"    {k:>38s}: {v}")

    # ------------------------------------------------------------------
    # Use sample_data returns for a simple empirical demo
    # ------------------------------------------------------------------
    print("\n--- Empirical Demo with generate_returns ---")
    returns_df = generate_returns(n=2000, n_assets=1, seed=42)
    ret_series = returns_df.iloc[:, 0].values

    rPos = ret_series[ret_series > 0].mean()
    rNeg = ret_series[ret_series <= 0].mean()
    obs_p = (ret_series > 0).sum() / len(ret_series)
    freq_emp = len(ret_series) / (len(ret_series) / 260)  # ~260 per year
    print(f"  Sample returns: n={len(ret_series)}, mean={ret_series.mean():.6f}")
    print(f"  Avg positive return:  {rPos:.6f}")
    print(f"  Avg negative return:  {rNeg:.6f}")
    print(f"  Observed precision:   {obs_p:.4f}")
    print(f"  Implied SR (asymmetric): {binSR(rNeg, rPos, freq_emp, obs_p):.4f}")

    for tsr in [0.5, 1.0, 1.5, 2.0]:
        pf = probFailure(ret_series, freq_emp, tsr)
        tag = f"{pf:.6f}" if not np.isnan(pf) else "N/A"
        print(f"  P[failure | tSR={tsr:.1f}] = {tag}")

    # ------------------------------------------------------------------
    # Exercise 15.1 walkthrough
    # ------------------------------------------------------------------
    print("\n--- Exercise 15.1: Strategy Viability ---")
    ex_sl, ex_pt, ex_freq, ex_p, ex_tSR = -0.02, 0.02, 52, 0.60, 2.0
    ex_sr = binSR(ex_sl, ex_pt, ex_freq, ex_p)
    print(f"  sl={ex_sl}, pt={ex_pt}, freq={ex_freq}, p={ex_p}, target SR={ex_tSR}")
    print(f"  (a) Achieved SR = {ex_sr:.4f}  => {'Viable' if ex_sr >= ex_tSR else 'NOT viable'}")

    req_p = binHR(ex_sl, ex_pt, ex_freq, ex_tSR)
    print(f"  (b) Required precision for SR={ex_tSR}: p = {req_p:.4f}")

    req_freq = binFreq(ex_sl, ex_pt, ex_p, ex_tSR)
    tag = f"{req_freq:.1f}" if req_freq is not None else "N/A"
    print(f"  (c) Required freq for SR={ex_tSR} with p={ex_p}: n = {tag}")

    # (d) Search for viable profit-taking threshold
    for pt_test in np.arange(0.02, 0.20, 0.01):
        sr_test = binSR(ex_sl, pt_test, ex_freq, ex_p)
        if sr_test >= ex_tSR:
            print(f"  (d) Min pt for SR>={ex_tSR}: pt = {pt_test:.2f} (SR={sr_test:.4f})")
            break

    # (e) Search for viable stop-loss
    for sl_test in np.arange(-0.02, -0.20, -0.01):
        sr_test = binSR(sl_test, ex_pt, ex_freq, ex_p)
        if sr_test >= ex_tSR:
            print(f"  (e) Viable stop-loss for SR>={ex_tSR}: sl = {sl_test:.2f} (SR={sr_test:.4f})")
            break

    print("\n" + "=" * 70)
    print("Done.")
