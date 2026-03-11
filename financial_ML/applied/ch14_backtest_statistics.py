"""
Chapter 14: Backtest Statistics
================================
Advances in Financial Machine Learning, Marcos Lopez de Prado

This chapter discusses the most commonly used performance evaluation
statistics for backtests: general characteristics (holding period, bet
timing), runs/drawdowns (HHI concentration, DD, TuW), efficiency
(Sharpe ratio, PSR, DSR), and classification scores.

Snippets
--------
14.1  getBetTiming()        – derive timestamps where positions are flattened or flipped
14.2  getHoldingPeriod()    – average holding period via entry-time pairing algorithm
14.3  getHHI()              – Herfindahl-Hirschman concentration index on returns/bets
14.4  computeDD_TuW()       – drawdown and time-under-water series
14.5  PSR / DSR / Sharpe    – probabilistic & deflated Sharpe ratio
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_returns, get_close_series

# ============================================================================
# Snippet 14.1 – Deriving the timing of bets from a series of target positions
# ============================================================================
# A bet takes place between flat positions or position flips.  This function
# returns the sorted index of timestamps where a position is flattened or
# flipped to the opposite side.

def getBetTiming(tPos):
    """
    Derive timestamps of flattening or flipping trades from a series of
    target positions.

    Parameters
    ----------
    tPos : pd.Series
        Series of target positions (indexed by datetime).

    Returns
    -------
    pd.DatetimeIndex
        Sorted timestamps where the position was flattened or flipped.
    """
    # Flattening: position goes to zero
    df0 = tPos[tPos == 0].index
    df1 = tPos.shift(1)
    df1 = df1[df1 != 0].index
    bets = df0.intersection(df1)  # flattening events

    # Flips: sign change in consecutive positions
    df0 = tPos.iloc[1:] * tPos.iloc[:-1].values
    bets = bets.union(df0[df0 < 0].index).sort_values()  # position flips

    # Ensure the last timestamp is included
    if tPos.index[-1] not in bets:
        bets = bets.append(tPos.index[-1:])
    return bets


# ============================================================================
# Snippet 14.2 – Implementation of a holding period estimator
# ============================================================================
# Uses an average entry time pairing algorithm to estimate how long, on
# average, each unit of position is held.

def getHoldingPeriod(tPos):
    """
    Derive the average holding period (in days) using an average entry time
    pairing algorithm.

    Parameters
    ----------
    tPos : pd.Series
        Series of target positions (indexed by datetime).

    Returns
    -------
    float
        Average holding period in days, or NaN if undefined.
    """
    hp = pd.DataFrame(columns=['dT', 'w'])
    tEntry = 0.0
    pDiff = tPos.diff()
    tDiff = (tPos.index - tPos.index[0]) / np.timedelta64(1, 'D')

    for i in range(1, tPos.shape[0]):
        if pDiff.iloc[i] * tPos.iloc[i - 1] >= 0:  # increased or unchanged
            if tPos.iloc[i] != 0:
                tEntry = (tEntry * tPos.iloc[i - 1] + tDiff[i] * pDiff.iloc[i]) / tPos.iloc[i]
        else:  # decreased
            if tPos.iloc[i] * tPos.iloc[i - 1] < 0:  # flip
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(tPos.iloc[i - 1]))
                tEntry = tDiff[i]  # reset entry time
            else:
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(pDiff.iloc[i]))

    if hp['w'].sum() > 0:
        hp = (hp['dT'] * hp['w']).sum() / hp['w'].sum()
    else:
        hp = np.nan
    return hp


# ============================================================================
# Snippet 14.3 – Algorithm for deriving HHI concentration
# ============================================================================
# The Herfindahl-Hirschman Index (HHI) measures how concentrated returns
# (or bets) are.  Values near 0 indicate uniform distribution; near 1
# indicates concentration in a single observation.

def getHHI(betRet):
    """
    Compute the Herfindahl-Hirschman Index (HHI) concentration.

    Parameters
    ----------
    betRet : pd.Series
        Series of bet returns (or counts for time concentration).

    Returns
    -------
    float
        HHI value in [0, 1], or NaN if too few observations.
    """
    if betRet.shape[0] <= 2:
        return np.nan
    wght = betRet / betRet.sum()
    hhi = (wght ** 2).sum()
    hhi = (hhi - betRet.shape[0] ** -1) / (1.0 - betRet.shape[0] ** -1)
    return hhi


# ============================================================================
# Snippet 14.4 – Deriving the sequence of DD and TuW
# ============================================================================
# Computes drawdown (DD) and time-under-water (TuW) series from either a
# returns series or dollar performance series.

def computeDD_TuW(series, dollars=False):
    """
    Compute the series of drawdowns and time under water.

    Parameters
    ----------
    series : pd.Series
        If dollars=False, a cumulative returns series (e.g. cumulative product
        of 1+r).  If dollars=True, a dollar PnL series.
    dollars : bool
        If True, DD is computed as dollar difference from HWM.
        If False, DD is computed as percentage from HWM.

    Returns
    -------
    dd : pd.Series
        Drawdown series (one entry per HWM that was followed by a drawdown).
    tuw : pd.Series
        Time under water in years between consecutive HWMs.
    """
    df0 = series.to_frame('pnl')
    df0['hwm'] = series.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = df0['hwm'].drop_duplicates(keep='first').index  # time of hwm
    df1 = df1[df1['hwm'] > df1['min']]  # hwm followed by a drawdown

    if dollars:
        dd = df1['hwm'] - df1['min']
    else:
        dd = 1 - df1['min'] / df1['hwm']

    tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, 'D')).values / 365.25  # in years
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw


# ============================================================================
# Snippet 14.5 – Sharpe ratio, Probabilistic Sharpe Ratio (PSR), and
#                 Deflated Sharpe Ratio (DSR)
# ============================================================================
# The PSR adjusts the observed SR for short series, skewness, and kurtosis.
# The DSR adjusts the rejection threshold for multiple-testing bias.

def sharpeRatio(returns, rfRate=0.0, periodsPerYear=252):
    """
    Compute the annualized Sharpe ratio from a return series.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.
    rfRate : float
        Risk-free rate per period (default 0).
    periodsPerYear : int
        Number of periods per year for annualisation (252 for daily).

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    excess = returns - rfRate
    sr = excess.mean() / excess.std()
    return sr * np.sqrt(periodsPerYear)


def probabilisticSharpeRatio(returns, srBenchmark=0.0):
    """
    Compute the Probabilistic Sharpe Ratio (PSR).

    PSR estimates the probability that the observed SR exceeds a benchmark
    SR*, adjusting for non-Normal returns and finite sample length.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.
    srBenchmark : float
        Benchmark Sharpe ratio (non-annualized).

    Returns
    -------
    float
        PSR value in [0, 1].
    """
    sr = returns.mean() / returns.std()  # non-annualized observed SR
    T = returns.shape[0]
    skew = returns.skew()
    kurt = returns.kurtosis()  # excess kurtosis; add 3 for raw kurtosis
    gamma4 = kurt + 3  # raw kurtosis

    numerator = (sr - srBenchmark) * np.sqrt(T - 1)
    denominator = np.sqrt(1 - skew * sr + (gamma4 - 1) / 4.0 * sr ** 2)

    if denominator <= 0:
        return np.nan
    psr = norm.cdf(numerator / denominator)
    return psr


def deflatedSharpeRatio(returns, srTrials, nTrials):
    """
    Compute the Deflated Sharpe Ratio (DSR).

    DSR is a PSR where the benchmark SR* is set to the expected maximum SR
    under the null hypothesis that the true SR is zero, adjusted for the
    number of independent trials.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns for the best strategy.
    srTrials : array-like
        Sharpe ratios from all N trials.
    nTrials : int
        Number of independent trials.

    Returns
    -------
    float
        DSR value in [0, 1].
    """
    srVar = np.var(srTrials)
    gamma = 0.5772156649  # Euler-Mascheroni constant

    # Expected maximum SR under null
    srStar = np.sqrt(srVar) * (
        (1 - gamma) * norm.ppf(1 - 1.0 / nTrials)
        + gamma * norm.ppf(1 - 1.0 / (nTrials * np.e))
    )

    # DSR = PSR evaluated at SR* = srStar
    return probabilisticSharpeRatio(returns, srBenchmark=srStar)


# ============================================================================
# Helper: generate a synthetic position series for demonstration
# ============================================================================

def _generateSyntheticPositions(close, seed=42):
    """
    Generate a synthetic target-position series based on a moving-average
    crossover signal.  Positions are +1 (long), -1 (short), or 0 (flat).
    """
    rng = np.random.default_rng(seed)
    fast = close.rolling(20).mean()
    slow = close.rolling(50).mean()
    signal = pd.Series(0.0, index=close.index)
    signal[fast > slow] = 1.0
    signal[fast < slow] = -1.0
    signal = signal.iloc[50:]  # skip warm-up NaNs

    # Introduce occasional flat periods (zero position)
    flat_mask = rng.random(len(signal)) < 0.05
    signal[flat_mask] = 0.0
    return signal


# ============================================================================
# Main – demonstrate all snippets
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 14: Backtest Statistics")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Generate data
    # ------------------------------------------------------------------
    print("\n[1] Generating data ...")
    close = get_close_series("MSFT")
    retDf = generate_returns(n=len(close), n_assets=1, seed=42)
    ret = retDf.iloc[:, 0]
    ret.index = close.index[:len(ret)]  # align dates

    tPos = _generateSyntheticPositions(close)
    print(f"    Close series  : {len(close)} bars")
    print(f"    Return series : {len(ret)} bars")
    print(f"    Position series: {len(tPos)} bars, "
          f"longs={int((tPos > 0).sum())}, shorts={int((tPos < 0).sum())}, "
          f"flat={int((tPos == 0).sum())}")

    # ------------------------------------------------------------------
    # Snippet 14.1 – Bet timing
    # ------------------------------------------------------------------
    print("\n[2] Snippet 14.1 – Bet timing (flatten / flip timestamps)")
    bets = getBetTiming(tPos)
    print(f"    Number of bets (flatten/flip events): {len(bets)}")
    print(f"    First 5 bet timestamps: {list(bets[:5])}")

    # ------------------------------------------------------------------
    # Snippet 14.2 – Average holding period
    # ------------------------------------------------------------------
    print("\n[3] Snippet 14.2 – Average holding period")
    hp = getHoldingPeriod(tPos)
    print(f"    Average holding period: {hp:.2f} days")

    # ------------------------------------------------------------------
    # Snippet 14.3 – HHI concentration
    # ------------------------------------------------------------------
    print("\n[4] Snippet 14.3 – HHI concentration")
    rHHIPos = getHHI(ret[ret >= 0])
    rHHINeg = getHHI(ret[ret < 0])
    # Concentration of bets per month
    tHHI = getHHI(ret.groupby(pd.Grouper(freq='M')).count())
    print(f"    HHI positive returns : {rHHIPos:.6f}")
    print(f"    HHI negative returns : {rHHINeg:.6f}")
    print(f"    HHI time concentration: {tHHI:.6f}")

    # ------------------------------------------------------------------
    # Snippet 14.4 – Drawdown and Time under Water
    # ------------------------------------------------------------------
    print("\n[5] Snippet 14.4 – Drawdown & Time under Water")
    cumRet = (1 + ret).cumprod()
    dd, tuw = computeDD_TuW(cumRet, dollars=False)
    if len(dd) > 0:
        print(f"    Number of DD episodes : {len(dd)}")
        print(f"    Max drawdown          : {dd.max():.4f} ({dd.max()*100:.2f}%)")
        print(f"    95th-percentile DD    : {dd.quantile(0.95):.4f}")
    else:
        print("    No drawdown episodes detected.")
    if len(tuw) > 0:
        print(f"    95th-percentile TuW   : {tuw.quantile(0.95):.4f} years")
    else:
        print("    No TuW data.")

    # ------------------------------------------------------------------
    # Snippet 14.5 – Sharpe, PSR, DSR
    # ------------------------------------------------------------------
    print("\n[6] Snippet 14.5 – Sharpe ratio, PSR, DSR")
    sr_ann = sharpeRatio(ret, rfRate=0.0, periodsPerYear=252)
    print(f"    Annualized Sharpe ratio  : {sr_ann:.4f}")

    psr = probabilisticSharpeRatio(ret, srBenchmark=0.0)
    print(f"    PSR (vs SR*=0)           : {psr:.4f}")

    # Simulate 100 trials for DSR
    rng = np.random.default_rng(123)
    srTrials = rng.normal(loc=0.0, scale=0.5, size=100)
    srTrials[0] = ret.mean() / ret.std()  # best trial = observed
    dsr = deflatedSharpeRatio(ret, srTrials, nTrials=100)
    print(f"    DSR (100 trials)         : {dsr:.4f}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Summary of Backtest Statistics")
    print("-" * 65)
    stats = {
        "Number of bets": len(bets),
        "Avg holding period (days)": round(hp, 2),
        "HHI positive returns": round(rHHIPos, 6),
        "HHI negative returns": round(rHHINeg, 6),
        "HHI time concentration": round(tHHI, 6),
        "Max drawdown (%)": round(dd.max() * 100, 2) if len(dd) > 0 else "N/A",
        "95th pctl DD (%)": round(dd.quantile(0.95) * 100, 2) if len(dd) > 0 else "N/A",
        "95th pctl TuW (yrs)": round(tuw.quantile(0.95), 4) if len(tuw) > 0 else "N/A",
        "Annualized SR": round(sr_ann, 4),
        "PSR (SR*=0)": round(psr, 4),
        "DSR (100 trials)": round(dsr, 4),
    }
    for k, v in stats.items():
        print(f"    {k:<28s}: {v}")

    print("\nAll Chapter 14 snippets executed successfully.")
