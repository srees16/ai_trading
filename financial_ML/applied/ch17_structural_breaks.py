"""
Chapter 17: Structural Breaks
==============================
Advances in Financial Machine Learning – Marcos López de Prado

Structural breaks, such as transitions from one market regime to another,
are among the best risk/reward opportunities. This chapter reviews methods
that measure the likelihood of structural breaks so that informative features
can be built upon them.

Tests implemented:
  - CUSUM tests (Brown-Durbin-Evans, Chu-Stinchcombe-White)
  - Explosiveness tests (Chow-type DF, Supremum ADF)
  - Sub-/super-martingale tests

Snippets:
  17.1 – get_bsadf()   – SADF's inner loop
  17.2 – getYX()        – Preparing the datasets
  17.3 – lagDF()        – Apply lags to dataframe
  17.4 – getBetas()     – Fitting the ADF specification
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import get_close_series, SYMBOLS


# ============================================================================
# Snippet 17.3 – Apply Lags to a DataFrame
# ============================================================================
def lagDF(df0, lags):
    """
    Apply lags to a DataFrame.

    Parameters
    ----------
    df0 : pd.DataFrame or pd.Series
        Input time series data.
    lags : int or list
        Number of lags to apply.

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged columns.
    """
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how='outer')
    return df1


# ============================================================================
# Snippet 17.2 – Preparing the Datasets (getYX)
# ============================================================================
def getYX(series, constant, lags):
    """
    Prepare the y and X arrays for the ADF regression.

    Parameters
    ----------
    series : pd.DataFrame
        Log-price series (single column).
    constant : str
        'nc' = no time trend, only a constant.
        'ct' = constant + linear time trend.
        'ctt' = constant + second-degree polynomial time trend.
    lags : int
        Number of lags for the ADF specification.

    Returns
    -------
    y : np.ndarray
        Dependent variable array.
    x : np.ndarray
        Regressor matrix.
    """
    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1:-1, 0]  # lagged level
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
        if constant == 'ctt':
            x = np.append(x, trend ** 2, axis=1)
    return y, x


# ============================================================================
# Snippet 17.4 – Fitting the ADF Specification (getBetas)
# ============================================================================
def getBetas(y, x):
    """
    Fit an ADF regression via OLS and return coefficient mean and variance.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable.
    x : np.ndarray
        Regressor matrix.

    Returns
    -------
    bMean : np.ndarray
        OLS coefficient estimates.
    bVar : np.ndarray
        Variance-covariance matrix of the estimates.
    """
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv, xy)
    err = y - np.dot(x, bMean)
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    return bMean, bVar


# ============================================================================
# Snippet 17.1 – SADF's Inner Loop (get_bsadf)
# ============================================================================
def get_bsadf(logP, minSL, constant, lags):
    """
    Compute the backward SADF statistic for a single end-point.

    This is SADF's inner loop: it estimates
        SADF_t = sup_{t0 in [1, t-tau]} { beta_{t0,t} / sigma_{beta_{t0,t}} }
    which is the backwards-expanding component of the algorithm.

    Parameters
    ----------
    logP : pd.Series
        Log-price series.
    minSL : int
        Minimum sample length (tau).
    constant : str
        Time trend specification ('nc', 'ct', or 'ctt').
    lags : int
        Number of lags in the ADF regression.

    Returns
    -------
    dict
        {'Time': end timestamp, 'gsadf': supremum ADF statistic}
    """
    # Convert to DataFrame if needed
    if isinstance(logP, pd.Series):
        logP = logP.to_frame()
    y, x = getYX(logP, constant=constant, lags=lags)
    startPoints = range(0, y.shape[0] + lags - minSL + 1)
    bsadf = -np.inf
    allADF = []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        if x_.shape[0] <= x_.shape[1]:
            continue  # not enough observations
        try:
            bMean_, bStd_ = getBetas(y_, x_)
            bMean_, bStd_ = bMean_[0, 0], bStd_[0, 0] ** 0.5
            adf_stat = bMean_ / bStd_
            allADF.append(adf_stat)
            if adf_stat > bsadf:
                bsadf = adf_stat
        except np.linalg.LinAlgError:
            continue
    out = {'Time': logP.index[-1], 'gsadf': bsadf}
    return out


# ============================================================================
# SADF series computation (outer loop)
# ============================================================================
def sadf_series(logP, minSL=20, constant='nc', lags=1):
    """
    Compute the SADF time series for a log-price series.

    For each end point t = minSL, ..., T the backwards-expanding ADF
    supremum is evaluated.

    Parameters
    ----------
    logP : pd.Series
        Log-price series.
    minSL : int
        Minimum sample length for the ADF regression.
    constant : str
        Time trend specification ('nc', 'ct', or 'ctt').
    lags : int
        Number of lags in the ADF regression.

    Returns
    -------
    pd.Series
        SADF values indexed by datetime.
    """
    if isinstance(logP, pd.Series):
        logP_df = logP.to_frame()
    else:
        logP_df = logP

    sadf_vals = {}
    for t in range(minSL + lags + 1, logP_df.shape[0]):
        window = logP_df.iloc[:t + 1]
        result = get_bsadf(window, minSL=minSL, constant=constant, lags=lags)
        sadf_vals[result['Time']] = result['gsadf']

    return pd.Series(sadf_vals, name='SADF')


# ============================================================================
# Chu-Stinchcombe-White CUSUM Test on Levels
# ============================================================================
def chu_stinchcombe_white(logP, n=None):
    """
    Chu-Stinchcombe-White CUSUM test on levels.

    Under H0: beta_t = 0 (random walk), computes the standardised departure
    of log-price relative to a reference level.

    Parameters
    ----------
    logP : pd.Series
        Log-price series.
    n : int, optional
        Reference index. Defaults to 0 (beginning of series).

    Returns
    -------
    pd.Series
        S_{n,t} statistics.
    """
    if n is None:
        n = 0
    y = logP.values
    yn = y[n]

    # Estimate sigma from first differences
    diff = np.diff(y[:n + 2]) if n > 0 else np.diff(y)
    sigma2 = np.var(diff, ddof=0) if len(diff) > 0 else 1e-10
    sigma = np.sqrt(max(sigma2, 1e-10))

    stats = {}
    for t in range(n + 1, len(y)):
        s_nt = (y[t] - yn) / (sigma * np.sqrt(t - n))
        stats[logP.index[t]] = s_nt

    return pd.Series(stats, name='CSW_CUSUM')


# ============================================================================
# Sub/Super-Martingale Tests
# ============================================================================
def smt_test(logP, minSL=20, specification='poly1'):
    """
    Sub-/Super-martingale test for explosive behaviour.

    Fits alternative trend specifications and evaluates the supremum
    t-value of the trend coefficient across backwards-expanding windows.

    Parameters
    ----------
    logP : pd.Series
        Log-price series.
    minSL : int
        Minimum sample length.
    specification : str
        One of 'poly1', 'poly2', 'exp', 'power'.

    Returns
    -------
    pd.Series
        SMT values indexed by date.
    """
    from numpy.linalg import lstsq

    y_vals = logP.values
    idx = logP.index
    smt_vals = {}

    for t in range(minSL, len(y_vals)):
        best_stat = -np.inf
        for t0 in range(0, t - minSL + 1):
            y_slice = y_vals[t0:t + 1]
            n_obs = len(y_slice)
            t_range = np.arange(1, n_obs + 1, dtype=float)

            if specification == 'poly1':
                # y_t = alpha + gamma*t + beta*t^2 + eps
                X = np.column_stack([np.ones(n_obs), t_range, t_range ** 2])
                y = y_slice
                beta_idx = 2
            elif specification == 'poly2':
                # log(y_t) = alpha + gamma*t + beta*t^2 + eps
                X = np.column_stack([np.ones(n_obs), t_range, t_range ** 2])
                y = y_slice  # already log prices
                beta_idx = 2
            elif specification == 'exp':
                # log(y_t) = log(alpha) + beta*t + eps
                X = np.column_stack([np.ones(n_obs), t_range])
                y = y_slice
                beta_idx = 1
            elif specification == 'power':
                # log(y_t) = log(alpha) + beta*log(t) + eps
                X = np.column_stack([np.ones(n_obs), np.log(t_range)])
                y = y_slice
                beta_idx = 1
            else:
                raise ValueError(f"Unknown specification: {specification}")

            try:
                beta_hat, residuals, _, _ = lstsq(X, y, rcond=None)
                err = y - X @ beta_hat
                sigma2 = np.dot(err, err) / max(n_obs - X.shape[1], 1)
                XtX_inv = np.linalg.inv(X.T @ X)
                se_beta = np.sqrt(max(sigma2 * XtX_inv[beta_idx, beta_idx], 1e-20))
                t_stat = abs(beta_hat[beta_idx]) / se_beta
                if t_stat > best_stat:
                    best_stat = t_stat
            except (np.linalg.LinAlgError, ValueError):
                continue

        smt_vals[idx[t]] = best_stat

    return pd.Series(smt_vals, name=f'SMT_{specification}')


# ============================================================================
# DEMO
# ============================================================================
def main():
    """Demonstrate structural break detection on real stock data."""
    OUTPUT_DIR = __import__('pathlib').Path(__file__).resolve().parent.parent / "_output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Chapter 17 – Structural Breaks")
    print("=" * 60)

    # --- 1) Download data ------------------------------------------------
    symbol = "NVDA"
    print(f"\n[1] Fetching close prices for {symbol} ...")
    close = get_close_series(symbol, start="2020-01-01", end="2024-12-31")
    logP = np.log(close)
    print(f"    {len(close)} daily bars")

    # --- 2) Demonstrate helper functions ---------------------------------
    print("\n[2] lagDF demo:")
    small = close.iloc[:10].to_frame()
    lagged = lagDF(small, 2)
    print(lagged.head())

    # --- 3) SADF series --------------------------------------------------
    print(f"\n[3] Computing SADF series (minSL=40, lags=1, constant='nc') ...")
    print("    (This uses a subset for speed – full series is O(n^2))")
    subset = logP.iloc[:200]  # use small subset for demo
    sadf = sadf_series(subset, minSL=40, constant='nc', lags=1)
    print(f"    SADF values: {len(sadf)}")
    print(f"    Max SADF = {sadf.max():.4f} at {sadf.idxmax()}")
    print(f"    Min SADF = {sadf.min():.4f}")

    # --- 4) CSW CUSUM test -----------------------------------------------
    print(f"\n[4] Chu-Stinchcombe-White CUSUM test ...")
    csw = chu_stinchcombe_white(logP)
    print(f"    Max S_{{n,t}} = {csw.max():.4f}")
    print(f"    Min S_{{n,t}} = {csw.min():.4f}")

    # --- 5) SMT test (polynomial) ----------------------------------------
    print(f"\n[5] Sub-/Super-Martingale test (poly1 specification) ...")
    subset2 = logP.iloc[:150]
    smt = smt_test(subset2, minSL=30, specification='poly1')
    print(f"    SMT values: {len(smt)}")
    print(f"    Max SMT = {smt.max():.4f}")

    # --- 6) Save results -------------------------------------------------
    results = pd.DataFrame({
        'logP': logP.iloc[:len(sadf)].reindex(sadf.index),
        'SADF': sadf
    }).dropna()
    results.to_csv(OUTPUT_DIR / "ch17_sadf.csv")
    print(f"\n[6] Results saved to _output/ch17_sadf.csv")

    print("\n✓ Chapter 17 complete")


if __name__ == "__main__":
    main()
