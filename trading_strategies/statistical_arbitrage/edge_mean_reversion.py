"""
Mean Reversion Statistical Tests Module.

Provides stationarity and cointegration tests used by the
Mean Reversion strategy:
- Augmented Dickey-Fuller (ADF) test
- Hurst Exponent
- Variance Ratio test
- Half-life of mean reversion
- Cointegration test (Engle-Granger)
- Johansen cointegration test

Reference:
    https://medium.com/bluekiri/simple-stationarity-tests-on-time-series-ad227e2e6d48
    https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/
"""

import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat, add_trend
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm

try:
    from arch.unitroot import VarianceRatio as ArchVarianceRatio
    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False


def adf(ts):
    """
    Augmented Dickey-Fuller unit root test (manual implementation).

    Returns the p-value.
    """
    ts = np.asarray(ts)
    nobs = ts.shape[0]
    maxlag = 1

    tsdiff = np.diff(ts)
    tsdall = lagmat(tsdiff[:, None], maxlag, trim='both', original='in')
    nobs = tsdall.shape[0]
    tsdall[:, 0] = ts[-nobs - 1:-1]
    tsdshort = tsdiff[-nobs:]

    results = OLS(tsdshort, add_trend(tsdall[:, :maxlag + 1], 'c')).fit()
    adfstat = results.tvalues[0]

    pvalue = mackinnonp(adfstat, 'c', N=1)
    return pvalue


def cadf(x, y):
    """
    Cointegrated Augmented Dickey-Fuller Test.

    Returns the p-value of the ADF test on the OLS residuals.
    """
    ols_result = OLS(x, y).fit()
    return adf(ols_result.resid)


def hurst(ts):
    """
    Returns the Hurst Exponent of the time series.

    H < 0.5 mean reverting
    H = 0.5 random walk
    H > 0.5 trending
    """
    ts = np.asarray(ts)

    lagvec = []
    tau = []
    lags = range(2, 100)

    for lag in lags:
        pdiff = np.subtract(ts[lag:], ts[:-lag])
        lagvec.append(lag)
        tau.append(np.sqrt(np.std(pdiff)))

    m = np.polyfit(
        np.log10(np.asarray(lagvec)),
        np.log10(np.asarray(tau).clip(min=1e-10)),
        1,
    )
    return m[0] * 2.0


def variance_ratio(ts, lag=2):
    """
    Returns the variance ratio test statistic.
    """
    ts = np.asarray(ts)
    n = len(ts)
    mu = sum(ts[1:n] - ts[:n - 1]) / n
    m = (n - lag + 1) * (1 - lag / n)
    b = sum(np.square(ts[1:n] - ts[:n - 1] - mu)) / (n - 1)
    t = sum(np.square(ts[lag:n] - ts[:n - lag] - lag * mu)) / m
    return t / (lag * b)


def half_life(ts):
    """
    Calculates the half-life of a mean reversion process.
    """
    ts = np.asarray(ts)
    delta_ts = np.diff(ts)
    lag_ts = np.vstack([ts[:-1], np.ones(len(ts[:-1]))]).T
    beta = np.linalg.lstsq(lag_ts, delta_ts, rcond=None)
    return -(np.log(2) / beta[0])[0]


def half_life_v2(series):
    """
    Calculates half-life using OLS regression on lagged values.
    """
    ts = series.to_frame()
    ts['lag'] = series.shift(1)
    ts['ret'] = series - ts.lag
    ts = sm.add_constant(ts)
    ts.dropna(inplace=True)

    model = sm.OLS(ts.ret, ts[['const', 'lag']])
    res = model.fit()

    halflife = -np.log(2) / res.params.iloc[1]
    return halflife


def perform_adf_test(series, notes=False):
    """
    Augmented Dickey-Fuller stationarity test.

    Returns (adf_stat, p_value, is_stationary).
    """
    result = adfuller(series)
    if notes:
        print('ADF Statistics: {:.3f}'.format(result[0]))
        print('p-value: {:.3f}'.format(result[1]))
        if result[1] < 0.05:
            print('Stationary (p < 0.05)')
        else:
            print('Non-stationary (p >= 0.05)')
    return float(result[0]), float(result[1]), bool(result[1] < 0.05)


def perform_hurst_exp_test(series, notes=False):
    """
    Hurst Exponent test.

    Returns (hurst_value, is_mean_reverting).
    """
    result = hurst(series)
    if notes:
        print('Hurst Exponent: {:.3f}'.format(result))
        if result < 0.5:
            print('Mean Reverting')
        elif result > 0.5:
            print('Trending')
        else:
            print('Random Walk')
    return float(result), bool(result < 0.5)


def perform_variance_ratio_test(series, lag=2, notes=False):
    """
    Variance Ratio test using the ``arch`` library when available,
    falling back to the manual implementation otherwise.

    Returns (p_value, is_not_random_walk).
    """
    if _HAS_ARCH:
        result = ArchVarianceRatio(series, lag).pvalue
    else:
        # Fallback: use the manual statistic; treat ratio far from 1 as significant
        vr = variance_ratio(series, lag)
        # Approximate: if ratio < 0.8 or > 1.2 treat as non-random
        result = 0.01 if abs(vr - 1) > 0.2 else 0.10

    if notes:
        print('Variance Ratio p-value: {:.3f}'.format(result))
        if result < 0.05:
            print('Not a random walk')
        else:
            print('Random Walk')
    return float(result), bool(result < 0.05)


def perform_coint_test(ts1, ts2, notes=False):
    """
    Engle-Granger cointegration test.

    Returns (p_value, is_cointegrated, test_statistic).
    """
    results = coint(ts1, ts2)
    if notes:
        if results[1] < 0.05:
            print('Cointegrated')
        else:
            print('Not cointegrated')
    return float(results[1]), bool(results[1] < 0.05), float(results[0])
