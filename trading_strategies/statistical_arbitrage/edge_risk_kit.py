"""
Risk Kit Module (lightweight).

Provides drawdown analysis and summary statistics used by the
Mean Reversion strategy. Adapted from the EDHEC Risk Kit.
"""

import pandas as pd
import numpy as np

DATA_FOLDER = '../data/'


def drawdown(return_series: pd.Series) -> pd.DataFrame:
    """
    Compute wealth index, previous peaks, and percentage drawdown.

    Args:
        return_series: Time series of asset returns.

    Returns:
        DataFrame with Wealth, Previous Peak, and Drawdown columns.
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Previous Peak": previous_peaks,
        "Drawdown": drawdowns,
    })


def annualize_rets(r, periods_per_year, **kwargs):
    """Annualise a set of returns."""
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    if n_periods == 0:
        return 0.0
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year, **kwargs):
    """Annualise volatility."""
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """Compute annualised Sharpe ratio."""
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return ann_ex_ret / ann_vol


def summary_stats(r, riskfree_rate=0.03, periods_per_year=12):
    """
    Return a DataFrame of summary statistics for return series.

    Args:
        r: DataFrame or Series of returns.
        riskfree_rate: Annual risk-free rate.
        periods_per_year: Number of return observations per year.

    Returns:
        DataFrame with Annualised Return, Annualised Vol, Sharpe Ratio,
        Max Drawdown, and Skewness/Kurtosis.
    """
    if isinstance(r, pd.Series):
        r = r.to_frame()

    stats = {
        "Annualised Return": r.aggregate(annualize_rets, periods_per_year=periods_per_year),
        "Annualised Vol": r.aggregate(annualize_vol, periods_per_year=periods_per_year),
        "Sharpe Ratio": r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year),
        "Skewness": r.aggregate(lambda s: s.skew()),
        "Kurtosis": r.aggregate(lambda s: s.kurtosis()),
        "Max Drawdown": r.aggregate(lambda s: drawdown(s)["Drawdown"].min()),
    }
    return pd.DataFrame(stats)
