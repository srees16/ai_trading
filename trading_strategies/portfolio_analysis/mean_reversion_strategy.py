"""
Mean Reversion Trading Strategy
by Chee-Foong on 15 Apr 2021

Summary:
This analysis creates a portfolio of assets that is cointegrated and mean reverting.
Using a linear mean reverting trading strategy on the portfolio, we assess its
performance and risk analytics.

Cryptocurrencies are trading assets used in this analysis. Namely, Bitcoin, Ethereum and Litecoin.

Reference:
1. https://letianzj.github.io/cointegration-pairs-trading.html
2. https://blog.quantinsti.com/johansen-test-cointegration-building-stationary-portfolio/
3. https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html
4. https://flare9xblog.wordpress.com/2017/09/27/half-life-of-mean-reversion-ornstein-uhlenbeck-formula-for-mean-reverting-process/
5. https://quant.stackexchange.com/questions/2076/how-to-interpret-the-eigenmatrix-from-a-johansen-cointegration-test
6. https://pythonforfinance.net/2016/07/10/python-backtesting-mean-reversion-part-4/#more-15487
7. https://medium.com/bluekiri/cointegration-tests-on-time-series-88702ea9c492
8. https://medium.com/bluekiri/simple-stationarity-tests-on-time-series-ad227e2e6d48
"""

# ============================================================
# Import Libraries
# ============================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

import time
import math
import os.path

from datetime import timedelta, datetime
from dateutil import parser

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.dates import MonthLocator

import seaborn as sns
sns.set()

import sys
sys.path.append('../src')

from edge_mean_reversion import *
from edge_binance import *
from edge import *

from edge_risk_kit import *
import edge_risk_kit as erk

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import statsmodels.api as sm

# ============================================================
# Constants
# ============================================================
PERIOD_PER_YEAR = 252 * 1
PERIOD_PER_DAY = 1


# ============================================================
# Helper Functions
# ============================================================
def draw_pair_plot(data, figsize=(10, 6)):
    """Plot three time series on separate subplots sharing the x-axis."""
    ts1 = data.iloc[:, 0]
    ts2 = data.iloc[:, 1]
    ts3 = data.iloc[:, 2]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=figsize)

    ax1.plot(ts1.index, ts1.values, label=ts1.name)
    ax2.plot(ts2.index, ts2.values, label=ts2.name, color='r')
    ax3.plot(ts3.index, ts3.values, label=ts3.name, color='g')

    ax1.set_ylabel(ts1.name)
    ax2.set_ylabel(ts2.name)
    ax3.set_ylabel(ts3.name)

    ax1.grid()
    ax2.grid()
    ax3.grid()

    plt.show()


def plot_reg_line(x, y):
    """Plot data points with a linear regression line."""
    reg = np.polyfit(x, y, deg=1)
    y_fitted = np.polyval(reg, x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'bo', label='data')
    plt.plot(x, y_fitted, 'r', lw=2.5, label='linear regression')
    plt.legend(loc=0)
    plt.show()


def plot_reg_pair(independent, dependent, showplot=False):
    """Determine hedge ratio via OLS regression, optionally plot."""
    model = sm.OLS(dependent, independent)
    coeff = model.fit().params

    if len(coeff) == 2:
        hedge_ratio = coeff[1]
        intercept = coeff[0]
    else:
        hedge_ratio = coeff[0]
        intercept = 0

    if showplot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(independent, dependent, 'bo', label='data')
        ax.plot(independent, independent * hedge_ratio + intercept, 'r', lw=2.5, label='linear regression')
        ax.set_xlabel(independent.name)
        ax.set_ylabel(dependent.name)
        plt.legend(loc=0)
        plt.show()

    return hedge_ratio


def Z_Score(values, n):
    """
    Return Z-Score of `values`, at each step taking into account
    `n` previous values for mean and standard deviation.
    """
    series = pd.Series(values)
    return (series - series.rolling(n).mean()) / series.rolling(n).std()


def neg_Volatility(stats):
    """Return negative volatility for minimisation in optimiser."""
    return -stats['Volatility (Ann.) [%]']


# ============================================================
# Trading Strategy
# ============================================================
class Z_Score_Naive(Strategy):
    """
    Adjusted Mean Reversion Strategy:
    1. Calculate Z-Score at time t, normalising the price by its rolling mean and std.
    2. Execute a trade if the Z-Score is below or above a threshold.
       Buy when Z-Score < -threshold, sell when Z-Score > threshold.
    3. Each trade is transacted with maximum cash on hand.
    4. A buying trade is closed when Z-Score becomes positive;
       a selling trade is closed when Z-Score becomes negative.
    """
    lookback = 30
    threshold = 2
    stoploss = 0.001

    def init(self):
        self.ZScore = self.I(Z_Score, self.data.Close, self.lookback)

    def next(self):
        if (self.position.is_long) & (self.ZScore > 0):
            self.position.close()

        if (self.position.is_short) & (self.ZScore < 0):
            self.position.close()

        if self.position.pl_pct < -self.stoploss:
            self.position.close()

        if (self.ZScore < -self.threshold) & (~self.position.is_long):
            self.position.close()
            self.buy()

        if (self.ZScore > self.threshold) & (~self.position.is_short):
            self.position.close()
            self.sell()


# ============================================================
# Main
# ============================================================
def main():
    # ----------------------------------------------------------
    # Import Data
    # Cryptocurrency prices downloaded from Binance API,
    # preprocessed, resampled daily and stored as HDFStore.
    # ----------------------------------------------------------
    prices = load_from_HDFStore(DATA_FOLDER + 'crypto.h5', 'crypto')
    print("Raw prices shape:", prices.shape)
    print(prices.head())

    # ----------------------------------------------------------
    # Data Preprocessing - select 2020 onwards
    # ----------------------------------------------------------
    prices.dropna(inplace=True)
    prices = prices['2020':]

    # ----------------------------------------------------------
    # Exploratory Data Analysis
    # ----------------------------------------------------------
    # Price Plots
    draw_pair_plot(prices[['eth', 'btc', 'ltc']], figsize=(12, 6))

    # Correlations of Returns
    returns = prices.pct_change().dropna()
    print("\nReturn Correlations:")
    print(returns.corr())

    # Pair Plots of Returns
    sns.pairplot(data=returns, plot_kws={'alpha': 0.5, 's': 2, 'edgecolor': 'b'})
    plt.show()

    # ----------------------------------------------------------
    # Mean Reversion Tests
    # ----------------------------------------------------------
    # Augmented Dickey Fuller Test
    print("\n--- Augmented Dickey Fuller Test ---")
    print(prices.apply(lambda x: perform_adf_test(x), axis=0))

    # Hurst Exponent
    print("\n--- Hurst Exponent ---")
    print(prices.apply(lambda x: perform_hurst_exp_test(x), axis=0))

    # Variance Ratio
    print("\n--- Variance Ratio ---")
    print(prices.apply(lambda x: perform_variance_ratio_test(x), axis=0))

    # Half-Life
    print("\n--- Half-Life (days) ---")
    print(prices.apply(lambda x: round(half_life_v2(x) / PERIOD_PER_DAY), axis=0))

    # Cointegration Tests
    print("\n--- Cointegration Tests ---")
    print("BTC vs ETH:", perform_coint_test(prices.btc, prices.eth, True))
    print("BTC vs LTC:", perform_coint_test(prices.btc, prices.ltc, True))
    print("ETH vs LTC:", perform_coint_test(prices.eth, prices.ltc, True))

    # ----------------------------------------------------------
    # Portfolio Construction - Two Assets (BTC & LTC)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("TWO-ASSET PORTFOLIO (BTC & LTC)")
    print("=" * 60)

    X = prices.btc
    Y = prices.ltc

    plot_reg_line(X, Y)

    hedge_ratio = plot_reg_pair(X, Y)
    print(f"Hedge Ratio: {hedge_ratio}")

    portf_2_assets = Y - hedge_ratio * X
    portf_2_assets.plot(figsize=(12, 6), title='2-Asset Portfolio Price Series')
    plt.show()

    # Portfolio Analysis
    print("\n--- 2-Asset Portfolio Analysis ---")
    print("ADF:", perform_adf_test(portf_2_assets, True))
    print("Hurst:", perform_hurst_exp_test(portf_2_assets, True))
    print("VR:", perform_variance_ratio_test(portf_2_assets, 2, True))
    print("Half-Life:", half_life(portf_2_assets) / PERIOD_PER_DAY)
    print("Cointegration:", perform_coint_test(X, Y, True))

    # Trading Strategy
    lookback = round(half_life(portf_2_assets))
    qty = -(portf_2_assets - portf_2_assets.rolling(lookback).mean()) / portf_2_assets.rolling(lookback).std()

    position = portf_2_assets * qty
    position.plot(figsize=(12, 6), title='Portfolio value over time')
    plt.show()

    # Performance Results
    pnl = position.pct_change().dropna()
    pnl.plot(figsize=(12, 6), title='Daily profit and loss')
    plt.show()

    erk.drawdown(pnl).Wealth.plot(figsize=(12, 6), title='Wealth Ratio')
    plt.show()

    erk.drawdown(pnl).Drawdown.plot(figsize=(12, 6), title='Drawdown')
    plt.show()

    print("\n--- 2-Asset Performance Summary ---")
    print(erk.summary_stats(pnl.to_frame(), riskfree_rate=0.02, periods_per_year=PERIOD_PER_YEAR))

    # ----------------------------------------------------------
    # Portfolio Construction - Three Assets (ETH, BTC, LTC)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("THREE-ASSET PORTFOLIO (ETH, BTC, LTC)")
    print("=" * 60)

    jres = coint_johansen(prices, det_order=0, k_ar_diff=1)
    coeff = jres.evec[:, 0]

    portf_3_assets = (prices * coeff).sum(axis=1)
    portf_3_assets.plot(figsize=(12, 6), title='3-Asset Portfolio Price Series')
    plt.show()

    # Portfolio Analysis
    print("\n--- 3-Asset Portfolio Analysis ---")
    print("ADF:", perform_adf_test(portf_3_assets, True))
    print("Hurst:", perform_hurst_exp_test(portf_3_assets, True))
    print("VR:", perform_variance_ratio_test(portf_3_assets, 2, True))
    print("Half-Life:", half_life(portf_3_assets) / PERIOD_PER_DAY)

    # Trading Strategy
    lookback = round(half_life(portf_3_assets))
    qty = -(portf_3_assets - portf_3_assets.rolling(lookback).mean()) / portf_3_assets.rolling(lookback).std()

    position = portf_3_assets * qty
    position.plot(figsize=(12, 6), title='Portfolio Value')
    plt.show()

    # Performance Results
    pnl = position.pct_change().dropna()
    pnl.plot(figsize=(12, 6), title='Daily Profit and Loss')
    plt.show()

    erk.drawdown(pnl).Wealth.plot(figsize=(12, 6), title='Wealth Ratio')
    plt.show()

    erk.drawdown(pnl).Drawdown.plot(figsize=(12, 6), title='Drawdown')
    plt.show()

    print("\n--- 3-Asset Performance Summary ---")
    print(erk.summary_stats(pnl.to_frame(), riskfree_rate=0.02, periods_per_year=PERIOD_PER_YEAR))

    # ----------------------------------------------------------
    # Enhanced Strategy with Backtesting Library
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("ENHANCED STRATEGY - BACKTESTING")
    print("=" * 60)

    bt_port = portf_3_assets.to_frame()
    bt_port.columns = ['Close']
    bt_port['Open'] = bt_port['Close']
    bt_port['High'] = bt_port['Close']
    bt_port['Low'] = bt_port['Close']

    # --- Initial Parameters ---
    # Threshold = -2 and 2, Lookback = 30, Stoploss = 0.001
    bt = Backtest(bt_port, Z_Score_Naive, cash=10000, commission=.002)
    stats = bt.run()
    print("\n--- Initial Run ---")
    print(stats)
    bt.plot()

    # --- Parameter Optimisation ---

    # 1. Maximising Final Equity
    print("\n--- Optimising: Maximise Final Equity ---")
    stats = bt.optimize(
        lookback=range(20, 40, 5),
        threshold=np.arange(2, 5.5, 0.5).tolist(),
        stoploss=np.arange(0.001, 0.005, 0.001).tolist(),
        maximize='Equity Final [$]'
    )
    print(stats)
    print("Strategy:", stats._strategy)
    print("Trades:\n", stats['_trades'])
    bt.plot(plot_volume=False)

    # 2. Minimising Maximum Drawdown
    print("\n--- Optimising: Minimise Max Drawdown ---")
    stats = bt.optimize(
        lookback=range(20, 40, 5),
        threshold=np.arange(2, 5.5, 0.5).tolist(),
        stoploss=np.arange(0.001, 0.005, 0.001).tolist(),
        maximize='Max. Drawdown [%]'
    )
    print(stats)
    print("Strategy:", stats._strategy)
    bt.plot(plot_volume=False)

    # 3. Minimising Volatility
    print("\n--- Optimising: Minimise Volatility ---")
    stats = bt.optimize(
        lookback=range(20, 40, 5),
        threshold=np.arange(2, 5.5, 0.5).tolist(),
        stoploss=np.arange(0.001, 0.005, 0.001).tolist(),
        maximize=neg_Volatility
    )
    print(stats)
    print("Strategy:", stats._strategy)
    bt.plot(plot_volume=False)

    # 4. Maximising Sharpe Ratio
    print("\n--- Optimising: Maximise Sharpe Ratio ---")
    stats = bt.optimize(
        lookback=range(20, 40, 5),
        threshold=np.arange(2, 5.5, 0.5).tolist(),
        stoploss=np.arange(0.001, 0.005, 0.001).tolist(),
        maximize='Sharpe Ratio'
    )
    print(stats)
    print("Strategy:", stats._strategy)
    bt.plot(plot_volume=False)

    # ----------------------------------------------------------
    # Closing Remarks
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLOSING REMARKS")
    print("=" * 60)
    print("""
1. Portfolio is constructed with a basket of cryptocurrencies.
   Weights (hedge ratios) are determined based on price data from Jan 2020 onwards.
2. Note that the same 'in-sample' data is used in backtesting and
   trading strategy performance measurements.
3. Further analysis needed to assess how strategies perform on 'out-of-sample' data.
""")


if __name__ == '__main__':
    main()
