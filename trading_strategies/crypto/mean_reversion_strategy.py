"""
Mean Reversion Trading Strategy (Crypto, Binance API)
by Chee-Foong on 15 Apr 2021 -- integrated as BaseStrategy adapter.

Summary:
This analysis creates a portfolio of assets that is cointegrated and mean reverting.
Using a linear mean reverting trading strategy on the portfolio, we assess its
performance and risk analytics.

Cryptocurrencies are trading assets.  Prices are fetched from the Binance public
REST API via ``binance_data.fetch_crypto_prices`` and cached locally as CSV.

The full standalone pipeline is preserved:
  - EDA (pair price plots, return correlations, seaborn pair plots)
  - Statistical mean-reversion tests (ADF, Hurst, Variance Ratio, Half-Life)
  - Cointegration tests (Engle-Granger pairwise + Johansen multi-asset)
  - 2-asset portfolio construction (OLS hedge ratio)
  - 3-asset portfolio construction (Johansen eigenvector weights)
  - Wealth / drawdown analysis via edge_risk_kit
  - Enhanced backtesting via the ``backtesting.py`` library
  - Parameter optimisation (max equity, min drawdown, min volatility, max Sharpe)

All charts and HTML artefacts are returned in the StrategyResult and
persisted to MinIO object storage and the PostgreSQL database.

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

import io
import logging
import math
import os
import sys
import tempfile
import time
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings("ignore", category=UserWarning, message="FigureCanvasAgg is non-interactive")
warnings.filterwarnings("ignore", category=UserWarning, message="Superimposed OHLC plot")
warnings.filterwarnings("ignore", category=UserWarning, message="no explicit representation of timezones")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*multi-process optimization.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*If you want to use multi-process.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from strategies.base_strategy import (
    BaseStrategy,
    StrategyResult,
    StrategyCategory,
    ChartData,
    TableData,
    RiskParams,
)
from strategies.registry import StrategyRegistry
from strategies.utils import matplotlib_to_base64
from trading_strategies.crypto.binance_data import (
    fetch_crypto_prices,
)
from trading_strategies.statistical_arbitrage.edge_mean_reversion import (
    perform_adf_test,
    perform_hurst_exp_test,
    perform_variance_ratio_test,
    half_life,
    half_life_v2,
    perform_coint_test,
)
from trading_strategies.statistical_arbitrage.edge_risk_kit import (
    drawdown as erk_drawdown,
    summary_stats as erk_summary_stats,
)

# -- Storage services (optional -- graceful fallback) -----------------
try:
    from storage.minio_service import get_minio_service
    _MINIO_AVAILABLE = True
except ImportError:
    _MINIO_AVAILABLE = False

try:
    from database.service import DatabaseService as _DatabaseService
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
PERIOD_PER_YEAR = 252
PERIOD_PER_DAY = 1


# ============================================================
# Helper Functions (preserved from standalone)
# ============================================================
def draw_pair_plot(data, figsize=(12, 6)):
    """Plot N time series on separate subplots sharing the x-axis."""
    n_cols = data.shape[1]
    fig, axes = plt.subplots(n_cols, sharex=True, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    colors = ["steelblue", "indianred", "seagreen", "darkorange", "purple"]
    for i, col in enumerate(data.columns):
        ax = axes[i]
        ax.plot(data.index, data[col].values, color=colors[i % len(colors)])
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_reg_line(x, y):
    """Plot data points with a linear regression line."""
    reg = np.polyfit(x, y, deg=1)
    y_fitted = np.polyval(reg, x)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, "bo", label="data", markersize=3, alpha=0.5)
    ax.plot(x, y_fitted, "r", lw=2.5, label="linear regression")
    ax.legend(loc=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_reg_pair(independent, dependent, showplot=False):
    """Determine hedge ratio via OLS regression, optionally return figure."""
    model = sm.OLS(dependent, independent)
    coeff = model.fit().params
    if len(coeff) == 2:
        hedge_ratio = coeff.iloc[1]
        intercept = coeff.iloc[0]
    else:
        hedge_ratio = coeff.iloc[0]
        intercept = 0

    fig = None
    if showplot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(independent, dependent, "bo", label="data", markersize=3, alpha=0.5)
        ax.plot(independent, independent * hedge_ratio + intercept, "r", lw=2.5, label="linear regression")
        ax.set_xlabel(independent.name)
        ax.set_ylabel(dependent.name)
        ax.legend(loc=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return hedge_ratio, fig


def Z_Score(values, n):
    """
    Return Z-Score of ``values``, at each step taking into account
    ``n`` previous values for mean and standard deviation.
    """
    series = pd.Series(values)
    return (series - series.rolling(n).mean()) / series.rolling(n).std()


def neg_Volatility(stats):
    """Return negative volatility for minimisation in optimiser."""
    return -stats["Volatility (Ann.) [%]"]


# ============================================================
# Backtesting.py Strategy Class (preserved exactly)
# ============================================================
class Z_Score_Naive(Strategy):
    """
    Adjusted Mean Reversion Strategy using the ``backtesting.py`` library.

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
# Integrated Strategy (BaseStrategy adapter)
# ============================================================
@StrategyRegistry.register_decorator
class CryptoMeanReversionStrategy(BaseStrategy):
    """
    Crypto Mean Reversion Strategy (Binance API, Z-Score).

    Downloads crypto prices from the Binance public API, applies
    the full standalone analysis pipeline (EDA, statistical tests,
    2-/3-asset portfolio construction, backtesting with parameter
    optimisation), and returns all artefacts via StrategyResult.

    Charts & HTML backtest plots are persisted to MinIO when available;
    metrics are stored in the PostgreSQL database.
    """

    name = "Crypto Mean Reversion (Z-Score)"
    description = (
        "Binance-sourced crypto portfolio with Z-Score mean-reversion "
        "signals -- full EDA, cointegration tests & backtesting optimisation"
    )
    category = StrategyCategory.STATISTICAL_ARBITRAGE
    version = "3.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 60

    # -- Configurable parameters ----------------------------------
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        return {
            "lookback": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 120,
                "description": "Rolling window for Z-Score (0 = auto from half-life)",
            },
            "threshold": {
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "description": "Z-Score entry threshold",
            },
            "stoploss": {
                "type": "float",
                "default": 0.001,
                "min": 0.001,
                "max": 0.20,
                "description": "Per-trade stop-loss fraction",
            },
            "cash": {
                "type": "float",
                "default": 10000,
                "min": 100,
                "max": 10000000,
                "description": "Initial backtesting cash",
            },
            "commission": {
                "type": "float",
                "default": 0.002,
                "min": 0.0,
                "max": 0.05,
                "description": "Commission per trade (as fraction)",
            },
            "run_optimisation": {
                "type": "bool",
                "default": True,
                "description": "Run parameter optimisation (slower but more thorough)",
            },
        }

    # -- Main entry point -----------------------------------------
    def run(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        capital: float,
        sentiment_data: Optional[dict] = None,
        risk_params: Optional[RiskParams | dict] = None,
        **kwargs,
    ) -> StrategyResult:
        start_time = time.time()
        charts: list[ChartData] = []
        tables: list[TableData] = []
        metrics: dict[str, Any] = {}
        html_artefacts: list[tuple[str, bytes]] = []  # (filename, html_bytes)

        # -- 1. Validate inputs -----------------------------------
        try:
            self._validate_crypto_inputs(tickers, start_date, end_date, capital)
        except ValueError as e:
            return StrategyResult(success=False, error_message=str(e))

        # -- 2. Parse kwargs parameters ---------------------------
        lookback = int(kwargs.get("lookback", 30))
        threshold = float(kwargs.get("threshold", 2.0))
        stoploss = float(kwargs.get("stoploss", 0.001))
        cash = float(kwargs.get("cash", capital))
        commission = float(kwargs.get("commission", 0.002))
        run_optimisation = kwargs.get("run_optimisation", True)

        # Build Binance symbol map from user-supplied tickers
        symbols = self._build_symbol_map(tickers)

        # -- 3. Fetch data from Binance ---------------------------
        try:
            raw_prices = fetch_crypto_prices(
                symbols=symbols,
                interval="1d",
                start=start_date,
                end=end_date,
            )
        except Exception as e:
            return StrategyResult(
                success=False,
                error_message=f"Failed to fetch data from Binance: {e}",
            )

        if raw_prices.empty or len(raw_prices) < self.min_data_points:
            return StrategyResult(
                success=False,
                error_message=(
                    f"Insufficient data: {len(raw_prices)} rows "
                    f"(need >= {self.min_data_points})"
                ),
            )

        metrics["raw_rows"] = len(raw_prices)
        col_names = list(raw_prices.columns)

        # -- 4. Preprocessing ------------------------------------
        prices = raw_prices.dropna()
        prices = prices[start_date:]
        if len(prices) < self.min_data_points:
            return StrategyResult(
                success=False,
                error_message=f"Only {len(prices)} data points after filtering; need {self.min_data_points}.",
            )
        metrics["filtered_rows"] = len(prices)

        # -- 5. EDA -----------------------------------------------
        eda_charts, eda_tables = self._run_eda(prices, col_names)
        charts.extend(eda_charts)
        tables.extend(eda_tables)

        # -- 6. Statistical tests ---------------------------------
        test_charts, test_tables, test_metrics = self._run_statistical_tests(prices, col_names)
        charts.extend(test_charts)
        tables.extend(test_tables)
        metrics.update(test_metrics)

        # -- 7. Two-asset portfolio -------------------------------
        if len(col_names) >= 2:
            p2_charts, p2_tables, p2_metrics = self._build_two_asset_portfolio(prices, col_names)
            charts.extend(p2_charts)
            tables.extend(p2_tables)
            metrics["two_asset"] = p2_metrics

        # -- 8. Three-asset portfolio (Johansen) ------------------
        portf_3_assets = None
        if len(col_names) >= 3:
            p3_charts, p3_tables, p3_metrics, portf_3_assets = self._build_three_asset_portfolio(prices)
            charts.extend(p3_charts)
            tables.extend(p3_tables)
            metrics["three_asset"] = p3_metrics

        # -- 9. Enhanced backtesting ------------------------------
        if portf_3_assets is not None:
            bt_series = portf_3_assets
        elif len(col_names) >= 2:
            X = prices[col_names[0]]
            Y = prices[col_names[1]]
            hr, _ = plot_reg_pair(X, Y)
            bt_series = Y - hr * X
        else:
            bt_series = prices.iloc[:, 0]

        bt_charts, bt_tables, bt_metrics, bt_html = self._run_backtesting(
            bt_series, lookback, threshold, stoploss, cash, commission, run_optimisation,
        )
        charts.extend(bt_charts)
        tables.extend(bt_tables)
        metrics["backtesting"] = bt_metrics
        html_artefacts.extend(bt_html)

        # -- 10. Build signals DataFrame --------------------------
        signals_df = self._build_signals_df(bt_series, lookback, threshold)

        # -- 11. Portfolio value curve ----------------------------
        portfolio_df = self._build_portfolio_df(signals_df, capital)

        # -- 12. Aggregate metrics --------------------------------
        base_metrics = self.calculate_metrics(portfolio_df, signals_df, capital)
        metrics.update(base_metrics)

        execution_time = time.time() - start_time
        metrics["execution_time"] = round(execution_time, 2)

        result = StrategyResult(
            charts=charts,
            tables=tables,
            metrics=metrics,
            signals=signals_df,
            portfolio=portfolio_df,
            success=True,
            execution_time=execution_time,
            metadata={
                "strategy": self.name,
                "tickers": col_names,
                "parameters": {
                    "lookback": lookback,
                    "threshold": threshold,
                    "stoploss": stoploss,
                    "cash": cash,
                    "commission": commission,
                    "run_optimisation": run_optimisation,
                },
                "binance_symbols": symbols,
            },
        )

        # -- 13. Persist to MinIO & Database ----------------------
        run_id = f"run_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._save_to_minio(run_id, result, html_artefacts)
        self._save_to_database(result, col_names, start_date, end_date, capital, kwargs)

        return result
    # ================================================================
    # Input validation
    # ================================================================
    def _validate_crypto_inputs(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        capital: float,
    ):
        """Validate crypto-specific inputs."""
        if not tickers or len(tickers) < 2:
            raise ValueError(
                "At least 2 crypto tickers required (e.g. ['ETH', 'BTC', 'LTC'])"
            )
        if len(tickers) > 10:
            raise ValueError("Maximum 10 tickers supported")

        try:
            s_dt = datetime.strptime(start_date, "%Y-%m-%d")
            e_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Dates must be YYYY-MM-DD format")

        if s_dt >= e_dt:
            raise ValueError("start_date must be before end_date")
        if (e_dt - s_dt).days < 30:
            raise ValueError("Date range must be at least 30 days")
        if capital <= 0:
            raise ValueError("Capital must be positive")

    def _build_symbol_map(self, tickers: list[str]) -> dict[str, str]:
        """
        Convert user tickers to Binance symbol map.

        Accepts:
            - Plain names: 'ETH', 'BTC', 'eth', 'btc'
            - Full pairs: 'ETHUSDT', 'BTCUSDT'
        """
        symbol_map: dict[str, str] = {}
        for t in tickers:
            t_clean = t.strip().upper()
            if t_clean.endswith("USDT"):
                short = t_clean.replace("USDT", "").lower()
                symbol_map[t_clean] = short
            elif t_clean.endswith("USD"):
                pair = t_clean.replace("USD", "USDT")
                short = t_clean.replace("USD", "").lower()
                symbol_map[pair] = short
            else:
                pair = f"{t_clean}USDT"
                symbol_map[pair] = t_clean.lower()
        return symbol_map

    # ================================================================
    # EDA
    # ================================================================
    def _run_eda(
        self, prices: pd.DataFrame, col_names: list[str]
    ) -> tuple[list[ChartData], list[TableData]]:
        """Exploratory data analysis -- price plots, correlations, pair plots."""
        charts: list[ChartData] = []
        tables: list[TableData] = []

        # Price subplots
        fig = draw_pair_plot(prices[col_names])
        charts.append(ChartData(
            title="Asset Price Series",
            data=matplotlib_to_base64(fig),
            chart_type="matplotlib",
            description=f"Daily close prices for {', '.join(c.upper() for c in col_names)}",
        ))

        # Return correlations
        returns = prices.pct_change().dropna()
        corr = returns.corr()

        corr_rows = []
        for c1 in corr.columns:
            row: dict[str, Any] = {"Ticker": c1.upper()}
            for c2 in corr.columns:
                row[c2.upper()] = round(float(corr.loc[c1, c2]), 4)
            corr_rows.append(row)
        tables.append(TableData(
            title="Return Correlations",
            data=corr_rows,
            columns=["Ticker"] + [c.upper() for c in corr.columns],
            description="Pairwise return correlations (daily)",
        ))

        # Seaborn pair plot
        try:
            pp = sns.pairplot(
                data=returns,
                plot_kws={"alpha": 0.5, "s": 2, "edgecolor": "b"},
            )
            charts.append(ChartData(
                title="Return Pair Plot",
                data=matplotlib_to_base64(pp.figure),
                chart_type="matplotlib",
                description="Scatter-plot matrix of daily returns",
            ))
        except Exception as e:
            logger.warning(f"Pair plot failed: {e}")

        return charts, tables

    # ================================================================
    # Statistical tests
    # ================================================================
    def _run_statistical_tests(
        self, prices: pd.DataFrame, col_names: list[str]
    ) -> tuple[list[ChartData], list[TableData], dict]:
        """ADF, Hurst, Variance Ratio, Half-Life, Cointegration."""
        charts: list[ChartData] = []
        tables: list[TableData] = []
        metrics: dict[str, Any] = {}

        adf_rows: list[dict] = []
        hurst_rows: list[dict] = []
        vr_rows: list[dict] = []
        hl_rows: list[dict] = []

        for col in col_names:
            s = prices[col]
            try:
                adf_stat, adf_p, adf_ok = perform_adf_test(s)
                adf_rows.append({
                    "Ticker": col.upper(), "ADF Stat": round(adf_stat, 4),
                    "p-value": round(adf_p, 4),
                    "Stationary": "Yes" if adf_ok else "No",
                })
            except Exception:
                adf_rows.append({"Ticker": col.upper(), "ADF Stat": "N/A", "p-value": "N/A", "Stationary": "N/A"})
            try:
                h_val, h_mr = perform_hurst_exp_test(s)
                hurst_rows.append({
                    "Ticker": col.upper(), "Hurst": round(h_val, 4),
                    "Mean Reverting": "Yes" if h_mr else "No",
                })
            except Exception:
                hurst_rows.append({"Ticker": col.upper(), "Hurst": "N/A", "Mean Reverting": "N/A"})
            try:
                vr_p, vr_ok = perform_variance_ratio_test(s)
                vr_rows.append({
                    "Ticker": col.upper(), "VR p-value": round(vr_p, 4),
                    "Not Random Walk": "Yes" if vr_ok else "No",
                })
            except Exception:
                vr_rows.append({"Ticker": col.upper(), "VR p-value": "N/A", "Not Random Walk": "N/A"})
            try:
                hl_val = half_life_v2(s) / PERIOD_PER_DAY
                if hl_val < 0:
                    hl_rows.append({"Ticker": col.upper(), "Half-Life": "N/A (trending)"})
                else:
                    hl_rows.append({"Ticker": col.upper(), "Half-Life": f"{round(hl_val)} days"})
            except Exception:
                hl_rows.append({"Ticker": col.upper(), "Half-Life": "N/A"})

        tables.append(TableData(title="ADF Test", data=adf_rows, columns=["Ticker", "ADF Stat", "p-value", "Stationary"]))
        tables.append(TableData(title="Hurst Exponent", data=hurst_rows, columns=["Ticker", "Hurst", "Mean Reverting"]))
        tables.append(TableData(title="Variance Ratio", data=vr_rows, columns=["Ticker", "VR p-value", "Not Random Walk"]))
        tables.append(TableData(title="Half-Life", data=hl_rows, columns=["Ticker", "Half-Life"]))

        # Cointegration tests (all pairs)
        coint_rows: list[dict] = []
        for i in range(len(col_names)):
            for j in range(i + 1, len(col_names)):
                try:
                    p_val, is_coint, stat = perform_coint_test(prices[col_names[i]], prices[col_names[j]])
                    coint_rows.append({
                        "Pair": f"{col_names[i].upper()} vs {col_names[j].upper()}",
                        "p-value": round(p_val, 4),
                        "Cointegrated": "Yes" if is_coint else "No",
                        "Statistic": round(stat, 4),
                    })
                except Exception:
                    coint_rows.append({
                        "Pair": f"{col_names[i].upper()} vs {col_names[j].upper()}",
                        "p-value": "N/A", "Cointegrated": "N/A", "Statistic": "N/A",
                    })
        tables.append(TableData(
            title="Cointegration Tests",
            data=coint_rows,
            columns=["Pair", "p-value", "Cointegrated", "Statistic"],
            description="Engle-Granger cointegration test (pairwise)",
        ))

        metrics["cointegration_tests"] = coint_rows
        return charts, tables, metrics
    # ================================================================
    # Two-asset portfolio
    # ================================================================
    def _build_two_asset_portfolio(
        self, prices: pd.DataFrame, col_names: list[str]
    ) -> tuple[list[ChartData], list[TableData], dict]:
        """Build and analyse a 2-asset spread portfolio."""
        charts: list[ChartData] = []
        tables: list[TableData] = []
        m: dict[str, Any] = {}

        X = prices[col_names[0]]
        Y = prices[col_names[1]]

        # Regression plot
        fig_reg = plot_reg_line(X, Y)
        charts.append(ChartData(
            title=f"Regression: {col_names[0].upper()} vs {col_names[1].upper()}",
            data=matplotlib_to_base64(fig_reg),
            chart_type="matplotlib",
        ))

        hedge_ratio, _ = plot_reg_pair(X, Y)
        m["hedge_ratio"] = float(hedge_ratio)

        portf = Y - hedge_ratio * X

        # Portfolio price series chart
        fig_p, ax_p = plt.subplots(figsize=(12, 6))
        ax_p.plot(portf.index, portf.values, color="steelblue")
        ax_p.set_title("2-Asset Portfolio Price Series")
        ax_p.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append(ChartData(
            title="2-Asset Portfolio Spread",
            data=matplotlib_to_base64(fig_p),
            chart_type="matplotlib",
        ))

        # Portfolio analysis
        try:
            adf_stat, adf_p, adf_ok = perform_adf_test(portf, False)
            m["adf_pvalue"] = round(adf_p, 4)
            m["adf_stationary"] = bool(adf_ok)
        except Exception:
            pass
        try:
            h_val, h_mr = perform_hurst_exp_test(portf, False)
            m["hurst"] = round(h_val, 4)
            m["hurst_mean_reverting"] = bool(h_mr)
        except Exception:
            pass
        try:
            vr_p, vr_ok = perform_variance_ratio_test(portf, 2, False)
            m["vr_pvalue"] = round(vr_p, 4)
        except Exception:
            pass
        try:
            m["half_life"] = round(float(half_life(portf) / PERIOD_PER_DAY), 2)
        except Exception:
            pass
        try:
            p_val, is_coint, stat = perform_coint_test(X, Y, False)
            m["coint_pvalue"] = round(p_val, 4)
            m["cointegrated"] = bool(is_coint)
        except Exception:
            pass

        # Trading strategy (linear mean reversion)
        try:
            lb = max(round(half_life(portf)), 5)
        except Exception:
            lb = 30
        qty = -(portf - portf.rolling(lb).mean()) / portf.rolling(lb).std()
        position = portf * qty
        pnl = position.pct_change().dropna()

        # PnL chart
        fig_pnl, ax_pnl = plt.subplots(figsize=(12, 5))
        ax_pnl.plot(pnl.index, pnl.values, color="navy", linewidth=0.5)
        ax_pnl.set_title("2-Asset Daily PnL")
        ax_pnl.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append(ChartData(
            title="2-Asset Daily PnL",
            data=matplotlib_to_base64(fig_pnl),
            chart_type="matplotlib",
        ))

        # Wealth / Drawdown charts
        dd = erk_drawdown(pnl)
        fig_wd, (ax_w, ax_d) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax_w.plot(dd.index, dd["Wealth"], color="darkgreen")
        ax_w.set_title("2-Asset Wealth Ratio")
        ax_w.grid(True, alpha=0.3)
        ax_d.fill_between(dd.index, dd["Drawdown"], 0, color="red", alpha=0.4)
        ax_d.set_title("2-Asset Drawdown")
        ax_d.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append(ChartData(
            title="2-Asset Wealth & Drawdown",
            data=matplotlib_to_base64(fig_wd),
            chart_type="matplotlib",
        ))

        # Summary stats table
        summary = erk_summary_stats(
            pnl.to_frame("2-Asset"), riskfree_rate=0.02, periods_per_year=PERIOD_PER_YEAR
        )
        summary_rows = summary.reset_index().rename(columns={"index": "Metric"}).to_dict(orient="records")
        tables.append(TableData(
            title="2-Asset Performance Summary",
            data=summary_rows,
            columns=list(summary.reset_index().columns),
        ))

        return charts, tables, m

    # ================================================================
    # Three-asset portfolio
    # ================================================================
    def _build_three_asset_portfolio(
        self, prices: pd.DataFrame,
    ) -> tuple[list[ChartData], list[TableData], dict, pd.Series]:
        """Build and analyse a 3+ asset Johansen portfolio."""
        charts: list[ChartData] = []
        tables: list[TableData] = []
        m: dict[str, Any] = {}

        try:
            jres = coint_johansen(prices, det_order=0, k_ar_diff=1)
            coeff = jres.evec[:, 0]
        except Exception as e:
            return charts, tables, {"error": str(e)}, None

        portf = (prices * coeff).sum(axis=1)
        m["johansen_weights"] = {c: round(float(w), 6) for c, w in zip(prices.columns, coeff)}

        # Portfolio chart
        fig_p, ax_p = plt.subplots(figsize=(12, 6))
        ax_p.plot(portf.index, portf.values, color="darkorange")
        ax_p.set_title(f"{len(prices.columns)}-Asset Portfolio Price Series (Johansen)")
        ax_p.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append(ChartData(
            title=f"{len(prices.columns)}-Asset Portfolio Spread",
            data=matplotlib_to_base64(fig_p),
            chart_type="matplotlib",
        ))

        # Analysis
        try:
            adf_stat, adf_p, adf_ok = perform_adf_test(portf, False)
            m["adf_pvalue"] = round(adf_p, 4)
            m["adf_stationary"] = bool(adf_ok)
        except Exception:
            pass
        try:
            h_val, h_mr = perform_hurst_exp_test(portf, False)
            m["hurst"] = round(h_val, 4)
        except Exception:
            pass
        try:
            vr_p, vr_ok = perform_variance_ratio_test(portf, 2, False)
            m["vr_pvalue"] = round(vr_p, 4)
        except Exception:
            pass
        try:
            m["half_life"] = round(float(half_life(portf) / PERIOD_PER_DAY), 2)
        except Exception:
            pass

        # Trading strategy
        try:
            lb = max(round(half_life(portf)), 5)
        except Exception:
            lb = 30
        qty = -(portf - portf.rolling(lb).mean()) / portf.rolling(lb).std()
        position = portf * qty
        pnl = position.pct_change().dropna()

        # Wealth / Drawdown
        dd = erk_drawdown(pnl)
        fig_wd, (ax_w, ax_d) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax_w.plot(dd.index, dd["Wealth"], color="darkgreen")
        ax_w.set_title(f"{len(prices.columns)}-Asset Wealth Ratio")
        ax_w.grid(True, alpha=0.3)
        ax_d.fill_between(dd.index, dd["Drawdown"], 0, color="red", alpha=0.4)
        ax_d.set_title(f"{len(prices.columns)}-Asset Drawdown")
        ax_d.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append(ChartData(
            title=f"{len(prices.columns)}-Asset Wealth & Drawdown",
            data=matplotlib_to_base64(fig_wd),
            chart_type="matplotlib",
        ))

        # Summary stats
        summary = erk_summary_stats(
            pnl.to_frame(f"{len(prices.columns)}-Asset"),
            riskfree_rate=0.02, periods_per_year=PERIOD_PER_YEAR,
        )
        summary_rows = summary.reset_index().rename(columns={"index": "Metric"}).to_dict(orient="records")
        tables.append(TableData(
            title=f"{len(prices.columns)}-Asset Performance Summary",
            data=summary_rows,
            columns=list(summary.reset_index().columns),
        ))

        return charts, tables, m, portf
    # ================================================================
    # Backtesting (backtesting.py library)
    # ================================================================
    def _run_backtesting(
        self,
        portfolio_series: pd.Series,
        lookback: int,
        threshold: float,
        stoploss: float,
        cash: float,
        commission: float,
        run_optimisation: bool,
    ) -> tuple[list[ChartData], list[TableData], dict, list[tuple[str, bytes]]]:
        """Run the Z_Score_Naive backtesting + optimisation."""
        charts: list[ChartData] = []
        tables: list[TableData] = []
        m: dict[str, Any] = {}
        html_artefacts: list[tuple[str, bytes]] = []

        # Build OHLC DataFrame for backtesting.py
        bt_port = portfolio_series.to_frame()
        bt_port.columns = ["Close"]
        bt_port["Open"] = bt_port["Close"]
        bt_port["High"] = bt_port["Close"]
        bt_port["Low"] = bt_port["Close"]

        bt = Backtest(bt_port, Z_Score_Naive, cash=cash, commission=commission, finalize_trades=True)

        # -- Initial run --
        stats = bt.run()
        m["initial_run"] = self._stats_to_dict(stats)

        initial_table = self._stats_to_table_rows(stats, "Initial Run")
        tables.append(TableData(
            title="Backtest: Initial Run",
            data=initial_table,
            columns=["Metric", "Value"],
            description=f"Z_Score_Naive(lookback={lookback}, threshold={threshold}, stoploss={stoploss})",
        ))

        # Capture bt.plot() as HTML bytes for MinIO
        html_bytes = self._capture_bt_plot(bt)
        if html_bytes:
            html_artefacts.append(("backtest_initial.html", html_bytes))

        # -- Parameter Optimisation --
        if run_optimisation:
            opt_configs = [
                ("Max Equity", {"maximize": "Equity Final [$]"}),
                ("Min Drawdown", {"maximize": "Max. Drawdown [%]"}),
                ("Min Volatility", {"maximize": neg_Volatility}),
                ("Max Sharpe", {"maximize": "Sharpe Ratio"}),
            ]
            for label, opt_kw in opt_configs:
                try:
                    opt_stats = bt.optimize(
                        lookback=range(20, 40, 5),
                        threshold=np.arange(2, 5.5, 0.5).tolist(),
                        stoploss=np.arange(0.001, 0.005, 0.001).tolist(),
                        **opt_kw,
                    )
                    m[f"opt_{label.lower().replace(' ', '_')}"] = self._stats_to_dict(opt_stats)

                    opt_table = self._stats_to_table_rows(opt_stats, label)
                    tables.append(TableData(
                        title=f"Optimised: {label}",
                        data=opt_table,
                        columns=["Metric", "Value"],
                        description=f"Strategy: {opt_stats._strategy}",
                    ))

                    # Trades table (for Max Equity only)
                    if label == "Max Equity":
                        trades_df = opt_stats["_trades"]
                        if trades_df is not None and len(trades_df) > 0:
                            trade_rows = trades_df.head(30).to_dict(orient="records")
                            tables.append(TableData(
                                title="Optimised Trades (Max Equity)",
                                data=trade_rows,
                                columns=list(trades_df.columns[:8]),
                                description=f"Strategy: {opt_stats._strategy}",
                            ))

                    # Capture optimised plot
                    html_bytes = self._capture_bt_plot(bt, plot_volume=False)
                    if html_bytes:
                        html_artefacts.append((f"backtest_opt_{label.lower().replace(' ', '_')}.html", html_bytes))

                except Exception as e:
                    logger.warning(f"Optimisation '{label}' failed: {e}")
                    m[f"opt_{label.lower().replace(' ', '_')}_error"] = str(e)

        return charts, tables, m, html_artefacts

    def _stats_to_dict(self, stats) -> dict:
        """Convert backtesting.py Stats series to a JSON-safe dict."""
        result = {}
        skip_keys = {"_strategy", "_equity_curve", "_trades"}
        for key in stats.index:
            if key in skip_keys:
                continue
            val = stats[key]
            try:
                if isinstance(val, (np.integer,)):
                    result[key] = int(val)
                elif isinstance(val, (np.floating, float)):
                    result[key] = round(float(val), 5) if not np.isnan(val) else None
                elif isinstance(val, (pd.Timedelta, timedelta)):
                    result[key] = str(val)
                else:
                    result[key] = str(val)
            except Exception:
                result[key] = str(val)
        return result

    def _stats_to_table_rows(self, stats, label: str) -> list[dict]:
        """Convert backtesting.py stats to table rows."""
        rows = []
        skip = {"_strategy", "_equity_curve", "_trades"}
        for key in stats.index:
            if key in skip:
                continue
            val = stats[key]
            try:
                if isinstance(val, float) and not np.isnan(val):
                    display = f"{val:.5f}"
                else:
                    display = str(val)
            except Exception:
                display = str(val)
            rows.append({"Metric": key, "Value": display})
        return rows

    def _capture_bt_plot(self, bt: Backtest, **plot_kwargs) -> Optional[bytes]:
        """Capture backtesting.py interactive HTML chart as bytes."""
        try:
            tmpdir = tempfile.mkdtemp()
            filepath = os.path.join(tmpdir, "bt_plot.html")
            bt.plot(filename=filepath, open_browser=False, **plot_kwargs)
            with open(filepath, "rb") as f:
                html_bytes = f.read()
            os.remove(filepath)
            os.rmdir(tmpdir)
            return html_bytes
        except Exception as e:
            logger.warning(f"Failed to capture bt.plot: {e}")
            return None
    # ================================================================
    # Signals & portfolio (for BaseStrategy compatibility)
    # ================================================================
    def _build_signals_df(
        self, portfolio_series: pd.Series, lookback: int, threshold: float
    ) -> pd.DataFrame:
        """Build a signals DataFrame matching BaseStrategy conventions."""
        df = pd.DataFrame(index=portfolio_series.index)
        df["Close"] = portfolio_series.values
        z = Z_Score(portfolio_series.values, lookback)
        df["z_score"] = z.values if isinstance(z, pd.Series) else z
        df["signals"] = 0
        df["positions"] = 0

        position = 0
        for i in range(lookback, len(df)):
            zv = df["z_score"].iat[i]
            if np.isnan(zv):
                df.iat[i, df.columns.get_loc("positions")] = position
                continue
            if position == 1 and zv > 0:
                df.iat[i, df.columns.get_loc("signals")] = -1
                position = 0
            elif position == -1 and zv < 0:
                df.iat[i, df.columns.get_loc("signals")] = 1
                position = 0
            if zv < -threshold and position != 1:
                df.iat[i, df.columns.get_loc("signals")] = 1
                position = 1
            elif zv > threshold and position != -1:
                df.iat[i, df.columns.get_loc("signals")] = -1
                position = -1
            df.iat[i, df.columns.get_loc("positions")] = position
        return df

    def _build_portfolio_df(
        self, signals: pd.DataFrame, capital: float
    ) -> pd.DataFrame:
        """Build equity curve from signals."""
        portfolio = pd.DataFrame(index=signals.index)
        portfolio["Close"] = signals["Close"]
        returns = signals["Close"].pct_change().fillna(0)
        pos_returns = signals["positions"].shift(1).fillna(0) * returns
        portfolio["returns"] = pos_returns
        portfolio["total_value"] = capital * (1 + pos_returns).cumprod()
        portfolio["positions"] = signals["positions"]
        return portfolio

    # ================================================================
    # Persistence -- MinIO
    # ================================================================
    def _save_to_minio(
        self,
        run_id: str,
        result: StrategyResult,
        html_artefacts: list[tuple[str, bytes]],
    ):
        """Persist charts and HTML artefacts to MinIO object storage."""
        if not _MINIO_AVAILABLE:
            logger.debug("MinIO not available -- skipping chart persistence")
            return

        try:
            minio_svc = get_minio_service()
            if not minio_svc.is_available:
                logger.debug("MinIO not reachable -- skipping")
                return

            # Save chart images
            saved = minio_svc.save_backtest_charts(
                run_id=run_id,
                charts=result.charts,
                strategy_name=self.name,
            )
            logger.info(f"MinIO: saved {len(saved)} chart images for run {run_id}")

            # Save HTML artefacts (backtesting.py interactive plots)
            for filename, html_bytes in html_artefacts:
                minio_svc.save_backtest_image(
                    run_id=run_id,
                    image_data=html_bytes,
                    filename=filename,
                    strategy_name=self.name,
                    chart_title=filename.replace(".html", "").replace("_", " ").title(),
                    chart_type="html",
                    content_type="text/html",
                )
            if html_artefacts:
                logger.info(f"MinIO: saved {len(html_artefacts)} HTML artefacts for run {run_id}")

        except Exception as e:
            logger.error(f"MinIO persistence failed: {e}")

    # ================================================================
    # Persistence -- Database
    # ================================================================
    def _save_to_database(
        self,
        result: StrategyResult,
        tickers: list[str],
        start_date: str,
        end_date: str,
        capital: float,
        params: dict,
    ):
        """Persist metrics and backtest results to PostgreSQL."""
        if not _DB_AVAILABLE:
            logger.debug("Database not available -- skipping persistence")
            return

        try:
            db = _DatabaseService()
            if not db.is_available:
                logger.debug("Database not reachable -- skipping")
                return

            backtest_data = {
                "strategy_name": self.name,
                "strategy_id": "crypto_mean_reversion",
                "tickers": [t.upper() for t in tickers],
                "start_date": datetime.strptime(start_date, "%Y-%m-%d"),
                "end_date": datetime.strptime(end_date, "%Y-%m-%d"),
                "initial_capital": capital,
                "parameters": params,
                "total_return": result.metrics.get("total_return"),
                "sharpe_ratio": result.metrics.get("sharpe_ratio"),
                "sortino_ratio": result.metrics.get("sortino_ratio"),
                "max_drawdown": result.metrics.get("max_drawdown"),
                "total_trades": result.metrics.get("total_trades", 0),
                "final_value": result.metrics.get("final_value"),
                "metrics": {
                    k: v for k, v in result.metrics.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            }
            db.save_backtest_result(backtest_data)
            logger.info("Database: saved backtest result")

        except Exception as e:
            logger.error(f"Database persistence failed: {e}")


# ================================================================
# Standalone test entry point
# ================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    strategy = CryptoMeanReversionStrategy()
    result = strategy.run(
        tickers=["ETH", "BTC", "LTC"],
        start_date="2020-01-01",
        end_date="2025-12-31",
        capital=10000,
        lookback=30,
        threshold=2.0,
        stoploss=0.001,
        run_optimisation=True,
    )

    print(f"\nSuccess: {result.success}")
    print(f"Execution time: {result.execution_time:.1f}s")
    print(f"Charts: {len(result.charts)}")
    print(f"Tables: {len(result.tables)}")

    if result.success:
        print(f"\nKey metrics:")
        for k in ("total_return", "sharpe_ratio", "max_drawdown", "total_trades", "final_value"):
            print(f"  {k}: {result.metrics.get(k)}")
        if "backtesting" in result.metrics:
            bt = result.metrics["backtesting"]
            if "initial_run" in bt:
                print(f"\nBacktest initial run:")
                for k, v in list(bt["initial_run"].items())[:10]:
                    print(f"  {k}: {v}")
    else:
        print(f"Error: {result.error_message}")