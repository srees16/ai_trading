"""
Mean Reversion Trading Strategy (Z-Score based).

Constructs a cointegrated portfolio from user-supplied tickers using
Johansen cointegration, then applies a Z-Score mean-reversion trading
strategy.  Statistical tests (ADF, Hurst, Variance Ratio, Half-Life)
are reported alongside performance analytics and charts.

Strategy Rules:
    1. Compute Z-Score of the portfolio price relative to its rolling mean/std.
    2. Buy when Z-Score < −threshold; Sell when Z-Score > +threshold.
    3. Close long when Z-Score > 0; close short when Z-Score < 0.
    4. Optional stop-loss on each trade.

Reference:
    - Letianzj: Cointegration Pairs Trading
    - QuantInsti: Johansen Test & Stationary Portfolio
    - Chee-Foong (original author, 15 Apr 2021)
"""

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.base_strategy import (
    BaseStrategy,
    StrategyResult,
    StrategyCategory,
    ChartData,
    TableData,
    RiskParams,
)
from strategies.registry import StrategyRegistry
from strategies.data_service import DataService
from strategies.utils import matplotlib_to_base64, dataframe_to_table
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

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────
PERIOD_PER_YEAR = 252
PERIOD_PER_DAY = 1


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _z_score_series(values: pd.Series, n: int) -> pd.Series:
    """Rolling Z-Score."""
    return (values - values.rolling(n).mean()) / values.rolling(n).std()


def _hedge_ratio_ols(independent: pd.Series, dependent: pd.Series) -> float:
    """Compute hedge ratio via OLS regression (no intercept)."""
    model = sm.OLS(dependent, independent)
    coeff = model.fit().params
    return float(coeff[0]) if len(coeff) >= 1 else 1.0


# ────────────────────────────────────────────────────────────
# Strategy
# ────────────────────────────────────────────────────────────
@StrategyRegistry.register_decorator
class MeanReversionStrategy(BaseStrategy):
    """
    Z-Score Mean Reversion Strategy.

    Builds a cointegrated portfolio from user-supplied tickers (2 or
    more), computes the Z-Score of the synthetic portfolio price, and
    trades mean-reversion signals.

    Parameters:
        lookback (int):   Rolling window for Z-Score (default: auto from half-life)
        threshold (float): Z-Score entry threshold (default: 2.0)
        stoploss (float):  Per-trade stop-loss fraction (default: 0.05)
    """

    name = "Mean Reversion (Z-Score)"
    description = (
        "Cointegrated portfolio with Z-Score mean-reversion signals "
        "— includes ADF, Hurst & Half-Life diagnostics"
    )
    category = StrategyCategory.STATISTICAL_ARBITRAGE
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 60

    # ── parameters ──────────────────────────────────────────
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
                "default": 0.0500,
                "min": 0.001,
                "max": 0.20,
                "description": "Per-trade stop-loss fraction (e.g. 0.05 = 5%)",
            },
        }

    # ── run ─────────────────────────────────────────────────
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

        # Validate
        try:
            self.validate_inputs(tickers, start_date, end_date, capital)
        except ValueError as e:
            return StrategyResult(success=False, error_message=str(e))

        if len(tickers) < 2:
            return StrategyResult(
                success=False,
                error_message="Mean reversion requires at least 2 tickers for cointegration analysis",
            )

        # ── Parameters ──
        lookback = int(kwargs.get("lookback", 30))
        threshold = float(kwargs.get("threshold", 2.0))
        stoploss = float(kwargs.get("stoploss", 0.05))
        risk = self.get_risk_params(risk_params)

        # ── Fetch Data ──
        data_service = DataService()
        price_frames: dict[str, pd.Series] = {}
        for t in tickers:
            try:
                df = data_service.get_ohlcv(t, start_date, end_date)
                if df.empty or len(df) < self.min_data_points:
                    continue
                price_frames[t] = df["Close"]
            except Exception:
                continue

        if len(price_frames) < 2:
            return StrategyResult(
                success=False,
                error_message=(
                    f"Need at least 2 tickers with sufficient data; "
                    f"got {len(price_frames)} ({', '.join(price_frames)})"
                ),
            )

        # ── Align prices ──
        prices = pd.DataFrame(price_frames).dropna()
        if len(prices) < self.min_data_points:
            return StrategyResult(
                success=False,
                error_message=f"Insufficient common data points ({len(prices)}). Need at least {self.min_data_points}.",
            )

        ticker_names = list(prices.columns)

        # ── Statistical Tests ──
        test_results = self._run_statistical_tests(prices)

        # ── Portfolio Construction ──
        if len(ticker_names) == 2:
            portfolio_series, method_desc = self._build_two_asset_portfolio(prices)
        else:
            portfolio_series, method_desc = self._build_multi_asset_portfolio(prices)

        # ── Lookback (auto from half-life if requested) ──
        try:
            hl = half_life(portfolio_series.values)
            hl_days = max(int(round(hl / PERIOD_PER_DAY)), 5)
        except Exception:
            hl_days = lookback
        if lookback <= 0:
            lookback = hl_days

        # ── Z-Score Trading Signals ──
        signals_df = self._generate_signals(portfolio_series, lookback, threshold, stoploss)

        # ── Portfolio Value ──
        portfolio_df = self._calculate_portfolio_value(signals_df, capital, risk)

        # ── Metrics ──
        metrics = self.calculate_metrics(portfolio_df, signals_df, capital)
        metrics["half_life_days"] = float(hl_days)
        metrics["lookback_used"] = lookback
        metrics["portfolio_method"] = method_desc
        metrics.update(test_results)

        # ── Charts ──
        charts = self._create_charts(
            prices, portfolio_series, signals_df, portfolio_df,
            ticker_names, lookback, threshold,
        )

        # ── Tables ──
        tables = self._create_tables(test_results, signals_df, ticker_names, metrics)

        execution_time = time.time() - start_time

        return StrategyResult(
            charts=charts,
            tables=tables,
            metrics=metrics,
            signals=signals_df,
            portfolio=portfolio_df,
            success=True,
            execution_time=execution_time,
            metadata={
                "strategy": self.name,
                "parameters": {
                    "lookback": lookback,
                    "threshold": threshold,
                    "stoploss": stoploss,
                },
                "tickers": ticker_names,
                "portfolio_method": method_desc,
            },
        )

    # ────────────────────────────────────────────────────────
    # Statistical tests
    # ────────────────────────────────────────────────────────
    def _run_statistical_tests(self, prices: pd.DataFrame) -> dict:
        """Run ADF, Hurst, Variance-Ratio for each ticker and pairs cointegration."""
        results: dict = {}
        tickers = list(prices.columns)

        # Per-ticker tests
        per_ticker = {}
        for t in tickers:
            try:
                adf_stat, adf_p, adf_stationary = perform_adf_test(prices[t])
                hurst_val, hurst_mr = perform_hurst_exp_test(prices[t])
                vr_p, vr_not_rw = perform_variance_ratio_test(prices[t])
                per_ticker[t] = {
                    "adf_pvalue": round(float(adf_p), 4),
                    "adf_stationary": bool(adf_stationary),
                    "hurst": round(float(hurst_val), 4),
                    "hurst_mean_reverting": bool(hurst_mr),
                    "var_ratio_pvalue": round(float(vr_p), 4),
                    "var_ratio_not_random": bool(vr_not_rw),
                }
            except Exception:
                per_ticker[t] = {"error": "test failed"}

        results["ticker_tests"] = per_ticker

        # Pair cointegration tests
        pair_tests = {}
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                try:
                    p_val, is_coint, stat = perform_coint_test(prices[tickers[i]], prices[tickers[j]])
                    pair_tests[f"{tickers[i]}/{tickers[j]}"] = {
                        "pvalue": round(float(p_val), 4),
                        "cointegrated": bool(is_coint),
                        "statistic": round(float(stat), 4),
                    }
                except Exception:
                    pair_tests[f"{tickers[i]}/{tickers[j]}"] = {"error": "test failed"}

        results["cointegration_tests"] = pair_tests
        return results

    # ────────────────────────────────────────────────────────
    # Portfolio construction
    # ────────────────────────────────────────────────────────
    def _build_two_asset_portfolio(self, prices: pd.DataFrame):
        """Build spread portfolio for two assets using OLS hedge ratio."""
        cols = list(prices.columns)
        X = prices[cols[0]]
        Y = prices[cols[1]]
        hedge_ratio = _hedge_ratio_ols(X, Y)
        portfolio = Y - hedge_ratio * X
        portfolio.name = "portfolio"
        desc = f"2-asset (hedge ratio={hedge_ratio:.4f})"
        return portfolio, desc

    def _build_multi_asset_portfolio(self, prices: pd.DataFrame):
        """Build portfolio using Johansen cointegration eigenvector weights."""
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen

            jres = coint_johansen(prices, det_order=0, k_ar_diff=1)
            coeff = jres.evec[:, 0]
            portfolio = (prices * coeff).sum(axis=1)
            portfolio.name = "portfolio"
            weight_str = ", ".join(f"{c}={w:.4f}" for c, w in zip(prices.columns, coeff))
            desc = f"Johansen ({weight_str})"
            return portfolio, desc
        except Exception:
            # Fallback: equal-weight spread of first two
            return self._build_two_asset_portfolio(prices)

    # ────────────────────────────────────────────────────────
    # Signal generation
    # ────────────────────────────────────────────────────────
    def _generate_signals(
        self,
        portfolio: pd.Series,
        lookback: int,
        threshold: float,
        stoploss: float,
    ) -> pd.DataFrame:
        """Generate Z-Score–based mean-reversion signals."""
        df = pd.DataFrame(index=portfolio.index)
        df["Close"] = portfolio.values
        df["z_score"] = _z_score_series(portfolio, lookback).values
        df["signals"] = 0
        df["positions"] = 0

        position = 0  # 1 = long, -1 = short, 0 = flat
        entry_price = None

        for i in range(lookback, len(df)):
            z = df["z_score"].iat[i]
            price = df["Close"].iat[i]

            if np.isnan(z):
                df.iat[i, df.columns.get_loc("positions")] = position
                continue

            # Stop-loss check
            if position != 0 and entry_price is not None and stoploss > 0:
                pct_change = (price / entry_price) - 1
                if (position == 1 and pct_change < -stoploss) or \
                   (position == -1 and -pct_change < -stoploss):
                    df.iat[i, df.columns.get_loc("signals")] = -position
                    position = 0
                    entry_price = None
                    df.iat[i, df.columns.get_loc("positions")] = position
                    continue

            # Exit rules (close long when z>0, close short when z<0)
            if position == 1 and z > 0:
                df.iat[i, df.columns.get_loc("signals")] = -1  # exit long
                position = 0
                entry_price = None
            elif position == -1 and z < 0:
                df.iat[i, df.columns.get_loc("signals")] = 1  # exit short
                position = 0
                entry_price = None

            # Entry rules
            if z < -threshold and position != 1:
                # Close opposing position if any, then go long
                if position == -1:
                    df.iat[i, df.columns.get_loc("signals")] = 1
                df.iat[i, df.columns.get_loc("signals")] = 1
                position = 1
                entry_price = price
            elif z > threshold and position != -1:
                if position == 1:
                    df.iat[i, df.columns.get_loc("signals")] = -1
                df.iat[i, df.columns.get_loc("signals")] = -1
                position = -1
                entry_price = price

            df.iat[i, df.columns.get_loc("positions")] = position

        return df

    # ────────────────────────────────────────────────────────
    # Portfolio value
    # ────────────────────────────────────────────────────────
    def _calculate_portfolio_value(
        self,
        signals: pd.DataFrame,
        capital: float,
        risk: RiskParams,
    ) -> pd.DataFrame:
        """Compute portfolio equity curve from signals."""
        portfolio = pd.DataFrame(index=signals.index)
        portfolio["Close"] = signals["Close"]

        returns = signals["Close"].pct_change().fillna(0)
        position_returns = signals["positions"].shift(1).fillna(0) * returns
        portfolio["returns"] = position_returns
        portfolio["total_value"] = capital * (1 + position_returns).cumprod()
        portfolio["positions"] = signals["positions"]

        return portfolio

    # ────────────────────────────────────────────────────────
    # Charts
    # ────────────────────────────────────────────────────────
    def _create_charts(
        self,
        prices: pd.DataFrame,
        portfolio_series: pd.Series,
        signals: pd.DataFrame,
        portfolio_df: pd.DataFrame,
        tickers: list[str],
        lookback: int,
        threshold: float,
    ) -> list[ChartData]:
        charts: list[ChartData] = []

        # ── Chart 1: Individual asset prices (normalised) ──
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        for t in tickers:
            norm = prices[t] / prices[t].iloc[0] * 100
            ax1.plot(norm.index, norm.values, label=t)
        ax1.set_title("Normalised Asset Prices (Base = 100)")
        ax1.set_ylabel("Price")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append(ChartData(
            title="Asset Prices",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Normalised closing prices of all assets in the portfolio",
        ))

        # ── Chart 2: Portfolio spread + Z-Score + Signals ──
        fig2, (ax2a, ax2b, ax2c) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Portfolio spread
        ax2a.plot(signals.index, signals["Close"], color="steelblue")
        ax2a.set_title("Portfolio Spread (Cointegrated)")
        ax2a.set_ylabel("Spread Value")
        ax2a.grid(True, alpha=0.3)

        # Z-Score with thresholds
        ax2b.plot(signals.index, signals["z_score"], color="purple", linewidth=0.8)
        ax2b.axhline(y=threshold, color="red", linestyle="--", label=f"+{threshold}")
        ax2b.axhline(y=-threshold, color="green", linestyle="--", label=f"−{threshold}")
        ax2b.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax2b.fill_between(signals.index, -threshold, threshold, alpha=0.07, color="gray")
        ax2b.set_title(f"Z-Score (lookback={lookback})")
        ax2b.set_ylabel("Z-Score")
        ax2b.legend(loc="best")
        ax2b.grid(True, alpha=0.3)

        # Position
        ax2c.fill_between(
            signals.index, 0, signals["positions"],
            where=signals["positions"] > 0, color="green", alpha=0.5, label="Long",
        )
        ax2c.fill_between(
            signals.index, 0, signals["positions"],
            where=signals["positions"] < 0, color="red", alpha=0.5, label="Short",
        )
        ax2c.set_title("Position")
        ax2c.set_xlabel("Date")
        ax2c.set_ylabel("Direction")
        ax2c.legend(loc="best")
        ax2c.grid(True, alpha=0.3)

        plt.tight_layout()
        charts.append(ChartData(
            title="Spread, Z-Score & Positions",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Portfolio spread, rolling Z-Score with entry thresholds, and position direction",
        ))

        # ── Chart 3: Equity Curve + Drawdown ──
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax3a.plot(portfolio_df.index, portfolio_df["total_value"], color="navy")
        ax3a.set_title("Equity Curve")
        ax3a.set_ylabel("Portfolio Value ($)")
        ax3a.grid(True, alpha=0.3)

        pnl = portfolio_df["total_value"].pct_change().dropna()
        if not pnl.empty:
            dd = erk_drawdown(pnl)
            ax3b.fill_between(dd.index, dd["Drawdown"], 0, color="red", alpha=0.4)
            ax3b.set_title("Drawdown")
            ax3b.set_ylabel("Drawdown (%)")
            ax3b.set_xlabel("Date")
            ax3b.grid(True, alpha=0.3)

        plt.tight_layout()
        charts.append(ChartData(
            title="Equity & Drawdown",
            data=matplotlib_to_base64(fig3),
            chart_type="matplotlib",
            description="Portfolio equity curve and underwater drawdown chart",
        ))

        return charts

    # ────────────────────────────────────────────────────────
    # Tables
    # ────────────────────────────────────────────────────────
    def _create_tables(
        self,
        test_results: dict,
        signals: pd.DataFrame,
        tickers: list[str],
        metrics: dict,
    ) -> list[TableData]:
        tables: list[TableData] = []

        # ── Table 1: Stationarity tests per ticker ──
        ticker_tests = test_results.get("ticker_tests", {})
        if ticker_tests:
            rows = []
            for t, vals in ticker_tests.items():
                if "error" in vals:
                    rows.append({"Ticker": t, "ADF p-value": "N/A", "Stationary": "N/A",
                                 "Hurst": "N/A", "Mean Reverting": "N/A",
                                 "VR p-value": "N/A", "Not Random": "N/A"})
                else:
                    rows.append({
                        "Ticker": t,
                        "ADF p-value": f"{vals['adf_pvalue']:.4f}",
                        "Stationary": "Yes" if vals["adf_stationary"] else "No",
                        "Hurst": f"{vals['hurst']:.4f}",
                        "Mean Reverting": "Yes" if vals["hurst_mean_reverting"] else "No",
                        "VR p-value": f"{vals['var_ratio_pvalue']:.4f}",
                        "Not Random": "Yes" if vals["var_ratio_not_random"] else "No",
                    })
            tables.append(TableData(
                title="Stationarity & Mean Reversion Tests",
                data=rows,
                columns=["Ticker", "ADF p-value", "Stationary", "Hurst",
                          "Mean Reverting", "VR p-value", "Not Random"],
                description="Per-ticker ADF, Hurst Exponent, and Variance Ratio tests",
            ))

        # ── Table 2: Pair cointegration results ──
        coint_tests = test_results.get("cointegration_tests", {})
        if coint_tests:
            rows = []
            for pair, vals in coint_tests.items():
                if "error" in vals:
                    rows.append({"Pair": pair, "p-value": "N/A",
                                 "Cointegrated": "N/A", "Statistic": "N/A"})
                else:
                    rows.append({
                        "Pair": pair,
                        "p-value": f"{vals['pvalue']:.4f}",
                        "Cointegrated": "Yes" if vals["cointegrated"] else "No",
                        "Statistic": f"{vals['statistic']:.4f}",
                    })
            tables.append(TableData(
                title="Cointegration Tests",
                data=rows,
                columns=["Pair", "p-value", "Cointegrated", "Statistic"],
                description="Engle-Granger cointegration test between each pair of tickers",
            ))

        # ── Table 3: Strategy parameters & diagnostics ──
        diag_rows = [
            {"Metric": "Half-Life (days)", "Value": str(metrics.get("half_life_days", "N/A"))},
            {"Metric": "Lookback", "Value": str(metrics.get("lookback_used", "N/A"))},
            {"Metric": "Portfolio Method", "Value": str(metrics.get("portfolio_method", "N/A"))},
            {"Metric": "Total Trades", "Value": str(metrics.get("total_trades", 0))},
        ]
        tables.append(TableData(
            title="Strategy Diagnostics",
            data=diag_rows,
            columns=["Metric", "Value"],
            description="Key diagnostic values for the backtest run",
        ))

        # ── Table 4: Recent signals ──
        signal_changes = signals[signals["signals"] != 0].tail(20)
        if not signal_changes.empty:
            sig_rows = signal_changes[["Close", "z_score", "signals"]].copy()
            sig_rows["Action"] = sig_rows["signals"].map({1: "Buy / Close Short", -1: "Sell / Close Long"})
            sig_rows["z_score"] = sig_rows["z_score"].round(3)
            sig_rows["Close"] = sig_rows["Close"].round(4)
            tables.append(TableData(
                title="Recent Trading Signals",
                data=sig_rows.reset_index().to_dict(orient="records"),
                columns=["Date", "Close", "z_score", "Action"],
                description="Most recent entry/exit signals",
            ))

        return tables


# ────────────────────────────────────────────────────────────
# Standalone entry point (for testing outside the UI)
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    strategy = MeanReversionStrategy()
    result = strategy.run(
        tickers=["GLD", "SLV"],
        start_date="2023-01-01",
        end_date="2025-01-01",
        capital=10000,
        lookback=30,
        threshold=2.0,
        stoploss=0.05,
    )
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    if result.success:
        print(f"Metrics: {result.metrics}")
    else:
        print(f"Error: {result.error_message}")
