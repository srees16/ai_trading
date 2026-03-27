"""
Swing Combo Strategy (Liquidity Sweep + Order Flow + RSI).

Ensemble strategy designed for **swing trades** (3–10 day hold).
Combines three complementary signals and requires 2-of-3 agreement
for a trade signal, significantly reducing false positives.

Components:
    1. Liquidity Sweep — detects stop-hunt reversals at swing levels.
    2. Order Flow Imbalance — confirms institutional accumulation/distribution.
    3. RSI confirmation — momentum filter to avoid counter-trend entries.

Signal Logic:
    BUY  when ≥2 of: (sweep_buy, order_flow_buy, RSI < oversold)
    SELL when ≥2 of: (sweep_sell, order_flow_sell, RSI > overbought)

Best suited for IND swing trades (3–10 day horizon).
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.base_strategy import (
    BaseStrategy,
    StrategyResult,
    StrategyCategory,
    ChartData,
    RiskParams,
)
from strategies.registry import StrategyRegistry
from strategies.data_service import DataService
from strategies.utils import matplotlib_to_base64, calculate_rsi


@StrategyRegistry.register_decorator
class SwingComboStrategy(BaseStrategy):
    """
    Ensemble of Liquidity Sweep + Order Flow + RSI for swing trades.

    Parameters:
        swing_lookback (int): Bars to identify swing pivots (default: 10)
        vol_mult (float):     Volume spike multiplier (default: 1.5)
        obv_lookback (int):   OBV divergence window (default: 20)
        mfi_period (int):     MFI period (default: 14)
        rsi_period (int):     RSI period (default: 14)
        rsi_oversold (int):   RSI oversold threshold (default: 35)
        rsi_overbought (int): RSI overbought threshold (default: 65)
        min_agreement (int):  Minimum sub-signals agreeing (default: 2)
    """

    name = "Swing Combo"
    description = "Ensemble of Liquidity Sweep + Order Flow + RSI for swing trades"
    category = StrategyCategory.BREAKOUT
    version = "1.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 60

    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        return {
            "swing_lookback": {
                "type": "int", "default": 10, "min": 5, "max": 30,
                "description": "Bars to identify swing high/low pivots",
            },
            "vol_mult": {
                "type": "float", "default": 1.5, "min": 1.0, "max": 3.0,
                "description": "Volume spike multiplier for sweep confirmation",
            },
            "obv_lookback": {
                "type": "int", "default": 20, "min": 10, "max": 50,
                "description": "Bars for OBV divergence detection",
            },
            "mfi_period": {
                "type": "int", "default": 14, "min": 7, "max": 21,
                "description": "Money Flow Index period",
            },
            "rsi_period": {
                "type": "int", "default": 14, "min": 7, "max": 21,
                "description": "RSI period for confirmation",
            },
            "rsi_oversold": {
                "type": "int", "default": 35, "min": 20, "max": 45,
                "description": "RSI oversold threshold (buy zone)",
            },
            "rsi_overbought": {
                "type": "int", "default": 65, "min": 55, "max": 80,
                "description": "RSI overbought threshold (sell zone)",
            },
            "min_agreement": {
                "type": "int", "default": 2, "min": 2, "max": 3,
                "description": "Minimum sub-signals required to confirm trade",
            },
        }

    # ── run ────────────────────────────────────────────────────

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
        try:
            self.validate_inputs(tickers, start_date, end_date, capital)
        except ValueError as e:
            return StrategyResult(success=False, error_message=str(e))

        swing_lb = kwargs.get("swing_lookback", 10)
        vol_mult = kwargs.get("vol_mult", 1.5)
        obv_lb = kwargs.get("obv_lookback", 20)
        mfi_period = kwargs.get("mfi_period", 14)
        rsi_period = kwargs.get("rsi_period", 14)
        rsi_oversold = kwargs.get("rsi_oversold", 35)
        rsi_overbought = kwargs.get("rsi_overbought", 65)
        min_agree = kwargs.get("min_agreement", 2)
        risk = self.get_risk_params(risk_params)

        data_service = DataService()
        all_charts, all_metrics = [], {}
        combined_signals, combined_portfolio = [], []

        for ticker in tickers:
            try:
                df = data_service.get_ohlcv(ticker, start_date, end_date)
                if df.empty or len(df) < self.min_data_points:
                    continue

                signals = self._generate_signals(
                    df, swing_lb, vol_mult, obv_lb, mfi_period,
                    rsi_period, rsi_oversold, rsi_overbought, min_agree,
                )
                portfolio = self._calculate_portfolio(signals, capital, risk)
                charts = self._create_charts(signals, portfolio, ticker, capital)
                all_charts.extend(charts)

                metrics = self.calculate_metrics(portfolio, signals, capital)
                metrics["ticker"] = ticker
                all_metrics[ticker] = metrics

                signals["ticker"] = ticker
                portfolio["ticker"] = ticker
                combined_signals.append(signals)
                combined_portfolio.append(portfolio)
            except Exception as e:
                all_metrics[ticker] = {"error": str(e)}

        if all_metrics:
            valid = [m for m in all_metrics.values() if "error" not in m]
            if valid:
                all_metrics["aggregate"] = {
                    "avg_return": np.mean([m.get("total_return", 0) for m in valid]),
                    "avg_sharpe": np.mean([m.get("sharpe_ratio", 0) for m in valid]),
                    "total_tickers": len(valid),
                }

        return StrategyResult(
            charts=all_charts,
            metrics=all_metrics,
            signals=pd.concat(combined_signals) if combined_signals else None,
            portfolio=pd.concat(combined_portfolio) if combined_portfolio else None,
            success=len(combined_signals) > 0,
            error_message="" if combined_signals else "No valid data for any ticker",
            execution_time=time.time() - start_time,
            metadata={
                "strategy": self.name,
                "parameters": {
                    "swing_lookback": swing_lb, "vol_mult": vol_mult,
                    "obv_lookback": obv_lb, "mfi_period": mfi_period,
                    "rsi_period": rsi_period, "rsi_oversold": rsi_oversold,
                    "rsi_overbought": rsi_overbought, "min_agreement": min_agree,
                },
                "tickers_processed": len(combined_signals),
            },
        )

    # ── sub-signal helpers ─────────────────────────────────────

    @staticmethod
    def _sweep_signals(
        df: pd.DataFrame, swing_lb: int, vol_mult: float,
    ) -> pd.Series:
        """Return +1 (buy sweep) / -1 (sell sweep) / 0."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
        avg_vol = volume.rolling(20, min_periods=5).mean()

        swing_high = high.rolling(swing_lb * 2 + 1, min_periods=swing_lb + 1).max()
        swing_low = low.rolling(swing_lb * 2 + 1, min_periods=swing_lb + 1).min()

        # Bearish sweep below swing low → buy reversal
        buy_sweep = (low < swing_low.shift(1)) & (close > swing_low.shift(1)) & (volume > avg_vol * vol_mult)
        # Bullish sweep above swing high → sell reversal
        sell_sweep = (high > swing_high.shift(1)) & (close < swing_high.shift(1)) & (volume > avg_vol * vol_mult)

        sig = pd.Series(0, index=df.index)
        sig[buy_sweep] = 1
        sig[sell_sweep] = -1
        return sig

    @staticmethod
    def _order_flow_signals(
        df: pd.DataFrame, obv_lb: int, mfi_period: int,
    ) -> pd.Series:
        """Return +1 (accumulation) / -1 (distribution) / 0."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)

        # OBV
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_slope = obv.diff(obv_lb)
        price_slope = close.diff(obv_lb)
        # Bullish divergence: price down but OBV up
        obv_bull = (price_slope < 0) & (obv_slope > 0)
        # Bearish divergence: price up but OBV down
        obv_bear = (price_slope > 0) & (obv_slope < 0)

        # CVD proxy: close position within bar
        mid = (high + low) / 2
        cvd_bar = np.where(close > mid, volume, np.where(close < mid, -volume, 0))
        cvd = pd.Series(cvd_bar, index=df.index).cumsum()
        cvd_rising = cvd.diff(5) > 0

        # MFI
        tp = (high + low + close) / 3
        mf = tp * volume
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(mfi_period).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(mfi_period).sum()
        mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, 1e-10)))

        sig = pd.Series(0, index=df.index)
        sig[(obv_bull | cvd_rising) & (mfi < 40)] = 1
        sig[(obv_bear | ~cvd_rising) & (mfi > 60)] = -1
        return sig

    @staticmethod
    def _rsi_signals(
        close: pd.Series, rsi_period: int, oversold: int, overbought: int,
    ) -> pd.Series:
        """Return +1 / -1 / 0 based on RSI thresholds."""
        rsi = calculate_rsi(close, period=rsi_period)
        sig = pd.Series(0, index=close.index)
        sig[rsi < oversold] = 1
        sig[rsi > overbought] = -1
        return sig

    # ── composite signal generation ────────────────────────────

    def _generate_signals(
        self,
        df: pd.DataFrame,
        swing_lb: int,
        vol_mult: float,
        obv_lb: int,
        mfi_period: int,
        rsi_period: int,
        rsi_oversold: int,
        rsi_overbought: int,
        min_agree: int,
    ) -> pd.DataFrame:
        signals = df.copy()

        sweep = self._sweep_signals(df, swing_lb, vol_mult)
        oflow = self._order_flow_signals(df, obv_lb, mfi_period)
        rsi_sig = self._rsi_signals(df["Close"], rsi_period, rsi_oversold, rsi_overbought)

        signals["sweep_signal"] = sweep
        signals["oflow_signal"] = oflow
        signals["rsi_signal"] = rsi_sig

        # Count agreement
        buy_count = (sweep == 1).astype(int) + (oflow == 1).astype(int) + (rsi_sig == 1).astype(int)
        sell_count = (sweep == -1).astype(int) + (oflow == -1).astype(int) + (rsi_sig == -1).astype(int)

        combo = pd.Series(0, index=df.index)
        combo[buy_count >= min_agree] = 1
        combo[sell_count >= min_agree] = -1
        signals["signal"] = combo

        # RSI for charting
        signals["RSI"] = calculate_rsi(df["Close"], period=rsi_period)

        return signals

    # ── portfolio & charts (reuse base helpers) ────────────────

    def _calculate_portfolio(
        self, signals: pd.DataFrame, capital: float, risk: RiskParams,
    ) -> pd.DataFrame:
        portfolio = signals[["Close"]].copy()
        portfolio.rename(columns={"Close": "close"}, inplace=True)
        portfolio["position"] = signals["signal"].shift(1).fillna(0)
        portfolio["returns"] = portfolio["close"].pct_change().fillna(0)
        portfolio["strategy_returns"] = portfolio["position"] * portfolio["returns"]
        portfolio["equity"] = capital * (1 + portfolio["strategy_returns"]).cumprod()
        portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1
        return portfolio

    def _create_charts(
        self, signals: pd.DataFrame, portfolio: pd.DataFrame,
        ticker: str, capital: float,
    ) -> list[ChartData]:
        charts: list[ChartData] = []
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

            # Price + signals
            ax = axes[0]
            ax.plot(signals.index, signals["Close"], label="Close", linewidth=0.8)
            buys = signals[signals["signal"] == 1]
            sells = signals[signals["signal"] == -1]
            ax.scatter(buys.index, buys["Close"], marker="^", color="green", s=40, label="BUY")
            ax.scatter(sells.index, sells["Close"], marker="v", color="red", s=40, label="SELL")
            ax.set_title(f"{ticker} — Swing Combo Signals")
            ax.legend(fontsize=7)

            # RSI
            ax = axes[1]
            ax.plot(signals.index, signals["RSI"], color="purple", linewidth=0.8)
            ax.axhline(35, color="green", linestyle="--", alpha=0.5)
            ax.axhline(65, color="red", linestyle="--", alpha=0.5)
            ax.set_title("RSI")

            # Equity
            ax = axes[2]
            ax.plot(portfolio.index, portfolio["equity"], color="blue", linewidth=0.8)
            ax.axhline(capital, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Portfolio Equity")

            plt.tight_layout()
            charts.append(ChartData(
                title=f"{ticker} Swing Combo",
                data=matplotlib_to_base64(fig),
                chart_type="matplotlib",
                ticker=ticker,
            ))
        except Exception:
            plt.close("all")
        return charts
