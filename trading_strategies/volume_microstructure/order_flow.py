"""
Order Flow Imbalance Strategy.

Approximates institutional order-flow using OHLCV data (no tick data
needed).  Uses three complementary proxies:

1. **On-Balance Volume (OBV)** divergence — price makes new high but OBV
   doesn't = distribution; price makes new low but OBV doesn't =
   accumulation.
2. **Cumulative Volume Delta (CVD) proxy** — estimates buying vs selling
   pressure from bar classification (close > midpoint ≈ buying).
3. **Money Flow Index (MFI)** — volume-weighted RSI for overbought/
   oversold.

Strategy Rules:
- BUY when all three align bullish: OBV divergence bullish, CVD rising,
  MFI < 40 (oversold with accumulation).
- SELL when all three align bearish: OBV divergence bearish, CVD falling,
  MFI > 60 (overbought with distribution).

Works for IND and US — uses only OHLCV; no Level-2 data required.
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
    TableData,
    RiskParams,
)
from strategies.registry import StrategyRegistry
from strategies.data_service import DataService
from strategies.utils import matplotlib_to_base64


@StrategyRegistry.register_decorator
class OrderFlowStrategy(BaseStrategy):
    """
    Multi-factor order-flow imbalance strategy using OBV divergence,
    cumulative volume delta proxy, and Money Flow Index.

    Parameters:
        obv_lookback (int):  Bars for OBV divergence detection (default: 20)
        mfi_period (int):    Money Flow Index period (default: 14)
        mfi_oversold (int):  MFI oversold threshold (default: 40)
        mfi_overbought (int): MFI overbought threshold (default: 60)
        cvd_smooth (int):    CVD smoothing window (default: 10)
    """

    name = "Order Flow Imbalance"
    description = "Multi-factor order-flow strategy using OBV divergence, CVD proxy, and MFI"
    category = StrategyCategory.MOMENTUM
    version = "1.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 60

    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        return {
            "obv_lookback": {
                "type": "int", "default": 20, "min": 10, "max": 50,
                "description": "Bars for OBV divergence detection",
            },
            "mfi_period": {
                "type": "int", "default": 14, "min": 7, "max": 21,
                "description": "Money Flow Index period",
            },
            "mfi_oversold": {
                "type": "int", "default": 40, "min": 20, "max": 45,
                "description": "MFI oversold threshold (buy zone)",
            },
            "mfi_overbought": {
                "type": "int", "default": 60, "min": 55, "max": 80,
                "description": "MFI overbought threshold (sell zone)",
            },
            "cvd_smooth": {
                "type": "int", "default": 10, "min": 3, "max": 30,
                "description": "CVD smoothing period",
            },
        }

    # ── core ───────────────────────────────────────────────────

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

        obv_lb = kwargs.get("obv_lookback", 20)
        mfi_period = kwargs.get("mfi_period", 14)
        mfi_oversold = kwargs.get("mfi_oversold", 40)
        mfi_overbought = kwargs.get("mfi_overbought", 60)
        cvd_smooth = kwargs.get("cvd_smooth", 10)
        risk = self.get_risk_params(risk_params)

        data_service = DataService()
        all_charts, all_tables, all_metrics = [], [], {}
        combined_signals, combined_portfolio = [], []

        for ticker in tickers:
            try:
                df = data_service.get_ohlcv(ticker, start_date, end_date)
                if df.empty or len(df) < self.min_data_points:
                    continue

                signals = self._generate_signals(
                    df, obv_lb, mfi_period, mfi_oversold, mfi_overbought, cvd_smooth,
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
            tables=all_tables,
            metrics=all_metrics,
            signals=pd.concat(combined_signals) if combined_signals else None,
            portfolio=pd.concat(combined_portfolio) if combined_portfolio else None,
            success=len(combined_signals) > 0,
            error_message="" if combined_signals else "No valid data for any ticker",
            execution_time=time.time() - start_time,
            metadata={
                "strategy": self.name,
                "parameters": {
                    "obv_lookback": obv_lb, "mfi_period": mfi_period,
                    "mfi_oversold": mfi_oversold, "mfi_overbought": mfi_overbought,
                    "cvd_smooth": cvd_smooth,
                },
                "tickers_processed": len(combined_signals),
            },
        )

    # ── indicators ─────────────────────────────────────────────

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()

    @staticmethod
    def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: int) -> pd.Series:
        """Money Flow Index (volume-weighted RSI)."""
        typical = (high + low + close) / 3
        mf = typical * volume
        delta = typical.diff()
        pos_mf = mf.where(delta > 0, 0).rolling(period).sum()
        neg_mf = mf.where(delta < 0, 0).rolling(period).sum()
        ratio = pos_mf / neg_mf.replace(0, np.nan)
        return 100 - (100 / (1 + ratio))

    @staticmethod
    def _cvd_proxy(high: pd.Series, low: pd.Series, close: pd.Series,
                   volume: pd.Series, smooth: int) -> pd.Series:
        """Cumulative Volume Delta proxy from bar classification.

        Buying pressure ≈ volume × (close − low) / (high − low)
        Selling pressure ≈ volume × (high − close) / (high − low)
        """
        bar_range = (high - low).replace(0, np.nan)
        buy_pct = (close - low) / bar_range
        sell_pct = (high - close) / bar_range
        delta = volume * (buy_pct - sell_pct)
        cvd = delta.cumsum()
        return cvd.rolling(smooth, min_periods=1).mean()

    # ── signal generation ──────────────────────────────────────

    def _generate_signals(
        self, df, obv_lb, mfi_period, mfi_os, mfi_ob, cvd_smooth,
    ) -> pd.DataFrame:
        signals = df.copy()
        close = signals["Close"]
        high = signals["High"]
        low = signals["Low"]
        volume = signals["Volume"] if "Volume" in signals.columns else pd.Series(1, index=signals.index)

        obv = self._obv(close, volume)
        mfi = self._mfi(high, low, close, volume, mfi_period)
        cvd = self._cvd_proxy(high, low, close, volume, cvd_smooth)

        signals["obv"] = obv
        signals["mfi"] = mfi
        signals["cvd"] = cvd

        # OBV divergence: compare price new-high/low vs OBV new-high/low
        price_high = close.rolling(obv_lb).max()
        price_low = close.rolling(obv_lb).min()
        obv_high = obv.rolling(obv_lb).max()
        obv_low = obv.rolling(obv_lb).min()

        # Bullish divergence: price at low but OBV above prior low
        bull_div = (close <= price_low * 1.01) & (obv > obv_low)
        # Bearish divergence: price at high but OBV below prior high
        bear_div = (close >= price_high * 0.99) & (obv < obv_high)

        # CVD trend: rising = buying, falling = selling
        cvd_rising = cvd > cvd.shift(1)
        cvd_falling = cvd < cvd.shift(1)

        # MFI zones
        mfi_oversold = mfi < mfi_os
        mfi_overbought = mfi > mfi_ob

        # Combined signals — require at least 2 of 3 factors
        buy_score = bull_div.astype(int) + cvd_rising.astype(int) + mfi_oversold.astype(int)
        sell_score = bear_div.astype(int) + cvd_falling.astype(int) + mfi_overbought.astype(int)

        signals["positions"] = 0
        signals.loc[buy_score >= 2, "positions"] = 1
        signals.loc[sell_score >= 2, "positions"] = 0

        # Forward-fill positions
        signals["positions"] = signals["positions"].replace(0, np.nan).ffill().fillna(0).astype(int)
        signals["signals"] = signals["positions"].diff().fillna(0)
        return signals

    # ── charting ───────────────────────────────────────────────

    def _create_charts(self, signals, portfolio, ticker, capital):
        charts = []
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1, 1]})

        ax1, ax2, ax3 = axes
        ax1.plot(signals.index, signals["Close"], label=ticker, color="steelblue", lw=1)
        buys = signals[signals["signals"] == 1]
        sells = signals[signals["signals"] == -1]
        if not buys.empty:
            ax1.scatter(buys.index, buys["Close"], marker="^", color="lime", s=80, zorder=5, label="BUY")
        if not sells.empty:
            ax1.scatter(sells.index, sells["Close"], marker="v", color="red", s=80, zorder=5, label="SELL")
        ax1.set_title(f"{ticker} — Order Flow Imbalance Signals")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        ax2.plot(signals.index, signals["cvd"], color="purple", lw=1, label="CVD")
        ax2.set_ylabel("CVD")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

        ax3.plot(signals.index, signals["mfi"], color="teal", lw=1, label="MFI")
        ax3.axhline(40, color="green", ls="--", alpha=0.5)
        ax3.axhline(60, color="red", ls="--", alpha=0.5)
        ax3.set_ylabel("MFI")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()

        charts.append(ChartData(
            title=f"{ticker} Order Flow", data=matplotlib_to_base64(fig),
            chart_type="matplotlib", description="OBV/CVD/MFI signals", ticker=ticker,
        ))
        return charts
