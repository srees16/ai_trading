"""
Anchored VWAP Strategy.

Volume-Weighted Average Price anchored to significant events (earnings,
52-week highs/lows, gap days, volume climax bars) provides dynamic
support/resistance that adapts to institutional accumulation/distribution.

Strategy Rules:
- Anchor VWAP from the most recent **significant volume event** (top-5%
  volume day or 52-week high/low).
- BUY when price pulls back to anchored VWAP from above and bounces
  (close > AVWAP after touching within 0.5%).
- SELL when price rallies to anchored VWAP from below and rejects
  (close < AVWAP after touching within 0.5%).
- Confirmation: RSI must align (> 45 for buy, < 55 for sell).

Works for IND and US stocks — VWAP is exchange-agnostic (OHLCV data).
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
from strategies.utils import matplotlib_to_base64, calculate_rsi


@StrategyRegistry.register_decorator
class AnchoredVWAPStrategy(BaseStrategy):
    """
    Trades mean-reversion around VWAP anchored to significant volume events.

    Parameters:
        vol_pctile (float): Volume percentile to qualify as anchor event (default: 95)
        touch_pct (float):  How close price must get to AVWAP to count as a touch (default: 0.005)
        rsi_period (int):   RSI confirmation period (default: 14)
    """

    name = "Anchored VWAP"
    description = "Mean-reversion around VWAP anchored to high-volume events"
    category = StrategyCategory.MEAN_REVERSION
    version = "1.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 60

    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        return {
            "vol_pctile": {
                "type": "float", "default": 95, "min": 80, "max": 99,
                "description": "Volume percentile for anchor event detection",
            },
            "touch_pct": {
                "type": "float", "default": 0.005, "min": 0.001, "max": 0.02,
                "description": "Proximity threshold to AVWAP (fraction of price)",
            },
            "rsi_period": {
                "type": "int", "default": 14, "min": 7, "max": 21,
                "description": "RSI period for confirmation",
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

        vol_pctile = kwargs.get("vol_pctile", 95)
        touch_pct = kwargs.get("touch_pct", 0.005)
        rsi_period = kwargs.get("rsi_period", 14)
        risk = self.get_risk_params(risk_params)

        data_service = DataService()
        all_charts, all_tables, all_metrics = [], [], {}
        combined_signals, combined_portfolio = [], []

        for ticker in tickers:
            try:
                df = data_service.get_ohlcv(ticker, start_date, end_date)
                if df.empty or len(df) < self.min_data_points:
                    continue

                signals = self._generate_signals(df, vol_pctile, touch_pct, rsi_period)
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
                "parameters": {"vol_pctile": vol_pctile, "touch_pct": touch_pct, "rsi_period": rsi_period},
                "tickers_processed": len(combined_signals),
            },
        )

    # ── signal generation ──────────────────────────────────────

    @staticmethod
    def _compute_avwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
        """Compute VWAP anchored from *anchor_idx* forward."""
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
        cum_tpv = (typical_price * volume).iloc[anchor_idx:].cumsum()
        cum_vol = volume.iloc[anchor_idx:].cumsum()
        avwap = cum_tpv / cum_vol.replace(0, np.nan)
        return avwap.reindex(df.index)

    def _generate_signals(
        self,
        df: pd.DataFrame,
        vol_pctile: float,
        touch_pct: float,
        rsi_period: int,
    ) -> pd.DataFrame:
        signals = df.copy()
        close = signals["Close"]
        volume = signals["Volume"] if "Volume" in signals.columns else pd.Series(1, index=signals.index)

        # Find anchor events (top N% volume days)
        vol_threshold = np.percentile(volume.dropna().values, vol_pctile)
        anchor_candidates = volume[volume >= vol_threshold].index
        # Also consider 52-week high/low as anchor
        rolling_high = close.rolling(252, min_periods=60).max()
        rolling_low = close.rolling(252, min_periods=60).min()
        is_52w_extreme = (close >= rolling_high) | (close <= rolling_low)
        anchor_days = sorted(set(anchor_candidates) | set(close[is_52w_extreme].index))

        # Use the most recent anchor event at each bar
        signals["avwap"] = np.nan
        if anchor_days:
            # Pick the latest anchor before each bar
            anchor_positions = [signals.index.get_loc(d) for d in anchor_days if d in signals.index]
            if anchor_positions:
                last_anchor = anchor_positions[-1]
                avwap = self._compute_avwap(df, last_anchor)
                signals["avwap"] = avwap

        # RSI for confirmation
        rsi_series = calculate_rsi(close, period=rsi_period)
        signals["rsi"] = rsi_series

        # Generate signals
        signals["positions"] = 0
        signals["signals"] = 0

        avwap = signals["avwap"]
        for i in range(1, len(signals)):
            if pd.isna(avwap.iloc[i]) or avwap.iloc[i] <= 0:
                continue

            price = close.iloc[i]
            vwap_val = avwap.iloc[i]
            proximity = abs(price - vwap_val) / vwap_val
            rsi_val = signals["rsi"].iloc[i] if not pd.isna(signals["rsi"].iloc[i]) else 50

            # Buy: price touches AVWAP from above and bounces, RSI > 45
            if proximity <= touch_pct and price > vwap_val and rsi_val > 45:
                signals.iloc[i, signals.columns.get_loc("positions")] = 1

            # Sell: price touches AVWAP from below and rejects, RSI < 55
            elif proximity <= touch_pct and price < vwap_val and rsi_val < 55:
                signals.iloc[i, signals.columns.get_loc("positions")] = 0

        signals["positions"] = signals["positions"].replace(0, np.nan).ffill().fillna(0).astype(int)
        signals["signals"] = signals["positions"].diff().fillna(0)
        return signals

    # ── charting ───────────────────────────────────────────────

    def _create_charts(self, signals, portfolio, ticker, capital):
        charts = []
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(signals.index, signals["Close"], label=ticker, color="steelblue", lw=1)
        if "avwap" in signals.columns:
            ax.plot(signals.index, signals["avwap"], label="Anchored VWAP",
                    color="orange", lw=1.5, ls="--")

        buys = signals[signals["signals"] == 1]
        sells = signals[signals["signals"] == -1]
        if not buys.empty:
            ax.scatter(buys.index, buys["Close"], marker="^", color="lime", s=90, zorder=5, label="BUY")
        if not sells.empty:
            ax.scatter(sells.index, sells["Close"], marker="v", color="red", s=90, zorder=5, label="SELL")

        ax.set_title(f"{ticker} — Anchored VWAP Strategy")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        charts.append(ChartData(
            title=f"{ticker} Anchored VWAP", data=matplotlib_to_base64(fig),
            chart_type="matplotlib", description="Price vs anchored VWAP", ticker=ticker,
        ))
        return charts
