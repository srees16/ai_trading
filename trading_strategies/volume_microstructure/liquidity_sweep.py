"""
Liquidity Sweep Strategy.

Detects stop-loss hunts by smart money: price wicks through key
support/resistance levels, sweeps resting liquidity (clusters of
stop-loss orders), then reverses.  This is a high-conviction
reversal pattern observable on both NSE and US equities.

Strategy Rules:
- Identify swing highs/lows over a lookback window.
- A **sweep** occurs when price briefly exceeds a swing high (sell-side)
  or undercuts a swing low (buy-side) then closes back within the range
  within the same or next bar.
- BUY after a bearish sweep below a swing low (stop-hunt → reversal up).
- SELL after a bullish sweep above a swing high (stop-hunt → reversal down).
- Confirmation via volume spike (>1.5× 20-day avg) at the sweep bar.

Works for IND and US stocks — liquidity sweeps are universal.
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
from strategies.utils import matplotlib_to_base64, create_metrics_summary


@StrategyRegistry.register_decorator
class LiquiditySweepStrategy(BaseStrategy):
    """
    Detects liquidity sweeps (stop-hunts) at key swing levels and
    trades the reversal.

    Parameters:
        swing_lookback (int): Bars to identify swing highs/lows (default: 10)
        vol_mult (float):     Volume spike multiplier threshold (default: 1.5)
        confirmation_bars (int): Bars after sweep for reversal confirmation (default: 2)
    """

    name = "Liquidity Sweep"
    description = "Detects stop-hunt sweeps at swing levels and trades the reversal"
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
                "description": "Volume spike multiplier over 20-day avg",
            },
            "confirmation_bars": {
                "type": "int", "default": 2, "min": 1, "max": 5,
                "description": "Bars after sweep for close-back confirmation",
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
        conf_bars = kwargs.get("confirmation_bars", 2)
        risk = self.get_risk_params(risk_params)

        data_service = DataService()
        all_charts, all_tables, all_metrics = [], [], {}
        combined_signals, combined_portfolio = [], []

        for ticker in tickers:
            try:
                df = data_service.get_ohlcv(ticker, start_date, end_date)
                if df.empty or len(df) < self.min_data_points:
                    continue

                signals = self._generate_signals(df, swing_lb, vol_mult, conf_bars)
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
                "parameters": {"swing_lookback": swing_lb, "vol_mult": vol_mult, "confirmation_bars": conf_bars},
                "tickers_processed": len(combined_signals),
            },
        )

    # ── signal generation ──────────────────────────────────────

    def _generate_signals(
        self,
        df: pd.DataFrame,
        swing_lb: int,
        vol_mult: float,
        conf_bars: int,
    ) -> pd.DataFrame:
        signals = df.copy()
        high = signals["High"]
        low = signals["Low"]
        close = signals["Close"]
        volume = signals["Volume"] if "Volume" in signals.columns else pd.Series(0, index=signals.index)

        # 20-day average volume
        avg_vol = volume.rolling(20, min_periods=5).mean()

        # Identify swing highs and swing lows using trailing (non-centered)
        # rolling window to avoid look-ahead bias.
        swing_high = high.rolling(window=swing_lb * 2 + 1, min_periods=swing_lb + 1).max()
        swing_low = low.rolling(window=swing_lb * 2 + 1, min_periods=swing_lb + 1).min()

        # Detect sweeps
        signals["positions"] = 0
        signals["signals"] = 0
        signals["sweep_type"] = ""

        for i in range(swing_lb * 2 + conf_bars, len(signals)):
            idx = signals.index[i]
            prev_swing_low = swing_low.iloc[i - conf_bars] if i >= conf_bars else np.nan
            prev_swing_high = swing_high.iloc[i - conf_bars] if i >= conf_bars else np.nan

            if pd.isna(prev_swing_low) or pd.isna(prev_swing_high):
                continue

            # Bearish sweep (buy signal): wick below swing low then close back above
            bars_to_check = range(max(0, i - conf_bars), i + 1)
            low_min = min(low.iloc[j] for j in bars_to_check)
            if low_min < prev_swing_low and close.iloc[i] > prev_swing_low:
                # Volume confirmation
                vol_ok = volume.iloc[i] > avg_vol.iloc[i] * vol_mult if avg_vol.iloc[i] > 0 else True
                if vol_ok:
                    signals.iloc[i, signals.columns.get_loc("positions")] = 1
                    signals.iloc[i, signals.columns.get_loc("sweep_type")] = "bear_sweep"

            # Bullish sweep (sell signal): wick above swing high then close back below
            high_max = max(high.iloc[j] for j in bars_to_check)
            if high_max > prev_swing_high and close.iloc[i] < prev_swing_high:
                vol_ok = volume.iloc[i] > avg_vol.iloc[i] * vol_mult if avg_vol.iloc[i] > 0 else True
                if vol_ok:
                    signals.iloc[i, signals.columns.get_loc("positions")] = 0
                    signals.iloc[i, signals.columns.get_loc("sweep_type")] = "bull_sweep"

        # Forward-fill positions, generate trade signals
        signals["positions"] = signals["positions"].replace(0, np.nan).ffill().fillna(0).astype(int)
        signals["signals"] = signals["positions"].diff().fillna(0)

        return signals

    # ── charting ───────────────────────────────────────────────

    def _create_charts(self, signals, portfolio, ticker, capital):
        charts = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(signals.index, signals["Close"], label=ticker, color="steelblue", lw=1)
        buys = signals[signals["signals"] == 1]
        sells = signals[signals["signals"] == -1]
        if not buys.empty:
            ax1.scatter(buys.index, buys["Close"], marker="^", color="lime", s=90, zorder=5, label="BUY (sweep)")
        if not sells.empty:
            ax1.scatter(sells.index, sells["Close"], marker="v", color="red", s=90, zorder=5, label="SELL (sweep)")
        ax1.set_title(f"{ticker} — Liquidity Sweep Signals")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        if "Volume" in signals.columns:
            ax2.bar(signals.index, signals["Volume"], color="gray", alpha=0.5, width=0.8)
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        charts.append(ChartData(
            title=f"{ticker} Liquidity Sweep", data=matplotlib_to_base64(fig),
            chart_type="matplotlib", description="Sweep signals with volume", ticker=ticker,
        ))
        return charts
