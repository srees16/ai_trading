"""
Volume Profile Strategy.

Constructs a volume-at-price histogram to identify high-volume nodes
(HVN) and low-volume nodes (LVN).  Institutional positions cluster at
HVN (support/resistance), and price tends to accelerate through LVN.

Strategy Rules:
- Build a volume profile over a rolling lookback window.
- **Point of Control (POC):** price level with the highest traded volume
  — acts as a strong magnet / support-resistance.
- **Value Area (VA):** price range containing 70 % of total volume.
- BUY when price pulls back to VA Low from above and holds
  (close > VA Low after touching within 0.3 %).
- SELL when price rallies to VA High from below and rejects
  (close < VA High after touching within 0.3 %).
- Extra confirmation: strong BUY if price reclaims POC from below;
  strong SELL if price loses POC from above.

Works for IND and US — uses daily OHLCV volume distribution.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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
class VolumeProfileStrategy(BaseStrategy):
    """
    Trades support/resistance derived from volume-at-price distribution.

    Parameters:
        profile_lookback (int): Bars for volume profile (default: 60)
        n_bins (int):           Price bins in the profile (default: 50)
        va_pct (float):         Value Area percentage (default: 0.70)
        touch_pct (float):      Proximity threshold (default: 0.003)
    """

    name = "Volume Profile"
    description = "Trades support/resistance from volume-at-price distribution"
    category = StrategyCategory.MEAN_REVERSION
    version = "1.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 80

    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        return {
            "profile_lookback": {
                "type": "int", "default": 60, "min": 20, "max": 120,
                "description": "Bars used to build volume profile",
            },
            "n_bins": {
                "type": "int", "default": 50, "min": 20, "max": 100,
                "description": "Number of price bins in profile histogram",
            },
            "va_pct": {
                "type": "float", "default": 0.70, "min": 0.50, "max": 0.90,
                "description": "Fraction of volume in value area",
            },
            "touch_pct": {
                "type": "float", "default": 0.003, "min": 0.001, "max": 0.01,
                "description": "Proximity threshold to VA/POC levels",
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

        profile_lb = kwargs.get("profile_lookback", 60)
        n_bins = kwargs.get("n_bins", 50)
        va_pct = kwargs.get("va_pct", 0.70)
        touch_pct = kwargs.get("touch_pct", 0.003)
        risk = self.get_risk_params(risk_params)

        data_service = DataService()
        all_charts, all_tables, all_metrics = [], [], {}
        combined_signals, combined_portfolio = [], []

        for ticker in tickers:
            try:
                df = data_service.get_ohlcv(ticker, start_date, end_date)
                if df.empty or len(df) < self.min_data_points:
                    continue

                signals = self._generate_signals(df, profile_lb, n_bins, va_pct, touch_pct)
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
                    "profile_lookback": profile_lb, "n_bins": n_bins,
                    "va_pct": va_pct, "touch_pct": touch_pct,
                },
                "tickers_processed": len(combined_signals),
            },
        )

    # ── volume profile computation ─────────────────────────────

    @staticmethod
    def _compute_profile(
        df: pd.DataFrame, n_bins: int, va_pct: float,
    ) -> Tuple[float, float, float]:
        """Compute POC, VA High, VA Low from a volume-at-price histogram.

        Returns (poc, va_high, va_low).
        """
        close = df["Close"].values
        volume = df["Volume"].values if "Volume" in df.columns else np.ones(len(df))

        price_min, price_max = close.min(), close.max()
        if price_min == price_max:
            return float(price_min), float(price_max), float(price_min)

        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_per_bin = np.zeros(n_bins)

        # Distribute each bar's volume across bins that its range covers
        high = df["High"].values if "High" in df.columns else close
        low = df["Low"].values if "Low" in df.columns else close
        for i in range(len(df)):
            bar_low, bar_high = low[i], high[i]
            for b in range(n_bins):
                if bin_edges[b + 1] >= bar_low and bin_edges[b] <= bar_high:
                    vol_per_bin[b] += volume[i]

        # POC: bin with the highest volume
        poc_idx = int(np.argmax(vol_per_bin))
        poc = float(bin_centers[poc_idx])

        # Value Area: expand from POC until va_pct of total volume covered
        total_vol = vol_per_bin.sum()
        if total_vol == 0:
            return poc, float(price_max), float(price_min)

        va_vol = vol_per_bin[poc_idx]
        lo_idx, hi_idx = poc_idx, poc_idx

        while va_vol / total_vol < va_pct:
            expand_lo = vol_per_bin[lo_idx - 1] if lo_idx > 0 else 0
            expand_hi = vol_per_bin[hi_idx + 1] if hi_idx < n_bins - 1 else 0
            if expand_lo >= expand_hi and lo_idx > 0:
                lo_idx -= 1
                va_vol += vol_per_bin[lo_idx]
            elif hi_idx < n_bins - 1:
                hi_idx += 1
                va_vol += vol_per_bin[hi_idx]
            else:
                break

        va_high = float(bin_edges[hi_idx + 1])
        va_low = float(bin_edges[lo_idx])

        return poc, va_high, va_low

    # ── signal generation ──────────────────────────────────────

    def _generate_signals(
        self, df, profile_lb, n_bins, va_pct, touch_pct,
    ) -> pd.DataFrame:
        signals = df.copy()
        close = signals["Close"]

        signals["poc"] = np.nan
        signals["va_high"] = np.nan
        signals["va_low"] = np.nan
        signals["positions"] = 0
        signals["signals"] = 0

        for i in range(profile_lb, len(signals)):
            window = signals.iloc[i - profile_lb: i]
            poc, va_hi, va_lo = self._compute_profile(window, n_bins, va_pct)
            signals.iloc[i, signals.columns.get_loc("poc")] = poc
            signals.iloc[i, signals.columns.get_loc("va_high")] = va_hi
            signals.iloc[i, signals.columns.get_loc("va_low")] = va_lo

            price = close.iloc[i]

            # Buy: price touches VA Low from above and holds
            if va_lo > 0 and abs(price - va_lo) / va_lo <= touch_pct and price >= va_lo:
                signals.iloc[i, signals.columns.get_loc("positions")] = 1
            # Strong buy: reclaim POC from below
            elif poc > 0 and price > poc and close.iloc[i - 1] <= poc:
                signals.iloc[i, signals.columns.get_loc("positions")] = 1
            # Sell: price touches VA High from below and rejects
            elif va_hi > 0 and abs(price - va_hi) / va_hi <= touch_pct and price <= va_hi:
                signals.iloc[i, signals.columns.get_loc("positions")] = 0
            # Strong sell: lose POC from above
            elif poc > 0 and price < poc and close.iloc[i - 1] >= poc:
                signals.iloc[i, signals.columns.get_loc("positions")] = 0

        # Forward-fill positions
        signals["positions"] = signals["positions"].replace(0, np.nan).ffill().fillna(0).astype(int)
        signals["signals"] = signals["positions"].diff().fillna(0)
        return signals

    # ── charting ───────────────────────────────────────────────

    def _create_charts(self, signals, portfolio, ticker, capital):
        charts = []
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(signals.index, signals["Close"], label=ticker, color="steelblue", lw=1)

        if "poc" in signals.columns:
            ax.plot(signals.index, signals["poc"], label="POC", color="gold", lw=1.2, ls="-.")
        if "va_high" in signals.columns:
            ax.plot(signals.index, signals["va_high"], label="VA High", color="salmon", lw=1, ls="--")
        if "va_low" in signals.columns:
            ax.plot(signals.index, signals["va_low"], label="VA Low", color="lightgreen", lw=1, ls="--")

        buys = signals[signals["signals"] == 1]
        sells = signals[signals["signals"] == -1]
        if not buys.empty:
            ax.scatter(buys.index, buys["Close"], marker="^", color="lime", s=90, zorder=5, label="BUY")
        if not sells.empty:
            ax.scatter(sells.index, sells["Close"], marker="v", color="red", s=90, zorder=5, label="SELL")

        ax.set_title(f"{ticker} — Volume Profile Strategy")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        charts.append(ChartData(
            title=f"{ticker} Volume Profile", data=matplotlib_to_base64(fig),
            chart_type="matplotlib", description="POC/VA with signals", ticker=ticker,
        ))
        return charts
