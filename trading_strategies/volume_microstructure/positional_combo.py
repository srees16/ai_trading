"""
Positional Combo Strategy (Volume Profile + Anchored VWAP + Liquidity Sweep).

Ensemble strategy designed for **positional trades** (2–6 week hold).
Combines three complementary signals and requires 2-of-3 agreement.

Components:
    1. Volume Profile — institutional S/R from volume-at-price distribution.
    2. Anchored VWAP — dynamic S/R anchored to high-volume events.
    3. Liquidity Sweep — detects stop-hunt reversals for timing entry.

Signal Logic:
    BUY  when ≥2 of: (VP buy at VA Low, AVWAP bounce buy, sweep_buy)
    SELL when ≥2 of: (VP sell at VA High, AVWAP reject sell, sweep_sell)

Best suited for IND positional trades (2–6 week horizon).
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
    RiskParams,
)
from strategies.registry import StrategyRegistry
from strategies.data_service import DataService
from strategies.utils import matplotlib_to_base64, calculate_rsi


@StrategyRegistry.register_decorator
class PositionalComboStrategy(BaseStrategy):
    """
    Ensemble of Volume Profile + Anchored VWAP + Liquidity Sweep for
    positional trades.

    Parameters:
        profile_lookback (int): Bars for volume profile (default: 60)
        n_bins (int):           Price bins in profile (default: 50)
        va_pct (float):         Value Area percentage (default: 0.70)
        vol_pctile (float):     AVWAP anchor volume percentile (default: 95)
        touch_pct (float):      Proximity threshold (default: 0.005)
        swing_lookback (int):   Bars for sweep detection (default: 12)
        vol_mult (float):       Volume spike multiplier (default: 1.5)
        min_agreement (int):    Minimum sub-signals required (default: 2)
    """

    name = "Positional Combo"
    description = "Ensemble of Volume Profile + Anchored VWAP + Liquidity Sweep for positional trades"
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
                "description": "Value Area percentage of total volume",
            },
            "vol_pctile": {
                "type": "float", "default": 95, "min": 80, "max": 99,
                "description": "Volume percentile for AVWAP anchor event",
            },
            "touch_pct": {
                "type": "float", "default": 0.005, "min": 0.001, "max": 0.02,
                "description": "Proximity threshold for VP/AVWAP touch detection",
            },
            "swing_lookback": {
                "type": "int", "default": 12, "min": 5, "max": 30,
                "description": "Bars for swing pivot / sweep detection",
            },
            "vol_mult": {
                "type": "float", "default": 1.5, "min": 1.0, "max": 3.0,
                "description": "Volume spike multiplier for sweep confirmation",
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

        profile_lb = kwargs.get("profile_lookback", 60)
        n_bins = kwargs.get("n_bins", 50)
        va_pct = kwargs.get("va_pct", 0.70)
        vol_pctile = kwargs.get("vol_pctile", 95)
        touch_pct = kwargs.get("touch_pct", 0.005)
        swing_lb = kwargs.get("swing_lookback", 12)
        vol_mult = kwargs.get("vol_mult", 1.5)
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
                    df, profile_lb, n_bins, va_pct, vol_pctile,
                    touch_pct, swing_lb, vol_mult, min_agree,
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
                    "profile_lookback": profile_lb, "n_bins": n_bins,
                    "va_pct": va_pct, "vol_pctile": vol_pctile,
                    "touch_pct": touch_pct, "swing_lookback": swing_lb,
                    "vol_mult": vol_mult, "min_agreement": min_agree,
                },
                "tickers_processed": len(combined_signals),
            },
        )

    # ── sub-signal helpers ─────────────────────────────────────

    @staticmethod
    def _volume_profile_signals(
        df: pd.DataFrame, profile_lb: int, n_bins: int, va_pct: float, touch_pct: float,
    ) -> pd.Series:
        """Return +1 / -1 / 0 based on Volume Profile VA boundaries."""
        sig = pd.Series(0, index=df.index)
        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)

        for i in range(profile_lb, len(df)):
            window = df.iloc[i - profile_lb : i]
            w_close = window["Close"]
            w_vol = volume.iloc[i - profile_lb : i]

            price_min, price_max = w_close.min(), w_close.max()
            if price_max == price_min:
                continue

            bins = np.linspace(price_min, price_max, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_vol = np.zeros(n_bins)

            for j in range(len(window)):
                idx = int((w_close.iloc[j] - price_min) / (price_max - price_min) * (n_bins - 1))
                idx = min(idx, n_bins - 1)
                bin_vol[idx] += w_vol.iloc[j]

            total_vol = bin_vol.sum()
            if total_vol == 0:
                continue

            # Value Area: bins around POC containing va_pct of volume
            poc_idx = np.argmax(bin_vol)
            va_lo_idx, va_hi_idx = poc_idx, poc_idx
            va_vol = bin_vol[poc_idx]
            while va_vol < total_vol * va_pct:
                lo_next = bin_vol[va_lo_idx - 1] if va_lo_idx > 0 else 0
                hi_next = bin_vol[va_hi_idx + 1] if va_hi_idx < n_bins - 1 else 0
                if lo_next >= hi_next and va_lo_idx > 0:
                    va_lo_idx -= 1
                    va_vol += lo_next
                elif va_hi_idx < n_bins - 1:
                    va_hi_idx += 1
                    va_vol += hi_next
                else:
                    break

            va_low = bin_centers[va_lo_idx]
            va_high = bin_centers[va_hi_idx]
            cur_close = close.iloc[i]

            # Buy: price near VA Low and closing above it
            if abs(cur_close - va_low) / va_low < touch_pct and cur_close >= va_low:
                sig.iloc[i] = 1
            # Sell: price near VA High and closing below it
            elif abs(cur_close - va_high) / va_high < touch_pct and cur_close <= va_high:
                sig.iloc[i] = -1

        return sig

    @staticmethod
    def _anchored_vwap_signals(
        df: pd.DataFrame, vol_pctile: float, touch_pct: float,
    ) -> pd.Series:
        """Return +1 / -1 / 0 based on Anchored VWAP bounce/reject."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)

        # Find anchor: most recent bar with volume above percentile
        vol_threshold = volume.quantile(vol_pctile / 100.0)
        anchor_bars = volume[volume >= vol_threshold].index

        sig = pd.Series(0, index=df.index)
        if len(anchor_bars) == 0:
            return sig

        # Compute AVWAP from most recent anchor
        tp = (high + low + close) / 3
        for i in range(len(df)):
            # Find latest anchor at or before i
            valid_anchors = [a for a in anchor_bars if a <= df.index[i]]
            if not valid_anchors:
                continue
            anchor = valid_anchors[-1]
            anchor_loc = df.index.get_loc(anchor)
            cur_loc = df.index.get_loc(df.index[i])
            if cur_loc <= anchor_loc:
                continue

            slc = slice(anchor_loc, cur_loc + 1)
            cum_tp_vol = (tp.iloc[slc] * volume.iloc[slc]).sum()
            cum_vol = volume.iloc[slc].sum()
            if cum_vol == 0:
                continue
            avwap = cum_tp_vol / cum_vol

            cur_close = close.iloc[i]
            # Buy bounce: price touches AVWAP from above and holds
            if abs(cur_close - avwap) / avwap < touch_pct and cur_close > avwap:
                sig.iloc[i] = 1
            # Sell reject: price touches AVWAP from below and fails
            elif abs(cur_close - avwap) / avwap < touch_pct and cur_close < avwap:
                sig.iloc[i] = -1

        return sig

    @staticmethod
    def _sweep_signals(
        df: pd.DataFrame, swing_lb: int, vol_mult: float,
    ) -> pd.Series:
        """Return +1 / -1 / 0 based on liquidity sweep detection."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
        avg_vol = volume.rolling(20, min_periods=5).mean()

        swing_high = high.rolling(swing_lb * 2 + 1, min_periods=swing_lb + 1).max()
        swing_low = low.rolling(swing_lb * 2 + 1, min_periods=swing_lb + 1).min()

        buy_sweep = (low < swing_low.shift(1)) & (close > swing_low.shift(1)) & (volume > avg_vol * vol_mult)
        sell_sweep = (high > swing_high.shift(1)) & (close < swing_high.shift(1)) & (volume > avg_vol * vol_mult)

        sig = pd.Series(0, index=df.index)
        sig[buy_sweep] = 1
        sig[sell_sweep] = -1
        return sig

    # ── composite signal generation ────────────────────────────

    def _generate_signals(
        self,
        df: pd.DataFrame,
        profile_lb: int,
        n_bins: int,
        va_pct: float,
        vol_pctile: float,
        touch_pct: float,
        swing_lb: int,
        vol_mult: float,
        min_agree: int,
    ) -> pd.DataFrame:
        signals = df.copy()

        vp_sig = self._volume_profile_signals(df, profile_lb, n_bins, va_pct, touch_pct)
        avwap_sig = self._anchored_vwap_signals(df, vol_pctile, touch_pct)
        sweep_sig = self._sweep_signals(df, swing_lb, vol_mult)

        signals["vp_signal"] = vp_sig
        signals["avwap_signal"] = avwap_sig
        signals["sweep_signal"] = sweep_sig

        buy_count = (vp_sig == 1).astype(int) + (avwap_sig == 1).astype(int) + (sweep_sig == 1).astype(int)
        sell_count = (vp_sig == -1).astype(int) + (avwap_sig == -1).astype(int) + (sweep_sig == -1).astype(int)

        combo = pd.Series(0, index=df.index)
        combo[buy_count >= min_agree] = 1
        combo[sell_count >= min_agree] = -1
        signals["signal"] = combo

        return signals

    # ── portfolio & charts ─────────────────────────────────────

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
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

            # Price + signals
            ax = axes[0]
            ax.plot(signals.index, signals["Close"], label="Close", linewidth=0.8)
            buys = signals[signals["signal"] == 1]
            sells = signals[signals["signal"] == -1]
            ax.scatter(buys.index, buys["Close"], marker="^", color="green", s=40, label="BUY")
            ax.scatter(sells.index, sells["Close"], marker="v", color="red", s=40, label="SELL")
            ax.set_title(f"{ticker} — Positional Combo Signals")
            ax.legend(fontsize=7)

            # Equity
            ax = axes[1]
            ax.plot(portfolio.index, portfolio["equity"], color="blue", linewidth=0.8)
            ax.axhline(capital, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Portfolio Equity")

            plt.tight_layout()
            charts.append(ChartData(
                title=f"{ticker} Positional Combo",
                data=matplotlib_to_base64(fig),
                chart_type="matplotlib",
                ticker=ticker,
            ))
        except Exception:
            plt.close("all")
        return charts
