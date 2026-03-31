"""
Ruggiero Cybernetic Trading Strategies — From Murray A. Ruggiero's
"Cybernetic Trading Strategies: Developing a Profitable Trading
System with State-of-the-Art Technologies".

Implements:
  1. Intermarket Analysis — correlated markets as leading indicators
  2. Regime-Adaptive Indicators — auto-tune RSI/MACD per market regime
  3. Multi-Timeframe Confirmation — daily + weekly + monthly alignment
  4. Seasonal Patterns — calendar-based bias (month-of-year, day-of-week)
  5. Trend Strength Classification — ADX + DI+/DI- flow analysis

Integration:
  - IND: Intermarket signals (USD/INR, VIX, crude, US10Y) → forecast → Kite
  - US: Intermarket (VIX, DXY, US10Y, SPY) + seasonal → API → UI display
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class IntermarketSignal:
    """Signal from a correlated market."""
    driver_symbol: str          # e.g., "USDINR=X", "^VIX"
    target_symbol: str          # stock being analyzed
    correlation: float          # rolling correlation
    lead_bars: int              # how many bars driver leads target
    signal_direction: float     # -1.0 to +1.0
    confidence: float           # 0.0 to 1.0


@dataclass
class SeasonalBias:
    """Calendar-based seasonal trading bias."""
    month: int                  # 1-12
    day_of_week: int            # 0=Mon, 4=Fri
    monthly_bias: float         # Historical avg monthly return
    dow_bias: float             # Historical avg day-of-week return
    combined_bias: float        # -1.0 to +1.0  (positive = bullish)


@dataclass
class TrendClassification:
    """ADX-based trend strength classification."""
    adx: float                  # ADX value (0-100)
    di_plus: float              # +DI value
    di_minus: float             # -DI value
    trend_strength: str         # "STRONG_UP", "WEAK_UP", "RANGE", "WEAK_DOWN", "STRONG_DOWN"
    trend_score: float          # -1.0 to +1.0


@dataclass
class CyberneticAnalysis:
    """Complete Ruggiero cybernetic analysis for a symbol."""
    symbol: str
    intermarket_signals: List[IntermarketSignal]
    intermarket_forecast: float          # Combined intermarket forecast (-20 to +20)
    seasonal_bias: SeasonalBias
    seasonal_forecast: float             # Seasonal component (-5 to +5)
    trend_class: TrendClassification
    adaptive_rsi_period: int             # Regime-tuned RSI period
    adaptive_macd_fast: int              # Regime-tuned MACD fast
    adaptive_macd_slow: int              # Regime-tuned MACD slow
    multi_tf_alignment: float            # -1.0 to +1.0 (daily/weekly/monthly)
    composite_forecast: float            # Combined cybernetic forecast (-20 to +20)


# ═══════════════════════════════════════════════════════════════
# Intermarket Analysis (Ruggiero Ch. 2-3)
# ═══════════════════════════════════════════════════════════════

# Default intermarket drivers for IND stocks
IND_INTERMARKET_DRIVERS = {
    "USDINR=X":   {"direction": -1, "description": "INR weakness → equity bearish"},
    "^VIX":       {"direction": -1, "description": "Global fear → equity bearish"},
    "CL=F":       {"direction": -1, "description": "Oil up → cost pressure → bearish"},
    "^TNX":       {"direction": -1, "description": "US 10Y yield up → EM outflow"},
    "GC=F":       {"direction":  1, "description": "Gold up → safe haven bid (mixed)"},
}

# Default intermarket drivers for US stocks
US_INTERMARKET_DRIVERS = {
    "^VIX":       {"direction": -1, "description": "VIX up → equity bearish"},
    "DX-Y.NYB":   {"direction": -1, "description": "Dollar up → equity bearish"},
    "^TNX":       {"direction": -1, "description": "Yields up → equity bearish"},
    "^GSPC":      {"direction":  1, "description": "SPY trend = broad market direction"},
    "CL=F":       {"direction": -0.5, "description": "Oil up → mixed (sector dependent)"},
}


def _squeeze_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract column as flat 1D array, handling yfinance multi-index."""
    s = df[col]
    if hasattr(s, 'squeeze'):
        s = s.squeeze()
    vals = s.values.astype(float)
    return vals.flatten() if vals.ndim > 1 else vals


def compute_intermarket_signals(
    target_df: pd.DataFrame,
    driver_dfs: Dict[str, pd.DataFrame],
    driver_config: Optional[Dict] = None,
    lookback: int = 60,
    lead_test_range: range = range(1, 6),
) -> List[IntermarketSignal]:
    """
    Compute intermarket leading signals.

    Ruggiero Ch. 2: "Intermarket analysis provides the trader with
    leading indicators that can significantly improve timing."

    For each driver market, computes:
      1. Rolling correlation with target
      2. Optimal lead (1-5 bars) that maximizes correlation
      3. Signal direction based on driver's recent move

    Args:
        target_df: target stock DataFrame with 'Close' column
        driver_dfs: {driver_symbol: DataFrame with 'Close'}
        driver_config: driver direction/description mapping
        lookback: rolling correlation window
        lead_test_range: range of lead values to test

    Returns:
        List of IntermarketSignal for each driver
    """
    if driver_config is None:
        driver_config = IND_INTERMARKET_DRIVERS

    target_close = _squeeze_col(target_df, "Close")
    target_returns = np.diff(target_close) / np.maximum(np.abs(target_close[:-1]), 1e-10)

    signals = []

    for driver_sym, config in driver_config.items():
        if driver_sym not in driver_dfs:
            continue

        driver_close = _squeeze_col(driver_dfs[driver_sym], "Close")
        driver_returns = np.diff(driver_close) / np.maximum(np.abs(driver_close[:-1]), 1e-10)

        # Find optimal lead
        best_corr = 0.0
        best_lead = 1

        for lead in lead_test_range:
            n = min(len(target_returns), len(driver_returns) - lead)
            if n < lookback:
                continue

            # Driver returns shifted forward by `lead` bars
            dr = driver_returns[:n]
            tr = target_returns[lead:lead + n]

            n_calc = min(len(dr), len(tr))
            if n_calc < lookback:
                continue

            # Rolling correlation (last `lookback` bars)
            dr_window = dr[-lookback:]
            tr_window = tr[-lookback:]

            corr = float(np.corrcoef(dr_window, tr_window)[0, 1])
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lead = lead

        # Signal direction: recent driver move × expected relationship
        expected_dir = config["direction"]
        recent_driver = float(np.mean(driver_returns[-5:])) if len(driver_returns) >= 5 else 0.0
        signal_dir = recent_driver * expected_dir

        # Normalize to [-1, 1]
        signal_dir = max(-1.0, min(1.0, signal_dir * 100.0))

        confidence = min(1.0, abs(best_corr))

        signals.append(IntermarketSignal(
            driver_symbol=driver_sym,
            target_symbol="",
            correlation=best_corr,
            lead_bars=best_lead,
            signal_direction=signal_dir,
            confidence=confidence,
        ))

    return signals


def compute_intermarket_forecast(
    signals: List[IntermarketSignal],
) -> float:
    """
    Combine intermarket signals into a Carver-compatible forecast.

    Weighted by confidence (|correlation|).
    """
    if not signals:
        return 0.0

    total_weight = 0.0
    weighted_signal = 0.0

    for sig in signals:
        w = sig.confidence
        weighted_signal += sig.signal_direction * w * 10.0  # Scale to ±10
        total_weight += w

    if total_weight == 0:
        return 0.0

    raw = weighted_signal / total_weight
    return float(max(-20.0, min(20.0, raw)))


# ═══════════════════════════════════════════════════════════════
# Seasonal Patterns (Ruggiero Ch. 4)
# ═══════════════════════════════════════════════════════════════

def compute_seasonal_bias(
    df: pd.DataFrame,
    current_date: Optional[datetime] = None,
) -> SeasonalBias:
    """
    Calendar-based seasonal trading bias.

    Ruggiero Ch. 4: "Seasonality provides a probabilistic edge
    that can be incorporated into a trading system."

    Computes:
      - Monthly bias: avg return for the current month across history
      - Day-of-week bias: avg return for current day-of-week
    """
    if current_date is None:
        current_date = datetime.now()

    close = _squeeze_col(df, "Close")
    dates = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
    returns = pd.Series(np.diff(close) / np.maximum(np.abs(close[:-1]), 1e-10))

    if len(returns) < 60:
        return SeasonalBias(
            month=current_date.month, day_of_week=current_date.weekday(),
            monthly_bias=0.0, dow_bias=0.0, combined_bias=0.0,
        )

    # Add dates to returns
    returns.index = dates[1:]

    # Monthly bias
    monthly_returns = returns.groupby(returns.index.month).mean()
    current_month = current_date.month
    monthly_bias = float(monthly_returns.get(current_month, 0.0)) * 252  # Annualize

    # Day-of-week bias
    dow_returns = returns.groupby(returns.index.dayofweek).mean()
    current_dow = current_date.weekday()
    dow_bias = float(dow_returns.get(current_dow, 0.0)) * 252  # Annualize

    # Combine: 70% monthly, 30% day-of-week (monthly is more reliable)
    combined = 0.7 * np.sign(monthly_bias) * min(1.0, abs(monthly_bias) / 0.20) + \
               0.3 * np.sign(dow_bias) * min(1.0, abs(dow_bias) / 0.20)

    return SeasonalBias(
        month=current_month,
        day_of_week=current_dow,
        monthly_bias=monthly_bias,
        dow_bias=dow_bias,
        combined_bias=float(max(-1.0, min(1.0, combined))),
    )


# ═══════════════════════════════════════════════════════════════
# Trend Strength Classification (Ruggiero Ch. 5)
# ═══════════════════════════════════════════════════════════════

def classify_trend_strength(
    df: pd.DataFrame, period: int = 14
) -> TrendClassification:
    """
    ADX + DI+/DI- based trend classification.

    Ruggiero Ch. 5: "The ADX is the most reliable indicator for
    determining whether a market is trending."

    Classification:
      ADX > 25 + DI+ > DI-: STRONG_UP → full long exposure
      ADX > 25 + DI- > DI+: STRONG_DOWN → reduce/avoid
      ADX 15-25 + DI+ > DI-: WEAK_UP → moderate long
      ADX 15-25 + DI- > DI+: WEAK_DOWN → minimize exposure
      ADX < 15: RANGE → use mean-reversion instead of trend
    """
    high = _squeeze_col(df, "High")
    low = _squeeze_col(df, "Low")
    close = _squeeze_col(df, "Close")
    n = len(close)

    if n < period + 2:
        return TrendClassification(
            adx=0, di_plus=0, di_minus=0,
            trend_strength="RANGE", trend_score=0.0,
        )

    # True Range
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)

    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)

        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]

        dm_plus[i] = up if (up > down and up > 0) else 0.0
        dm_minus[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing
    atr = np.zeros(n)
    smooth_dm_plus = np.zeros(n)
    smooth_dm_minus = np.zeros(n)

    atr[period] = np.mean(tr[1:period + 1])
    smooth_dm_plus[period] = np.mean(dm_plus[1:period + 1])
    smooth_dm_minus[period] = np.mean(dm_minus[1:period + 1])

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        smooth_dm_plus[i] = (smooth_dm_plus[i - 1] * (period - 1) + dm_plus[i]) / period
        smooth_dm_minus[i] = (smooth_dm_minus[i - 1] * (period - 1) + dm_minus[i]) / period

    # DI+ and DI-
    di_plus_val = 100.0 * smooth_dm_plus[-1] / max(atr[-1], 1e-10)
    di_minus_val = 100.0 * smooth_dm_minus[-1] / max(atr[-1], 1e-10)

    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        dp = 100.0 * smooth_dm_plus[i] / max(atr[i], 1e-10)
        dm = 100.0 * smooth_dm_minus[i] / max(atr[i], 1e-10)
        dx[i] = 100.0 * abs(dp - dm) / max(dp + dm, 1e-10)

    adx_val = float(np.mean(dx[-period:]))

    # Classification
    if adx_val > 25:
        if di_plus_val > di_minus_val:
            strength = "STRONG_UP"
            score = min(1.0, adx_val / 50.0)
        else:
            strength = "STRONG_DOWN"
            score = -min(1.0, adx_val / 50.0)
    elif adx_val > 15:
        if di_plus_val > di_minus_val:
            strength = "WEAK_UP"
            score = 0.3
        else:
            strength = "WEAK_DOWN"
            score = -0.3
    else:
        strength = "RANGE"
        score = 0.0

    return TrendClassification(
        adx=adx_val,
        di_plus=di_plus_val,
        di_minus=di_minus_val,
        trend_strength=strength,
        trend_score=score,
    )


# ═══════════════════════════════════════════════════════════════
# Regime-Adaptive Indicator Parameters (Ruggiero Ch. 6-7)
# ═══════════════════════════════════════════════════════════════

def compute_adaptive_parameters(
    trend_class: TrendClassification,
) -> Dict[str, int]:
    """
    Auto-tune indicator parameters based on trend classification.

    Ruggiero: "The biggest mistake traders make is using the same
    indicator settings regardless of market conditions."

    STRONG_UP/DOWN: faster indicators (RSI 7, MACD 8/17)
    WEAK_UP/DOWN: medium indicators (RSI 14, MACD 12/26)
    RANGE: slower indicators (RSI 21, MACD 19/39)
    """
    if "STRONG" in trend_class.trend_strength:
        return {
            "rsi_period": 7,
            "macd_fast": 8,
            "macd_slow": 17,
            "macd_signal": 9,
            "bb_period": 15,
            "stoch_k": 9,
        }
    elif "WEAK" in trend_class.trend_strength:
        return {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "stoch_k": 14,
        }
    else:  # RANGE
        return {
            "rsi_period": 21,
            "macd_fast": 19,
            "macd_slow": 39,
            "macd_signal": 9,
            "bb_period": 30,
            "stoch_k": 21,
        }


# ═══════════════════════════════════════════════════════════════
# Multi-Timeframe Confirmation (Ruggiero Ch. 8)
# ═══════════════════════════════════════════════════════════════

def compute_multi_timeframe_alignment(
    daily_df: pd.DataFrame,
) -> float:
    """
    Measure alignment across daily, weekly, and monthly timeframes.

    Ruggiero Ch. 8: "Multi-timeframe analysis confirms trade direction.
    The higher timeframe takes precedence."

    Returns:
        -1.0 to +1.0 alignment score
        +1.0 = all timeframes bullish
        -1.0 = all timeframes bearish
        0.0 = conflicting timeframes
    """
    close = _squeeze_col(daily_df, "Close")
    n = len(close)

    if n < 60:
        return 0.0

    # Daily: 10d vs 20d SMA
    sma_10 = np.mean(close[-10:])
    sma_20 = np.mean(close[-20:])
    daily_bull = 1.0 if sma_10 > sma_20 else -1.0

    # Weekly: 5w vs 13w (approx 25d vs 65d)
    if n >= 65:
        sma_25 = np.mean(close[-25:])
        sma_65 = np.mean(close[-65:])
        weekly_bull = 1.0 if sma_25 > sma_65 else -1.0
    else:
        weekly_bull = 0.0

    # Monthly: 20d vs 60d
    if n >= 60:
        sma_20m = np.mean(close[-20:])
        sma_60m = np.mean(close[-60:])
        monthly_bull = 1.0 if sma_20m > sma_60m else -1.0
    else:
        monthly_bull = 0.0

    # Weighted: monthly (50%) > weekly (30%) > daily (20%)
    alignment = 0.2 * daily_bull + 0.3 * weekly_bull + 0.5 * monthly_bull

    return float(max(-1.0, min(1.0, alignment)))


# ═══════════════════════════════════════════════════════════════
# Composite Cybernetic Forecast
# ═══════════════════════════════════════════════════════════════

def compute_cybernetic_analysis(
    target_df: pd.DataFrame,
    driver_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    driver_config: Optional[Dict] = None,
    symbol: str = "UNKNOWN",
) -> CyberneticAnalysis:
    """
    Full Ruggiero cybernetic analysis for a single stock.

    Combines:
      1. Intermarket signals (40% of forecast)
      2. Trend classification + adaptive params (30%)
      3. Multi-timeframe alignment (20%)
      4. Seasonal bias (10%)

    Args:
        target_df: stock OHLCV DataFrame
        driver_dfs: {driver_symbol: OHLCV DataFrame}
        driver_config: intermarket driver configuration

    Returns:
        CyberneticAnalysis with all components and composite forecast
    """
    # Intermarket analysis
    if driver_dfs:
        im_signals = compute_intermarket_signals(
            target_df, driver_dfs, driver_config
        )
        im_forecast = compute_intermarket_forecast(im_signals)
    else:
        im_signals = []
        im_forecast = 0.0

    # Seasonal bias
    seasonal = compute_seasonal_bias(target_df)
    seasonal_forecast = seasonal.combined_bias * 5.0  # Scale to ±5

    # Trend classification
    trend_class = classify_trend_strength(target_df)
    adaptive_params = compute_adaptive_parameters(trend_class)

    # Multi-timeframe alignment
    mtf = compute_multi_timeframe_alignment(target_df)

    # Composite forecast
    # Intermarket: 40%, Trend: 30%, Multi-TF: 20%, Seasonal: 10%
    composite = (0.40 * im_forecast
                 + 0.30 * trend_class.trend_score * 10.0
                 + 0.20 * mtf * 10.0
                 + 0.10 * seasonal_forecast)

    composite = max(-20.0, min(20.0, composite))

    return CyberneticAnalysis(
        symbol=symbol,
        intermarket_signals=im_signals,
        intermarket_forecast=im_forecast,
        seasonal_bias=seasonal,
        seasonal_forecast=seasonal_forecast,
        trend_class=trend_class,
        adaptive_rsi_period=adaptive_params["rsi_period"],
        adaptive_macd_fast=adaptive_params["macd_fast"],
        adaptive_macd_slow=adaptive_params["macd_slow"],
        multi_tf_alignment=mtf,
        composite_forecast=composite,
    )


# ═══════════════════════════════════════════════════════════════
# Batch Processing for Pipeline Integration
# ═══════════════════════════════════════════════════════════════

def compute_cybernetic_forecast_batch(
    ohlcv_dict: Dict[str, pd.DataFrame],
    driver_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    driver_config: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute Ruggiero cybernetic forecast for multiple symbols.

    Args:
        ohlcv_dict: {symbol: OHLCV DataFrame}
        driver_dfs: {driver_symbol: OHLCV DataFrame}
        driver_config: intermarket driver config

    Returns:
        {symbol: forecast_value (-20 to +20)}
    """
    results = {}
    for symbol, df in ohlcv_dict.items():
        try:
            analysis = compute_cybernetic_analysis(
                df, driver_dfs, driver_config, symbol
            )
            results[symbol] = analysis.composite_forecast

            logger.info(
                "Cybernetic %s: intermarket=%.1f trend=%s mtf=%.1f seasonal=%.1f → forecast=%.1f",
                symbol, analysis.intermarket_forecast,
                analysis.trend_class.trend_strength,
                analysis.multi_tf_alignment,
                analysis.seasonal_forecast,
                analysis.composite_forecast,
            )
        except Exception as e:
            logger.error("Cybernetic analysis failed for %s: %s", symbol, e)
            results[symbol] = 0.0

    return results


def compute_cybernetic_analysis_batch(
    ohlcv_dict: Dict[str, pd.DataFrame],
    driver_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    driver_config: Optional[Dict] = None,
) -> Dict[str, Optional[CyberneticAnalysis]]:
    """
    Full cybernetic analysis for multiple symbols (for API/UI display).
    """
    results = {}
    for symbol, df in ohlcv_dict.items():
        try:
            results[symbol] = compute_cybernetic_analysis(
                df, driver_dfs, driver_config, symbol
            )
        except Exception as e:
            logger.error("Cybernetic analysis failed for %s: %s", symbol, e)
            results[symbol] = None
    return results
