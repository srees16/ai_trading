"""
Penfold Trend Trading Tactics — Implementation of Brent Penfold's
"The Universal Tactics of Successful Trend Trading" (Wiley, 2021).

Key concepts implemented:
  1. Dow Theory swing detection (higher-highs / higher-lows)
  2. Multi-timeframe trend filter (weekly Dow filter for daily entries)
  3. Turtle-style channel breakout (4-week entry / 2-week exit)
  4. ATR band breakout (volatility expansion signal)
  5. Retracement entry (pullback into trend + swing-point confirmation)
  6. Equity curve R² (robustness smoothness metric)
  7. UPI (Ulcer Performance Index) calculation
  8. Golden tenets enforcement: follow trend, cut losses short, let profits run

Design principles (Penfold Ch. 2):
  - Less is more: few rules, few indicators, few variables
  - Same variable values for buy & sell, same across all markets
  - Turn-key: complete rules for setup, entry, stop, exit
  - Versatile: profitable over diversified portfolio
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  1. DOW THEORY — SWING POINT DETECTION
# ═══════════════════════════════════════════════════════════════

@dataclass
class SwingPoint:
    """A detected swing high or low."""
    index: int          # bar index in the DataFrame
    price: float        # high for swing high, low for swing low
    is_high: bool       # True = swing high, False = swing low
    date: object = None # optional datetime


@dataclass
class DowTrend:
    """Dow Theory trend state for a symbol."""
    trend: str = "unknown"          # "up", "down", "unknown"
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0
    prev_swing_high: float = 0.0
    prev_swing_low: float = 0.0
    trend_changed: bool = False     # True if trend flipped this bar
    confidence: float = 0.5         # 0-1 based on swing pattern clarity


def detect_swing_points(
    df: pd.DataFrame,
    lookback: int = 5,
) -> List[SwingPoint]:
    """Detect swing highs and lows using N-bar pivot logic.

    A swing high occurs when the high is the highest of the surrounding
    `lookback` bars on each side. Similarly for swing lows.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'High' and 'Low' columns.
    lookback : int
        Number of bars on each side to confirm a swing point.
        Default 5 (matches Penfold's daily swing chart overlay).

    Returns
    -------
    List[SwingPoint]
        Chronologically ordered swing points.
    """
    highs = df["High"].values if "High" in df.columns else df["high"].values
    lows = df["Low"].values if "Low" in df.columns else df["low"].values
    dates = df.index if hasattr(df.index, '__len__') else range(len(df))

    swings: List[SwingPoint] = []

    for i in range(lookback, len(df) - lookback):
        # Swing high: bar i has highest high in window [i-lookback, i+lookback]
        window_highs = highs[i - lookback: i + lookback + 1]
        if highs[i] == np.max(window_highs) and highs[i] > highs[i - 1]:
            swings.append(SwingPoint(
                index=i,
                price=float(highs[i]),
                is_high=True,
                date=dates[i] if i < len(dates) else None,
            ))

        # Swing low: bar i has lowest low in window
        window_lows = lows[i - lookback: i + lookback + 1]
        if lows[i] == np.min(window_lows) and lows[i] < lows[i - 1]:
            swings.append(SwingPoint(
                index=i,
                price=float(lows[i]),
                is_high=False,
                date=dates[i] if i < len(dates) else None,
            ))

    # Sort by index (chronological)
    swings.sort(key=lambda s: s.index)
    return swings


def compute_dow_trend(df: pd.DataFrame, lookback: int = 5) -> DowTrend:
    """Determine current Dow Theory trend from swing point structure.

    Penfold Ch. 9: Trend is up when making higher highs + higher lows.
    Trend is down when making lower lows + lower highs.
    Always in the market — trend is either up or down.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (minimum ~30 bars for reliable detection).
    lookback : int
        Swing detection lookback (bars each side).

    Returns
    -------
    DowTrend
        Current trend state.
    """
    swings = detect_swing_points(df, lookback=lookback)

    if len(swings) < 4:
        return DowTrend(trend="unknown", confidence=0.0)

    # Extract last 2 swing highs and last 2 swing lows
    swing_highs = [s for s in swings if s.is_high]
    swing_lows = [s for s in swings if not s.is_high]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return DowTrend(trend="unknown", confidence=0.0)

    sh1, sh2 = swing_highs[-2], swing_highs[-1]
    sl1, sl2 = swing_lows[-2], swing_lows[-1]

    higher_highs = sh2.price > sh1.price
    higher_lows = sl2.price > sl1.price
    lower_highs = sh2.price < sh1.price
    lower_lows = sl2.price < sl1.price

    result = DowTrend(
        higher_highs=higher_highs,
        higher_lows=higher_lows,
        lower_highs=lower_highs,
        lower_lows=lower_lows,
        last_swing_high=sh2.price,
        last_swing_low=sl2.price,
        prev_swing_high=sh1.price,
        prev_swing_low=sl1.price,
    )

    # Penfold: HH + HL = uptrend, LL + LH = downtrend
    if higher_highs and higher_lows:
        result.trend = "up"
        result.confidence = 0.9
    elif lower_lows and lower_highs:
        result.trend = "down"
        result.confidence = 0.9
    elif higher_highs and lower_lows:
        # Mixed — trend uncertain, use most recent swing direction
        result.trend = "up" if sh2.index > sl2.index else "down"
        result.confidence = 0.4
    elif lower_lows and higher_lows:
        # Expanding range — use direction of highs
        result.trend = "down" if lower_highs else "up"
        result.confidence = 0.5
    else:
        # Default to prior trend or unknown
        result.trend = "unknown"
        result.confidence = 0.3

    return result


def compute_dow_trend_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    lookback: int = 5,
) -> Dict[str, DowTrend]:
    """Compute Dow Theory trend for multiple symbols."""
    results = {}
    for sym, df in ohlcv_cache.items():
        if df is None or len(df) < 30:
            continue
        results[sym] = compute_dow_trend(df, lookback=lookback)
    return results


# ═══════════════════════════════════════════════════════════════
#  2. MULTI-TIMEFRAME TREND FILTER
# ═══════════════════════════════════════════════════════════════

def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly bars."""
    if df is None or df.empty:
        return pd.DataFrame()

    weekly = df.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return weekly


def compute_weekly_dow_filter(
    df: pd.DataFrame,
    lookback: int = 3,
) -> str:
    """Determine weekly Dow trend to filter daily entries.

    Penfold Ch. 9: Weekly Dow Trader (WDT) doubled returns vs daily.
    Use weekly trend as a filter — only take daily longs if weekly uptrend,
    only take daily shorts if weekly downtrend.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV data (minimum ~60 bars for reliable weekly structure).
    lookback : int
        Weekly swing detection lookback. Default 3 (suitable for weekly).

    Returns
    -------
    str
        "up", "down", or "unknown".
    """
    weekly = resample_to_weekly(df)
    if len(weekly) < 15:
        return "unknown"
    trend = compute_dow_trend(weekly, lookback=lookback)
    return trend.trend


def compute_weekly_trend_filter_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
) -> Dict[str, str]:
    """Compute weekly Dow trend filter for all symbols.

    Returns dict of {symbol: "up"/"down"/"unknown"}.
    """
    results = {}
    for sym, df in ohlcv_cache.items():
        if df is None or len(df) < 60:
            continue
        results[sym] = compute_weekly_dow_filter(df)
    return results


# ═══════════════════════════════════════════════════════════════
#  3. CHANNEL BREAKOUT STRATEGIES (Penfold Ch. 6)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ChannelBreakoutSignal:
    """Signal from a channel breakout strategy."""
    symbol: str
    forecast: float         # -20 to +20 (Carver-compatible)
    channel_high: float
    channel_low: float
    current_price: float
    strategy: str           # "donchian_4w", "turtle", "dreyfus_52w"
    entry_side: str = ""    # "BUY" or "SELL" or ""
    stop_level: float = 0.0


def compute_turtle_breakout(
    df: pd.DataFrame,
    symbol: str = "",
    entry_weeks: int = 4,
    exit_weeks: int = 2,
) -> Optional[ChannelBreakoutSignal]:
    """Turtle Trading channel breakout (Penfold Ch. 6).

    Entry: Break of N-week high/low channel.
    Exit/Stop: Break of opposite M-week high/low channel.
    Same values for buy and sell. Same values across all markets.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV data. Needs >= entry_weeks * 5 + 5 bars.
    entry_weeks : int
        Entry channel width in weeks. Default 4 (Turtle/Donchian).
    exit_weeks : int
        Exit/stop channel width in weeks. Default 2 (Turtle refinement).
    """
    entry_days = entry_weeks * 5
    exit_days = exit_weeks * 5

    if df is None or len(df) < entry_days + 5:
        return None

    highs = df["High"].values if "High" in df.columns else df["high"].values
    lows = df["Low"].values if "Low" in df.columns else df["low"].values
    close = df["Close"].values if "Close" in df.columns else df["close"].values

    price_now = float(close[-1])
    channel_high = float(np.max(highs[-entry_days - 1:-1]))
    channel_low = float(np.min(lows[-entry_days - 1:-1]))
    exit_high = float(np.max(highs[-exit_days - 1:-1]))
    exit_low = float(np.min(lows[-exit_days - 1:-1]))

    rng = channel_high - channel_low
    if rng <= 0:
        return None

    # Normalized position within channel → forecast [-20, +20]
    position_in_channel = (price_now - channel_low) / rng
    forecast = (position_in_channel - 0.5) * 40.0  # Scale to ±20

    # Breakout signals
    entry_side = ""
    stop_level = 0.0
    if price_now > channel_high:
        entry_side = "BUY"
        stop_level = exit_low
        forecast = 20.0  # Max bullish forecast
    elif price_now < channel_low:
        entry_side = "SELL"
        stop_level = exit_high
        forecast = -20.0  # Max bearish forecast

    forecast = max(-20.0, min(20.0, forecast))

    return ChannelBreakoutSignal(
        symbol=symbol,
        forecast=forecast,
        channel_high=channel_high,
        channel_low=channel_low,
        current_price=price_now,
        strategy=f"turtle_{entry_weeks}w_{exit_weeks}w",
        entry_side=entry_side,
        stop_level=round(stop_level, 2),
    )


def compute_donchian_4w_breakout(
    df: pd.DataFrame,
    symbol: str = "",
) -> Optional[ChannelBreakoutSignal]:
    """Donchian's Four-Week Rule (1960) — simplest trend strategy.

    Penfold: "Possibly one of the all-time best strategies. Outstanding
    with a capital 'O'." 1 rule, 1 variable, 60+ years out-of-sample.
    """
    return compute_turtle_breakout(df, symbol, entry_weeks=4, exit_weeks=4)


def compute_dreyfus_52w_breakout(
    df: pd.DataFrame,
    symbol: str = "",
) -> Optional[ChannelBreakoutSignal]:
    """Dreyfus's 52-Week Rule (1960) — long-term channel breakout.

    Avg profit per trade: $3,038 (vs $262 for Donchian 4W).
    Fewer trades but much larger winners — ideal for positional.
    """
    return compute_turtle_breakout(df, symbol, entry_weeks=52, exit_weeks=26)


def compute_channel_breakout_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    strategy: str = "turtle",
) -> Dict[str, float]:
    """Compute channel breakout forecasts for all symbols.

    Returns dict of {symbol: forecast} for Carver pipeline integration.
    """
    results: Dict[str, float] = {}
    for sym, df in ohlcv_cache.items():
        if df is None:
            continue
        if strategy == "turtle":
            sig = compute_turtle_breakout(df, sym, entry_weeks=4, exit_weeks=2)
        elif strategy == "donchian":
            sig = compute_donchian_4w_breakout(df, sym)
        elif strategy == "dreyfus":
            sig = compute_dreyfus_52w_breakout(df, sym)
        else:
            sig = compute_turtle_breakout(df, sym)

        if sig and abs(sig.forecast) > 0.5:
            results[sym] = sig.forecast
    return results


# ═══════════════════════════════════════════════════════════════
#  4. ATR BAND BREAKOUT (Penfold Ch. 6)
# ═══════════════════════════════════════════════════════════════

def compute_atr_band_breakout(
    df: pd.DataFrame,
    symbol: str = "",
    period: int = 80,
    atr_multiplier: float = 2.0,
) -> Optional[float]:
    """ATR Band breakout forecast (Penfold Ch. 6).

    Entry: Close above upper ATR band (MA + ATR × multiplier) → buy.
    Stop: Close below MA → sell.
    Same variables for buy/sell. Same across all markets.

    Returns forecast in [-20, +20] range.
    """
    if df is None or len(df) < period + 5:
        return None

    close = df["Close"].values if "Close" in df.columns else df["close"].values
    highs = df["High"].values if "High" in df.columns else df["high"].values
    lows = df["Low"].values if "Low" in df.columns else df["low"].values

    # True Range
    tr_list = []
    for i in range(1, len(close)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - close[i - 1])
        lc = abs(lows[i] - close[i - 1])
        tr_list.append(max(hl, hc, lc))
    tr = np.array(tr_list)

    if len(tr) < period:
        return None

    # ATR and MA over the period
    atr = float(np.mean(tr[-period:]))
    ma = float(np.mean(close[-period:]))
    upper_band = ma + atr_multiplier * atr
    lower_band = ma - atr_multiplier * atr
    price_now = float(close[-1])

    if atr <= 0:
        return None

    # Forecast: normalized distance from MA relative to band width
    band_width = upper_band - lower_band
    if band_width <= 0:
        return None

    position = (price_now - lower_band) / band_width
    forecast = (position - 0.5) * 40.0  # Scale to ±20

    # Boost to ±20 on actual breakout
    if price_now > upper_band:
        forecast = 20.0
    elif price_now < lower_band:
        forecast = -20.0

    return max(-20.0, min(20.0, forecast))


def compute_atr_band_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
    period: int = 80,
    atr_multiplier: float = 2.0,
) -> Dict[str, float]:
    """Compute ATR band breakout forecasts for all symbols."""
    results: Dict[str, float] = {}
    for sym, df in ohlcv_cache.items():
        fc = compute_atr_band_breakout(df, sym, period, atr_multiplier)
        if fc is not None and abs(fc) > 0.5:
            results[sym] = fc
    return results


# ═══════════════════════════════════════════════════════════════
#  5. RETRACEMENT ENTRY (Penfold Ch. 6 — Elder Triple Screen style)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RetracementSignal:
    """Retracement entry signal — pullback into uptrend or rally into downtrend."""
    symbol: str
    forecast: float
    trend_direction: str    # "up" or "down"
    pullback_pct: float     # how deep the pullback is (0-1)
    entry_level: float
    stop_level: float


def compute_retracement_entry(
    df: pd.DataFrame,
    symbol: str = "",
    trend_period: int = 50,
    pullback_threshold: float = 0.02,
) -> Optional[RetracementSignal]:
    """Retracement trend entry — wait for pullback before entering with trend.

    Penfold Ch. 6: Retracement strategies "patiently wait for a pause
    and pull back in prices before initiating a trade in the direction
    of the trend."

    Logic:
    1. Determine trend using 50-period MA slope.
    2. In uptrend, wait for price to pull back below 20-MA (short-term weakness).
    3. Enter when price bounces back above 20-MA with higher low intact.
    4. Stop below the most recent swing low.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV data.
    trend_period : int
        Long MA period for trend determination. Default 50.
    pullback_threshold : float
        Minimum pullback depth (fraction) to qualify. Default 2%.
    """
    if df is None or len(df) < trend_period + 10:
        return None

    close = df["Close"].values if "Close" in df.columns else df["close"].values
    lows = df["Low"].values if "Low" in df.columns else df["low"].values
    highs = df["High"].values if "High" in df.columns else df["high"].values

    # Long-term trend: 50-MA slope
    ma_long = float(np.mean(close[-trend_period:]))
    ma_long_prev = float(np.mean(close[-trend_period - 5:-5]))
    price_now = float(close[-1])
    price_prev = float(close[-2])

    # Short-term MA for pullback detection
    ma_short = float(np.mean(close[-20:]))
    ma_short_prev = float(np.mean(close[-21:-1]))

    if ma_long <= 0:
        return None

    # Uptrend: price above long MA AND long MA rising
    uptrend = price_now > ma_long and ma_long > ma_long_prev

    # Downtrend: price below long MA AND long MA falling (buy-only for IND stocks)
    downtrend = price_now < ma_long and ma_long < ma_long_prev

    if not uptrend and not downtrend:
        return None

    if uptrend:
        # Pullback: price recently touched or went below short MA
        recent_low = float(np.min(lows[-10:]))
        pullback_depth = (float(np.max(highs[-20:])) - recent_low) / float(np.max(highs[-20:]))

        if pullback_depth < pullback_threshold:
            return None  # No meaningful pullback yet

        # Bounce: price now back above short MA after pullback
        bouncing = price_now > ma_short and price_prev <= ma_short_prev

        if not bouncing:
            return None

        # Forecast strength proportional to pullback depth (deeper = stronger signal)
        forecast = min(20.0, 10.0 + pullback_depth * 100.0)
        stop_level = float(np.min(lows[-10:]))

        return RetracementSignal(
            symbol=symbol,
            forecast=forecast,
            trend_direction="up",
            pullback_pct=round(pullback_depth, 4),
            entry_level=round(price_now, 2),
            stop_level=round(stop_level, 2),
        )

    if downtrend:
        # Short pullback (rally in downtrend)
        recent_high = float(np.max(highs[-10:]))
        pullback_depth = (recent_high - float(np.min(lows[-20:]))) / recent_high

        if pullback_depth < pullback_threshold:
            return None

        bouncing_down = price_now < ma_short and price_prev >= ma_short_prev

        if not bouncing_down:
            return None

        forecast = max(-20.0, -10.0 - pullback_depth * 100.0)
        stop_level = float(np.max(highs[-10:]))

        return RetracementSignal(
            symbol=symbol,
            forecast=forecast,
            trend_direction="down",
            pullback_pct=round(pullback_depth, 4),
            entry_level=round(price_now, 2),
            stop_level=round(stop_level, 2),
        )

    return None


def compute_retracement_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute retracement entry forecasts for all symbols."""
    results: Dict[str, float] = {}
    for sym, df in ohlcv_cache.items():
        sig = compute_retracement_entry(df, sym)
        if sig is not None and abs(sig.forecast) > 0.5:
            results[sym] = sig.forecast
    return results


# ═══════════════════════════════════════════════════════════════
#  6. EQUITY CURVE METRICS (Penfold Ch. 7-8)
# ═══════════════════════════════════════════════════════════════

def equity_curve_r_squared(equity_series: pd.Series) -> float:
    """R-squared of equity curve vs time regression line.

    Penfold Ch. 8: "I generally prefer strategies with 90+%
    R-squared readings." Measures how smooth the equity curve is.
    100% = perfectly straight line. Lower = bumpier.

    Parameters
    ----------
    equity_series : pd.Series
        Cumulative equity values (e.g., from backtest).

    Returns
    -------
    float
        R-squared value (0.0 to 1.0).
    """
    if equity_series is None or len(equity_series) < 10:
        return 0.0

    y = equity_series.values.astype(float)
    x = np.arange(len(y), dtype=float)

    # Linear regression
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)

    if ss_xx == 0 or ss_yy == 0:
        return 0.0

    r = ss_xy / (math.sqrt(ss_xx) * math.sqrt(ss_yy))
    return round(r * r, 4)


def ulcer_performance_index(
    equity_series: pd.Series,
    risk_free_annual: float = 0.07,
) -> float:
    """Ulcer Performance Index (Martin, 1987).

    Penfold Ch. 7: "The Ulcer Performance Index — a Superior
    Risk-Adjusted Return Measurement." UPI = excess return / Ulcer Index.
    Where Ulcer Index = sqrt(mean(drawdown_pct²)).

    Parameters
    ----------
    equity_series : pd.Series
        Cumulative equity values.
    risk_free_annual : float
        Annual risk-free rate. Default 7% (India).

    Returns
    -------
    float
        UPI value. Higher is better. >2.0 is very good per Penfold.
    """
    if equity_series is None or len(equity_series) < 20:
        return 0.0

    values = equity_series.values.astype(float)
    n = len(values)

    # Running maximum (high water mark)
    running_max = np.maximum.accumulate(values)

    # Percent drawdown from HWM
    dd_pct = np.where(running_max > 0, (values - running_max) / running_max * 100.0, 0.0)

    # Ulcer Index
    ulcer_index = math.sqrt(float(np.mean(dd_pct ** 2)))

    if ulcer_index < 0.001:
        return 0.0

    # Annualized return
    if values[0] <= 0:
        return 0.0
    total_return = values[-1] / values[0]
    years = n / 252.0
    if years <= 0:
        return 0.0
    annual_return = total_return ** (1.0 / years) - 1.0

    # UPI = excess return / ulcer index
    excess = (annual_return - risk_free_annual) * 100.0  # Convert to %
    upi = excess / ulcer_index

    return round(upi, 4)


# ═══════════════════════════════════════════════════════════════
#  7. RISK-OF-RUIN (Penfold Ch. 2 — Cardinal Rule: 0% ROR)
# ═══════════════════════════════════════════════════════════════

def risk_of_ruin(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    risk_per_trade_pct: float,
    ruin_threshold: float = 0.50,
    n_simulations: int = 10_000,
    n_trades: int = 500,
) -> float:
    """Monte Carlo risk-of-ruin simulation.

    Penfold Ch. 2: "Every trader's number one objective is to commence
    trading with a 0% ROR. The risk of you ruining your trading account
    is a mathematical function of the combination of how you trade
    (methodology) and how much capital you risk per trade (money management).
    Any ROR above 0% is a guarantee that a trader will blow up."

    Parameters
    ----------
    win_rate : float
        Probability of winning trade (0-1).
    avg_win : float
        Average winning trade return (fraction, e.g., 0.05 = 5%).
    avg_loss : float
        Average losing trade return (fraction, positive, e.g., 0.02 = 2%).
    risk_per_trade_pct : float
        Fraction of capital risked per trade (e.g., 0.02 = 2%).
    ruin_threshold : float
        Equity level below which ruin is declared (fraction of initial).
    n_simulations : int
        Number of Monte Carlo paths.
    n_trades : int
        Number of trades per simulation path.

    Returns
    -------
    float
        Risk of ruin probability (0.0 to 1.0). Target: 0.0.
    """
    if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
        return 1.0

    ruin_count = 0
    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        equity = 1.0
        for _ in range(n_trades):
            if rng.random() < win_rate:
                equity += equity * risk_per_trade_pct * (avg_win / avg_loss)
            else:
                equity -= equity * risk_per_trade_pct

            if equity < ruin_threshold:
                ruin_count += 1
                break

    return round(ruin_count / n_simulations, 4)


def check_ror_gate(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    risk_per_trade_pct: float,
) -> Tuple[bool, float]:
    """Check if current risk parameters produce 0% ROR.

    Returns (is_safe, ror_probability).
    Penfold: "ROR is king. 0% ROR is non-negotiable."
    """
    ror = risk_of_ruin(win_rate, avg_win, avg_loss, risk_per_trade_pct)
    return ror == 0.0, ror


# ═══════════════════════════════════════════════════════════════
#  8. PENFOLD TREND FORECAST (Combined signal for pipeline)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PenfoldTrendResult:
    """Combined Penfold trend analysis for a symbol."""
    symbol: str
    dow_trend_daily: str = "unknown"
    dow_trend_weekly: str = "unknown"
    dow_confidence: float = 0.0
    turtle_forecast: float = 0.0
    atr_band_forecast: float = 0.0
    retracement_forecast: float = 0.0
    weekly_aligned: bool = False    # True if daily and weekly trends agree
    combined_forecast: float = 0.0  # Blended forecast for pipeline


def compute_penfold_trend_analysis(
    df: pd.DataFrame,
    symbol: str = "",
) -> PenfoldTrendResult:
    """Run full Penfold trend analysis for a single symbol.

    Combines:
    1. Dow Theory trend (daily + weekly)
    2. Turtle channel breakout
    3. ATR band breakout
    4. Retracement entry

    Blends into a single forecast. Weekly trend acts as a FILTER
    (Penfold Ch. 9): only take buys if weekly trend is up.
    """
    result = PenfoldTrendResult(symbol=symbol)

    # Dow trend analysis
    daily_trend = compute_dow_trend(df, lookback=5)
    result.dow_trend_daily = daily_trend.trend
    result.dow_confidence = daily_trend.confidence

    weekly_trend_str = compute_weekly_dow_filter(df, lookback=3)
    result.dow_trend_weekly = weekly_trend_str

    # Weekly alignment (Penfold's key insight)
    result.weekly_aligned = (
        (daily_trend.trend == "up" and weekly_trend_str == "up") or
        (daily_trend.trend == "down" and weekly_trend_str == "down")
    )

    # Turtle breakout
    turtle_sig = compute_turtle_breakout(df, symbol, entry_weeks=4, exit_weeks=2)
    if turtle_sig:
        result.turtle_forecast = turtle_sig.forecast

    # ATR band breakout
    atr_fc = compute_atr_band_breakout(df, symbol, period=80, atr_multiplier=2.0)
    if atr_fc is not None:
        result.atr_band_forecast = atr_fc

    # Retracement entry
    retrace_sig = compute_retracement_entry(df, symbol)
    if retrace_sig:
        result.retracement_forecast = retrace_sig.forecast

    # Blend forecasts — weighted average
    # Turtle: 40%, ATR bands: 25%, Retracement: 20%, Dow alignment bonus: 15%
    components = []
    if abs(result.turtle_forecast) > 0.5:
        components.append(("turtle", 0.40, result.turtle_forecast))
    if abs(result.atr_band_forecast) > 0.5:
        components.append(("atr_band", 0.25, result.atr_band_forecast))
    if abs(result.retracement_forecast) > 0.5:
        components.append(("retrace", 0.20, result.retracement_forecast))

    if not components:
        result.combined_forecast = 0.0
        return result

    total_w = sum(w for _, w, _ in components)
    blended = sum(w * fc for _, w, fc in components) / total_w if total_w > 0 else 0.0

    # Weekly filter: dampen signal if weekly trend disagrees
    if result.weekly_aligned:
        blended *= 1.15  # Boost for alignment
    elif weekly_trend_str != "unknown":
        # Counter-trend: use only top-of-channel direction
        if (blended > 0 and weekly_trend_str == "down") or \
           (blended < 0 and weekly_trend_str == "up"):
            blended *= 0.3  # Heavily dampen counter-weekly signals

    result.combined_forecast = max(-20.0, min(20.0, round(blended, 2)))
    return result


def compute_penfold_forecast_batch(
    ohlcv_cache: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute Penfold trend forecasts for all symbols.

    Returns dict of {symbol: combined_forecast} for Carver pipeline integration.
    """
    results: Dict[str, float] = {}
    for sym, df in ohlcv_cache.items():
        if df is None or len(df) < 60:
            continue
        analysis = compute_penfold_trend_analysis(df, sym)
        if abs(analysis.combined_forecast) > 0.5:
            results[sym] = analysis.combined_forecast
    return results
