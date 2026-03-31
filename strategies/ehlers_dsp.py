"""
Ehlers Digital Signal Processing — Advanced indicators from
John F. Ehlers' "Cybernetic Analysis for Stocks & Futures" and
"Rocket Science for Traders".

Implements zero-lag adaptive filters and leading indicators:
  1. Super Smoother (2-pole Butterworth digital filter)
  2. Fisher Transform (normalize price to Gaussian PDF)
  3. Instantaneous Trendline (zero-lag via cycle notching)
  4. Cyber Cycle Oscillator (pure cycle extraction)
  5. MAMA/FAMA (adaptive EMA via Hilbert Transform phase)
  6. Sinewave Indicator (leading turning point detector)
  7. Relative Vigor Index (RVI)
  8. Signal-to-Noise Ratio (SNR)
  9. Adaptive RSI (self-tuning period via dominant cycle)
  10. Dominant Cycle Period (Homodyne Discriminator)

All indicators produce Carver-compatible forecasts (avg abs ≈ 10, ±20).
Combined via compute_ehlers_composite_forecast() for the pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class EhlersAnalysis:
    """Complete Ehlers DSP analysis for a single symbol."""
    symbol: str
    super_smoother: float          # Current smoothed price
    fisher_transform: float        # Fisher value
    fisher_trigger: float          # Fisher[1] for crossover
    instantaneous_trendline: float # Zero-lag trend
    cyber_cycle: float             # Cycle oscillator value
    mama: float                    # MAMA value
    fama: float                    # FAMA value (slower adaptive)
    sinewave: float                # Sine component
    leadsine: float                # Lead sine (leading indicator)
    rvi: float                     # Relative Vigor Index
    rvi_signal: float              # RVI signal line
    snr: float                     # Signal-to-Noise Ratio (dB)
    adaptive_rsi: float            # Self-tuning RSI (0-100)
    dominant_cycle: float          # Measured cycle period (bars)
    composite_forecast: float      # Combined Ehlers forecast (-20 to +20)


# ═══════════════════════════════════════════════════════════════
# Core DSP Filters (Ehlers, Cybernetic Analysis Ch. 13)
# ═══════════════════════════════════════════════════════════════

def super_smoother(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Two-pole Butterworth digital filter — superior smoothing with
    minimal lag. Ehlers, Cybernetic Analysis Ch. 13, Eq. 13.10.

    Transfer function: flat passband below cutoff, steep rolloff above.
    Lag is approximately period/4 (vs period/2 for SMA).
    """
    n = len(series)
    if n < 3:
        return series.copy()

    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2.0 * a1 * math.cos(math.radians(1.414 * 180.0 / period))
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = (1.0 - b1 + a1 * a1) / 4.0

    vals = series.values.astype(float)
    out = np.empty(n)
    out[0] = vals[0]
    out[1] = vals[1] if n > 1 else vals[0]

    for i in range(2, n):
        out[i] = (coef1 * (vals[i] + 2.0 * vals[i - 1] + vals[i - 2])
                  + coef2 * out[i - 1] + coef3 * out[i - 2])

    return pd.Series(out, index=series.index, name="super_smoother")


def three_pole_super_smoother(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Three-pole Butterworth — even steeper rolloff for long-period smoothing.
    Ehlers, Cybernetic Analysis Ch. 13, Eq. 13.11.
    """
    n = len(series)
    if n < 4:
        return series.copy()

    a = math.exp(-math.pi / period)
    b = 2.0 * a * math.cos(math.radians(1.738 * 180.0 / period))
    c = a * a
    coef2 = b + c
    coef3 = -(c + b * c)
    coef4 = c * c
    coef1 = (1.0 - b + c) * (1.0 - c) / 8.0

    vals = series.values.astype(float)
    out = np.full(n, vals[0])

    for i in range(3, n):
        out[i] = (coef1 * (vals[i] + 3.0 * vals[i - 1]
                           + 3.0 * vals[i - 2] + vals[i - 3])
                  + coef2 * out[i - 1] + coef3 * out[i - 2]
                  + coef4 * out[i - 3])

    return pd.Series(out, index=series.index, name="3pole_super_smoother")


# ═══════════════════════════════════════════════════════════════
# Fisher Transform (Ehlers, Cybernetic Analysis Ch. 1)
# ═══════════════════════════════════════════════════════════════

def fisher_transform(series: pd.Series, period: int = 10
                     ) -> Tuple[pd.Series, pd.Series]:
    """
    Fisher Transform — normalizes price channel to Gaussian PDF.
    Returns (fisher, trigger) where trigger = fisher[1].

    Ehlers Ch. 1: "The turning points are not only sharp and distinct,
    but they also occur in a timely fashion."
    """
    n = len(series)
    vals = series.values.astype(float)
    fish = np.zeros(n)
    val1 = np.zeros(n)

    for i in range(period, n):
        window = vals[max(0, i - period + 1): i + 1]
        max_h = np.max(window)
        min_l = np.min(window)

        if max_h == min_l:
            raw = 0.0
        else:
            raw = 2.0 * ((vals[i] - min_l) / (max_h - min_l) - 0.5)

        # EMA smoothing (alpha=0.5)
        val1[i] = 0.5 * raw + 0.5 * val1[i - 1]

        # Clamp to avoid log(0)
        val1[i] = max(-0.9999, min(0.9999, val1[i]))

        # Fisher transform
        fish[i] = (0.25 * math.log((1.0 + val1[i]) / (1.0 - val1[i]))
                   + 0.5 * fish[i - 1])

    fisher_s = pd.Series(fish, index=series.index, name="fisher")
    trigger_s = pd.Series(np.roll(fish, 1), index=series.index, name="trigger")
    trigger_s.iloc[0] = 0.0

    return fisher_s, trigger_s


# ═══════════════════════════════════════════════════════════════
# Instantaneous Trendline (Ehlers, Cybernetic Analysis Ch. 2-3)
# ═══════════════════════════════════════════════════════════════

def instantaneous_trendline(series: pd.Series) -> pd.Series:
    """
    Zero-lag trendline via cycle-notch SMA.
    Ehlers Ch. 2: "By dividing the market into a trend component
    and a cycle component, I create a zero-lag cycle oscillator."

    Uses dominant cycle to set the SMA length, then applies
    4-tap FIR weighting to remove ringing.
    """
    n = len(series)
    vals = series.values.astype(float)
    dc = _compute_dominant_cycle_array(vals, n)
    itrend = np.copy(vals)

    for i in range(12, n):
        period = max(2, int(dc[i]))
        # Sum over dominant cycle period
        s = 0.0
        for j in range(period):
            if i - j >= 0:
                s += vals[i - j]
        if period > 0:
            s /= period
        # 4-tap FIR smoother to reduce ringing
        itrend[i] = (4.0 * s + 3.0 * itrend[i - 1]
                     + 2.0 * itrend[i - 2] + itrend[i - 3]) / 10.0

    return pd.Series(itrend, index=series.index, name="inst_trendline")


# ═══════════════════════════════════════════════════════════════
# Cyber Cycle Oscillator (Ehlers, Cybernetic Analysis Ch. 4)
# ═══════════════════════════════════════════════════════════════

def cyber_cycle(series: pd.Series, alpha: float = 0.07) -> pd.Series:
    """
    Extract pure cycle component via high-pass + smoothing.
    Ehlers Ch. 4: Difference between price and trend = cycle.
    """
    n = len(series)
    vals = series.values.astype(float)
    smooth = np.zeros(n)
    cycle = np.zeros(n)

    for i in range(3, n):
        smooth[i] = (vals[i] + 2.0 * vals[i - 1]
                     + 2.0 * vals[i - 2] + vals[i - 3]) / 6.0

    for i in range(6, n):
        cycle[i] = ((1.0 - 0.5 * alpha) ** 2
                    * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
                    + 2.0 * (1.0 - alpha) * cycle[i - 1]
                    - (1.0 - alpha) ** 2 * cycle[i - 2])

    return pd.Series(cycle, index=series.index, name="cyber_cycle")


# ═══════════════════════════════════════════════════════════════
# MAMA / FAMA (Ehlers, Rocket Science Ch. 17)
# ═══════════════════════════════════════════════════════════════

def mama_fama(series: pd.Series,
              fast_limit: float = 0.5,
              slow_limit: float = 0.05
              ) -> Tuple[pd.Series, pd.Series]:
    """
    MESA Adaptive Moving Average — adapts alpha via Hilbert Transform
    phase rate of change. Ehlers, Rocket Science Ch. 17.

    Returns (mama, fama) where:
      - MAMA: fast adaptive (hugs price at turning points)
      - FAMA: slow adaptive (acts as trigger line)

    "The variable alpha is guaranteed to be set to the FastLimit
     every half cycle due to the measured phase snap back."
    """
    n = len(series)
    vals = series.values.astype(float)

    smooth = np.zeros(n)
    detrend = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re_arr = np.zeros(n)
    im_arr = np.zeros(n)
    period = np.full(n, 6.0)
    smooth_period = np.full(n, 6.0)
    phase = np.zeros(n)
    mama_out = np.copy(vals)
    fama_out = np.copy(vals)

    for i in range(6, n):
        # 4-tap Gaussian filter
        smooth[i] = (4.0 * vals[i] + 3.0 * vals[i - 1]
                     + 2.0 * vals[i - 2] + vals[i - 3]) / 10.0

        adj = 0.075 * period[i - 1] + 0.54

        # Detrend via Hilbert Transform approximation
        detrend[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
                      - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6]) * adj

        # In-phase and Quadrature
        q1[i] = (0.0962 * detrend[i] + 0.5769 * detrend[i - 2]
                 - 0.5769 * detrend[i - 4] - 0.0962 * detrend[i - 6]) * adj
        i1[i] = detrend[i - 3]

        # Advance phase by 90 degrees
        ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i - 2]
                 - 0.5769 * i1[i - 4] - 0.0962 * i1[i - 6]) * adj
        jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i - 2]
                 - 0.5769 * q1[i - 4] - 0.0962 * q1[i - 6]) * adj

        # Phasor addition for 3-bar averaging
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]

        # Smooth I and Q
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        # Homodyne Discriminator
        re_arr[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im_arr[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re_arr[i] = 0.2 * re_arr[i] + 0.8 * re_arr[i - 1]
        im_arr[i] = 0.2 * im_arr[i] + 0.8 * im_arr[i - 1]

        if im_arr[i] != 0.0 and re_arr[i] != 0.0:
            period[i] = 360.0 / math.degrees(math.atan2(im_arr[i], re_arr[i]))
        else:
            period[i] = period[i - 1]

        period[i] = max(6.0, min(50.0, period[i]))
        period[i] = max(0.67 * period[i - 1], min(1.5 * period[i - 1], period[i]))

        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]

        # Phase
        if i1[i] != 0.0:
            phase[i] = math.degrees(math.atan2(q1[i], i1[i]))
        else:
            phase[i] = phase[i - 1]

        delta_phase = phase[i - 1] - phase[i]
        delta_phase = max(1.0, delta_phase)

        alpha = max(slow_limit, fast_limit / delta_phase)

        mama_out[i] = alpha * vals[i] + (1.0 - alpha) * mama_out[i - 1]
        fama_out[i] = 0.5 * alpha * mama_out[i] + (1.0 - 0.5 * alpha) * fama_out[i - 1]

    return (pd.Series(mama_out, index=series.index, name="mama"),
            pd.Series(fama_out, index=series.index, name="fama"))


# ═══════════════════════════════════════════════════════════════
# Sinewave Indicator (Ehlers, Cybernetic Analysis Ch. 11)
# ═══════════════════════════════════════════════════════════════

def sinewave_indicator(series: pd.Series
                       ) -> Tuple[pd.Series, pd.Series]:
    """
    Leading indicator for cycle turning points.
    Returns (sine, leadsine) where leadsine leads by ~quarter cycle.

    Ehlers Ch. 11: "The Sinewave Indicator provides a clear,
    unambiguous buy and sell signal."

    Buy when sine crosses above leadsine; sell on cross below.
    """
    n = len(series)
    vals = series.values.astype(float)
    dc = _compute_dominant_cycle_array(vals, n)

    sine = np.zeros(n)
    leadsine = np.zeros(n)

    for i in range(1, n):
        dc_period = max(2.0, dc[i])
        # Current phase position in the cycle
        phase_rad = 2.0 * math.pi / dc_period
        sine[i] = math.sin(phase_rad * (i % int(dc_period)))
        leadsine[i] = math.sin(phase_rad * (i % int(dc_period)) + math.pi / 4.0)

    return (pd.Series(sine, index=series.index, name="sine"),
            pd.Series(leadsine, index=series.index, name="leadsine"))


# ═══════════════════════════════════════════════════════════════
# Relative Vigor Index (Ehlers, Cybernetic Analysis Ch. 6)
# ═══════════════════════════════════════════════════════════════

def relative_vigor_index(df: pd.DataFrame, period: int = 10
                         ) -> Tuple[pd.Series, pd.Series]:
    """
    RVI = (Close - Open) / (High - Low), convolved with symmetric FIR.
    Ehlers Ch. 6: "Relative Vigor Index measures the conviction of
    a recent price move."

    Returns (rvi, signal).
    """
    close = df["Close"].values.astype(float)
    open_ = df["Open"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    n = len(close)

    # Numerator & denominator with 4-tap symmetric FIR
    num = np.zeros(n)
    den = np.zeros(n)

    for i in range(3, n):
        num[i] = ((close[i] - open_[i])
                  + 2.0 * (close[i - 1] - open_[i - 1])
                  + 2.0 * (close[i - 2] - open_[i - 2])
                  + (close[i - 3] - open_[i - 3])) / 6.0
        range_hl = high[i] - low[i]
        den[i] = ((range_hl)
                  + 2.0 * (high[i - 1] - low[i - 1])
                  + 2.0 * (high[i - 2] - low[i - 2])
                  + (high[i - 3] - low[i - 3])) / 6.0

    rvi = np.zeros(n)
    for i in range(period + 3, n):
        sum_num = np.sum(num[i - period + 1: i + 1])
        sum_den = np.sum(den[i - period + 1: i + 1])
        rvi[i] = sum_num / sum_den if sum_den != 0.0 else 0.0

    # Signal line: 4-tap symmetric FIR of RVI
    signal = np.zeros(n)
    for i in range(3, n):
        signal[i] = (rvi[i] + 2.0 * rvi[i - 1]
                     + 2.0 * rvi[i - 2] + rvi[i - 3]) / 6.0

    return (pd.Series(rvi, index=df.index, name="rvi"),
            pd.Series(signal, index=df.index, name="rvi_signal"))


# ═══════════════════════════════════════════════════════════════
# Signal-to-Noise Ratio (Ehlers, Rocket Science Ch. 8)
# ═══════════════════════════════════════════════════════════════

def signal_to_noise_ratio(series: pd.Series, period: int = 10) -> pd.Series:
    """
    SNR in dB. Higher = cleaner trend, lower = noisy/choppy.
    Ehlers Ch. 8: "The Signal-to-Noise Ratio enables us to
    determine when a signal is clear enough to act upon."

    SNR > 6 dB: strong trend, take signal
    SNR < 3 dB: noisy, reduce position / skip
    """
    n = len(series)
    vals = series.values.astype(float)
    snr = np.zeros(n)

    smooth = super_smoother(series, period).values

    for i in range(period, n):
        signal_power = (smooth[i] - smooth[i - period]) ** 2
        noise_power = 0.0
        for j in range(i - period + 1, i + 1):
            noise_power += (vals[j] - smooth[j]) ** 2
        noise_power /= period

        if noise_power > 0:
            snr[i] = 10.0 * math.log10(max(1e-10, signal_power / noise_power))
        else:
            snr[i] = 20.0  # Perfect signal

    return pd.Series(snr, index=series.index, name="snr")


# ═══════════════════════════════════════════════════════════════
# Dominant Cycle Period (Homodyne Discriminator)
# ═══════════════════════════════════════════════════════════════

def dominant_cycle_period(series: pd.Series) -> pd.Series:
    """
    Measure the dominant cycle period using the Homodyne Discriminator.
    Ehlers, Rocket Science Ch. 7.

    Returns cycle period in bars. Typical range: 6-50 bars.
    """
    vals = series.values.astype(float)
    n = len(vals)
    dc = _compute_dominant_cycle_array(vals, n)
    return pd.Series(dc, index=series.index, name="dominant_cycle")


def _compute_dominant_cycle_array(vals: np.ndarray, n: int) -> np.ndarray:
    """Internal: compute dominant cycle period array via Homodyne Discriminator."""
    smooth = np.zeros(n)
    detrend = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re_arr = np.zeros(n)
    im_arr = np.zeros(n)
    period = np.full(n, 15.0)
    smooth_period = np.full(n, 15.0)

    for i in range(6, n):
        smooth[i] = (4.0 * vals[i] + 3.0 * vals[i - 1]
                     + 2.0 * vals[i - 2] + vals[i - 3]) / 10.0

        adj = 0.075 * period[i - 1] + 0.54

        detrend[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
                      - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6]) * adj

        q1[i] = (0.0962 * detrend[i] + 0.5769 * detrend[i - 2]
                 - 0.5769 * detrend[i - 4] - 0.0962 * detrend[i - 6]) * adj
        i1[i] = detrend[i - 3]

        # Phasor addition
        i2[i] = i1[i] - 0.0  # No jQ in simplified version
        q2[i] = q1[i]

        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        # Homodyne Discriminator
        re_arr[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im_arr[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re_arr[i] = 0.2 * re_arr[i] + 0.8 * re_arr[i - 1]
        im_arr[i] = 0.2 * im_arr[i] + 0.8 * im_arr[i - 1]

        if im_arr[i] != 0.0 and re_arr[i] != 0.0:
            period[i] = 360.0 / math.degrees(math.atan2(im_arr[i], re_arr[i]))
        else:
            period[i] = period[i - 1]

        period[i] = max(6.0, min(50.0, period[i]))
        period[i] = max(0.67 * period[i - 1], min(1.5 * period[i - 1], period[i]))
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]

    return smooth_period


# ═══════════════════════════════════════════════════════════════
# Adaptive RSI (Ehlers, Rocket Science Ch. 22)
# ═══════════════════════════════════════════════════════════════

def adaptive_rsi(series: pd.Series) -> pd.Series:
    """
    RSI that adapts its period to the dominant cycle.
    Ehlers Ch. 22: "Making Standard Indicators Adaptive."

    Instead of fixed RSI(14), uses RSI(dominant_cycle/2).
    """
    n = len(series)
    vals = series.values.astype(float)
    dc = _compute_dominant_cycle_array(vals, n)
    rsi_out = np.full(n, 50.0)

    for i in range(20, n):
        period = max(3, int(dc[i] / 2.0))
        lookback = min(period, i)

        gains = 0.0
        losses = 0.0
        for j in range(1, lookback + 1):
            if i - j >= 0:
                change = vals[i - j + 1] - vals[i - j]
                if change > 0:
                    gains += change
                else:
                    losses -= change

        if gains + losses > 0:
            rsi_out[i] = 100.0 * gains / (gains + losses)
        else:
            rsi_out[i] = 50.0

    return pd.Series(rsi_out, index=series.index, name="adaptive_rsi")


# ═══════════════════════════════════════════════════════════════
# Composite Forecast (Combines all Ehlers indicators)
# ═══════════════════════════════════════════════════════════════

def compute_ehlers_analysis(df: pd.DataFrame) -> Optional[EhlersAnalysis]:
    """
    Full Ehlers DSP analysis for a single stock's OHLCV DataFrame.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume]
            Must have at least 50 bars of data.

    Returns:
        EhlersAnalysis dataclass with all indicators and composite forecast.
    """
    if df is None or len(df) < 50:
        return None

    try:
        # Handle yfinance multi-index columns
        _df = df.copy()
        if hasattr(_df.columns, 'nlevels') and _df.columns.nlevels > 1:
            _df.columns = _df.columns.get_level_values(0)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in _df.columns and hasattr(_df[col], 'squeeze'):
                _df[col] = _df[col].squeeze()

        hl2 = (_df["High"] + _df["Low"]) / 2.0
        close = _df["Close"]

        # Compute all indicators
        ss = super_smoother(hl2, period=10)
        fisher, trigger = fisher_transform(hl2, period=10)
        itrend = instantaneous_trendline(hl2)
        cc = cyber_cycle(hl2)
        mama_v, fama_v = mama_fama(hl2)
        sine, leadsine = sinewave_indicator(hl2)
        snr = signal_to_noise_ratio(hl2, period=10)
        arsi = adaptive_rsi(close)
        dc = dominant_cycle_period(hl2)

        # RVI needs OHLC
        rvi_v, rvi_sig = relative_vigor_index(_df, period=10)

        # Last values
        idx = -1
        ss_val = float(ss.iloc[idx])
        fisher_val = float(fisher.iloc[idx])
        trigger_val = float(trigger.iloc[idx])
        itrend_val = float(itrend.iloc[idx])
        cc_val = float(cc.iloc[idx])
        mama_val = float(mama_v.iloc[idx])
        fama_val = float(fama_v.iloc[idx])
        sine_val = float(sine.iloc[idx])
        leadsine_val = float(leadsine.iloc[idx])
        rvi_val = float(rvi_v.iloc[idx])
        rvi_sig_val = float(rvi_sig.iloc[idx])
        snr_val = float(snr.iloc[idx])
        arsi_val = float(arsi.iloc[idx])
        dc_val = float(dc.iloc[idx])

        # ── Construct composite forecast ────────────────────
        forecast = _compute_composite_forecast(
            fisher_val, trigger_val,
            mama_val, fama_val,
            itrend_val, float(close.iloc[idx]),
            cc_val, float(cc.iloc[idx - 1]) if len(cc) > 1 else 0.0,
            sine_val, leadsine_val,
            rvi_val, rvi_sig_val,
            snr_val, arsi_val
        )

        symbol = df.attrs.get("symbol", "UNKNOWN") if hasattr(df, "attrs") else "UNKNOWN"

        return EhlersAnalysis(
            symbol=symbol,
            super_smoother=ss_val,
            fisher_transform=fisher_val,
            fisher_trigger=trigger_val,
            instantaneous_trendline=itrend_val,
            cyber_cycle=cc_val,
            mama=mama_val,
            fama=fama_val,
            sinewave=sine_val,
            leadsine=leadsine_val,
            rvi=rvi_val,
            rvi_signal=rvi_sig_val,
            snr=snr_val,
            adaptive_rsi=arsi_val,
            dominant_cycle=dc_val,
            composite_forecast=forecast,
        )

    except Exception as e:
        logger.error("Ehlers analysis failed: %s", e)
        return None


def _compute_composite_forecast(
    fisher: float, trigger: float,
    mama: float, fama: float,
    itrend: float, price: float,
    cc: float, cc_prev: float,
    sine: float, leadsine: float,
    rvi: float, rvi_signal: float,
    snr: float, arsi: float,
) -> float:
    """
    Combine Ehlers indicators into a single Carver-compatible forecast.

    Sub-signals and weights:
      1. Fisher crossover (25%): sharp turning points
      2. MAMA vs FAMA (25%): adaptive trend direction
      3. Price vs Instantaneous Trendline (15%): zero-lag trend
      4. Sinewave crossover (15%): leading cycle turning points
      5. RVI crossover (10%): vigor confirmation
      6. Adaptive RSI (10%): overbought/oversold

    SNR acts as confidence multiplier:
      SNR > 6 dB: 1.0 (strong signal)
      SNR 3-6 dB: 0.6 (moderate)
      SNR < 3 dB: 0.3 (weak — dampen forecast)
    """
    # 1. Fisher crossover → forecast
    fisher_score = 0.0
    diff = fisher - trigger
    fisher_score = max(-10.0, min(10.0, diff * 5.0))

    # 2. MAMA vs FAMA
    mama_score = 0.0
    if mama > fama:
        mama_score = min(10.0, (mama - fama) / max(1e-6, abs(fama)) * 100.0)
    else:
        mama_score = max(-10.0, (mama - fama) / max(1e-6, abs(fama)) * 100.0)

    # 3. Price vs Instantaneous Trendline
    trend_score = 0.0
    if itrend != 0:
        pct_above = (price - itrend) / abs(itrend) * 100.0
        trend_score = max(-10.0, min(10.0, pct_above * 5.0))

    # 4. Sinewave crossover
    sine_score = 0.0
    sine_diff = sine - leadsine
    sine_score = max(-10.0, min(10.0, sine_diff * 10.0))

    # 5. RVI crossover
    rvi_score = 0.0
    rvi_diff = rvi - rvi_signal
    rvi_score = max(-10.0, min(10.0, rvi_diff * 20.0))

    # 6. Adaptive RSI
    rsi_score = 0.0
    if arsi > 70:
        rsi_score = -min(10.0, (arsi - 70) / 3.0)  # Overbought → reduce
    elif arsi < 30:
        rsi_score = min(10.0, (30 - arsi) / 3.0)    # Oversold → buy signal
    # 30-70: neutral contribution

    # Weighted combination
    raw = (0.25 * fisher_score
           + 0.25 * mama_score
           + 0.15 * trend_score
           + 0.15 * sine_score
           + 0.10 * rvi_score
           + 0.10 * rsi_score)

    # SNR confidence multiplier
    if snr >= 6.0:
        confidence = 1.0
    elif snr >= 3.0:
        confidence = 0.6
    else:
        confidence = 0.3

    forecast = raw * confidence

    # Carver scale: cap at ±20
    return max(-20.0, min(20.0, forecast))


# ═══════════════════════════════════════════════════════════════
# Batch Processing for Pipeline Integration
# ═══════════════════════════════════════════════════════════════

def compute_ehlers_forecast_batch(
    ohlcv_dict: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """
    Compute Ehlers composite forecast for multiple symbols.

    Args:
        ohlcv_dict: {symbol: DataFrame with OHLCV}

    Returns:
        {symbol: forecast_value} where forecast is -20 to +20
    """
    results = {}
    for symbol, df in ohlcv_dict.items():
        if df is not None and hasattr(df, 'attrs'):
            df.attrs["symbol"] = symbol
        elif df is not None:
            df.attrs = {"symbol": symbol}

        analysis = compute_ehlers_analysis(df)
        if analysis is not None:
            results[symbol] = analysis.composite_forecast
            logger.info(
                "Ehlers %s: fisher=%.2f mama_fama=%s snr=%.1fdB dc=%.0f → forecast=%.1f",
                symbol, analysis.fisher_transform,
                "BULL" if analysis.mama > analysis.fama else "BEAR",
                analysis.snr, analysis.dominant_cycle,
                analysis.composite_forecast,
            )
        else:
            results[symbol] = 0.0

    return results


def compute_ehlers_analysis_batch(
    ohlcv_dict: Dict[str, pd.DataFrame],
) -> Dict[str, Optional[EhlersAnalysis]]:
    """
    Full Ehlers analysis for multiple symbols (for API/UI display).

    Returns:
        {symbol: EhlersAnalysis or None}
    """
    results = {}
    for symbol, df in ohlcv_dict.items():
        if df is not None:
            df.attrs = {"symbol": symbol}
        results[symbol] = compute_ehlers_analysis(df)
    return results
