"""
Advanced Technical Indicators — computed locally on OHLCV data via the ``ta`` library.

Extends the existing 6 indicators (RSI, MACD, BB, ADX, OBV, Fibonacci) with:
- Supertrend, Ichimoku Cloud, Parabolic SAR       (trend)
- Stochastic RSI, Williams %R, CCI, MFI           (momentum/oscillator)
- ATR, Keltner Channels                            (volatility)
- CMF (Chaikin Money Flow), VWAP                   (volume)

All functions accept a standard OHLCV pandas DataFrame and return
either a single float value or a typed dict.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AdvancedIndicators:
    """Container for all advanced TA indicators computed locally."""

    # Trend
    supertrend_direction: Optional[float] = None   # +1 bullish, -1 bearish
    supertrend_value: Optional[float] = None
    ichimoku_conversion: Optional[float] = None
    ichimoku_base: Optional[float] = None
    ichimoku_span_a: Optional[float] = None
    ichimoku_span_b: Optional[float] = None
    parabolic_sar: Optional[float] = None

    # Momentum / Oscillator
    stoch_rsi_k: Optional[float] = None   # 0–100
    stoch_rsi_d: Optional[float] = None   # 0–100
    williams_r: Optional[float] = None    # -100 to 0
    cci: Optional[float] = None           # typically -200 to +200
    mfi: Optional[float] = None           # 0–100

    # Volatility
    atr: Optional[float] = None
    keltner_upper: Optional[float] = None
    keltner_lower: Optional[float] = None

    # Volume
    cmf: Optional[float] = None           # -1 to +1
    vwap: Optional[float] = None

    # Enhanced existing
    rsi_enhanced: Optional[float] = None  # ta-lib RSI for cross-validation
    macd_enhanced: Optional[float] = None
    macd_signal_enhanced: Optional[float] = None
    macd_histogram_enhanced: Optional[float] = None
    bb_upper_enhanced: Optional[float] = None
    bb_lower_enhanced: Optional[float] = None
    bb_pband: Optional[float] = None      # %B — position within bands (0–1)
    bb_wband: Optional[float] = None      # bandwidth — volatility measure
    adx_enhanced: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    obv_enhanced: Optional[float] = None

    # Meta
    indicator_count: int = 0              # how many were successfully computed


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------

def compute_advanced_indicators(
    df: pd.DataFrame,
    *,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
    stoch_rsi_period: int = 14,
    stoch_rsi_smooth_k: int = 3,
    stoch_rsi_smooth_d: int = 3,
    williams_r_period: int = 14,
    cci_period: int = 20,
    mfi_period: int = 14,
    atr_period: int = 14,
    keltner_period: int = 20,
    keltner_atr_mult: float = 1.5,
    cmf_period: int = 20,
    ichimoku_conv: int = 9,
    ichimoku_base: int = 26,
    ichimoku_span_b: int = 52,
) -> AdvancedIndicators:
    """Compute all advanced indicators on a standard OHLCV DataFrame.

    Expects columns: Open, High, Low, Close, Volume (case-sensitive).
    Requires at least 60 rows of data; gracefully degrades with fewer.

    Returns an ``AdvancedIndicators`` dataclass.
    """
    result = AdvancedIndicators()

    if df is None or df.empty or len(df) < 20:
        logger.warning("Insufficient data for advanced indicators (%s rows)",
                       0 if df is None else len(df))
        return result

    # Flatten MultiIndex columns (yfinance v2 returns tuples)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Normalise column names (yfinance sometimes uses lowercase)
    cols = {c.lower(): c for c in df.columns}
    close = df[cols.get("close", "Close")]
    high = df[cols.get("high", "High")]
    low = df[cols.get("low", "Low")]
    opn = df[cols.get("open", "Open")]
    volume = df[cols.get("volume", "Volume")]

    count = 0

    # ------------------------------------------------------------------
    # 1.  Enhanced existing indicators (cross-validation with ta library)
    # ------------------------------------------------------------------
    try:
        from ta.momentum import RSIIndicator
        rsi_ind = RSIIndicator(close=close, window=14)
        result.rsi_enhanced = _last_float(rsi_ind.rsi())
        count += 1
    except Exception as e:
        logger.debug("RSI enhanced failed: %s", e)

    try:
        from ta.trend import MACD
        macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        result.macd_enhanced = _last_float(macd_ind.macd())
        result.macd_signal_enhanced = _last_float(macd_ind.macd_signal())
        result.macd_histogram_enhanced = _last_float(macd_ind.macd_diff())
        count += 1
    except Exception as e:
        logger.debug("MACD enhanced failed: %s", e)

    try:
        from ta.volatility import BollingerBands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        result.bb_upper_enhanced = _last_float(bb.bollinger_hband())
        result.bb_lower_enhanced = _last_float(bb.bollinger_lband())
        result.bb_pband = _last_float(bb.bollinger_pband())
        result.bb_wband = _last_float(bb.bollinger_wband())
        count += 1
    except Exception as e:
        logger.debug("BB enhanced failed: %s", e)

    try:
        from ta.trend import ADXIndicator
        adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
        result.adx_enhanced = _last_float(adx_ind.adx())
        result.plus_di = _last_float(adx_ind.adx_pos())
        result.minus_di = _last_float(adx_ind.adx_neg())
        count += 1
    except Exception as e:
        logger.debug("ADX enhanced failed: %s", e)

    try:
        from ta.volume import OnBalanceVolumeIndicator
        obv_ind = OnBalanceVolumeIndicator(close=close, volume=volume)
        result.obv_enhanced = _last_float(obv_ind.on_balance_volume())
        count += 1
    except Exception as e:
        logger.debug("OBV enhanced failed: %s", e)

    # ------------------------------------------------------------------
    # 2.  New trend indicators
    # ------------------------------------------------------------------

    # Supertrend
    try:
        st_dir, st_val = _compute_supertrend(high, low, close,
                                              supertrend_period, supertrend_multiplier)
        result.supertrend_direction = st_dir
        result.supertrend_value = st_val
        count += 1
    except Exception as e:
        logger.debug("Supertrend failed: %s", e)

    # Ichimoku Cloud
    try:
        from ta.trend import IchimokuIndicator
        ich = IchimokuIndicator(high=high, low=low,
                                window1=ichimoku_conv,
                                window2=ichimoku_base,
                                window3=ichimoku_span_b)
        result.ichimoku_conversion = _last_float(ich.ichimoku_conversion_line())
        result.ichimoku_base = _last_float(ich.ichimoku_base_line())
        result.ichimoku_span_a = _last_float(ich.ichimoku_a())
        result.ichimoku_span_b = _last_float(ich.ichimoku_b())
        count += 1
    except Exception as e:
        logger.debug("Ichimoku failed: %s", e)

    # Parabolic SAR
    try:
        from ta.trend import PSARIndicator
        psar = PSARIndicator(high=high, low=low, close=close)
        result.parabolic_sar = _last_float(psar.psar())
        count += 1
    except Exception as e:
        logger.debug("PSAR failed: %s", e)

    # ------------------------------------------------------------------
    # 3.  New momentum / oscillator indicators
    # ------------------------------------------------------------------

    # Stochastic RSI
    try:
        from ta.momentum import StochRSIIndicator
        srsi = StochRSIIndicator(close=close, window=stoch_rsi_period,
                                  smooth1=stoch_rsi_smooth_k,
                                  smooth2=stoch_rsi_smooth_d)
        result.stoch_rsi_k = _last_float(srsi.stochrsi_k()) 
        result.stoch_rsi_d = _last_float(srsi.stochrsi_d())
        # Convert 0-1 range to 0-100
        if result.stoch_rsi_k is not None:
            result.stoch_rsi_k *= 100
        if result.stoch_rsi_d is not None:
            result.stoch_rsi_d *= 100
        count += 1
    except Exception as e:
        logger.debug("StochRSI failed: %s", e)

    # Williams %R
    try:
        from ta.momentum import WilliamsRIndicator
        wr = WilliamsRIndicator(high=high, low=low, close=close,
                                 lbp=williams_r_period)
        result.williams_r = _last_float(wr.williams_r())
        count += 1
    except Exception as e:
        logger.debug("Williams %%R failed: %s", e)

    # CCI (Commodity Channel Index)
    try:
        from ta.trend import CCIIndicator
        cci_ind = CCIIndicator(high=high, low=low, close=close,
                                window=cci_period)
        result.cci = _last_float(cci_ind.cci())
        count += 1
    except Exception as e:
        logger.debug("CCI failed: %s", e)

    # MFI (Money Flow Index)
    try:
        from ta.volume import MFIIndicator
        mfi_ind = MFIIndicator(high=high, low=low, close=close,
                                volume=volume, window=mfi_period)
        result.mfi = _last_float(mfi_ind.money_flow_index())
        count += 1
    except Exception as e:
        logger.debug("MFI failed: %s", e)

    # ------------------------------------------------------------------
    # 4.  Volatility indicators
    # ------------------------------------------------------------------

    # ATR (Average True Range)
    try:
        from ta.volatility import AverageTrueRange
        atr_ind = AverageTrueRange(high=high, low=low, close=close,
                                    window=atr_period)
        result.atr = _last_float(atr_ind.average_true_range())
        count += 1
    except Exception as e:
        logger.debug("ATR failed: %s", e)

    # Keltner Channels
    try:
        from ta.volatility import KeltnerChannel
        kc = KeltnerChannel(high=high, low=low, close=close,
                             window=keltner_period,
                             window_atr=atr_period,
                             multiplier=keltner_atr_mult)
        result.keltner_upper = _last_float(kc.keltner_channel_hband())
        result.keltner_lower = _last_float(kc.keltner_channel_lband())
        count += 1
    except Exception as e:
        logger.debug("Keltner failed: %s", e)

    # ------------------------------------------------------------------
    # 5.  Volume indicators
    # ------------------------------------------------------------------

    # CMF (Chaikin Money Flow)
    try:
        from ta.volume import ChaikinMoneyFlowIndicator
        cmf_ind = ChaikinMoneyFlowIndicator(high=high, low=low, close=close,
                                             volume=volume, window=cmf_period)
        result.cmf = _last_float(cmf_ind.chaikin_money_flow())
        count += 1
    except Exception as e:
        logger.debug("CMF failed: %s", e)

    # VWAP (Volume Weighted Average Price)
    try:
        from ta.volume import VolumeWeightedAveragePrice
        vwap_ind = VolumeWeightedAveragePrice(high=high, low=low,
                                               close=close, volume=volume)
        result.vwap = _last_float(vwap_ind.volume_weighted_average_price())
        count += 1
    except Exception as e:
        logger.debug("VWAP failed: %s", e)

    result.indicator_count = count
    logger.info("Computed %d/%d advanced indicators", count, 17)
    return result


# ---------------------------------------------------------------------------
# Supertrend (not in ``ta`` — manual implementation)
# ---------------------------------------------------------------------------

def _compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple:
    """Compute Supertrend indicator.

    Returns (direction, value):
        direction = +1.0 → bullish (price above supertrend)
        direction = -1.0 → bearish (price below supertrend)
        value     = supertrend line value
    """
    hl2 = (high + low) / 2

    # ATR via true range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(0.0, index=close.index)
    direction = pd.Series(1.0, index=close.index)

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1.0 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1.0 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1.0 else upper_band.iloc[i]

    return float(direction.iloc[-1]), float(supertrend.iloc[-1])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _last_float(series: pd.Series) -> Optional[float]:
    """Extract last non-NaN float from a pandas Series."""
    if series is None or series.empty:
        return None
    last = series.dropna()
    if last.empty:
        return None
    val = float(last.iloc[-1])
    return val if np.isfinite(val) else None
