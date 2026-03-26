"""
Technical Analysis Aggregator.

Fuses three signal sources into a single TA score (-1 to +1):

1. **Legacy 6 indicators** (RSI, MACD, BB, ADX, OBV, Fib) — from MetricsCalculator
2. **Advanced local indicators** (Supertrend, Ichimoku, StochRSI, CCI, MFI, …) — ``ta`` lib
3. **TradingView consensus** (26 oscillators + MAs across 1h/4h/1D/1W)

Weighting (configurable via Config):
    Local advanced indicators  50%   (17 signals)
    TradingView consensus      30%   (multi-timeframe agreement)
    Cross-validation bonus     20%   (agreement between local & TV amplifies conviction)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import Config
from services.technical_analysis.indicators import (
    AdvancedIndicators,
    compute_advanced_indicators,
)
from services.technical_analysis.tradingview import (
    TVConsensus,
    fetch_tradingview_consensus,
)

logger = logging.getLogger(__name__)


@dataclass
class TAResult:
    """Unified technical analysis result."""
    # Component scores
    local_score: float = 0.0       # From advanced local indicators
    tv_score: float = 0.0          # From TradingView consensus
    cross_validation_bonus: float = 0.0  # Agreement amplifier

    # Fused result
    fused_score: float = 0.0       # Final -1 to +1

    # Breakdown
    trend_score: float = 0.0       # Supertrend + Ichimoku + PSAR + MAs
    momentum_score: float = 0.0    # StochRSI + Williams%R + CCI + RSI
    volatility_score: float = 0.0  # BB%B + Keltner + ATR context
    volume_score: float = 0.0      # CMF + MFI + OBV + VWAP

    # Raw data
    advanced_indicators: Optional[AdvancedIndicators] = None
    tv_consensus: Optional[TVConsensus] = None

    # Meta
    indicator_count: int = 0
    tv_available: bool = False
    confidence: float = 0.0        # 0–1 based on data completeness

    @property
    def summary(self) -> str:
        parts = [
            f"fused={self.fused_score:+.3f}",
            f"local={self.local_score:+.3f}",
            f"tv={self.tv_score:+.3f}({'+' if self.tv_available else '-'})",
            f"xval={self.cross_validation_bonus:+.3f}",
            f"conf={self.confidence:.0%}",
        ]
        return f"TA[{', '.join(parts)}]"


class TechnicalAnalysisAggregator:
    """Fuses local indicators + TradingView into a unified TA score."""

    def analyze(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        *,
        skip_tradingview: bool = False,
    ) -> TAResult:
        """Run full technical analysis for a ticker.

        Args:
            ticker: Stock symbol (e.g. 'RELIANCE.NS', 'AAPL')
            ohlcv: OHLCV DataFrame (at least 60 rows recommended)
            skip_tradingview: If True, skip the TV API call

        Returns:
            ``TAResult`` with fused score and breakdowns.
        """
        result = TAResult()

        # 1. Compute advanced local indicators
        adv = compute_advanced_indicators(ohlcv)
        result.advanced_indicators = adv
        result.indicator_count = adv.indicator_count

        # Normalise OHLCV columns for downstream scoring
        _ohlcv = ohlcv
        if _ohlcv is not None and isinstance(_ohlcv.columns, pd.MultiIndex):
            _ohlcv = _ohlcv.copy()
            _ohlcv.columns = [c[0] if isinstance(c, tuple) else c for c in _ohlcv.columns]

        # 2. Score local indicators by category
        result.trend_score = self._score_trend(adv, _ohlcv)
        result.momentum_score = self._score_momentum(adv)
        result.volatility_score = self._score_volatility(adv, _ohlcv)
        result.volume_score = self._score_volume(adv, _ohlcv)

        # Composite local score (equal weight across categories)
        cat_scores = [result.trend_score, result.momentum_score,
                      result.volatility_score, result.volume_score]
        result.local_score = _clamp(sum(cat_scores) / len(cat_scores))

        # 3. Fetch TradingView consensus
        if not skip_tradingview:
            try:
                tv = fetch_tradingview_consensus(ticker)
                result.tv_consensus = tv
                result.tv_available = tv.available
                result.tv_score = tv.overall_score if tv.available else 0.0
            except Exception as e:
                logger.warning("TV consensus failed for %s: %s", ticker, e)

        # 4. Cross-validation bonus
        if result.tv_available:
            agreement = result.local_score * result.tv_score
            if agreement > 0:
                # Both agree on direction → amplify conviction
                result.cross_validation_bonus = min(0.15, abs(agreement) * 0.3)
                # Preserve sign of local score
                if result.local_score < 0:
                    result.cross_validation_bonus *= -1
            else:
                # Disagreement → dampen
                result.cross_validation_bonus = 0.0

        # 5. Fuse scores
        w_local = Config.TA_LOCAL_WEIGHT
        w_tv = Config.TA_TV_WEIGHT if result.tv_available else 0.0
        w_xval = Config.TA_XVAL_WEIGHT if result.tv_available else 0.0

        total_w = w_local + w_tv + w_xval
        if total_w == 0:
            total_w = 1.0

        result.fused_score = _clamp(
            (result.local_score * w_local
             + result.tv_score * w_tv
             + result.cross_validation_bonus * w_xval)
            / total_w
        )

        # 6. Confidence (data completeness)
        max_indicators = 17
        local_pct = min(1.0, adv.indicator_count / max_indicators)
        tv_pct = 1.0 if result.tv_available else 0.0
        result.confidence = local_pct * 0.6 + tv_pct * 0.4

        logger.info("%s TA: %s", ticker, result.summary)
        return result

    # ------------------------------------------------------------------
    # Category scorers — each returns -1 to +1
    # ------------------------------------------------------------------

    def _score_trend(self, adv: AdvancedIndicators, ohlcv: pd.DataFrame) -> float:
        """Score trend indicators: Supertrend, Ichimoku, PSAR, ADX+DI."""
        signals = []

        # Supertrend direction
        if adv.supertrend_direction is not None:
            signals.append(adv.supertrend_direction * 0.5)  # +0.5 or -0.5

        # Ichimoku: price vs cloud
        if (adv.ichimoku_span_a is not None and adv.ichimoku_span_b is not None
                and ohlcv is not None and len(ohlcv) > 0):
            price = _last_close(ohlcv)
            cloud_top = max(adv.ichimoku_span_a, adv.ichimoku_span_b)
            cloud_bot = min(adv.ichimoku_span_a, adv.ichimoku_span_b)
            if price > cloud_top:
                signals.append(0.5)   # Above cloud — bullish
            elif price < cloud_bot:
                signals.append(-0.5)  # Below cloud — bearish
            else:
                signals.append(0.0)   # Inside cloud — neutral

        # Ichimoku: conversion vs base line
        if adv.ichimoku_conversion is not None and adv.ichimoku_base is not None:
            if adv.ichimoku_conversion > adv.ichimoku_base:
                signals.append(0.3)
            else:
                signals.append(-0.3)

        # Parabolic SAR
        if adv.parabolic_sar is not None and ohlcv is not None and len(ohlcv) > 0:
            price = _last_close(ohlcv)
            if price > adv.parabolic_sar:
                signals.append(0.3)   # SAR below price — uptrend
            else:
                signals.append(-0.3)

        # ADX with +DI/-DI directional
        if adv.adx_enhanced is not None and adv.plus_di is not None and adv.minus_di is not None:
            if adv.adx_enhanced >= 20:  # Trending
                if adv.plus_di > adv.minus_di:
                    signals.append(0.4)
                else:
                    signals.append(-0.4)
            # ADX < 20 → range-bound, no directional signal

        return _clamp(_safe_avg(signals))

    def _score_momentum(self, adv: AdvancedIndicators) -> float:
        """Score momentum oscillators: StochRSI, Williams%R, CCI, RSI, MFI."""
        signals = []

        # Stochastic RSI (0-100)
        if adv.stoch_rsi_k is not None:
            if adv.stoch_rsi_k < 20:
                signals.append(0.5)   # Oversold
            elif adv.stoch_rsi_k < 35:
                signals.append(0.2)
            elif adv.stoch_rsi_k > 80:
                signals.append(-0.5)  # Overbought
            elif adv.stoch_rsi_k > 65:
                signals.append(-0.2)
            else:
                signals.append(0.0)

        # Williams %R (-100 to 0)
        if adv.williams_r is not None:
            if adv.williams_r < -80:
                signals.append(0.4)   # Oversold
            elif adv.williams_r < -60:
                signals.append(0.1)
            elif adv.williams_r > -20:
                signals.append(-0.4)  # Overbought
            elif adv.williams_r > -40:
                signals.append(-0.1)
            else:
                signals.append(0.0)

        # CCI
        if adv.cci is not None:
            if adv.cci < -200:
                signals.append(0.5)   # Deeply oversold
            elif adv.cci < -100:
                signals.append(0.3)
            elif adv.cci > 200:
                signals.append(-0.5)  # Deeply overbought
            elif adv.cci > 100:
                signals.append(-0.3)
            else:
                signals.append(0.0)

        # RSI cross-validation (ta lib vs existing)
        if adv.rsi_enhanced is not None:
            if adv.rsi_enhanced < 30:
                signals.append(0.4)
            elif adv.rsi_enhanced < 40:
                signals.append(0.15)
            elif adv.rsi_enhanced > 70:
                signals.append(-0.4)
            elif adv.rsi_enhanced > 60:
                signals.append(-0.15)
            else:
                signals.append(0.0)

        # MFI (0-100) — volume-weighted RSI
        if adv.mfi is not None:
            if adv.mfi < 20:
                signals.append(0.4)   # Oversold with volume confirmation
            elif adv.mfi < 35:
                signals.append(0.15)
            elif adv.mfi > 80:
                signals.append(-0.4)
            elif adv.mfi > 65:
                signals.append(-0.15)
            else:
                signals.append(0.0)

        return _clamp(_safe_avg(signals))

    def _score_volatility(self, adv: AdvancedIndicators, ohlcv: pd.DataFrame) -> float:
        """Score volatility indicators: BB %B, Keltner position, ATR context."""
        signals = []

        # Bollinger %B (0 = lower band, 1 = upper band)
        if adv.bb_pband is not None:
            if adv.bb_pband < 0.1:
                signals.append(0.4)   # Below lower band — oversold
            elif adv.bb_pband < 0.3:
                signals.append(0.2)
            elif adv.bb_pband > 0.9:
                signals.append(-0.4)  # Above upper band — overbought
            elif adv.bb_pband > 0.7:
                signals.append(-0.2)
            else:
                signals.append(0.0)

        # Keltner Channel position
        if (adv.keltner_upper is not None and adv.keltner_lower is not None
                and ohlcv is not None and len(ohlcv) > 0):
            price = _last_close(ohlcv)
            kc_range = adv.keltner_upper - adv.keltner_lower
            if kc_range > 0:
                kc_pos = (price - adv.keltner_lower) / kc_range
                if kc_pos < 0.15:
                    signals.append(0.3)
                elif kc_pos > 0.85:
                    signals.append(-0.3)
                else:
                    signals.append(0.0)

        # Bollinger Bandwidth — squeeze detection
        if adv.bb_wband is not None:
            # Low bandwidth = volatility squeeze → breakout imminent
            # (not directional, but reduces dampening)
            if adv.bb_wband < 0.05:
                signals.append(0.1)  # Slight bullish bias (breakouts tend up)

        return _clamp(_safe_avg(signals))

    def _score_volume(self, adv: AdvancedIndicators, ohlcv: pd.DataFrame) -> float:
        """Score volume indicators: CMF, VWAP position, OBV."""
        signals = []

        # CMF (-1 to +1)
        if adv.cmf is not None:
            if adv.cmf > 0.1:
                signals.append(0.4)   # Strong buying pressure
            elif adv.cmf > 0:
                signals.append(0.15)
            elif adv.cmf < -0.1:
                signals.append(-0.4)  # Strong selling pressure
            elif adv.cmf < 0:
                signals.append(-0.15)
            else:
                signals.append(0.0)

        # VWAP — price above VWAP = bullish intraday bias
        if adv.vwap is not None and ohlcv is not None and len(ohlcv) > 0:
            price = _last_close(ohlcv)
            if price > 0 and adv.vwap > 0:
                vwap_ratio = price / adv.vwap
                if vwap_ratio > 1.02:
                    signals.append(0.3)
                elif vwap_ratio > 1.0:
                    signals.append(0.1)
                elif vwap_ratio < 0.98:
                    signals.append(-0.3)
                elif vwap_ratio < 1.0:
                    signals.append(-0.1)

        # MFI as volume confirmation (already scored in momentum but useful here too)
        if adv.mfi is not None:
            if adv.mfi > 60:
                signals.append(0.2)
            elif adv.mfi < 40:
                signals.append(-0.2)

        return _clamp(_safe_avg(signals))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_avg(values: list) -> float:
    """Average of non-empty list, or 0."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _last_close(ohlcv: pd.DataFrame) -> float:
    """Extract last close price from OHLCV DataFrame."""
    if ohlcv is None or ohlcv.empty:
        return 0.0
    row = ohlcv.iloc[-1]
    for col in ("Close", "close"):
        if col in ohlcv.columns:
            return float(row[col])
    return 0.0
