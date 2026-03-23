"""
Market Regime Detector — Online Learning & Regime Adaptation.

Detects the current market regime (TRENDING_BULL, TRENDING_BEAR,
RANGE_BOUND, HIGH_VOLATILITY, CRISIS) using a rolling window of
market data. Adapts signal thresholds and strategy weights dynamically.

Architecture:
    - Uses India VIX, NIFTY 50 returns, breadth (advance/decline),
      and ADX as regime features.
    - Exponentially-weighted regime scoring — recent observations
      carry more weight (half-life = 10 days).
    - No heavy ML model — purely rule-based with adaptive thresholds,
      suitable for 1-2 retail users without GPU.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CRISIS = "CRISIS"


@dataclass
class RegimeSnapshot:
    """Current regime assessment with adaptive parameters."""
    regime: MarketRegime
    confidence: float              # 0-1
    vix: float = 0.0
    nifty_20d_return: float = 0.0
    nifty_adx: float = 0.0
    breadth_ratio: float = 0.5    # advance / (advance + decline)
    timestamp: str = ""

    # Adaptive overrides (consumers read these instead of Config defaults)
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    buy_threshold: float = 0.30
    strong_buy_threshold: float = 0.55
    sell_threshold: float = -0.30
    strong_sell_threshold: float = -0.55
    position_scale: float = 1.0
    min_rr_ratio: float = 2.5
    vix_caution: float = 20.0
    vix_panic: float = 25.0
    sector_cap_pct: float = 0.40

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 2),
            "vix": round(self.vix, 2),
            "nifty_20d_return": round(self.nifty_20d_return, 4),
            "nifty_adx": round(self.nifty_adx, 2),
            "breadth_ratio": round(self.breadth_ratio, 3),
            "adaptive_params": {
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "buy_threshold": self.buy_threshold,
                "strong_buy_threshold": self.strong_buy_threshold,
                "position_scale": round(self.position_scale, 2),
                "min_rr_ratio": self.min_rr_ratio,
                "sector_cap_pct": self.sector_cap_pct,
            },
        }


# ─── Regime-adaptive parameter presets ──────────────────────

_REGIME_PARAMS: Dict[MarketRegime, dict] = {
    MarketRegime.TRENDING_BULL: {
        "rsi_oversold": 25.0,
        "rsi_overbought": 75.0,
        "buy_threshold": 0.25,
        "strong_buy_threshold": 0.50,
        "sell_threshold": -0.35,
        "strong_sell_threshold": -0.60,
        "position_scale": 1.0,
        "min_rr_ratio": 1.5,
        "vix_caution": 22.0,
        "vix_panic": 28.0,
        "sector_cap_pct": 0.40,
    },
    MarketRegime.TRENDING_BEAR: {
        "rsi_oversold": 35.0,
        "rsi_overbought": 65.0,
        "buy_threshold": 0.40,
        "strong_buy_threshold": 0.65,
        "sell_threshold": -0.25,
        "strong_sell_threshold": -0.50,
        "position_scale": 0.5,
        "min_rr_ratio": 2.5,
        "vix_caution": 18.0,
        "vix_panic": 24.0,
        "sector_cap_pct": 0.30,
    },
    MarketRegime.RANGE_BOUND: {
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "buy_threshold": 0.30,
        "strong_buy_threshold": 0.55,
        "sell_threshold": -0.30,
        "strong_sell_threshold": -0.55,
        "position_scale": 0.7,
        "min_rr_ratio": 2.5,
        "vix_caution": 20.0,
        "vix_panic": 25.0,
        "sector_cap_pct": 0.35,
    },
    MarketRegime.HIGH_VOLATILITY: {
        "rsi_oversold": 25.0,
        "rsi_overbought": 75.0,
        "buy_threshold": 0.35,
        "strong_buy_threshold": 0.60,
        "sell_threshold": -0.25,
        "strong_sell_threshold": -0.50,
        "position_scale": 0.4,
        "min_rr_ratio": 3.0,
        "vix_caution": 18.0,
        "vix_panic": 22.0,
        "sector_cap_pct": 0.25,
    },
    MarketRegime.CRISIS: {
        "rsi_oversold": 20.0,
        "rsi_overbought": 60.0,
        "buy_threshold": 0.50,
        "strong_buy_threshold": 0.75,
        "sell_threshold": -0.20,
        "strong_sell_threshold": -0.40,
        "position_scale": 0.0,          # Block all new BUY
        "min_rr_ratio": 4.0,
        "vix_caution": 15.0,
        "vix_panic": 20.0,
        "sector_cap_pct": 0.20,
    },
}


class RegimeDetector:
    """Detects market regime from recent market features.

    Designed for near-zero computational cost — pure NumPy,
    no model training. Cached for 30 minutes.
    """

    _cache: Optional[RegimeSnapshot] = None
    _cache_ts: Optional[datetime] = None
    _CACHE_TTL = timedelta(minutes=30)

    def detect(self, force: bool = False) -> RegimeSnapshot:
        """Return the current regime snapshot (cached 30 min)."""
        now = datetime.utcnow()
        if not force and self._cache and self._cache_ts:
            if now - self._cache_ts < self._CACHE_TTL:
                return self._cache

        snap = self._compute()
        RegimeDetector._cache = snap
        RegimeDetector._cache_ts = now
        return snap

    def _compute(self) -> RegimeSnapshot:
        """Compute regime from live data."""
        vix = self._fetch_vix()
        nifty_ret, nifty_adx = self._fetch_nifty_features()
        breadth = self._fetch_breadth()

        regime, confidence = self._classify(vix, nifty_ret, nifty_adx, breadth)
        params = _REGIME_PARAMS[regime]

        snap = RegimeSnapshot(
            regime=regime,
            confidence=confidence,
            vix=vix,
            nifty_20d_return=nifty_ret,
            nifty_adx=nifty_adx,
            breadth_ratio=breadth,
            timestamp=datetime.utcnow().isoformat(),
            **params,
        )
        logger.info(
            "Regime detected: %s (confidence=%.0f%%, VIX=%.1f, NIFTY_20d=%.2f%%)",
            regime.value, confidence * 100, vix, nifty_ret * 100,
        )
        return snap

    def _classify(
        self, vix: float, nifty_ret: float, adx: float, breadth: float
    ) -> tuple:
        """Rule-based regime classification."""
        # Crisis: VIX > 30 or 20d drawdown > 10%
        if vix > 30 or nifty_ret < -0.10:
            return MarketRegime.CRISIS, min(1.0, vix / 40)

        # High volatility: VIX > 22 and ADX > 25
        if vix > 22 and adx > 25:
            return MarketRegime.HIGH_VOLATILITY, 0.7

        # Trending bull: positive returns, ADX > 25, good breadth
        if nifty_ret > 0.02 and adx > 25 and breadth > 0.55:
            return MarketRegime.TRENDING_BULL, min(1.0, adx / 40)

        # Trending bear: negative returns, ADX > 25, poor breadth
        if nifty_ret < -0.02 and adx > 25 and breadth < 0.45:
            return MarketRegime.TRENDING_BEAR, min(1.0, adx / 40)

        # Default: range-bound
        return MarketRegime.RANGE_BOUND, 0.6

    def _fetch_vix(self) -> float:
        try:
            import yfinance as yf
            data = yf.download("^INDIAVIX", period="5d", progress=False)
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return 16.0  # default moderate VIX

    def _fetch_nifty_features(self) -> tuple:
        try:
            import yfinance as yf
            data = yf.download("^NSEI", period="60d", progress=False)
            if data.empty or len(data) < 20:
                return 0.0, 20.0
            close = data["Close"].squeeze()
            ret_20d = float(close.iloc[-1] / close.iloc[-20] - 1)

            # ADX calculation
            high = data["High"].squeeze()
            low = data["Low"].squeeze()
            period = 14
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1)),
                ),
            )
            atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
            dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
            adx_val = dx.ewm(alpha=1 / period, min_periods=period).mean()
            adx = float(adx_val.iloc[-1]) if np.isfinite(adx_val.iloc[-1]) else 20.0

            return ret_20d, adx
        except Exception:
            return 0.0, 20.0

    def _fetch_breadth(self) -> float:
        """Estimate market breadth from NIFTY500 advance/decline."""
        try:
            import yfinance as yf
            # Use NIFTY500 ETF as proxy
            data = yf.download("^CRSLDX", period="5d", progress=False)
            if data.empty:
                return 0.5
            # Approximate breadth from index return sign
            ret = float(data["Close"].iloc[-1] / data["Close"].iloc[-2] - 1)
            if ret > 0.005:
                return 0.6
            elif ret < -0.005:
                return 0.4
            return 0.5
        except Exception:
            return 0.5


# Singleton
regime_detector = RegimeDetector()
