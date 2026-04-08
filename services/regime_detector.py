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
        """Return the current regime snapshot (cached 30 min).

        C9: Auto-invalidates cache if VIX spikes above 25 while
        cached regime was non-crisis (flash-crash protection).
        """
        now = datetime.utcnow()
        if not force and self._cache and self._cache_ts:
            if now - self._cache_ts < self._CACHE_TTL:
                # C9: VIX spike auto-invalidation
                if self._cache.regime not in (MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY):
                    try:
                        live_vix = self._fetch_vix()
                        if live_vix > 25:
                            logger.warning(
                                "C9: VIX=%.1f spike detected, cached regime=%s — forcing recompute",
                                live_vix, self._cache.regime,
                            )
                            force = True
                    except Exception:
                        pass
                if not force:
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
                return float(data["Close"].iloc[-1].item())
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
        """Compute market breadth from NIFTY 50 constituent advance/decline ratio."""
        try:
            import yfinance as yf
            # Use NIFTY 50 top constituents as breadth proxy
            _NIFTY_TICKERS = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
                "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS",
                "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
                "BAJFINANCE.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS",
                "TATAMOTORS.NS", "WIPRO.NS", "HCLTECH.NS", "NTPC.NS",
            ]
            data = yf.download(_NIFTY_TICKERS, period="5d", progress=False, group_by="ticker")
            if data.empty:
                return 0.5
            advances = 0
            declines = 0
            for tkr in _NIFTY_TICKERS:
                try:
                    close = data[tkr]["Close"].dropna()
                    if len(close) >= 2:
                        ret = float(close.iloc[-1] / close.iloc[-2] - 1)
                        if ret > 0:
                            advances += 1
                        elif ret < 0:
                            declines += 1
                except Exception:
                    continue
            total = advances + declines
            if total == 0:
                return 0.5
            return round(advances / total, 3)
        except Exception:
            return 0.5


# Singleton
regime_detector = RegimeDetector()


def get_current_regime() -> RegimeSnapshot:
    """Module-level convenience function for callers that import get_current_regime."""
    return regime_detector.detect()


def detect_regime() -> RegimeSnapshot:
    """Alias used by auto_executor."""
    return regime_detector.detect()


def get_current_vix() -> Optional[float]:
    """Return current VIX value (India VIX from NSE), or None if unavailable."""
    try:
        snap = regime_detector.detect()
        return getattr(snap, 'vix', None)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# Phase 4 (R19d): Backtest-Compatible Regime Detection
# ═══════════════════════════════════════════════════════════════
# Pure OHLCV-based — no live API calls, no yfinance fetches.
# Uses equal-weight index proxy + realized vol + breadth from
# the backtest's own OHLCV data.

import pandas as pd  # noqa: E402 (already imported via numpy above)


class BacktestRegime(str, Enum):
    """Simplified regime for backtest vol-target asymmetry."""
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    CRISIS = "CRISIS"


@dataclass
class BacktestRegimeState:
    """Regime state computed from backtest OHLCV data."""
    regime: BacktestRegime
    vol_target: float          # Recommended annual vol target
    stop_sigma: float          # Recommended stop width (σ)
    index_above_sma200: bool
    realized_vol_pct: float    # 20-day annualized realized vol
    breadth_pct: float         # % symbols above own SMA200
    sma200_distance_pct: float # Index distance from SMA200


# Thresholds calibrated for Indian equity markets:
#   2008 GFC: vol ~50%, SMA200 breach
#   2020 COVID: vol ~45%, SMA200 breach
#   2022 chop: vol ~22%, brief SMA200 breach
#   Bull runs: vol 12-18%, well above SMA200
_BT_VOL_CALM = 18.0     # Below = calm
_BT_VOL_STRESS = 28.0   # Above = stress

# Vol targets per regime (annual, decimal)
_BT_REGIME_VOL = {
    BacktestRegime.BULL:    0.85,
    BacktestRegime.NEUTRAL: 0.65,
    BacktestRegime.BEAR:    0.40,
    BacktestRegime.CRISIS:  0.20,
}

# Stop widths per regime (sigma of daily vol)
_BT_REGIME_STOP = {
    BacktestRegime.BULL:    10.0,  # Wide — let winners run
    BacktestRegime.NEUTRAL:  7.0,
    BacktestRegime.BEAR:     4.0,  # Tighter — cut losers faster
    BacktestRegime.CRISIS:   3.0,  # Tight — capital preservation
}


def detect_backtest_regime(
    ohlcv_slice: dict,
    day_idx: int,
    sma_window: int = 200,
    vol_window: int = 20,
) -> BacktestRegimeState:
    """Detect regime from backtest OHLCV data (no network calls).

    Parameters
    ----------
    ohlcv_slice : dict[str, DataFrame]
        Symbol → OHLCV DataFrame sliced up to current day.
    day_idx : int
        Current simulation day index (used only for logging).
    sma_window : int
        SMA lookback (default 200).
    vol_window : int
        Realized vol lookback (default 20).

    Returns
    -------
    BacktestRegimeState
    """
    # ── 1. Build equal-weight index proxy ──────────────────────
    close_series = []
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_window:
            continue
        first = float(c.iloc[0])
        if first > 0 and np.isfinite(first):
            close_series.append(c / first * 100)

    if len(close_series) < 10:
        return BacktestRegimeState(
            regime=BacktestRegime.NEUTRAL,
            vol_target=_BT_REGIME_VOL[BacktestRegime.NEUTRAL],
            stop_sigma=_BT_REGIME_STOP[BacktestRegime.NEUTRAL],
            index_above_sma200=True,
            realized_vol_pct=20.0,
            breadth_pct=0.50,
            sma200_distance_pct=0.0,
        )

    index = pd.concat(close_series, axis=1).mean(axis=1)

    # ── 2. SMA200 ─────────────────────────────────────────────
    current_price = float(index.iloc[-1])
    sma200 = float(index.iloc[-sma_window:].mean())
    above_sma200 = current_price > sma200
    sma_dist = (current_price / sma200 - 1) * 100 if sma200 > 0 else 0.0

    # ── 3. Realized vol (annualized %) ─────────────────────────
    rets = index.pct_change().dropna()
    if len(rets) >= vol_window:
        realized_vol = float(rets.iloc[-vol_window:].std()) * np.sqrt(252) * 100
    else:
        realized_vol = 20.0

    # ── 4. Market breadth ──────────────────────────────────────
    above_own = 0
    total = 0
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_window:
            continue
        total += 1
        if float(c.iloc[-1]) > float(c.iloc[-sma_window:].mean()):
            above_own += 1
    breadth = above_own / total if total > 0 else 0.50

    # ── 5. Classify ────────────────────────────────────────────
    if above_sma200:
        if realized_vol < _BT_VOL_CALM:
            regime = BacktestRegime.BULL
        elif realized_vol < _BT_VOL_STRESS:
            regime = BacktestRegime.NEUTRAL
        else:
            regime = BacktestRegime.NEUTRAL  # High vol but uptrend
    else:
        if realized_vol >= _BT_VOL_STRESS:
            regime = BacktestRegime.CRISIS
        else:
            regime = BacktestRegime.BEAR

    # ── 6. Breadth confirmation ────────────────────────────────
    if breadth < 0.35:
        if regime == BacktestRegime.BULL:
            regime = BacktestRegime.NEUTRAL
        elif regime == BacktestRegime.NEUTRAL and not above_sma200:
            regime = BacktestRegime.BEAR
        elif regime == BacktestRegime.BEAR:
            regime = BacktestRegime.CRISIS

    # ── 7. Smooth boundary transition (±3% SMA200) ────────────
    vol_target = _BT_REGIME_VOL[regime]
    stop_sigma = _BT_REGIME_STOP[regime]

    if abs(sma_dist) < 3.0:
        if above_sma200:
            lower = BacktestRegime.BEAR if realized_vol < _BT_VOL_STRESS else BacktestRegime.CRISIS
            blend = (3.0 - sma_dist) / 3.0
            vol_target = vol_target * (1 - blend * 0.5) + _BT_REGIME_VOL[lower] * blend * 0.5
            stop_sigma = stop_sigma * (1 - blend * 0.5) + _BT_REGIME_STOP[lower] * blend * 0.5
        else:
            blend = max(0.0, min(1.0, (3.0 + sma_dist) / 3.0))
            vol_target = vol_target * (1 - blend * 0.5) + _BT_REGIME_VOL[BacktestRegime.NEUTRAL] * blend * 0.5
            stop_sigma = stop_sigma * (1 - blend * 0.5) + _BT_REGIME_STOP[BacktestRegime.NEUTRAL] * blend * 0.5

    return BacktestRegimeState(
        regime=regime,
        vol_target=round(vol_target, 4),
        stop_sigma=round(stop_sigma, 2),
        index_above_sma200=above_sma200,
        realized_vol_pct=round(realized_vol, 2),
        breadth_pct=round(breadth, 4),
        sma200_distance_pct=round(sma_dist, 2),
    )


# ═══════════════════════════════════════════════════════════════
# Phase 5 (R19e): FAST Regime Detection
# ═══════════════════════════════════════════════════════════════
# Fixes from R19d failure analysis:
#   1. SMA200 reacts too late → add SMA50 as early-warning
#   2. Vol threshold 28% too high → lower to 22%
#   3. Breadth 35% too generous → tighten to 45%
#   4. 20d momentum (index ROC) as leading indicator
#   5. Vol acceleration: short_vol/long_vol ratio detects spikes
#   6. Narrower blend zone: ±2% instead of ±3%
#   7. BEAR vol target 0.30 (was 0.40) — more defensive

# Vol targets per regime — R19e recalibrated
_BT_REGIME_VOL_FAST = {
    BacktestRegime.BULL:    0.80,   # Slightly less than R19d (0.85) to reduce whipsaw
    BacktestRegime.NEUTRAL: 0.55,   # Reduced from 0.65
    BacktestRegime.BEAR:    0.30,   # Reduced from 0.40 — get small faster
    BacktestRegime.CRISIS:  0.15,   # Reduced from 0.20 — near-flat in crisis
}

# Stop widths — tighter across the board
_BT_REGIME_STOP_FAST = {
    BacktestRegime.BULL:    8.0,    # Was 10 — slightly tighter even in bull
    BacktestRegime.NEUTRAL: 5.0,    # Was 7
    BacktestRegime.BEAR:    3.0,    # Was 4
    BacktestRegime.CRISIS:  2.0,    # Was 3 — very tight capital preservation
}


def detect_backtest_regime_fast(
    ohlcv_slice: dict,
    day_idx: int,
    sma_slow: int = 200,
    sma_fast: int = 50,
    vol_window: int = 20,
) -> BacktestRegimeState:
    """Fast regime detection — R19e version.

    Key differences from detect_backtest_regime (R19d):
      - Dual SMA: SMA50 breach triggers early downgrade
      - Momentum: 20-day index return as leading signal
      - Vol acceleration: 20d vol / 60d vol ratio
      - Tighter breadth threshold (45% vs 35%)
      - Narrower SMA blend zone (±2% vs ±3%)
    """
    # ── 1. Build equal-weight index proxy ──────────────────────
    close_series = []
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_slow:
            continue
        first = float(c.iloc[0])
        if first > 0 and np.isfinite(first):
            close_series.append(c / first * 100)

    if len(close_series) < 10:
        return BacktestRegimeState(
            regime=BacktestRegime.NEUTRAL,
            vol_target=_BT_REGIME_VOL_FAST[BacktestRegime.NEUTRAL],
            stop_sigma=_BT_REGIME_STOP_FAST[BacktestRegime.NEUTRAL],
            index_above_sma200=True,
            realized_vol_pct=20.0,
            breadth_pct=0.50,
            sma200_distance_pct=0.0,
        )

    index = pd.concat(close_series, axis=1).mean(axis=1)
    current_price = float(index.iloc[-1])

    # ── 2. Dual SMA ───────────────────────────────────────────
    sma200_val = float(index.iloc[-sma_slow:].mean())
    sma50_val = float(index.iloc[-sma_fast:].mean()) if len(index) >= sma_fast else sma200_val
    above_sma200 = current_price > sma200_val
    above_sma50 = current_price > sma50_val
    sma_dist = (current_price / sma200_val - 1) * 100 if sma200_val > 0 else 0.0

    # ── 3. Realized vol + vol acceleration ─────────────────────
    rets = index.pct_change().dropna()
    if len(rets) >= vol_window:
        short_vol = float(rets.iloc[-vol_window:].std()) * np.sqrt(252) * 100
    else:
        short_vol = 20.0
    if len(rets) >= 60:
        long_vol = float(rets.iloc[-60:].std()) * np.sqrt(252) * 100
    else:
        long_vol = short_vol
    # Vol acceleration: >1.5 means vol is spiking
    vol_accel = short_vol / long_vol if long_vol > 0 else 1.0

    # ── 4. 20-day momentum (index ROC) ─────────────────────────
    if len(index) >= 20:
        mom_20d = (current_price / float(index.iloc[-20]) - 1) * 100
    else:
        mom_20d = 0.0

    # ── 5. Market breadth (% above own SMA50 — faster than SMA200) ───
    above_own = 0
    total = 0
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_fast:
            continue
        total += 1
        if float(c.iloc[-1]) > float(c.iloc[-sma_fast:].mean()):
            above_own += 1
    breadth = above_own / total if total > 0 else 0.50

    # ── 6. Classify with multi-signal logic ────────────────────
    # Start with base classification
    vol_calm = 16.0     # Tighter than R19d's 18
    vol_stress = 22.0   # Tighter than R19d's 28

    if above_sma200 and above_sma50:
        # Both SMAs confirm uptrend
        if short_vol < vol_calm and mom_20d > -3.0:
            regime = BacktestRegime.BULL
        elif short_vol < vol_stress:
            regime = BacktestRegime.NEUTRAL
        else:
            regime = BacktestRegime.NEUTRAL
    elif above_sma200 and not above_sma50:
        # SMA50 breach = EARLY WARNING — downgrade one level
        if short_vol < vol_stress:
            regime = BacktestRegime.NEUTRAL
        else:
            regime = BacktestRegime.BEAR
    elif not above_sma200 and above_sma50:
        # Mixed — recovering or initial breach
        regime = BacktestRegime.NEUTRAL
    else:
        # Both SMAs breached
        if short_vol >= vol_stress:
            regime = BacktestRegime.CRISIS
        else:
            regime = BacktestRegime.BEAR

    # ── 7. Momentum override — fast crash detection ────────────
    if mom_20d < -8.0:
        # 8%+ drop in 20 days — escalate regardless
        if regime == BacktestRegime.BULL:
            regime = BacktestRegime.NEUTRAL
        elif regime == BacktestRegime.NEUTRAL:
            regime = BacktestRegime.BEAR
        elif regime == BacktestRegime.BEAR:
            regime = BacktestRegime.CRISIS

    if mom_20d < -15.0:
        # 15%+ crash — force CRISIS
        regime = BacktestRegime.CRISIS

    # ── 8. Vol acceleration override ───────────────────────────
    if vol_accel > 1.8 and regime in (BacktestRegime.BULL, BacktestRegime.NEUTRAL):
        # Vol spiking fast — downgrade one level
        if regime == BacktestRegime.BULL:
            regime = BacktestRegime.NEUTRAL
        else:
            regime = BacktestRegime.BEAR

    # ── 9. Breadth confirmation (tighter: 45%) ─────────────────
    if breadth < 0.45:
        if regime == BacktestRegime.BULL:
            regime = BacktestRegime.NEUTRAL
        elif regime == BacktestRegime.NEUTRAL and not above_sma200:
            regime = BacktestRegime.BEAR
        elif regime == BacktestRegime.BEAR:
            regime = BacktestRegime.CRISIS

    # ── 10. Smooth boundary (±2% SMA200 — narrower) ───────────
    vol_target = _BT_REGIME_VOL_FAST[regime]
    stop_sigma = _BT_REGIME_STOP_FAST[regime]

    if abs(sma_dist) < 2.0:
        if above_sma200:
            lower = BacktestRegime.BEAR if short_vol < vol_stress else BacktestRegime.CRISIS
            blend = (2.0 - sma_dist) / 2.0
            vol_target = vol_target * (1 - blend * 0.5) + _BT_REGIME_VOL_FAST[lower] * blend * 0.5
            stop_sigma = stop_sigma * (1 - blend * 0.5) + _BT_REGIME_STOP_FAST[lower] * blend * 0.5
        else:
            blend = max(0.0, min(1.0, (2.0 + sma_dist) / 2.0))
            vol_target = vol_target * (1 - blend * 0.5) + _BT_REGIME_VOL_FAST[BacktestRegime.NEUTRAL] * blend * 0.5
            stop_sigma = stop_sigma * (1 - blend * 0.5) + _BT_REGIME_STOP_FAST[BacktestRegime.NEUTRAL] * blend * 0.5

    return BacktestRegimeState(
        regime=regime,
        vol_target=round(vol_target, 4),
        stop_sigma=round(stop_sigma, 2),
        index_above_sma200=above_sma200,
        realized_vol_pct=round(short_vol, 2),
        breadth_pct=round(breadth, 4),
        sma200_distance_pct=round(sma_dist, 2),
    )


# ═══════════════════════════════════════════════════════════════
# Phase 5b (R19f): Regime Detection with Hysteresis & DD Feedback
# ═══════════════════════════════════════════════════════════════
# Fixes from R19e gap analysis (Day 500-750 correction, -56.3% DD):
#   Gap 1: CRISIS→BULL whipsaw in 25 days — bear rally trap at Day 600
#          reloaded 9 positions into falling market, added ~₹300K loss
#   Gap 2: Regime jumped CRISIS→BULL, skipping BEAR/NEUTRAL
#   Gap 3: Market looked BULL (SMA bounce, breadth 68%) but equity was
#          already -34% from peak — no equity DD feedback
#   Gap 4: 5-day re-check too slow during rapid corrections
#   Gap 5: Positions exploded 2→9 instantly on false BULL
#
# Solutions:
#   Fix 1: Cooldown — min 25 days in CRISIS, 15 in BEAR before upgrade
#   Fix 2: One-step upgrades only (CRISIS→BEAR→NEUTRAL→BULL)
#   Fix 3: Equity DD overlay — >20% DD caps at NEUTRAL, >35% caps at BEAR
#   Fix 4: Adaptive re-check frequency (CRISIS=2d, BEAR=3d, else=5d)
#   Fix 5: Position ramp cap after crisis — handled in pipeline

_REGIME_LEVEL = {
    BacktestRegime.CRISIS:  0,
    BacktestRegime.BEAR:    1,
    BacktestRegime.NEUTRAL: 2,
    BacktestRegime.BULL:    3,
}
_LEVEL_TO_REGIME = {v: k for k, v in _REGIME_LEVEL.items()}

# Minimum days in a defensive regime before allowing upgrade
_UPGRADE_COOLDOWN = {
    BacktestRegime.CRISIS:  25,
    BacktestRegime.BEAR:    15,
    BacktestRegime.NEUTRAL: 0,
    BacktestRegime.BULL:    0,
}

# Vol targets & stops — same as R19e (proven reasonable per-regime)
_BT_REGIME_VOL_V2 = dict(_BT_REGIME_VOL_FAST)
_BT_REGIME_STOP_V2 = dict(_BT_REGIME_STOP_FAST)


def detect_backtest_regime_v2(
    ohlcv_slice: dict,
    day_idx: int,
    prev_state: Optional[BacktestRegimeState] = None,
    days_in_regime: int = 0,
    equity_dd_pct: float = 0.0,
    sma_slow: int = 200,
    sma_fast: int = 50,
    vol_window: int = 20,
) -> BacktestRegimeState:
    """V2 regime detection — R19e signals + hysteresis + equity DD feedback.

    Fixes over R19e:
      - Cooldown: min 25d in CRISIS / 15d in BEAR before upgrade
      - One-step upgrades: CRISIS→BEAR→NEUTRAL→BULL (no skipping)
      - Equity DD overlay: >20% caps NEUTRAL, >35% caps BEAR
      - Downgrades remain instant (safety first)
    """
    # ── 1. Build equal-weight index proxy ──────────────────────
    close_series = []
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_slow:
            continue
        first = float(c.iloc[0])
        if first > 0 and np.isfinite(first):
            close_series.append(c / first * 100)

    if len(close_series) < 10:
        return BacktestRegimeState(
            regime=BacktestRegime.NEUTRAL,
            vol_target=_BT_REGIME_VOL_V2[BacktestRegime.NEUTRAL],
            stop_sigma=_BT_REGIME_STOP_V2[BacktestRegime.NEUTRAL],
            index_above_sma200=True,
            realized_vol_pct=20.0,
            breadth_pct=0.50,
            sma200_distance_pct=0.0,
        )

    index = pd.concat(close_series, axis=1).mean(axis=1)
    current_price = float(index.iloc[-1])

    # ── 2. Dual SMA ───────────────────────────────────────────
    sma200_val = float(index.iloc[-sma_slow:].mean())
    sma50_val = float(index.iloc[-sma_fast:].mean()) if len(index) >= sma_fast else sma200_val
    above_sma200 = current_price > sma200_val
    above_sma50 = current_price > sma50_val
    sma_dist = (current_price / sma200_val - 1) * 100 if sma200_val > 0 else 0.0

    # ── 3. Realized vol + vol acceleration ─────────────────────
    rets = index.pct_change().dropna()
    if len(rets) >= vol_window:
        short_vol = float(rets.iloc[-vol_window:].std()) * np.sqrt(252) * 100
    else:
        short_vol = 20.0
    if len(rets) >= 60:
        long_vol = float(rets.iloc[-60:].std()) * np.sqrt(252) * 100
    else:
        long_vol = short_vol
    vol_accel = short_vol / long_vol if long_vol > 0 else 1.0

    # ── 4. 20-day momentum ────────────────────────────────────
    if len(index) >= 20:
        mom_20d = (current_price / float(index.iloc[-20]) - 1) * 100
    else:
        mom_20d = 0.0

    # ── 5. Market breadth (% above own SMA50) ─────────────────
    above_own = 0
    total = 0
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_fast:
            continue
        total += 1
        if float(c.iloc[-1]) > float(c.iloc[-sma_fast:].mean()):
            above_own += 1
    breadth = above_own / total if total > 0 else 0.50

    # ── 6. Raw regime classification (same as R19e) ────────────
    vol_calm = 16.0
    vol_stress = 22.0

    if above_sma200 and above_sma50:
        if short_vol < vol_calm and mom_20d > -3.0:
            regime = BacktestRegime.BULL
        elif short_vol < vol_stress:
            regime = BacktestRegime.NEUTRAL
        else:
            regime = BacktestRegime.NEUTRAL
    elif above_sma200 and not above_sma50:
        if short_vol < vol_stress:
            regime = BacktestRegime.NEUTRAL
        else:
            regime = BacktestRegime.BEAR
    elif not above_sma200 and above_sma50:
        regime = BacktestRegime.NEUTRAL
    else:
        if short_vol >= vol_stress:
            regime = BacktestRegime.CRISIS
        else:
            regime = BacktestRegime.BEAR

    # ── 7. Momentum override ──────────────────────────────────
    if mom_20d < -8.0:
        if regime == BacktestRegime.BULL:
            regime = BacktestRegime.NEUTRAL
        elif regime == BacktestRegime.NEUTRAL:
            regime = BacktestRegime.BEAR
        elif regime == BacktestRegime.BEAR:
            regime = BacktestRegime.CRISIS
    if mom_20d < -15.0:
        regime = BacktestRegime.CRISIS

    # ── 8. Vol acceleration override ──────────────────────────
    if vol_accel > 1.8 and regime in (BacktestRegime.BULL, BacktestRegime.NEUTRAL):
        if regime == BacktestRegime.BULL:
            regime = BacktestRegime.NEUTRAL
        else:
            regime = BacktestRegime.BEAR

    # ── 9. Breadth confirmation (45%) ─────────────────────────
    if breadth < 0.45:
        if regime == BacktestRegime.BULL:
            regime = BacktestRegime.NEUTRAL
        elif regime == BacktestRegime.NEUTRAL and not above_sma200:
            regime = BacktestRegime.BEAR
        elif regime == BacktestRegime.BEAR:
            regime = BacktestRegime.CRISIS

    # ═══ NEW FIX 1: Equity DD overlay ═════════════════════════
    # If our own equity is bleeding, market signals don't matter
    if equity_dd_pct > 0.35:
        max_level = _REGIME_LEVEL[BacktestRegime.BEAR]
        if _REGIME_LEVEL[regime] > max_level:
            regime = BacktestRegime.BEAR
    elif equity_dd_pct > 0.20:
        max_level = _REGIME_LEVEL[BacktestRegime.NEUTRAL]
        if _REGIME_LEVEL[regime] > max_level:
            regime = BacktestRegime.NEUTRAL

    # ═══ NEW FIX 2: Cooldown + one-step upgrades ══════════════
    if prev_state is not None:
        prev_level = _REGIME_LEVEL[prev_state.regime]
        new_level = _REGIME_LEVEL[regime]

        if new_level > prev_level:
            # Attempting upgrade — enforce cooldown
            min_days = _UPGRADE_COOLDOWN.get(prev_state.regime, 0)
            if days_in_regime < min_days:
                # Not enough time in defensive regime — stay put
                regime = prev_state.regime
            else:
                # Cooldown passed — but only upgrade ONE step
                regime = _LEVEL_TO_REGIME[min(prev_level + 1, 3)]
        # Downgrades are always instant (no restriction)

    # ── 10. Apply vol target and stops ─────────────────────────
    vol_target = _BT_REGIME_VOL_V2[regime]
    stop_sigma = _BT_REGIME_STOP_V2[regime]

    # Smooth SMA boundary (±2%)
    if abs(sma_dist) < 2.0:
        if above_sma200:
            lower = BacktestRegime.BEAR if short_vol < vol_stress else BacktestRegime.CRISIS
            blend = (2.0 - sma_dist) / 2.0
            vol_target = vol_target * (1 - blend * 0.5) + _BT_REGIME_VOL_V2[lower] * blend * 0.5
            stop_sigma = stop_sigma * (1 - blend * 0.5) + _BT_REGIME_STOP_V2[lower] * blend * 0.5
        else:
            blend = max(0.0, min(1.0, (2.0 + sma_dist) / 2.0))
            vol_target = vol_target * (1 - blend * 0.5) + _BT_REGIME_VOL_V2[BacktestRegime.NEUTRAL] * blend * 0.5
            stop_sigma = stop_sigma * (1 - blend * 0.5) + _BT_REGIME_STOP_V2[BacktestRegime.NEUTRAL] * blend * 0.5

    return BacktestRegimeState(
        regime=regime,
        vol_target=round(vol_target, 4),
        stop_sigma=round(stop_sigma, 2),
        index_above_sma200=above_sma200,
        realized_vol_pct=round(short_vol, 2),
        breadth_pct=round(breadth, 4),
        sma200_distance_pct=round(sma_dist, 2),
    )


# ═══════════════════════════════════════════════════════════════
# Phase 5c (R19f): Sigmoid Blend Regime — Continuous Vol Scalar
# ═══════════════════════════════════════════════════════════════
# Fixes R19e's CRISIS→BULL whipsaw by eliminating hard regime
# transitions. Instead of mapping discrete states to vol targets,
# compute a continuous "market health" score → sigmoid → vol/stop.
#
# Key advantages over R19e:
#   - No whipsaw: sigmoid moves gradually, no jumps
#   - No cooldown logic needed: smoothness is inherent
#   - Bear rally trap proof: a bounce lifts score slightly but
#     can't push through sigmoid fast enough to reload positions
#   - Equity DD overlay: bleeds vol_target when our equity hurts
#     regardless of what market signals say

# Sigmoid tuning — k controls slope (steepness)
# k=4.5: steeper curve — reaches 85%/15% output faster, more decisive
# (k=3.0 in v1 was too gentle, mild bull only got vol=0.62)
_SIGMOID_K = 4.5
_SIGMOID_MID = 0.0

# Vol target range [floor, ceiling]
_SIGMOID_VOL_FLOOR = 0.15    # Maximum defensive (≈ CRISIS)
_SIGMOID_VOL_CEIL = 0.85     # Maximum aggressive (≈ BULL) — was 0.80

# Stop sigma range [tight, wide]
_SIGMOID_STOP_FLOOR = 2.0    # Tightest (crisis-level)
_SIGMOID_STOP_CEIL = 8.5     # Widest (bull-level) — was 8.0

# Feature weights for composite score
_FEAT_W_SMA = 0.30       # SMA distance (trend strength)
_FEAT_W_VOL = 0.25       # Realized vol (inverse — low vol = good)
_FEAT_W_BREADTH = 0.25   # Market breadth (participation)
_FEAT_W_MOM = 0.20       # 20-day momentum (direction)

# Feature normalization anchors (calibrated from Indian equity 2012-2025)
# sma_dist: range [-15%, +30%], center ~+2% (was +5%, too conservative)
# Lowered so that being +10% above SMA already scores strongly bullish
_SMA_CENTER = 2.0
_SMA_SCALE = 15.0     # ±15% from center = ±1 unit
# rvol: range [8%, 45%], center ~16%
_VOL_CENTER = 16.0
_VOL_SCALE = 12.0     # ±12% from center = ±1 unit (inverted)
# breadth: range [10%, 95%], center ~50%
_BREADTH_CENTER = 0.50
_BREADTH_SCALE = 0.30  # ±30pp from center = ±1 unit
# mom_20d: range [-20%, +15%], center ~+1%
_MOM_CENTER = 1.0
_MOM_SCALE = 8.0       # ±8% from center = ±1 unit


def _sigmoid(x: float, k: float = _SIGMOID_K, mid: float = _SIGMOID_MID) -> float:
    """Sigmoid with tunable slope and midpoint. Returns [0, 1]."""
    z = k * (x - mid)
    z = max(-20.0, min(20.0, z))  # clamp to prevent overflow
    return 1.0 / (1.0 + np.exp(-z))


def detect_backtest_regime_sigmoid(
    ohlcv_slice: dict,
    day_idx: int,
    equity_dd_pct: float = 0.0,
    sma_slow: int = 200,
    sma_fast: int = 50,
    vol_window: int = 20,
) -> BacktestRegimeState:
    """Sigmoid blend regime detection — R19f.

    Returns continuous vol_target and stop_sigma via sigmoid of
    composite market health score + equity DD bleed-down.
    Regime label is advisory only (for logging).
    """
    # ── 1. Build equal-weight index proxy ──────────────────────
    close_series = []
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_slow:
            continue
        first = float(c.iloc[0])
        if first > 0 and np.isfinite(first):
            close_series.append(c / first * 100)

    if len(close_series) < 10:
        return BacktestRegimeState(
            regime=BacktestRegime.NEUTRAL,
            vol_target=0.55,
            stop_sigma=5.0,
            index_above_sma200=True,
            realized_vol_pct=20.0,
            breadth_pct=0.50,
            sma200_distance_pct=0.0,
        )

    index = pd.concat(close_series, axis=1).mean(axis=1)
    current_price = float(index.iloc[-1])

    # ── 2. Dual SMA ───────────────────────────────────────────
    sma200_val = float(index.iloc[-sma_slow:].mean())
    sma50_val = float(index.iloc[-sma_fast:].mean()) if len(index) >= sma_fast else sma200_val
    above_sma200 = current_price > sma200_val
    sma_dist = (current_price / sma200_val - 1) * 100 if sma200_val > 0 else 0.0

    # ── 3. Realized vol ───────────────────────────────────────
    rets = index.pct_change().dropna()
    if len(rets) >= vol_window:
        short_vol = float(rets.iloc[-vol_window:].std()) * np.sqrt(252) * 100
    else:
        short_vol = 20.0

    # ── 4. 20-day momentum ────────────────────────────────────
    if len(index) >= 20:
        mom_20d = (current_price / float(index.iloc[-20]) - 1) * 100
    else:
        mom_20d = 0.0

    # ── 5. Market breadth (% above own SMA50) ─────────────────
    above_own = 0
    total = 0
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_fast:
            continue
        total += 1
        if float(c.iloc[-1]) > float(c.iloc[-sma_fast:].mean()):
            above_own += 1
    breadth = above_own / total if total > 0 else 0.50

    # ── 6. Normalize features to ~[-2, +2] ────────────────────
    f_sma = (sma_dist - _SMA_CENTER) / _SMA_SCALE
    f_vol = -((short_vol - _VOL_CENTER) / _VOL_SCALE)   # Inverted: low vol = positive
    f_breadth = (breadth - _BREADTH_CENTER) / _BREADTH_SCALE
    f_mom = (mom_20d - _MOM_CENTER) / _MOM_SCALE

    # ── 7. Composite health score ─────────────────────────────
    score = (_FEAT_W_SMA * f_sma
             + _FEAT_W_VOL * f_vol
             + _FEAT_W_BREADTH * f_breadth
             + _FEAT_W_MOM * f_mom)

    # ── 8. Sigmoid → continuous [0, 1] ────────────────────────
    sig = _sigmoid(score)

    # ── 9. Map to vol_target and stop_sigma ────────────────────
    vol_target = _SIGMOID_VOL_FLOOR + sig * (_SIGMOID_VOL_CEIL - _SIGMOID_VOL_FLOOR)
    stop_sigma = _SIGMOID_STOP_FLOOR + sig * (_SIGMOID_STOP_CEIL - _SIGMOID_STOP_FLOOR)

    # ── 10. Equity DD overlay — bleed down when our equity hurts
    # Starts at 20% DD (was 10% — don't penalize normal pullbacks for a
    # high-CAGR strategy), maxes at 45% DD (70% penalty)
    if equity_dd_pct > 0.20:
        dd_penalty = min(0.7, (equity_dd_pct - 0.20) / 0.25 * 0.7)
        vol_target *= (1.0 - dd_penalty)
        stop_sigma *= (1.0 - dd_penalty * 0.5)  # Stops tighten at half rate

    # Floor: never go below crisis minimums
    vol_target = max(_SIGMOID_VOL_FLOOR, vol_target)
    stop_sigma = max(_SIGMOID_STOP_FLOOR, stop_sigma)

    # ── 11. Advisory regime label (for logging only) ───────────
    if sig > 0.70:
        regime = BacktestRegime.BULL
    elif sig > 0.45:
        regime = BacktestRegime.NEUTRAL
    elif sig > 0.20:
        regime = BacktestRegime.BEAR
    else:
        regime = BacktestRegime.CRISIS

    return BacktestRegimeState(
        regime=regime,
        vol_target=round(vol_target, 4),
        stop_sigma=round(stop_sigma, 2),
        index_above_sma200=above_sma200,
        realized_vol_pct=round(short_vol, 2),
        breadth_pct=round(breadth, 4),
        sma200_distance_pct=round(sma_dist, 2),
    )


# ═══════════════════════════════════════════════════════════════════
# R19g — Sigmoid v2: fixes R19f's three killers
#   1. Higher vol floor (0.40 vs 0.15) — never go to zero positions
#   2. Higher stop floor (6.0σ vs 2.0σ) — no premature stop-outs
#   3. Gentler equity DD overlay (starts 35%, max 40% penalty at 55%)
#      with NO stop tightening — only vol_target is penalized
#   4. Gentler sigmoid (k=3.5 vs 4.5) — smoother transitions
# ═══════════════════════════════════════════════════════════════════
_V2_SIGMOID_K = 3.5
_V2_SIGMOID_MID = 0.0

_V2_VOL_FLOOR = 0.40      # R19c floor = 0.40 — never below this
_V2_VOL_CEIL = 0.80       # Slight cap to limit leverage in euphoria

_V2_STOP_FLOOR = 6.0      # Never below 6σ (R19c uses fixed 10σ)
_V2_STOP_CEIL = 10.0      # Match R19c bull-level stops

# Same feature weights as v1 — these worked well
_V2_FEAT_W_SMA = 0.30
_V2_FEAT_W_VOL = 0.25
_V2_FEAT_W_BREADTH = 0.25
_V2_FEAT_W_MOM = 0.20

# Same normalization anchors
_V2_SMA_CENTER = 2.0
_V2_SMA_SCALE = 15.0
_V2_VOL_CENTER = 16.0
_V2_VOL_SCALE = 12.0
_V2_BREADTH_CENTER = 0.50
_V2_BREADTH_SCALE = 0.30
_V2_MOM_CENTER = 1.0
_V2_MOM_SCALE = 8.0

# Equity DD overlay — much gentler than v1
_V2_DD_START = 0.35        # Don't penalize until 35% DD (was 20%)
_V2_DD_FULL = 0.55         # Full penalty at 55% DD (was 45%)
_V2_DD_MAX_PENALTY = 0.40  # Max 40% vol reduction (was 70%)


def detect_backtest_regime_sigmoid_v2(
    ohlcv_slice: dict,
    day_idx: int,
    equity_dd_pct: float = 0.0,
    sma_slow: int = 200,
    sma_fast: int = 50,
    vol_window: int = 20,
) -> BacktestRegimeState:
    """Sigmoid v2 regime detection — R19g.

    Fixes R19f's three performance killers:
    1. Vol floor raised from 0.15→0.40 (always maintain positions)
    2. Stop floor raised from 2.0σ→6.0σ (no premature stop-outs)
    3. Equity DD overlay gentler: starts 35%, max 40% penalty, no stop tightening

    Returns continuous vol_target [0.40, 0.80] and stop_sigma [6.0, 10.0].
    """
    # ── 1. Build equal-weight index proxy ──────────────────────
    close_series = []
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_slow:
            continue
        first = float(c.iloc[0])
        if first > 0 and np.isfinite(first):
            close_series.append(c / first * 100)

    if len(close_series) < 10:
        return BacktestRegimeState(
            regime=BacktestRegime.NEUTRAL,
            vol_target=0.60,
            stop_sigma=8.0,
            index_above_sma200=True,
            realized_vol_pct=20.0,
            breadth_pct=0.50,
            sma200_distance_pct=0.0,
        )

    index = pd.concat(close_series, axis=1).mean(axis=1)
    current_price = float(index.iloc[-1])

    # ── 2. Dual SMA ───────────────────────────────────────────
    sma200_val = float(index.iloc[-sma_slow:].mean())
    above_sma200 = current_price > sma200_val
    sma_dist = (current_price / sma200_val - 1) * 100 if sma200_val > 0 else 0.0

    # ── 3. Realized vol ───────────────────────────────────────
    rets = index.pct_change().dropna()
    if len(rets) >= vol_window:
        short_vol = float(rets.iloc[-vol_window:].std()) * np.sqrt(252) * 100
    else:
        short_vol = 20.0

    # ── 4. 20-day momentum ────────────────────────────────────
    if len(index) >= 20:
        mom_20d = (current_price / float(index.iloc[-20]) - 1) * 100
    else:
        mom_20d = 0.0

    # ── 5. Market breadth (% above own SMA50) ─────────────────
    above_own = 0
    total = 0
    for sym, df in ohlcv_slice.items():
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        if len(c) < sma_fast:
            continue
        total += 1
        if float(c.iloc[-1]) > float(c.iloc[-sma_fast:].mean()):
            above_own += 1
    breadth = above_own / total if total > 0 else 0.50

    # ── 6. Normalize features to ~[-2, +2] ────────────────────
    f_sma = (sma_dist - _V2_SMA_CENTER) / _V2_SMA_SCALE
    f_vol = -((short_vol - _V2_VOL_CENTER) / _V2_VOL_SCALE)
    f_breadth = (breadth - _V2_BREADTH_CENTER) / _V2_BREADTH_SCALE
    f_mom = (mom_20d - _V2_MOM_CENTER) / _V2_MOM_SCALE

    # ── 7. Composite health score ─────────────────────────────
    score = (_V2_FEAT_W_SMA * f_sma
             + _V2_FEAT_W_VOL * f_vol
             + _V2_FEAT_W_BREADTH * f_breadth
             + _V2_FEAT_W_MOM * f_mom)

    # ── 8. Sigmoid → continuous [0, 1] ────────────────────────
    sig = _sigmoid(score, k=_V2_SIGMOID_K, mid=_V2_SIGMOID_MID)

    # ── 9. Map to vol_target and stop_sigma ────────────────────
    vol_target = _V2_VOL_FLOOR + sig * (_V2_VOL_CEIL - _V2_VOL_FLOOR)
    #          = 0.40           + sig * 0.40 → range [0.40, 0.80]
    stop_sigma = _V2_STOP_FLOOR + sig * (_V2_STOP_CEIL - _V2_STOP_FLOOR)
    #          = 6.0            + sig * 4.0  → range [6.0, 10.0]

    # ── 10. Equity DD overlay — GENTLE, vol_target only ────────
    # Starts at 35% DD, maxes at 55% DD (40% penalty)
    # NO stop tightening — stops must stay wide to avoid death-by-stops
    if equity_dd_pct > _V2_DD_START:
        dd_frac = min(1.0, (equity_dd_pct - _V2_DD_START)
                       / (_V2_DD_FULL - _V2_DD_START))
        dd_penalty = dd_frac * _V2_DD_MAX_PENALTY
        vol_target *= (1.0 - dd_penalty)

    # Floor: never go below minimums
    vol_target = max(_V2_VOL_FLOOR, vol_target)
    stop_sigma = max(_V2_STOP_FLOOR, stop_sigma)

    # ── 11. Advisory regime label (logging only) ───────────────
    if sig > 0.70:
        regime = BacktestRegime.BULL
    elif sig > 0.45:
        regime = BacktestRegime.NEUTRAL
    elif sig > 0.20:
        regime = BacktestRegime.BEAR
    else:
        regime = BacktestRegime.CRISIS

    return BacktestRegimeState(
        regime=regime,
        vol_target=round(vol_target, 4),
        stop_sigma=round(stop_sigma, 2),
        index_above_sma200=above_sma200,
        realized_vol_pct=round(short_vol, 2),
        breadth_pct=round(breadth, 4),
        sma200_distance_pct=round(sma_dist, 2),
    )
