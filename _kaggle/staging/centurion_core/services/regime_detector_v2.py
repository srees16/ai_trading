"""
Enhanced Regime Detector v2 — Gap #3 Implementation.

Multi-indicator regime detection with crisis triggers to avoid 2-week lags
in identifying bear markets. Combines:
  • SMA200 crossovers (primary trend)
  • RSI (momentum confirmation)
  • VIX-equivalent (market fear)
  • Current drawdown (equity line stress)

Expected Impact:
  • Backtest: -0.5–1.5% CAGR (fewer bull trades during fake recoveries)
  • MaxDD: -2–3% reduction (earlier regime shifts to defensive sizing)
  • Sharpe: +0.1–0.2 (better capture of regime transitions)

Crisis Trigger Rules:
  1. If DD > 30% OR VIX > 35 → CRISIS (immediate, override SMA)
  2. If RSI < 25 AND equity < SMA×0.95 → SEVERE_BEAR
  3. If equity < SMA×0.98 OR RSI < 40 → BEAR
  4. If SMA×0.98 < equity < SMA×1.05 → NEUTRAL
  5. If equity > SMA×1.05 OR RSI > 60 → BULL
  6. If equity > SMA×1.10 AND RSI > 70 → STRONG_BULL
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime states."""
    CRISIS = "CRISIS"           # VIX >35 or DD >30%
    SEVERE_BEAR = "SEVERE_BEAR" # RSI <25 + SMA<0.95
    BEAR = "BEAR"               # Equity <SMA ×0.98 or RSI <40
    NEUTRAL = "NEUTRAL"         # Range-bound (0.98 < SMA < 1.05)
    BULL = "BULL"               # Uptrend (SMA >1.05 or RSI >60)
    STRONG_BULL = "STRONG_BULL" # Strong uptrend (SMA >1.10 and RSI >70)


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Compute RSI (Relative Strength Index) for recent price series.
    
    Parameters
    ----------
    prices : pd.Series
        Historical prices (should be at least period+1 elements).
    period : int
        RSI period (default 14 days).
    
    Returns
    -------
    float
        RSI value (0 to 100). Returns 50.0 if insufficient data.
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI if insufficient data
    
    try:
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.0001)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    except Exception as e:
        logger.warning("RSI computation failed: %s", e)
        return 50.0


def compute_vix_equivalent(returns: pd.Series, window: int = 20) -> float:
    """
    Compute VIX-equivalent (annualized volatility of returns).
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    window : int
        Rolling window for volatility (default 20 days = 1 month).
    
    Returns
    -------
    float
        Annualized volatility as a level (0 to 100 scale, where 20 ≈ 20% vol).
    """
    if len(returns) < window:
        return 20.0  # Default neutral VIX
    
    try:
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        vix_equiv = float(rolling_vol.iloc[-1] * 100) if not rolling_vol.empty else 20.0
        return max(10.0, min(100.0, vix_equiv))  # Clamp [10, 100]
    except Exception as e:
        logger.warning("VIX equivalent computation failed: %s", e)
        return 20.0


def compute_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute current drawdown from peak.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Daily equity values.
    
    Returns
    -------
    float
        Drawdown as decimal (0.0 = at peak, 0.35 = 35% below peak).
    """
    if len(equity_curve) < 1 or equity_curve.iloc[-1] <= 0:
        return 0.0
    
    try:
        peak = equity_curve.max()
        current = equity_curve.iloc[-1]
        if peak <= 0:
            return 0.0
        dd = (peak - current) / peak
        return max(0.0, dd)
    except Exception:
        return 0.0


def detect_regime_v2(
    equity_curve: pd.Series,
    prices: pd.Series,
    lookback_sma: int = 200,
    rsi_period: int = 14,
    vix_window: int = 20,
    prev_regime: str = "NEUTRAL",
) -> Tuple[str, Dict[str, float]]:
    """
    Multi-indicator regime detection with crisis triggers.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Daily equity values.
    prices : pd.Series
        Daily prices (for SMA + RSI computation).
    lookback_sma : int
        SMA period (default 200).
    rsi_period : int
        RSI period (default 14).
    vix_window : int
        Window for VIX-equivalent (default 20).
    prev_regime : str
        Previous regime (for hysteresis/smoothing, optional).
    
    Returns
    -------
    Tuple[str, Dict]
        (regime, {"sma": float, "rsi": float, "vix": float, "dd": float})
    
    Examples
    --------
    >>> prices = pd.Series([100, 101, 102, ...])  # 200+ points
    >>> equity = pd.Series([500000, 501000, ...])
    >>> regime, metrics = detect_regime_v2(equity, prices)
    >>> print(f"Regime: {regime}, RSI: {metrics['rsi']:.1f}, VIX: {metrics['vix']:.1f}")
    """
    
    # ── Compute SMA200 ──
    if len(prices) < lookback_sma:
        sma = prices.iloc[-1] if len(prices) > 0 else 100.0
    else:
        sma = float(prices.rolling(lookback_sma).mean().iloc[-1])
    
    current_price = float(prices.iloc[-1]) if len(prices) > 0 else sma
    
    # ── Compute RSI ──
    rsi = compute_rsi(prices, period=rsi_period)
    
    # ── Compute VIX-equivalent ──
    if len(prices) > 1:
        returns = np.log(prices / prices.shift(1))
        vix = compute_vix_equivalent(returns, window=vix_window)
    else:
        vix = 20.0
    
    # ── Compute Drawdown ──
    dd = compute_drawdown(equity_curve)
    
    # ── Regime Decision Logic (priority order) ──
    
    # 1. CRISIS check (immediate, regardless of SMA)
    if dd > 0.30 or vix > 35.0:
        regime = Regime.CRISIS.value
        logger.info(
            "[CRISIS TRIGGER] DD=%.1f%% or VIX=%.1f (threshold: DD>30%% or VIX>35)",
            dd * 100, vix
        )
    
    # 2. SEVERE_BEAR check (RSI very low + below SMA)
    elif rsi < 25.0 and current_price < sma * 0.95:
        regime = Regime.SEVERE_BEAR.value
    
    # 3. BEAR check (below SMA or RSI weak)
    elif current_price < sma * 0.98 or rsi < 40.0:
        regime = Regime.BEAR.value
    
    # 4. NEUTRAL check (range-bound)
    elif sma * 0.98 <= current_price <= sma * 1.05:
        regime = Regime.NEUTRAL.value
    
    # 5. BULL / STRONG_BULL check
    elif current_price > sma * 1.10 and rsi > 70.0:
        regime = Regime.STRONG_BULL.value
    
    elif current_price > sma * 1.05 or rsi > 60.0:
        regime = Regime.BULL.value
    
    else:
        regime = prev_regime  # No clear signal, maintain previous
    
    metrics = {
        "sma": sma,
        "current_price": current_price,
        "rsi": rsi,
        "vix": vix,
        "dd": dd,
    }
    
    logger.debug(
        "Regime: %s | SMA=%.0f Price=%.0f RSI=%.1f VIX=%.1f DD=%.1f%%",
        regime, sma, current_price, rsi, vix, dd * 100
    )
    
    return regime, metrics


def get_regime_position_caps_v2(regime: str) -> Tuple[float, int, int]:
    """
    Get position sizing parameters for the detected regime.
    
    Returns
    -------
    Tuple[float, int, int]
        (investable_threshold, max_positions_cap, min_positions)
    
    Usage
    -----
    >>> regime, _ = detect_regime_v2(equity, prices)
    >>> threshold, cap, min_pos = get_regime_position_caps_v2(regime)
    """
    
    regime_caps = {
        Regime.CRISIS.value: (6.0, 0, 0),           # HALT ALL TRADING
        Regime.SEVERE_BEAR.value: (5.0, 5, 1),      # Ultra defensive
        Regime.BEAR.value: (4.0, 12, 1),            # Defensive
        Regime.NEUTRAL.value: (3.0, 15, 1),         # Balanced
        Regime.BULL.value: (2.5, 20, 2),            # Offensive
        Regime.STRONG_BULL.value: (2.0, 25, 3),     # Very offensive
    }
    
    return regime_caps.get(regime, (3.0, 15, 1))


def get_regime_vol_scalar_v2(regime: str) -> float:
    """
    Get volatility scaling multiplier for the regime.
    
    Returns
    -------
    float
        Multiplier to apply to vol target (e.g., 0.55× in bear = lower risk)
    """
    
    regime_scalars = {
        Regime.CRISIS.value: 0.0,           # No trading
        Regime.SEVERE_BEAR.value: 0.40,     # 40% of vol target
        Regime.BEAR.value: 0.55,            # 55% of vol target
        Regime.NEUTRAL.value: 1.0,          # 100% of vol target
        Regime.BULL.value: 1.25,            # 125% of vol target (boost)
        Regime.STRONG_BULL.value: 1.35,     # 135% of vol target (strong boost)
    }
    
    return regime_scalars.get(regime, 1.0)
