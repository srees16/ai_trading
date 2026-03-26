"""
RL Trading Bot — Feature Builder.

Reuses existing Centurion components to construct the observation
(state) vector for the RL agent.  No indicator re-computation —
everything delegates to existing services.

Feature groups:
  1. Technical indicators  (from MetricsCalculator + TA aggregator)
  2. Quant signals         (momentum, volatility, volume)
  3. Fundamental features  (PE, ROE, EPS, Z-score, F-score)
  4. Regime context        (RegimeDetector snapshot)
  5. Portfolio state       (cash, holdings, entry price, drawdown)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature dimension registry ──────────────────────────────────────
# Each group defines its feature names.  The total observation size
# is the sum across all groups.

TECHNICAL_FEATURES = [
    "rsi_norm",                # RSI / 100
    "macd_histogram_norm",     # MACD-hist / price
    "bb_pband",                # Bollinger %B (0-1)
    "adx_norm",                # ADX / 100
    "obv_divergence",          # OBV vs OBV-SMA (binary-ish)
    "supertrend_dir",          # +1 / -1
    "stoch_rsi_k_norm",        # StochRSI-K / 100
    "williams_r_norm",         # (Williams%R + 100) / 100
    "cci_norm",                # CCI / 400 clamped
    "mfi_norm",                # MFI / 100
    "cmf",                     # CMF raw (-1..+1)
    "ta_fused_score",          # Advanced TA fused score (-1..+1)
    "ta_trend_score",          # Trend category
    "ta_momentum_score",       # Momentum category
    "ta_volatility_score",     # Volatility category
    "ta_volume_score",         # Volume category
]

QUANT_FEATURES = [
    "return_1d",               # 1-day % return
    "return_5d",               # 5-day % return
    "return_20d",              # 20-day % return
    "volatility_20d",          # 20-day annualised volatility
    "volume_ratio",            # current volume / 20-day avg
    "atr_pct",                 # ATR / price (normalised)
    "momentum_score",          # 20d return / 20d vol (mini Sharpe)
    "price_sma20_ratio",       # price / SMA20
    "price_sma50_ratio",       # price / SMA50
]

FUNDAMENTAL_FEATURES = [
    "peg_ratio_norm",          # PEG clamped to [0, 5] then / 5
    "roe_norm",                # ROE clamped [-50, 50] then / 50
    "piotroski_f_norm",        # F-Score / 9
    "altman_z_norm",           # Z-Score clamped [0, 5] then / 5
    "intrinsic_value_ratio",   # intrinsic / price
]

REGIME_FEATURES = [
    "regime_bull",             # one-hot
    "regime_bear",
    "regime_range",
    "regime_vol",
    "regime_crisis",
    "regime_position_scale",   # 0-1
]

PORTFOLIO_FEATURES = [
    "cash_ratio",              # cash / initial_capital
    "holdings_ratio",          # holdings_value / portfolio_value
    "unrealised_pnl_pct",     # (current - entry) / entry
    "position_drawdown",       # drawdown from peak position value
    "days_in_position",        # normalised (days / 60)
]

ALL_FEATURES = (
    TECHNICAL_FEATURES
    + QUANT_FEATURES
    + FUNDAMENTAL_FEATURES
    + REGIME_FEATURES
    + PORTFOLIO_FEATURES
)

FEATURE_DIM = len(ALL_FEATURES)


# ── Builder ─────────────────────────────────────────────────────────

def build_features_from_ohlcv(
    ohlcv: pd.DataFrame,
    step: int,
    *,
    lookback: int = 60,
    metrics: Optional[object] = None,
    regime: Optional[object] = None,
    portfolio_state: Optional[Dict] = None,
) -> np.ndarray:
    """Build a single observation vector from OHLCV + optional live data.

    Args:
        ohlcv: Full OHLCV DataFrame (already flattened, standard columns).
        step: Current bar index into ohlcv.
        lookback: How many bars of history we're allowed to use.
        metrics: Optional ``StockMetrics`` from MetricsCalculator.
        regime: Optional ``RegimeSnapshot`` from RegimeDetector.
        portfolio_state: Optional dict with keys:
            cash, holdings_value, portfolio_value, entry_price,
            current_price, initial_capital, peak_value, days_held.

    Returns:
        np.ndarray of shape (FEATURE_DIM,), dtype float32.
    """
    obs = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Slice historical data for quant features
    start = max(0, step - lookback)
    window = ohlcv.iloc[start:step + 1]

    if len(window) < 2:
        return obs

    close = window["Close"]
    high = window["High"]
    low = window["Low"]
    volume = window["Volume"]
    current_price = float(close.iloc[-1])

    idx = 0

    # ── 1. Technical features ───────────────────────────────
    if metrics is not None:
        obs[idx] = _norm(getattr(metrics, "rsi", None), 0, 100)
        idx += 1
        # MACD histogram normalised by price
        mh = getattr(metrics, "macd_histogram", None)
        obs[idx] = _clamp((mh / current_price * 100) if mh and current_price else 0, -2, 2) / 2
        idx += 1
        obs[idx] = _clamp(getattr(metrics, "ta_volatility_score", None) or _bb_pband(metrics, current_price), -1, 1)
        idx += 1
        obs[idx] = _norm(getattr(metrics, "adx", None), 0, 100)
        idx += 1
        # OBV divergence
        obv = getattr(metrics, "obv", None)
        obv_sma = getattr(metrics, "obv_sma", None)
        obs[idx] = 1.0 if (obv and obv_sma and obv > obv_sma) else -1.0 if (obv and obv_sma) else 0.0
        idx += 1
        obs[idx] = float(getattr(metrics, "supertrend_direction", None) or 0)
        idx += 1
        obs[idx] = _norm(getattr(metrics, "stoch_rsi_k", None), 0, 100)
        idx += 1
        obs[idx] = _norm(_shift(getattr(metrics, "williams_r", None), 100), 0, 100)
        idx += 1
        obs[idx] = _clamp((getattr(metrics, "cci", None) or 0) / 400, -1, 1)
        idx += 1
        obs[idx] = _norm(getattr(metrics, "mfi", None), 0, 100)
        idx += 1
        obs[idx] = _clamp(getattr(metrics, "cmf", None) or 0, -1, 1)
        idx += 1
        obs[idx] = _clamp(getattr(metrics, "ta_fused_score", None) or 0, -1, 1)
        idx += 1
        obs[idx] = _clamp(getattr(metrics, "ta_trend_score", None) or 0, -1, 1)
        idx += 1
        obs[idx] = _clamp(getattr(metrics, "ta_momentum_score", None) or 0, -1, 1)
        idx += 1
        obs[idx] = _clamp(getattr(metrics, "ta_volatility_score", None) or 0, -1, 1)
        idx += 1
        obs[idx] = _clamp(getattr(metrics, "ta_volume_score", None) or 0, -1, 1)
        idx += 1
    else:
        # Compute from raw OHLCV (lightweight fallback)
        obs[idx] = _norm(_compute_rsi(close), 0, 100); idx += 1
        obs[idx] = 0.0; idx += 1  # MACD skipped
        obs[idx] = _compute_bb_pband(close); idx += 1
        obs[idx] = 0.0; idx += 1  # ADX skipped
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1
        obs[idx] = 0.0; idx += 1

    # ── 2. Quant features ───────────────────────────────────
    obs[idx] = _clamp(_pct_return(close, 1), -0.2, 0.2) / 0.2; idx += 1
    obs[idx] = _clamp(_pct_return(close, 5), -0.5, 0.5) / 0.5; idx += 1
    obs[idx] = _clamp(_pct_return(close, 20), -1, 1); idx += 1
    vol20 = _volatility(close, 20)
    obs[idx] = _clamp(vol20, 0, 1); idx += 1
    vol_ratio = float(volume.iloc[-1]) / float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 and volume.rolling(20).mean().iloc[-1] > 0 else 1.0
    obs[idx] = _clamp(vol_ratio / 3, 0, 1); idx += 1
    atr_val = getattr(metrics, "atr", None) if metrics else None
    obs[idx] = _clamp((atr_val / current_price) if atr_val and current_price else 0, 0, 0.1) / 0.1; idx += 1
    # Momentum score = 20d return / 20d vol
    mom = _pct_return(close, 20) / vol20 if vol20 > 0.001 else 0
    obs[idx] = _clamp(mom / 3, -1, 1); idx += 1
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else current_price
    obs[idx] = _clamp(current_price / sma20 - 1, -0.3, 0.3) / 0.3; idx += 1
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current_price
    obs[idx] = _clamp(current_price / sma50 - 1, -0.5, 0.5) / 0.5; idx += 1

    # ── 3. Fundamental features ─────────────────────────────
    if metrics is not None:
        obs[idx] = _clamp((getattr(metrics, "peg_ratio", None) or 2.5) / 5, 0, 1); idx += 1
        obs[idx] = _clamp((getattr(metrics, "roe", None) or 0) / 50, -1, 1); idx += 1
        obs[idx] = (getattr(metrics, "piotroski_f_score", None) or 5) / 9; idx += 1
        obs[idx] = _clamp((getattr(metrics, "altman_z_score", None) or 2.5) / 5, 0, 1); idx += 1
        iv = getattr(metrics, "intrinsic_value", None)
        cp = getattr(metrics, "current_price", None)
        obs[idx] = _clamp(iv / cp - 1, -1, 1) if (iv and cp and cp > 0) else 0; idx += 1
    else:
        idx += len(FUNDAMENTAL_FEATURES)

    # ── 4. Regime features ──────────────────────────────────
    if regime is not None:
        regime_name = getattr(regime, "regime", None)
        r_str = regime_name.value if hasattr(regime_name, "value") else str(regime_name or "")
        obs[idx] = 1.0 if "BULL" in r_str else 0.0; idx += 1
        obs[idx] = 1.0 if "BEAR" in r_str else 0.0; idx += 1
        obs[idx] = 1.0 if "RANGE" in r_str else 0.0; idx += 1
        obs[idx] = 1.0 if "VOLATILITY" in r_str else 0.0; idx += 1
        obs[idx] = 1.0 if "CRISIS" in r_str else 0.0; idx += 1
        obs[idx] = _clamp(getattr(regime, "position_scale", 1.0), 0, 1); idx += 1
    else:
        idx += len(REGIME_FEATURES)

    # ── 5. Portfolio state ──────────────────────────────────
    ps = portfolio_state or {}
    init_cap = ps.get("initial_capital", 1.0)
    obs[idx] = _clamp(ps.get("cash", init_cap) / init_cap, 0, 1); idx += 1
    pv = ps.get("portfolio_value", init_cap)
    obs[idx] = _clamp(ps.get("holdings_value", 0) / pv if pv > 0 else 0, 0, 1); idx += 1
    entry = ps.get("entry_price", 0)
    obs[idx] = _clamp((current_price / entry - 1) if entry > 0 else 0, -0.5, 0.5) / 0.5; idx += 1
    peak = ps.get("peak_value", pv)
    obs[idx] = _clamp((pv / peak - 1) if peak > 0 else 0, -0.5, 0); idx += 1
    obs[idx] = _clamp(ps.get("days_held", 0) / 60, 0, 1); idx += 1

    return obs


# ── Lightweight local compute helpers (fallback when metrics absent) ─

def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if np.isfinite(val) else 50.0


def _compute_bb_pband(close: pd.Series, period: int = 20) -> float:
    if len(close) < period:
        return 0.5
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    rng = float(upper.iloc[-1] - lower.iloc[-1])
    if rng <= 0:
        return 0.5
    return _clamp((float(close.iloc[-1]) - float(lower.iloc[-1])) / rng, 0, 1)


def _pct_return(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def _volatility(close: pd.Series, window: int = 20) -> float:
    if len(close) < window:
        return 0.0
    rets = close.pct_change().dropna()
    if len(rets) < window:
        return 0.0
    return float(rets.rolling(window).std().iloc[-1] * np.sqrt(252))


def _bb_pband(metrics, price: float) -> float:
    bu = getattr(metrics, "bollinger_upper", None)
    bl = getattr(metrics, "bollinger_lower", None)
    if bu and bl and (bu - bl) > 0:
        return _clamp((price - bl) / (bu - bl), 0, 1)
    return 0.5


# ── Numeric helpers ─────────────────────────────────────────────────

def _norm(val, lo, hi):
    if val is None:
        return 0.0
    return _clamp((float(val) - lo) / (hi - lo) if hi > lo else 0.0, 0, 1)


def _shift(val, offset):
    if val is None:
        return None
    return float(val) + offset


def _clamp(v, lo=-1.0, hi=1.0):
    if v is None:
        return 0.0
    return max(lo, min(hi, float(v)))
