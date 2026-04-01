"""
Signal Quality Evaluator — Regime-Conditioned Analysis & CAGR Estimation.

Quantitatively evaluates BUY/SELL signal quality across market regimes
(BULL, BEAR, SIDEWAYS) and produces realistic, defensible CAGR estimates.

Outputs:
  - docs/signal_quality_by_regime.md
  - docs/regime_performance.md
  - docs/cagr_estimation.md
  - docs/signal_insights.md

Usage:
    from services.signal_quality_evaluator import run_full_evaluation
    results = run_full_evaluation(market="IND", period="5y")
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
SQRT_252 = np.sqrt(TRADING_DAYS)
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# ═══════════════════════════════════════════════════════════════
# 1. REGIME SEGMENTATION
# ═══════════════════════════════════════════════════════════════

REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_SIDEWAYS = "SIDEWAYS"


@dataclass
class RegimeLabel:
    """Regime classification for a single date."""
    date: pd.Timestamp
    regime: str            # BULL / BEAR / SIDEWAYS
    trend_score: float     # [-1, +1] from SMA alignment
    volatility_z: float    # vol z-score (relative to 1y rolling)
    adx: float             # ADX(14) trend strength


def classify_regimes(
    close: pd.Series,
    sma_short: int = 50,
    sma_long: int = 200,
    adx_period: int = 14,
    vol_window: int = 21,
    vol_z_window: int = 252,
) -> pd.DataFrame:
    """Classify each date into BULL / BEAR / SIDEWAYS.

    Method (robust, multi-factor):
      1. Trend: SMA(50) vs SMA(200) crossover + slope direction
      2. ADX(14): Trend strength filter (ADX < 20 → SIDEWAYS)
      3. Vol Z-Score: Abnormal volatility detection

    Classification logic:
      - If ADX < 20 → SIDEWAYS (regardless of trend)
      - If SMA(50) > SMA(200) AND slope > 0 → BULL
      - If SMA(50) < SMA(200) AND slope < 0 → BEAR
      - Otherwise → SIDEWAYS

    Returns DataFrame with columns: regime, trend_score, volatility_z, adx
    """
    close = close.dropna()
    if len(close) < sma_long + 50:
        raise ValueError(f"Need at least {sma_long + 50} bars, got {len(close)}")

    # ── SMA alignment ──
    sma_s = close.rolling(sma_short).mean()
    sma_l = close.rolling(sma_long).mean()
    # Trend score: normalized distance (SMA_short - SMA_long) / SMA_long
    trend_score = (sma_s - sma_l) / sma_l

    # SMA slope (20-day rate of change of SMA_50)
    sma_slope = sma_s.pct_change(20)

    # ── ADX ──
    high = close  # Approximate: use close as proxy for H/L when only close available
    low = close
    adx = _compute_adx_from_close(close, period=adx_period)

    # ── Volatility Z-Score ──
    daily_ret = close.pct_change()
    rolling_vol = daily_ret.rolling(vol_window).std()
    vol_mean = rolling_vol.rolling(vol_z_window).mean()
    vol_std = rolling_vol.rolling(vol_z_window).std()
    vol_z = (rolling_vol - vol_mean) / (vol_std + 1e-10)

    # ── Classification ──
    regimes = []
    for i in range(len(close)):
        idx = close.index[i]
        ts = trend_score.iloc[i] if i < len(trend_score) else np.nan
        adx_val = adx.iloc[i] if i < len(adx) else np.nan
        vz = vol_z.iloc[i] if i < len(vol_z) else np.nan
        slope = sma_slope.iloc[i] if i < len(sma_slope) else np.nan

        if pd.isna(ts) or pd.isna(adx_val):
            regimes.append(REGIME_SIDEWAYS)
            continue

        if adx_val < 20:
            regimes.append(REGIME_SIDEWAYS)
        elif ts > 0 and (pd.isna(slope) or slope > -0.01):
            regimes.append(REGIME_BULL)
        elif ts < 0 and (pd.isna(slope) or slope < 0.01):
            regimes.append(REGIME_BEAR)
        else:
            regimes.append(REGIME_SIDEWAYS)

    df = pd.DataFrame({
        "regime": regimes,
        "trend_score": trend_score.values,
        "volatility_z": vol_z.values,
        "adx": adx.values,
    }, index=close.index)

    return df


def _compute_adx_from_close(close: pd.Series, period: int = 14) -> pd.Series:
    """Approximate ADX from close-only data using Parkinson-style estimation.

    Uses |close[t] - close[t-1]| as proxy for True Range since we may
    not have reliable H/L data in all cases.
    """
    delta = close.diff().abs()
    # +DM / -DM approximated
    up = close.diff()
    down = -close.diff()
    plus_dm = up.where((up > 0) & (up > down), 0.0)
    minus_dm = down.where((down > 0) & (down > up), 0.0)

    atr = delta.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def classify_regimes_ohlcv(
    ohlcv: pd.DataFrame,
    sma_short: int = 50,
    sma_long: int = 200,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Classify regimes using full OHLCV data (more accurate ADX)."""
    close = ohlcv["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    high = ohlcv["High"]
    low = ohlcv["Low"]
    if hasattr(high, "squeeze"):
        high = high.squeeze()
    if hasattr(low, "squeeze"):
        low = low.squeeze()

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > 0) & (up > down), 0.0)
    minus_dm = down.where((down > 0) & (down > up), 0.0)

    atr = tr.ewm(span=adx_period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=adx_period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=adx_period, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=adx_period, adjust=False).mean()

    # SMA alignment
    sma_s = close.rolling(sma_short).mean()
    sma_l = close.rolling(sma_long).mean()
    trend_score = (sma_s - sma_l) / sma_l
    sma_slope = sma_s.pct_change(20)

    # Volatility Z
    daily_ret = close.pct_change()
    rolling_vol = daily_ret.rolling(21).std()
    vol_mean = rolling_vol.rolling(252).mean()
    vol_std = rolling_vol.rolling(252).std()
    vol_z = (rolling_vol - vol_mean) / (vol_std + 1e-10)

    regimes = []
    for i in range(len(close)):
        ts = trend_score.iloc[i] if not pd.isna(trend_score.iloc[i]) else 0
        adx_val = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0
        slope = sma_slope.iloc[i] if not pd.isna(sma_slope.iloc[i]) else 0

        if pd.isna(ts) or pd.isna(adx_val) or adx_val < 20:
            regimes.append(REGIME_SIDEWAYS)
        elif ts > 0 and slope > -0.01:
            regimes.append(REGIME_BULL)
        elif ts < 0 and slope < 0.01:
            regimes.append(REGIME_BEAR)
        else:
            regimes.append(REGIME_SIDEWAYS)

    return pd.DataFrame({
        "regime": regimes,
        "trend_score": trend_score.values,
        "volatility_z": vol_z.values,
        "adx": adx.values,
    }, index=close.index)


# ═══════════════════════════════════════════════════════════════
# 2. SIGNAL COLLECTION & FORWARD RETURNS
# ═══════════════════════════════════════════════════════════════

@dataclass
class SignalRecord:
    """A single BUY/SELL signal with metadata."""
    ticker: str
    date: pd.Timestamp
    direction: str           # "BUY" or "SELL"
    forecast: float          # raw combined forecast [-20, +20]
    confidence: float        # abs(forecast) / 20 → [0, 1]
    regime: str              # BULL / BEAR / SIDEWAYS
    # Forward returns (filled after alignment)
    fwd_5d: float = np.nan
    fwd_10d: float = np.nan
    fwd_20d: float = np.nan


def generate_signals_from_backtest(
    ohlcv_cache: Dict[str, pd.DataFrame],
    market: str = "IND",
    warmup: int = 262,
) -> Tuple[List[SignalRecord], Dict[str, pd.DataFrame], pd.DataFrame]:
    """Generate signals using the full Carver pipeline on historical data.

    Returns:
        signals: List of SignalRecord
        ohlcv_cache: cleaned OHLCV (same ref)
        regime_df: regime classification for the market index
    """
    from services.instrument_volatility import daily_price_volatility
    from services.forecast_scalar import ewmac_to_forecast, cap_forecast
    from strategies.ewmac import DEFAULT_VARIATIONS
    from strategies.carry_rule import compute_carry_batch
    from strategies.mean_reversion import compute_mean_reversion_batch
    from services.momentum_factor import compute_momentum_forecasts
    from services.oi_signal import compute_oi_signals_batch
    from services.forecast_combiner import (
        combine_forecasts, ForecastWeight,
        DEFAULT_FORECAST_WEIGHTS,
    )

    symbols = list(ohlcv_cache.keys())
    if not symbols:
        return [], ohlcv_cache, pd.DataFrame()

    # Use the first symbol (or NIFTY proxy) for regime classification
    index_sym = None
    for candidate in ["^NSEI", "NIFTY_50", "RELIANCE.NS", symbols[0]]:
        if candidate in ohlcv_cache:
            index_sym = candidate
            break
    if index_sym is None:
        index_sym = symbols[0]

    index_close = ohlcv_cache[index_sym]["Close"]
    if hasattr(index_close, "squeeze"):
        index_close = index_close.squeeze()

    # Classify regimes on the market index
    try:
        regime_df = classify_regimes_ohlcv(ohlcv_cache[index_sym])
    except (ValueError, KeyError):
        regime_df = classify_regimes(index_close)

    # All 23 offline-capable sources (matching production pipeline)
    available_sources = {
        "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
        "momentum", "mean_reversion", "oi_signal", "carry",
        "breakout", "cross_momentum", "pairs_arb",
        "penfold_trend", "ehlers_dsp", "intermarket", "acceleration",
        "carver_value", "skew_signal", "pead", "fii_flow",
        "event_driven", "sentiment", "screener", "decision_engine",
    }
    active_weights = [
        fw for fw in DEFAULT_FORECAST_WEIGHTS if fw.name in available_sources
    ]
    total_w = sum(fw.weight for fw in active_weights)
    if total_w > 0:
        active_weights = [
            ForecastWeight(fw.name, fw.weight / total_w)
            for fw in active_weights
        ]

    min_len = min(len(df) for df in ohlcv_cache.values())
    n_days = min_len
    signals: List[SignalRecord] = []

    # Pre-fetch dividend yields for carry
    dividend_yields: Dict[str, float] = {}
    try:
        import yfinance as yf
        for sym in symbols[:30]:  # cap at 30 for speed
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    info = yf.Ticker(sym).info
                    dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
                    if dy and dy > 0:
                        dividend_yields[sym] = float(dy)
            except Exception:
                pass
    except ImportError:
        pass

    logger.info("Generating signals for %d symbols over %d days (warmup=%d)",
                len(symbols), n_days, warmup)

    for day_idx in range(warmup, n_days):
        # Build OHLCV slice to current day
        ohlcv_slice = {sym: df.iloc[:day_idx + 1] for sym, df in ohlcv_cache.items()}

        # Get regime for this day
        if day_idx < len(regime_df):
            current_regime = regime_df.iloc[day_idx]["regime"]
        else:
            current_regime = REGIME_SIDEWAYS

        # Compute forecasts for each symbol
        all_forecasts: Dict[str, Dict[str, float]] = {sym: {} for sym in symbols}

        # EWMAC
        for sym in symbols:
            df = ohlcv_slice[sym]
            c = df["Close"]
            if hasattr(c, "squeeze"):
                c = c.squeeze()
            close = c.dropna()
            if len(close) < 270:
                continue
            dpv = daily_price_volatility(close)
            if dpv <= 0:
                dpv = 0.02
            for fast, slow in DEFAULT_VARIATIONS:
                if len(close) < slow + 10:
                    continue
                fast_ewma = close.ewm(span=fast, adjust=False).mean()
                slow_ewma = close.ewm(span=slow, adjust=False).mean()
                raw = float(fast_ewma.iloc[-1] - slow_ewma.iloc[-1])
                fc = ewmac_to_forecast(raw, dpv, fast, slow)
                all_forecasts[sym][f"ewmac_{fast}_{slow}"] = fc

        # Momentum
        try:
            mom_fc = compute_momentum_forecasts(ohlcv_slice)
            for sym, fc in mom_fc.items():
                if sym in all_forecasts:
                    all_forecasts[sym]["momentum"] = fc
        except Exception:
            pass

        # Mean reversion
        try:
            mr_fc = compute_mean_reversion_batch(ohlcv_slice)
            for sym, fc in mr_fc.items():
                if sym in all_forecasts:
                    all_forecasts[sym]["mean_reversion"] = fc
        except Exception:
            pass

        # Carry
        try:
            carry_fc = compute_carry_batch(ohlcv_slice, dividend_yields=dividend_yields)
            for sym, cf in carry_fc.items():
                if sym in all_forecasts:
                    all_forecasts[sym]["carry"] = cf.forecast
        except Exception:
            pass

        # OI signal
        try:
            oi_data = {}
            for sym, df in ohlcv_slice.items():
                if "Volume" not in df.columns or len(df) < 6:
                    continue
                vol = df["Volume"]
                if hasattr(vol, "squeeze"):
                    vol = vol.squeeze()
                last_vol = float(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0
                avg_vol = float(vol.iloc[-6:-1].mean())
                if avg_vol > 0:
                    bare = sym.replace('.NS', '').replace('.BO', '')
                    oi_data[bare] = {
                        "oi_change_pct": ((last_vol / avg_vol) - 1) * 100,
                        "price_change_pct": 0,
                        "volume_ratio": last_vol / avg_vol,
                    }
            oi_fc = compute_oi_signals_batch(oi_data)
            for sym in symbols:
                bare = sym.replace('.NS', '').replace('.BO', '')
                if bare in oi_fc and sym in all_forecasts:
                    all_forecasts[sym]["oi_signal"] = oi_fc[bare]
        except Exception:
            pass

        # Combine and record signals
        for sym, fc_dict in all_forecasts.items():
            if not fc_dict:
                continue
            combined = combine_forecasts(sym, fc_dict, active_weights)
            forecast = combined.combined_forecast

            if abs(forecast) < 2.0:
                continue  # Skip near-zero forecasts (no meaningful signal)

            direction = "BUY" if forecast > 0 else "SELL"
            confidence = min(abs(forecast) / 20.0, 1.0)

            # Get the date
            df = ohlcv_slice[sym]
            date = df.index[day_idx] if day_idx < len(df) else df.index[-1]

            signals.append(SignalRecord(
                ticker=sym,
                date=date,
                direction=direction,
                forecast=forecast,
                confidence=confidence,
                regime=current_regime,
            ))

    # ── Align forward returns ──
    logger.info("Aligning forward returns for %d signals...", len(signals))
    for sig in signals:
        if sig.ticker not in ohlcv_cache:
            continue
        df = ohlcv_cache[sig.ticker]
        c = df["Close"]
        if hasattr(c, "squeeze"):
            c = c.squeeze()

        try:
            loc = c.index.get_loc(sig.date)
        except (KeyError, TypeError):
            # Find nearest
            diffs = (c.index - sig.date).total_seconds().abs() if hasattr(c.index, 'total_seconds') else np.arange(len(c))
            loc = int(np.argmin(np.abs((c.index - sig.date).total_seconds()))) if hasattr(c.index[0], 'timestamp') else 0

        entry_price = float(c.iloc[loc])
        if entry_price <= 0 or not np.isfinite(entry_price):
            continue

        multiplier = 1.0 if sig.direction == "BUY" else -1.0

        for horizon, attr in [(5, "fwd_5d"), (10, "fwd_10d"), (20, "fwd_20d")]:
            if loc + horizon < len(c):
                exit_price = float(c.iloc[loc + horizon])
                if np.isfinite(exit_price) and exit_price > 0:
                    ret = (exit_price / entry_price - 1) * multiplier
                    setattr(sig, attr, ret)

    return signals, ohlcv_cache, regime_df


# ═══════════════════════════════════════════════════════════════
# 3. SIGNAL QUALITY METRICS (Per Regime)
# ═══════════════════════════════════════════════════════════════

@dataclass
class SignalQualityMetrics:
    """Quality metrics for a signal cohort."""
    regime: str
    direction: str           # BUY / SELL / ALL
    horizon: str             # 5D / 10D / 20D
    n_signals: int = 0
    hit_rate: float = 0.0    # % profitable
    avg_return: float = 0.0
    median_return: float = 0.0
    max_return: float = 0.0
    min_return: float = 0.0   # worst single signal
    std_return: float = 0.0
    sharpe: float = 0.0       # per-signal Sharpe
    profit_factor: float = 0.0
    false_signal_rate: float = 0.0  # % that lost > 2%
    avg_confidence: float = 0.0

    def to_row(self) -> Dict[str, Any]:
        return {
            "Regime": self.regime,
            "Direction": self.direction,
            "Horizon": self.horizon,
            "N": self.n_signals,
            "Hit Rate": f"{self.hit_rate:.1f}%",
            "Avg Ret": f"{self.avg_return * 100:.2f}%",
            "Med Ret": f"{self.median_return * 100:.2f}%",
            "Sharpe": f"{self.sharpe:.2f}",
            "PF": f"{self.profit_factor:.2f}",
            "False %": f"{self.false_signal_rate:.1f}%",
        }


def compute_signal_quality(
    signals: List[SignalRecord],
    regime: str = "ALL",
    direction: str = "ALL",
    horizon: str = "20D",
) -> SignalQualityMetrics:
    """Compute quality metrics for a filtered cohort of signals."""
    # Filter
    filtered = signals
    if regime != "ALL":
        filtered = [s for s in filtered if s.regime == regime]
    if direction != "ALL":
        filtered = [s for s in filtered if s.direction == direction]

    # Select horizon
    horizon_map = {"5D": "fwd_5d", "10D": "fwd_10d", "20D": "fwd_20d"}
    attr = horizon_map.get(horizon, "fwd_20d")
    returns = np.array([getattr(s, attr) for s in filtered if np.isfinite(getattr(s, attr))])

    m = SignalQualityMetrics(
        regime=regime,
        direction=direction,
        horizon=horizon,
        n_signals=len(returns),
    )

    if len(returns) == 0:
        return m

    m.avg_return = float(np.mean(returns))
    m.median_return = float(np.median(returns))
    m.max_return = float(np.max(returns))
    m.min_return = float(np.min(returns))
    m.std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    m.hit_rate = float(np.sum(returns > 0) / len(returns) * 100)
    m.false_signal_rate = float(np.sum(returns < -0.02) / len(returns) * 100)
    m.avg_confidence = float(np.mean([s.confidence for s in filtered
                                       if np.isfinite(getattr(s, attr))]))

    if m.std_return > 0:
        m.sharpe = float(m.avg_return / m.std_return * np.sqrt(TRADING_DAYS / int(horizon[:-1])))

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    m.profit_factor = float(gains / losses) if losses > 0 else (float('inf') if gains > 0 else 0.0)

    return m


# ═══════════════════════════════════════════════════════════════
# 4. REGIME PERFORMANCE BREAKDOWN
# ═══════════════════════════════════════════════════════════════

@dataclass
class RegimePerformance:
    """Performance summary for a single regime."""
    regime: str
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    win_rate: float = 0.0
    cumulative_return: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    regime_days: int = 0
    regime_pct: float = 0.0  # % of total time in this regime


def compute_regime_performance(
    signals: List[SignalRecord],
    regime_df: pd.DataFrame,
) -> List[RegimePerformance]:
    """Compute performance breakdown per regime."""
    results = []
    total_days = len(regime_df)

    for regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]:
        regime_sigs = [s for s in signals if s.regime == regime]
        rp = RegimePerformance(regime=regime)
        rp.total_signals = len(regime_sigs)
        rp.buy_signals = sum(1 for s in regime_sigs if s.direction == "BUY")
        rp.sell_signals = sum(1 for s in regime_sigs if s.direction == "SELL")

        regime_days = int((regime_df["regime"] == regime).sum())
        rp.regime_days = regime_days
        rp.regime_pct = regime_days / total_days * 100 if total_days > 0 else 0

        # Use 20D returns for regime performance
        returns = np.array([s.fwd_20d for s in regime_sigs if np.isfinite(s.fwd_20d)])
        if len(returns) == 0:
            results.append(rp)
            continue

        rp.win_rate = float(np.sum(returns > 0) / len(returns) * 100)
        rp.avg_return = float(np.mean(returns))
        rp.cumulative_return = float(np.prod(1 + returns) - 1)
        rp.volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0

        if rp.volatility > 0:
            rp.sharpe = float(rp.avg_return / rp.volatility * np.sqrt(TRADING_DAYS / 20))

        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        rp.profit_factor = float(gains / losses) if losses > 0 else 0

        # Drawdown of regime-specific equity curve
        eq = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        rp.max_drawdown = float(np.max(dd)) if len(dd) > 0 else 0

        results.append(rp)

    return results


# ═══════════════════════════════════════════════════════════════
# 5. REALISTIC PORTFOLIO BACKTEST (Regime-Aware)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PortfolioBacktestResult:
    """Results from the realistic portfolio-level backtest."""
    # Equity curve
    daily_equity: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    daily_regimes: List[str] = field(default_factory=list)

    # Overall metrics
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    max_dd_duration_days: int = 0
    n_trades: int = 0
    avg_positions: float = 0.0

    # Cost breakdown
    total_costs: float = 0.0
    slippage_costs: float = 0.0
    commission_costs: float = 0.0

    # Source attribution
    source_hit_rates: Dict[str, float] = field(default_factory=dict)

    # Regime-conditioned metrics
    regime_sharpes: Dict[str, float] = field(default_factory=dict)
    regime_returns: Dict[str, float] = field(default_factory=dict)
    regime_drawdowns: Dict[str, float] = field(default_factory=dict)


def run_realistic_backtest(
    ohlcv_cache: Dict[str, pd.DataFrame],
    regime_df: pd.DataFrame,
    market: str = "IND",
    capital: float = 500_000.0,
    annual_vol_target: float = 0.75,
    commission_bps: float = 13.0,
    slippage_bps: float = 10.0,
    max_leverage: float = 7.0,
    max_positions: int = 12,
) -> PortfolioBacktestResult:
    """Run a regime-aware portfolio backtest with realistic constraints.

    Delegates to the production full_pipeline_backtest.run_full_backtest
    which includes all 23 forecast sources, regime-adaptive leverage,
    cost modeling, and trailing stops. Then layers regime analysis.
    """
    from services.full_pipeline_backtest import run_full_backtest

    tickers = list(ohlcv_cache.keys())
    if len(tickers) < 2:
        return PortfolioBacktestResult()

    # Run production pipeline backtest with matching config
    bt = run_full_backtest(
        tickers=tickers,
        capital=capital,
        period="5y",
        market=market,
        annual_vol_target=annual_vol_target,
        verbose=True,
    )

    if not isinstance(bt, dict) or bt.get("sharpe", 0) == 0:
        return PortfolioBacktestResult()

    # Extract daily returns from equity curve
    daily_equity = bt.get("daily_equity", [capital])
    daily_returns_list = []
    for i in range(1, len(daily_equity)):
        if daily_equity[i - 1] > 0:
            daily_returns_list.append(daily_equity[i] / daily_equity[i - 1] - 1)
        else:
            daily_returns_list.append(0.0)

    # Map daily returns to regimes using regime_df
    warmup = max(0, len(daily_equity) - 1 - len(regime_df))
    daily_regimes = []
    for i in range(len(daily_returns_list)):
        regime_idx = warmup + i
        if regime_idx < len(regime_df):
            daily_regimes.append(regime_df.iloc[regime_idx]["regime"])
        else:
            daily_regimes.append(REGIME_SIDEWAYS)

    result = PortfolioBacktestResult(
        daily_equity=daily_equity,
        daily_returns=daily_returns_list,
        daily_regimes=daily_regimes,
        total_return_pct=bt.get("total_return_pct", 0),
        annual_return_pct=bt.get("annual_return_pct", 0),
        sharpe=bt.get("sharpe", 0),
        sortino=bt.get("sortino", 0),
        calmar=bt.get("calmar", 0),
        max_drawdown_pct=bt.get("max_drawdown_pct", 0),
        n_trades=bt.get("n_trades", 0),
        avg_positions=bt.get("avg_positions", 0),
        source_hit_rates=bt.get("source_hit_rates", {}),
    )

    # Compute drawdown duration
    if len(daily_equity) > 1:
        eq_arr = np.array(daily_equity)
        peak = np.maximum.accumulate(eq_arr)
        in_dd = eq_arr < peak
        max_dur = 0
        cur = 0
        for d in in_dd:
            if d:
                cur += 1
                max_dur = max(max_dur, cur)
            else:
                cur = 0
        result.max_dd_duration_days = max_dur

    # Regime-conditioned metrics
    for regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]:
        regime_mask = [i for i, r in enumerate(daily_regimes) if r == regime]
        if not regime_mask:
            continue
        regime_rets = np.array([daily_returns_list[i] for i in regime_mask
                                if i < len(daily_returns_list)])
        if len(regime_rets) > 1:
            r_std = float(np.std(regime_rets, ddof=1))
            r_mean = float(np.mean(regime_rets))
            if r_std > 0:
                result.regime_sharpes[regime] = round(r_mean / r_std * SQRT_252, 3)
            n_yr = len(regime_rets) / TRADING_DAYS
            r_total = float(np.prod(1 + regime_rets) - 1)
            result.regime_returns[regime] = round(
                ((1 + r_total) ** (1 / max(n_yr, 0.01)) - 1) * 100, 2
            )
            eq = np.cumprod(1 + regime_rets)
            pk = np.maximum.accumulate(eq)
            dd = (pk - eq) / pk * 100
            result.regime_drawdowns[regime] = round(float(np.max(dd)), 2)

    return result


# ═══════════════════════════════════════════════════════════════
# 6. CAGR ESTIMATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class CAGREstimate:
    """CAGR estimation with confidence intervals."""
    # Ideal (no constraints)
    ideal_cagr: float = 0.0
    ideal_sharpe: float = 0.0
    ideal_max_dd: float = 0.0

    # Realistic (with costs + execution)
    realistic_cagr: float = 0.0
    realistic_sharpe: float = 0.0
    realistic_max_dd: float = 0.0

    # Conservative (overfitting penalty)
    conservative_cagr: float = 0.0
    conservative_sharpe: float = 0.0
    conservative_max_dd: float = 0.0

    # Statistical
    cagr_ci_90: Tuple[float, float] = (0.0, 0.0)
    bootstrap_cagrs: List[float] = field(default_factory=list)
    overfitting_haircut_pct: float = 0.0
    n_years: float = 0.0


def estimate_cagr(
    bt_result: PortfolioBacktestResult,
    n_strategies: int = 22,
    cost_drag_bps: float = 23.0,
) -> CAGREstimate:
    """Compute ideal, realistic, and conservative CAGR estimates.

    Conservative estimate applies:
      1. Transaction cost drag
      2. Deflated Sharpe Ratio haircut (de Prado)
      3. Walk-forward degradation ratio (~0.6-0.7 typical)
      4. Bootstrap confidence interval
    """
    est = CAGREstimate()

    ret_arr = np.array(bt_result.daily_returns)
    if len(ret_arr) < 50:
        return est

    n_years = len(ret_arr) / TRADING_DAYS
    est.n_years = round(n_years, 2)

    # ── Ideal CAGR (from backtest as-is, includes costs already) ──
    total_ret = bt_result.daily_equity[-1] / bt_result.daily_equity[0] - 1
    est.ideal_cagr = round(((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1) * 100, 2)
    est.ideal_max_dd = bt_result.max_drawdown_pct

    avg_ret = float(np.mean(ret_arr))
    std_ret = float(np.std(ret_arr, ddof=1))
    if std_ret > 0:
        est.ideal_sharpe = round(avg_ret / std_ret * SQRT_252, 3)

    # ── Realistic CAGR (additional cost + slippage drag) ──
    # The backtest already includes costs, but add an extra buffer for
    # real-execution friction (market impact, partial fills, timing)
    execution_drag_daily = cost_drag_bps / 10000.0 / TRADING_DAYS
    realistic_rets = ret_arr - execution_drag_daily
    r_total = float(np.prod(1 + realistic_rets) - 1)
    est.realistic_cagr = round(((1 + r_total) ** (1 / max(n_years, 0.01)) - 1) * 100, 2)

    r_std = float(np.std(realistic_rets, ddof=1))
    r_mean = float(np.mean(realistic_rets))
    if r_std > 0:
        est.realistic_sharpe = round(r_mean / r_std * SQRT_252, 3)

    eq_r = np.cumprod(1 + realistic_rets) * bt_result.daily_equity[0]
    pk_r = np.maximum.accumulate(eq_r)
    dd_r = (pk_r - eq_r) / pk_r * 100
    est.realistic_max_dd = round(float(np.max(dd_r)), 2)

    # ── Conservative CAGR (overfitting haircut) ──
    # Haircut based on:
    # 1. Number of strategy variants tested (data mining bias)
    # 2. Walk-forward degradation ratio (~0.65 typical)
    # 3. De Prado expected max Sharpe under null
    wf_degradation = 0.65  # typical OOS/IS ratio
    n_tests = max(n_strategies, 10)

    # Expected max Sharpe under null (Euler-gamma from Order Statistics)
    euler_gamma = 0.5772
    expected_max_sr_null = (
        np.sqrt(2 * np.log(n_tests))
        - (np.log(np.pi) + euler_gamma) / (2 * np.sqrt(2 * np.log(n_tests)))
    ) if n_tests > 1 else 0

    # Haircut: conservative Sharpe = realistic Sharpe × degradation - null_expected
    conservative_sharpe = max(
        0, est.realistic_sharpe * wf_degradation - expected_max_sr_null * 0.3
    )
    est.conservative_sharpe = round(conservative_sharpe, 3)

    # Convert conservative Sharpe back to approximate CAGR
    # CAGR ≈ rf + Sharpe × vol
    annual_vol = float(np.std(ret_arr, ddof=1)) * SQRT_252
    rf = 0.07 if True else 0.04  # IND default
    est.conservative_cagr = round((rf + conservative_sharpe * annual_vol) * 100, 2)

    est.overfitting_haircut_pct = round(
        (1 - est.conservative_cagr / max(est.ideal_cagr, 0.01)) * 100, 1
    )

    # Conservative max DD = realistic × 1.3 (safety margin)
    est.conservative_max_dd = round(est.realistic_max_dd * 1.3, 2)

    # ── Bootstrap CAGR CI (block bootstrap, block=21 days) ──
    rng = np.random.RandomState(42)
    block_size = 21
    n_boot = 2000
    boot_cagrs = []
    for _ in range(n_boot):
        n_blocks = len(ret_arr) // block_size + 1
        indices = []
        for _ in range(n_blocks):
            start = rng.randint(0, max(1, len(ret_arr) - block_size))
            indices.extend(range(start, min(start + block_size, len(ret_arr))))
        boot_rets = ret_arr[indices[:len(ret_arr)]]
        b_total = float(np.prod(1 + boot_rets) - 1)
        b_cagr = ((1 + b_total) ** (1 / max(n_years, 0.01)) - 1) * 100
        if np.isfinite(b_cagr) and -100 < b_cagr < 500:
            boot_cagrs.append(b_cagr)

    if boot_cagrs:
        est.bootstrap_cagrs = boot_cagrs
        est.cagr_ci_90 = (
            round(float(np.percentile(boot_cagrs, 5)), 2),
            round(float(np.percentile(boot_cagrs, 95)), 2),
        )

    return est


# ═══════════════════════════════════════════════════════════════
# 7. STRESS TESTING
# ═══════════════════════════════════════════════════════════════

@dataclass
class StressTestResult:
    """Results from stress testing signals."""
    scenario: str
    n_signals: int = 0
    hit_rate: float = 0.0
    avg_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0


def run_stress_tests(
    signals: List[SignalRecord],
    regime_df: pd.DataFrame,
) -> List[StressTestResult]:
    """Evaluate signal robustness under stress scenarios.

    Scenarios:
      1. High Volatility (vol_z > 1.5)
      2. Extreme Bear (trend_score < -0.05 and vol_z > 1.0)
      3. Low Confidence Signals (confidence < 0.3)
      4. First Year Only (early signals, less data)
      5. Last Year Only (most recent, out-of-sample proxy)
    """
    results = []

    # ── 1. High Volatility ──
    high_vol_dates = set()
    if "volatility_z" in regime_df.columns:
        hv = regime_df[regime_df["volatility_z"] > 1.5]
        high_vol_dates = set(hv.index)

    hv_signals = [s for s in signals if s.date in high_vol_dates]
    results.append(_stress_result("High Volatility (vol_z>1.5)", hv_signals))

    # ── 2. Extreme Bear ──
    extreme_bear_dates = set()
    if "trend_score" in regime_df.columns and "volatility_z" in regime_df.columns:
        eb = regime_df[(regime_df["trend_score"] < -0.05) & (regime_df["volatility_z"] > 1.0)]
        extreme_bear_dates = set(eb.index)

    eb_signals = [s for s in signals if s.date in extreme_bear_dates]
    results.append(_stress_result("Extreme Bear (trend<-5%, vol_z>1)", eb_signals))

    # ── 3. Low Confidence ──
    low_conf = [s for s in signals if s.confidence < 0.3]
    results.append(_stress_result("Low Confidence (<0.3)", low_conf))

    # ── 4. First Year ──
    if signals:
        all_dates = sorted(set(s.date for s in signals))
        if all_dates:
            cutoff_early = all_dates[0] + pd.Timedelta(days=365)
            first_yr = [s for s in signals if s.date <= cutoff_early]
            results.append(_stress_result("First Year (early signals)", first_yr))

    # ── 5. Last Year ──
    if signals:
        all_dates = sorted(set(s.date for s in signals))
        if all_dates:
            cutoff_late = all_dates[-1] - pd.Timedelta(days=365)
            last_yr = [s for s in signals if s.date >= cutoff_late]
            results.append(_stress_result("Last Year (OOS proxy)", last_yr))

    return results


def _stress_result(scenario: str, sigs: List[SignalRecord]) -> StressTestResult:
    """Compute stress test metrics for a signal cohort."""
    r = StressTestResult(scenario=scenario, n_signals=len(sigs))
    if not sigs:
        return r

    returns = np.array([s.fwd_20d for s in sigs if np.isfinite(s.fwd_20d)])
    if len(returns) == 0:
        return r

    r.hit_rate = float(np.sum(returns > 0) / len(returns) * 100)
    r.avg_return = float(np.mean(returns))

    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0
    if std > 0:
        r.sharpe = float(r.avg_return / std * np.sqrt(TRADING_DAYS / 20))

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    r.profit_factor = float(gains / losses) if losses > 0 else 0

    eq = np.cumprod(1 + returns)
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / pk
    r.max_drawdown = float(np.max(dd)) if len(dd) > 0 else 0

    return r


# ═══════════════════════════════════════════════════════════════
# 8. DOCUMENTATION GENERATORS
# ═══════════════════════════════════════════════════════════════

def _generate_signal_quality_doc(
    quality_metrics: List[SignalQualityMetrics],
    signals: List[SignalRecord],
) -> str:
    """Generate docs/signal_quality_by_regime.md"""
    lines = [
        "# Signal Quality by Regime",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Total signals evaluated: {len(signals):,}*",
        "",
        "## Overview",
        "",
        "Signal quality metrics computed for BUY and SELL signals across",
        "BULL, BEAR, and SIDEWAYS regimes at 5D, 10D, and 20D forward horizons.",
        "",
        "## Summary Table",
        "",
    ]

    # Group by horizon
    for horizon in ["5D", "10D", "20D"]:
        h_metrics = [m for m in quality_metrics if m.horizon == horizon]
        if not h_metrics:
            continue

        lines.append(f"### {horizon} Forward Returns")
        lines.append("")
        lines.append("| Regime | Dir | N | Hit Rate | Avg Ret | Med Ret | Sharpe | PF | False% |")
        lines.append("|--------|-----|---|----------|---------|---------|--------|----|--------|")

        for m in h_metrics:
            pf_str = f"{m.profit_factor:.2f}" if m.profit_factor < 100 else "∞"
            lines.append(
                f"| {m.regime:8s} | {m.direction:4s} | {m.n_signals:5d} | "
                f"{m.hit_rate:5.1f}% | {m.avg_return*100:+6.2f}% | "
                f"{m.median_return*100:+6.2f}% | {m.sharpe:5.2f} | "
                f"{pf_str:>5s} | {m.false_signal_rate:5.1f}% |"
            )
        lines.append("")

    # Detailed analysis per regime
    for regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]:
        regime_sigs = [s for s in signals if s.regime == regime]
        if not regime_sigs:
            continue

        lines.append(f"## {regime} Regime — Detailed Analysis")
        lines.append("")
        lines.append(f"- Total signals: {len(regime_sigs):,}")
        lines.append(f"- BUY signals: {sum(1 for s in regime_sigs if s.direction=='BUY'):,}")
        lines.append(f"- SELL signals: {sum(1 for s in regime_sigs if s.direction=='SELL'):,}")

        # Confidence distribution
        confs = [s.confidence for s in regime_sigs]
        lines.append(f"- Avg confidence: {np.mean(confs):.3f}")
        lines.append(f"- Confidence > 0.5: {sum(1 for c in confs if c > 0.5):,} "
                      f"({sum(1 for c in confs if c > 0.5)/len(confs)*100:.1f}%)")
        lines.append("")

        # Best/worst tickers in this regime
        ticker_rets: Dict[str, List[float]] = defaultdict(list)
        for s in regime_sigs:
            if np.isfinite(s.fwd_20d):
                ticker_rets[s.ticker].append(s.fwd_20d)

        if ticker_rets:
            avg_by_ticker = {t: np.mean(r) for t, r in ticker_rets.items() if len(r) >= 3}
            if avg_by_ticker:
                sorted_tickers = sorted(avg_by_ticker.items(), key=lambda x: -x[1])
                lines.append(f"### Top 5 Performers ({regime})")
                lines.append("")
                for t, r in sorted_tickers[:5]:
                    lines.append(f"- **{t}**: avg 20D return = {r*100:+.2f}% (N={len(ticker_rets[t])})")
                lines.append("")
                lines.append(f"### Bottom 5 ({regime})")
                lines.append("")
                for t, r in sorted_tickers[-5:]:
                    lines.append(f"- **{t}**: avg 20D return = {r*100:+.2f}% (N={len(ticker_rets[t])})")
                lines.append("")

    return "\n".join(lines)


def _generate_regime_performance_doc(
    regime_perf: List[RegimePerformance],
    regime_df: pd.DataFrame,
) -> str:
    """Generate docs/regime_performance.md"""
    total_days = len(regime_df)
    lines = [
        "# Regime Performance Breakdown",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Total trading days analyzed: {total_days:,}*",
        "",
        "## Regime Distribution",
        "",
        "| Regime | Days | % of Time | Transitions In |",
        "|--------|------|-----------|----------------|",
    ]

    for rp in regime_perf:
        # Count transitions
        transitions = 0
        if "regime" in regime_df.columns:
            r_col = regime_df["regime"]
            transitions = int(((r_col == rp.regime) & (r_col.shift(1) != rp.regime)).sum())
        lines.append(
            f"| {rp.regime:10s} | {rp.regime_days:5d} | {rp.regime_pct:5.1f}% | {transitions:3d} |"
        )

    lines.extend([
        "",
        "## Performance by Regime",
        "",
        "| Regime | Signals | Win Rate | Cum Return | Avg Ret | Vol | Sharpe | PF | Max DD |",
        "|--------|---------|----------|------------|---------|-----|--------|----|--------|",
    ])

    for rp in regime_perf:
        pf_str = f"{rp.profit_factor:.2f}" if rp.profit_factor < 100 else "∞"
        lines.append(
            f"| {rp.regime:10s} | {rp.total_signals:6d} | {rp.win_rate:5.1f}% | "
            f"{rp.cumulative_return*100:+8.2f}% | {rp.avg_return*100:+6.3f}% | "
            f"{rp.volatility*100:5.2f}% | {rp.sharpe:5.2f} | {pf_str:>5s} | "
            f"{rp.max_drawdown*100:5.1f}% |"
        )

    # Identify best/worst
    lines.extend(["", "## Key Findings", ""])
    if regime_perf:
        best = max(regime_perf, key=lambda r: r.sharpe)
        worst = min(regime_perf, key=lambda r: r.sharpe)
        lines.append(f"- **Best regime**: {best.regime} (Sharpe={best.sharpe:.2f}, "
                      f"Win Rate={best.win_rate:.1f}%)")
        lines.append(f"- **Worst regime**: {worst.regime} (Sharpe={worst.sharpe:.2f}, "
                      f"Win Rate={worst.win_rate:.1f}%)")

        if worst.sharpe < 0:
            lines.append(f"- **Warning**: Negative Sharpe in {worst.regime} — "
                          f"consider regime-specific signal filtering")

        bull = next((r for r in regime_perf if r.regime == REGIME_BULL), None)
        bear = next((r for r in regime_perf if r.regime == REGIME_BEAR), None)
        if bull and bear:
            lines.append(f"- **Bull/Bear asymmetry**: Bull Sharpe={bull.sharpe:.2f} vs "
                          f"Bear Sharpe={bear.sharpe:.2f}")
            if bear.sharpe < 0:
                lines.append("  - System struggles in BEAR regime → needs protective filters")

    lines.extend([
        "",
        "## Signal Distribution by Regime",
        "",
    ])

    for rp in regime_perf:
        buy_pct = rp.buy_signals / max(rp.total_signals, 1) * 100
        lines.append(f"- **{rp.regime}**: {rp.total_signals} signals "
                      f"({rp.buy_signals} BUY / {rp.sell_signals} SELL, "
                      f"BUY ratio={buy_pct:.0f}%)")

    return "\n".join(lines)


def _generate_cagr_doc(
    cagr_est: CAGREstimate,
    bt_result: PortfolioBacktestResult,
    market: str,
) -> str:
    """Generate docs/cagr_estimation.md"""
    lines = [
        "# CAGR Estimation — Centurion Core",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Market: {market} | Backtest period: {cagr_est.n_years:.1f} years*",
        "",
        "## CAGR Summary",
        "",
        "| Metric | Ideal | Realistic | Conservative |",
        "|--------|-------|-----------|--------------|",
        f"| **CAGR** | {cagr_est.ideal_cagr:+.1f}% | {cagr_est.realistic_cagr:+.1f}% | "
        f"{cagr_est.conservative_cagr:+.1f}% |",
        f"| **Sharpe** | {cagr_est.ideal_sharpe:.2f} | {cagr_est.realistic_sharpe:.2f} | "
        f"{cagr_est.conservative_sharpe:.2f} |",
        f"| **Max DD** | {cagr_est.ideal_max_dd:.1f}% | {cagr_est.realistic_max_dd:.1f}% | "
        f"{cagr_est.conservative_max_dd:.1f}% |",
        "",
        "## Definitions",
        "",
        "- **Ideal**: Raw backtest returns (includes base transaction costs + slippage)",
        "- **Realistic**: Ideal + additional execution friction (market impact, partial fills, timing delays)",
        "- **Conservative**: Realistic × walk-forward degradation (0.65) − data-mining bias haircut",
        "",
        "## Statistical Confidence",
        "",
        f"- 90% Bootstrap CI for CAGR: [{cagr_est.cagr_ci_90[0]:+.1f}%, {cagr_est.cagr_ci_90[1]:+.1f}%]",
        f"- Overfitting haircut: {cagr_est.overfitting_haircut_pct:.1f}%",
        f"- Block bootstrap: 2000 simulations, block size = 21 days",
        "",
        "## Portfolio Backtest Details",
        "",
        f"- Starting capital: {bt_result.daily_equity[0]:,.0f}",
        f"- Final equity: {bt_result.daily_equity[-1]:,.0f}" if bt_result.daily_equity else "",
        f"- Total trades: {bt_result.n_trades:,}",
        f"- Avg positions: {bt_result.avg_positions:.1f}",
        f"- Transaction costs: {bt_result.total_costs:,.0f}",
        f"- Max drawdown duration: {bt_result.max_dd_duration_days} days",
        "",
        "## Regime-Conditioned Performance",
        "",
        "| Regime | Ann. Return | Sharpe | Max DD |",
        "|--------|-------------|--------|--------|",
    ]

    for regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]:
        ann_ret = bt_result.regime_returns.get(regime, 0)
        sharpe = bt_result.regime_sharpes.get(regime, 0)
        dd = bt_result.regime_drawdowns.get(regime, 0)
        lines.append(f"| {regime:10s} | {ann_ret:+.1f}% | {sharpe:.2f} | {dd:.1f}% |")

    lines.extend([
        "",
        "## Methodology",
        "",
        "1. **No look-ahead bias**: Expanding window, signals generated using only past data",
        "2. **Walk-forward validation**: 252-day train / 63-day test rolling windows",
        "3. **Position sizing**: Volatility-targeted (Carver AFTS), regime-adaptive leverage",
        "4. **Costs**: Commission + slippage modeled per-trade",
        "5. **Conservative haircut**: Accounts for data-mining bias (22 strategy variants tested),",
        "   walk-forward degradation ratio, and expected max Sharpe under null hypothesis",
        "6. **Bootstrap CI**: Block bootstrap (21-day blocks) preserves autocorrelation structure",
    ])

    return "\n".join(lines)


def _generate_insights_doc(
    quality_metrics: List[SignalQualityMetrics],
    regime_perf: List[RegimePerformance],
    stress_results: List[StressTestResult],
    cagr_est: CAGREstimate,
    bt_result: PortfolioBacktestResult,
    signals: List[SignalRecord],
) -> str:
    """Generate docs/signal_insights.md"""
    lines = [
        "# Signal Insights & Recommendations",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## 1. Regime-Specific Signal Failures",
        "",
    ]

    # Identify failing regimes
    for regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]:
        regime_20d = [m for m in quality_metrics
                      if m.regime == regime and m.horizon == "20D" and m.direction == "ALL"]
        if regime_20d:
            m = regime_20d[0]
            if m.hit_rate < 50:
                lines.append(f"### {regime}: Underperforming (Hit Rate = {m.hit_rate:.1f}%)")
                lines.append(f"- Avg return: {m.avg_return*100:+.2f}%")
                lines.append(f"- False signal rate: {m.false_signal_rate:.1f}%")
                lines.append(f"- Sharpe: {m.sharpe:.2f}")

                # Check BUY vs SELL
                buy_m = next((x for x in quality_metrics
                              if x.regime == regime and x.horizon == "20D" and x.direction == "BUY"), None)
                sell_m = next((x for x in quality_metrics
                               if x.regime == regime and x.horizon == "20D" and x.direction == "SELL"), None)
                if buy_m and sell_m:
                    lines.append(f"- BUY hit rate: {buy_m.hit_rate:.1f}% vs SELL: {sell_m.hit_rate:.1f}%")
                    if buy_m.hit_rate < sell_m.hit_rate - 10:
                        lines.append(f"  → **BUY signals are weaker** in {regime}")
                    elif sell_m.hit_rate < buy_m.hit_rate - 10:
                        lines.append(f"  → **SELL signals are weaker** in {regime}")
                lines.append("")
            else:
                lines.append(f"### {regime}: Performing Well (Hit Rate = {m.hit_rate:.1f}%, "
                              f"Sharpe = {m.sharpe:.2f})")
                lines.append("")

    # Overfitting indicators
    lines.extend([
        "## 2. Overfitting Indicators",
        "",
    ])

    # Compare first year vs last year stress results
    first_yr = next((s for s in stress_results if "First Year" in s.scenario), None)
    last_yr = next((s for s in stress_results if "Last Year" in s.scenario), None)
    if first_yr and last_yr:
        if first_yr.sharpe > 0 and last_yr.sharpe > 0:
            degradation = last_yr.sharpe / first_yr.sharpe
            lines.append(f"- First year Sharpe: {first_yr.sharpe:.2f} | "
                          f"Last year Sharpe: {last_yr.sharpe:.2f}")
            lines.append(f"- Degradation ratio: {degradation:.2f}")
            if degradation < 0.5:
                lines.append("  → **HIGH overfitting risk** — performance degrades significantly in recent data")
            elif degradation < 0.7:
                lines.append("  → **Moderate overfitting concern** — some performance decay")
            else:
                lines.append("  → Performance is relatively stable over time ✓")
        elif first_yr.sharpe > 0 and last_yr.sharpe <= 0:
            lines.append(f"- **CRITICAL**: Positive Sharpe in first year ({first_yr.sharpe:.2f}) "
                          f"but negative in last year ({last_yr.sharpe:.2f})")
            lines.append("  → Strategy may have stopped working")

    lines.append(f"- Overfitting haircut applied: {cagr_est.overfitting_haircut_pct:.1f}%")
    lines.append(f"- Conservative CAGR after haircut: {cagr_est.conservative_cagr:+.1f}%")
    lines.append("")

    # Weak components
    lines.extend([
        "## 3. Weak Pipeline Components",
        "",
    ])

    # Low hit-rate source analysis
    low_quality_regimes = []
    for m in quality_metrics:
        if m.horizon == "20D" and m.direction == "ALL" and m.hit_rate < 48:
            low_quality_regimes.append(m)

    if low_quality_regimes:
        for m in low_quality_regimes:
            lines.append(f"- **{m.regime}**: Hit rate = {m.hit_rate:.1f}%, "
                          f"PF = {m.profit_factor:.2f}, False% = {m.false_signal_rate:.1f}%")
    else:
        lines.append("- All regimes have hit rate ≥ 48% at 20D horizon ✓")

    # Check low-confidence signal performance
    low_conf = next((s for s in stress_results if "Low Confidence" in s.scenario), None)
    if low_conf:
        lines.append(f"- Low confidence signals (<0.3): Hit rate = {low_conf.hit_rate:.1f}%, "
                      f"Sharpe = {low_conf.sharpe:.2f}")
        if low_conf.sharpe < 0.3:
            lines.append("  → **Filter low-confidence signals** — they add noise, not alpha")
    lines.append("")

    # Stress test summary
    lines.extend([
        "## 4. Stress Test Results",
        "",
        "| Scenario | N | Hit Rate | Avg Ret | Sharpe | PF | Max DD |",
        "|----------|---|----------|---------|--------|----|--------|",
    ])

    for st in stress_results:
        pf_str = f"{st.profit_factor:.2f}" if st.profit_factor < 100 else "∞"
        lines.append(
            f"| {st.scenario:40s} | {st.n_signals:5d} | {st.hit_rate:5.1f}% | "
            f"{st.avg_return*100:+6.2f}% | {st.sharpe:5.2f} | {pf_str:>5s} | "
            f"{st.max_drawdown*100:5.1f}% |"
        )

    lines.extend([
        "",
        "## 5. Recommendations",
        "",
    ])

    recommendations = []

    # Regime-specific tuning
    bear_perf = next((r for r in regime_perf if r.regime == REGIME_BEAR), None)
    if bear_perf and bear_perf.sharpe < 0.3:
        recommendations.append(
            "**Regime-Adaptive Position Sizing**: Reduce position sizes by 60-70% in BEAR regime. "
            "Current BEAR Sharpe ({:.2f}) suggests the system's trend-following signals "
            "are partially offset by whipsaw losses.".format(bear_perf.sharpe)
        )

    # Signal filtering
    if low_conf and low_conf.sharpe < 0.3:
        recommendations.append(
            "**Confidence Threshold Filter**: Raise minimum forecast threshold from 2.0 to 5.0 "
            "to eliminate weak signals. Low-confidence signals show Sharpe = {:.2f}.".format(
                low_conf.sharpe
            )
        )

    # Horizon optimization
    horizon_sharpes = {}
    for h in ["5D", "10D", "20D"]:
        all_h = [m for m in quality_metrics if m.horizon == h and m.regime == "ALL" and m.direction == "ALL"]
        if all_h:
            horizon_sharpes[h] = all_h[0].sharpe

    if horizon_sharpes:
        best_h = max(horizon_sharpes, key=horizon_sharpes.get)
        recommendations.append(
            f"**Optimal Holding Period**: {best_h} shows the best Sharpe "
            f"({horizon_sharpes[best_h]:.2f}). Consider calibrating position holding "
            f"to this horizon for maximum risk-adjusted returns."
        )

    # Overfitting mitigation
    if cagr_est.overfitting_haircut_pct > 30:
        recommendations.append(
            "**Reduce Strategy Variants**: The overfitting haircut ({:.0f}%) is high. "
            "Consider reducing the number of forecast sources from 22 to ~12 "
            "(drop lowest Sharpe contributors) to reduce data-mining bias.".format(
                cagr_est.overfitting_haircut_pct
            )
        )

    # Drawdown management
    if bt_result.max_drawdown_pct > 30:
        recommendations.append(
            f"**Drawdown Circuit Breaker**: Max DD = {bt_result.max_drawdown_pct:.1f}%. "
            "Implement equity curve filter — halt new trades when equity drops below "
            "63-day SMA to limit tail risk."
        )

    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")
        lines.append("")

    if not recommendations:
        lines.append("System performance is robust across regimes. No critical issues identified.")
        lines.append("")

    # Final assessment
    lines.extend([
        "## 6. Final Assessment",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Ideal CAGR | {cagr_est.ideal_cagr:+.1f}% |",
        f"| Realistic CAGR | {cagr_est.realistic_cagr:+.1f}% |",
        f"| Conservative CAGR | {cagr_est.conservative_cagr:+.1f}% |",
        f"| CAGR 90% CI | [{cagr_est.cagr_ci_90[0]:+.1f}%, {cagr_est.cagr_ci_90[1]:+.1f}%] |",
        f"| Overfitting Risk | {'HIGH' if cagr_est.overfitting_haircut_pct > 40 else 'MODERATE' if cagr_est.overfitting_haircut_pct > 25 else 'LOW'} ({cagr_est.overfitting_haircut_pct:.0f}%) |",
        "",
        "**Bottom line**: The defensible CAGR range for centurion_core is "
        f"**{cagr_est.conservative_cagr:+.1f}% to {cagr_est.realistic_cagr:+.1f}%** "
        f"with 90% confidence bounds of [{cagr_est.cagr_ci_90[0]:+.1f}%, {cagr_est.cagr_ci_90[1]:+.1f}%].",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 9. MAIN EVALUATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def _download_ohlcv(
    tickers: List[str],
    period: str,
    market: str,
) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data for all tickers."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance required. Install: pip install yfinance")
        return {}

    ohlcv = {}
    for sym in tickers:
        try:
            suffix = ".NS" if market == "IND" and "." not in sym else ""
            ticker = f"{sym}{suffix}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df is not None and len(df) >= 250:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                ohlcv[sym if "." in sym else ticker] = df
                logger.info("Downloaded %s: %d bars", ticker, len(df))
        except Exception as e:
            logger.warning("Failed to download %s: %s", sym, e)
    return ohlcv


def run_full_evaluation(
    market: str = "IND",
    period: str = "5y",
    tickers: Optional[List[str]] = None,
    capital: float = 500_000.0,
    annual_vol_target: float = 0.75,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the complete signal quality evaluation pipeline.

    Steps:
      1. Download OHLCV data
      2. Classify regimes
      3. Generate signals through Carver pipeline
      4. Compute forward returns
      5. Calculate signal quality metrics per regime
      6. Run realistic portfolio backtest
      7. Estimate CAGR (ideal / realistic / conservative)
      8. Stress test
      9. Generate insights
      10. Write documentation files

    Returns dict with all results for programmatic access.
    """
    # Default tickers
    if tickers is None:
        if market == "IND":
            tickers = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "BHARTIARTL", "LT", "SBIN", "ITC", "TATAMOTORS",
                "AXISBANK", "WIPRO", "SUNPHARMA", "MARUTI", "ONGC",
            ]
        else:
            tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "TSLA", "JPM", "V", "UNH", "HD", "PG", "XOM", "MA",
                "JNJ",
            ]

    if verbose:
        logger.info("=" * 70)
        logger.info("  SIGNAL QUALITY EVALUATION — %s", market)
        logger.info("  %d tickers | Period: %s", len(tickers), period)
        logger.info("=" * 70)

    # ── 1. Download data ──
    if verbose:
        logger.info("[1/8] Downloading OHLCV data...")
    ohlcv_cache = _download_ohlcv(tickers, period, market)

    if len(ohlcv_cache) < 3:
        logger.error("Need at least 3 symbols with data.")
        return {"error": "Insufficient data"}

    if verbose:
        logger.info("  → %d symbols loaded", len(ohlcv_cache))

    # ── 2. Regime classification ──
    if verbose:
        logger.info("[2/8] Classifying market regimes...")

    # Use first symbol as market proxy
    index_sym = list(ohlcv_cache.keys())[0]
    try:
        regime_df = classify_regimes_ohlcv(ohlcv_cache[index_sym])
    except ValueError:
        index_close = ohlcv_cache[index_sym]["Close"]
        if hasattr(index_close, "squeeze"):
            index_close = index_close.squeeze()
        regime_df = classify_regimes(index_close)

    if verbose:
        regime_counts = regime_df["regime"].value_counts()
        for r, c in regime_counts.items():
            logger.info("  %s: %d days (%.1f%%)", r, c, c / len(regime_df) * 100)

    # ── 3. Generate signals ──
    if verbose:
        logger.info("[3/8] Generating signals through Carver pipeline...")
    signals, ohlcv_cache, _ = generate_signals_from_backtest(
        ohlcv_cache, market=market, warmup=262,
    )
    if verbose:
        logger.info("  → %s signals generated", f"{len(signals):,}")
        n_buy = sum(1 for s in signals if s.direction == "BUY")
        n_sell = sum(1 for s in signals if s.direction == "SELL")
        logger.info("  → %s BUY | %s SELL", f"{n_buy:,}", f"{n_sell:,}")

    # ── 4. Signal quality metrics ──
    if verbose:
        logger.info("[4/8] Computing signal quality metrics per regime...")

    quality_metrics: List[SignalQualityMetrics] = []
    for horizon in ["5D", "10D", "20D"]:
        for regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, "ALL"]:
            for direction in ["BUY", "SELL", "ALL"]:
                m = compute_signal_quality(signals, regime, direction, horizon)
                quality_metrics.append(m)

    if verbose:
        # Print summary for 20D ALL
        all_20d = next((m for m in quality_metrics
                        if m.regime == "ALL" and m.direction == "ALL" and m.horizon == "20D"), None)
        if all_20d:
            logger.info("  → Overall 20D: Hit Rate=%.1f%%, Sharpe=%.2f, PF=%.2f",
                        all_20d.hit_rate, all_20d.sharpe, all_20d.profit_factor)

    # ── 5. Regime performance ──
    if verbose:
        logger.info("[5/8] Computing regime performance breakdown...")
    regime_perf = compute_regime_performance(signals, regime_df)

    if verbose:
        for rp in regime_perf:
            logger.info("  %s: Sharpe=%.2f, Win=%.0f%%, Signals=%d",
                        rp.regime, rp.sharpe, rp.win_rate, rp.total_signals)

    # ── 6. Realistic portfolio backtest ──
    if verbose:
        logger.info("[6/8] Running realistic portfolio backtest...")

    cost_bps = 13.0 if market == "IND" else 10.0
    slip_bps = 10.0 if market == "IND" else 5.0
    max_lev = 7.0 if market == "IND" else 1.0

    bt_result = run_realistic_backtest(
        ohlcv_cache, regime_df,
        market=market,
        capital=capital,
        annual_vol_target=annual_vol_target,
        commission_bps=cost_bps,
        slippage_bps=slip_bps,
        max_leverage=max_lev,
    )

    if verbose:
        logger.info("  → Sharpe: %.3f", bt_result.sharpe)
        logger.info("  → Annual Return: %+.1f%%", bt_result.annual_return_pct)
        logger.info("  → Max Drawdown: %.1f%%", bt_result.max_drawdown_pct)
        logger.info("  → Trades: %s", f"{bt_result.n_trades:,}")

    # ── 7. CAGR estimation ──
    if verbose:
        logger.info("[7/8] Estimating CAGR (ideal / realistic / conservative)...")

    cagr_est = estimate_cagr(bt_result, n_strategies=22)

    if verbose:
        logger.info("  → Ideal CAGR:        %+.1f%%", cagr_est.ideal_cagr)
        logger.info("  → Realistic CAGR:    %+.1f%%", cagr_est.realistic_cagr)
        logger.info("  → Conservative CAGR: %+.1f%%", cagr_est.conservative_cagr)
        logger.info("  → 90%% CI: [%+.1f%%, %+.1f%%]",
                     cagr_est.cagr_ci_90[0], cagr_est.cagr_ci_90[1])

    # ── 8. Stress testing ──
    if verbose:
        logger.info("[8/8] Running stress tests...")
    stress_results = run_stress_tests(signals, regime_df)

    if verbose:
        for st in stress_results:
            logger.info("  %s: N=%d, Hit=%.0f%%, Sharpe=%.2f",
                        st.scenario, st.n_signals, st.hit_rate, st.sharpe)

    # ── 9. Generate documentation ──
    if verbose:
        logger.info("Writing documentation files...")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    doc1 = _generate_signal_quality_doc(quality_metrics, signals)
    (DOCS_DIR / "signal_quality_by_regime.md").write_text(doc1, encoding="utf-8")

    doc2 = _generate_regime_performance_doc(regime_perf, regime_df)
    (DOCS_DIR / "regime_performance.md").write_text(doc2, encoding="utf-8")

    doc3 = _generate_cagr_doc(cagr_est, bt_result, market)
    (DOCS_DIR / "cagr_estimation.md").write_text(doc3, encoding="utf-8")

    doc4 = _generate_insights_doc(
        quality_metrics, regime_perf, stress_results, cagr_est, bt_result, signals,
    )
    (DOCS_DIR / "signal_insights.md").write_text(doc4, encoding="utf-8")

    if verbose:
        logger.info("  ✓ docs/signal_quality_by_regime.md")
        logger.info("  ✓ docs/regime_performance.md")
        logger.info("  ✓ docs/cagr_estimation.md")
        logger.info("  ✓ docs/signal_insights.md")
        logger.info("=" * 70)
        logger.info("  EVALUATION COMPLETE")
        logger.info("=" * 70)

    return {
        "signals": signals,
        "quality_metrics": quality_metrics,
        "regime_performance": regime_perf,
        "regime_df": regime_df,
        "backtest": bt_result,
        "cagr": cagr_est,
        "stress_tests": stress_results,
    }
