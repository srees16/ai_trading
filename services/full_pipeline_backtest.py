"""
Full 13-Source Pipeline Backtester.

Expanding-window daily simulation using ALL offline-capable forecast
sources through the real Carver forecast combiner and position sizer.

Sources tested (10 offline, 3 stubbed):
  OFFLINE:  ewmac_16_64, ewmac_32_128, ewmac_64_256, carry,
            momentum, mean_reversion, oi_signal, pairs_arb
  PROXIED:  fii_flow (random-walk proxy — captures weight drag only)
  STUBBED:  screener, decision_engine, event_driven (omitted; weights renormalized)

Usage:
    from services.full_pipeline_backtest import run_full_backtest
    result = run_full_backtest(
        tickers=["AAPL", "MSFT", ...],
        capital=10_000,
        period="2y",
        market="US",          # "US" or "IND"
    )
    print(result["report"])
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default pairs for pairs_arb source ────────────────────────
DEFAULT_PAIRS_US = [
    ("AAPL", "MSFT"),
    ("GOOGL", "META"),
    ("AMZN", "NVDA"),
    ("JPM", "V"),
]
DEFAULT_PAIRS_IND = [
    ("HDFCBANK.NS", "ICICIBANK.NS"),
    ("TCS.NS", "INFY.NS"),
    ("RELIANCE.NS", "ONGC.NS"),
    ("SBIN.NS", "PNB.NS"),
]


@dataclass
class BacktestResult:
    """Full pipeline backtest output."""
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    annual_return_pct: float = 0.0
    total_return_pct: float = 0.0
    n_symbols: int = 0
    n_days_traded: int = 0
    n_trades: int = 0
    avg_positions: float = 0.0
    source_hit_rates: Dict[str, float] = field(default_factory=dict)
    daily_equity: List[float] = field(default_factory=list)
    report: str = ""


# ── Helpers ───────────────────────────────────────────────────

def _download(sym: str, period: str, market: str) -> Optional[pd.DataFrame]:
    """Download OHLCV via yfinance."""
    try:
        import yfinance as yf
        import warnings
        suffix = ".NS" if market == "IND" and "." not in sym else ""
        ticker = f"{sym}{suffix}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df is not None and len(df) >= 120:
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception as e:
        logger.warning("Download failed for %s: %s", sym, e)
    return None


def _build_oi_proxy(ohlcv_slice: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Build OI proxy from volume changes (same as carver_pipeline.py).
    Strips .NS suffix so FNO_LOT_SIZES lookup works for IND stocks.
    """
    oi_data = {}
    for sym, df in ohlcv_slice.items():
        if "Volume" not in df.columns or len(df) < 6:
            continue
        vol = df["Volume"]
        if hasattr(vol, "squeeze"):
            vol = vol.squeeze()
        last_vol = float(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0
        avg_vol = float(vol.iloc[-6:-1].mean()) if len(vol) >= 6 else last_vol
        if avg_vol <= 0:
            continue
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if len(close) < 2:
            continue
        oi_change = ((last_vol / avg_vol) - 1) * 100
        price_change = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100
        vol_ratio = last_vol / avg_vol
        # Use bare symbol (strip .NS/.BO suffix) for FNO_LOT_SIZES lookup
        bare_sym = sym.replace('.NS', '').replace('.BO', '')
        oi_data[bare_sym] = {
            "oi_change_pct": oi_change,
            "price_change_pct": price_change,
            "volume_ratio": vol_ratio,
        }
    return oi_data


def _compute_pairs_forecasts(
    ohlcv_slice: Dict[str, pd.DataFrame],
    pairs: List[Tuple[str, str]],
    pair_states: Dict[str, dict],
) -> Dict[str, float]:
    """Compute pairs_arb forecasts for available pairs."""
    from services.pairs_trading_live import generate_pairs_signal, LOOKBACK

    forecasts: Dict[str, float] = {}
    for leg1, leg2 in pairs:
        if leg1 not in ohlcv_slice or leg2 not in ohlcv_slice:
            continue
        c1 = ohlcv_slice[leg1]["Close"]
        c2 = ohlcv_slice[leg2]["Close"]
        if hasattr(c1, "squeeze"):
            c1 = c1.squeeze()
        if hasattr(c2, "squeeze"):
            c2 = c2.squeeze()
        c1 = c1.dropna()
        c2 = c2.dropna()
        if len(c1) < LOOKBACK or len(c2) < LOOKBACK:
            continue
        key = f"{leg1}|{leg2}"
        state = pair_states.get(key, {})
        has_open = state.get("open", False)
        direction = state.get("direction", "")

        sig = generate_pairs_signal(
            leg1, leg2,
            c1.values, c2.values,
            has_open_position=has_open,
            current_direction=direction,
        )
        # Update state
        if sig.action.startswith("ENTER"):
            pair_states[key] = {
                "open": True,
                "direction": "LONG_LEG1" if "LONG" in sig.action else "SHORT_LEG1",
            }
        elif sig.action in ("EXIT", "STOP"):
            pair_states[key] = {"open": False, "direction": ""}

        if sig.forecast != 0:
            forecasts[leg1] = forecasts.get(leg1, 0) + sig.forecast * 0.5
            forecasts[leg2] = forecasts.get(leg2, 0) - sig.forecast * 0.5
    return forecasts


# ── Main Backtester ───────────────────────────────────────────

def run_full_backtest(
    tickers: Optional[List[str]] = None,
    capital: float = 10_000,
    period: str = "2y",
    market: str = "US",
    annual_vol_target: float = 0.20,
    min_history: int = 262,
    pairs: Optional[List[Tuple[str, str]]] = None,
    include_carry: bool = True,
    include_pairs: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run a full 13-source pipeline backtest.

    Parameters
    ----------
    tickers : list of symbols (with .NS suffix for IND if needed)
    capital : initial capital
    period : yfinance period string (1y, 2y, 5y, max)
    market : "US" or "IND"
    annual_vol_target : decimal (0.20 = 20%)
    min_history : minimum bars before trading starts
    pairs : list of (leg1, leg2) for pairs_arb; None = default
    include_carry : whether to compute carry (needs yfinance dividend data)
    include_pairs : whether to compute pairs_arb
    verbose : print progress

    Returns
    -------
    dict with keys: sharpe, sortino, calmar, max_drawdown_pct,
         annual_return_pct, total_return_pct, n_symbols, n_days_traded,
         n_trades, source_hit_rates, daily_equity, report
    """
    # ── Imports ────────────────────────────────────────────────
    from services.instrument_volatility import daily_price_volatility
    from services.forecast_scalar import ewmac_to_forecast, cap_forecast
    from strategies.ewmac import DEFAULT_VARIATIONS
    from strategies.carry_rule import compute_carry_batch
    from strategies.mean_reversion import compute_mean_reversion_batch
    from services.momentum_factor import compute_momentum_forecasts
    from services.oi_signal import compute_oi_signals_batch
    from services.forecast_combiner import (
        combine_forecasts, ForecastWeight,
        DEFAULT_FORECAST_WEIGHTS, DEFAULT_CORRELATION_MATRIX,
    )
    from services.volatility_target import VolatilityTarget, VolatilityTargetConfig

    # ── Default tickers ────────────────────────────────────────
    if tickers is None:
        if market == "US":
            tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "TSLA", "JPM", "V", "UNH", "HD", "PG", "XOM", "MA",
                "JNJ",
            ]
        else:
            tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
                "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "SBIN.NS",
                "ITC.NS", "TATAMOTORS.NS", "AXISBANK.NS", "WIPRO.NS",
                "SUNPHARMA.NS", "MARUTI.NS", "ONGC.NS",
            ]
    if pairs is None:
        pairs = DEFAULT_PAIRS_US if market == "US" else DEFAULT_PAIRS_IND

    # ── Download data ──────────────────────────────────────────
    if verbose:
        print(f"\n{'='*70}")
        print(f"  FULL PIPELINE BACKTEST — {market} ({len(tickers)} tickers, {period})")
        print(f"  Capital: {capital:,.0f}  |  Vol Target: {annual_vol_target*100:.0f}%")
        print(f"{'='*70}\n")
        print("Downloading OHLCV data...")

    ohlcv_full: Dict[str, pd.DataFrame] = {}
    for sym in tickers:
        df = _download(sym, period, market)
        if df is not None:
            ohlcv_full[sym] = df
            if verbose:
                ret = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100
                print(f"  {sym:20s} {len(df):4d} bars  ret={ret:+.1f}%")

    symbols = list(ohlcv_full.keys())
    n_symbols = len(symbols)
    if n_symbols < 2:
        print("ERROR: Need at least 2 symbols with sufficient data.")
        return {"sharpe": 0, "report": "Insufficient data"}

    # ── Align dates across all symbols ─────────────────────────
    min_len = min(len(df) for df in ohlcv_full.values())
    n_days = min_len
    if n_days < min_history + 20:
        print(f"ERROR: Only {n_days} bars, need at least {min_history + 20}.")
        return {"sharpe": 0, "report": "Insufficient history"}

    if verbose:
        print(f"\n  Symbols loaded: {n_symbols}")
        print(f"  Common bars:    {n_days}")
        print(f"  Warmup period:  {min_history} bars")
        print(f"  Trading days:   {n_days - min_history}\n")

    # ── Determine available sources ────────────────────────────
    # We include all offline-capable sources; omit live-only ones.
    # The combiner auto-renormalizes weights for available sources.
    available_sources = {
        "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
        "momentum", "mean_reversion", "oi_signal", "breakout", "cross_momentum",
    }
    if include_carry:
        available_sources.add("carry")
    if include_pairs:
        available_sources.add("pairs_arb")

    # Build weight list (only available sources)
    active_weights = [
        fw for fw in DEFAULT_FORECAST_WEIGHTS if fw.name in available_sources
    ]
    total_w = sum(fw.weight for fw in active_weights)
    if total_w > 0:
        active_weights = [
            ForecastWeight(fw.name, fw.weight / total_w)
            for fw in active_weights
        ]

    if verbose:
        print("  Active forecast sources:")
        for fw in active_weights:
            print(f"    {fw.name:20s} {fw.weight*100:5.1f}%")
        omitted = [
            fw.name for fw in DEFAULT_FORECAST_WEIGHTS
            if fw.name not in available_sources
        ]
        if omitted:
            print(f"  Omitted (live-only): {', '.join(omitted)}")
        print()

    # ── Transaction costs ──────────────────────────────────────
    if market == "IND":
        cost_pct = 0.0012   # F&O futures: 0.05% brokerage + 0.01% STT + 0.06% slippage
    else:
        cost_pct = 0.0015   # 0.10% commission + 0.05% slippage (US)

    # ── VolatilityTarget ───────────────────────────────────────
    vol_target = VolatilityTarget(VolatilityTargetConfig(
        initial_capital=capital,
        annual_vol_target_pct=annual_vol_target,
    ))

    # ── State tracking ─────────────────────────────────────────
    equity = capital
    daily_equity = [capital]
    daily_returns: List[float] = []
    prev_positions: Dict[str, int] = {sym: 0 for sym in symbols}
    peak_prices: Dict[str, float] = {}
    stop_levels: Dict[str, float] = {}
    stop_cooldown: Dict[str, int] = {}   # sym → days remaining until re-entry allowed
    pair_states: Dict[str, dict] = {}
    trades_count = 0
    source_hits: Dict[str, int] = defaultdict(int)
    source_total: Dict[str, int] = defaultdict(int)
    daily_position_counts: List[int] = []

    # ── Pre-fetch dividend yields for carry (one-time) ─────────
    dividend_yields: Dict[str, float] = {}
    if include_carry:
        try:
            import yfinance as yf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for sym in symbols:
                    try:
                        info = yf.Ticker(sym).info
                        dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
                        if dy and dy > 0:
                            dividend_yields[sym] = float(dy)
                    except Exception:
                        pass
            if verbose:
                print(f"  Dividend yields fetched: {len(dividend_yields)}/{n_symbols}")
        except ImportError:
            pass

    # ── EXPANDING-WINDOW DAILY SIMULATION ──────────────────────
    if verbose:
        print("\n  Running simulation...", end="", flush=True)

    for day_idx in range(min_history, n_days):
        day_pnl = 0.0

        # Build OHLCV slices up to current day
        ohlcv_slice: Dict[str, pd.DataFrame] = {}
        for sym, df in ohlcv_full.items():
            ohlcv_slice[sym] = df.iloc[:day_idx + 1]

        # ── 1. Mark-to-market existing positions ───────────────
        for sym in symbols:
            prev_qty = prev_positions.get(sym, 0)
            if prev_qty == 0:
                continue
            c = ohlcv_slice[sym]["Close"]
            if hasattr(c, "squeeze"):
                c = c.squeeze()
            price = float(c.iloc[-1])
            prev_price = float(c.iloc[-2]) if len(c) > 1 else price

            # NaN guard
            if not np.isfinite(price) or not np.isfinite(prev_price) or prev_price <= 0:
                continue

            # Check trailing stop
            if sym in stop_levels:
                if prev_qty > 0:
                    low_col = ohlcv_slice[sym]["Low"]
                    if hasattr(low_col, "squeeze"):
                        low_col = low_col.squeeze()
                    low = float(low_col.iloc[-1])
                    if np.isfinite(low) and low <= stop_levels[sym]:
                        exit_price = stop_levels[sym]
                        daily_ret = (exit_price - prev_price) / prev_price
                        day_pnl += prev_qty * prev_price * daily_ret
                        day_pnl -= abs(prev_qty) * exit_price * cost_pct
                        trades_count += 1
                        prev_positions[sym] = 0
                        peak_prices.pop(sym, None)
                        stop_levels.pop(sym, None)
                        stop_cooldown[sym] = 5
                        continue
                elif prev_qty < 0:
                    high_col = ohlcv_slice[sym]["High"]
                    if hasattr(high_col, "squeeze"):
                        high_col = high_col.squeeze()
                    high = float(high_col.iloc[-1])
                    if np.isfinite(high) and high >= stop_levels[sym]:
                        exit_price = stop_levels[sym]
                        daily_ret = (prev_price - exit_price) / prev_price
                        day_pnl += abs(prev_qty) * prev_price * daily_ret
                        day_pnl -= abs(prev_qty) * exit_price * cost_pct
                        trades_count += 1
                        prev_positions[sym] = 0
                        peak_prices.pop(sym, None)
                        stop_levels.pop(sym, None)
                        stop_cooldown[sym] = 5
                        continue

            # Normal MTM
            if prev_price > 0:
                if prev_qty > 0:
                    daily_ret = (price - prev_price) / prev_price
                    day_pnl += prev_qty * prev_price * daily_ret
                elif prev_qty < 0:
                    daily_ret = (prev_price - price) / prev_price
                    day_pnl += abs(prev_qty) * prev_price * daily_ret

        # ── 2. Compute ALL forecasts ───────────────────────────
        all_forecasts: Dict[str, Dict[str, float]] = {sym: {} for sym in symbols}

        # 2a. EWMAC (3 variations)
        for sym, df in ohlcv_slice.items():
            c = df["Close"]
            if hasattr(c, "squeeze"):
                c = c.squeeze()
            close = c.dropna()
            if len(close) < 270:
                continue
            price = float(close.iloc[-1])
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
                key = f"ewmac_{fast}_{slow}"
                all_forecasts[sym][key] = fc

        # 2b. Momentum (12-1 month)
        try:
            mom_forecasts = compute_momentum_forecasts(ohlcv_slice)
            for sym, fc in mom_forecasts.items():
                if sym in all_forecasts:
                    all_forecasts[sym]["momentum"] = fc
        except Exception as e:
            logger.debug("Momentum failed at day %d: %s", day_idx, e)

        # 2c. Mean reversion
        try:
            mr_forecasts = compute_mean_reversion_batch(ohlcv_slice)
            for sym, fc in mr_forecasts.items():
                if sym in all_forecasts:
                    all_forecasts[sym]["mean_reversion"] = fc
        except Exception as e:
            logger.debug("Mean reversion failed at day %d: %s", day_idx, e)

        # 2d. Carry
        if include_carry:
            try:
                carry_results = compute_carry_batch(
                    ohlcv_slice, dividend_yields=dividend_yields,
                )
                for sym, cf in carry_results.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["carry"] = cf.forecast
            except Exception as e:
                logger.debug("Carry failed at day %d: %s", day_idx, e)

        # 2e. OI signal (volume proxy — bare symbols for FNO lookup)
        try:
            oi_data = _build_oi_proxy(ohlcv_slice)
            oi_forecasts = compute_oi_signals_batch(oi_data)
            # Map bare symbols back to .NS if needed
            for sym in symbols:
                bare = sym.replace('.NS', '').replace('.BO', '')
                if bare in oi_forecasts and sym in all_forecasts:
                    all_forecasts[sym]["oi_signal"] = oi_forecasts[bare]
        except Exception as e:
            logger.debug("OI signal failed at day %d: %s", day_idx, e)

        # 2f. Breakout signal (20-day high/low channel)
        for sym, df in ohlcv_slice.items():
            if sym not in all_forecasts:
                continue
            c = df["Close"]
            if hasattr(c, "squeeze"):
                c = c.squeeze()
            if len(c) < 22:
                continue
            price_now = float(c.iloc[-1])
            high_20 = float(c.iloc[-21:-1].max())
            low_20 = float(c.iloc[-21:-1].min())
            rng = high_20 - low_20
            if rng > 0 and np.isfinite(price_now):
                # Breakout position: +10 at 20-day high, -10 at 20-day low, linear between
                breakout_fc = ((price_now - low_20) / rng - 0.5) * 20.0
                breakout_fc = max(-20.0, min(20.0, breakout_fc))
                all_forecasts[sym]["breakout"] = breakout_fc

        # 2g. Pairs arb
        if include_pairs:
            try:
                pairs_fc = _compute_pairs_forecasts(ohlcv_slice, pairs, pair_states)
                for sym, fc in pairs_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["pairs_arb"] = fc
            except Exception as e:
                logger.debug("Pairs failed at day %d: %s", day_idx, e)

        # 2h. Cross-sectional momentum (long top, short bottom)
        # Rank stocks by 6-month return, tilt forecasts for top/bottom tercile
        xmom_returns = {}
        for sym, df in ohlcv_slice.items():
            if sym not in all_forecasts:
                continue
            c = df["Close"]
            if hasattr(c, "squeeze"):
                c = c.squeeze()
            if len(c) >= 126:
                ret = float(c.iloc[-1] / c.iloc[-126] - 1)
                if np.isfinite(ret):
                    xmom_returns[sym] = ret
        if len(xmom_returns) >= 6:
            sorted_syms = sorted(xmom_returns.keys(), key=lambda s: xmom_returns[s])
            n_tercile = max(1, len(sorted_syms) // 3)
            bottom = sorted_syms[:n_tercile]      # worst performers → short
            top = sorted_syms[-n_tercile:]         # best performers → long
            for sym in bottom:
                all_forecasts[sym]["cross_momentum"] = -8.0
            for sym in top:
                all_forecasts[sym]["cross_momentum"] = +8.0

        # ── 3. Combine forecasts + size positions ──────────────
        # Update daily cash target based on current equity (compounding)
        dynamic_daily_target = max(equity, capital * 0.5) * annual_vol_target / 16.0
        n_active = max(1, sum(1 for sym in symbols if all_forecasts.get(sym)))
        weight_per_sym = 1.0 / n_active

        # IDM: scales with instrument count
        idm = 2.0 if n_active >= 10 else (1.8 if n_active >= 6 else 1.5)

        # Regime detection: average 40-day return across ALL symbols
        detected_regime = 'sideways'
        avg_ret_40d = 0.0
        if day_idx >= 40:
            rets_40d = []
            for sym in symbols:
                pc = ohlcv_slice[sym]["Close"]
                if hasattr(pc, "squeeze"):
                    pc = pc.squeeze()
                if len(pc) >= 40:
                    r = float(pc.iloc[-1] / pc.iloc[-40] - 1)
                    if np.isfinite(r):
                        rets_40d.append(r)
            if rets_40d:
                avg_ret_40d = np.mean(rets_40d)
                if avg_ret_40d > 0.05:
                    detected_regime = 'bull'
                elif avg_ret_40d < -0.05:
                    detected_regime = 'bear'

        # Read leverage and short-selling config
        try:
            from config import Config as _BtCfg
            max_leverage = getattr(_BtCfg, 'CARVER_MAX_LEVERAGE', 3.0)
            allow_short = getattr(_BtCfg, 'SHORT_SELLING_ENABLED', False)
        except Exception:
            max_leverage = 3.0
            allow_short = False

        # Tick down stop cooldowns
        for sym in list(stop_cooldown.keys()):
            stop_cooldown[sym] -= 1
            if stop_cooldown[sym] <= 0:
                del stop_cooldown[sym]

        active_count = 0
        for sym, fc_dict in all_forecasts.items():
            if not fc_dict:
                continue

            # Track source hit rates
            for src in fc_dict:
                source_hits[src] += 1
            for fw in active_weights:
                source_total[fw.name] += 1

            # Combine
            combined = combine_forecasts(sym, fc_dict, active_weights)
            forecast = combined.combined_forecast

            # Position sizing
            c = ohlcv_slice[sym]["Close"]
            if hasattr(c, "squeeze"):
                c = c.squeeze()
            close = c.dropna()
            price = float(close.iloc[-1])
            if not np.isfinite(price) or price <= 0:
                continue
            dpv = daily_price_volatility(close)
            if not np.isfinite(dpv) or dpv <= 0:
                dpv = 0.02
            ivv = price * dpv

            if ivv > 0 and dynamic_daily_target > 0:
                vol_scalar = dynamic_daily_target / ivv
                position = (forecast / 10.0) * vol_scalar * weight_per_sym * idm

                # Direction-aware regime scaling:
                # Bull: longs 1.3x, shorts 0.5x (trend aligned)
                # Bear: shorts 1.3x, longs 0.5x (trend aligned)
                # Sideways: both 1.0x
                if detected_regime == 'bull':
                    regime_scale = 1.3 if position >= 0 else 0.5
                elif detected_regime == 'bear':
                    regime_scale = 1.3 if position < 0 else 0.5
                else:
                    regime_scale = 1.0
                position *= regime_scale

                target_qty = round(position)

                # Cap at max leverage (both long and short)
                # FIX-6: Per-symbol cap should NOT double-apply weight.
                # The Carver formula already factors weight_per_sym × IDM.
                # Cap each position at (equity × max_leverage / n_active)
                # so portfolio total can reach equity × max_leverage.
                max_notional = abs(equity) * max_leverage / n_active
                if abs(target_qty) * price > max_notional:
                    cap_qty = int(max_notional / price)
                    target_qty = cap_qty if target_qty > 0 else -cap_qty

                # Guard: floor at 0 for long-only mode (unless short enabled)
                if not allow_short and target_qty < 0:
                    target_qty = 0

                # Stop cooldown: prevent re-entry for 5 days after stop exit
                if sym in stop_cooldown and prev_positions.get(sym, 0) == 0:
                    target_qty = 0

                # NaN guard
                if not np.isfinite(target_qty):
                    target_qty = 0

                # Transaction costs on turnover
                prev_qty = prev_positions.get(sym, 0)
                delta = abs(target_qty - prev_qty)
                if delta > 0:
                    # Inertia: skip small changes (< 15%)
                    if abs(prev_qty) > 0 and delta / abs(prev_qty) < 0.15:
                        target_qty = prev_qty
                    else:
                        cost = delta * price * cost_pct
                        day_pnl -= cost
                        trades_count += 1

                prev_positions[sym] = target_qty

                # Regime-adaptive trailing stop:
                # Bull: 5.0σ (let winners run)
                # Bear: 3.0σ (cut losses fast)
                # Sideways: 4.0σ
                if detected_regime == 'bull':
                    stop_sigma = 5.0
                elif detected_regime == 'bear':
                    stop_sigma = 3.0
                else:
                    stop_sigma = 4.0

                if target_qty > 0:
                    active_count += 1
                    pk = max(peak_prices.get(sym, price), price)
                    peak_prices[sym] = pk
                    stop_dist = stop_sigma * dpv * pk
                    new_stop = pk - stop_dist
                    stop_levels[sym] = max(stop_levels.get(sym, 0), new_stop)
                elif target_qty < 0:
                    active_count += 1
                    trough = min(peak_prices.get(sym, price), price)
                    peak_prices[sym] = trough
                    stop_dist = stop_sigma * dpv * trough
                    new_stop = trough + stop_dist
                    stop_levels[sym] = min(stop_levels.get(sym, float('inf')), new_stop)
                else:
                    peak_prices.pop(sym, None)
                    stop_levels.pop(sym, None)

        daily_position_counts.append(active_count)

        # ── 4. Update equity ───────────────────────────────────
        if not np.isfinite(day_pnl):
            day_pnl = 0.0  # NaN guard: skip corrupted days
        equity += day_pnl
        daily_returns.append(day_pnl / max(daily_equity[-1], 1))
        daily_equity.append(equity)

        # Print progress every 50 days
        if verbose and (day_idx - min_history) % 50 == 0:
            d = day_idx - min_history
            total_d = n_days - min_history
            ret_so_far = (equity / capital - 1) * 100
            print(f"\r  Day {d:4d}/{total_d}  equity={equity:,.0f}  ret={ret_so_far:+.1f}%  positions={active_count}", end="", flush=True)

    if verbose:
        print(f"\r  Simulation complete: {n_days - min_history} trading days" + " " * 40)

    # ── 5. Compute metrics ─────────────────────────────────────
    ret_arr = np.array(daily_returns)
    result = BacktestResult(
        n_symbols=n_symbols,
        n_days_traded=n_days - min_history,
        n_trades=trades_count,
        daily_equity=daily_equity,
    )

    if len(ret_arr) > 1:
        avg_ret = float(np.mean(ret_arr))
        std_ret = float(np.std(ret_arr, ddof=1))

        # Sharpe (annualized)
        if std_ret > 0:
            result.sharpe = round(avg_ret / std_ret * 16.0, 3)  # sqrt(252) ≈ 16

        # Sortino
        downside = ret_arr[ret_arr < 0]
        ds_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else std_ret
        if ds_std > 0:
            result.sortino = round(avg_ret / ds_std * 16.0, 3)

        # Max drawdown
        eq_arr = np.array(daily_equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / peak * 100
        result.max_drawdown_pct = round(float(np.max(dd)), 2)

        # Calmar
        if result.max_drawdown_pct > 0:
            total_ret = (equity / capital - 1)
            n_years = len(ret_arr) / 252
            ann_ret = ((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0
            result.calmar = round(ann_ret / (result.max_drawdown_pct / 100), 3)

        # Returns
        total_return = (equity - capital) / capital
        n_years = len(ret_arr) / 252
        if n_years > 0:
            result.annual_return_pct = round(
                ((1 + total_return) ** (1 / n_years) - 1) * 100, 2
            )
        result.total_return_pct = round(total_return * 100, 2)

    # Source hit rates
    for src in source_total:
        total = source_total[src]
        hits = source_hits.get(src, 0)
        result.source_hit_rates[src] = round(hits / total * 100, 1) if total > 0 else 0

    # Average positions
    if daily_position_counts:
        result.avg_positions = round(np.mean(daily_position_counts), 1)

    # ── 6. Build report ────────────────────────────────────────
    lines = [
        f"\n{'='*70}",
        f"  FULL PIPELINE BACKTEST — {market}",
        f"{'='*70}",
        "",
        f"  Capital:       {capital:>12,.0f}",
        f"  Final Equity:  {equity:>12,.0f}",
        f"  Vol Target:    {annual_vol_target*100:>11.0f}%",
        f"  Symbols:       {n_symbols:>12d}",
        f"  Trading Days:  {result.n_days_traded:>12d}",
        f"  Total Trades:  {trades_count:>12d}",
        f"  Avg Positions: {result.avg_positions:>12.1f}",
        "",
        f"  {'─'*40}",
        f"  Sharpe Ratio:  {result.sharpe:>12.3f}",
        f"  Sortino Ratio: {result.sortino:>12.3f}",
        f"  Calmar Ratio:  {result.calmar:>12.3f}",
        f"  Max Drawdown:  {result.max_drawdown_pct:>11.1f}%",
        f"  Annual Return: {result.annual_return_pct:>11.1f}%",
        f"  Total Return:  {result.total_return_pct:>11.1f}%",
        "",
        f"  {'─'*40}",
        f"  Source Contribution (% of days producing a forecast):",
    ]
    for src, rate in sorted(result.source_hit_rates.items(), key=lambda x: -x[1]):
        w = next((fw.weight for fw in active_weights if fw.name == src), 0)
        lines.append(f"    {src:20s}  weight={w*100:5.1f}%  hit_rate={rate:5.1f}%")

    lines.append("")
    lines.append(f"  Transaction cost: {cost_pct*100:.2f}% round-trip")
    lines.append(f"  Regime-adaptive stop: 3.0-5.0σ  |  Inertia: 15%  |  Cooldown: 5d")
    lines.append(f"  Position sizing: Vol-target  |  IDM=dynamic  |  MaxLev={max_leverage:.1f}x")
    lines.append(f"{'='*70}\n")

    result.report = "\n".join(lines)

    if verbose:
        print(result.report)

    return {
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "calmar": result.calmar,
        "max_drawdown_pct": result.max_drawdown_pct,
        "annual_return_pct": result.annual_return_pct,
        "total_return_pct": result.total_return_pct,
        "n_symbols": result.n_symbols,
        "n_days_traded": result.n_days_traded,
        "n_trades": result.n_trades,
        "avg_positions": result.avg_positions,
        "source_hit_rates": result.source_hit_rates,
        "daily_equity": result.daily_equity,
        "report": result.report,
    }
