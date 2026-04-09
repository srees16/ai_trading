"""
Full 23-Source Pipeline Backtester.

Expanding-window daily simulation using ALL offline-capable forecast
sources through the real Carver forecast combiner and position sizer.

Sources tested (23 total, matching live CarverPipeline):
  TREND:      ewmac_8_32, ewmac_16_64, ewmac_32_128, ewmac_64_256
  TREND+:     penfold_trend, acceleration
  ADAPTIVE:   ehlers_dsp
  INTERMARKET: intermarket (Ruggiero cybernetic)
  VALUE:      carry, carver_value
  MOMENTUM:   momentum, cross_momentum
  MEAN-REV:   mean_reversion
  DERIVATIVES: oi_signal, skew_signal
  EVENT:      pead, event_driven
  FLOW/MACRO: fii_flow
  SENTIMENT:  sentiment (FinBERT proxy)
  COMPOSITE:  screener, decision_engine
  STAT-ARB:   pairs_arb
  CHANNEL:    breakout (0% weight, included for completeness)

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
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── R21A active configuration ──────────────────────────────────────────
_SAVE_FORECASTS_MODE = False  # R21a: save per-source forecasts for weight optimization
_forecast_log: list = []      # accumulator: [(day_idx, {sym: {source: val}}, {sym: next_ret})]
_R21A_REGIME_VOL = True       # R21a: regime-adaptive vol target (aggressive uptrend, defensive downtrend)
_R21A_REGIME_BOOST = 1.25     # uptrend vol multiplier
_R21A_REGIME_DEFEND = 0.55    # downtrend vol multiplier

# ── R22: Bull-Run Capital Infusion (Centurion Compounder enhancement) ──
# Optional: user injects fresh capital at confirmed bear→bull crossover
# Strategy compounds normally even WITHOUT infusion — infusion is a booster
_R22_BULL_INFUSION = False         # master toggle
_R22_INFUSION_AMOUNT = 50_000.0    # fixed ₹ amount per infusion event
_R22_INFUSION_COOLDOWN_DAYS = 200  # min trading days between infusions
_R22_BULL_CONFIRM_DAYS = 5         # require N consecutive days above SMA200+2% to confirm bull

# ── Centurion Harvest (CH) flags: profit booking in bull → reinvest in bear dips ──
# V1: Bear dip-buyer — mean-reversion signals get 3.3× vol boost in downtrend
#     (equivalent to 0.50× instead of 0.15× in live VolatilityTarget)
_HARVEST_DIP_BUYER = False
_HARVEST_MR_BEAR_VOL_MULT = 3.33   # 0.50 / 0.15 ≈ 3.33× boost for MR in downtrend

# V2: Bull profit-taker — tighter trailing stops in uptrend to book profits earlier
_HARVEST_PROFIT_TAKER = False
_HARVEST_BULL_STOP_SIGMA = 6.0     # 6σ stops in bull (vs default 10σ) — tighter exit

# V4: Bear-bottom capital injection + bull profit booking (Centurion Harvest mode)
# Simulates user adding extra funds at bear→bull crossover and booking profits in bull
_HARVEST_ENABLED = False
_HARVEST_INJECT_PCT = 0.20          # inject 20% of initial capital at bear→bull crossover
_HARVEST_BOOK_PCT = 0.15            # book 15% of gains above peak when sustained bull
_HARVEST_BULL_SUSTAIN_DAYS = 30     # require 30 days above SMA200 before booking profits
_HARVEST_MIN_GAIN_TO_BOOK = 0.10    # only book if gains > 10% above injected capital
_HARVEST_INJECT_COOLDOWN_DAYS = 200 # min 200 days between injections (avoid churn)

# ── Default pairs for pairs_arb source ────────────────────────
DEFAULT_PAIRS_US = [
    ("AAPL", "MSFT"),
    ("GOOGL", "META"),
    ("AMZN", "NVDA"),
    ("JPM", "V"),
]
DEFAULT_PAIRS_IND = [
    ("HDFCBANK.NS", "ICICIBANK.NS"),   # Large-cap banking
    ("TCS.NS", "INFY.NS"),             # IT services
    ("RELIANCE.NS", "ONGC.NS"),        # Energy
    ("SBIN.NS", "PNB.NS"),             # PSU banking
    ("SUNPHARMA.NS", "DRREDDY.NS"),    # Pharma
    ("TATASTEEL.NS", "JSWSTEEL.NS"),   # Metal
    ("MARUTI.NS", "M&M.NS"),           # Auto
    ("HINDUNILVR.NS", "ITC.NS"),       # FMCG
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
    win_rate: float = 0.0
    profit_factor: float = 0.0
    report: str = ""
    # Aronson EBTA enrichment fields
    detrended_sharpe: float = 0.0
    trimmed_sharpe: float = 0.0
    per_signal_tstats: Dict[str, float] = field(default_factory=dict)
    dm_bias_estimate: float = 0.0
    bootstrap_ci_sharpe: tuple = (0.0, 0.0)



# ── Helpers ───────────────────────────────────────────────────

def _download(sym: str, period: str, market: str,
              start: str = "", end: str = "") -> Optional[pd.DataFrame]:
    """Download OHLCV via yfinance.  Prefers start/end dates over period."""
    try:
        import yfinance as yf
        import warnings
        suffix = ".NS" if market == "IND" and "." not in sym else ""
        ticker = f"{sym}{suffix}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if start:
                df = yf.download(ticker, start=start,
                                 end=end or None,
                                 auto_adjust=True, progress=False)
            else:
                df = yf.download(ticker, period=period,
                                 auto_adjust=True, progress=False)
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
    start_date: str = "",
    end_date: str = "",
) -> Dict:
    """Run a full 13-source pipeline backtest.

    Parameters
    ----------
    tickers : list of symbols (with .NS suffix for IND if needed)
    capital : initial capital
    period : yfinance period string (1y, 2y, 5y, max). Ignored if start_date is set.
    market : "US" or "IND"
    annual_vol_target : decimal (0.20 = 20%)
    min_history : minimum bars before trading starts
    pairs : list of (leg1, leg2) for pairs_arb; None = default
    include_carry : whether to compute carry (needs yfinance dividend data)
    include_pairs : whether to compute pairs_arb
    verbose : print progress
    start_date : ISO date string e.g. "2012-01-01". Overrides period if set.
    end_date : ISO date string e.g. "2025-12-31". Empty = latest available.

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
    # R18: regime_strategy_mix import removed — equity velocity replaces regime detection
    # GAP-1: Import all 12 previously missing forecast sources
    from strategies.penfold_trend import compute_penfold_forecast_batch
    from strategies.ehlers_dsp import compute_ehlers_forecast_batch
    from strategies.ruggiero_cybernetic import compute_cybernetic_forecast_batch
    from strategies.acceleration import compute_acceleration_batch
    from strategies.carver_value import compute_value_batch
    from strategies.skew_signal import compute_skew_batch
    from services.pead_strategy import PEADStrategy
    from services.fii_flow_signal import compute_fii_forecast
    from services.event_strategy import generate_event_forecasts
    from services.sentiment_forecast import compute_sentiment_batch

    # ── Default tickers ────────────────────────────────────────
    if tickers is None:
        if market == "US":
            tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "TSLA", "JPM", "V", "UNH", "HD", "PG", "XOM", "MA",
                "JNJ",
            ]
        else:
            # FIX: Use NSE universe (respects Config.NSE_UNIVERSE_TIER)
            # instead of hardcoded 15-stock list.  Default tier is
            # Config.NSE_UNIVERSE_TIER = "BROAD" for NIFTY50+NEXT50+
            # sectoral+thematic (~800-1200 stocks).  For backtesting
            # we use DEFAULT tier (~100 NIFTY50+NEXT50) for speed;
            # override via Config.NSE_UNIVERSE_TIER for broader runs.
            try:
                from kite_connect.nse.nse_universe import get_nse_default_tickers
                raw_syms = get_nse_default_tickers()
                if raw_syms and len(raw_syms) >= 10:
                    tickers = [f"{s}.NS" for s in raw_syms]
                    if verbose:
                        print(f"  NSE universe: {len(tickers)} tickers (NIFTY50+NEXT50)")
                else:
                    raise ValueError("NSE download returned too few symbols")
            except Exception as exc:
                logger.warning("NSE universe fetch failed (%s), using fallback", exc)
                # Fallback: hardcoded NIFTY50+NEXT50 core names
                from kite_connect.core.config import INDEX_CONSTITUENTS
                n50 = INDEX_CONSTITUENTS.get("NIFTY50", [])
                nn50 = INDEX_CONSTITUENTS.get("NIFTY_NEXT50", [])
                tickers = [f"{s}.NS" for s in sorted(set(n50 + nn50))]
                if verbose:
                    print(f"  NSE fallback: {len(tickers)} tickers (hardcoded NIFTY50+NEXT50)")
    if pairs is None:
        pairs = DEFAULT_PAIRS_US if market == "US" else DEFAULT_PAIRS_IND

    # ── Download data ──────────────────────────────────────────
    if verbose:
        date_label = f"{start_date} to {end_date or 'latest'}" if start_date else period
        print(f"\n{'='*70}")
        print(f"  FULL PIPELINE BACKTEST — {market} ({len(tickers)} tickers, {date_label})")
        print(f"  Capital: {capital:,.0f}  |  Vol Target: {annual_vol_target*100:.0f}%")
        print(f"{'='*70}\n")
        print("Downloading OHLCV data...")

    ohlcv_full: Dict[str, pd.DataFrame] = {}
    for sym in tickers:
        df = _download(sym, period, market, start=start_date, end=end_date)
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

    # ── G3: Drop symbols with insufficient history ─────────────
    # With 100-stock universe, some newer listings won't have 13yr data.
    # Drop those instead of failing the entire backtest.
    _min_required = min_history + 20
    _short_syms = [s for s, df in ohlcv_full.items() if len(df) < _min_required]
    if _short_syms:
        for s in _short_syms:
            del ohlcv_full[s]
        if verbose:
            print(f"\n  Dropped {len(_short_syms)} symbols with <{_min_required} bars: "
                  f"{', '.join(_short_syms[:5])}{'...' if len(_short_syms) > 5 else ''}")
    symbols = list(ohlcv_full.keys())
    n_symbols = len(symbols)
    if n_symbols < 2:
        print("ERROR: Need at least 2 symbols with sufficient data after filtering.")
        return {"sharpe": 0, "report": "Insufficient data"}

    # ── Date-align all symbols to a common calendar ──────────
    # Use the longest-running symbol's dates as the master calendar.
    # For efficiency: precompute each symbol's offset into the master index
    # instead of reindexing (which adds NaN rows that must be dropna'd every day).
    _longest_sym = max(ohlcv_full.keys(), key=lambda s: len(ohlcv_full[s]))
    master_index = ohlcv_full[_longest_sym].index
    n_days = len(master_index)

    # Map each symbol to its starting position in master calendar
    # sym_start[sym] = master_index position where this symbol's first date falls
    _master_dates_set = {d: i for i, d in enumerate(master_index)}
    sym_start: Dict[str, int] = {}
    for sym, df in ohlcv_full.items():
        first_date = df.index[0]
        sym_start[sym] = _master_dates_set.get(first_date, 0)

    if verbose:
        print(f"\n  Symbols loaded: {n_symbols}")
        print(f"  Master bars:    {n_days}  (from {_longest_sym})")
        print(f"  Warmup period:  {min_history} bars")
        print(f"  Trading days:   {n_days - min_history}\n")

    # ── Determine available sources ────────────────────────────
    # We include all offline-capable sources; omit live-only ones.
    # The combiner auto-renormalizes weights for available sources.
    # T3-2: Match live 24-source parity for realistic backtest
    available_sources = {
        "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
        "carry", "screener", "momentum", "pead", "mean_reversion",
        "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
        "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
        "intermarket", "acceleration", "carver_value", "skew_signal",
        "sentiment", "breakout", "order_flow",
    }  # 24 sources — full parity with live pipeline
    if not include_carry:
        available_sources.discard("carry")
    if not include_pairs:
        available_sources.discard("pairs_arb")

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
    # T3-3: Realistic transaction costs (NSE delivery: STT+brokerage+GST+slippage)
    if market == "IND":
        cost_pct = 0.0033   # 33 bps round-trip (0.10% STT + 0.05% brokerage + 0.18% stamp+GST+slippage)
    else:
        cost_pct = 0.0015   # 15 bps round-trip (US zero-commission + spread slippage)

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
    entry_prices: Dict[str, float] = {}    # avg entry price per open position
    trade_pnls: List[float] = []            # round-trip trade PnL %
    source_hits: Dict[str, int] = defaultdict(int)
    source_total: Dict[str, int] = defaultdict(int)
    daily_position_counts: List[int] = []

    # V4: Capital rotation state — bear-bottom injection + bull profit booking
    _cr_prev_below_sma = False        # was equity below SMA200 yesterday?
    _cr_days_above_sma = 0            # consecutive days equity > SMA200
    _cr_last_inject_day = -9999       # day_idx of last capital injection
    _cr_total_injected = 0.0          # cumulative extra funds injected
    _cr_total_booked = 0.0            # cumulative profits booked (withdrawn)
    _cr_inject_events: list = []      # log: [(day_idx, amount, equity_before, equity_after)]
    _cr_book_events: list = []        # log: [(day_idx, amount, equity_before, equity_after)]
    _cr_base_capital = capital         # original starting capital (for injection sizing)

    # R22: Bull-run capital infusion state (Centurion Compounder)
    _r22_was_bear = False              # was equity in bear regime (< SMA200*0.98) ?
    _r22_bull_streak = 0               # consecutive days in bull regime (> SMA200*1.02)
    _r22_last_infusion_day = -9999     # day_idx of last infusion
    _r22_total_infused = 0.0           # cumulative fresh capital infused
    _r22_infusion_events: list = []    # log: [(day_idx, amount, equity_before, equity_after)]
    _r22_alert_events: list = []       # log: [(day_idx, date_str)] — when alert was generated (even if not infused)

    # FIX-DD-v2: Smooth continuous drawdown scaling (no force-liquidation)
    # Force-liquidation at bottoms caused whipsaw death spiral (-60% in bull market).
    # New approach: smooth scale-down curve, let trailing stops handle exits organically.
    peak_equity = capital
    dd_deep_days = 0          # consecutive days with DD > 25% (for gradual peak decay)
    PEAK_DECAY_GRACE_DAYS = 60  # R14 baseline
    PEAK_DECAY_RATE = 0.01      # R14 baseline: 1%/day blend

    # R13: Bear lockout REMOVED — binary exit/re-enter causes whipsaw in all variants
    # R11 (return-based) and R12 (vol-based) both destroyed equity via whipsaw.
    # R13 uses dynamic vol target (Fix C) instead — continuous, no churn.

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

    _cached_forecasts: Dict[str, Dict[str, float]] = {}  # signal cache for recompute optimization
    _cached_idm: float = 1.7  # IDM cache — recompute on recompute days only

    # OPT: Pre-load config once outside the loop
    try:
        from config import Config as _BtCfg
        max_leverage = getattr(_BtCfg, 'CARVER_MAX_LEVERAGE', 3.0)
    except Exception:
        max_leverage = 3.0
    allow_short = False  # FIX-SHORT: disabled — short Sharpe ≈ -0.01, bleeds in secular bull

    # ── Checkpoint save/resume ─────────────────────────────────
    import pickle, os, signal, traceback
    _checkpoint_path = os.environ.get(
        "CENTURION_BT_CHECKPOINT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'backtest_checkpoint.pkl'),
    )
    _start_day_idx = min_history  # default: start from beginning

    if os.path.exists(_checkpoint_path):
        try:
            with open(_checkpoint_path, 'rb') as _ckf:
                _ckpt = pickle.load(_ckf)
            # Validate checkpoint matches current run config
            if (_ckpt.get('n_symbols') == n_symbols
                    and _ckpt.get('capital') == capital
                    and _ckpt.get('n_days') == n_days):
                _start_day_idx = _ckpt['day_idx'] + 1  # resume from next day
                equity = _ckpt['equity']
                daily_equity = _ckpt['daily_equity']
                daily_returns = _ckpt['daily_returns']
                prev_positions = _ckpt['prev_positions']
                peak_prices = _ckpt['peak_prices']
                stop_levels = _ckpt['stop_levels']
                stop_cooldown = _ckpt['stop_cooldown']
                pair_states = _ckpt['pair_states']
                trades_count = _ckpt['trades_count']
                source_hits = defaultdict(int, _ckpt['source_hits'])
                source_total = defaultdict(int, _ckpt['source_total'])
                daily_position_counts = _ckpt['daily_position_counts']
                peak_equity = _ckpt['peak_equity']
                dd_deep_days = _ckpt['dd_deep_days']
                _cached_forecasts = _ckpt.get('cached_forecasts', {})
                _cached_idm = _ckpt.get('cached_idm', 1.7)
                trade_pnls = _ckpt.get('trade_pnls', [])
                entry_prices = _ckpt.get('entry_prices', {})
                if verbose:
                    _resume_day = _start_day_idx - min_history
                    print(f"\n  Resuming from checkpoint: Day {_resume_day}/{n_days - min_history} "
                          f"equity={equity:,.0f}", flush=True)
            else:
                if verbose:
                    print("\n  Checkpoint found but config mismatch — starting fresh", flush=True)
                os.remove(_checkpoint_path)
        except Exception as e:
            if verbose:
                print(f"\n  Checkpoint load failed ({e}) — starting fresh", flush=True)
            if os.path.exists(_checkpoint_path):
                os.remove(_checkpoint_path)

    # ── Signal handler: save checkpoint on SIGINT/SIGTERM ──────
    _graceful_exit_requested = False

    def _graceful_shutdown(signum, frame):
        nonlocal _graceful_exit_requested
        _graceful_exit_requested = True
        if verbose:
            print(f"\n  Signal {signum} received — will save checkpoint and exit after current day", flush=True)

    _prev_sigint = signal.signal(signal.SIGINT, _graceful_shutdown)
    _prev_sigterm = signal.signal(signal.SIGTERM, _graceful_shutdown)

    _consecutive_day_errors = 0
    _MAX_CONSECUTIVE_ERRORS = 10  # abort if 10+ days crash in a row

    for day_idx in range(_start_day_idx, n_days):
      try:
        day_pnl = 0.0

        # Build OHLCV slices up to current day (views, not copies)
        ohlcv_slice: Dict[str, pd.DataFrame] = {}
        # T3-1: Extract current simulation date for look-ahead bias prevention
        current_date = master_index[day_idx]
        if hasattr(current_date, 'date'):
            current_date = current_date.date()
        for sym, df in ohlcv_full.items():
            # Compute how many bars this symbol has up to current master day_idx
            local_len = day_idx - sym_start.get(sym, 0) + 1
            actual_len = len(df)
            use_len = min(local_len, actual_len)
            if use_len >= 50:  # need at least 50 valid bars
                ohlcv_slice[sym] = df.iloc[:use_len]

        # ── 1. Mark-to-market existing positions ───────────────
        for sym in symbols:
            prev_qty = prev_positions.get(sym, 0)
            if prev_qty == 0:
                continue
            if sym not in ohlcv_slice:
                continue  # no data for this symbol yet
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
                        if sym in entry_prices and entry_prices[sym] > 0:
                            trade_pnls.append((exit_price - entry_prices[sym]) / entry_prices[sym] * 100)
                            del entry_prices[sym]
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
                        if sym in entry_prices and entry_prices[sym] > 0:
                            trade_pnls.append((entry_prices[sym] - exit_price) / entry_prices[sym] * 100)
                            del entry_prices[sym]
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
        # Optimization: expensive signals recomputed every RECOMPUTE_FREQ days,
        # cached between. EWMAC/momentum change slowly over 5 days.
        _RECOMPUTE_FREQ = 5  # R14 baseline: recompute every 5 days
        _trading_day = day_idx - min_history
        _recompute = (_trading_day % _RECOMPUTE_FREQ == 0)

        if not _recompute:
            all_forecasts = {sym: dict(fc) for sym, fc in _cached_forecasts.items()}
        else:  # ── full signal recompute (serial) ──
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

            # ── 2i. Penfold Trend (Turtle + ATR + Retracement + Dow filter) ──
            try:
                penfold_fc = compute_penfold_forecast_batch(ohlcv_slice)
                for sym, fc in penfold_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["penfold_trend"] = fc
            except Exception as e:
                logger.debug("Penfold failed at day %d: %s", day_idx, e)

            # ── 2j. Ehlers DSP (Fisher, MAMA/FAMA, SuperSmoother) ──
            try:
                ehlers_fc = compute_ehlers_forecast_batch(ohlcv_slice)
                for sym, fc in ehlers_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["ehlers_dsp"] = fc
            except Exception as e:
                logger.debug("Ehlers DSP failed at day %d: %s", day_idx, e)

            # ── 2k. Ruggiero Cybernetic (intermarket + seasonal + multi-TF) ──
            try:
                intermarket_fc = compute_cybernetic_forecast_batch(ohlcv_slice)
                for sym, fc in intermarket_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["intermarket"] = fc
            except Exception as e:
                logger.debug("Intermarket failed at day %d: %s", day_idx, e)

            # ── 2l. Acceleration (AFTS S23: rate-of-change of EWMAC) ──
            try:
                accel_fc = compute_acceleration_batch(ohlcv_slice)
                for sym, fc in accel_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["acceleration"] = fc
            except Exception as e:
                logger.debug("Acceleration failed at day %d: %s", day_idx, e)

            # ── 2m. Carver Value (AFTS S22: 5-year mean reversion) ──
            try:
                value_fc = compute_value_batch(ohlcv_slice)
                for sym, fc in value_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["carver_value"] = fc
            except Exception as e:
                logger.debug("Carver value failed at day %d: %s", day_idx, e)

            # ── 2n. Skew Signal (AFTS S24: realized skew risk premium) ──
            try:
                skew_fc = compute_skew_batch(ohlcv_slice)
                for sym, fc in skew_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["skew_signal"] = fc
            except Exception as e:
                logger.debug("Skew signal failed at day %d: %s", day_idx, e)

            # ── 2o. PEAD (Post-Earnings Announcement Drift) ──
            # T3-1: pass as_of_date to prevent look-ahead bias
            try:
                pead = PEADStrategy()
                for sym in symbols:
                    if sym not in all_forecasts:
                        continue
                    try:
                        pead_fc = pead.get_forecast(sym, as_of_date=current_date)
                    except TypeError:
                        pead_fc = pead.get_forecast(sym)
                    if pead_fc is not None and np.isfinite(pead_fc):
                        all_forecasts[sym]["pead"] = pead_fc
            except Exception as e:
                logger.debug("PEAD failed at day %d: %s", day_idx, e)

            # ── 2p. FII Flow (institutional net buy/sell) ──
            try:
                fii_fc = compute_fii_forecast()
                if fii_fc is not None and np.isfinite(fii_fc):
                    for sym in symbols:
                        if sym in all_forecasts:
                            all_forecasts[sym]["fii_flow"] = fii_fc
            except Exception as e:
                logger.debug("FII flow failed at day %d: %s", day_idx, e)

            # ── 2q. Event-Driven (earnings/RBI/expiry/rebalance) ──
            # T3-1: pass as_of_date to prevent look-ahead bias
            try:
                try:
                    event_fcs = generate_event_forecasts(symbols, as_of_date=current_date)
                except TypeError:
                    event_fcs = generate_event_forecasts(symbols)
                if isinstance(event_fcs, dict):
                    for sym, fc_val in event_fcs.items():
                        if sym in all_forecasts and fc_val is not None:
                            f_v = fc_val.forecast if hasattr(fc_val, 'forecast') else fc_val
                            if np.isfinite(f_v):
                                all_forecasts[sym]["event_driven"] = f_v
            except Exception as e:
                logger.debug("Event-driven failed at day %d: %s", day_idx, e)

            # ── 2r. Sentiment (FinBERT news-based) ──
            try:
                sent_fc = compute_sentiment_batch(symbols)
                for sym, fc in sent_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["sentiment"] = fc
            except Exception as e:
                logger.debug("Sentiment failed at day %d: %s", day_idx, e)

            # ── 2s. Screener + Decision Engine (composite technical/fundamental) ──
            # In backtest mode these use simplified technical proxies
            # (the live pipeline uses NSEScreener + IntegratedScorer which need real-time data)
            for sym, df in ohlcv_slice.items():
                if sym not in all_forecasts:
                    continue
                c = df["Close"]
                if hasattr(c, "squeeze"):
                    c = c.squeeze()
                if len(c) < 50:
                    continue
                # Screener proxy: RSI(14) + 50-day MA slope composite
                delta = c.diff()
                gain = delta.where(delta > 0, 0.0).ewm(span=14).mean()
                loss = (-delta).where(delta < 0, 0.0).ewm(span=14).mean()
                rs = gain / (loss + 1e-10)
                rsi = float(100 - (100 / (1 + rs.iloc[-1])))
                ma50 = c.rolling(50).mean()
                ma_slope = float((ma50.iloc[-1] - ma50.iloc[-5]) / (ma50.iloc[-5] + 1e-10)) * 100
                screener_fc = ((rsi - 50) / 5.0) + ma_slope * 2.0
                screener_fc = max(-20.0, min(20.0, screener_fc))
                all_forecasts[sym]["screener"] = screener_fc
                # Decision engine proxy: blend of technical + fundamental-ish signals
                # Uses available forecast average as a simple proxy
                existing_fcs = [v for v in all_forecasts[sym].values() if np.isfinite(v)]
                if existing_fcs:
                    de_fc = np.mean(existing_fcs) * 0.3  # dampened consensus
                    de_fc = max(-20.0, min(20.0, de_fc))
                    all_forecasts[sym]["decision_engine"] = de_fc

            # ── 2t. Order flow (OBV + CVD + MFI microstructure) — T3-2 ──
            try:
                from strategies.order_flow import compute_order_flow_forecasts_batch
                of_fc = compute_order_flow_forecasts_batch(ohlcv_slice)
                for sym, fc in of_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["order_flow"] = fc
            except Exception as e:
                logger.debug("Order flow failed at day %d: %s", day_idx, e)


            _cached_forecasts = {sym: dict(fc) for sym, fc in all_forecasts.items()}
        # ── 3. Combine forecasts + size positions ──────────────
        # R12: SIMPLEST POSSIBLE SYSTEM — strip all whipsaw sources
        # Meta-analysis of R1-R11: every scaling mechanism (DD, regime, warmup)
        # either creates death spiral or amplifies whipsaw losses.
        # R12 approach: STAY INVESTED through corrections, exit ONLY on panic vol.
        peak_equity = max(peak_equity, equity)
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        # R14: Gradual peak decay — prevents unreachable peak trapping vol target
        # at low levels forever, but does NOT snap DD to zero (which re-enabled
        # full risk into a continuing crash every 30 days in R13).
        if current_dd >= 0.25:
            dd_deep_days += 1
            if dd_deep_days >= PEAK_DECAY_GRACE_DAYS:
                # Slowly blend peak toward current equity (1%/day)
                peak_equity = peak_equity * (1.0 - PEAK_DECAY_RATE) + equity * PEAK_DECAY_RATE
                current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        else:
            dd_deep_days = 0

        # R13: NO DD SCALING — replaced with dynamic vol target (Fix C)
        dd_scale = 1.0

        # R14: 5-tier step function — dynamic vol target based on DD depth
        if current_dd < 0.10:
            annual_vol_target = 0.75   # Full risk — no DD
        elif current_dd < 0.20:
            annual_vol_target = 0.65   # Mild pullback — slight reduction
        elif current_dd < 0.30:
            annual_vol_target = 0.55   # Moderate DD — meaningful reduction
        elif current_dd < 0.40:
            annual_vol_target = 0.45   # Severe DD — significant reduction
        else:
            annual_vol_target = 0.40   # Extreme DD — floor (never go to zero)

        # FIX-FLOOR: Use actual equity for daily target
        sizing_equity = max(equity, capital * 0.10)  # 10% ruin floor
        dynamic_daily_target = sizing_equity * annual_vol_target / 16.0

        # R21a: Regime-adaptive vol target
        # Aggressive in sustained uptrends, defensive in downtrends
        if _R21A_REGIME_VOL and len(daily_equity) >= 200:
            _eq_sma200 = sum(daily_equity[-200:]) / 200.0
            if equity > _eq_sma200 * 1.02:
                dynamic_daily_target *= _R21A_REGIME_BOOST  # uptrend: aggressive
            elif equity < _eq_sma200 * 0.98:
                dynamic_daily_target *= _R21A_REGIME_DEFEND  # downtrend: defensive

        # ── Centurion Harvest: inject at bear→bull, book in sustained bull ──
        if _HARVEST_ENABLED and len(daily_equity) >= 200:
            _cr_sma200 = sum(daily_equity[-200:]) / 200.0
            _cr_currently_below = (equity < _cr_sma200 * 0.98)
            _cr_currently_above = (equity > _cr_sma200 * 1.02)
            _trading_day_num = day_idx - min_history

            # Bear→Bull crossover: equity was below SMA200, now above → INJECT
            if _cr_prev_below_sma and _cr_currently_above:
                _days_since_inject = day_idx - _cr_last_inject_day
                if _days_since_inject >= _HARVEST_INJECT_COOLDOWN_DAYS:
                    _inject_amount = _cr_base_capital * _HARVEST_INJECT_PCT
                    _eq_before = equity
                    equity += _inject_amount
                    capital += _inject_amount  # increase base capital for sizing
                    _cr_total_injected += _inject_amount
                    _cr_last_inject_day = day_idx
                    _cr_inject_events.append((_trading_day_num, _inject_amount, _eq_before, equity))
                    # Recalculate sizing targets with new capital
                    sizing_equity = max(equity, capital * 0.10)
                    dynamic_daily_target = sizing_equity * annual_vol_target / 16.0
                    if verbose:
                        print(f"  CAPITAL INJECT Day {_trading_day_num}: "
                              f"+₹{_inject_amount:,.0f} ({_HARVEST_INJECT_PCT:.0%} of base) | "
                              f"equity ₹{_eq_before:,.0f} → ₹{equity:,.0f} | "
                              f"total injected: ₹{_cr_total_injected:,.0f}", flush=True)

            # Sustained bull: equity above SMA200 for N days → BOOK PROFITS
            if _cr_currently_above:
                _cr_days_above_sma += 1
            else:
                _cr_days_above_sma = 0

            if (_cr_days_above_sma >= _HARVEST_BULL_SUSTAIN_DAYS
                    and _cr_days_above_sma % _HARVEST_BULL_SUSTAIN_DAYS == 0):
                # Book profits = portion of gains above total invested capital
                _total_invested = _cr_base_capital  # base includes all injections
                _gain = equity - _total_invested
                if _gain > _total_invested * _HARVEST_MIN_GAIN_TO_BOOK:
                    _book_amount = _gain * _HARVEST_BOOK_PCT
                    _eq_before = equity
                    equity -= _book_amount
                    _cr_total_booked += _book_amount
                    _cr_book_events.append((_trading_day_num, _book_amount, _eq_before, equity))
                    # Recalculate sizing targets
                    sizing_equity = max(equity, capital * 0.10)
                    dynamic_daily_target = sizing_equity * annual_vol_target / 16.0
                    if verbose:
                        print(f"  PROFIT BOOK Day {_trading_day_num}: "
                              f"-₹{_book_amount:,.0f} ({_HARVEST_BOOK_PCT:.0%} of ₹{_gain:,.0f} gains) | "
                              f"equity ₹{_eq_before:,.0f} → ₹{equity:,.0f} | "
                              f"total booked: ₹{_cr_total_booked:,.0f}", flush=True)

            _cr_prev_below_sma = _cr_currently_below

        # ── R22: Bull-Run Capital Infusion (Centurion Compounder) ──
        # Detects confirmed bear→bull regime transition and either:
        #   (a) infuses fresh capital if _R22_BULL_INFUSION=True, or
        #   (b) logs an alert event so the UI/email can notify the user
        # Compounding continues normally regardless of infusion.
        if len(daily_equity) >= 200:
            _r22_sma200 = sum(daily_equity[-200:]) / 200.0
            _r22_in_bear = (equity < _r22_sma200 * 0.98)
            _r22_in_bull = (equity > _r22_sma200 * 1.02)
            _trading_day_num_r22 = day_idx - min_history

            if _r22_in_bear:
                _r22_was_bear = True
                _r22_bull_streak = 0
            elif _r22_in_bull:
                _r22_bull_streak += 1
            else:
                _r22_bull_streak = 0  # neutral zone → reset

            # Confirmed bull: was in bear, now N consecutive bull days
            if (_r22_was_bear and _r22_bull_streak >= _R22_BULL_CONFIRM_DAYS
                    and _r22_bull_streak == _R22_BULL_CONFIRM_DAYS):
                _r22_was_bear = False  # reset so we don't re-trigger until next bear
                _days_since_r22 = day_idx - _r22_last_infusion_day

                # Always log alert (for UI/email notification)
                _r22_date_str = ""
                try:
                    _r22_date_str = ohlcv_slice[symbols[0]]["Close"].index[day_idx].strftime("%Y-%m-%d")
                except Exception:
                    _r22_date_str = f"Day {_trading_day_num_r22}"
                _r22_alert_events.append((_trading_day_num_r22, _r22_date_str))

                if verbose:
                    print(f"  ★ BULL CONFIRMED Day {_trading_day_num_r22} ({_r22_date_str}): "
                          f"equity ₹{equity:,.0f} > SMA200 ₹{_r22_sma200:,.0f} "
                          f"for {_R22_BULL_CONFIRM_DAYS} consecutive days", flush=True)

                # Infuse capital if enabled and cooldown satisfied
                if (_R22_BULL_INFUSION
                        and _days_since_r22 >= _R22_INFUSION_COOLDOWN_DAYS):
                    _r22_eq_before = equity
                    equity += _R22_INFUSION_AMOUNT
                    capital += _R22_INFUSION_AMOUNT
                    _r22_total_infused += _R22_INFUSION_AMOUNT
                    _r22_last_infusion_day = day_idx
                    _r22_infusion_events.append((
                        _trading_day_num_r22, _R22_INFUSION_AMOUNT,
                        _r22_eq_before, equity,
                    ))
                    # Recalculate sizing with new capital
                    sizing_equity = max(equity, capital * 0.10)
                    dynamic_daily_target = sizing_equity * annual_vol_target / 16.0
                    if verbose:
                        print(f"  💰 CAPITAL INFUSION Day {_trading_day_num_r22}: "
                              f"+₹{_R22_INFUSION_AMOUNT:,.0f} | "
                              f"equity ₹{_r22_eq_before:,.0f} → ₹{equity:,.0f} | "
                              f"total infused: ₹{_r22_total_infused:,.0f}", flush=True)

        # Tick down stop cooldowns
        for sym in list(stop_cooldown.keys()):
            stop_cooldown[sym] -= 1
            if stop_cooldown[sym] <= 0:
                del stop_cooldown[sym]

        # ── G3: Top-N conviction filter ────────────────────────
        # R13 Fix B: ADAPTIVE position count based on forecast consensus
        # When signals are strong (avg forecast > 8), deploy widely (15 positions).
        # When signals are mixed (avg 3-8), concentrate (10 positions).
        # When signals are weak (avg < 3), hold few (6 positions).
        # This prevents over-deployment in low-conviction regimes without binary exit.
        
        # Pre-compute combined forecasts for ALL symbols, then rank
        _all_combined: Dict[str, float] = {}
        for sym, fc_dict in all_forecasts.items():
            if not fc_dict:
                continue
            combined = combine_forecasts(sym, fc_dict, active_weights)
            fc_val = combined.combined_forecast
            _all_combined[sym] = fc_val

        # R21a: Save per-source forecasts + close prices for weight optimization
        if _SAVE_FORECASTS_MODE:
            _fc_snap = {}
            _px_snap = {}
            _vol_snap = {}
            for sym, fc_dict in all_forecasts.items():
                if fc_dict:
                    _fc_snap[sym] = dict(fc_dict)
                if sym in ohlcv_slice:
                    c = ohlcv_slice[sym]["Close"]
                    if hasattr(c, "squeeze"):
                        c = c.squeeze()
                    _c = c.dropna()
                    if len(_c) > 0:
                        _px_snap[sym] = float(_c.iloc[-1])
                        _dpv = daily_price_volatility(_c)
                        _vol_snap[sym] = float(_dpv) if np.isfinite(_dpv) else 0.02
            _forecast_log.append((day_idx, str(current_date), _fc_snap, _px_snap, _vol_snap))

        # Adaptive position count based on top-15 average forecast strength
        _ranked_for_count = sorted(_all_combined.values(), key=lambda x: abs(x), reverse=True)
        _top15_avg = np.mean([abs(f) for f in _ranked_for_count[:15]]) if _ranked_for_count else 0.0
        if _top15_avg > 8.0:
            MAX_POSITIONS = 15  # Strong consensus — deploy widely
        elif _top15_avg > 5.0:
            MAX_POSITIONS = 12  # Moderate consensus
        elif _top15_avg > 3.0:
            MAX_POSITIONS = 8   # Weak consensus — concentrate
        else:
            MAX_POSITIONS = 5   # Very weak — minimal deployment

        MAX_HOLD_GRACE = MAX_POSITIONS + 7  # Grace zone scales with positions
        # Rank by absolute forecast strength
        _ranked = sorted(_all_combined.items(), key=lambda x: abs(x[1]), reverse=True)
        _top_syms = set(s for s, _ in _ranked[:MAX_POSITIONS])      # top-20 → eligible for new entries
        _grace_syms = set(s for s, _ in _ranked[:MAX_HOLD_GRACE])   # top-30 → held positions stay

        # R8 CRITICAL FIX: Dynamic weight_per_sym based on ACTUAL investable count
        # Bug in R4-R7: weight_per_sym = 1/MAX_POSITIONS = 1/20 = 5% always.
        # When only 5 stocks have strong signals, 75% of capital sat idle.
        # Fix: count stocks with |forecast| > 2.0 in top set, use that for weight.
        # Floor at 5 (prevent over-concentration), cap at MAX_POSITIONS.
        _investable = [s for s in _top_syms if abs(_all_combined.get(s, 0)) > 2.0]
        n_investable = max(5, min(len(_investable), MAX_POSITIONS))
        weight_per_sym = 1.0 / n_investable

        # IDM: T3-6 — compute dynamically from instrument return correlations
        # IDM = 1/sqrt(avg_pairwise_correlation) for >6 instruments
        # OPT: Only recompute on recompute days (correlation changes slowly)
        if _recompute and n_investable >= 6 and day_idx >= min_history + 60:
            _rets_for_idm = []
            for sym, df in ohlcv_slice.items():
                if sym not in all_forecasts or not all_forecasts[sym]:
                    continue
                c = df["Close"]
                if hasattr(c, "squeeze"):
                    c = c.squeeze()
                if len(c) >= 60:
                    _rets_for_idm.append(c.pct_change().iloc[-60:].rename(sym))
            if len(_rets_for_idm) >= 4:
                _rets_df = pd.concat(_rets_for_idm, axis=1).dropna()
                if len(_rets_df) >= 30:
                    _corr_mat = _rets_df.corr()
                    _n = len(_corr_mat)
                    _off_diag = (_corr_mat.values.sum() - _n) / max(_n * (_n - 1), 1)
                    _avg_corr = max(0.05, min(0.95, _off_diag))
                    _cached_idm = min(2.5, 1.0 / np.sqrt(_avg_corr))
                else:
                    _cached_idm = 1.7
            else:
                _cached_idm = 1.7
        elif n_investable < 6:
            _cached_idm = 1.5
        idm = _cached_idm

        active_count = 0
        for sym, fc_dict in all_forecasts.items():
            if not fc_dict:
                continue

            # Track source hit rates
            for src in fc_dict:
                source_hits[src] += 1
            for fw in active_weights:
                source_total[fw.name] += 1

            # R7: Soft grace-zone position management
            # - Top-20: eligible for NEW entries and re-sizing
            # - Top-21 to top-30: HOLD existing positions, but no new entries
            # - Below top-30: force exit (truly low-conviction stocks)
            # This prevents the R6 churn (hard exit at 15) and R4/R5 leak (no exit at all)
            forecast = _all_combined.get(sym, 0.0)
            is_held = prev_positions.get(sym, 0) != 0
            if sym not in _top_syms:
                if sym in _grace_syms and is_held:
                    # Grace zone: keep position, don't resize, let existing stops/forecasts manage
                    continue
                elif is_held:
                    # Below grace zone: force exit
                    if sym in ohlcv_slice:
                        _exit_c = ohlcv_slice[sym]["Close"]
                        if hasattr(_exit_c, "squeeze"):
                            _exit_c = _exit_c.squeeze()
                        _exit_price = float(_exit_c.iloc[-1])
                        _exit_qty = prev_positions[sym]
                        day_pnl -= abs(_exit_qty) * _exit_price * cost_pct
                        trades_count += 1
                    prev_positions[sym] = 0
                    peak_prices.pop(sym, None)
                    stop_levels.pop(sym, None)
                continue

            # Position sizing (forecast already computed in conviction filter above)
            if sym not in ohlcv_slice:
                continue
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
                # V1: Bear dip-buyer — boost vol target for mean-reversion signals in downtrend
                _sym_daily_target = dynamic_daily_target
                if _HARVEST_DIP_BUYER and len(daily_equity) >= 200:
                    _eq_sma = sum(daily_equity[-200:]) / 200.0
                    if equity < _eq_sma * 0.98:  # downtrend
                        _mr_fc = fc_dict.get("mean_reversion", 0.0)
                        _max_src_fc = max(abs(v) for v in fc_dict.values()) if fc_dict else 0.0
                        if abs(_mr_fc) > 0 and abs(_mr_fc) >= _max_src_fc * 0.5:
                            # MR is a significant contributor — boost vol target
                            _sym_daily_target *= _HARVEST_MR_BEAR_VOL_MULT

                vol_scalar = _sym_daily_target / ivv
                position = (forecast / 10.0) * vol_scalar * weight_per_sym * idm

                # R13: NO REGIME SCALING — regime-agnostic position sizing
                # Dynamic vol target (Fix C) handles risk continuously.
                # No binary regime multipliers — those caused R4-R11 whipsaw.

                target_qty = round(position)

                # Cap at max leverage per-symbol (dynamic based on investable count)
                max_notional = abs(equity) * max_leverage / n_investable
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
                    # R14 baseline: Inertia 10%
                    _inertia_pct = 0.10
                    if abs(prev_qty) > 0 and delta / abs(prev_qty) < _inertia_pct:
                        target_qty = prev_qty
                    else:
                        cost = delta * price * cost_pct
                        day_pnl -= cost
                        trades_count += 1

                # ── Per-trade tracking (round-trip PnL) ───────
                if prev_qty == 0 and target_qty != 0:
                    # New position opened — record entry price
                    entry_prices[sym] = price
                elif prev_qty != 0 and target_qty == 0:
                    # Position fully closed — log round-trip PnL
                    if sym in entry_prices and entry_prices[sym] > 0:
                        ep = entry_prices.pop(sym)
                        if prev_qty > 0:
                            trade_pnls.append((price - ep) / ep * 100)
                        else:
                            trade_pnls.append((ep - price) / ep * 100)
                elif prev_qty != 0 and target_qty != 0 and abs(target_qty) != abs(prev_qty):
                    # Position resized — update VWAP entry
                    if sym in entry_prices and abs(target_qty) > abs(prev_qty):
                        added = abs(target_qty) - abs(prev_qty)
                        entry_prices[sym] = (entry_prices[sym] * abs(prev_qty) + price * added) / abs(target_qty)

                prev_positions[sym] = target_qty

                # R14 baseline: 10σ stops
                stop_sigma = 10.0

                # V2: Bull profit-taker — tighter stops in uptrend to book profits
                if _HARVEST_PROFIT_TAKER and len(daily_equity) >= 200:
                    _eq_sma = sum(daily_equity[-200:]) / 200.0
                    if equity > _eq_sma * 1.02:  # uptrend
                        stop_sigma = min(stop_sigma, _HARVEST_BULL_STOP_SIGMA)

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

        # FIX-LEV: Portfolio-wide leverage cap enforcement
        # Sum of all |position × price| must not exceed equity × max_leverage
        total_exposure = 0.0
        for sym, qty in prev_positions.items():
            if qty == 0 or sym not in ohlcv_slice:
                continue
            c = ohlcv_slice[sym]["Close"]
            if hasattr(c, "squeeze"):
                c = c.squeeze()
            p = float(c.dropna().iloc[-1]) if len(c.dropna()) > 0 else 0
            total_exposure += abs(qty) * p
        max_total_exposure = max(equity, capital * 0.10) * max_leverage
        if total_exposure > max_total_exposure and total_exposure > 0:
            scale_down = max_total_exposure / total_exposure
            for sym in list(prev_positions.keys()):
                if prev_positions[sym] != 0:
                    prev_positions[sym] = round(prev_positions[sym] * scale_down)

        # ── 4. Update equity ───────────────────────────────────
        if not np.isfinite(day_pnl):
            day_pnl = 0.0  # NaN guard: skip corrupted days
        equity += day_pnl
        daily_returns.append(day_pnl / max(daily_equity[-1], 1))
        daily_equity.append(equity)

        _consecutive_day_errors = 0  # reset on successful day

        # Print progress every 50 days
        if verbose and (day_idx - min_history) % 50 == 0:
            d = day_idx - min_history
            total_d = n_days - min_history
            ret_so_far = (equity / capital - 1) * 100
            _line = f"  Day {d:4d}/{total_d}  equity={equity:,.0f}  ret={ret_so_far:+.1f}%  positions={active_count}"
            print(_line, flush=True)

            # Save checkpoint every 50-day block
            try:
                _ckpt_data = {
                    'day_idx': day_idx,
                    'n_symbols': n_symbols, 'capital': capital, 'n_days': n_days,
                    'equity': equity,
                    'daily_equity': daily_equity,
                    'daily_returns': daily_returns,
                    'prev_positions': dict(prev_positions),
                    'peak_prices': dict(peak_prices),
                    'stop_levels': dict(stop_levels),
                    'stop_cooldown': dict(stop_cooldown),
                    'pair_states': dict(pair_states),
                    'trades_count': trades_count,
                    'source_hits': dict(source_hits),
                    'source_total': dict(source_total),
                    'daily_position_counts': list(daily_position_counts),
                    'peak_equity': peak_equity,
                    'dd_deep_days': dd_deep_days,
                    'cached_forecasts': {s: dict(f) for s, f in _cached_forecasts.items()},
                    'cached_idm': _cached_idm,
                    'trade_pnls': list(trade_pnls),
                    'entry_prices': dict(entry_prices),
                }
                with open(_checkpoint_path, 'wb') as _ckf:
                    pickle.dump(_ckpt_data, _ckf, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass  # non-fatal — checkpoint save failure shouldn't stop simulation

        # ── Graceful exit: save checkpoint and break ───────────
        if _graceful_exit_requested:
            if verbose:
                d = day_idx - min_history
                print(f"\n  Graceful exit at Day {d}/{n_days - min_history} — saving checkpoint...", flush=True)
            try:
                _ckpt_data = {
                    'day_idx': day_idx,
                    'n_symbols': n_symbols, 'capital': capital, 'n_days': n_days,
                    'equity': equity,
                    'daily_equity': daily_equity,
                    'daily_returns': daily_returns,
                    'prev_positions': dict(prev_positions),
                    'peak_prices': dict(peak_prices),
                    'stop_levels': dict(stop_levels),
                    'stop_cooldown': dict(stop_cooldown),
                    'pair_states': dict(pair_states),
                    'trades_count': trades_count,
                    'source_hits': dict(source_hits),
                    'source_total': dict(source_total),
                    'daily_position_counts': list(daily_position_counts),
                    'peak_equity': peak_equity,
                    'dd_deep_days': dd_deep_days,
                    'cached_forecasts': {s: dict(f) for s, f in _cached_forecasts.items()},
                    'cached_idm': _cached_idm,
                    'trade_pnls': list(trade_pnls),
                    'entry_prices': dict(entry_prices),
                }
                with open(_checkpoint_path, 'wb') as _ckf:
                    pickle.dump(_ckpt_data, _ckf, protocol=pickle.HIGHEST_PROTOCOL)
                if verbose:
                    print(f"  Checkpoint saved. Re-run to resume from Day {d+1}.", flush=True)
            except Exception as _save_err:
                if verbose:
                    print(f"  WARNING: checkpoint save failed on exit: {_save_err}", flush=True)
            break

      except Exception as _day_err:
        # Per-day fault tolerance: log error, skip day (pnl=0), continue
        _consecutive_day_errors += 1
        d = day_idx - min_history
        if verbose:
            print(f"  WARNING: Day {d} error ({type(_day_err).__name__}: {_day_err}) — skipping day", flush=True)
            traceback.print_exc()
        # Record zero PnL so equity array stays aligned with day_idx
        # Guard: only append if this day hasn't already been appended (partial execution)
        _expected_len = day_idx - min_history + 2  # +1 for initial, +1 for current
        if len(daily_equity) < _expected_len:
            daily_returns.append(0.0)
            daily_equity.append(equity)
        if _consecutive_day_errors >= _MAX_CONSECUTIVE_ERRORS:
            if verbose:
                print(f"\n  FATAL: {_MAX_CONSECUTIVE_ERRORS} consecutive day errors — aborting simulation", flush=True)
            # Save emergency checkpoint before aborting
            try:
                _ckpt_data = {
                    'day_idx': day_idx,
                    'n_symbols': n_symbols, 'capital': capital, 'n_days': n_days,
                    'equity': equity,
                    'daily_equity': daily_equity,
                    'daily_returns': daily_returns,
                    'prev_positions': dict(prev_positions),
                    'peak_prices': dict(peak_prices),
                    'stop_levels': dict(stop_levels),
                    'stop_cooldown': dict(stop_cooldown),
                    'pair_states': dict(pair_states),
                    'trades_count': trades_count,
                    'source_hits': dict(source_hits),
                    'source_total': dict(source_total),
                    'daily_position_counts': list(daily_position_counts),
                    'peak_equity': peak_equity,
                    'dd_deep_days': dd_deep_days,
                    'cached_forecasts': {s: dict(f) for s, f in _cached_forecasts.items()},
                    'cached_idm': _cached_idm,
                    'trade_pnls': list(trade_pnls),
                    'entry_prices': dict(entry_prices),
                }
                with open(_checkpoint_path, 'wb') as _ckf:
                    pickle.dump(_ckpt_data, _ckf, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass
            break

    # Restore original signal handlers
    signal.signal(signal.SIGINT, _prev_sigint)
    signal.signal(signal.SIGTERM, _prev_sigterm)

    # Clean up checkpoint on successful completion
    if os.path.exists(_checkpoint_path):
        os.remove(_checkpoint_path)

    if verbose:
        print(f"  Simulation complete: {n_days - min_history} trading days")

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

    # Win Rate & Profit Factor (from round-trip trade PnLs)
    if trade_pnls:
        _tp = np.array(trade_pnls)
        _wins = _tp[_tp > 0]
        _losses = _tp[_tp < 0]
        result.win_rate = round(len(_wins) / len(_tp) * 100, 1)
        _gross_profit = float(np.sum(_wins)) if len(_wins) > 0 else 0.0
        _gross_loss = float(np.abs(np.sum(_losses))) if len(_losses) > 0 else 0.0
        result.profit_factor = round(_gross_profit / _gross_loss, 2) if _gross_loss > 0 else 99.99

    # ── 5b. Aronson EBTA enrichment metrics ─────────────────
    try:
        from services.aronson_validator import (
            detrend_returns, trimmed_sharpe as _trimmed_sharpe,
            compute_signal_tstat, estimate_data_mining_bias,
        )
        _ret_series = pd.Series(daily_returns)

        # Detrended Sharpe: subtract rolling mean to isolate timing skill
        _detrended = detrend_returns(_ret_series, window=252)
        _dt_arr = _detrended.dropna().values
        if len(_dt_arr) > 10:
            _dt_mean = float(np.mean(_dt_arr))
            _dt_std = float(np.std(_dt_arr, ddof=1))
            if _dt_std > 0:
                result.detrended_sharpe = round(_dt_mean / _dt_std * 16.0, 3)

        # Trimmed Sharpe (5% winsorized)
        result.trimmed_sharpe = round(_trimmed_sharpe(ret_arr, trim_pct=0.05), 3)

        # Per-signal t-statistics (from source_daily_returns if available)
        if source_total:
            n_sigs = len(source_total)
            best_std = 0.0
            for src in source_total:
                total = source_total[src]
                hits = source_hits.get(src, 0)
                if total > 10:
                    # Approximate signal returns: hit contributes +mean, miss contributes -mean
                    _hit_r = hits / total if total > 0 else 0.5
                    _sim_rets = np.array([1.0] * hits + [-1.0] * (total - hits))
                    _ts, _pv = compute_signal_tstat(_sim_rets)
                    result.per_signal_tstats[src] = round(_ts, 3)
                    _src_std = float(np.std(_sim_rets, ddof=1)) if len(_sim_rets) > 1 else 1.0
                    if _src_std > best_std:
                        best_std = _src_std
            # DM bias estimate
            if n_sigs >= 2 and best_std > 0:
                result.dm_bias_estimate = round(
                    estimate_data_mining_bias(n_sigs, best_std) * 100, 2  # as percentage
                )

        # Bootstrap CI for Sharpe (quick: 1000 resamples)
        if len(ret_arr) > 30:
            rng = np.random.RandomState(42)
            boot_sharpes = []
            for _ in range(1000):
                idx = rng.randint(0, len(ret_arr), size=len(ret_arr))
                _b = ret_arr[idx]
                _bm = float(np.mean(_b))
                _bs = float(np.std(_b, ddof=1))
                if _bs > 0:
                    boot_sharpes.append(_bm / _bs * 16.0)
            if boot_sharpes:
                result.bootstrap_ci_sharpe = (
                    round(float(np.percentile(boot_sharpes, 5)), 3),
                    round(float(np.percentile(boot_sharpes, 95)), 3),
                )
    except Exception as _aronson_exc:
        logger.debug("Aronson enrichment skipped: %s", _aronson_exc)

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
        f"  Win Rate:      {result.win_rate:>11.1f}%",
        f"  Profit Factor: {result.profit_factor:>12.2f}",
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

    # V4: Capital rotation summary
    if _HARVEST_ENABLED:
        lines.append(f"")
        lines.append(f"  {'─'*40}")
        lines.append(f"  Centurion Harvest — Capital Rotation:")
        lines.append(f"  Total Injected:    ₹{_cr_total_injected:>12,.0f}  ({len(_cr_inject_events)} events)")
        lines.append(f"  Total Booked:      ₹{_cr_total_booked:>12,.0f}  ({len(_cr_book_events)} events)")
        _cr_net = _cr_total_injected - _cr_total_booked
        lines.append(f"  Net Capital Added: ₹{_cr_net:>12,.0f}")
        # Adjusted return: factor out injected capital for fair comparison
        _cr_adjusted_equity = equity - _cr_total_injected + _cr_total_booked
        _cr_adj_ret = (_cr_adjusted_equity / (capital - _cr_total_injected) - 1) * 100 if (capital - _cr_total_injected) > 0 else 0
        lines.append(f"  Adj. Total Return: {_cr_adj_ret:>11.1f}%  (excl. injections)")
        if _cr_inject_events:
            lines.append(f"  Injection Events:")
            for _ev in _cr_inject_events:
                lines.append(f"    Day {_ev[0]:4d}: +₹{_ev[1]:,.0f}  (equity ₹{_ev[2]:,.0f} → ₹{_ev[3]:,.0f})")
        if _cr_book_events:
            lines.append(f"  Profit Booking Events:")
            for _ev in _cr_book_events:
                lines.append(f"    Day {_ev[0]:4d}: -₹{_ev[1]:,.0f}  (equity ₹{_ev[2]:,.0f} → ₹{_ev[3]:,.0f})")

    # R22: Bull-run capital infusion summary
    if _r22_alert_events or _r22_infusion_events:
        lines.append(f"")
        lines.append(f"  {'─'*40}")
        lines.append(f"  R22 — Bull-Run Capital Infusion (Centurion Compounder):")
        lines.append(f"  Infusion Enabled:  {'YES' if _R22_BULL_INFUSION else 'NO (alerts only)'}")
        lines.append(f"  Bull Alerts:       {len(_r22_alert_events):>12d}")
        lines.append(f"  Total Infused:     ₹{_r22_total_infused:>12,.0f}  ({len(_r22_infusion_events)} events)")
        if _r22_total_infused > 0:
            _r22_orig_capital = capital - _r22_total_infused
            _r22_adj_equity = equity - _r22_total_infused
            _r22_adj_ret = (_r22_adj_equity / _r22_orig_capital - 1) * 100 if _r22_orig_capital > 0 else 0
            lines.append(f"  Adj. Total Return: {_r22_adj_ret:>11.1f}%  (excl. infusions — organic compounding)")
        if _r22_alert_events:
            lines.append(f"  Bull Confirmation Alerts:")
            for _ev in _r22_alert_events:
                _infused_tag = ""
                for _ie in _r22_infusion_events:
                    if _ie[0] == _ev[0]:
                        _infused_tag = f"  → INFUSED +₹{_ie[1]:,.0f}"
                        break
                lines.append(f"    Day {_ev[0]:4d} ({_ev[1]}){_infused_tag}")

    # Aronson EBTA enrichment
    if result.detrended_sharpe or result.trimmed_sharpe:
        lines.append(f"")
        lines.append(f"  {'─'*40}")
        lines.append(f"  Aronson EBTA Metrics:")
        lines.append(f"  Detrended Sharpe:  {result.detrended_sharpe:>10.3f}")
        lines.append(f"  Trimmed Sharpe:    {result.trimmed_sharpe:>10.3f}")
        lines.append(f"  DM Bias Est (%):   {result.dm_bias_estimate:>10.2f}")
        if result.bootstrap_ci_sharpe != (0.0, 0.0):
            lines.append(f"  Sharpe 90% CI:     [{result.bootstrap_ci_sharpe[0]:.3f}, {result.bootstrap_ci_sharpe[1]:.3f}]")
        if result.per_signal_tstats:
            lines.append(f"  Signals t≥2.0:     {sum(1 for t in result.per_signal_tstats.values() if abs(t) >= 2.0)}/{len(result.per_signal_tstats)}")

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
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "source_hit_rates": result.source_hit_rates,
        "daily_equity": result.daily_equity,
        "report": result.report,
        # Aronson EBTA enrichment
        "detrended_sharpe": result.detrended_sharpe,
        "trimmed_sharpe": result.trimmed_sharpe,
        "per_signal_tstats": result.per_signal_tstats,
        "dm_bias_estimate": result.dm_bias_estimate,
        "bootstrap_ci_sharpe": result.bootstrap_ci_sharpe,
        # V4: Capital rotation
        "capital_rotation": {
            "total_injected": _cr_total_injected,
            "total_booked": _cr_total_booked,
            "net_added": _cr_total_injected - _cr_total_booked,
            "inject_events": _cr_inject_events,
            "book_events": _cr_book_events,
        } if _HARVEST_ENABLED else None,
        # R22: Bull-run capital infusion
        "r22_bull_infusion": {
            "enabled": _R22_BULL_INFUSION,
            "infusion_amount": _R22_INFUSION_AMOUNT,
            "total_infused": _r22_total_infused,
            "infusion_events": _r22_infusion_events,   # [(day, amount, eq_before, eq_after)]
            "alert_events": _r22_alert_events,           # [(day, date_str)]
            "n_alerts": len(_r22_alert_events),
            "n_infusions": len(_r22_infusion_events),
        } if (_r22_alert_events or _R22_BULL_INFUSION) else None,
    }


if __name__ == "__main__":
    import sys, os
    # Ensure centurion_core root is on sys.path for internal imports
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    result = run_full_backtest(
        tickers=None,
        capital=500_000,
        period="13y",
        market="IND",
        verbose=True,
        start_date="2012-01-01",
        end_date="2025-12-31",
    )
    # Report already printed by verbose=True; print metrics summary
    print(f"\n=== KEY METRICS ===")
    for k in ["annual_return_pct", "total_return_pct", "sharpe", "sortino",
              "calmar", "max_drawdown_pct", "n_trades", "avg_positions",
              "detrended_sharpe", "trimmed_sharpe", "dm_bias_estimate"]:
        print(f"  {k:25s} = {result.get(k)}")
    if result.get("bootstrap_ci_sharpe"):
        lo, hi = result["bootstrap_ci_sharpe"]
        print(f"  {'bootstrap_ci_sharpe':25s} = [{lo:.3f}, {hi:.3f}]")
