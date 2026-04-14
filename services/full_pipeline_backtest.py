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

# ── Centralized config import (single source of truth for all params) ──
try:
    from config import Config as _Cfg
except ImportError:
    _Cfg = None

# ── Rolling forecast history for empirical FDM (C3 fix) ──
from collections import deque
_rolling_forecast_history: Dict[str, deque] = {}  # {source_name: deque(maxlen=252)}
_ROLLING_FDM_LOOKBACK = 252  # 1 year of daily forecasts

# ── R21A active configuration ──────────────────────────────────────────
_SAVE_FORECASTS_MODE = False  # R21a: save per-source forecasts for weight optimization
_forecast_log: list = []      # accumulator: [(day_idx, {sym: {source: val}}, {sym: next_ret})]
_R21A_REGIME_VOL = True       # R21a: regime-adaptive vol target (aggressive uptrend, defensive downtrend)
_R21A_REGIME_BOOST = getattr(_Cfg, 'R21A_REGIME_BOOST', 1.25) if _Cfg else 1.25
_R21A_REGIME_DEFEND = getattr(_Cfg, 'R21A_REGIME_DEFEND', 0.55) if _Cfg else 0.55

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
    turnover_penalized_ci: tuple = (0.0, 0.0)  # L5: turnover-penalized bootstrap CI



# ── Helpers ───────────────────────────────────────────────────

def _download(sym: str, period: str, market: str,
              start: str = "", end: str = "") -> Optional[pd.DataFrame]:
    """Download OHLCV via yfinance.  Prefers start/end dates over period."""
    try:
        import yfinance as yf
        import warnings
        # Add .NS for IND market, but NOT for symbols already suffixed (.NS/.BO)
        # or non-NSE tickers like BTC-USD, USDINR=X, ^NSEI
        suffix = ".NS" if market == "IND" and not any(c in sym for c in '.-=^') else ""
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


def _annual_yield_at(sym: str, date, div_history: Dict[str, pd.Series],
                     price: float) -> float:
    """Compute trailing-12-month dividend yield at a given date (point-in-time).

    Sums all dividend payments in [date - 365d, date] and divides by current price.
    Returns 0.0 if no dividends found or price is invalid.
    """
    if price <= 0 or not np.isfinite(price):
        return 0.0
    divs = div_history.get(sym)
    if divs is None or len(divs) == 0:
        return 0.0
    # Convert date to Timestamp for comparison
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    start = date - pd.Timedelta(days=365)
    mask = (divs.index >= start) & (divs.index <= date)
    trailing_divs = divs.loc[mask]
    if len(trailing_divs) == 0:
        return 0.0
    return float(trailing_divs.sum()) / price


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

    # Phase B: Point-in-time universe flag (read once, used in ticker loading + sim loop)
    _pit_universe_on = getattr(_Cfg, 'PIT_UNIVERSE_ENABLED', False) if _Cfg else False

    # ── Default tickers ────────────────────────────────────────
    if tickers is None:
        if market == "US":
            tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "TSLA", "JPM", "V", "UNH", "HD", "PG", "XOM", "MA",
                "JNJ",
            ]
        else:
            # FIX (v29): Respect Config.NSE_UNIVERSE_TIER instead of
            # always using get_nse_default_tickers (95 stocks).
            # Tiers: DEFAULT=~100, NIFTY500=~500, BROAD=~800-1200
            # Phase B: If PIT_UNIVERSE_ENABLED, download the union of ALL historical
            # NIFTY500 constituents (~800 unique tickers). The sim loop will filter
            # to the correct ~500 at each point in time.
            try:
                from kite_connect.nse.nse_universe import get_nse_universe
                if _pit_universe_on:
                    from kite_connect.nse.nse_universe import get_nse_universe_pit_union
                    raw_syms = get_nse_universe_pit_union()
                    if raw_syms and len(raw_syms) >= 10:
                        tickers = [f"{s}.NS" for s in raw_syms]
                        if verbose:
                            print(f"  NSE universe: {len(tickers)} tickers (PIT union)")
                    else:
                        raise ValueError("PIT union returned too few symbols")
                else:
                    _tier = getattr(_Cfg, 'NSE_UNIVERSE_TIER', 'DEFAULT') if _Cfg else 'DEFAULT'
                    raw_syms = get_nse_universe(tier=_tier)
                    if raw_syms and len(raw_syms) >= 10:
                        tickers = [f"{s}.NS" for s in raw_syms]
                        if verbose:
                            print(f"  NSE universe: {len(tickers)} tickers (tier={_tier})")
                    else:
                        raise ValueError(f"NSE universe ({_tier}) returned too few symbols")
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

    # ── Godmode Phase 2b: Multi-asset diversification ──────────
    # Add uncorrelated assets (gold ETFs, CPSE, liquid) to reduce portfolio correlation.
    _multi_asset_on = getattr(_Cfg, 'MULTI_ASSET_ENABLED', False) if _Cfg else False
    if _multi_asset_on and market == "IND":
        _ma_tickers = getattr(_Cfg, 'MULTI_ASSET_TICKERS_IND', []) if _Cfg else []
        for _mat in _ma_tickers:
            if _mat not in tickers:
                tickers.append(_mat)
        if verbose and _ma_tickers:
            print(f"  Multi-asset: +{len(_ma_tickers)} tickers ({', '.join(_ma_tickers)})")

    # ── Crypto ticker for crypto_correlation signal ────────────
    if _SAVE_FORECASTS_MODE or _multi_asset_on:
        _crypto_tk = getattr(_Cfg, 'CRYPTO_TICKER', 'BTC-USD') if _Cfg else 'BTC-USD'
        if _crypto_tk and _crypto_tk not in tickers:
            tickers.append(_crypto_tk)
            if verbose:
                print(f"  Crypto proxy: +{_crypto_tk} (for crypto_correlation signal)")

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
        "carry", "screener", "momentum", "pead",
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
    # Phase 2: Per-symbol Zerodha delivery costs (replaces flat cost_pct)
    # Zerodha equity delivery: ₹0 brokerage. Costs = STT + exchange txn + GST + stamp + SEBI
    # Buy-side: ~0.059% (stamp 0.015% + exchange 0.00345% + GST 0.000621% + SEBI 0.0001%)
    # Sell-side: ~0.159% (STT 0.1% + exchange 0.00345% + GST 0.000621% + SEBI 0.0001%)
    # Round-trip statutory: ~0.218% ≈ 22 bps
    _statutory_rt_cost = 0.0022  # 22 bps round-trip (Zerodha equity delivery)
    # Tiered slippage by market-cap tier
    _slip_nifty50 = 0.0005   # 5 bps — NIFTY50 very liquid
    _slip_next50  = 0.0020   # 20 bps — mid-cap
    _slip_small   = 0.0050   # 50 bps — smallcap / illiquid
    _nifty50_cost = _statutory_rt_cost + _slip_nifty50  # 27 bps
    _next50_cost  = _statutory_rt_cost + _slip_next50   # 42 bps
    _smallcap_cost = _statutory_rt_cost + _slip_small   # 72 bps
    # H2 FIX: kept for backward compat in report line
    _slippage_bps = 0.0020  # default mid-cap slippage
    if market == "IND":
        cost_pct = _next50_cost   # default fallback (mid-cap)
        # Build per-symbol cost lookup from INDEX_CONSTITUENTS
        try:
            from kite_connect.core.config import INDEX_CONSTITUENTS as _IDX
            _n50_set = set(_IDX.get("NIFTY50", []))
            _nn50_set = set(_IDX.get("NIFTY_NEXT50", []))
        except ImportError:
            _n50_set, _nn50_set = set(), set()
        _sym_cost_map: Dict[str, float] = {}
        for sym in symbols:
            _bare = sym.replace('.NS', '').replace('.BO', '')
            if _bare in _n50_set:
                _sym_cost_map[sym] = _nifty50_cost
            elif _bare in _nn50_set:
                _sym_cost_map[sym] = _next50_cost
            else:
                _sym_cost_map[sym] = _smallcap_cost
    else:
        cost_pct = 0.0015 + 0.0003   # 15 bps cost + 3 bps slippage (US)
        _sym_cost_map = {sym: cost_pct for sym in symbols}

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
    # PBO-FIX: Per-signal daily return attribution for valid CSCV
    source_daily_returns: Dict[str, List[float]] = defaultdict(list)

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
    _true_peak_equity = capital  # H1 FIX: TRUE peak that NEVER decays — used for DD halt
    dd_deep_days = 0          # consecutive days with DD > 25% (for gradual peak decay)
    PEAK_DECAY_GRACE_DAYS = 60  # R14 baseline
    PEAK_DECAY_RATE = 0.01      # R14 baseline: 1%/day blend

    # R13: Bear lockout REMOVED — binary exit/re-enter causes whipsaw in all variants
    # R11 (return-based) and R12 (vol-based) both destroyed equity via whipsaw.
    # R13 uses dynamic vol target (Fix C) instead — continuous, no churn.

    # ── Pre-fetch historical dividend series for carry (point-in-time) ────
    # Phase 3: Use historical dividend payments instead of current yield (look-ahead fix)
    # Stores full dividend payment series per symbol → compute trailing-12m yield at any date
    dividend_yields: Dict[str, float] = {}  # backward compat: stores latest yield as fallback
    _dividend_history: Dict[str, pd.Series] = {}  # sym → DatetimeIndex Series of dividend payments
    if include_carry:
        try:
            import yfinance as yf
            import warnings, io, contextlib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for sym in symbols:
                    # Skip non-equity symbols (no dividends for crypto, FX, indices)
                    if "-" in sym or sym.startswith("^") or "=" in sym:
                        dividend_yields[sym] = 0.0
                        continue
                    try:
                        suffix = ".NS" if market == "IND" and not any(c in sym for c in '.-=^') else ""
                        _tk = f"{sym}{suffix}"
                        with contextlib.redirect_stdout(io.StringIO()), \
                             contextlib.redirect_stderr(io.StringIO()):
                            _divs = yf.Ticker(_tk).dividends
                        if _divs is not None and len(_divs) > 0:
                            # Ensure timezone-naive index for comparison with simulation dates
                            if _divs.index.tz is not None:
                                _divs.index = _divs.index.tz_localize(None)
                            _dividend_history[sym] = _divs
                            # Also store latest trailing yield as fallback
                            dividend_yields[sym] = 0.01  # placeholder — real yield computed point-in-time
                        else:
                            dividend_yields[sym] = 0.0
                    except Exception:
                        dividend_yields[sym] = 0.0  # store 0 to prevent per-day re-fetch
            if verbose:
                _actual = sum(1 for v in _dividend_history.values() if len(v) > 0)
                print(f"  Dividend history fetched: {_actual}/{n_symbols} symbols with dividends")
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
        max_leverage = getattr(_BtCfg, 'CARVER_MAX_LEVERAGE', 2.0)
    except Exception:
        max_leverage = 2.0
    allow_short = False  # FIX-SHORT: disabled — short Sharpe ≈ -0.01, bleeds in secular bull

    # Phase 4: Execution gap model — penalty for close→open gap on new entries/exits
    _EXECUTION_GAP_ENABLED = getattr(_Cfg, 'EXECUTION_GAP_ENABLED', True) if _Cfg else True
    _EXECUTION_GAP_BPS = getattr(_Cfg, 'EXECUTION_GAP_BPS', 0.0050) if _Cfg else 0.0050  # 50 bps

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
            # In extraction mode, allow n_symbols drift (NIFTY500 may vary ±5% across sessions)
            _ckpt_n_sym = _ckpt.get('n_symbols', 0)
            if _SAVE_FORECASTS_MODE:
                _sym_match = abs(_ckpt_n_sym - n_symbols) <= max(20, int(n_symbols * 0.10))
            else:
                _sym_match = (_ckpt_n_sym == n_symbols)
            if (_sym_match
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
                _true_peak_equity = _ckpt.get('true_peak_equity', peak_equity)  # H1: restore true peak
                dd_deep_days = _ckpt['dd_deep_days']
                _cached_forecasts = _ckpt.get('cached_forecasts', {})
                _cached_idm = _ckpt.get('cached_idm', 1.7)
                trade_pnls = _ckpt.get('trade_pnls', [])
                entry_prices = _ckpt.get('entry_prices', {})
                # PBO-FIX: restore per-signal daily returns
                _sdr = _ckpt.get('source_daily_returns', {})
                if _sdr:
                    source_daily_returns = defaultdict(list, {k: list(v) for k, v in _sdr.items()})
                # Restore forecast log for extraction mode resume
                if _SAVE_FORECASTS_MODE and 'forecast_log' in _ckpt:
                    _forecast_log.clear()
                    _forecast_log.extend(_ckpt['forecast_log'])
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

    _prev_sigint = None
    _prev_sigterm = None
    try:
        _prev_sigint = signal.signal(signal.SIGINT, _graceful_shutdown)
        _prev_sigterm = signal.signal(signal.SIGTERM, _graceful_shutdown)
    except ValueError:
        pass  # Not on main thread (e.g. FastAPI asyncio.to_thread) — skip signal handling

    # ── Timer-based graceful exit for cloud environments (Kaggle 12h limit) ──
    # If CENTURION_MAX_RUNTIME_SECS is set, schedule a graceful exit before the
    # cloud platform kills the process with SIGKILL (no chance to save).
    _max_runtime = int(os.environ.get("CENTURION_MAX_RUNTIME_SECS", "0"))
    _deadline_timer = None
    if _max_runtime > 0:
        import threading

        def _deadline_shutdown():
            nonlocal _graceful_exit_requested
            _graceful_exit_requested = True
            if verbose:
                print(f"\n  ⏱ Runtime limit ({_max_runtime}s) reached — saving checkpoint and exiting", flush=True)

        _deadline_timer = threading.Timer(_max_runtime, _deadline_shutdown)
        _deadline_timer.daemon = True
        _deadline_timer.start()
        if verbose:
            print(f"  Timer: graceful exit in {_max_runtime // 3600}h {(_max_runtime % 3600) // 60}m")

    _consecutive_day_errors = 0
    _MAX_CONSECUTIVE_ERRORS = 10  # abort if 10+ days crash in a row

    # ── Phase B: Point-in-Time universe state ──────────────────
    # Tracks which subset of downloaded symbols are valid NIFTY500 constituents
    # at the current simulation date.  Updated at semi-annual boundaries.
    _pit_active_set: Optional[set] = None   # set of ".NS" tickers currently in-universe
    _pit_last_period: Optional[str] = None  # "YYYY-MM" of last universe reload
    if _pit_universe_on and market == "IND":
        from kite_connect.nse.nse_universe import get_nse_universe_pit
        # Initialize with earliest period
        _pit_syms = get_nse_universe_pit("2012-01-01")
        _pit_active_set = {f"{s}.NS" for s in _pit_syms}
        _pit_last_period = "2012-03"
        if verbose:
            print(f"  PIT Universe: initialized with {len(_pit_active_set)} symbols (2012-03)")

    for day_idx in range(_start_day_idx, n_days):
      try:
        day_pnl = 0.0

        # Build OHLCV slices up to current day (views, not copies)
        ohlcv_slice: Dict[str, pd.DataFrame] = {}
        # T3-1: Extract current simulation date for look-ahead bias prevention
        current_date = master_index[day_idx]
        if hasattr(current_date, 'date'):
            current_date = current_date.date()

        # ── Phase B: PIT universe rotation at semi-annual boundaries ──
        if _pit_active_set is not None:
            import datetime as _dt
            _cd = current_date if isinstance(current_date, _dt.date) else _dt.date.fromisoformat(str(current_date))
            _m = _cd.month
            if _m >= 9:
                _cur_period = f"{_cd.year}-09"
            elif _m >= 3:
                _cur_period = f"{_cd.year}-03"
            else:
                _cur_period = f"{_cd.year - 1}-09"
            if _cur_period != _pit_last_period:
                _pit_syms_new = get_nse_universe_pit(_cd)
                _pit_new_set = {f"{s}.NS" for s in _pit_syms_new}
                _removed = _pit_active_set - _pit_new_set
                _added = _pit_new_set - _pit_active_set
                # Force-sell positions in removed tickers (delisted / dropped from index)
                for _rsym in _removed:
                    _rqty = prev_positions.get(_rsym, 0)
                    if _rqty != 0 and _rsym in ohlcv_full:
                        _rc = ohlcv_full[_rsym]["Close"]
                        if hasattr(_rc, "squeeze"):
                            _rc = _rc.squeeze()
                        _ridx = day_idx - sym_start.get(_rsym, 0)
                        if 0 < _ridx < len(_rc):
                            _rpx = float(_rc.iloc[_ridx])
                            if np.isfinite(_rpx) and _rpx > 0:
                                day_pnl -= abs(_rqty) * _rpx * _sym_cost_map.get(_rsym, cost_pct)
                                trades_count += 1
                                if _rsym in entry_prices and entry_prices[_rsym] > 0:
                                    _ep = entry_prices.pop(_rsym)
                                    if _rqty > 0:
                                        trade_pnls.append((_rpx - _ep) / _ep * 100)
                                    else:
                                        trade_pnls.append((_ep - _rpx) / _ep * 100)
                        prev_positions[_rsym] = 0
                        peak_prices.pop(_rsym, None)
                        stop_levels.pop(_rsym, None)
                _pit_active_set = _pit_new_set
                _pit_last_period = _cur_period
                if verbose and (_removed or _added):
                    print(f"  PIT Rebalance {_cur_period}: +{len(_added)} / -{len(_removed)} → {len(_pit_active_set)} symbols", flush=True)

        for sym, df in ohlcv_full.items():
            # Phase B: Skip symbols not in current PIT universe
            if _pit_active_set is not None and sym not in _pit_active_set:
                continue
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
                        day_pnl -= abs(prev_qty) * exit_price * _sym_cost_map.get(sym, cost_pct)
                        trades_count += 1
                        if sym in entry_prices and entry_prices[sym] > 0:
                            trade_pnls.append((exit_price - entry_prices[sym]) / entry_prices[sym] * 100)
                            del entry_prices[sym]
                        prev_positions[sym] = 0
                        peak_prices.pop(sym, None)
                        stop_levels.pop(sym, None)
                        stop_cooldown[sym] = 2  # Phase C: reduced from 5 → 2 days
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
                        day_pnl -= abs(prev_qty) * exit_price * _sym_cost_map.get(sym, cost_pct)
                        trades_count += 1
                        if sym in entry_prices and entry_prices[sym] > 0:
                            trade_pnls.append((entry_prices[sym] - exit_price) / entry_prices[sym] * 100)
                            del entry_prices[sym]
                        prev_positions[sym] = 0
                        peak_prices.pop(sym, None)
                        stop_levels.pop(sym, None)
                        stop_cooldown[sym] = 2  # Phase C: reduced from 5 → 2 days
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
        # Optimization: tiered recompute — fast signals daily, slow signals every 5 days.
        # Fast:   ewmac_8_32, breakout (daily)
        # Medium: momentum, ehlers_dsp, acceleration, penfold_trend (3-day)
        # Slow:   carver_value, ewmac_64_256, mean_reversion, carry (5-day)
        _RECOMPUTE_FREQ_FAST = getattr(_Cfg, 'RECOMPUTE_FREQ_FAST', 1) if _Cfg else 1
        _RECOMPUTE_FREQ_MED = getattr(_Cfg, 'RECOMPUTE_FREQ_MEDIUM', 3) if _Cfg else 3
        _RECOMPUTE_FREQ_SLOW = getattr(_Cfg, 'RECOMPUTE_FREQ_SLOW', 5) if _Cfg else 5
        _trading_day = day_idx - min_history
        _recompute_fast = (_trading_day % _RECOMPUTE_FREQ_FAST == 0)
        _recompute_med = (_trading_day % _RECOMPUTE_FREQ_MED == 0)
        _recompute_slow = (_trading_day % _RECOMPUTE_FREQ_SLOW == 0)
        _recompute = _recompute_fast  # at minimum, fast signals recompute daily

        if not _recompute:
            all_forecasts = {sym: dict(fc) for sym, fc in _cached_forecasts.items()}
        else:  # ── full signal recompute (serial) ──
            all_forecasts: Dict[str, Dict[str, float]] = {sym: {} for sym in symbols}
            # C5 FIX: For slow signals, reuse cache unless _recompute_slow is True.
            # Pre-seed from cache so slow signals carry forward on non-slow days.
            if not _recompute_slow and _cached_forecasts:
                _slow_sources = {'carver_value', 'ewmac_64_256', 'mean_reversion', 'carry', 'skew_signal'}
                for sym, cached_fc in _cached_forecasts.items():
                    if sym in all_forecasts:
                        for src, val in cached_fc.items():
                            if src in _slow_sources:
                                all_forecasts[sym][src] = val
            if not _recompute_med and _cached_forecasts:
                _med_sources = {'momentum', 'ehlers_dsp', 'acceleration', 'penfold_trend',
                                'cross_momentum', 'intermarket'}
                for sym, cached_fc in _cached_forecasts.items():
                    if sym in all_forecasts:
                        for src, val in cached_fc.items():
                            if src in _med_sources:
                                all_forecasts[sym][src] = val

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

            # 2c. Mean reversion (SLOW signal — skip if cached on non-slow days)
            if _recompute_slow or not any('mean_reversion' in fc for fc in all_forecasts.values() if fc):
              try:
                mr_forecasts = compute_mean_reversion_batch(ohlcv_slice)
                for sym, fc in mr_forecasts.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["mean_reversion"] = fc
              except Exception as e:
                logger.debug("Mean reversion failed at day %d: %s", day_idx, e)

            # 2d. Carry (SLOW signal — skip if cached on non-slow days)
            if include_carry and (_recompute_slow or not any('carry' in fc for fc in all_forecasts.values() if fc)):
                try:
                    # Phase 3: Compute point-in-time dividend yields for this date
                    _pit_yields: Dict[str, float] = {}
                    for _csym, _cdf in ohlcv_slice.items():
                        if _csym in _dividend_history:
                            _cc = _cdf["Close"]
                            if hasattr(_cc, "squeeze"):
                                _cc = _cc.squeeze()
                            _cpx = float(_cc.dropna().iloc[-1]) if len(_cc.dropna()) > 0 else 0.0
                            _pit_yields[_csym] = _annual_yield_at(
                                _csym, current_date, _dividend_history, _cpx)
                        else:
                            _pit_yields[_csym] = dividend_yields.get(_csym, 0.0)
                    carry_results = compute_carry_batch(
                        ohlcv_slice, dividend_yields=_pit_yields,
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
            # M1 FIX: Percentile-scaled forecasts instead of fixed ±8
            # H5 FIX: Bottom tercile = 0 in long-only mode (can't short anyway)
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
                _n_xmom = len(sorted_syms)
                for _rank_i, sym in enumerate(sorted_syms):
                    # Percentile: 0 (worst) → 1 (best)
                    _pctl = _rank_i / max(_n_xmom - 1, 1)
                    # Scale to [-20, +20] forecast range, centered at 0
                    _xmom_fc = (_pctl - 0.5) * 40.0  # e.g. top=+20, bottom=-20
                    _xmom_fc = max(-20.0, min(20.0, _xmom_fc))
                    # H5: floor at 0 for long-only (negative xmom drags combined forecast
                    # for stocks we can't short — just wastes forecast weight)
                    if not allow_short and _xmom_fc < 0:
                        _xmom_fc = 0.0
                    all_forecasts[sym]["cross_momentum"] = _xmom_fc

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

            # ── 2o-2r. DEAD SIGNALS SKIPPED (M4 FIX) ──────────
            # PEAD, fii_flow, event_driven, sentiment: all at 0% weight with
            # 0% hit rate in backtest. Skip computation to save ~40% per-day time.
            # Re-enable when proper data sources are available.

            # ── NEW ALPHA SOURCES (6 uncorrelated) ──────────────

            # ── 2o-new. Calendar Effects (month-end, expiry, budget, Diwali) ──
            try:
                from strategies.calendar_effects import compute_calendar_forecast_batch
                cal_fc = compute_calendar_forecast_batch(ohlcv_slice)
                for sym, fc in cal_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["calendar_effects"] = fc
            except Exception as e:
                logger.debug("Calendar effects failed at day %d: %s", day_idx, e)

            # ── 2p-new. Fundamental Momentum (52w-high proximity + 3m momentum) ──
            try:
                from strategies.fundamental_momentum import compute_fundamental_momentum_batch
                fmom_fc = compute_fundamental_momentum_batch(ohlcv_slice)
                for sym, fc in fmom_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["fundamental_momentum"] = fc
            except Exception as e:
                logger.debug("Fundamental momentum failed at day %d: %s", day_idx, e)

            # ── 2q-new. Insider Activity (volume surge + directional proxy) ──
            try:
                from strategies.insider_activity import compute_insider_activity_batch
                insider_fc = compute_insider_activity_batch(ohlcv_slice)
                for sym, fc in insider_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["insider_activity"] = fc
            except Exception as e:
                logger.debug("Insider activity failed at day %d: %s", day_idx, e)

            # ── 2r-new. Dispersion Trading (constituent vol vs portfolio vol) ──
            try:
                from strategies.dispersion_trading import compute_dispersion_forecast_batch
                disp_fc = compute_dispersion_forecast_batch(ohlcv_slice)
                for sym, fc in disp_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["dispersion"] = fc
            except Exception as e:
                logger.debug("Dispersion trading failed at day %d: %s", day_idx, e)

            # ── 2s-new. Gold-Equity Rotation (GOLDBEES relative strength) ──
            if _multi_asset_on or _SAVE_FORECASTS_MODE:
                try:
                    from strategies.gold_equity_rotation import compute_gold_equity_rotation_batch
                    gold_fc = compute_gold_equity_rotation_batch(ohlcv_slice)
                    for sym, fc in gold_fc.items():
                        if sym in all_forecasts:
                            all_forecasts[sym]["gold_equity_rotation"] = fc
                except Exception as e:
                    logger.debug("Gold-equity rotation failed at day %d: %s", day_idx, e)

            # ── 2t-new. Crypto Correlation (BTC as risk-on indicator) ──
            try:
                from strategies.crypto_correlation import compute_crypto_correlation_batch
                crypto_fc = compute_crypto_correlation_batch(ohlcv_slice)
                for sym, fc in crypto_fc.items():
                    if sym in all_forecasts:
                        all_forecasts[sym]["crypto_correlation"] = fc
            except Exception as e:
                logger.debug("Crypto correlation failed at day %d: %s", day_idx, e)

            # ── 2u-new. Sector Rotation Signal (macro sector momentum overlay) ──
            if _recompute_slow or not any('sector_rotation' in fc for fc in all_forecasts.values() if fc):
                try:
                    from strategies.sector_rotation_signal import compute_sector_rotation_forecast_batch
                    sr_fc = compute_sector_rotation_forecast_batch(ohlcv_slice)
                    for sym, fc in sr_fc.items():
                        if sym in all_forecasts:
                            all_forecasts[sym]["sector_rotation"] = fc
                except Exception as e:
                    logger.debug("Sector rotation signal failed at day %d: %s", day_idx, e)

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

            # ── 2u. Multi-timeframe EWMAC blending (Godmode Phase 2) ──
            # Compute weekly and monthly EWMAC and blend into daily signals.
            # Daily signals keep DAILY_SIGNAL_WEIGHT, remaining split weekly+monthly.
            _mtf_enabled = getattr(_Cfg, 'MULTI_TIMEFRAME_ENABLED', False) if _Cfg else False
            if _mtf_enabled:
                _w_daily = getattr(_Cfg, 'DAILY_SIGNAL_WEIGHT', 0.65) if _Cfg else 0.65
                _w_weekly = getattr(_Cfg, 'WEEKLY_SIGNAL_WEIGHT', 0.25) if _Cfg else 0.25
                _w_monthly = getattr(_Cfg, 'MONTHLY_SIGNAL_WEIGHT', 0.10) if _Cfg else 0.10
                for sym, df in ohlcv_slice.items():
                    if sym not in all_forecasts:
                        continue
                    c = df["Close"]
                    if hasattr(c, "squeeze"):
                        c = c.squeeze()
                    close = c.dropna()
                    if len(close) < 270:
                        continue
                    dpv = daily_price_volatility(close)
                    if dpv <= 0:
                        dpv = 0.02
                    # Weekly: resample to 5-day bars, compute EWMAC(4,16) ~ weekly 8_32
                    try:
                        weekly_close = close.iloc[::5]  # subsample every 5 bars
                        if len(weekly_close) >= 20:
                            w_fast = weekly_close.ewm(span=4, adjust=False).mean()
                            w_slow = weekly_close.ewm(span=16, adjust=False).mean()
                            w_raw = float(w_fast.iloc[-1] - w_slow.iloc[-1])
                            w_dpv = max(dpv * np.sqrt(5), 1e-6)  # weekly vol scale
                            w_fc = max(-20.0, min(20.0, w_raw / (float(close.iloc[-1]) * w_dpv)))
                            # Blend EWMAC signals: daily_fc × w_daily + weekly_fc × w_weekly
                            for key in list(all_forecasts[sym].keys()):
                                if key.startswith("ewmac_"):
                                    daily_fc = all_forecasts[sym][key]
                                    all_forecasts[sym][key] = daily_fc * _w_daily + w_fc * _w_weekly
                    except Exception:
                        pass
                    # Monthly: resample to ~21-day bars, compute EWMAC(2,8) ~ monthly 64_256
                    try:
                        monthly_close = close.iloc[::21]
                        if len(monthly_close) >= 10:
                            m_fast = monthly_close.ewm(span=2, adjust=False).mean()
                            m_slow = monthly_close.ewm(span=8, adjust=False).mean()
                            m_raw = float(m_fast.iloc[-1] - m_slow.iloc[-1])
                            m_dpv = max(dpv * np.sqrt(21), 1e-6)
                            m_fc = max(-20.0, min(20.0, m_raw / (float(close.iloc[-1]) * m_dpv)))
                            for key in list(all_forecasts[sym].keys()):
                                if key.startswith("ewmac_"):
                                    all_forecasts[sym][key] += m_fc * _w_monthly
                    except Exception:
                        pass


            _cached_forecasts = {sym: dict(fc) for sym, fc in all_forecasts.items()}
        # ── 3. Combine forecasts + size positions ──────────────
        # R12: SIMPLEST POSSIBLE SYSTEM — strip all whipsaw sources
        # Meta-analysis of R1-R11: every scaling mechanism (DD, regime, warmup)
        # either creates death spiral or amplifies whipsaw losses.
        # R12 approach: STAY INVESTED through corrections, exit ONLY on panic vol.
        peak_equity = max(peak_equity, equity)
        _true_peak_equity = max(_true_peak_equity, equity)  # H1 FIX: TRUE peak never decays
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        # H1/H2 FIX: True drawdown from absolute peak for halt decisions
        _true_dd = (_true_peak_equity - equity) / _true_peak_equity if _true_peak_equity > 0 else 0.0

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

        # REVERTED: Use current_dd (decaying peak) for vol tier decisions.
        # _true_dd from non-decaying peak caused permanent halt — system went
        # to 0 positions at day ~100 and NEVER recovered (true peak never decays).
        if current_dd < 0.15:
            annual_vol_target = 0.50   # Full risk — no DD
        elif current_dd < 0.25:
            annual_vol_target = 0.45   # Mild pullback
        elif current_dd < 0.30:
            annual_vol_target = 0.35   # Moderate DD
        elif current_dd < 0.35:
            annual_vol_target = 0.20   # Severe DD
        else:
            annual_vol_target = 0.05   # Near-halt (NOT zero — allow slow recovery)

        # FIX-FLOOR: Use actual equity for daily target
        sizing_equity = max(equity, capital * 0.10)  # 10% ruin floor
        dynamic_daily_target = sizing_equity * annual_vol_target / 16.0

        # R21a: Regime-adaptive vol target
        # Aggressive in sustained uptrends, defensive in downtrends
        # Godmode: smooth sigmoid interpolation (replaces binary threshold)
        # C1 FIX: Determine equity-curve regime for sizing, stops, leverage
        _eq_regime = 'neutral'  # default
        if len(daily_equity) >= 200:
            _eq_sma200 = sum(daily_equity[-200:]) / 200.0
            if equity < _eq_sma200 * 0.95:
                _eq_regime = 'severe_bear'   # C1: go to ZERO exposure
            elif equity < _eq_sma200 * 0.98:
                _eq_regime = 'bear'
            elif equity > _eq_sma200 * 1.05:
                _eq_regime = 'strong_bull'
            elif equity > _eq_sma200 * 1.02:
                _eq_regime = 'bull'
        else:
            _eq_sma200 = equity  # not enough data

        # C1: BEAR regime → floor at 10% of normal exposure (was 0.0 — positions=0 trap)
        _SEVERE_BEAR_FLOOR = getattr(_Cfg, 'SEVERE_BEAR_EXPOSURE_FLOOR', 0.10) if _Cfg else 0.10
        if _eq_regime == 'severe_bear':
            dynamic_daily_target = max(dynamic_daily_target * _SEVERE_BEAR_FLOOR,
                                       sizing_equity * 0.05 / 16.0)  # absolute min: 5% vol target
        elif _R21A_REGIME_VOL and len(daily_equity) >= 200:
            _use_smooth = getattr(_Cfg, 'SMOOTH_BEAR_DEFENSE', False) if _Cfg else False
            if _use_smooth:
                from services.volatility_target import smooth_regime_scale
                _steepness = getattr(_Cfg, 'SMOOTH_DEFENSE_STEEPNESS', 10.0) if _Cfg else 10.0
                _regime_mult = smooth_regime_scale(
                    equity, _eq_sma200,
                    boost=_R21A_REGIME_BOOST, defend=_R21A_REGIME_DEFEND,
                    steepness=_steepness,
                )
                dynamic_daily_target *= _regime_mult
            else:
                if equity > _eq_sma200 * 1.02:
                    dynamic_daily_target *= _R21A_REGIME_BOOST
                elif equity < _eq_sma200 * 0.98:
                    dynamic_daily_target *= _R21A_REGIME_DEFEND

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
        # Only runs when R22 is explicitly enabled — no detection/logging in R21A-only runs.
        if _R22_BULL_INFUSION and len(daily_equity) >= 200:
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

                # Log alert (for UI/email notification)
                _r22_date_str = ""
                try:
                    _r22_date_str = ohlcv_slice[symbols[0]]["Close"].index[day_idx].strftime("%Y-%m-%d")
                except Exception:
                    _r22_date_str = f"Day {_trading_day_num_r22}"
                _r22_alert_events.append((_trading_day_num_r22, _r22_date_str))

                if verbose:
                    print(f"Bull Confirmed Day {_trading_day_num_r22} ({_r22_date_str}): "
                          f"equity ₹{equity:,.0f} > SMA200 ₹{_r22_sma200:,.0f} "
                          f"for {_R22_BULL_CONFIRM_DAYS} consecutive days", flush=True)

                # Infuse capital if cooldown satisfied
                if _days_since_r22 >= _R22_INFUSION_COOLDOWN_DAYS:
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
                        print(f"Capital Infusion Day {_trading_day_num_r22}: "
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
            # C3 FIX: Pass rolling forecast history for empirical FDM
            _fh_for_sym = None
            _use_empirical_fdm = getattr(_Cfg, 'EMPIRICAL_FDM_ENABLED', False) if _Cfg else False
            if _use_empirical_fdm and _rolling_forecast_history:
                _fh_for_sym = {
                    src: list(vals) for src, vals in _rolling_forecast_history.items()
                    if len(vals) >= 30
                }
                if len(_fh_for_sym) < 3:
                    _fh_for_sym = None  # not enough history yet
            combined = combine_forecasts(
                sym, fc_dict, active_weights,
                forecast_history=_fh_for_sym,
                regime=_eq_regime,
            )
            fc_val = combined.combined_forecast
            _all_combined[sym] = fc_val

        # ── Godmode: Meta-labeling filter (AFML Ch.3) ──────────
        # Scale forecasts by meta-label probability to filter false signals.
        _meta_labeling_on = getattr(_Cfg, 'META_LABELING_ENABLED', False) if _Cfg else False
        if _meta_labeling_on and _recompute:
            try:
                from services.meta_labeling import apply_meta_labels
                _meta_min_prob = getattr(_Cfg, 'META_LABEL_MIN_PROBABILITY', 0.50) if _Cfg else 0.50
                _meta_result = apply_meta_labels(
                    combined_forecasts=_all_combined,
                    ohlcv_cache=ohlcv_slice,
                    min_probability=_meta_min_prob,
                    market=market,
                )
                _all_combined = dict(_meta_result.scaled_forecasts)
            except Exception as _ml_err:
                logger.debug("Meta-labeling skipped at day %d: %s", day_idx, _ml_err)

        # ── H3/M1: Signal strength gate — DISABLED (was filtering valid trades) ──
        # Keeping code but all thresholds at 0.0 so no filtering occurs
        _MIN_FORECAST_BULL = 0.0     # DISABLED: gate killed returns
        _MIN_FORECAST_NEUTRAL = 0.0  # DISABLED
        _MIN_FORECAST_BEAR = 0.0     # DISABLED
        if _eq_regime in ('severe_bear', 'bear'):
            _min_fc_gate = _MIN_FORECAST_BEAR
        elif _eq_regime in ('bull', 'strong_bull'):
            _min_fc_gate = _MIN_FORECAST_BULL
        else:
            _min_fc_gate = _MIN_FORECAST_NEUTRAL
        _gated_combined = {}
        for _gs, _gf in _all_combined.items():
            if abs(_gf) >= _min_fc_gate:
                _gated_combined[_gs] = _gf
            else:
                _gated_combined[_gs] = 0.0  # Zero forecast = no new position
        _all_combined = _gated_combined

        # ── M8 FIX: Distribution shift detector — scale down in regime breaks ──
        _dist_shift_on = getattr(_Cfg, 'DISTRIBUTION_SHIFT_ENABLED', True) if _Cfg else True
        if _dist_shift_on and len(daily_returns) >= 90:
            try:
                from services.distribution_shift import detect_distribution_shift
                _recent_rets = np.array(daily_returns[-60:])
                _baseline_rets = np.array(daily_returns[-252:-60]) if len(daily_returns) > 252 else np.array(daily_returns[:-60])
                if len(_baseline_rets) >= 30:
                    _shift = detect_distribution_shift(_baseline_rets, _recent_rets)
                    if _shift.get('verdict') == 'regime_break':
                        # Severe shift — halve all forecasts
                        _all_combined = {s: f * 0.50 for s, f in _all_combined.items()}
                    elif _shift.get('verdict') == 'drifting':
                        # Moderate shift — reduce by 25%
                        _all_combined = {s: f * 0.75 for s, f in _all_combined.items()}
            except Exception:
                pass

        # C3 FIX: Accumulate rolling forecast history for empirical FDM
        # Uses first symbol's per-source values as representative (cross-sectional avg)
        if _recompute:
            _src_avg: Dict[str, List[float]] = defaultdict(list)
            for sym, fc_dict in all_forecasts.items():
                for src, val in fc_dict.items():
                    if np.isfinite(val):
                        _src_avg[src].append(val)
            for src, vals in _src_avg.items():
                if src not in _rolling_forecast_history:
                    _rolling_forecast_history[src] = deque(maxlen=_ROLLING_FDM_LOOKBACK)
                _rolling_forecast_history[src].append(float(np.mean(vals)))

        # M5 FIX: Time-based exits — force close positions held too long
        _time_exit_on = getattr(_Cfg, 'TIME_EXIT_ENABLED', False) if _Cfg else False
        if _time_exit_on:
            _max_hold = 15  # default
            try:
                _max_hold = _Cfg.get_regime_hold_days(_eq_regime) if _Cfg else 15
            except Exception:
                pass
            for sym in list(prev_positions.keys()):
                if prev_positions.get(sym, 0) == 0:
                    continue
                if sym in entry_prices:
                    # entry_day tracked via day_idx offset
                    _held_days = _trading_day - entry_prices.get(f'_day_{sym}', _trading_day)
                    if _held_days >= _max_hold:
                        # Force exit stale position
                        if sym in ohlcv_slice:
                            _exit_c = ohlcv_slice[sym]["Close"]
                            if hasattr(_exit_c, "squeeze"):
                                _exit_c = _exit_c.squeeze()
                            _exit_price = float(_exit_c.iloc[-1])
                            day_pnl -= abs(prev_positions[sym]) * _exit_price * _sym_cost_map.get(sym, cost_pct)
                            trades_count += 1
                            if sym in entry_prices and entry_prices[sym] > 0:
                                ep = entry_prices.pop(sym)
                                if prev_positions[sym] > 0:
                                    trade_pnls.append((_exit_price - ep) / ep * 100)
                                else:
                                    trade_pnls.append((ep - _exit_price) / ep * 100)
                        prev_positions[sym] = 0
                        peak_prices.pop(sym, None)
                        stop_levels.pop(sym, None)

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
        # Phase C: Increased from 10 → 15 for broader signal capture on NIFTY500
        MAX_POSITIONS = 15

        MAX_HOLD_GRACE = MAX_POSITIONS + 7  # Grace zone scales with positions
        # Rank by conviction: in long-only mode, positive forecasts first (descending),
        # then negative by abs (for grace-zone exits). Prevents bearish stocks from
        # filling top slots and getting zeroed by long-only guard → zero-position trap.
        if not allow_short:
            # Phase 1 fix: split positive and negative — only positive can fill top slots
            _positive_ranked = sorted([(s, f) for s, f in _all_combined.items() if f > 0],
                                      key=lambda x: x[1], reverse=True)
            _negative_ranked = sorted([(s, f) for s, f in _all_combined.items() if f <= 0],
                                      key=lambda x: -abs(x[1]))
            _ranked = _positive_ranked + _negative_ranked
        else:
            _ranked = sorted(_all_combined.items(), key=lambda x: abs(x[1]), reverse=True)
        _top_syms_list = [s for s, _ in _ranked[:MAX_POSITIONS]]    # ordered list for min-1-share
        _top_syms = set(_top_syms_list)                             # set for O(1) membership
        _grace_syms = set(s for s, _ in _ranked[:MAX_HOLD_GRACE])   # top-30 → held positions stay

        # ── Godmode: Sector enforcement — max N stocks per sector ──
        _sector_enforce = getattr(_Cfg, 'SECTOR_ENFORCEMENT_ENABLED', False) if _Cfg else False
        if _sector_enforce:
            _max_per_sector = getattr(_Cfg, 'MAX_STOCKS_PER_SECTOR', 3) if _Cfg else 3
            _sector_map = getattr(_Cfg, 'NSE_SECTOR_MAP', {}) if _Cfg else {}
            _sector_counts: Dict[str, int] = defaultdict(int)
            _filtered_top: List[str] = []
            for s, _ in _ranked:
                _sym_clean = s.replace('.NS', '').replace('.BO', '')
                _sector = _sector_map.get(_sym_clean, 'Unknown')
                if _sector_counts[_sector] < _max_per_sector:
                    _filtered_top.append(s)
                    _sector_counts[_sector] += 1
                if len(_filtered_top) >= MAX_POSITIONS:
                    break
            _top_syms_list = _filtered_top
            _top_syms = set(_top_syms_list)

        # R8 CRITICAL FIX: Dynamic weight_per_sym based on ACTUAL investable count
        # Phase C: Lowered threshold from 2.0 → 1.0 to fill more position slots
        _investable = [s for s in _top_syms if abs(_all_combined.get(s, 0)) > 1.0]
        n_investable = max(5, min(len(_investable), MAX_POSITIONS))

        # ── Godmode: Forecast-proportional sizing (replaces flat 1/N) ──
        _fc_prop_sizing = getattr(_Cfg, 'FORECAST_PROPORTIONAL_SIZING', False) if _Cfg else False
        _fc_sizing_floor = getattr(_Cfg, 'FORECAST_SIZING_FLOOR', 3.0) if _Cfg else 3.0
        _sym_weights: Dict[str, float] = {}
        if _fc_prop_sizing and _investable:
            _abs_fcs = {s: max(abs(_all_combined.get(s, 0)), _fc_sizing_floor) for s in _investable}
            _total_fc = sum(_abs_fcs.values())
            if _total_fc > 0:
                for s in _investable:
                    _sym_weights[s] = _abs_fcs[s] / _total_fc
            else:
                for s in _investable:
                    _sym_weights[s] = 1.0 / n_investable
            weight_per_sym = 1.0  # weight_per_sym is now per-sym via _sym_weights
        else:
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
                        day_pnl -= abs(_exit_qty) * _exit_price * _sym_cost_map.get(sym, cost_pct)
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
                # Godmode: use per-symbol weight if forecast-proportional sizing is on
                _w = _sym_weights.get(sym, weight_per_sym) if _fc_prop_sizing else weight_per_sym
                position = (forecast / 10.0) * vol_scalar * _w * idm

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
                    # Godmode: Cost-aware inertia — only trade if expected alpha > N × cost
                    _cost_aware = getattr(_Cfg, 'COST_AWARE_INERTIA', False) if _Cfg else False
                    if _cost_aware and abs(prev_qty) > 0:
                        # H5 FIX: Regime-adaptive alpha-cost ratio
                        # Bear: require 4× alpha/cost (more conservative trading)
                        # Bull: require 1.5× (trade more freely)
                        if _eq_regime in ('severe_bear', 'bear'):
                            _alpha_cost_ratio = 4.0  # H5: very conservative in bear
                        elif _eq_regime in ('bull', 'strong_bull'):
                            _alpha_cost_ratio = 1.5  # H5: permissive in bull
                        else:
                            _alpha_cost_ratio = 2.5  # H5: moderate in neutral/sideways
                        _expected_cost = delta * price * _sym_cost_map.get(sym, cost_pct)
                        # H3 FIX: Correct alpha estimate — forecast-implied daily PnL
                        # alpha = |forecast/10| × vol_target_weight × daily_vol_target
                        _w = _sym_weights.get(sym, weight_per_sym) if _fc_prop_sizing else weight_per_sym
                        _expected_alpha = abs(forecast / 10.0) * _w * dynamic_daily_target
                        if _expected_alpha < _alpha_cost_ratio * _expected_cost:
                            target_qty = prev_qty  # cost exceeds alpha — skip trade
                        else:
                            cost = delta * price * _sym_cost_map.get(sym, cost_pct)
                            day_pnl -= cost
                            trades_count += 1
                    else:
                        # Legacy: fixed 20% inertia threshold
                        _inertia_pct = 0.20
                        if abs(prev_qty) > 0 and delta / abs(prev_qty) < _inertia_pct:
                            target_qty = prev_qty
                        else:
                            cost = delta * price * _sym_cost_map.get(sym, cost_pct)
                            day_pnl -= cost
                            trades_count += 1

                # ── Per-trade tracking (round-trip PnL) ───────
                if prev_qty == 0 and target_qty != 0:
                    # New position opened — record entry price + day
                    entry_prices[sym] = price
                    entry_prices[f'_day_{sym}'] = _trading_day  # M5: track holding period
                    # Phase 4: Execution gap penalty — new entry fills at next-day open (approx 50bps penalty)
                    if _EXECUTION_GAP_ENABLED:
                        _gap_penalty = abs(target_qty) * price * _EXECUTION_GAP_BPS
                        day_pnl -= _gap_penalty
                elif prev_qty != 0 and target_qty == 0:
                    # Position fully closed — log round-trip PnL
                    if sym in entry_prices and entry_prices[sym] > 0:
                        ep = entry_prices.pop(sym)
                        if prev_qty > 0:
                            trade_pnls.append((price - ep) / ep * 100)
                        else:
                            trade_pnls.append((ep - price) / ep * 100)
                    # Phase 4: Execution gap penalty on full exits too
                    if _EXECUTION_GAP_ENABLED:
                        _gap_penalty = abs(prev_qty) * price * _EXECUTION_GAP_BPS
                        day_pnl -= _gap_penalty
                elif prev_qty != 0 and target_qty != 0 and abs(target_qty) != abs(prev_qty):
                    # Position resized — update VWAP entry
                    if sym in entry_prices and abs(target_qty) > abs(prev_qty):
                        added = abs(target_qty) - abs(prev_qty)
                        entry_prices[sym] = (entry_prices[sym] * abs(prev_qty) + price * added) / abs(target_qty)

                prev_positions[sym] = target_qty

                # M3 FIX: Tighter stops overall — current stops too wide (49.9% DD)
                if _Cfg:
                    if _eq_regime == 'severe_bear' or _eq_regime == 'bear':
                        stop_sigma = getattr(_Cfg, 'STOP_SIGMA_BEAR', 2.0)
                    elif _eq_regime == 'strong_bull':
                        stop_sigma = getattr(_Cfg, 'STOP_SIGMA_STRONG_TREND', 4.0)  # M3: tightened from 5.0
                    elif _eq_regime == 'bull':
                        stop_sigma = getattr(_Cfg, 'STOP_SIGMA_BULL', 2.5)  # M3: tightened from 3.0
                    else:
                        stop_sigma = getattr(_Cfg, 'STOP_SIGMA_NEUTRAL', 3.0)  # M3: tightened from 4.0
                else:
                    stop_sigma = 3.0  # M3: tightened from 5.0

                # V2: Bull profit-taker — tighter stops in uptrend to book profits (Harvest only)
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

        # Phase 1: Min-1-position enforcement — avoid 0-position days entirely
        if active_count == 0 and not allow_short:
            # Find highest positive-forecast symbol from top_syms with valid price
            for _m1_sym in _top_syms_list:
                _m1_fc = _all_combined.get(_m1_sym, 0.0)
                if _m1_fc <= 0:
                    continue
                if _m1_sym not in ohlcv_slice:
                    continue
                _m1_c = ohlcv_slice[_m1_sym]["Close"]
                if hasattr(_m1_c, "squeeze"):
                    _m1_c = _m1_c.squeeze()
                _m1_close = _m1_c.dropna()
                if len(_m1_close) == 0:
                    continue
                _m1_price = float(_m1_close.iloc[-1])
                if not np.isfinite(_m1_price) or _m1_price <= 0:
                    continue
                # Force minimum 1-share position
                prev_positions[_m1_sym] = 1
                entry_prices[_m1_sym] = _m1_price
                entry_prices[f'_day_{_m1_sym}'] = _trading_day
                daily_position_counts[-1] = 1
                break

        # ── Godmode: Dynamic leverage per regime (uses _eq_regime) ──
        _dyn_lev = getattr(_Cfg, 'DYNAMIC_LEVERAGE_ENABLED', False) if _Cfg else False
        _effective_max_lev = max_leverage
        if _dyn_lev:
            if _eq_regime in ('severe_bear', 'bear'):
                _effective_max_lev = getattr(_Cfg, 'LEVERAGE_BEAR', 1.0) if _Cfg else 1.0
            elif _eq_regime == 'strong_bull':
                _effective_max_lev = getattr(_Cfg, 'LEVERAGE_BULL_CONFIRMED', 2.5) if _Cfg else 2.5
            elif _eq_regime == 'bull':
                _effective_max_lev = getattr(_Cfg, 'LEVERAGE_NEUTRAL', 2.0) if _Cfg else 2.0
            else:
                _effective_max_lev = getattr(_Cfg, 'LEVERAGE_NEUTRAL', 2.0) if _Cfg else 2.0

        # M7 FIX: VIX-like leverage scaler — realized vol proxy from portfolio
        # Use cross-sectional 20-day realized vol of top positions as India VIX proxy.
        # When vol > CAUTION → reduce leverage by VIX_POSITION_SCALE; PANIC → halve again.
        _vix_scaling = getattr(_Cfg, 'VIX_PIPELINE_SCALING_ENABLED', False) if _Cfg else False
        if _vix_scaling and len(ohlcv_slice) >= 5:
            _vix_rets = []
            for _vs, _vdf in list(ohlcv_slice.items())[:20]:
                _vc = _vdf["Close"]
                if hasattr(_vc, "squeeze"):
                    _vc = _vc.squeeze()
                if len(_vc) >= 25:
                    _vr = _vc.pct_change().dropna().iloc[-20:]
                    if len(_vr) >= 15:
                        _vix_rets.append(float(_vr.std()) * 16.0 * 100)  # annualized vol %
            if _vix_rets:
                _realized_vix = float(np.median(_vix_rets))  # median across stocks
                _caution = getattr(_Cfg, 'VIX_CAUTION_THRESHOLD', 20.0) if _Cfg else 20.0
                _panic = getattr(_Cfg, 'VIX_PANIC_THRESHOLD', 30.0) if _Cfg else 30.0
                _kill = getattr(_Cfg, 'KILL_SWITCH_VIX_THRESHOLD', 40.0) if _Cfg else 40.0
                _vix_scale = getattr(_Cfg, 'VIX_POSITION_SCALE', 0.5) if _Cfg else 0.5
                if _realized_vix >= _kill:
                    _effective_max_lev = 1.0  # kill switch: delever to 1×
                elif _realized_vix >= _panic:
                    _effective_max_lev *= _vix_scale * 0.5  # severe scale-down
                elif _realized_vix >= _caution:
                    _effective_max_lev *= _vix_scale  # cautious scale-down

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
        max_total_exposure = max(equity, capital * 0.10) * _effective_max_lev
        if total_exposure > max_total_exposure and total_exposure > 0:
            scale_down = max_total_exposure / total_exposure
            for sym in list(prev_positions.keys()):
                if prev_positions[sym] != 0:
                    prev_positions[sym] = round(prev_positions[sym] * scale_down)

        # M6 FIX: Multi-asset allocation cap — non-equity assets capped at 15%
        if _multi_asset_on:
            _ma_cap = getattr(_Cfg, 'MULTI_ASSET_MAX_ALLOCATION', 0.15) if _Cfg else 0.15
            _ma_set = set(getattr(_Cfg, 'MULTI_ASSET_TICKERS_IND', []) if _Cfg else [])
            if _ma_set:
                _ma_exposure = 0.0
                for sym, qty in prev_positions.items():
                    if qty == 0 or sym not in _ma_set or sym not in ohlcv_slice:
                        continue
                    c = ohlcv_slice[sym]["Close"]
                    if hasattr(c, "squeeze"):
                        c = c.squeeze()
                    _ma_exposure += abs(qty) * float(c.dropna().iloc[-1]) if len(c.dropna()) > 0 else 0
                _ma_max = max(equity, capital * 0.10) * _ma_cap
                if _ma_exposure > _ma_max and _ma_exposure > 0:
                    _ma_scale = _ma_max / _ma_exposure
                    for sym in list(prev_positions.keys()):
                        if sym in _ma_set and prev_positions[sym] != 0:
                            prev_positions[sym] = round(prev_positions[sym] * _ma_scale)

        # ── 4. Update equity ───────────────────────────────────
        if not np.isfinite(day_pnl):
            day_pnl = 0.0  # NaN guard: skip corrupted days
        equity += day_pnl
        _day_ret = day_pnl / max(daily_equity[-1], 1)
        daily_returns.append(_day_ret)
        daily_equity.append(equity)

        # PBO-FIX: Attribute daily return to signals by forecast weight share
        # For each signal, its daily attributed return = day_ret × (signal's avg
        # weighted forecast contribution across all held symbols / total combined).
        # If a signal disagreed with the final position direction, its contribution
        # is effectively negative — faithful to actual portfolio construction.
        try:
            _day_fc_contribs: Dict[str, float] = defaultdict(float)
            _day_fc_total = 0.0
            for _sym, _fc_dict in all_forecasts.items():
                if not _fc_dict or _sym not in _all_combined:
                    continue
                _comb = _all_combined[_sym]
                if abs(_comb) < 1e-9:
                    continue
                _pos = prev_positions.get(_sym, 0)
                if _pos == 0:
                    continue
                # Weight each signal's contribution by its normalized share
                for _fw in active_weights:
                    _src_fc = _fc_dict.get(_fw.name, 0.0)
                    _weighted = _src_fc * _fw.weight
                    _day_fc_contribs[_fw.name] += _weighted
                    _day_fc_total += abs(_weighted)
            if _day_fc_total > 1e-9 and abs(_day_ret) > 0:
                for _src_name, _src_contrib in _day_fc_contribs.items():
                    _frac = _src_contrib / _day_fc_total
                    source_daily_returns[_src_name].append(_day_ret * _frac)
                # Fill 0 for signals that didn't contribute today
                for _fw in active_weights:
                    if _fw.name not in _day_fc_contribs:
                        source_daily_returns[_fw.name].append(0.0)
            else:
                for _fw in active_weights:
                    source_daily_returns[_fw.name].append(0.0)
        except Exception:
            pass

        _consecutive_day_errors = 0  # reset on successful day

        # Print progress every N days (L1: configurable checkpoint)
        _ckpt_interval = getattr(_Cfg, 'CHECKPOINT_INTERVAL_DAYS', 50) if _Cfg else 50
        if verbose and (day_idx - min_history) % _ckpt_interval == 0:
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
                    'true_peak_equity': _true_peak_equity,  # H1: persist true peak
                    'dd_deep_days': dd_deep_days,
                    'cached_forecasts': {s: dict(f) for s, f in _cached_forecasts.items()},
                    'cached_idm': _cached_idm,
                    'trade_pnls': list(trade_pnls),
                    'entry_prices': dict(entry_prices),
                    'source_daily_returns': {k: list(v) for k, v in source_daily_returns.items()},
                }
                if _SAVE_FORECASTS_MODE:
                    _ckpt_data['forecast_log'] = list(_forecast_log)
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
                    'true_peak_equity': _true_peak_equity,  # H1
                    'dd_deep_days': dd_deep_days,
                    'cached_forecasts': {s: dict(f) for s, f in _cached_forecasts.items()},
                    'cached_idm': _cached_idm,
                    'trade_pnls': list(trade_pnls),
                    'entry_prices': dict(entry_prices),
                    'source_daily_returns': {k: list(v) for k, v in source_daily_returns.items()},
                }
                if _SAVE_FORECASTS_MODE:
                    _ckpt_data['forecast_log'] = list(_forecast_log)
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
                    'true_peak_equity': _true_peak_equity,  # H1
                    'dd_deep_days': dd_deep_days,
                    'cached_forecasts': {s: dict(f) for s, f in _cached_forecasts.items()},
                    'cached_idm': _cached_idm,
                    'trade_pnls': list(trade_pnls),
                    'entry_prices': dict(entry_prices),
                    'source_daily_returns': {k: list(v) for k, v in source_daily_returns.items()},
                }
                with open(_checkpoint_path, 'wb') as _ckf:
                    pickle.dump(_ckpt_data, _ckf, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass
            break

    # Restore original signal handlers
    try:
        if _prev_sigint is not None:
            signal.signal(signal.SIGINT, _prev_sigint)
        if _prev_sigterm is not None:
            signal.signal(signal.SIGTERM, _prev_sigterm)
    except ValueError:
        pass  # Not on main thread

    # Cancel deadline timer if still running
    if _deadline_timer is not None:
        _deadline_timer.cancel()

    # Clean up checkpoint on successful completion (only if NOT interrupted)
    if not _graceful_exit_requested and os.path.exists(_checkpoint_path):
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
    # C1 FIX: PBO/CSCV computation
    # C2 FIX: Alpha-beta separation
    _alpha_beta_result = {}
    _pbo_result = {}
    try:
        from services.aronson_validator import (
            detrend_returns, trimmed_sharpe as _trimmed_sharpe,
            compute_signal_tstat, estimate_data_mining_bias,
            compute_pbo, compute_alpha_beta,
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

        # C2 FIX: Alpha-Beta Separation — measure true alpha vs market beta
        # Download NIFTY50 benchmark for IND market
        try:
            _nifty_ticker = getattr(_Cfg, 'NIFTY_BENCHMARK_TICKER', '^NSEI') if _Cfg else '^NSEI'
            if market == "IND":
                import yfinance as yf
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _bench_df = yf.download(_nifty_ticker, start=start_date,
                                            end=end_date or None,
                                            auto_adjust=True, progress=False)
                if _bench_df is not None and len(_bench_df) > 100:
                    _bench_close = _bench_df["Close"]
                    if hasattr(_bench_close, "squeeze"):
                        _bench_close = _bench_close.squeeze()
                    _bench_rets = _bench_close.pct_change().dropna().values
                    _alpha_beta_result = compute_alpha_beta(
                        np.array(daily_returns), _bench_rets,
                    )
        except Exception as _ab_err:
            logger.debug("Alpha-beta computation failed: %s", _ab_err)

        # C1 FIX: PBO/CSCV — Probability of Backtest Overfitting
        # PBO-FIX: Use actual per-signal daily return attribution (not data availability)
        try:
            _sdr_signals = [src for src in sorted(source_daily_returns.keys())
                            if len(source_daily_returns[src]) >= 60]
            if len(_sdr_signals) >= 4:
                _min_len = min(len(source_daily_returns[s]) for s in _sdr_signals)
                _returns_matrix = np.array([
                    source_daily_returns[s][:_min_len] for s in _sdr_signals
                ])
                _pbo_result = compute_pbo(_returns_matrix, n_partitions=10)
                if verbose:
                    logger.info("PBO computed from %d signals × %d days: %.1f%% (%s)",
                                len(_sdr_signals), _min_len,
                                _pbo_result.get('pbo_pct', 0),
                                _pbo_result.get('interpretation', 'N/A'))
            elif source_total and len(source_total) >= 4:
                # Fallback: old method (data-availability-based) if per-signal returns unavailable
                _n_sigs = len(source_total)
                _T = len(daily_returns)
                _strat_returns = []
                _strat_names = []
                for src in sorted(source_total.keys()):
                    total = source_total[src]
                    hits = source_hits.get(src, 0)
                    if total > 10:
                        _hit_rate = hits / total if total > 0 else 0.5
                        rng_pbo = np.random.RandomState(hash(src) % (2**31))
                        _syn_rets = np.where(
                            rng_pbo.random(_T) < _hit_rate,
                            abs(np.mean(daily_returns)) if daily_returns else 0.001,
                            -abs(np.mean(daily_returns)) if daily_returns else -0.001,
                        )
                        _strat_returns.append(_syn_rets)
                        _strat_names.append(src)
                if len(_strat_returns) >= 4:
                    _returns_matrix = np.array(_strat_returns)
                    _pbo_result = compute_pbo(_returns_matrix, n_partitions=10)
                    _pbo_result['method'] = 'data_availability_fallback'
        except Exception as _pbo_err:
            logger.debug("PBO computation failed: %s", _pbo_err)

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

        # Bootstrap CI for Sharpe (circular block bootstrap — preserves autocorrelation)
        if len(ret_arr) > 30:
            rng = np.random.RandomState(42)
            boot_sharpes = []
            _block_len = getattr(_Cfg, 'BOOTSTRAP_BLOCK_LENGTH', 20) if _Cfg else 20
            _n_ret = len(ret_arr)
            for _ in range(1000):
                # Circular block bootstrap: draw blocks of _block_len, wrap around
                _n_blocks = int(np.ceil(_n_ret / _block_len))
                _starts = rng.randint(0, _n_ret, size=_n_blocks)
                _boot_idx = []
                for _st in _starts:
                    _boot_idx.extend([(_st + j) % _n_ret for j in range(_block_len)])
                _boot_idx = _boot_idx[:_n_ret]
                _b = ret_arr[_boot_idx]
                _bm = float(np.mean(_b))
                _bs = float(np.std(_b, ddof=1))
                if _bs > 0:
                    boot_sharpes.append(_bm / _bs * 16.0)
            if boot_sharpes:
                result.bootstrap_ci_sharpe = (
                    round(float(np.percentile(boot_sharpes, 5)), 3),
                    round(float(np.percentile(boot_sharpes, 95)), 3),
                )

            # L5 FIX: Turnover-penalized bootstrap Sharpe
            # Penalize bootstrap resamples by estimated turnover drag
            _tp_lambda = getattr(_Cfg, 'TURNOVER_PENALTY_LAMBDA', 0.005) if _Cfg else 0.005
            if _tp_lambda > 0 and trades_count > 0 and n_days > min_history:
                _days_traded = n_days - min_history
                _daily_turnover = trades_count / _days_traded  # avg trades/day
                _tp_drag = _tp_lambda * _daily_turnover  # daily penalty
                tp_boot_sharpes = []
                for _bs_val in boot_sharpes:
                    # Subtract turnover drag from annualized Sharpe
                    tp_boot_sharpes.append(_bs_val - _tp_drag * 16.0)
                if tp_boot_sharpes:
                    result.turnover_penalized_ci = (
                        round(float(np.percentile(tp_boot_sharpes, 5)), 3),
                        round(float(np.percentile(tp_boot_sharpes, 95)), 3),
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

    # Aronson EBTA enrichment + C1 PBO + C2 Alpha-Beta
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
        # C1: PBO/CSCV results
        if _pbo_result:
            _pbo_method = _pbo_result.get('method', 'per_signal_daily_pnl')
            lines.append(f"")
            lines.append(f"  {'─'*40}")
            lines.append(f"  C1: Probability of Backtest Overfitting (CSCV):")
            lines.append(f"  PBO:               {_pbo_result.get('pbo_pct', 0):.1f}%")
            lines.append(f"  CSCV Combinations: {_pbo_result.get('n_combinations', 0)}")
            lines.append(f"  Median Logit:      {_pbo_result.get('median_logit', 0):.3f}")
            lines.append(f"  Method:            {_pbo_method}")
            lines.append(f"  Interpretation:    {_pbo_result.get('interpretation', 'N/A')}")
        # C2: Alpha-Beta decomposition
        if _alpha_beta_result:
            lines.append(f"")
            lines.append(f"  {'─'*40}")
            lines.append(f"  C2: Alpha-Beta Decomposition (vs {getattr(_Cfg, 'NIFTY_BENCHMARK_TICKER', '^NSEI') if _Cfg else '^NSEI'}):")
            lines.append(f"  Beta:              {_alpha_beta_result.get('beta', 0):.3f}")
            lines.append(f"  Alpha (ann. %):    {_alpha_beta_result.get('alpha_annual_pct', 0):.3f}%")
            lines.append(f"  Alpha Sharpe:      {_alpha_beta_result.get('alpha_sharpe', 0):.3f}")
            lines.append(f"  R-squared:         {_alpha_beta_result.get('r_squared', 0):.3f}")
            lines.append(f"  Info Ratio:        {_alpha_beta_result.get('information_ratio', 0):.3f}")
            lines.append(f"  Beta Contrib (%):  {_alpha_beta_result.get('beta_contribution_pct', 0):.1f}%")

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
        # C1: PBO/CSCV results
        "pbo": _pbo_result if _pbo_result else None,
        # C2: Alpha-beta decomposition
        "alpha_beta": _alpha_beta_result if _alpha_beta_result else None,
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
