"""
US Stocks Carver Pipeline — Adapts the Carver Systematic Trading framework
for the US equity market (USD-based, DriveWealth execution).

Mirrors the IND pipeline (services/carver_pipeline.py) with US-specific:
  - USD capital base and cost config
  - yfinance OHLCV (no .NS suffix)
  - US sector classification
  - Lower transaction costs (zero-commission brokers)

Steps:
  1. Download OHLCV via yfinance for US tickers
  2. Compute instrument volatilities
  3. Run EWMAC forecasts (16/64, 32/128, 64/256 — swing-appropriate)
  4. Convert decision engine scores → Carver forecasts (-20/+20)
  5. Combine forecasts with FDM
  6. Apply cost speed limit filter (US cost config)
  7. Compute handcrafted weights + IDM
  8. Size positions via vol-targeting
  9. Generate trade plan dicts for DriveWealth execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ── US Sector Map (GICS-inspired top-level sectors) ──────────────

US_SECTOR_MAP: Dict[str, str] = {
    # Tech
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "ADBE": "Technology", "CSCO": "Technology",
    "AVGO": "Technology", "QCOM": "Technology", "TXN": "Technology",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "NKE": "Consumer",
    "SBUX": "Consumer", "MCD": "Consumer", "HD": "Consumer",
    "WMT": "Consumer", "COST": "Consumer", "TGT": "Consumer",
    "PG": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "BMY": "Healthcare",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "C": "Financials", "AXP": "Financials", "V": "Financials",
    "MA": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "DE": "Industrials", "LMT": "Industrials",
    # Communication
    "DIS": "Communication", "NFLX": "Communication", "CMCSA": "Communication",
    "T": "Communication", "VZ": "Communication",
}

# Default US tickers for Carver analysis (diversified top-20)
DEFAULT_US_CARVER_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JNJ", "JPM", "V",
    "UNH", "HD", "PG", "XOM", "MA",
    "BAC", "ABBV", "KO", "PFE", "COST",
]


def get_us_universe(mode: Optional[str] = None) -> List[str]:
    """
    Return the US stock universe based on mode.

    Parameters
    ----------
    mode : str | None
        ``"DEFAULT"`` (top-20), ``"NASDAQ100"`` (~100),
        ``"SP500"`` (~500), ``"NASDAQ_FULL"`` (~3000+).
        If ``None``, reads from ``Config.US_UNIVERSE_MODE``.

    Returns
    -------
    list[str]
        US ticker symbols.
    """
    if mode is None:
        try:
            from config import Config
            mode = getattr(Config, "US_UNIVERSE_MODE", "NASDAQ_FULL")
        except Exception:
            mode = "NASDAQ_FULL"

    mode = mode.upper().strip()

    if mode == "DEFAULT":
        return list(DEFAULT_US_CARVER_TICKERS)

    if mode == "NASDAQ100":
        tickers = _fetch_nasdaq100()
        return tickers if tickers else list(DEFAULT_US_CARVER_TICKERS)

    if mode == "SP500":
        tickers = _fetch_sp500()
        return tickers if tickers else _fetch_nasdaq100() or list(DEFAULT_US_CARVER_TICKERS)

    # NASDAQ_FULL — get all NASDAQ-listed stocks
    tickers = _fetch_nasdaq_full()
    if len(tickers) >= 100:
        return tickers
    # Fallback chain
    tickers = _fetch_sp500()
    if tickers:
        return tickers
    tickers = _fetch_nasdaq100()
    return tickers if tickers else list(DEFAULT_US_CARVER_TICKERS)


def _fetch_nasdaq100() -> List[str]:
    """Fetch NASDAQ-100 constituents from Wikipedia."""
    try:
        import io, requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), match="Ticker")
        if tables:
            df = tables[0]
            col = "Ticker" if "Ticker" in df.columns else df.columns[1]
            tickers = sorted(df[col].dropna().str.strip().unique().tolist())
            logger.info("Fetched %d NASDAQ-100 tickers", len(tickers))
            return tickers
    except Exception as exc:
        logger.warning("NASDAQ-100 fetch failed: %s", exc)
    return []


def _fetch_sp500() -> List[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    try:
        import io, requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), match="Symbol")
        if tables:
            df = tables[0]
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            tickers = sorted(
                df[col].dropna().str.strip().str.replace(".", "-", regex=False)
                .unique().tolist()
            )
            logger.info("Fetched %d S&P 500 tickers", len(tickers))
            return tickers
    except Exception as exc:
        logger.warning("S&P 500 fetch failed: %s", exc)
    return []


def _fetch_nasdaq_full() -> List[str]:
    """
    Fetch all NASDAQ-listed stocks via the NASDAQ screener API.

    Filters to common stocks only (no ETFs, warrants, rights).
    Applies minimum market cap ($100M) and price ($5) filters
    to exclude penny stocks and illiquid micro-caps.
    """
    import requests

    try:
        # NASDAQ screener API — returns JSON with all listed securities
        api_url = (
            "https://api.nasdaq.com/api/screener/stocks"
            "?tableonly=true&limit=25&offset=0&download=true"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        resp = requests.get(api_url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("data", {}).get("rows", [])
        if not rows:
            logger.warning("NASDAQ screener returned empty rows")
            return []

        tickers = []
        for row in rows:
            symbol = row.get("symbol", "").strip()
            if not symbol or "/" in symbol or "^" in symbol:
                continue
            # Filter: market cap >= $100M, price >= $5
            try:
                mcap_str = row.get("marketCap", "0").replace(",", "")
                price_str = row.get("lastsale", "$0").replace("$", "").replace(",", "")
                mcap = float(mcap_str) if mcap_str else 0
                price = float(price_str) if price_str else 0
                if mcap < 100_000_000 or price < 5.0:
                    continue
            except (ValueError, TypeError):
                continue
            tickers.append(symbol)

        tickers = sorted(set(tickers))
        logger.info("Fetched %d NASDAQ stocks (filtered: mcap>$100M, price>$5)", len(tickers))
        return tickers

    except Exception as exc:
        logger.warning("NASDAQ full fetch failed: %s", exc)
        return []


@dataclass
class USCostConfig:
    """Cost parameters for US equity trades (zero-commission brokers)."""
    round_trip_cost_pct: float = 0.0010   # 10 bps (SEC fee + minimal platform cost)
    spread_slippage_pct: float = 0.0005   # 5 bps for large-cap US equities
    speed_limit_factor: float = 3.0
    default_annual_turnover: float = 15.0


@dataclass
class USCarverResult:
    """Result from running the US Carver pipeline."""
    trade_plans: List[Dict[str, Any]] = field(default_factory=list)
    combined_forecasts: Dict[str, float] = field(default_factory=dict)
    instrument_vols: Dict[str, float] = field(default_factory=dict)
    instrument_weights: Dict[str, float] = field(default_factory=dict)
    idm: float = 1.0
    symbols_processed: int = 0
    symbols_filtered_by_cost: int = 0
    pipeline_log: List[str] = field(default_factory=list)
    # Aronson EBTA: analysis-only validation stats for US module
    validation_stats: Dict[str, Any] = field(default_factory=dict)


def run_us_carver_pipeline(
    tickers: List[str],
    capital: Optional[float] = None,
    annual_vol_target: Optional[float] = None,
) -> USCarverResult:
    """Run the full Carver systematic trading pipeline for US stocks.

    Parameters
    ----------
    tickers : list[str]
        US stock tickers (e.g. ["AAPL", "MSFT", "GOOGL"]).
    capital : float | None
        Starting capital in USD. Defaults to Config.CARVER_US_INITIAL_CAPITAL.
    annual_vol_target : float | None
        Annual vol target (fraction). Defaults to Config.CARVER_US_ANNUAL_VOL_TARGET.

    Returns
    -------
    USCarverResult
        Trade plans and pipeline metadata.
    """
    from config import Config
    from utils import download_us_ohlcv
    from services.instrument_volatility import compute_volatilities_batch
    from strategies.ewmac import compute_ewmac_batch
    from services.forecast_scalar import screener_to_forecast
    from services.forecast_combiner import combine_forecasts_batch
    from services.cost_speed_limit import CostConfig, filter_by_cost
    from services.instrument_weights import compute_handcrafted_weights, get_default_idm
    from services.volatility_target import VolatilityTarget, VolatilityTargetConfig
    from services.position_sizer import compute_position_size

    result = USCarverResult()
    cap = capital or getattr(Config, "CARVER_US_INITIAL_CAPITAL", 10_000.0)
    vol_tgt = annual_vol_target or getattr(Config, "CARVER_US_ANNUAL_VOL_TARGET", 0.20)
    max_lev = getattr(Config, "CARVER_US_MAX_LEVERAGE", 1.0)

    # ── Pre-check: Portfolio drawdown halt ────────────────────
    try:
        cum_pnl = getattr(Config, "_CUMULATIVE_REALIZED_PNL", 0.0)
        peak_eq = getattr(Config, "_PEAK_EQUITY", cap)
        current_eq = cap + cum_pnl
        if peak_eq > 0 and current_eq < peak_eq:
            dd_pct = (peak_eq - current_eq) / peak_eq * 100
            if dd_pct > 15.0:
                result.pipeline_log.append(f"HALT: US portfolio drawdown {dd_pct:.1f}% > 15% — no new trades")
                return result
            elif dd_pct > 10.0:
                result.pipeline_log.append(f"WARNING: US portfolio drawdown {dd_pct:.1f}% — position sizes halved")
    except Exception:
        pass

    # ── Step 1: Download OHLCV ───────────────────────────────
    result.pipeline_log.append(f"Step 1: Downloading OHLCV for {len(tickers)} US tickers")
    ohlcv_cache: Dict[str, pd.DataFrame] = {}

    # Use batch parallel download for large universes
    if len(tickers) > 30:
        try:
            from utils import download_ohlcv_batch_parallel
            ohlcv_cache = download_ohlcv_batch_parallel(
                tickers, market="US", period="6mo",
            )
            # Filter to minimum bar count
            ohlcv_cache = {s: d for s, d in ohlcv_cache.items() if len(d) >= 64}
        except Exception as exc:
            result.pipeline_log.append(f"  → Batch download failed ({exc}), falling back to sequential")
            ohlcv_cache = {}

    # Fallback: sequential download for remaining
    if not ohlcv_cache:
        for sym in tickers:
            try:
                df = download_us_ohlcv(sym, period="6mo")
                if df is not None and len(df) >= 64:
                    ohlcv_cache[sym] = df
            except Exception:
                pass

    result.pipeline_log.append(f"  → {len(ohlcv_cache)}/{len(tickers)} tickers have sufficient data")
    if not ohlcv_cache:
        result.pipeline_log.append("ABORT: No OHLCV data available")
        return result

    # ── Step 2: Compute volatilities ─────────────────────────
    result.pipeline_log.append("Step 2: Computing instrument volatilities")
    vol_data = compute_volatilities_batch(ohlcv_cache)
    result.instrument_vols = {s: v["instrument_value_vol"] for s, v in vol_data.items()}
    result.pipeline_log.append(f"  → {len(result.instrument_vols)} volatilities computed")

    # ── Step 3: EWMAC forecasts ──────────────────────────────
    result.pipeline_log.append("Step 3: Computing EWMAC forecasts")
    ewmac_batch = compute_ewmac_batch(ohlcv_cache)

    # ── Step 3b: Carry forecasts (US funding ~5.25%) ─────────
    carry_batch = {}
    try:
        from strategies.carry_rule import compute_carry_batch
        us_funding = 0.0525  # US Fed Funds rate
        carry_batch = compute_carry_batch(ohlcv_cache, funding_cost=us_funding)
        result.pipeline_log.append(f"  Carry: {len(carry_batch)}/{len(ohlcv_cache)} symbols")
    except Exception:
        result.pipeline_log.append("  Carry: unavailable (non-fatal)")

    # ── G10: Momentum factor forecasts ───────────────────────
    momentum_forecasts: Dict[str, float] = {}
    try:
        from services.momentum_factor import compute_momentum_forecasts
        momentum_forecasts = compute_momentum_forecasts(ohlcv_cache)
        if momentum_forecasts:
            result.pipeline_log.append(f"  Momentum: {len(momentum_forecasts)} symbols")
    except Exception:
        result.pipeline_log.append("  Momentum: unavailable (non-fatal)")

    # ── G10: Mean-reversion forecasts ────────────────────────
    mean_rev_forecasts: Dict[str, float] = {}
    try:
        from strategies.mean_reversion import compute_mean_reversion_batch
        mean_rev_forecasts = compute_mean_reversion_batch(ohlcv_cache)
        if mean_rev_forecasts:
            result.pipeline_log.append(f"  Mean-reversion: {len(mean_rev_forecasts)} symbols")
    except Exception:
        result.pipeline_log.append("  Mean-reversion: unavailable (non-fatal)")

    # ── Penfold trend tactics: Turtle + ATR band + retracement + weekly Dow ──
    penfold_forecasts: Dict[str, float] = {}
    penfold_weekly_trends: Dict[str, str] = {}
    try:
        from strategies.penfold_trend import (
            compute_penfold_forecast_batch,
            compute_weekly_trend_filter_batch,
        )
        penfold_forecasts = compute_penfold_forecast_batch(ohlcv_cache)
        penfold_weekly_trends = compute_weekly_trend_filter_batch(ohlcv_cache)
        if penfold_forecasts:
            result.pipeline_log.append(f"  Penfold trend: {len(penfold_forecasts)} symbols")
    except Exception as exc:
        result.pipeline_log.append(f"  Penfold trend: skipped ({exc})")

    # ── Ehlers DSP: Fisher Transform + MAMA/FAMA + Super Smoother + Sinewave ──
    ehlers_forecasts: Dict[str, float] = {}
    try:
        from strategies.ehlers_dsp import compute_ehlers_forecast_batch
        ehlers_forecasts = compute_ehlers_forecast_batch(ohlcv_cache)
        if ehlers_forecasts:
            result.pipeline_log.append(f"  Ehlers DSP: {len(ehlers_forecasts)} symbols")
    except Exception as exc:
        result.pipeline_log.append(f"  Ehlers DSP: skipped ({exc})")

    # ── Ruggiero Cybernetic: intermarket + seasonal + trend strength + MTF ──
    cybernetic_forecasts: Dict[str, float] = {}
    try:
        from strategies.ruggiero_cybernetic import (
            compute_cybernetic_forecast_batch,
            US_INTERMARKET_DRIVERS,
        )
        import yfinance as _yf_drivers
        driver_dfs: Dict[str, pd.DataFrame] = {}
        for driver_sym in US_INTERMARKET_DRIVERS:
            try:
                d = _yf_drivers.download(driver_sym, period="120d", progress=False)
                if d is not None and len(d) >= 30:
                    driver_dfs[driver_sym] = d
            except Exception:
                pass
        if driver_dfs:
            cybernetic_forecasts = compute_cybernetic_forecast_batch(
                ohlcv_cache, driver_dfs, US_INTERMARKET_DRIVERS
            )
            if cybernetic_forecasts:
                result.pipeline_log.append(
                    f"  Cybernetic intermarket: {len(cybernetic_forecasts)} symbols "
                    f"({len(driver_dfs)} drivers)")
    except Exception as exc:
        result.pipeline_log.append(f"  Cybernetic intermarket: skipped ({exc})")

    # ── AFTS S23: Acceleration — rate of change of EWMAC forecast ──
    acceleration_forecasts: Dict[str, float] = {}
    try:
        from strategies.acceleration import compute_acceleration_batch
        acceleration_forecasts = compute_acceleration_batch(ohlcv_cache)
        if acceleration_forecasts:
            result.pipeline_log.append(f"  Acceleration (S23): {len(acceleration_forecasts)} symbols")
    except Exception as exc:
        result.pipeline_log.append(f"  Acceleration: skipped ({exc})")

    # ── AFTS S22: Carver Value — 5-year mean reversion ──
    value_forecasts: Dict[str, float] = {}
    try:
        from strategies.carver_value import compute_value_batch
        value_forecasts = compute_value_batch(ohlcv_cache)
        if value_forecasts:
            result.pipeline_log.append(f"  Carver Value (S22): {len(value_forecasts)} symbols")
    except Exception as exc:
        result.pipeline_log.append(f"  Carver Value: skipped ({exc})")

    # ── AFTS S24: Skew Signal — realized skew risk premium ──
    skew_forecasts: Dict[str, float] = {}
    try:
        from strategies.skew_signal import compute_skew_batch
        skew_forecasts = compute_skew_batch(ohlcv_cache)
        if skew_forecasts:
            result.pipeline_log.append(f"  Skew Signal (S24): {len(skew_forecasts)} symbols")
    except Exception as exc:
        result.pipeline_log.append(f"  Skew Signal: skipped ({exc})")

    # ── Sentiment Forecast — news-driven signal ──
    sentiment_forecasts: Dict[str, float] = {}
    try:
        from services.sentiment_forecast import compute_sentiment_batch
        sentiment_forecasts = compute_sentiment_batch(ohlcv_cache)
        if sentiment_forecasts:
            result.pipeline_log.append(f"  Sentiment: {len(sentiment_forecasts)} symbols")
    except Exception as exc:
        result.pipeline_log.append(f"  Sentiment: skipped ({exc})")

    # ── Step 4: Build forecast dicts ─────────────────────────
    result.pipeline_log.append("Step 4: Building per-symbol forecast dicts")
    all_forecasts: Dict[str, Dict[str, float]] = {}
    for sym in ohlcv_cache:
        fc: Dict[str, float] = {}
        if sym in ewmac_batch:
            for ef in ewmac_batch[sym]:
                fc[f"ewmac_{ef.fast}_{ef.slow}"] = ef.forecast
        if sym in carry_batch:
            fc["carry"] = carry_batch[sym].forecast
        if sym in momentum_forecasts:
            fc["momentum"] = momentum_forecasts[sym]
        if sym in mean_rev_forecasts:
            fc["mean_reversion"] = mean_rev_forecasts[sym]
        if sym in penfold_forecasts:
            fc["penfold_trend"] = penfold_forecasts[sym]
        # Ehlers DSP (Fisher Transform + MAMA/FAMA + Super Smoother + Sinewave + SNR)
        if sym in ehlers_forecasts:
            fc["ehlers_dsp"] = ehlers_forecasts[sym]
        # Ruggiero Cybernetic (intermarket + seasonal + trend strength + multi-TF)
        if sym in cybernetic_forecasts:
            fc["intermarket"] = cybernetic_forecasts[sym]
        # AFTS S23: Acceleration (rate of change of trend forecast)
        if sym in acceleration_forecasts:
            fc["acceleration"] = acceleration_forecasts[sym]
        # AFTS S22: Carver Value (5-year mean reversion)
        if sym in value_forecasts:
            fc["carver_value"] = value_forecasts[sym]
        # AFTS S24: Skew Signal (realized skew risk premium)
        if sym in skew_forecasts:
            fc["skew_signal"] = skew_forecasts[sym]
        # News sentiment forecast
        if sym in sentiment_forecasts:
            fc["sentiment"] = sentiment_forecasts[sym]
        if fc:
            all_forecasts[sym] = fc

    # Apply weekly Dow filter: dampen counter-trend signals
    # Aggressive dampening (×0.15 in broad bear) preserves capital
    if penfold_weekly_trends:
        n_down = sum(1 for v in penfold_weekly_trends.values() if v == "down")
        n_up = sum(1 for v in penfold_weekly_trends.values() if v == "up")
        broad_bear = n_down > n_up * 2
        for sym, fc in all_forecasts.items():
            wt = penfold_weekly_trends.get(sym, "unknown")
            for key in list(fc.keys()):
                if wt == "down" and fc[key] > 5.0:
                    dampen = 0.15 if broad_bear else 0.35
                    fc[key] *= dampen
                elif wt == "up" and fc[key] < -5.0:
                    fc[key] *= 0.5
                elif wt == "up" and fc[key] < -5.0:
                    fc[key] *= 0.5  # dampen sell in weekly uptrend

    if not all_forecasts:
        result.pipeline_log.append("ABORT: No forecasts generated")
        return result

    # ── Step 5: Combine forecasts ────────────────────────────
    result.pipeline_log.append("Step 5: Combining forecasts with FDM")
    combined = combine_forecasts_batch(all_forecasts)
    combined_values = {s: cf.combined_forecast for s, cf in combined.items()}

    # ── Step 5b: Masters prediction quality gate ─────────────
    try:
        from services.forecast_combiner import apply_masters_quality_gate
        gated = apply_masters_quality_gate(combined, ohlcv_data)
        gated_values = {sym: cf.combined_forecast for sym, cf in gated.items()}
        n_dampened = sum(
            1 for sym in gated_values
            if abs(gated_values[sym]) < abs(combined_values.get(sym, 0))
        )
        combined_values = gated_values
        if n_dampened > 0:
            result.pipeline_log.append(f"  → Masters quality gate dampened {n_dampened} low-quality forecasts")
    except Exception as mqe:
        result.pipeline_log.append(f"  → Masters quality gate skipped: {mqe}")

    # ── Step 5b: RL confidence modifier ──────────────────────
    if Config.RL_ENABLED:
        try:
            from services.rl_bot.rl_signal_integrator import get_rl_layer_score
            rl_modified = 0
            for sym in list(combined_values.keys()):
                try:
                    rl_score = get_rl_layer_score(sym, market="US")
                    if rl_score is None or rl_score == 0.0:
                        continue
                    original = combined_values[sym]
                    if abs(original) < 1.0:
                        continue
                    modifier = 1.0 + rl_score * 0.15
                    combined_values[sym] = max(-20.0, min(20.0, original * modifier))
                    rl_modified += 1
                except Exception:
                    pass
            if rl_modified:
                result.pipeline_log.append(f"  → RL confidence modifier applied to {rl_modified} forecasts")
                result.combined_forecasts = combined_values
            else:
                result.pipeline_log.append("  → RL enabled but no trained models matched current symbols")
        except ImportError:
            result.pipeline_log.append("  → RL module not available, skipping")
        except Exception as rl_exc:
            result.pipeline_log.append(f"  → RL confidence modifier skipped: {rl_exc}")

    # ── Step 5c: Meta-labeling confidence gate ───────────────
    try:
        from services.meta_labeling import apply_meta_labels
        ml_result = apply_meta_labels(
            combined_forecasts=combined_values,
            ohlcv_cache=ohlcv_cache,
            market="US",
        )
        if ml_result.blocked_count > 0 or ml_result.modified_count > 0:
            combined_values = ml_result.scaled_forecasts
            result.combined_forecasts = combined_values
            result.pipeline_log.append(
                f"  → Meta-label: {ml_result.modified_count} scaled, "
                f"{ml_result.blocked_count} blocked, {ml_result.passed_count} passed"
            )
        else:
            result.pipeline_log.append("  → Meta-label: no model or all passed through")
    except ImportError:
        result.pipeline_log.append("  → Meta-labeling module not available, skipping")
    except Exception as ml_exc:
        result.pipeline_log.append(f"  → Meta-labeling skipped: {ml_exc}")

    # ── Step 6: Cost speed limit (US costs) ──────────────────
    result.pipeline_log.append("Step 6: Applying cost speed limit (US cost config)")
    us_cost = CostConfig(
        round_trip_cost_pct=getattr(Config, "CARVER_US_COST_ROUND_TRIP_PCT", 0.0010),
        spread_slippage_pct=getattr(Config, "CARVER_US_SPREAD_SLIPPAGE_PCT", 0.0005),
        speed_limit_factor=getattr(Config, "CARVER_COST_SPEED_LIMIT", 3.0),
    )
    pre_filter = len(combined_values)
    combined_values = filter_by_cost(combined_values, vol_tgt, config=us_cost)
    result.symbols_filtered_by_cost = pre_filter - len(combined_values)
    result.combined_forecasts = combined_values
    result.pipeline_log.append(
        f"  → {len(combined_values)} passed, {result.symbols_filtered_by_cost} blocked by cost"
    )

    if not combined_values:
        result.pipeline_log.append("ABORT: All symbols filtered by cost speed limit")
        return result

    # ── Step 6b: Earnings blackout filter (P1 fix) ───────────
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        blackout_before = getattr(Config, "EARNINGS_BLACKOUT_DAYS_BEFORE", 2)
        blackout_after = getattr(Config, "EARNINGS_BLACKOUT_DAYS_AFTER", 1)
        today = datetime.now().date()
        blackout_syms = set()
        for sym in list(combined_values.keys()):
            try:
                ticker = yf.Ticker(sym)
                cal = ticker.calendar
                if cal is None or (hasattr(cal, 'empty') and cal.empty):
                    continue
                earnings_date = None
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        earnings_date = ed[0] if isinstance(ed, list) else ed
                elif hasattr(cal, "loc"):
                    try:
                        ed = cal.loc["Earnings Date"]
                        earnings_date = ed.iloc[0] if hasattr(ed, 'iloc') else ed
                    except Exception:
                        pass
                if earnings_date is not None:
                    if hasattr(earnings_date, 'date'):
                        earnings_date = earnings_date.date()
                    window_start = earnings_date - timedelta(days=blackout_before)
                    window_end = earnings_date + timedelta(days=blackout_after)
                    if window_start <= today <= window_end:
                        blackout_syms.add(sym)
            except Exception:
                continue
        if blackout_syms:
            for sym in blackout_syms:
                combined_values.pop(sym, None)
            result.pipeline_log.append(
                f"  Earnings blackout: {len(blackout_syms)} symbols suppressed — {', '.join(blackout_syms)}"
            )
    except Exception:
        pass  # Earnings data unavailable — proceed without filter

    if not combined_values:
        result.pipeline_log.append("ABORT: All symbols in earnings blackout")
        return result

    # ── Step 7: Instrument weights + IDM ─────────────────────
    result.pipeline_log.append("Step 7: Computing handcrafted weights + IDM")
    active = [s for s in combined_values if combined_values[s] > 0]
    weights = compute_handcrafted_weights(active, US_SECTOR_MAP)
    idm = get_default_idm(len(active))
    result.instrument_weights = weights
    result.idm = idm
    result.pipeline_log.append(f"  → {len(active)} active, IDM={idm:.2f}")

    # ── Step 8: Vol-targeted position sizing ─────────────────
    result.pipeline_log.append("Step 8: Computing vol-targeted position sizes")
    vt_cfg = VolatilityTargetConfig(
        initial_capital=cap,
        annual_vol_target_pct=vol_tgt,
        max_leverage_factor=max_lev,
        vince_insurance_pct=getattr(Config, 'VINCE_INSURANCE_PCT_US', 0.0),
    )
    vol_target = VolatilityTarget(vt_cfg)

    # ── Step 9: Generate trade plans ─────────────────────────
    result.pipeline_log.append("Step 9: Generating trade plans")
    for sym, forecast in combined_values.items():
        if forecast <= 0:
            continue
        inst_vol = result.instrument_vols.get(sym, 0)
        if inst_vol <= 0:
            continue
        weight = weights.get(sym, 1.0 / max(len(active), 1))

        # Get current price from OHLCV
        ohlcv_df = ohlcv_cache.get(sym)
        if ohlcv_df is None or ohlcv_df.empty:
            continue
        price = float(ohlcv_df["Close"].iloc[-1])
        if price <= 0:
            continue

        try:
            ps = compute_position_size(
                symbol=sym,
                combined_forecast=forecast,
                instrument_value_vol=inst_vol,
                daily_cash_vol_target=vol_target.daily_cash_vol_target,
                price=price,
                capital=cap,
                instrument_weight=weight,
                idm=idm,
            )
            qty = max(0, ps.target_quantity)
            if qty <= 0:
                continue

            # Adaptive stop-loss: 2.5σ for swing (Carver Ch. 13)
            daily_vol = result.instrument_vols.get(sym, 0)
            daily_pct_vol = daily_vol / price if price > 0 else 0.03
            stop_distance = 2.5 * daily_pct_vol * price
            stop_loss = round(price - stop_distance, 2)
            target = round(price + 3.5 * daily_pct_vol * price, 2)

            result.trade_plans.append({
                "symbol": sym,
                "side": "BUY",
                "quantity": qty,
                "entry_price": round(price, 2),
                "stop_loss": max(stop_loss, 0.01),
                "target": target,
                "forecast": round(forecast, 2),
                "instrument_weight": round(weight, 4),
                "instrument_vol": round(inst_vol, 2),
                "position_value": round(price * qty, 2),
                "pct_of_capital": round(price * qty / cap * 100, 1),
            })
        except Exception as exc:
            logger.debug("Position sizing failed for %s: %s", sym, exc)

    result.symbols_processed = len(ohlcv_cache)
    result.pipeline_log.append(f"  → {len(result.trade_plans)} trade plans generated")

    # ── Step 10 (Vince): Attach risk metrics summary ─────────
    try:
        from services.vince_metrics import get_vince_tracker
        vt = get_vince_tracker()
        snap = vt.get_snapshot("__portfolio__")
        if snap and snap.n_trades >= 5:
            result.pipeline_log.append(
                f"  Vince: G={snap.geometric_mean:.4f}, "
                f"optimal_f={snap.optimal_f:.3f}, "
                f"kelly_half={snap.kelly_half:.3f}, "
                f"trades={snap.n_trades}"
            )
    except Exception:
        pass

    return result


def run_us_carver_backtest(
    tickers: Optional[List[str]] = None,
    capital: Optional[float] = None,
    annual_vol_target: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the Carver calibration backtest on US stocks.

    Returns a dict with before/after metrics and calibration data.
    """
    from config import Config
    from utils import download_us_ohlcv
    from services.carver_calibration import CarverCalibrator, generate_efficiency_report

    tickers = tickers or DEFAULT_US_CARVER_TICKERS[:15]
    cap = capital or getattr(Config, "CARVER_US_INITIAL_CAPITAL", 10_000.0)
    vol_tgt = annual_vol_target or getattr(Config, "CARVER_US_ANNUAL_VOL_TARGET", 0.20)

    # Fetch 1 year OHLCV
    ohlcv_cache: Dict[str, pd.DataFrame] = {}
    for sym in tickers:
        try:
            df = download_us_ohlcv(sym, period="1y")
            if df is not None and len(df) >= 120:
                ohlcv_cache[sym] = df
        except Exception:
            pass

    if not ohlcv_cache:
        return {"error": "No OHLCV data available for US backtest"}

    calibrator = CarverCalibrator(
        annual_vol_target=vol_tgt,
        initial_capital=cap,
    )
    report = calibrator.run_expanding_backtest(ohlcv_cache)
    text_report = generate_efficiency_report(report)

    return {
        "report_text": text_report,
        "backtest_sharpe": report.backtest_sharpe,
        "backtest_sortino": report.backtest_sortino,
        "backtest_max_drawdown_pct": report.backtest_max_drawdown_pct,
        "backtest_annual_return_pct": report.backtest_annual_return_pct,
        "n_symbols": report.n_symbols,
        "n_days": report.n_days,
        "ewmac_scalars": report.ewmac_scalars,
        "calibrated_fdm": report.calibrated_fdm,
        "calibrated_idm": report.calibrated_idm,
        "market": "US",
        "capital_currency": "USD",
        "initial_capital": cap,
    }
