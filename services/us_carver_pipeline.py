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

    # ── Step 1: Download OHLCV ───────────────────────────────
    result.pipeline_log.append(f"Step 1: Downloading OHLCV for {len(tickers)} US tickers")
    ohlcv_cache: Dict[str, pd.DataFrame] = {}
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

    # ── Step 4: Build forecast dicts ─────────────────────────
    result.pipeline_log.append("Step 4: Building per-symbol forecast dicts")
    all_forecasts: Dict[str, Dict[str, float]] = {}
    for sym in ohlcv_cache:
        fc: Dict[str, float] = {}
        if sym in ewmac_batch:
            for ef in ewmac_batch[sym]:
                fc[f"ewmac_{ef.fast}_{ef.slow}"] = ef.forecast
        if fc:
            all_forecasts[sym] = fc

    if not all_forecasts:
        result.pipeline_log.append("ABORT: No forecasts generated")
        return result

    # ── Step 5: Combine forecasts ────────────────────────────
    result.pipeline_log.append("Step 5: Combining forecasts with FDM")
    combined = combine_forecasts_batch(all_forecasts)
    combined_values = {s: cf.combined_forecast for s, cf in combined.items()}

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
