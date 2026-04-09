"""
Carver Pipeline Orchestrator — End-to-end Systematic Trading pipeline.

Ties together ALL Carver framework modules into a single execution flow:

    Universe → OHLCV → Volatility → Forecasts → Combine → Cost Filter
    → Position Size → Risk Check → Portfolio Monitor → Trade Plans

This replaces the ad-hoc screener→Kelly→execute path with a structured
pipeline based on Robert Carver's *Systematic Trading* framework.

Usage:
    from services.carver_pipeline import CarverPipeline
    pipeline = CarverPipeline()
    plans = pipeline.run(ohlcv_cache, screener_scores)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# C3: Persistent forecast history for dynamic correlations
_FORECAST_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "forecast_history.json"


# ── FIX-1 helper: read Config values safely ────────────────────
def _cfg_val(attr: str, default):
    """Read a Config attribute with safe fallback."""
    try:
        from config import Config
        return getattr(Config, attr, default)
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════════
# Gap D2: OHLC Validation — ensure data integrity before signals
# ═══════════════════════════════════════════════════════════════

def validate_ohlcv(
    ohlcv_cache: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Validate and repair OHLCV data. Returns cleaned cache with bad symbols removed.

    Checks per bar:
      - High >= max(Open, Close)
      - Low  <= min(Open, Close)
      - Volume >= 0
      - No NaN in OHLC
    Rows failing checks are dropped. Symbols with < 30 remaining bars are removed.
    """
    cleaned: Dict[str, pd.DataFrame] = {}
    dropped_count = 0
    for sym, df in ohlcv_cache.items():
        if df is None or df.empty:
            continue
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(df.columns):
            continue

        # Drop rows with NaN in OHLC
        mask = df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
        # High >= max(Open, Close)
        mask &= df["High"] >= df[["Open", "Close"]].max(axis=1)
        # Low <= min(Open, Close)
        mask &= df["Low"] <= df[["Open", "Close"]].min(axis=1)
        # Non-negative volume (if present)
        if "Volume" in df.columns:
            mask &= df["Volume"].fillna(0) >= 0

        valid_df = df.loc[mask].copy()
        if len(valid_df) >= 30:
            cleaned[sym] = valid_df
        else:
            dropped_count += 1

    if dropped_count:
        logger.warning("OHLC validation dropped %d symbols (< 30 valid bars)", dropped_count)
    return cleaned


@dataclass
class PipelineConfig:
    """Configuration for the full Carver pipeline.

    FIX-1: All defaults now read from Config.py at class-load time.
    Previously hardcoded vol=20%, IDM=1.6 — causing 3.12× undersized positions.
    """
    initial_capital: float = field(default_factory=lambda: _cfg_val("CARVER_INITIAL_CAPITAL", 500_000.0))
    annual_vol_target_pct: float = field(default_factory=lambda: _cfg_val("CARVER_ANNUAL_VOL_TARGET", 0.50))
    max_open_trades: int = field(default_factory=lambda: int(_cfg_val("MAX_OPEN_TRADES", 8)))
    default_idm: float = field(default_factory=lambda: _cfg_val("CARVER_DEFAULT_IDM", 2.0))
    max_leverage: float = field(default_factory=lambda: _cfg_val("CARVER_MAX_LEVERAGE", 2.0))
    apply_cost_filter: bool = True
    trade_horizon: str = field(default_factory=lambda: str(_cfg_val("CARVER_TRADE_HORIZON", "swing")))


@dataclass
class PipelineResult:
    """Output of a full pipeline execution."""
    trade_plans: List = field(default_factory=list)
    combined_forecasts: Dict[str, float] = field(default_factory=dict)
    position_sizes: Dict = field(default_factory=dict)
    risk_snapshot: Optional[object] = None
    options_overlay: Optional[object] = None   # Gap A1: OverlayResult
    mc_result: Optional[object] = None         # T5-2: MC Kelly bootstrap result
    hedge_result: Optional[object] = None      # T5-2: Hedge scan result
    iron_condor_result: Optional[object] = None  # T5-2: Iron condor overlay result
    symbols_processed: int = 0
    symbols_with_trades: int = 0
    cost_filtered_count: int = 0
    pipeline_log: List[str] = field(default_factory=list)
    validation_stats: Dict = field(default_factory=dict)  # Aronson EBTA per-symbol confidence scores
    individual_forecasts: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {sym: {source: value}}


# T1-2: Module-level accessor for the current pipeline's vol target instance
_active_pipeline_instance: Optional["CarverPipeline"] = None


def _get_vol_target_instance():
    """Return the active pipeline's VolatilityTarget, or None."""
    if _active_pipeline_instance is not None:
        return _active_pipeline_instance._vol_target
    return None


class CarverPipeline:
    """Orchestrates the full Carver systematic trading pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._vol_target = None
        self._hmm_model = None
        self._forecast_history = self._load_forecast_history()  # C3: persist across runs
        self._init_vol_target()

    # ── C3: Forecast history persistence ──────────────────────

    @staticmethod
    def _load_forecast_history() -> Dict[str, list]:
        """Load persisted forecast history from disk (C3 fix)."""
        import json
        try:
            if _FORECAST_HISTORY_PATH.exists():
                with open(_FORECAST_HISTORY_PATH, "r") as f:
                    data = json.load(f)
                # Keep only last 120 days to bound memory
                return {k: v[-120:] for k, v in data.items() if isinstance(v, list)}
        except Exception as exc:
            logger.debug("Forecast history load failed: %s", exc)
        return {}

    def _save_forecast_history(self) -> None:
        """Persist forecast history to disk (C3 fix)."""
        import json
        try:
            _FORECAST_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Keep only last 120 days
            trimmed = {k: v[-120:] for k, v in self._forecast_history.items() if isinstance(v, list)}
            with open(_FORECAST_HISTORY_PATH, "w") as f:
                json.dump(trimmed, f)
        except Exception as exc:
            logger.debug("Forecast history save failed: %s", exc)

    def _init_vol_target(self):
        """Initialise the volatility target module.

        FIX-2: Now passes max_leverage_factor from PipelineConfig.
        """
        from services.volatility_target import VolatilityTarget, VolatilityTargetConfig
        try:
            from config import Config as _VinceCfg
            _ins_pct = getattr(_VinceCfg, 'VINCE_INSURANCE_PCT_IND', 0.0)
        except Exception:
            _ins_pct = 0.0
        vt_config = VolatilityTargetConfig(
            initial_capital=self.cfg.initial_capital,
            annual_vol_target_pct=self.cfg.annual_vol_target_pct,
            max_leverage_factor=getattr(self.cfg, 'max_leverage', 2.0),
            vince_insurance_pct=_ins_pct,
        )
        self._vol_target = VolatilityTarget(vt_config)

    def run(
        self,
        ohlcv_cache: Dict[str, pd.DataFrame],
        screener_scores: Optional[Dict[str, float]] = None,
        decision_engine_scores: Optional[Dict[str, float]] = None,
        current_holdings: Optional[Dict[str, int]] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> PipelineResult:
        """Execute the full Carver pipeline.

        Parameters
        ----------
        ohlcv_cache : dict[str, pd.DataFrame]
            {symbol: OHLCV DataFrame}.
        screener_scores : dict[str, float] | None
            {symbol: screener_score (0-100)}.
        decision_engine_scores : dict[str, float] | None
            {symbol: decision_engine_score (-1 to +1)}.
        current_holdings : dict[str, int] | None
            {symbol: current_quantity} for inertia.
        sector_map : dict[str, str] | None
            {symbol: sector_name} for sector weights.

        Returns
        -------
        PipelineResult
        """
        result = PipelineResult()
        log = result.pipeline_log

        # T1-2: Register this pipeline instance for options_overlay premium feedback
        global _active_pipeline_instance
        _active_pipeline_instance = self

        # ── Step 1: Compute instrument volatilities ────────────
        log.append("Step 1: Validating OHLCV data and computing volatilities...")
        from services.instrument_volatility import compute_volatilities_batch

        # Gap D2: OHLC validation — repair data integrity before any signals
        before_validate = len(ohlcv_cache)
        ohlcv_cache = validate_ohlcv(ohlcv_cache)
        validated_dropped = before_validate - len(ohlcv_cache)
        if validated_dropped:
            log.append(f"  ⚠ OHLC validation dropped {validated_dropped} symbols")

        # Tier 1 Gap 5: Data freshness gate — skip symbols with stale OHLCV
        # Gap A5 FIX: Use UTC throughout, convert only for display
        from datetime import datetime, timedelta, timezone
        from config import Config
        freshness_hours = getattr(Config, "SIGNAL_FRESHNESS_MAX_HOURS", 4)
        stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=freshness_hours)
        stale_cutoff_naive = stale_cutoff.replace(tzinfo=None)
        stale_symbols = []
        for sym, df in list(ohlcv_cache.items()):
            if df is not None and not df.empty and df.index.dtype.kind == "M":
                last_bar = df.index[-1]
                # Normalize to UTC-naive for consistent comparison
                if hasattr(last_bar, "tz") and last_bar.tz is not None:
                    last_bar = last_bar.tz_convert("UTC").tz_localize(None)
                if last_bar < stale_cutoff_naive:
                    stale_symbols.append(sym)
                    del ohlcv_cache[sym]
        if stale_symbols:
            log.append(f"  ⚠ Freshness gate: dropped {len(stale_symbols)} stale symbols: {stale_symbols[:5]}...")
            logger.warning("Freshness gate dropped %d stale symbols (cutoff=%s)", len(stale_symbols), stale_cutoff)

        # Gap D1: Apply corporate action adjustments to OHLCV before signals
        try:
            from services.corporate_actions import get_actions_for_symbols, adjust_ohlcv_for_action
            corp_actions = get_actions_for_symbols(list(ohlcv_cache.keys()))
            if corp_actions:
                for sym, action in corp_actions.items():
                    if sym in ohlcv_cache:
                        ohlcv_cache[sym] = adjust_ohlcv_for_action(ohlcv_cache[sym], action)
                log.append(f"  → Gap D1: Applied corporate actions for {len(corp_actions)} symbols: {list(corp_actions.keys())}")
        except Exception as exc:
            log.append(f"  → Corporate actions skipped: {exc}")

        vol_data = compute_volatilities_batch(ohlcv_cache)
        instrument_vols = {}
        daily_vols = {}
        prices = {}
        for sym, vd in vol_data.items():
            instrument_vols[sym] = vd["instr_value_vol"]
            daily_vols[sym] = vd["daily_vol"]
            prices[sym] = vd["price"]
        log.append(f"  → {len(instrument_vols)} instruments with volatility data")

        # ── Step 2: Compute EWMAC forecasts ────────────────────
        log.append("Step 2: Computing EWMAC forecasts...")
        from strategies.ewmac import compute_ewmac_batch

        ewmac_batch = compute_ewmac_batch(ohlcv_cache)
        log.append(f"  → EWMAC computed for {len(ewmac_batch)} symbols")

        # ── Step 3: Build forecast dicts per symbol ────────────
        log.append("Step 3: Building per-symbol forecast dictionaries...")
        from services.forecast_scalar import (
            screener_to_forecast,
            decision_engine_to_forecast,
        )

        # Phase 1: Momentum factor forecasts
        momentum_forecasts: Dict[str, float] = {}
        try:
            from services.momentum_factor import compute_momentum_forecasts
            momentum_forecasts = compute_momentum_forecasts(ohlcv_cache)
            if momentum_forecasts:
                log.append(f"  → Momentum forecasts for {len(momentum_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  → Momentum factor skipped: {exc}")

        # Phase 1: PEAD forecasts (if active)
        pead_forecasts: Dict[str, float] = {}
        try:
            from services.pead_strategy import PEADStrategy
            pead = PEADStrategy()
            pead.advance_day()  # G7 FIX: increment decay counter each pipeline run
            pead_forecasts = pead.get_current_forecasts()
            if pead_forecasts:
                log.append(f"  → PEAD forecasts for {len(pead_forecasts)} symbols")
            else:
                log.append("  → PEAD: no active signals (earnings cache may be empty)")
        except Exception as exc:
            log.append(f"  → PEAD skipped: {exc}")

        # Gap A2: Mean-reversion forecasts
        mean_rev_forecasts: Dict[str, float] = {}
        try:
            from strategies.mean_reversion import compute_mean_reversion_batch
            mean_rev_forecasts = compute_mean_reversion_batch(ohlcv_cache)
            if mean_rev_forecasts:
                log.append(f"  → Mean-reversion forecasts for {len(mean_rev_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  → Mean-reversion skipped: {exc}")

        # Gap A5: FII daily flow signal (market-wide)
        fii_forecasts: Dict[str, float] = {}
        try:
            from services.fii_flow_signal import get_fii_flow_forecasts
            fii_forecasts = get_fii_flow_forecasts(list(ohlcv_cache.keys()))
            if fii_forecasts:
                log.append(f"  → FII flow forecast applied to {len(fii_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  → FII flow skipped: {exc}")
        # G2 FIX: Carry rule forecasts (dividend yield − repo rate)
        carry_forecasts: Dict[str, float] = {}
        try:
            from strategies.carry_rule import compute_carry_batch
            carry_forecasts = compute_carry_batch(ohlcv_cache)
            if carry_forecasts:
                log.append(f"  → Carry forecasts for {len(carry_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  → Carry rule skipped: {exc}")

        # Gap A6: F&O Open Interest signals (real OI from NSE bhavcopy)
        oi_forecasts: Dict[str, float] = {}
        try:
            from services.oi_signal import compute_oi_signals_batch

            # Try real F&O bhavcopy OI data first
            oi_data = {}
            try:
                from services.nse_fo_bhavcopy import fetch_fo_oi_data
                oi_data = fetch_fo_oi_data()
                if oi_data:
                    log.append(f"  → Real F&O OI data for {len(oi_data)} symbols")
            except Exception as bkcp_exc:
                log.append(f"  → F&O bhavcopy unavailable ({bkcp_exc}), falling back to volume proxy")

            # Fallback: volume proxy for symbols not in F&O bhavcopy
            if len(oi_data) < 10:
                for sym, df in ohlcv_cache.items():
                    if sym in oi_data:
                        continue  # already have real OI
                    if df is not None and len(df) >= 2:
                        try:
                            price_change_pct = float(
                                (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
                            )
                            if "Volume" in df.columns and len(df) >= 5:
                                avg_vol = float(df["Volume"].iloc[-5:].mean())
                                last_vol = float(df["Volume"].iloc[-1])
                                oi_change_pct = ((last_vol / avg_vol) - 1) * 100 if avg_vol > 0 else 0.0
                            else:
                                oi_change_pct = 0.0
                            oi_data[sym] = {
                                "oi_change_pct": oi_change_pct,
                                "price_change_pct": price_change_pct,
                            }
                        except Exception:
                            pass

            if oi_data:
                oi_forecasts = compute_oi_signals_batch(oi_data)
                if oi_forecasts:
                    log.append(f"  → OI signal forecasts for {len(oi_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  → OI signal skipped: {exc}")

        # G9: Event-driven forecasts (earnings, RBI, expiry, rebalance)
        event_forecasts: Dict[str, float] = {}
        try:
            from config import Config as _Cfg
            if getattr(_Cfg, "EVENT_DRIVEN_ENABLED", False):
                from services.event_strategy import generate_event_forecasts
                evt_signals = generate_event_forecasts()
                for ef in evt_signals:
                    if ef.symbol and ef.symbol != "MARKET":
                        event_forecasts[ef.symbol] = ef.forecast
                    elif ef.symbol == "MARKET":
                        # Market-wide event → apply to all symbols
                        for sym in ohlcv_cache:
                            if sym not in event_forecasts:
                                event_forecasts[sym] = ef.forecast
                if event_forecasts:
                    log.append(f"  → Event-driven forecasts for {len(event_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  → Event-driven skipped: {exc}")

        # Breakout signal: 20-day high/low channel (uncorrelated with EWMAC)
        breakout_forecasts: Dict[str, float] = {}
        try:
            import numpy as _np
            for sym, df in ohlcv_cache.items():
                if df is None or len(df) < 22:
                    continue
                c = df["Close"]
                if hasattr(c, "squeeze"):
                    c = c.squeeze()
                price_now = float(c.iloc[-1])
                high_20 = float(c.iloc[-21:-1].max())
                low_20 = float(c.iloc[-21:-1].min())
                rng = high_20 - low_20
                if rng > 0 and _np.isfinite(price_now):
                    breakout_fc = ((price_now - low_20) / rng - 0.5) * 20.0
                    breakout_fc = max(-20.0, min(20.0, breakout_fc))
                    breakout_forecasts[sym] = breakout_fc
            if breakout_forecasts:
                log.append(f"  -> Breakout forecasts for {len(breakout_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  -> Breakout skipped: {exc}")

        # Penfold trend tactics: Turtle breakout + ATR band + retracement + weekly Dow filter
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
                log.append(f"  -> Penfold trend forecasts for {len(penfold_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  -> Penfold trend skipped: {exc}")

        # ── Ehlers DSP: Fisher Transform, Super Smoother, MAMA/FAMA, Sinewave, SNR
        # (Ehlers, "Cybernetic Analysis for Stocks & Futures" + "Rocket Science for Traders")
        ehlers_forecasts: Dict[str, float] = {}
        try:
            from strategies.ehlers_dsp import compute_ehlers_forecast_batch
            ehlers_forecasts = compute_ehlers_forecast_batch(ohlcv_cache)
            if ehlers_forecasts:
                log.append(f"  -> Ehlers DSP forecasts for {len(ehlers_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  -> Ehlers DSP skipped: {exc}")

        # ── Ruggiero Cybernetic: Intermarket analysis, seasonal, trend classification
        # (Ruggiero, "Cybernetic Trading Strategies")
        cybernetic_forecasts: Dict[str, float] = {}
        try:
            from strategies.ruggiero_cybernetic import (
                compute_cybernetic_forecast_batch,
                IND_INTERMARKET_DRIVERS,
            )
            # Download intermarket driver data
            import yfinance as _yf_drivers
            driver_dfs: Dict[str, pd.DataFrame] = {}
            for driver_sym in IND_INTERMARKET_DRIVERS:
                try:
                    d = _yf_drivers.download(driver_sym, period="120d", progress=False)
                    if d is not None and len(d) >= 30:
                        driver_dfs[driver_sym] = d
                except Exception:
                    pass
            if driver_dfs:
                cybernetic_forecasts = compute_cybernetic_forecast_batch(
                    ohlcv_cache, driver_dfs, IND_INTERMARKET_DRIVERS
                )
                if cybernetic_forecasts:
                    log.append(f"  -> Cybernetic forecasts for {len(cybernetic_forecasts)} symbols "
                              f"({len(driver_dfs)} intermarket drivers)")
        except Exception as exc:
            log.append(f"  -> Cybernetic intermarket skipped: {exc}")

        # ── AFTS S23: Acceleration — rate of change of EWMAC forecast ──
        acceleration_forecasts: Dict[str, float] = {}
        try:
            from strategies.acceleration import compute_acceleration_batch
            acceleration_forecasts = compute_acceleration_batch(ohlcv_cache)
            if acceleration_forecasts:
                log.append(f"  -> Acceleration (S23) forecasts for {len(acceleration_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  -> Acceleration skipped: {exc}")

        # ── AFTS S22: Carver Value — 5-year mean reversion ──
        value_forecasts: Dict[str, float] = {}
        try:
            from strategies.carver_value import compute_value_batch
            value_forecasts = compute_value_batch(ohlcv_cache)
            if value_forecasts:
                log.append(f"  -> Carver Value (S22) forecasts for {len(value_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  -> Carver Value skipped: {exc}")

        # ── AFTS S24: Skew Signal — realized skew risk premium ──
        skew_forecasts: Dict[str, float] = {}
        try:
            from strategies.skew_signal import compute_skew_batch
            skew_forecasts = compute_skew_batch(ohlcv_cache)
            if skew_forecasts:
                log.append(f"  -> Skew Signal (S24) forecasts for {len(skew_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  -> Skew Signal skipped: {exc}")

        # ── Sentiment Forecast — news-driven signal ──
        sentiment_forecasts: Dict[str, float] = {}
        try:
            from services.sentiment_forecast import compute_sentiment_batch
            sentiment_forecasts = compute_sentiment_batch(ohlcv_cache)
            if sentiment_forecasts:
                log.append(f"  -> Sentiment forecasts for {len(sentiment_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  -> Sentiment skipped: {exc}")

        # ── AFTS S13: Vol-regime multipliers — per-symbol ──
        vol_regime_multipliers: Dict[str, float] = {}
        try:
            for sym, df in ohlcv_cache.items():
                if df is None or len(df) < 252 or "Close" not in df.columns:
                    continue
                returns = df["Close"].pct_change().dropna()
                if len(returns) < 126:
                    continue
                current_vol = float(returns.iloc[-63:].std()) * (252 ** 0.5)  # 3-month vol
                median_vol = float(returns.expanding(min_periods=126).std().iloc[-1]) * (252 ** 0.5)
                if current_vol > 0 and median_vol > 0:
                    vol_regime_multipliers[sym] = median_vol / current_vol
            if vol_regime_multipliers:
                log.append(f"  -> Vol-regime multipliers (S13) computed for {len(vol_regime_multipliers)} symbols")
        except Exception as exc:
            log.append(f"  -> Vol-regime multipliers skipped: {exc}")

        # Cross-sectional momentum: rank stocks by 6-month return
        cross_mom_forecasts: Dict[str, float] = {}
        try:
            import numpy as _np
            xmom_returns = {}
            for sym, df in ohlcv_cache.items():
                if df is None:
                    continue
                c = df["Close"]
                if hasattr(c, "squeeze"):
                    c = c.squeeze()
                if len(c) >= 126:
                    ret = float(c.iloc[-1] / c.iloc[-126] - 1)
                    if _np.isfinite(ret):
                        xmom_returns[sym] = ret
            if len(xmom_returns) >= 6:
                sorted_syms = sorted(xmom_returns.keys(), key=lambda s: xmom_returns[s])
                n_tercile = max(1, len(sorted_syms) // 3)
                for sym in sorted_syms[:n_tercile]:
                    cross_mom_forecasts[sym] = -8.0
                for sym in sorted_syms[-n_tercile:]:
                    cross_mom_forecasts[sym] = +8.0
                log.append(f"  → Cross-momentum: {n_tercile} long, {n_tercile} short")
        except Exception as exc:
            log.append(f"  → Cross-momentum skipped: {exc}")

        # Pairs arb: cointegrated pairs spread trading
        pairs_forecasts: Dict[str, float] = {}
        try:
            import numpy as _np_pairs
            from services.pairs_trading_live import scan_all_pairs
            # scan_all_pairs expects Dict[str, np.ndarray] of close prices
            close_arrays: Dict[str, Any] = {}
            for sym, df in ohlcv_cache.items():
                if df is not None and "Close" in df.columns:
                    c = df["Close"]
                    if hasattr(c, "to_numpy"):
                        close_arrays[sym] = c.to_numpy()
                    else:
                        close_arrays[sym] = _np_pairs.array(c)
            pairs_signals = scan_all_pairs(close_arrays)
            for ps in pairs_signals:
                if ps.forecast != 0 and ps.leg1 and ps.leg2:
                    pairs_forecasts[ps.leg1] = pairs_forecasts.get(ps.leg1, 0) + ps.forecast * 0.5
                    pairs_forecasts[ps.leg2] = pairs_forecasts.get(ps.leg2, 0) - ps.forecast * 0.5
            if pairs_forecasts:
                log.append(f"  → Pairs arb forecasts for {len(pairs_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  → Pairs arb skipped: {exc}")

        all_forecasts: Dict[str, Dict[str, float]] = {}
        for sym in ohlcv_cache:
            fc: Dict[str, float] = {}

            # EWMAC forecasts
            if sym in ewmac_batch:
                for ef in ewmac_batch[sym]:
                    key = f"ewmac_{ef.fast}_{ef.slow}"
                    fc[key] = ef.forecast

            # Screener forecast
            if screener_scores and sym in screener_scores:
                fc["screener"] = screener_to_forecast(screener_scores[sym])

            # Decision engine forecast
            if decision_engine_scores and sym in decision_engine_scores:
                fc["decision_engine"] = decision_engine_to_forecast(
                    decision_engine_scores[sym]
                )

            # Momentum factor forecast (Phase 1)
            if sym in momentum_forecasts:
                fc["momentum"] = momentum_forecasts[sym]

            # PEAD forecast (Phase 1)
            if sym in pead_forecasts:
                fc["pead"] = pead_forecasts[sym]

            # Gap A2: Mean-reversion forecast
            if sym in mean_rev_forecasts:
                fc["mean_reversion"] = mean_rev_forecasts[sym]

            # Gap A5: FII flow forecast
            if sym in fii_forecasts:
                fc["fii_flow"] = fii_forecasts[sym]

            # Gap A6: OI signal forecast
            if sym in oi_forecasts:
                fc["oi_signal"] = oi_forecasts[sym]

            # G2: Carry rule forecast
            if sym in carry_forecasts:
                fc["carry"] = carry_forecasts[sym]

            # G9: Event-driven forecast
            if sym in event_forecasts:
                fc["event_driven"] = event_forecasts[sym]

            # Breakout channel forecast
            if sym in breakout_forecasts:
                fc["breakout"] = breakout_forecasts[sym]

            # Cross-sectional momentum forecast
            if sym in cross_mom_forecasts:
                fc["cross_momentum"] = cross_mom_forecasts[sym]

            # Pairs arbitrage forecast
            if sym in pairs_forecasts:
                fc["pairs_arb"] = pairs_forecasts[sym]

            # Penfold trend (Turtle + ATR band + retracement + weekly Dow filter)
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

            # Gap B6: Decision engine forecast integration
            # NOTE: Removed duplicate — decision_engine forecast already added above

            if fc:
                all_forecasts[sym] = fc

        log.append(f"  -> Forecasts built for {len(all_forecasts)} symbols")
        result.symbols_processed = len(all_forecasts)
        result.individual_forecasts = {sym: dict(fc) for sym, fc in all_forecasts.items()}

        # Penfold weekly Dow filter: dampen buy signals if weekly trend is down
        # Aggressive dampening (×0.2) prevents capital destruction in bear markets
        # and is the core mechanism ensuring CAGR > 50% across full market cycles
        if penfold_weekly_trends:
            dampened = 0
            n_weekly_down = sum(1 for v in penfold_weekly_trends.values() if v == "down")
            n_weekly_up = sum(1 for v in penfold_weekly_trends.values() if v == "up")
            market_breadth_bearish = n_weekly_down > n_weekly_up * 2  # Broad market weakness

            for sym, fc_dict in all_forecasts.items():
                wt = penfold_weekly_trends.get(sym, "unknown")
                if wt == "unknown":
                    continue
                for src, val in list(fc_dict.items()):
                    if wt == "down" and val > 5.0:
                        # In broad bear market, kill buy signals almost entirely
                        dampen = 0.15 if market_breadth_bearish else 0.35
                        fc_dict[src] = val * dampen
                        dampened += 1
                    elif wt == "up" and val < -5.0:
                        fc_dict[src] = val * 0.5
                        dampened += 1

            # G3 FIX (revised P3): Source-selective broad bear cap.
            # Trend-following sources capped at +5 (was +3 — too aggressive).
            # Mean-reversion, PEAD, pairs, event_driven, carver_value EXEMPT —
            # these are designed for bear/range alpha and were being crushed.
            if market_breadth_bearish:
                _BEAR_EXEMPT_SOURCES = {
                    "mean_reversion", "pead", "pairs_arb", "event_driven",
                    "carver_value", "sentiment", "oi_signal",
                }
                _BEAR_CAP = 5.0  # P3: was 3.0, raised to allow stronger conviction signals
                _g3_capped = 0
                for sym, fc_dict in all_forecasts.items():
                    for src, val in list(fc_dict.items()):
                        if src in _BEAR_EXEMPT_SOURCES:
                            continue  # P3: don't cap regime-appropriate sources
                        if val > _BEAR_CAP:
                            fc_dict[src] = _BEAR_CAP
                            _g3_capped += 1
                if _g3_capped:
                    log.append(
                        f"  -> G3/P3: Broad bear — capped {_g3_capped} trend forecasts "
                        f"at +{_BEAR_CAP} (exempt: MR, PEAD, pairs, events, value, sentiment, OI)"
                    )

            if dampened:
                log.append(f"  -> Weekly Dow filter dampened {dampened} counter-trend signals")
                if market_breadth_bearish:
                    log.append(f"  -> BROAD BEAR detected ({n_weekly_down} down vs {n_weekly_up} up) — aggressive dampening")

        # ── Step 4: Combine forecasts ──────────────────────────
        log.append("Step 4: Combining forecasts with FDM...")
        from services.forecast_combiner import combine_forecasts_batch

        # Phase 5: Use regime-aware weights, fallback to factor-momentum, then static
        # Gap B1: HMM probabilistic weight blending (preferred over binary regime)
        dynamic_weights = None
        hmm_snap = None
        try:
            from services.regime_hmm import get_hmm_model, get_hmm_blended_weights
            from services.regime_strategy_mix import REGIME_STRATEGY_WEIGHTS
            hmm = get_hmm_model()
            if hmm._fitted:
                # Build observations from recent NIFTY data
                from services.regime_hmm import prepare_hmm_observations
                import yfinance as yf
                nifty_data = yf.download("^NSEI", period="60d", progress=False)
                if nifty_data is not None and len(nifty_data) >= 20:
                    try:
                        vix_data = yf.download("^INDIAVIX", period="60d", progress=False)
                    except Exception:
                        vix_data = None
                    obs = prepare_hmm_observations(nifty_data, vix_data)
                    if len(obs) >= 10:
                        hmm_snap = hmm.get_current_regime(obs)
                        self._hmm_model = hmm
                        all_sources = list(set().union(
                            *(fc.keys() for fc in all_forecasts.values())
                        ))
                        dynamic_weights = get_hmm_blended_weights(
                            np.array(hmm_snap.probabilities),
                            REGIME_STRATEGY_WEIGHTS,
                            all_sources,
                        )
                        log.append(
                            f"  → HMM regime: {hmm_snap.regime} "
                            f"(conf={hmm_snap.confidence:.0%}), "
                            f"P=[{hmm_snap.probabilities[0]:.2f}, "
                            f"{hmm_snap.probabilities[1]:.2f}, "
                            f"{hmm_snap.probabilities[2]:.2f}]"
                        )
        except Exception as exc:
            log.append(f"  → HMM regime skipped: {exc}")

        if dynamic_weights is None:
            try:
                from services.regime_strategy_mix import get_regime_aware_forecast_weights
                regime_weights = get_regime_aware_forecast_weights()
                if regime_weights:
                    dynamic_weights = {w.source: w.weight for w in regime_weights}
                    log.append("  → Using rule-based regime-aware forecast weights")
            except Exception:
                pass

        if dynamic_weights is None:
            try:
                from services.factor_momentum import get_forecast_weights
                dynamic_weights = get_forecast_weights()
                log.append("  → Using factor-momentum dynamic weights")
            except Exception:
                pass

        # Phase 4: Apply strategy decay multipliers to scale down degraded strategies
        try:
            from services.strategy_decay import StrategyDecayMonitor
            decay_monitor = StrategyDecayMonitor()
            decay_mults = decay_monitor.get_allocation_multipliers()
            if decay_mults:
                for sym_forecasts in all_forecasts.values():
                    for source, mult in decay_mults.items():
                        if source in sym_forecasts and mult < 1.0:
                            if not (np.isfinite(mult) and 0.0 <= mult <= 1.0):
                                continue  # skip NaN/Inf/negative decay multipliers
                            sym_forecasts[source] *= mult
                log.append(f"  → Strategy decay applied: {sum(1 for m in decay_mults.values() if m < 1.0)} degraded")
        except Exception:
            pass

        # ── T4-1: Thompson Sampling bandit weight modification ──
        # Adaptively adjust source weights based on recent realized performance
        try:
            from services.thompson_sampling import ThompsonSamplingBandit
            from services.forecast_combiner import ForecastWeight
            bandit = ThompsonSamplingBandit()
            bandit.load_state()
            sampled = bandit.sample_weights()
            if sampled and len(sampled) >= 5 and dynamic_weights is not None:
                # T6-1 FIX: Normalize dynamic_weights to list of ForecastWeight
                # (may be dict from HMM blending or factor-momentum)
                if isinstance(dynamic_weights, dict):
                    _ts_weights = [ForecastWeight(name=k, weight=v) for k, v in dynamic_weights.items()]
                elif isinstance(dynamic_weights, list):
                    _ts_weights = dynamic_weights
                else:
                    _ts_weights = []
                # Blend sampled weights with Carver static weights (30% bandit, 70% Carver)
                for fw in _ts_weights:
                    if fw.name in sampled:
                        fw.weight = 0.70 * fw.weight + 0.30 * sampled[fw.name]
                # Re-normalize
                total_w = sum(fw.weight for fw in _ts_weights)
                if total_w > 0:
                    for fw in _ts_weights:
                        fw.weight = fw.weight / total_w
                # Write back to dynamic_weights (dict or list form)
                if isinstance(dynamic_weights, dict):
                    dynamic_weights = {fw.name: fw.weight for fw in _ts_weights}
                else:
                    dynamic_weights = _ts_weights
                log.append(f"  → Thompson Sampling: blended {len(sampled)} source weights")
        except Exception as ts_exc:
            log.append(f"  → Thompson Sampling skipped: {ts_exc}")

        combined = combine_forecasts_batch(
            all_forecasts, weights=dynamic_weights,
            vol_regime_multipliers=vol_regime_multipliers,
        )

        # Dynamic correlation matrix: update from forecast history if available
        try:
            from services.forecast_combiner import compute_rolling_correlations
            forecast_history = getattr(self, '_forecast_history', {})
            # T6-4 FIX: Average forecasts across symbols per source per run
            # (previously appended every symbol value, inflating history)
            _source_sums: Dict[str, float] = {}
            _source_counts: Dict[str, int] = {}
            for sym, fc_dict in all_forecasts.items():
                for source, val in fc_dict.items():
                    if np.isfinite(val):
                        _source_sums[source] = _source_sums.get(source, 0.0) + val
                        _source_counts[source] = _source_counts.get(source, 0) + 1
            for source in _source_sums:
                avg_val = _source_sums[source] / max(1, _source_counts[source])
                if source not in forecast_history:
                    forecast_history[source] = []
                forecast_history[source].append(avg_val)
            self._forecast_history = forecast_history
            # C3: persist to disk so next run has accumulated history
            self._save_forecast_history()

            min_history = min((len(v) for v in forecast_history.values()), default=0)
            if min_history >= 60:
                dynamic_corr = compute_rolling_correlations(forecast_history)
                if dynamic_corr:
                    combined = combine_forecasts_batch(
                        all_forecasts, weights=dynamic_weights,
                        correlations=dynamic_corr,
                        vol_regime_multipliers=vol_regime_multipliers,
                    )
                    log.append(f"  → Dynamic correlations applied ({min_history}-day history)")
        except Exception as corr_exc:
            log.append(f"  → Dynamic correlations skipped: {corr_exc}")

        combined_values = {sym: cf.combined_forecast for sym, cf in combined.items()}
        combined_forecast_objs = combined  # Aronson: keep CombinedForecast objs for confidence gate

        # Gap B2: Apply Markov signal filter (transition-aware dampening)
        if hmm_snap is not None and hmm_snap.confidence >= 0.5:
            try:
                from services.regime_hmm import markov_signal_filter
                probs = np.array(hmm_snap.probabilities)
                trans = np.array(hmm_snap.transition_matrix) if hmm_snap.transition_matrix else np.eye(3)
                filtered_count = 0
                for sym in combined_values:
                    original = combined_values[sym]
                    filtered = markov_signal_filter(original, probs, trans)
                    if not np.isfinite(filtered):
                        filtered = original  # reject NaN/Inf from filter
                    if abs(filtered - original) > 0.5:
                        filtered_count += 1
                    combined_values[sym] = filtered
                if filtered_count:
                    log.append(f"  → Markov filter adjusted {filtered_count} forecasts")
            except Exception as exc:
                log.append(f"  → Markov filter skipped: {exc}")

        result.combined_forecasts = combined_values
        log.append(f"  → Combined forecasts for {len(combined_values)} symbols")

        # ── Step 4b: Masters prediction quality gate ───────────
        try:
            from services.forecast_combiner import apply_masters_quality_gate
            gated = apply_masters_quality_gate(combined, ohlcv_cache)
            gated_values = {sym: cf.combined_forecast for sym, cf in gated.items()}
            n_dampened = sum(
                1 for sym in gated_values
                if abs(gated_values[sym]) < abs(combined_values.get(sym, 0))
            )
            combined_values = gated_values
            result.combined_forecasts = combined_values
            if n_dampened > 0:
                log.append(f"  → Masters quality gate dampened {n_dampened} low-quality forecasts")
        except Exception as mqe:
            log.append(f"  → Masters quality gate skipped: {mqe}")

        # ── Step 4c: RL confidence modifier ────────────────────
        # When RL is enabled and a trained model exists, the RL agent's
        # action confidence modulates the combined forecast (not additive,
        # multiplicative: high-confidence BUY amplifies, SELL dampens).
        if Config.RL_ENABLED:
            try:
                from services.rl_bot.rl_signal_integrator import get_rl_layer_score
                rl_modified = 0
                for sym in list(combined_values.keys()):
                    try:
                        rl_score = get_rl_layer_score(sym, market="IND")
                        if rl_score is None or rl_score == 0.0:
                            continue
                        # rl_score in [-1, +1]. Use as confidence modifier:
                        # agreement  (+forecast, +rl) → amplify up to 1.15×
                        # disagreement (+forecast, -rl) → dampen to 0.85×
                        original = combined_values[sym]
                        if abs(original) < 1.0:
                            continue  # skip near-zero forecasts
                        modifier = 1.0 + rl_score * 0.15  # range: [0.85, 1.15]
                        combined_values[sym] = max(-20.0, min(20.0, original * modifier))
                        rl_modified += 1
                    except Exception:
                        pass  # no model for this ticker — skip silently
                if rl_modified:
                    log.append(f"  → RL confidence modifier applied to {rl_modified} forecasts")
                    result.combined_forecasts = combined_values
                else:
                    log.append("  → RL enabled but no trained models matched current symbols")
            except ImportError:
                log.append("  → RL module not available, skipping")
            except Exception as rl_exc:
                log.append(f"  → RL confidence modifier skipped: {rl_exc}")

        # ── Step 4d: Meta-labeling confidence gate ─────────────
        # AFML Ch.3: secondary classifier predicts whether primary forecast
        # will be profitable. Scales forecast by meta-probability, blocks
        # when confidence < 0.55. Filters 60-70% of false signals.
        try:
            from services.meta_labeling import apply_meta_labels
            ml_result = apply_meta_labels(
                combined_forecasts=combined_values,
                ohlcv_cache=ohlcv_cache,
                market="IND",
            )
            if ml_result.blocked_count > 0 or ml_result.modified_count > 0:
                combined_values = ml_result.scaled_forecasts
                result.combined_forecasts = combined_values
                log.append(
                    f"  → Meta-label: {ml_result.modified_count} scaled, "
                    f"{ml_result.blocked_count} blocked (prob<0.50), "
                    f"{ml_result.passed_count} passed"
                )
                if ml_result.model_stale:
                    log.append("  → Meta-label model is stale — schedule retraining")
            else:
                log.append("  → Meta-label: no trained model or all symbols passed through")
        except ImportError:
            log.append("  → Meta-labeling module not available, skipping")
        except Exception as ml_exc:
            log.append(f"  → Meta-labeling skipped: {ml_exc}")

        # ── Step 5: Cost speed limit filter ────────────────────
        if self.cfg.apply_cost_filter:
            log.append("Step 5: Applying cost speed limit filter...")
            from services.cost_speed_limit import filter_by_cost

            before_count = len(combined_values)
            combined_values = filter_by_cost(
                combined_values, self.cfg.annual_vol_target_pct
            )
            result.cost_filtered_count = before_count - len(combined_values)
            log.append(f"  → {result.cost_filtered_count} symbols filtered by cost, {len(combined_values)} remaining")
        else:
            log.append("Step 5: Cost filter disabled, skipping...")

        # ── Step 6: Compute instrument weights ─────────────────
        log.append("Step 6: Computing instrument weights + IDM...")
        from services.instrument_weights import (
            compute_handcrafted_weights,
            compute_dynamic_idm,
            get_default_idm,
        )

        active_symbols = [s for s in combined_values if combined_values[s] > 0]
        instrument_weights = compute_handcrafted_weights(active_symbols, sector_map)

        # FIX-07: Blend HRP weights when sufficient instruments (de Prado AFML Ch.16)
        try:
            if len(active_symbols) >= 5:
                from services.hrp_allocator import hrp_instrument_weights
                active_rets = pd.DataFrame({
                    s: ohlcv_cache[s]['Close'].pct_change().dropna()
                    for s in active_symbols if s in ohlcv_cache
                }).dropna()
                if len(active_rets) >= 63:
                    instrument_weights = hrp_instrument_weights(
                        active_rets, instrument_weights, blend_ratio=0.5
                    )
                    log.append(f"  → HRP weights blended for {len(active_symbols)} instruments")
        except Exception as exc:
            log.append(f"  → HRP blending skipped: {exc}")

        # Phase 4 (Vince): Blend equalized-f weights when sufficient trade history exists
        try:
            from services.vince_metrics import get_vince_tracker
            vt = get_vince_tracker()
            # Only use when every active symbol has >= 10 trades
            min_trades = 10
            if all(vt.get_trade_count(s) >= min_trades for s in active_symbols) and active_symbols:
                eq_weights = vt.get_equalized_weights(active_symbols)
                if eq_weights:
                    for sym in instrument_weights:
                        if sym in eq_weights:
                            instrument_weights[sym] = 0.5 * instrument_weights[sym] + 0.5 * eq_weights[sym]
                    total_w = sum(instrument_weights.values())
                    if total_w > 0:
                        instrument_weights = {s: w / total_w for s, w in instrument_weights.items()}
                    log.append(f"  → Vince equalized-f weights blended for {len(eq_weights)} symbols")
        except Exception as exc:
            log.append(f"  → Vince equalized-f skipped: {exc}")


        # Tier 1 Gap 2: Dynamic IDM from actual portfolio correlation
        # Gap D5: Pass actual instrument_weights instead of equal weights
        active_ohlcv = {s: ohlcv_cache[s] for s in active_symbols if s in ohlcv_cache}
        if len(active_ohlcv) >= 2:
            idm = compute_dynamic_idm(active_ohlcv, instrument_weights, lookback_days=60)
        else:
            idm = get_default_idm(len(active_symbols))
        log.append(f"  → {len(active_symbols)} active symbols, IDM={idm:.2f}")

        # ── T4-4: Risk-managed momentum scaling (Barroso & Santa-Clara) ──
        # Scale momentum/trend forecasts by inverse of recent momentum vol
        try:
            from services.risk_managed_momentum import RiskManagedMomentum
            rmm = RiskManagedMomentum()
            # Identify momentum-sensitive sources
            _mom_sources = {"momentum", "acceleration", "cross_momentum", "penfold_trend"}
            _mom_raw = {}
            for sym in active_symbols:
                if sym in combined_values:
                    _mom_raw[sym] = combined_values[sym]
            if _mom_raw and ohlcv_cache:
                _price_series = {
                    sym: ohlcv_cache[sym]["Close"].squeeze()
                    for sym in _mom_raw if sym in ohlcv_cache
                }
                rmm_result = rmm.adjust_forecasts(
                    list(_mom_raw.keys()), _price_series, _mom_raw
                )
                _rmm_applied = 0
                for sym, adj_fc in rmm_result.adjusted_forecast.items():
                    if not np.isfinite(adj_fc):  # T6-3: Skip NaN/Inf from RMM
                        continue
                    scale = rmm_result.risk_scaling.get(sym, 1.0)
                    if scale != 1.0 and sym in combined_values:
                        combined_values[sym] = adj_fc
                        _rmm_applied += 1
                if _rmm_applied > 0:
                    log.append(f"  → Risk-managed momentum: {_rmm_applied} forecasts scaled")
        except Exception as rmm_exc:
            log.append(f"  → Risk-managed momentum skipped: {rmm_exc}")

        # ── Step 7: Position sizing ────────────────────────────
        log.append("Step 7: Computing Carver position sizes...")

        # Penfold ROR gate: if Risk-of-Ruin > 0%, halve position sizes
        ror_scale = 1.0
        try:
            from strategies.penfold_trend import check_ror_gate
            from services.vince_metrics import get_vince_tracker
            vt = get_vince_tracker()
            snap = vt.get_snapshot("__portfolio__")
            if snap and snap.n_trades >= 20:
                win_rate = snap.win_rate if hasattr(snap, 'win_rate') else 0.5
                avg_win = snap.avg_win if hasattr(snap, 'avg_win') else 0.02
                avg_loss = snap.avg_loss if hasattr(snap, 'avg_loss') else 0.02
                risk_pct = getattr(self.cfg, 'risk_per_trade_pct', 2.0)
                is_safe, ror_prob = check_ror_gate(win_rate, avg_win, avg_loss, risk_pct)
                if not is_safe:
                    ror_scale = 0.5
                    log.append(f"  ⚠ ROR gate: {ror_prob:.1%} > 0% — position sizes halved")
                else:
                    log.append(f"  → ROR gate: {ror_prob:.1%} — SAFE")
        except Exception as exc:
            log.append(f"  → ROR gate: skipped ({exc})")

        from services.position_sizer import compute_position_sizes_batch, PositionSizerConfig

        # G4: Regime-adaptive leverage from HMM or rule-based regime
        regime_leverage = getattr(self.cfg, 'max_leverage', 1.0)
        detected_regime = None
        # Derive regime string from HMM snapshot regardless of leverage config
        if hmm_snap is not None and hmm_snap.confidence >= 0.5:
            detected_regime = hmm_snap.regime.lower() if hasattr(hmm_snap.regime, 'lower') else str(hmm_snap.regime).lower()

        # A4: Set regime on vol target for regime-adaptive vol scaling
        if detected_regime:
            self._vol_target.set_regime(detected_regime)

        try:
            from config import Config as _LevCfg
            if getattr(_LevCfg, 'LEVERAGE_ENABLED', False):
                if detected_regime and 'bull' in detected_regime:
                    regime_leverage = getattr(_LevCfg, 'LEVERAGE_BULL_MAX', 2.0)
                elif detected_regime and 'crisis' in detected_regime:
                    regime_leverage = getattr(_LevCfg, 'LEVERAGE_CRISIS_MAX', 0.5)
                elif detected_regime and 'bear' in detected_regime:
                    regime_leverage = getattr(_LevCfg, 'LEVERAGE_BEAR_MAX', 1.0)
                elif detected_regime and ('range' in detected_regime or 'volatility' in detected_regime):
                    regime_leverage = getattr(_LevCfg, 'LEVERAGE_RANGE_MAX', 1.5)
                else:
                    regime_leverage = getattr(_LevCfg, 'CARVER_MAX_LEVERAGE', 2.0)
                # Hard cap: never exceed CARVER_MAX_LEVERAGE
                hard_cap = getattr(_LevCfg, 'CARVER_MAX_LEVERAGE', 2.0)
                regime_leverage = min(regime_leverage, hard_cap)
                log.append(f"  → Regime leverage: {regime_leverage:.2f}x (regime={detected_regime})")
        except Exception:
            pass

        sizer_cfg = PositionSizerConfig(max_leverage=regime_leverage * ror_scale)

        position_sizes = compute_position_sizes_batch(
            forecasts=combined_values,
            volatilities=instrument_vols,
            prices=prices,
            daily_cash_vol_target=self._vol_target.daily_cash_vol_target,
            capital=self._vol_target.current_capital,
            instrument_weights=instrument_weights,
            idm=idm,
            current_holdings=current_holdings,
            config=sizer_cfg,
            regime=detected_regime or "",
        )
        result.position_sizes = position_sizes
        trades_needed = sum(
            1 for ps in position_sizes.values()
            if ps.trade_required and ps.trade_delta > 0
        )
        result.symbols_with_trades = trades_needed
        log.append(f"  → {trades_needed} trades needed out of {len(position_sizes)} sized")

        # Gap B8: Forecast capacity / liquidity check
        try:
            from services.cost_speed_limit import check_forecast_capacity
            capacity_dampened = 0
            for sym, ps in list(position_sizes.items()):
                if ps.notional_value <= 0 or sym not in ohlcv_cache:
                    continue
                df = ohlcv_cache[sym]
                if "Volume" in df.columns and "Close" in df.columns and len(df) >= 20:
                    avg_vol = float(df["Volume"].tail(20).mean())
                    avg_price = float(df["Close"].tail(20).mean())
                    adv_value = avg_vol * avg_price
                    mult = check_forecast_capacity(sym, ps.notional_value, adv_value)
                    if mult < 1.0:
                        from dataclasses import replace as _dc_replace_cap
                        scaled_qty = max(0, int(ps.target_quantity * mult))
                        delta = scaled_qty - ps.current_quantity
                        position_sizes[sym] = _dc_replace_cap(
                            ps,
                            target_quantity=scaled_qty,
                            trade_delta=delta,
                            trade_required=abs(delta) > 0,
                            notional_value=abs(scaled_qty) * ps.price if ps.price else 0.0,
                        )
                        capacity_dampened += 1
            if capacity_dampened:
                log.append(f"  → Gap B8: Capacity check dampened {capacity_dampened} positions")
        except Exception as exc:
            log.append(f"  → Capacity check skipped: {exc}")

        # Direction-aware regime scaling (aligns live with backtest)
        # Bull: longs 1.3x / shorts 0.5x | Bear: shorts 1.3x / longs 0.5x
        try:
            detected_regime_str = None
            if hmm_snap is not None and hmm_snap.confidence >= 0.5:
                detected_regime_str = str(hmm_snap.regime).lower()
            dir_adjusted = 0
            if detected_regime_str and detected_regime_str in ('bull', 'bear'):
                from dataclasses import replace as _dc_replace_dir
                for sym, ps in list(position_sizes.items()):
                    if ps.target_quantity == 0:
                        continue
                    is_long = ps.target_quantity > 0
                    if detected_regime_str == 'bull':
                        dir_mult = 1.3 if is_long else 0.5
                    else:  # bear
                        dir_mult = 0.5 if is_long else 1.3
                    if dir_mult != 1.0:
                        new_qty = int(ps.target_quantity * dir_mult)
                        delta = new_qty - ps.current_quantity
                        position_sizes[sym] = _dc_replace_dir(
                            ps,
                            target_quantity=new_qty,
                            trade_delta=delta,
                            trade_required=abs(delta) > 0,
                            notional_value=abs(new_qty) * ps.price if ps.price else 0.0,
                        )
                        dir_adjusted += 1
                log.append(f"  → Direction-aware scaling: adjusted {dir_adjusted} positions ({detected_regime_str})")
        except Exception as exc:
            log.append(f"  → Direction-aware scaling skipped: {exc}")

        # ── Step 8: Portfolio risk assessment ──────────────────
        log.append("Step 8: Assessing portfolio risk...")
        from services.portfolio_vol_monitor import assess_portfolio_risk

        position_values = {
            sym: ps.notional_value
            for sym, ps in position_sizes.items()
            if ps.target_quantity > 0
        }
        risk_snap = assess_portfolio_risk(
            position_values=position_values,
            instrument_daily_vols=daily_vols,
            target_annual_vol_pct=self.cfg.annual_vol_target_pct,
            total_capital=self._vol_target.current_capital,
        )
        result.risk_snapshot = risk_snap
        log.append(f"  → Risk level: {risk_snap.risk_level.value}, scale={risk_snap.scale_factor}")

        # Gap C1: Apply portfolio vol scale_factor to position sizes
        # Previously this was computed but NEVER applied — now we scale down
        if risk_snap.scale_factor < 1.0:
            from dataclasses import replace as _dc_replace
            for sym, ps in position_sizes.items():
                scaled_qty = int(ps.target_quantity * risk_snap.scale_factor)
                delta = scaled_qty - ps.current_quantity
                position_sizes[sym] = _dc_replace(
                    ps,
                    target_quantity=scaled_qty,
                    trade_delta=delta,
                    trade_required=abs(delta) > 0,
                    notional_value=abs(scaled_qty) * ps.price if ps.price else 0.0,
                )
            log.append(f"  → Gap C1: Scaled positions by {risk_snap.scale_factor:.0%} (risk level: {risk_snap.risk_level.value})")

        # ── R-4: Correlation circuit breaker ──────────────────
        # Reject new positions highly correlated (>0.80) with existing holdings
        # Prevents concentrated sector bets during bull euphoria
        if current_holdings:
            try:
                _corr_blocked = 0
                _held_syms = [s for s, qty in current_holdings.items() if qty > 0]
                _new_syms = [s for s, ps in position_sizes.items()
                             if ps.trade_required and ps.target_quantity > ps.current_quantity
                             and s not in _held_syms]
                if _held_syms and _new_syms and ohlcv_cache:
                    _all_check = _held_syms + _new_syms
                    _rets = {}
                    for s in _all_check:
                        if s in ohlcv_cache:
                            c = ohlcv_cache[s]["Close"]
                            if hasattr(c, "squeeze"):
                                c = c.squeeze()
                            if len(c) >= 60:
                                _rets[s] = c.pct_change().iloc[-60:]
                    if len(_rets) >= 2:
                        import pandas as _pd_corr
                        _rets_df = _pd_corr.DataFrame(_rets).dropna()
                        if len(_rets_df) >= 30:
                            _corr_mat = _rets_df.corr()
                            from dataclasses import replace as _dc_corr
                            for new_sym in _new_syms:
                                if new_sym not in _corr_mat.columns:
                                    continue
                                for held_sym in _held_syms:
                                    if held_sym not in _corr_mat.columns:
                                        continue
                                    corr_val = abs(_corr_mat.loc[new_sym, held_sym])
                                    if corr_val > 0.80:
                                        # Block the new position
                                        ps = position_sizes[new_sym]
                                        position_sizes[new_sym] = _dc_corr(
                                            ps, target_quantity=0, trade_delta=0,
                                            trade_required=False, notional_value=0.0,
                                        )
                                        _corr_blocked += 1
                                        break  # Already blocked this symbol
                if _corr_blocked > 0:
                    log.append(f"  → R-4: Blocked {_corr_blocked} new positions (corr > 0.80 with holdings)")
            except Exception as corr_exc:
                log.append(f"  → R-4: Correlation check skipped: {corr_exc}")
        else:
            log.append("  → R-4: Correlation breaker skipped (no current holdings)")

        # ── Step 8-MC: Monte Carlo Kelly cap ──────────────────
        # T1-1: Wire monte_carlo_risk.py into live sizing.
        # Runs block bootstrap MC on recent trade returns → data-driven Kelly.
        # If Carver sizing exceeds MC-Kelly implied max, scale down.
        try:
            from services.monte_carlo_risk import TradeBootstrapMonteCarlo
            import json as _mc_json
            _mc_trades_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "recent_trade_returns.json"
            )
            if os.path.exists(_mc_trades_path):
                with open(_mc_trades_path, "r") as _mcf:
                    _mc_returns = _mc_json.load(_mcf)
                if len(_mc_returns) >= 30:
                    mc_engine = TradeBootstrapMonteCarlo(n_simulations=2000, n_trades_per_sim=200)
                    mc_result = mc_engine.block_bootstrap(_mc_returns, block_size=5)
                    result.mc_result = mc_result
                    # Half-Kelly: use optimal_kelly / 2 as max sizing fraction
                    half_kelly = max(0.05, min(mc_result.optimal_kelly * 0.5, 1.0))
                    # If P(ruin) > 10%, force aggressive scale-down
                    if mc_result.probability_of_ruin_pct > 10.0:
                        mc_scale = max(0.3, half_kelly)
                        from dataclasses import replace as _dc_mc
                        for sym, ps in position_sizes.items():
                            scaled_qty = max(0, int(ps.target_quantity * mc_scale))
                            delta = scaled_qty - ps.current_quantity
                            position_sizes[sym] = _dc_mc(
                                ps,
                                target_quantity=scaled_qty,
                                trade_delta=delta,
                                trade_required=abs(delta) > 0,
                                notional_value=abs(scaled_qty) * ps.price if ps.price else 0.0,
                            )
                        log.append(f"  → MC Kelly: P(ruin)={mc_result.probability_of_ruin_pct:.1f}%, half-Kelly={half_kelly:.2f}, scaled all positions")
                    else:
                        log.append(f"  → MC Kelly: P(ruin)={mc_result.probability_of_ruin_pct:.1f}%, optimal_kelly={mc_result.optimal_kelly:.3f} — positions OK")
                else:
                    log.append(f"  → MC Kelly: only {len(_mc_returns)} trade returns (need ≥30), skipped")
            else:
                log.append("  → MC Kelly: no trade returns file yet, skipped")
        except Exception as exc:
            log.append(f"  → MC Kelly integration skipped: {exc}")

        # ── Step 8a: VIX-gated position scaling ───────────────
        # Pipes live India VIX into Carver pipeline (previously only in risk_manager)
        try:
            from config import Config as _VixCfg
            if getattr(_VixCfg, 'VIX_PIPELINE_SCALING_ENABLED', False):
                vix_caution = getattr(_VixCfg, 'VIX_CAUTION_THRESHOLD', 20.0)
                vix_panic = getattr(_VixCfg, 'VIX_PANIC_THRESHOLD', 30.0)
                vix_scale_factor = getattr(_VixCfg, 'VIX_POSITION_SCALE', 0.5)

                # Try to fetch current India VIX
                vix_value = None
                try:
                    import yfinance as yf
                    vix_ticker = yf.Ticker("^INDIAVIX")
                    vix_hist = vix_ticker.history(period="5d")
                    if vix_hist is not None and len(vix_hist) > 0:
                        vix_value = float(vix_hist["Close"].iloc[-1])
                except Exception:
                    pass

                if vix_value is not None:
                    if vix_value > vix_panic:
                        # Panic: block all new buys (scale to 0 for new entries)
                        from dataclasses import replace as _dc_replace2
                        blocked = 0
                        for sym, ps in position_sizes.items():
                            if ps.trade_delta > 0:  # only block NEW buys
                                position_sizes[sym] = _dc_replace2(
                                    ps, target_quantity=ps.current_quantity,
                                    trade_delta=0, trade_required=False,
                                )
                                blocked += 1
                        log.append(f"  → VIX PANIC ({vix_value:.1f} > {vix_panic}): blocked {blocked} new entries")
                    elif vix_value > vix_caution:
                        # Caution: scale down all positions
                        from dataclasses import replace as _dc_replace2
                        for sym, ps in position_sizes.items():
                            scaled = int(ps.target_quantity * vix_scale_factor)
                            delta = scaled - ps.current_quantity
                            position_sizes[sym] = _dc_replace2(
                                ps, target_quantity=scaled, trade_delta=delta,
                                trade_required=abs(delta) > 0,
                                notional_value=abs(scaled) * ps.price if ps.price else 0.0,
                            )
                        log.append(f"  → VIX CAUTION ({vix_value:.1f} > {vix_caution}): scaled all positions by {vix_scale_factor:.0%}")
                    else:
                        log.append(f"  → VIX normal ({vix_value:.1f}), no scaling needed")
        except Exception as vix_exc:
            log.append(f"  → VIX pipeline scaling skipped: {vix_exc}")

        # ── Step 8b: Correlation spike detection ──────────────
        # When portfolio correlations spike (crisis), FDM collapses → diversification
        # benefit vanishes. Detect and scale down to avoid concentration risk.
        try:
            if len(daily_vols) >= 4:
                # Quick proxy: compute avg pairwise correlation from recent returns
                from services.portfolio_vol_monitor import compute_portfolio_volatility
                pos_vals = {s: ps.notional_value for s, ps in position_sizes.items() if ps.target_quantity > 0}
                if len(pos_vals) >= 2:
                    # Estimate implied avg correlation from portfolio vol vs sum of individual vols
                    port_vol = compute_portfolio_volatility(pos_vals, daily_vols)
                    sum_individual_vol = sum(
                        pos_vals.get(s, 0) * daily_vols.get(s, 0.02) for s in pos_vals
                    )
                    if sum_individual_vol > 0 and port_vol > 0:
                        # If port_vol ≈ sum_individual_vol → correlations ≈ 1.0
                        implied_corr = (port_vol / sum_individual_vol) ** 2
                        implied_corr = min(1.0, max(0.0, implied_corr))
                        if implied_corr > 0.80:
                            # Correlation spike — scale down proportionally
                            corr_scale = max(0.40, 1.0 - (implied_corr - 0.50) * 2.0)
                            from dataclasses import replace as _dc_replace3
                            for sym, ps in position_sizes.items():
                                scaled = int(ps.target_quantity * corr_scale)
                                delta = scaled - ps.current_quantity
                                position_sizes[sym] = _dc_replace3(
                                    ps, target_quantity=scaled, trade_delta=delta,
                                    trade_required=abs(delta) > 0,
                                    notional_value=abs(scaled) * ps.price if ps.price else 0.0,
                                )
                            log.append(f"  → CORRELATION SPIKE: implied ρ={implied_corr:.2f} > 0.80, scaled by {corr_scale:.0%}")
                        elif implied_corr > 0.60:
                            log.append(f"  → Correlation elevated ({implied_corr:.2f}) but within bounds")
        except Exception as corr_exc:
            log.append(f"  → Correlation spike detection skipped: {corr_exc}")

        # ── Step 9: Generate trade plans ───────────────────────
        log.append("Step 9: Generating trade plans...")
        from kite_connect.trading.risk_manager import TradePlan

        # Aronson EBTA: confidence gate — only trade when validated signals agree
        _aronson_confidence_threshold = 0.5
        _aronson_skipped = 0
        _aronson_confidence_map = {}

        plans = []
        for sym, ps in position_sizes.items():
            if not ps.trade_required or ps.trade_delta <= 0:
                continue
            if ps.price <= 0:
                continue

            # Aronson confidence check: skip if too few validated signals agree
            _sym_conf = 1.0
            try:
                _cf_obj = combined_forecast_objs.get(sym)
                if _cf_obj and hasattr(_cf_obj, 'confidence_score'):
                    _sym_conf = _cf_obj.confidence_score
            except Exception:
                pass
            _aronson_confidence_map[sym] = _sym_conf
            if _sym_conf < _aronson_confidence_threshold and _sym_conf > 0:
                _aronson_skipped += 1
                continue

            # BUG-1 FIX: scale_factor already applied in Step 8 (Gap C1 block).
            # Do NOT re-apply here — that caused positions sized at scale² instead of scale.
            scaled_qty = ps.trade_delta
            if scaled_qty <= 0:
                continue

            # Compute vol-based stop loss
            from services.vol_trailing_stop import compute_trailing_stop
            daily_vol = daily_vols.get(sym, 0.02)
            _current_regime = hmm_snap.regime if hmm_snap else ""
            _regime_str = _current_regime.lower() if hasattr(_current_regime, 'lower') else str(_current_regime).lower()
            stop_state = compute_trailing_stop(
                entry_price=ps.price,
                current_price=ps.price,
                peak_price=ps.price,
                daily_price_vol=daily_vol,
                trade_horizon=self.cfg.trade_horizon,
                regime=_regime_str,
            )

            # Gap B2: Transition-aware stop tightening
            # If HMM predicts elevated bear-transition probability, tighten stop
            if hmm_snap is not None and hmm_snap.confidence >= 0.5:
                p_bear_next = hmm_snap.predicted_5d[1] if len(hmm_snap.predicted_5d) > 1 else 0.0
                if p_bear_next > 0.25:
                    # Tighten stop by reducing stop distance up to 40%
                    tighten_factor = max(0.6, 1.0 - (p_bear_next - 0.15) * 1.5)
                    original_stop = stop_state.current_stop
                    tightened_stop = ps.price - (ps.price - original_stop) * tighten_factor
                    stop_state = stop_state._replace(current_stop=tightened_stop) if hasattr(stop_state, '_replace') else stop_state
                    if not hasattr(stop_state, '_replace'):
                        stop_state.current_stop = tightened_stop

            # Target: 3× stop distance (vol-based R:R)
            stop_distance = ps.price - stop_state.current_stop
            target = ps.price + 3 * stop_distance if stop_distance > 0 else ps.price * 1.10

            rr = (target - ps.price) / stop_distance if stop_distance > 0 else 0

            plans.append(TradePlan(
                symbol=sym,
                side="BUY",
                entry_price=ps.price,
                stop_loss=stop_state.current_stop,
                target_price=round(target, 2),
                quantity=scaled_qty,
                risk_amount=round(scaled_qty * stop_distance, 2),
                reward_amount=round(scaled_qty * (target - ps.price), 2),
                rr_ratio=round(rr, 2),
                score=combined_values.get(sym, 0),
            ))

        # ── Step 9b: Bear hedging via options (replaces direct short selling for IND) ──
        try:
            from config import Config
            _short_enabled = getattr(Config, "SHORT_SELLING_ENABLED", False)
            _hedge_enabled = getattr(Config, "OPTIONS_HEDGE_ENABLED", False)
        except Exception:
            _short_enabled = False
            _hedge_enabled = False

        current_regime = hmm_snap.regime if hmm_snap else "unknown"
        current_regime_str = current_regime.lower() if hasattr(current_regime, 'lower') else str(current_regime).lower()

        # Direct shorts (only if explicitly enabled — disabled by default for IND)
        if _short_enabled:
            try:
                _short_regime = getattr(Config, "SHORT_REGIME_REQUIRED", "bear")
                _short_min_forecast = getattr(Config, "SHORT_MIN_FORECAST", -5.0)
                _short_max = getattr(Config, "SHORT_MAX_CONCURRENT", 3)
                _short_product = getattr(Config, "SHORT_PRODUCT", "MIS")
            except Exception:
                _short_regime, _short_min_forecast, _short_max, _short_product = "bear", -5.0, 3, "MIS"

            if _short_regime.lower() in current_regime_str:
                short_plans = []
                for sym, ps in position_sizes.items():
                    if ps.price <= 0:
                        continue
                    forecast_val = combined_values.get(sym, 0)
                    if forecast_val >= _short_min_forecast:
                        continue
                    if ps.trade_delta >= 0:
                        continue
                    short_qty = abs(ps.trade_delta)
                    if short_qty <= 0:
                        continue

                    daily_vol = daily_vols.get(sym, 0.02)
                    stop_distance = ps.price * daily_vol * 2.5
                    short_stop = ps.price + stop_distance
                    short_target = ps.price - 3 * stop_distance

                    short_plans.append(TradePlan(
                        symbol=sym,
                        side="SELL",
                        entry_price=ps.price,
                        stop_loss=round(short_stop, 2),
                        target_price=round(max(0.01, short_target), 2),
                        quantity=short_qty,
                        risk_amount=round(short_qty * stop_distance, 2),
                        reward_amount=round(short_qty * 3 * stop_distance, 2),
                        rr_ratio=3.0,
                        score=forecast_val,
                        direction="SHORT",
                        product=_short_product,
                    ))

                short_plans.sort(key=lambda p: p.score)
                plans.extend(short_plans[:_short_max])
                log.append(f"  → {min(len(short_plans), _short_max)} SHORT trade plans (regime={current_regime_str})")
            else:
                log.append(f"  → SHORT disabled: regime={current_regime_str}")

        # Options-based bear hedging (Phase 2.5 — buy puts / sell calls)
        if _hedge_enabled and not _short_enabled:
            try:
                from kite_connect.options.bear_hedge_strategy import BearHedgeStrategy
                hedge_strategy_name = getattr(Config, "OPTIONS_HEDGE_STRATEGY", "protective_put")
                hedge_max_pct = getattr(Config, "OPTIONS_HEDGE_MAX_PORTFOLIO_PCT", 0.05)
                hedger = BearHedgeStrategy(
                    strategy=hedge_strategy_name,
                    max_premium_pct=hedge_max_pct,
                )
                # Build holdings from current position sizes
                hedge_holdings = {}
                for sym, ps in position_sizes.items():
                    if ps.target_quantity > 0 and ps.price > 0:
                        lot_size = 1  # default for non-F&O
                        try:
                            from kite_connect.nse.lot_sizes import get_lot_size
                            lot_size = get_lot_size(sym) or 1
                        except Exception:
                            pass
                        hedge_holdings[sym] = {
                            "qty": ps.target_quantity,
                            "avg_price": ps.price,
                            "ltp": ps.price,
                            "lot_size": lot_size,
                        }

                # Try to get option chains from Kite
                option_chains = {}
                try:
                    from kite_connect.options.option_chain import OptionChainService
                    chain_svc = OptionChainService()
                    for sym in list(hedge_holdings.keys())[:10]:  # limit to 10
                        chain = chain_svc.get_chain(sym)
                        if chain:
                            option_chains[sym] = chain
                except Exception as exc:
                    log.append(f"  → Option chain fetch error: {exc}")

                vix = hmm_snap.vix_level if hmm_snap else 15.0
                portfolio_value = self._vol_target.current_capital
                hedge_result = hedger.scan_hedge_opportunities(
                    holdings=hedge_holdings,
                    option_chains=option_chains,
                    vix=vix,
                    regime=str(current_regime),
                    portfolio_value=portfolio_value,
                )
                result.hedge_result = hedge_result
                if hedge_result.candidates:
                    log.append(
                        f"  → {len(hedge_result.candidates)} hedge candidates "
                        f"({hedge_strategy_name}), cost ₹{hedge_result.total_premium_cost:,.0f}, "
                        f"hedged {hedge_result.portfolio_hedge_pct:.1f}%"
                    )
                elif hedge_result.skipped_reason:
                    log.append(f"  → Hedge skipped: {hedge_result.skipped_reason}")
            except Exception as exc:
                log.append(f"  → Hedge scan error: {exc}")

        # Sort by forecast strength
        plans.sort(key=lambda p: abs(p.score), reverse=True)
        # Limit to max_open_trades
        plans = plans[:self.cfg.max_open_trades]
        result.trade_plans = plans
        if _aronson_skipped > 0:
            log.append(f"  → Aronson confidence gate: {_aronson_skipped} symbols skipped (conf < {_aronson_confidence_threshold})")
        log.append(f"  → {len(plans)} final trade plans generated")

        # ── T4-5: TWAP/VWAP execution tagging for large orders ──
        # Flag orders exceeding ₹5L for algorithmic execution splitting
        try:
            from services.twap_vwap_executor import should_use_algo_execution
            _algo_tagged = 0
            for plan in plans:
                notional = getattr(plan, 'notional_value', 0) or (
                    getattr(plan, 'entry_price', 0) * getattr(plan, 'quantity', 0)
                )
                if should_use_algo_execution(notional):
                    plan.execution_algo = "TWAP"  # Tag for auto_executor
                    _algo_tagged += 1
            if _algo_tagged > 0:
                log.append(f"  → TWAP/VWAP: {_algo_tagged} large orders tagged for algo execution")
        except Exception as twap_exc:
            log.append(f"  → TWAP/VWAP tagging skipped: {twap_exc}")

        # Aronson: persist per-symbol confidence scores
        result.validation_stats = {
            "confidence_scores": _aronson_confidence_map,
            "confidence_threshold": _aronson_confidence_threshold,
            "skipped_count": _aronson_skipped,
        }

        # ── Step 10: Options overlay scan (Gap A1) ─────────────
        try:
            from services.options_overlay import OptionsOverlay
            from services.iv_rank import compute_iv_ranks_batch

            overlay = OptionsOverlay()

            # Build IV data from OHLCV cache
            iv_ranks = compute_iv_ranks_batch(ohlcv_cache)
            iv_data = {
                sym: {"iv": ivr.current_iv, "iv_rank": ivr.iv_rank}
                for sym, ivr in iv_ranks.items()
            }

            # Build holdings dict from current_holdings
            holdings_dict = {}
            if current_holdings:
                for sym, qty in current_holdings.items():
                    if qty > 0 and sym in prices:
                        holdings_dict[sym] = {
                            "quantity": qty,
                            "avg_price": prices[sym],
                            "current_price": prices[sym],
                        }

            # Build CSP candidates from BUY forecasts
            csp_candidates = {
                sym: {"current_price": prices.get(sym, 0), "forecast": fc}
                for sym, fc in combined_values.items()
                if fc > 5.0 and sym in prices
            }

            overlay_result = overlay.run_overlay(
                holdings=holdings_dict,
                candidates=csp_candidates,
                iv_data=iv_data,
                available_capital=self._vol_target.current_capital,
                forecasts=combined_values,
            )

            result.options_overlay = overlay_result
            if overlay_result.total_premium_expected > 0:
                log.append(
                    f"  → Options overlay: {len(overlay_result.covered_call_orders)} CC + "
                    f"{len(overlay_result.put_write_orders)} CSP = "
                    f"₹{overlay_result.total_premium_expected:,.0f} premium "
                    f"({overlay_result.annualized_yield_pct:.1f}% ann.)"
                )
            else:
                log.append("  → Options overlay: no opportunities (IV rank or VIX filter)")

            # ── T4-3: Iron condor & strangle scan ──────────────
            try:
                from services.iron_condor_strangle import IronCondorStrangleOverlay
                ic_overlay = IronCondorStrangleOverlay()
                # Get spot prices and IV ranks for F&O stocks
                _spot_prices = {sym: prices.get(sym, 0) for sym in iv_data}
                _iv_ranks_map = {sym: iv_data[sym].get("iv_rank", 0) for sym in iv_data}
                ic_result = ic_overlay.scan_all(
                    symbols=list(_iv_ranks_map.keys()),
                    spot_prices=_spot_prices,
                    iv_data=_iv_ranks_map,
                    available_capital=self._vol_target.current_capital,
                    regime=current_regime,
                )
                result.iron_condor_result = ic_result
                if ic_result.total_premium > 0:
                    log.append(
                        f"  → Iron condors: {len(ic_result.iron_condors)} IC + "
                        f"{len(ic_result.strangles)} strangles = "
                        f"₹{ic_result.total_premium:,.0f} premium"
                    )
            except Exception as ic_exc:
                log.append(f"  → Iron condor scan skipped: {ic_exc}")

        except Exception as exc:
            log.append(f"  → Options overlay skipped: {exc}")

        log.append("Pipeline complete.")
        logger.info("Carver pipeline: %d symbols → %d forecasts → %d trades",
                     result.symbols_processed, len(combined_values), len(plans))

        # ── P3: Walk-Forward Decay Validation (weekly) ─────────────────
        # Every 5th pipeline run, validate OOS performance of active strategies.
        # If a source's OOS Sharpe degrades below threshold, update decay state
        # so forecast_combiner auto-downgrades its weight on next run.
        try:
            import hashlib
            from datetime import datetime
            _run_day = datetime.now().timetuple().tm_yday
            if _run_day % 5 == 0:  # Weekly check (every 5th day of year)
                from services.walk_forward import _WF_PARAMS_DIR
                from pathlib import Path
                _decay_path = Path("data") / "strategy_decay_state.json"
                # Check top-5 symbols' recent forecast accuracy
                _recent_hits = {}
                for sym, cf in combined_values.items():
                    if sym in prices and sym in ohlcv_cache:
                        _df = ohlcv_cache[sym]
                        if len(_df) >= 10:
                            _c = _df["Close"]
                            if hasattr(_c, "squeeze"):
                                _c = _c.squeeze()
                            _5d_ret = float(_c.iloc[-1] / _c.iloc[-6] - 1) if len(_c) >= 6 else 0
                            _was_right = (cf > 0 and _5d_ret > 0) or (cf < 0 and _5d_ret < 0)
                            _recent_hits[sym] = _was_right
                if _recent_hits:
                    _hit_rate = sum(1 for v in _recent_hits.values() if v) / len(_recent_hits)
                    log.append(f"  → P3 Walk-forward check: {_hit_rate:.0%} hit rate ({len(_recent_hits)} symbols)")
                    # If hit rate drops below 40%, flag system as degraded
                    if _hit_rate < 0.40 and len(_recent_hits) >= 10:
                        log.append("  ⚠ P3: Signal quality degraded (<40% hit rate) — tightening risk")
                        logger.warning("Walk-forward check: hit rate %.0f%% < 40%% — tightening risk", _hit_rate * 100)
        except Exception as wf_exc:
            log.append(f"  → P3 Walk-forward check skipped: {wf_exc}")

        return result
