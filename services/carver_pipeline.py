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
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


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
    """Configuration for the full Carver pipeline."""
    initial_capital: float = 500_000.0
    annual_vol_target_pct: float = 0.20
    max_open_trades: int = 8
    default_idm: float = 1.6
    apply_cost_filter: bool = True
    trade_horizon: str = "swing"  # "swing" or "positional"


@dataclass
class PipelineResult:
    """Output of a full pipeline execution."""
    trade_plans: List = field(default_factory=list)
    combined_forecasts: Dict[str, float] = field(default_factory=dict)
    position_sizes: Dict = field(default_factory=dict)
    risk_snapshot: Optional[object] = None
    options_overlay: Optional[object] = None   # Gap A1: OverlayResult
    symbols_processed: int = 0
    symbols_with_trades: int = 0
    cost_filtered_count: int = 0
    pipeline_log: List[str] = field(default_factory=list)


class CarverPipeline:
    """Orchestrates the full Carver systematic trading pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._vol_target = None
        self._hmm_model = None
        self._forecast_history = {}  # Rolling forecast history for dynamic correlations
        self._init_vol_target()

    def _init_vol_target(self):
        """Initialise the volatility target module."""
        from services.volatility_target import VolatilityTarget, VolatilityTargetConfig
        vt_config = VolatilityTargetConfig(
            initial_capital=self.cfg.initial_capital,
            annual_vol_target_pct=self.cfg.annual_vol_target_pct,
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
                log.append(f"  → Breakout forecasts for {len(breakout_forecasts)} symbols")
        except Exception as exc:
            log.append(f"  → Breakout skipped: {exc}")

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

            # Gap B6: Decision engine forecast integration
            # NOTE: Removed duplicate — decision_engine forecast already added above

            if fc:
                all_forecasts[sym] = fc

        log.append(f"  → Forecasts built for {len(all_forecasts)} symbols")
        result.symbols_processed = len(all_forecasts)

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

        combined = combine_forecasts_batch(all_forecasts, weights=dynamic_weights)

        # Dynamic correlation matrix: update from forecast history if available
        try:
            from services.forecast_combiner import compute_rolling_correlations
            forecast_history = getattr(self, '_forecast_history', {})
            # Append today's forecasts to history (averaged across symbols)
            for sym, fc_dict in all_forecasts.items():
                for source, val in fc_dict.items():
                    if source not in forecast_history:
                        forecast_history[source] = []
                    forecast_history[source].append(val)
            self._forecast_history = forecast_history

            min_history = min((len(v) for v in forecast_history.values()), default=0)
            if min_history >= 60:
                dynamic_corr = compute_rolling_correlations(forecast_history)
                if dynamic_corr:
                    combined = combine_forecasts_batch(
                        all_forecasts, weights=dynamic_weights,
                        correlations=dynamic_corr,
                    )
                    log.append(f"  → Dynamic correlations applied ({min_history}-day history)")
        except Exception as corr_exc:
            log.append(f"  → Dynamic correlations skipped: {corr_exc}")

        combined_values = {sym: cf.combined_forecast for sym, cf in combined.items()}

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

        # Tier 1 Gap 2: Dynamic IDM from actual portfolio correlation
        # Gap D5: Pass actual instrument_weights instead of equal weights
        active_ohlcv = {s: ohlcv_cache[s] for s in active_symbols if s in ohlcv_cache}
        if len(active_ohlcv) >= 2:
            idm = compute_dynamic_idm(active_ohlcv, instrument_weights, lookback_days=60)
        else:
            idm = get_default_idm(len(active_symbols))
        log.append(f"  → {len(active_symbols)} active symbols, IDM={idm:.2f}")

        # ── Step 7: Position sizing ────────────────────────────
        log.append("Step 7: Computing Carver position sizes...")
        from services.position_sizer import compute_position_sizes_batch, PositionSizerConfig

        # G4: Regime-adaptive leverage from HMM or rule-based regime
        regime_leverage = getattr(self.cfg, 'max_leverage', 1.0)
        try:
            from config import Config as _LevCfg
            if getattr(_LevCfg, 'LEVERAGE_ENABLED', False):
                detected_regime = None
                if hmm_snap is not None and hmm_snap.confidence >= 0.5:
                    detected_regime = hmm_snap.regime.lower() if hasattr(hmm_snap.regime, 'lower') else str(hmm_snap.regime).lower()
                if detected_regime and 'bull' in detected_regime:
                    regime_leverage = getattr(_LevCfg, 'LEVERAGE_BULL_MAX', 1.3)
                elif detected_regime and 'bear' in detected_regime:
                    regime_leverage = getattr(_LevCfg, 'LEVERAGE_BEAR_MAX', 0.8)
                elif detected_regime and 'range' in detected_regime:
                    regime_leverage = getattr(_LevCfg, 'LEVERAGE_RANGE_MAX', 1.15)
                else:
                    regime_leverage = getattr(_LevCfg, 'CARVER_MAX_LEVERAGE', 1.0)
                log.append(f"  → Regime leverage: {regime_leverage:.2f}x (regime={detected_regime})")
        except Exception:
            pass

        sizer_cfg = PositionSizerConfig(max_leverage=regime_leverage)

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

        # ── Step 9: Generate trade plans ───────────────────────
        log.append("Step 9: Generating trade plans...")
        from kite_connect.trading.risk_manager import TradePlan

        plans = []
        for sym, ps in position_sizes.items():
            if not ps.trade_required or ps.trade_delta <= 0:
                continue
            if ps.price <= 0:
                continue

            # BUG-1 FIX: scale_factor already applied in Step 8 (Gap C1 block).
            # Do NOT re-apply here — that caused positions sized at scale² instead of scale.
            scaled_qty = ps.trade_delta
            if scaled_qty <= 0:
                continue

            # Compute vol-based stop loss
            from services.vol_trailing_stop import compute_trailing_stop
            daily_vol = daily_vols.get(sym, 0.02)
            stop_state = compute_trailing_stop(
                entry_price=ps.price,
                current_price=ps.price,
                peak_price=ps.price,
                daily_price_vol=daily_vol,
                trade_horizon=self.cfg.trade_horizon,
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

        # ── Step 9b: SHORT trade plans (Phase 2 — Short Selling) ──
        try:
            from config import Config
            _short_enabled = getattr(Config, "SHORT_SELLING_ENABLED", False)
            _short_regime = getattr(Config, "SHORT_REGIME_REQUIRED", "bear")
            _short_min_forecast = getattr(Config, "SHORT_MIN_FORECAST", -5.0)
            _short_max = getattr(Config, "SHORT_MAX_CONCURRENT", 3)
            _short_product = getattr(Config, "SHORT_PRODUCT", "MIS")
        except Exception:
            _short_enabled = False

        if _short_enabled:
            current_regime = hmm_snap.regime if hmm_snap else "unknown"
            if current_regime.lower() == _short_regime.lower():
                short_plans = []
                for sym, ps in position_sizes.items():
                    if ps.price <= 0:
                        continue
                    forecast_val = combined_values.get(sym, 0)
                    if forecast_val >= _short_min_forecast:
                        continue  # Not bearish enough
                    if ps.trade_delta >= 0:
                        continue  # No negative delta
                    short_qty = abs(ps.trade_delta)
                    if short_qty <= 0:
                        continue

                    from services.vol_trailing_stop import compute_trailing_stop
                    daily_vol = daily_vols.get(sym, 0.02)
                    # For shorts: stop is ABOVE entry, target is BELOW
                    stop_distance = ps.price * daily_vol * 2.5  # 2.5σ stop
                    short_stop = ps.price + stop_distance
                    short_target = ps.price - 3 * stop_distance
                    rr = 3.0 if stop_distance > 0 else 0

                    short_plans.append(TradePlan(
                        symbol=sym,
                        side="SELL",
                        entry_price=ps.price,
                        stop_loss=round(short_stop, 2),
                        target_price=round(max(0.01, short_target), 2),
                        quantity=short_qty,
                        risk_amount=round(short_qty * stop_distance, 2),
                        reward_amount=round(short_qty * 3 * stop_distance, 2),
                        rr_ratio=round(rr, 2),
                        score=forecast_val,
                        direction="SHORT",
                        product=_short_product,
                    ))

                # Limit to max concurrent short positions
                short_plans.sort(key=lambda p: p.score)  # Most bearish first
                plans.extend(short_plans[:_short_max])
                log.append(f"  → {min(len(short_plans), _short_max)} SHORT trade plans added (regime={current_regime})")
            else:
                log.append(f"  → SHORT disabled: regime={current_regime} (need {_short_regime})")

        # Sort by forecast strength
        plans.sort(key=lambda p: abs(p.score), reverse=True)
        # Limit to max_open_trades
        plans = plans[:self.cfg.max_open_trades]
        result.trade_plans = plans
        log.append(f"  → {len(plans)} final trade plans generated")

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
        except Exception as exc:
            log.append(f"  → Options overlay skipped: {exc}")

        log.append("Pipeline complete.")
        logger.info("Carver pipeline: %d symbols → %d forecasts → %d trades",
                     result.symbols_processed, len(combined_values), len(plans))

        return result
