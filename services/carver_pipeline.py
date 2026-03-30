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
        log.append("Step 1: Computing instrument volatilities...")
        from services.instrument_volatility import compute_volatilities_batch

        # Tier 1 Gap 5: Data freshness gate — skip symbols with stale OHLCV
        from datetime import datetime, timedelta
        from config import Config
        freshness_hours = getattr(Config, "SIGNAL_FRESHNESS_MAX_HOURS", 4)
        stale_cutoff = datetime.now() - timedelta(hours=freshness_hours)
        stale_symbols = []
        for sym, df in list(ohlcv_cache.items()):
            if df is not None and not df.empty and df.index.dtype.kind == "M":
                last_bar = df.index[-1]
                # Convert to tz-naive if needed
                if hasattr(last_bar, "tz") and last_bar.tz is not None:
                    last_bar = last_bar.tz_localize(None)
                if last_bar < stale_cutoff:
                    stale_symbols.append(sym)
                    del ohlcv_cache[sym]
        if stale_symbols:
            log.append(f"  ⚠ Freshness gate: dropped {len(stale_symbols)} stale symbols: {stale_symbols[:5]}...")
            logger.warning("Freshness gate dropped %d stale symbols (cutoff=%s)", len(stale_symbols), stale_cutoff)

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

        # Gap A6: F&O Open Interest signals
        oi_forecasts: Dict[str, float] = {}
        try:
            from services.oi_signal import compute_oi_signals_batch
            # OI data would come from Kite or cached; skip if unavailable
        except Exception as exc:
            log.append(f"  → OI signal skipped: {exc}")

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

            # Gap B6: Decision engine forecast integration
            if decision_engine_scores and sym in decision_engine_scores:
                fc["decision_engine"] = decision_engine_to_forecast(
                    decision_engine_scores[sym]
                )

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
                            sym_forecasts[source] *= mult
                log.append(f"  → Strategy decay applied: {sum(1 for m in decay_mults.values() if m < 1.0)} degraded")
        except Exception:
            pass

        combined = combine_forecasts_batch(all_forecasts, weights=dynamic_weights)
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
        from services.position_sizer import compute_position_sizes_batch

        position_sizes = compute_position_sizes_batch(
            forecasts=combined_values,
            volatilities=instrument_vols,
            prices=prices,
            daily_cash_vol_target=self._vol_target.daily_cash_vol_target,
            capital=self._vol_target.current_capital,
            instrument_weights=instrument_weights,
            idm=idm,
            current_holdings=current_holdings,
        )
        result.position_sizes = position_sizes
        trades_needed = sum(
            1 for ps in position_sizes.values()
            if ps.trade_required and ps.trade_delta > 0
        )
        result.symbols_with_trades = trades_needed
        log.append(f"  → {trades_needed} trades needed out of {len(position_sizes)} sized")

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

            # Apply risk snapshot scale factor
            scaled_qty = max(0, int(ps.trade_delta * risk_snap.scale_factor))
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

        # Sort by forecast strength
        plans.sort(key=lambda p: abs(p.score), reverse=True)
        # Limit to max_open_trades
        plans = plans[:self.cfg.max_open_trades]
        result.trade_plans = plans
        log.append(f"  → {len(plans)} final trade plans generated")

        log.append("Pipeline complete.")
        logger.info("Carver pipeline: %d symbols → %d forecasts → %d trades",
                     result.symbols_processed, len(combined_values), len(plans))

        return result
