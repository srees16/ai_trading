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

        vol_data = compute_volatilities_batch(ohlcv_cache)
        instrument_vols = {}
        daily_vols = {}
        prices = {}
        for sym, vd in vol_data.items():
            instrument_vols[sym] = vd["instrument_value_vol"]
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

            if fc:
                all_forecasts[sym] = fc

        log.append(f"  → Forecasts built for {len(all_forecasts)} symbols")
        result.symbols_processed = len(all_forecasts)

        # ── Step 4: Combine forecasts ──────────────────────────
        log.append("Step 4: Combining forecasts with FDM...")
        from services.forecast_combiner import combine_forecasts_batch

        combined = combine_forecasts_batch(all_forecasts)
        combined_values = {sym: cf.combined_forecast for sym, cf in combined.items()}
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
            get_default_idm,
        )

        active_symbols = [s for s in combined_values if combined_values[s] > 0]
        instrument_weights = compute_handcrafted_weights(active_symbols, sector_map)
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
