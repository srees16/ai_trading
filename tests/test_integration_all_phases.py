"""Comprehensive integration test for all Phases 0-7."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

def main():
    print("=== FULL INTEGRATION TEST ===")

    # Phase 0
    from services.risk_metrics import RiskMetrics
    rm = RiskMetrics.compute_all(pd.Series(np.random.normal(0.001, 0.015, 200)))
    print(f"P0 RiskMetrics: Sharpe={rm.sharpe_ratio:.2f} OK")

    from services.volatility_target import VolatilityTarget, VolatilityTargetConfig
    vt = VolatilityTarget(VolatilityTargetConfig(initial_capital=500000, annual_vol_target_pct=25.0))
    vt.update_pnl(realized=-60000)
    print(f"P0 VolTarget: DD={vt.drawdown_pct:.1f}%, scale={vt.risk_scale_factor:.2f} OK")

    from services.forecast_scalar import calibrate_scalar_from_data
    s = calibrate_scalar_from_data(pd.Series(np.random.normal(0, 2.5, 500)))
    print(f"P0 ForecastScalar: {s:.2f} OK")

    # Phase 1
    from services.momentum_factor import MomentumFactor
    ohlcv = {}
    for sym in list("ABCDEFGHIJ"):
        p = 100 * np.cumprod(1 + np.random.normal(np.random.uniform(-0.001, 0.003), 0.02, 300))
        ohlcv[sym] = pd.DataFrame({"Close": p, "High": p * 1.01, "Low": p * 0.99, "Open": p, "Volume": 1e6})
    fc = MomentumFactor().get_forecasts(ohlcv)
    print(f"P1 Momentum: {len(fc)} forecasts OK")

    from services.pead_strategy import PEADStrategy, EarningsSurprise
    pead = PEADStrategy()
    pead.process_earnings([
        EarningsSurprise(ticker="INFY", announcement_date="2025-01-15",
                        eps_actual=25, eps_consensus=20, sue=2.5,
                        surprise_pct=0.25, direction="POSITIVE"),
    ])
    print(f"P1 PEAD: {len(pead.get_current_forecasts())} forecasts OK")

    from services.factor_momentum import FactorMomentum
    dw = FactorMomentum().compute_dynamic_weights({
        k: pd.Series(np.random.normal(0.001, 0.01, 63))
        for k in ["ewmac_16_64", "carry", "screener"]
    })
    print(f"P1 FactorMom: {len(dw.weights)} weights OK")

    # Phase 2
    from kite_connect.options.covered_call_strategy import CoveredCallStrategy
    cc = CoveredCallStrategy()
    print(f"P2 CoveredCall: delta={cc.delta_target} OK")

    from kite_connect.options.put_selling_strategy import PutSellingStrategy
    ps = PutSellingStrategy()
    print(f"P2 PutSelling: max={ps.max_concurrent} OK")

    # Phase 3
    from services.monte_carlo_risk import TradeBootstrapMonteCarlo
    sim = TradeBootstrapMonteCarlo(n_simulations=1000, n_trades_per_sim=200).simulate(
        list(np.random.normal(0.005, 0.03, 50))
    )
    print(f"P3 MonteCarlo: median={sim.median_cagr_pct:.1f}% OK")

    from services.portfolio_correlation import PortfolioCorrelationRisk
    ret_dict = {s: pd.Series(np.random.normal(0.001, 0.02, 100)) for s in ["A", "B", "C"]}
    returns_df = pd.DataFrame(ret_dict)
    pos_weights = {"A": 0.4, "B": 0.35, "C": 0.25}
    a = PortfolioCorrelationRisk().assess(
        position_weights=pos_weights,
        returns_data=returns_df,
    )
    print(f"P3 PortCorr: div={a.diversification_ratio:.2f} OK")

    from services.tail_risk_hedge import TailRiskHedge
    h = TailRiskHedge().assess(
        portfolio_value=1e6, vix=28, vix_3d_ago=18, drawdown_pct=12
    )
    print(f"P3 TailHedge: {h.hedge_urgency} OK")

    # Phase 4
    from services.regime_performance import RegimePerformance
    idx = pd.date_range("2024-01-01", periods=500, freq="B")
    r = RegimePerformance().stratify_returns(
        "test_strategy",
        pd.Series(np.random.normal(0.001, 0.015, 500), index=idx),
        pd.Series(np.random.choice(["BULL", "BEAR", "RANGE", "CRISIS"], 500, p=[0.4, 0.2, 0.3, 0.1]), index=idx),
    )
    print(f"P4 RegimePerf: {len(r.regime_stats)} regimes, weakest={r.weakest_regime} OK")

    from services.strategy_decay import StrategyDecayMonitor
    sd = StrategyDecayMonitor()
    strat_rets = {
        "ewmac": pd.Series(np.random.normal(0.002, 0.01, 63)),
        "carry": pd.Series(np.random.normal(-0.001, 0.01, 63)),
    }
    hist_sharpes = {"ewmac": 2.0, "carry": 1.5}
    report = sd.check_all(strat_rets, hist_sharpes)
    mults = sd.get_allocation_multipliers(strat_rets, hist_sharpes)
    print(f"P4 StratDecay: {mults} OK")

    # Phase 5-7
    from services.regime_strategy_mix import get_regime_weights
    w = get_regime_weights("TRENDING_BULL", blend_with_static=0.3)
    print(f"P5 RegimeMix: {len(w)} sources OK")

    from services.strategy_tournament import StrategyTournament
    tourney = StrategyTournament(top_n=3)
    all_rets = {
        n: pd.Series(np.random.normal(d, 0.012, 100))
        for n, d in [("ewmac", 0.002), ("carry", 0.001), ("screener", 0.0015),
                     ("momentum", 0.003), ("pead", -0.001)]
    }
    tr = tourney.run_tournament(all_rets)
    print(f"P7 Tournament: top={tr.top_strategies} OK")

    # Wiring checks
    pipe_src = open(os.path.join(os.path.dirname(__file__), "..", "services", "carver_pipeline.py"), encoding="utf-8").read()
    for mod in ["regime_strategy_mix", "strategy_decay", "factor_momentum"]:
        assert mod in pipe_src, f"{mod} NOT wired in pipeline!"

    sched_src = open(os.path.join(os.path.dirname(__file__), "..", "scheduler.py"), encoding="utf-8").read()
    assert "strategy_tournament" in sched_src, "tournament NOT in scheduler!"

    print("Wiring: pipeline + scheduler OK")
    print()
    print("=== ALL PHASES 0-7 PASS ===")


if __name__ == "__main__":
    main()
