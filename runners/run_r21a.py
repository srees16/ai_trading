"""
R21a Runner — Optimized Signal Weights (Walk-Forward).

Runs full backtest with signal weights optimized by optimize_weights_r21a.py.
Base: R21A (sole active configuration).
Only change: signal weight allocation.

Usage:
    python run_r21a.py
"""
import sys
import os
import pickle

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

_R21A_CHECKPOINT = os.path.join(_root, "data", "backtest_checkpoint_r21a.pkl")
_OPT_RESULTS = os.path.join(_root, "data", "r21a_optimization_results.pkl")

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")


def main():
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

    # ── Load optimized weights ────
    if os.path.exists(_OPT_RESULTS):
        with open(_OPT_RESULTS, "rb") as f:
            opt = pickle.load(f)
        _R21A_WEIGHTS = opt["best_weights"]
        # Ensure all 24 signals are present (missing = 0)
        all_24 = [
            "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
            "carry", "screener", "momentum", "pead", "mean_reversion",
            "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
            "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
            "intermarket", "acceleration", "carver_value", "skew_signal",
            "sentiment", "breakout", "order_flow",
        ]
        for sig in all_24:
            if sig not in _R21A_WEIGHTS:
                _R21A_WEIGHTS[sig] = 0.0
        print(f"  Loaded optimized weights from {_OPT_RESULTS}")
    else:
        print(f"  WARNING: {_OPT_RESULTS} not found!")
        print(f"  Run optimize_weights_r21a.py first.")
        print(f"  Using DEFAULT_FORECAST_WEIGHTS (R21A-optimized) as fallback.")
        _R21A_WEIGHTS = {fw.name: fw.weight for fw in DEFAULT_FORECAST_WEIGHTS}

    for fw in DEFAULT_FORECAST_WEIGHTS:
        if fw.name in _R21A_WEIGHTS:
            fw.weight = _R21A_WEIGHTS[fw.name]

    # ── Set checkpoint path + mode flags ──
    import services.full_pipeline_backtest as bt_mod

    os.environ["CENTURION_BT_CHECKPOINT"] = _R21A_CHECKPOINT
    bt_mod._SAVE_FORECASTS_MODE = False
    # R21a: Enable regime-adaptive vol target
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25   # 25% more aggressive in uptrends
    bt_mod._R21A_REGIME_DEFEND = 0.15  # P1f: near-cash in downtrends (was 0.55)

    print("=" * 70)
    print("  R21a — Optimized Signal Weights (Walk-Forward)")
    print("  Base: R21A (sole active configuration)")
    print("  Change: signal weight allocation only")
    print(f"  Checkpoint: {_R21A_CHECKPOINT}")
    print("=" * 70)
    print("  Weights:")
    for sig in sorted(_R21A_WEIGHTS, key=lambda s: _R21A_WEIGHTS[s], reverse=True):
        w = _R21A_WEIGHTS[sig]
        if w > 0.005:
            print(f"    {sig:20s}  {w*100:5.1f}%")

    result = bt_mod.run_full_backtest(
        tickers=None,
        capital=500_000,
        period="13y",
        market="IND",
        verbose=True,
        start_date="2012-01-01",
        end_date="2025-12-31",
    )

    print(f"\n{'='*70}")
    print(f"  R21a KEY METRICS")
    print(f"{'='*70}")
    for k in ["annual_return_pct", "total_return_pct", "sharpe", "sortino",
              "calmar", "max_drawdown_pct", "n_trades", "avg_positions",
              "win_rate", "profit_factor",
              "detrended_sharpe", "trimmed_sharpe", "dm_bias_estimate"]:
        print(f"  {k:25s} = {result.get(k)}")
    if result.get("bootstrap_ci_sharpe"):
        lo, hi = result["bootstrap_ci_sharpe"]
        print(f"  {'sharpe_90pct_ci':25s} = [{lo:.3f}, {hi:.3f}]")

    # Comparison vs R21a OOS benchmark
    r21a_oos_sharpe = 2.093
    r21a_oos_maxdd = 25.2
    r21a_oos_cagr = 74.1
    r21a_sharpe = result.get("sharpe", 0)
    r21a_maxdd = result.get("max_drawdown_pct", 100)
    r21a_cagr = result.get("annual_return_pct", 0)
    print(f"\n  ── R21a OOS Benchmark vs This Run ──")
    print(f"  Sharpe:  {r21a_oos_sharpe:.3f} → {r21a_sharpe:.3f}  (Δ{r21a_sharpe - r21a_oos_sharpe:+.3f})")
    print(f"  MaxDD:   {r21a_oos_maxdd:.1f}% → {r21a_maxdd:.1f}%  (Δ{r21a_maxdd - r21a_oos_maxdd:+.1f}%)")
    print(f"  CAGR:    {r21a_oos_cagr:.1f}% → {r21a_cagr:.1f}%")

    if r21a_sharpe >= 1.5 and r21a_maxdd <= 50.0 and r21a_cagr >= 50.0:
        print(f"\n  TARGET HIT!")
    elif r21a_sharpe > r21a_oos_sharpe:
        print(f"\n  IMPROVEMENT: Sharpe Δ{r21a_sharpe - r21a_oos_sharpe:+.3f}")
    else:
        print(f"\n  BELOW OOS BENCHMARK")


if __name__ == "__main__":
    main()
