"""
R21a Full Pipeline — Runs all 3 steps sequentially (overnight).

  Step 1: Extract per-source forecasts (~5 hours)
  Step 2: Optimize signal weights (~5 minutes)
  Step 3: Full validation backtest with optimal weights (~5 hours)

Usage:
    python run_r21a_pipeline.py
"""
import sys
import os
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")


def step1_extract():
    """Step 1: Extract per-source forecasts."""
    import pickle
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS
    import services.full_pipeline_backtest as bt_mod

    _CHECKPOINT = os.path.join(_root, "data", "backtest_checkpoint_extract.pkl")
    _OUTPUT = os.path.join(_root, "data", "extracted_forecasts.pkl")

    _R19C_WEIGHTS = {
        "ewmac_8_32": 0.07, "ewmac_16_64": 0.09, "ewmac_32_128": 0.00,
        "ewmac_64_256": 0.08, "carry": 0.00, "screener": 0.05,
        "momentum": 0.16, "pead": 0.00, "mean_reversion": 0.13,
        "fii_flow": 0.00, "decision_engine": 0.00, "oi_signal": 0.00,
        "cross_momentum": 0.00, "pairs_arb": 0.00, "event_driven": 0.00,
        "penfold_trend": 0.12, "ehlers_dsp": 0.12, "intermarket": 0.00,
        "acceleration": 0.04, "carver_value": 0.07, "skew_signal": 0.00,
        "sentiment": 0.00, "breakout": 0.07, "order_flow": 0.00,
    }
    for fw in DEFAULT_FORECAST_WEIGHTS:
        if fw.name in _R19C_WEIGHTS:
            fw.weight = _R19C_WEIGHTS[fw.name]

    os.environ["CENTURION_BT_CHECKPOINT"] = _CHECKPOINT
    bt_mod._SAVE_FORECASTS_MODE = True
    bt_mod._R20A_MAXDD_MODE = False
    bt_mod._R20B_MAXDD_MODE = False
    bt_mod._R20C_MAXDD_MODE = False
    bt_mod._R20D_HYBRID_MODE = False
    bt_mod._R19D_REGIME_MODE = False
    bt_mod._R19E_REGIME_MODE = False
    bt_mod._R19F_REGIME_MODE = False
    bt_mod._R19G_REGIME_MODE = False
    bt_mod._R19H_REGIME_MODE = False
    bt_mod._forecast_log.clear()

    print("=" * 70)
    print("  STEP 1/3 — Forecast Extraction (R19c base)")
    print(f"  Output: {_OUTPUT}")
    print("=" * 70)

    result = bt_mod.run_full_backtest(
        tickers=None, capital=500_000, period="13y", market="IND",
        verbose=True, start_date="2012-01-01", end_date="2025-12-31",
    )

    log = bt_mod._forecast_log
    print(f"\n  Extracted {len(log)} day-snapshots")

    payload = {
        "log": log,
        "r19c_result": {
            k: result.get(k)
            for k in ["sharpe", "sortino", "calmar", "max_drawdown_pct",
                       "annual_return_pct", "total_return_pct", "n_trades"]
        },
    }
    with open(_OUTPUT, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    sz_mb = os.path.getsize(_OUTPUT) / 1e6
    print(f"  Saved to {_OUTPUT} ({sz_mb:.1f} MB)")
    print(f"  R19c baseline: Sharpe={result.get('sharpe'):.3f}  CAGR={result.get('annual_return_pct'):.1f}%")
    return True


def step2_optimize():
    """Step 2: Optimize signal weights."""
    print("\n" + "=" * 70)
    print("  STEP 2/3 — Weight Optimization")
    print("=" * 70)

    # Import and run optimizer
    from optimizer.optimize_weights_r21a import run_optimization
    run_optimization()
    return True


def step3_validate():
    """Step 3: Full validation backtest with optimal weights."""
    import pickle
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS
    import importlib
    import services.full_pipeline_backtest as bt_mod

    # Force reimport to reset state from step 1
    importlib.reload(bt_mod)

    _OPT_RESULTS = os.path.join(_root, "data", "r21a_optimization_results.pkl")
    _CHECKPOINT = os.path.join(_root, "data", "backtest_checkpoint_r21a.pkl")

    if not os.path.exists(_OPT_RESULTS):
        print(f"\n  ERROR: {_OPT_RESULTS} not found. Step 2 may have failed.")
        return False

    with open(_OPT_RESULTS, "rb") as f:
        opt = pickle.load(f)
    best_weights = opt["best_weights"]

    # Ensure all 24 signals present
    all_24 = [
        "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
        "carry", "screener", "momentum", "pead", "mean_reversion",
        "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
        "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
        "intermarket", "acceleration", "carver_value", "skew_signal",
        "sentiment", "breakout", "order_flow",
    ]
    for sig in all_24:
        if sig not in best_weights:
            best_weights[sig] = 0.0

    # Reload DEFAULT_FORECAST_WEIGHTS fresh
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS as DFW
    for fw in DFW:
        if fw.name in best_weights:
            fw.weight = best_weights[fw.name]

    os.environ["CENTURION_BT_CHECKPOINT"] = _CHECKPOINT
    bt_mod._SAVE_FORECASTS_MODE = False
    bt_mod._R20A_MAXDD_MODE = False
    bt_mod._R20B_MAXDD_MODE = False
    bt_mod._R20C_MAXDD_MODE = False
    bt_mod._R20D_HYBRID_MODE = False
    bt_mod._R19D_REGIME_MODE = False
    bt_mod._R19E_REGIME_MODE = False
    bt_mod._R19F_REGIME_MODE = False
    bt_mod._R19G_REGIME_MODE = False
    bt_mod._R19H_REGIME_MODE = False

    print("\n" + "=" * 70)
    print("  STEP 3/3 — Full Validation Backtest (R21a optimal weights)")
    print("=" * 70)
    print("  Weights:")
    for sig in sorted(best_weights, key=lambda s: best_weights[s], reverse=True):
        w = best_weights[sig]
        if w > 0.005:
            print(f"    {sig:20s}  {w*100:5.1f}%")

    result = bt_mod.run_full_backtest(
        tickers=None, capital=500_000, period="13y", market="IND",
        verbose=True, start_date="2012-01-01", end_date="2025-12-31",
    )

    # Final report
    print(f"\n{'='*70}")
    print(f"  R21a FINAL RESULTS")
    print(f"{'='*70}")
    for k in ["annual_return_pct", "total_return_pct", "sharpe", "sortino",
              "calmar", "max_drawdown_pct", "n_trades", "avg_positions",
              "win_rate", "profit_factor",
              "detrended_sharpe", "trimmed_sharpe", "dm_bias_estimate"]:
        print(f"  {k:25s} = {result.get(k)}")
    if result.get("bootstrap_ci_sharpe"):
        lo, hi = result["bootstrap_ci_sharpe"]
        print(f"  {'sharpe_90pct_ci':25s} = [{lo:.3f}, {hi:.3f}]")

    r19c_sharpe = 1.025
    r19c_maxdd = 67.41
    r21a_sharpe = result.get("sharpe", 0)
    r21a_maxdd = result.get("max_drawdown_pct", 100)
    r21a_cagr = result.get("annual_return_pct", 0)
    print(f"\n  ── R19c vs R21a ──")
    print(f"  Sharpe:  {r19c_sharpe:.3f} → {r21a_sharpe:.3f}  (Δ{r21a_sharpe - r19c_sharpe:+.3f})")
    print(f"  MaxDD:   {r19c_maxdd:.1f}% → {r21a_maxdd:.1f}%  (Δ{r21a_maxdd - r19c_maxdd:+.1f}%)")
    print(f"  CAGR:    48.78% → {r21a_cagr:.1f}%")

    if r21a_sharpe >= 1.5 and r21a_maxdd <= 50.0 and r21a_cagr >= 50.0:
        print(f"\n  ★ TARGET HIT! ★")
    elif r21a_sharpe > r19c_sharpe:
        print(f"\n  ✓ IMPROVEMENT: Sharpe Δ{r21a_sharpe - r19c_sharpe:+.3f}")
    else:
        print(f"\n  ✗ NO IMPROVEMENT vs R19c")

    return True


def main():
    t0 = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║  R21a OVERNIGHT PIPELINE                                         ║")
    print("║  Step 1: Extract forecasts (~5 hrs)                              ║")
    print("║  Step 2: Optimize weights  (~5 min)                              ║")
    print("║  Step 3: Validate backtest (~5 hrs)                              ║")
    print("╚" + "═" * 68 + "╝")

    # Step 1
    t1 = time.time()
    ok = step1_extract()
    elapsed1 = (time.time() - t1) / 3600
    print(f"\n  Step 1 complete in {elapsed1:.1f} hours")
    if not ok:
        print("  ABORT: Step 1 failed")
        return

    # Step 2
    t2 = time.time()
    ok = step2_optimize()
    elapsed2 = (time.time() - t2) / 60
    print(f"\n  Step 2 complete in {elapsed2:.1f} minutes")
    if not ok:
        print("  ABORT: Step 2 failed")
        return

    # Step 3
    t3 = time.time()
    ok = step3_validate()
    elapsed3 = (time.time() - t3) / 3600
    print(f"\n  Step 3 complete in {elapsed3:.1f} hours")

    total = (time.time() - t0) / 3600
    print(f"\n{'='*70}")
    print(f"  R21a PIPELINE COMPLETE — Total: {total:.1f} hours")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
