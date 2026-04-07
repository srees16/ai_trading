"""
Unified Kaggle Runner — run any centurion_core task on Kaggle free tier.

Supports all backtest revisions, forecast extraction, and weight optimization.

Usage (in Kaggle notebook cell):
    !python centurion_core/cloud/run_kaggle.py --task r19c
    !python centurion_core/cloud/run_kaggle.py --task r20c
    !python centurion_core/cloud/run_kaggle.py --task r20d
    !python centurion_core/cloud/run_kaggle.py --task r21a
    !python centurion_core/cloud/run_kaggle.py --task extract      # forecast extraction
    !python centurion_core/cloud/run_kaggle.py --task optimize     # weight optimization
    !python centurion_core/cloud/run_kaggle.py --task pipeline     # extract + optimize + validate

Tasks:
    r19c      – Baseline R19c backtest (all modes OFF)
    r20a      – R20a: vol attenuation + sector caps + tight DD tiers
    r20b      – R20b: redesigned MaxDD guardrails
    r20c      – R20c: asymmetric vol boost (calm-only, never cuts)
    r20d      – R20d: R20c + position floor + tighter stops
    r21a      – R21a: optimized weights + regime-adaptive vol
    extract   – Extract per-source forecasts for weight optimizer
    optimize  – Run differential evolution weight optimizer
    pipeline  – Full 3-step: extract → optimize → validate (R21a)
"""
import sys
import os
import argparse
import time

# ── Path setup ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.dirname(_SCRIPT_DIR)  # centurion_core/
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

# Detect Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle/working")

VALID_TASKS = ["r19c", "r20a", "r20b", "r20c", "r20d", "r21a",
               "extract", "optimize", "pipeline"]


def _print_header(task: str):
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    mem_gb = "?"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_gb = f"{int(line.split()[1]) / 1024 / 1024:.1f}"
                    break
    except Exception:
        pass
    print("=" * 70)
    print(f"  Centurion Backtest — Task: {task.upper()}")
    print(f"  CPUs: {n_cpus}  |  RAM: {mem_gb} GB  |  Kaggle: {IS_KAGGLE}")
    print("=" * 70)


def _set_checkpoint(name: str):
    """Set checkpoint path for the task."""
    ckpt = os.path.join(_CORE_DIR, "data", f"backtest_checkpoint_{name}.pkl")
    if IS_KAGGLE:
        ckpt = f"/kaggle/working/backtest_checkpoint_{name}.pkl"
    os.environ["CENTURION_BT_CHECKPOINT"] = ckpt
    print(f"  Checkpoint: {ckpt}")
    return ckpt


def _set_all_modes_off(bt_mod):
    """Set all revision flags to OFF (pure R19c baseline)."""
    bt_mod._R19D_REGIME_MODE = False
    bt_mod._R19E_REGIME_MODE = False
    bt_mod._R19F_REGIME_MODE = False
    bt_mod._R19G_REGIME_MODE = False
    bt_mod._R19H_REGIME_MODE = False
    bt_mod._R20A_MAXDD_MODE = False
    bt_mod._R20B_MAXDD_MODE = False
    bt_mod._R20C_MAXDD_MODE = False
    bt_mod._R20D_HYBRID_MODE = False
    bt_mod._SAVE_FORECASTS_MODE = False
    bt_mod._R21A_REGIME_VOL = False


def _run_backtest(bt_mod, label: str):
    """Run the full backtest with retry and print results."""
    MAX_RETRIES = 3
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            result = bt_mod.run_full_backtest(
                tickers=None,
                capital=500_000,
                period="13y",
                market="IND",
                verbose=True,
                start_date="2012-01-01",
                end_date="2025-12-31",
            )
            elapsed = (time.time() - t0) / 60.0
            print(f"\n  {label} completed in {elapsed:.1f} minutes")
            break
        except Exception as e:
            print(f"\n  CRASH on attempt {attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}", flush=True)
            if attempt < MAX_RETRIES:
                print(f"  Retrying in 5s (checkpoint will auto-resume)...", flush=True)
                time.sleep(5)
            else:
                print(f"  All {MAX_RETRIES} attempts exhausted.", flush=True)
                raise

    if result is not None:
        print(f"\n{'='*70}")
        print(f"  {label} — KEY METRICS")
        print(f"{'='*70}")
        for k in ["annual_return_pct", "total_return_pct", "sharpe", "sortino",
                   "calmar", "max_drawdown_pct", "n_trades", "avg_positions",
                   "win_rate", "profit_factor"]:
            print(f"  {k:25s} = {result.get(k)}")

    return result


def _print_result_comparison(result, label: str):
    """Print comparison vs R19c baseline."""
    r19c_sharpe = 1.025
    r19c_maxdd = 67.41
    r19c_cagr = 48.28
    sharpe = result.get("sharpe", 0)
    maxdd = result.get("max_drawdown_pct", 100)
    cagr = result.get("annual_return_pct", 0)
    print(f"\n  ── R19c vs {label} ──")
    print(f"  Sharpe:  {r19c_sharpe:.3f} → {sharpe:.3f}  (Δ{sharpe - r19c_sharpe:+.3f})")
    print(f"  MaxDD:   {r19c_maxdd:.1f}% → {maxdd:.1f}%  (Δ{maxdd - r19c_maxdd:+.1f}%)")
    print(f"  CAGR:    {r19c_cagr:.1f}% → {cagr:.1f}%  (Δ{cagr - r19c_cagr:+.1f}%)")


# ───────────────────────────────────────────────────
#  Task Runners
# ───────────────────────────────────────────────────

def task_r19c():
    """Pure R19c baseline backtest."""
    import services.full_pipeline_backtest as bt_mod
    _set_all_modes_off(bt_mod)
    _set_checkpoint("r19c")
    _print_header("r19c")
    result = _run_backtest(bt_mod, "R19c")
    return result


def task_r20a():
    """R20a: vol attenuation + sector caps + tight DD tiers."""
    import services.full_pipeline_backtest as bt_mod
    _set_all_modes_off(bt_mod)
    bt_mod._R20A_MAXDD_MODE = True
    _set_checkpoint("r20a")
    _print_header("r20a")
    result = _run_backtest(bt_mod, "R20a")
    if result:
        _print_result_comparison(result, "R20a")
    return result


def task_r20b():
    """R20b: redesigned MaxDD guardrails."""
    import services.full_pipeline_backtest as bt_mod
    _set_all_modes_off(bt_mod)
    bt_mod._R20B_MAXDD_MODE = True
    _set_checkpoint("r20b")
    _print_header("r20b")
    result = _run_backtest(bt_mod, "R20b")
    if result:
        _print_result_comparison(result, "R20b")
    return result


def task_r20c():
    """R20c: asymmetric vol boost (calm-only, never cuts below R19c)."""
    import services.full_pipeline_backtest as bt_mod
    _set_all_modes_off(bt_mod)
    bt_mod._R20C_MAXDD_MODE = True
    _set_checkpoint("r20c")
    _print_header("r20c")
    result = _run_backtest(bt_mod, "R20c")
    if result:
        _print_result_comparison(result, "R20c")
    return result


def task_r20d():
    """R20d: R20c + position floor (min 6) + tighter stops (8σ)."""
    import services.full_pipeline_backtest as bt_mod
    _set_all_modes_off(bt_mod)
    bt_mod._R20D_HYBRID_MODE = True
    _set_checkpoint("r20d")
    _print_header("r20d")
    result = _run_backtest(bt_mod, "R20d")
    if result:
        _print_result_comparison(result, "R20d")
    return result


def task_r21a():
    """R21a: optimized signal weights + regime-adaptive vol."""
    import pickle
    import services.full_pipeline_backtest as bt_mod
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

    _set_all_modes_off(bt_mod)
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25
    bt_mod._R21A_REGIME_DEFEND = 0.55
    _set_checkpoint("r21a")
    _print_header("r21a")

    # Load optimized weights
    opt_path = os.path.join(_CORE_DIR, "data", "r21a_optimization_results.pkl")
    if IS_KAGGLE:
        # Check working dir first (from optimizer output), then data dir
        for p in ["/kaggle/working/r21a_optimization_results.pkl", opt_path]:
            if os.path.exists(p):
                opt_path = p
                break

    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            opt = pickle.load(f)
        weights = opt["best_weights"]
        print(f"  Loaded optimized weights from {opt_path}")
    else:
        print(f"  WARNING: No optimized weights found!")
        print(f"  Run: !python run_kaggle.py --task optimize")
        print(f"  Using R19c weights as fallback.")
        weights = {
            "ewmac_8_32": 0.07, "ewmac_16_64": 0.09, "ewmac_64_256": 0.08,
            "screener": 0.05, "momentum": 0.16, "mean_reversion": 0.13,
            "penfold_trend": 0.12, "ehlers_dsp": 0.12, "acceleration": 0.04,
            "carver_value": 0.07, "breakout": 0.07,
        }

    # Fill missing signals with 0
    all_24 = [
        "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
        "carry", "screener", "momentum", "pead", "mean_reversion",
        "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
        "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
        "intermarket", "acceleration", "carver_value", "skew_signal",
        "sentiment", "breakout", "order_flow",
    ]
    for sig in all_24:
        if sig not in weights:
            weights[sig] = 0.0

    for fw in DEFAULT_FORECAST_WEIGHTS:
        if fw.name in weights:
            fw.weight = weights[fw.name]

    print("  Weights:")
    for sig in sorted(weights, key=lambda s: weights[s], reverse=True):
        w = weights[sig]
        if w > 0.005:
            print(f"    {sig:20s}  {w*100:5.1f}%")

    result = _run_backtest(bt_mod, "R21a")
    if result:
        _print_result_comparison(result, "R21a")
        sharpe = result.get("sharpe", 0)
        maxdd = result.get("max_drawdown_pct", 100)
        cagr = result.get("annual_return_pct", 0)
        if sharpe >= 1.5 and maxdd <= 50.0 and cagr >= 50.0:
            print(f"\n  TARGET HIT!")
        elif sharpe > 1.025:
            print(f"\n  IMPROVEMENT: Sharpe Δ{sharpe - 1.025:+.3f}")
        else:
            print(f"\n  NO IMPROVEMENT vs R19c")
    return result


def task_extract():
    """Extract per-source forecasts for weight optimizer."""
    import pickle
    import services.full_pipeline_backtest as bt_mod
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

    _set_all_modes_off(bt_mod)
    bt_mod._SAVE_FORECASTS_MODE = True
    bt_mod._forecast_log.clear()
    _set_checkpoint("extract")
    _print_header("extract")

    # R19c weights
    _R19C = {
        "ewmac_8_32": 0.07, "ewmac_16_64": 0.09, "ewmac_64_256": 0.08,
        "screener": 0.05, "momentum": 0.16, "mean_reversion": 0.13,
        "penfold_trend": 0.12, "ehlers_dsp": 0.12, "acceleration": 0.04,
        "carver_value": 0.07, "breakout": 0.07,
    }
    for fw in DEFAULT_FORECAST_WEIGHTS:
        if fw.name in _R19C:
            fw.weight = _R19C[fw.name]

    result = _run_backtest(bt_mod, "Forecast Extraction")

    # Save extracted forecasts
    log = bt_mod._forecast_log
    print(f"\n  Extracted {len(log)} day-snapshots")

    out_path = os.path.join(_CORE_DIR, "data", "extracted_forecasts.pkl")
    if IS_KAGGLE:
        out_path = "/kaggle/working/extracted_forecasts.pkl"

    output = {"log": log, "r19c_result": {k: result.get(k) for k in [
        "sharpe", "sortino", "calmar", "max_drawdown_pct",
        "annual_return_pct", "total_return_pct", "n_trades",
    ]} if result else {}}
    with open(out_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved to {out_path}")
    return result


def task_optimize():
    """Run differential evolution weight optimizer."""
    _print_header("optimize")
    # Import and run the standalone optimizer
    from optimizer.optimize_weights_r21a import run_optimization
    run_optimization()


def task_pipeline():
    """Full 3-step pipeline: extract → optimize → validate."""
    _print_header("pipeline")

    print("\n  ═══ STEP 1/3: Forecast Extraction ═══\n")
    task_extract()

    print("\n  ═══ STEP 2/3: Weight Optimization ═══\n")
    task_optimize()

    print("\n  ═══ STEP 3/3: R21a Validation Backtest ═══\n")
    task_r21a()


# ───────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────

TASK_MAP = {
    "r19c": task_r19c,
    "r20a": task_r20a,
    "r20b": task_r20b,
    "r20c": task_r20c,
    "r20d": task_r20d,
    "r21a": task_r21a,
    "extract": task_extract,
    "optimize": task_optimize,
    "pipeline": task_pipeline,
}


def main():
    parser = argparse.ArgumentParser(description="Centurion Backtest — Unified Kaggle Runner")
    parser.add_argument("--task", required=True, choices=VALID_TASKS,
                        help="Which task to run")
    args = parser.parse_args()

    task_fn = TASK_MAP[args.task]
    t0 = time.time()
    task_fn()
    total = (time.time() - t0) / 60.0
    print(f"\n  Total time: {total:.1f} minutes")


if __name__ == "__main__":
    main()
