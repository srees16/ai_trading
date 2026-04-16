"""
Unified Kaggle Runner — run any centurion_core task on Kaggle free tier.

Supports R21A backtest, forecast extraction, weight optimization, and Harvest variants.

Usage (in Kaggle notebook cell):
    !python centurion_core/cloud/run_kaggle.py --task r21a
    !python centurion_core/cloud/run_kaggle.py --task extract      # forecast extraction
    !python centurion_core/cloud/run_kaggle.py --task optimize     # weight optimization
    !python centurion_core/cloud/run_kaggle.py --task pipeline     # extract + optimize + validate
    !python centurion_core/cloud/run_kaggle.py --task validate_hybrid  # H1 hybrid regime validation
    !python centurion_core/cloud/run_kaggle.py --task contra_all   # all 5 Harvest variants
    !python centurion_core/cloud/run_kaggle.py --task contra_v4    # capital rotation (inject+book)

Tasks:
    r21a      — R21A: optimized weights + regime-adaptive vol (benchmark)
    extract   — Extract per-source forecasts for weight optimizer
    optimize  — Run differential evolution weight optimizer
    pipeline  — Full 3-step: extract → optimize → validate (R21a)
    contra_v0 – Contra V0: R21a baseline (control)
    contra_v1 – Contra V1: Bear dip-buyer (MR vol boost in downtrend)
    contra_v2 – Contra V2: Bull profit-taker (tighter stops in uptrend)
    contra_v3 – Contra V3: Combined V1+V2
    contra_v4 – Contra V4: Capital rotation (inject at bear→bull, book in bull)
    contra_all– Run all 5 variants sequentially + comparison table
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

VALID_TASKS = ["r21a", "r22",
               "extract", "optimize", "optimize_v28", "pipeline", "pipeline_v28",
               "validate_hybrid",
               "contra_v0", "contra_v1", "contra_v2", "contra_v3", "contra_v4", "contra_all"]


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
    """Reset backtest module to clean slate (used before enabling specific config)."""
    bt_mod._SAVE_FORECASTS_MODE = False
    bt_mod._R21A_REGIME_VOL = False
    bt_mod._R22_BULL_INFUSION = False


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
                annual_vol_target=0.20,  # R24 FIX: explicit (matches R21A calibration)
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
    """Print comparison vs R21A OOS benchmark."""
    r21a = {"sharpe": 2.093, "sortino": 3.200, "calmar": 2.937,
            "max_drawdown_pct": 25.2, "annual_return_pct": 74.1}
    metrics = [
        ("Sharpe",  "sharpe",            0,   ".3f"),
        ("Sortino", "sortino",           0,   ".3f"),
        ("Calmar",  "calmar",            0,   ".3f"),
        ("CAGR",    "annual_return_pct",  0,   ".1f"),
        ("MaxDD",   "max_drawdown_pct", 100,  ".1f"),
    ]
    print(f"\n  ── R21A OOS vs {label} ──")
    for name, key, default, fmt in metrics:
        base = r21a[key]
        curr = result.get(key, default)
        suffix = "%" if key in ("annual_return_pct", "max_drawdown_pct") else ""
        print(f"  {name:8s} {base:{fmt}}{suffix} → {curr:{fmt}}{suffix}  (Δ{curr - base:+{fmt}}{suffix})")


# ───────────────────────────────────────────────────
#  Task Runners
# ───────────────────────────────────────────────────

def task_r21a():
    """R21a: optimized signal weights + regime-adaptive vol."""
    import pickle
    import services.full_pipeline_backtest as bt_mod
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

    _set_all_modes_off(bt_mod)
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25
    bt_mod._R21A_REGIME_DEFEND = 0.55  # R21A original (v25) — was 0.15 (P1f over-correction)

    # R24 FIX: Enable harvest features (were built but disabled)
    bt_mod._HARVEST_DIP_BUYER = True     # MR vol boost in bear (Sharpe +0.03)
    bt_mod._HARVEST_PROFIT_TAKER = True  # Tighter stops in bull (Sharpe +0.02)

    _set_checkpoint("r21a")
    _print_header("r21a")

    # Load optimized weights — fall back to DEFAULT_FORECAST_WEIGHTS (R21A)
    opt_path = os.path.join(_CORE_DIR, "data", "r21a_optimization_results.pkl")
    if IS_KAGGLE:
        for p in ["/kaggle/working/r21a_optimization_results.pkl", opt_path]:
            if os.path.exists(p):
                opt_path = p
                break

    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            opt = pickle.load(f)
        weights = opt["best_weights"]
        print(f"  Loaded optimized weights from {opt_path}")
        for fw in DEFAULT_FORECAST_WEIGHTS:
            if fw.name in weights:
                fw.weight = weights[fw.name]
            elif fw.weight > 0:
                fw.weight = 0.0
    else:
        print(f"  WARNING: No optimized weights found!")
        print(f"  Using DEFAULT_FORECAST_WEIGHTS (R21A-optimized)")
        weights = {fw.name: fw.weight for fw in DEFAULT_FORECAST_WEIGHTS}

    print("  Weights:")
    for fw in sorted(DEFAULT_FORECAST_WEIGHTS, key=lambda f: f.weight, reverse=True):
        if fw.weight > 0.005:
            print(f"    {fw.name:20s}  {fw.weight*100:5.1f}%")

    result = _run_backtest(bt_mod, "R21a")
    if result:
        _print_result_comparison(result, "R21a")
    return result


def task_r22():
    """R22: Centurion Compounder + Bull-Run Capital Infusion."""
    import services.full_pipeline_backtest as bt_mod
    _setup_r21a_base(bt_mod)

    # R22-specific: enable bull-run infusion
    bt_mod._R22_BULL_INFUSION = True
    bt_mod._R22_INFUSION_AMOUNT = 50_000.0
    bt_mod._R22_INFUSION_COOLDOWN_DAYS = 200
    bt_mod._R22_BULL_CONFIRM_DAYS = 5

    # Disable Harvest (this is Compounder-only)
    bt_mod._HARVEST_ENABLED = False
    bt_mod._HARVEST_DIP_BUYER = False
    bt_mod._HARVEST_PROFIT_TAKER = False

    _set_checkpoint("r22")
    _print_header("r22")
    print(f"  R22 Bull-Run Infusion: +\u20b9{bt_mod._R22_INFUSION_AMOUNT:,.0f} per event")
    print(f"  Cooldown: {bt_mod._R22_INFUSION_COOLDOWN_DAYS}d | Bull confirm: {bt_mod._R22_BULL_CONFIRM_DAYS}d")

    result = _run_backtest(bt_mod, "R22 (Bull-Run Capital Infusion)")
    if result:
        _print_result_comparison(result, "R22")
        r22_data = result.get("r22_bull_infusion")
        if r22_data:
            print(f"\n  R22 Infusion Summary:")
            print(f"  Bull Alerts:     {r22_data['n_alerts']}")
            print(f"  Infusions Made:  {r22_data['n_infusions']}")
            print(f"  Total Infused:   \u20b9{r22_data['total_infused']:,.0f}")
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

    # Use DEFAULT_FORECAST_WEIGHTS as-is (R21A-optimized) for extraction base
    # The optimizer needs the current best weights to extract forecasts from

    result = _run_backtest(bt_mod, "Forecast Extraction")

    # Save extracted forecasts
    log = bt_mod._forecast_log
    print(f"\n  Extracted {len(log)} day-snapshots")

    out_path = os.path.join(_CORE_DIR, "data", "extracted_forecasts.pkl")
    if IS_KAGGLE:
        out_path = "/kaggle/working/extracted_forecasts.pkl"

    output = {"log": log, "extraction_result": {k: result.get(k) for k in [
        "sharpe", "sortino", "calmar", "max_drawdown_pct",
        "annual_return_pct", "total_return_pct", "n_trades",
    ]} if result else {}}
    with open(out_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved to {out_path}")
    return result


def task_optimize():
    """Run differential evolution weight optimizer (v27: 10 signals)."""
    _print_header("optimize")
    from optimizer.optimize_weights_r21a import run_optimization
    run_optimization()


def task_optimize_v28():
    """Run v28 optimizer (12 signals: +carry, +skew_signal, stricter guards)."""
    _print_header("optimize_v28")
    from optimizer.optimize_weights_v28 import run_optimization
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


def task_pipeline_v28():
    """v28 pipeline: extract → optimize_v28 (12 signals + anti-overfit)."""
    _print_header("pipeline_v28")

    print("\n  ═══ STEP 1/2: Forecast Extraction ═══\n")
    task_extract()

    print("\n  ═══ STEP 2/2: v28 Weight Optimization (12 signals) ═══\n")
    task_optimize_v28()


def task_validate_hybrid():
    """Validate hybrid HMM×SMA200 regime scaling against R21a baseline."""
    _print_header("validate_hybrid")
    from optimizer.validate_r21a_hybrid import main as validate_main
    validate_main()


# ───────────────────────────────────────────────────
#  Contra-Regime Strategy Backtest Variants
# ───────────────────────────────────────────────────
# V0: R21a baseline (contra flags OFF) — control
# V1: Bear dip-buyer — MR signals get 3.33× vol boost in downtrend
# V2: Bull profit-taker — tighter trailing stops (6σ) in uptrend
# V3: Combined V1+V2
# R21a benchmark: Sharpe=2.093, CAGR=74.1%, MaxDD=25.2%, Calmar=2.937

def _setup_r21a_base(bt_mod):
    """Set R21a base configuration (shared by all Harvest variants)."""
    import pickle
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

    _set_all_modes_off(bt_mod)
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25
    bt_mod._R21A_REGIME_DEFEND = 0.55  # R21A original (v25) — was 0.15 (P1f over-correction)

    # Load optimized weights — fall back to DEFAULT_FORECAST_WEIGHTS (R21A)
    opt_path = os.path.join(_CORE_DIR, "data", "r21a_optimization_results.pkl")
    if IS_KAGGLE:
        for p in ["/kaggle/working/r21a_optimization_results.pkl", opt_path]:
            if os.path.exists(p):
                opt_path = p
                break

    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            opt = pickle.load(f)
        weights = opt["best_weights"]
        print(f"  Loaded optimized weights from {opt_path}")
        for fw in DEFAULT_FORECAST_WEIGHTS:
            if fw.name in weights:
                fw.weight = weights[fw.name]
            elif fw.weight > 0:
                fw.weight = 0.0
    else:
        print(f"  WARNING: No optimized weights — using DEFAULT_FORECAST_WEIGHTS (R21A)")
        # DEFAULT_FORECAST_WEIGHTS already has R21A-optimized values


def _set_contra_flags(bt_mod, dip_buyer: bool, profit_taker: bool, capital_rotation: bool = False):
    """Set Centurion Harvest flags on the backtest module."""
    bt_mod._HARVEST_DIP_BUYER = dip_buyer
    bt_mod._HARVEST_PROFIT_TAKER = profit_taker
    bt_mod._HARVEST_ENABLED = capital_rotation
    flags = []
    if dip_buyer:
        flags.append(f"BEAR_DIP(MR×{bt_mod._HARVEST_MR_BEAR_VOL_MULT:.1f})")
    if profit_taker:
        flags.append(f"BULL_PROFIT(stop={bt_mod._HARVEST_BULL_STOP_SIGMA:.0f}σ)")
    if capital_rotation:
        flags.append(f"HARVEST(inject={bt_mod._HARVEST_INJECT_PCT:.0%},book={bt_mod._HARVEST_BOOK_PCT:.0%})")
    print(f"  Harvest flags: {', '.join(flags) if flags else 'OFF (Compounder baseline)'}")


def _print_contra_comparison(results: dict):
    """Print comparison table for all contra-regime variants."""
    r21a_sharpe = 2.093
    r21a_cagr = 74.1
    r21a_maxdd = 25.2
    r21a_calmar = 2.937

    print(f"\n{'='*78}")
    print(f"  CENTURION HARVEST vs COMPOUNDER — COMPARISON")
    print(f"{'='*78}")
    print(f"  {'Variant':<12} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'Calmar':>8} {'Trades':>8} {'WinRate':>8} {'Injected':>10} {'Booked':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    print(f"  {'R21a(ref)':<12} {r21a_sharpe:>8.3f} {r21a_cagr:>8.1f} {r21a_maxdd:>8.1f} {r21a_calmar:>8.3f} {'—':>8} {'—':>8} {'—':>10} {'—':>10}")

    for name, r in results.items():
        if r is None:
            print(f"  {name:<12} {'FAILED':>8}")
            continue
        s = r.get('sharpe', 0)
        c = r.get('annual_return_pct', 0)
        m = r.get('max_drawdown_pct', 100)
        cal = r.get('calmar', 0)
        t = r.get('n_trades', 0)
        w = r.get('win_rate', 0)
        cr = r.get('capital_rotation')
        inj = f"₹{cr['total_injected']:,.0f}" if cr else '—'
        bkd = f"₹{cr['total_booked']:,.0f}" if cr else '—'
        # Accept/reject
        ok = s >= 2.0 and m <= 30.0 and cal >= 2.5
        tag = ' ✓' if ok else ' ✗'
        print(f"  {name:<12} {s:>8.3f} {c:>8.1f} {m:>8.1f} {cal:>8.3f} {t:>8d} {w:>7.1f}%{tag} {inj:>10} {bkd:>10}")

    print(f"\n  Accept criteria: Sharpe ≥ 2.0 AND MaxDD ≤ 30% AND Calmar ≥ 2.5")
    print(f"{'='*78}")


def task_contra_v0():
    """Contra V0: R21a baseline (control — contra flags OFF)."""
    import services.full_pipeline_backtest as bt_mod
    _setup_r21a_base(bt_mod)
    _set_contra_flags(bt_mod, dip_buyer=False, profit_taker=False)
    _set_checkpoint("contra_v0")
    _print_header("contra_v0")
    return _run_backtest(bt_mod, "Contra V0 (R21a baseline)")


def task_contra_v1():
    """Contra V1: Bear dip-buyer — MR gets 3.33× vol boost in downtrend."""
    import services.full_pipeline_backtest as bt_mod
    _setup_r21a_base(bt_mod)
    _set_contra_flags(bt_mod, dip_buyer=True, profit_taker=False)
    _set_checkpoint("contra_v1")
    _print_header("contra_v1")
    return _run_backtest(bt_mod, "Contra V1 (Bear Dip-Buyer)")


def task_contra_v2():
    """Contra V2: Bull profit-taker — tighter stops (6σ) in uptrend."""
    import services.full_pipeline_backtest as bt_mod
    _setup_r21a_base(bt_mod)
    _set_contra_flags(bt_mod, dip_buyer=False, profit_taker=True)
    _set_checkpoint("contra_v2")
    _print_header("contra_v2")
    return _run_backtest(bt_mod, "Contra V2 (Bull Profit-Taker)")


def task_contra_v3():
    """Contra V3: Combined dip-buyer + profit-taker."""
    import services.full_pipeline_backtest as bt_mod
    _setup_r21a_base(bt_mod)
    _set_contra_flags(bt_mod, dip_buyer=True, profit_taker=True)
    _set_checkpoint("contra_v3")
    _print_header("contra_v3")
    return _run_backtest(bt_mod, "Contra V3 (Combined)")


def task_contra_v4():
    """Centurion Harvest: Capital rotation — inject at bear→bull + book profits in bull."""
    import services.full_pipeline_backtest as bt_mod
    _setup_r21a_base(bt_mod)
    _set_contra_flags(bt_mod, dip_buyer=True, profit_taker=True, capital_rotation=True)
    _set_checkpoint("contra_v4")
    _print_header("contra_v4")
    print(f"  Capital inject: {bt_mod._HARVEST_INJECT_PCT:.0%} of base at bear→bull crossover")
    print(f"  Profit booking: {bt_mod._HARVEST_BOOK_PCT:.0%} of gains after {bt_mod._HARVEST_BULL_SUSTAIN_DAYS}d sustained bull")
    print(f"  Injection cooldown: {bt_mod._HARVEST_INJECT_COOLDOWN_DAYS}d | Min gain to book: {bt_mod._HARVEST_MIN_GAIN_TO_BOOK:.0%}")
    return _run_backtest(bt_mod, "Centurion Harvest (Capital Rotation)")


def task_contra_all():
    """Run all 5 contra-regime variants and print comparison table."""
    _print_header("contra_all")
    print("\n  Running 5 contra-regime variants sequentially...")
    print("  Paper trading is NOT affected (backtest uses historical data only).\n")

    results = {}

    print("\n" + "═" * 70)
    print("  VARIANT 1/5: V0 — R21a Baseline (control)")
    print("═" * 70)
    results["V0-base"] = task_contra_v0()

    print("\n" + "═" * 70)
    print("  VARIANT 2/5: V1 — Bear Dip-Buyer")
    print("═" * 70)
    results["V1-dip"] = task_contra_v1()

    print("\n" + "═" * 70)
    print("  VARIANT 3/5: V2 — Bull Profit-Taker")
    print("═" * 70)
    results["V2-profit"] = task_contra_v2()

    print("\n" + "═" * 70)
    print("  VARIANT 4/5: V3 — Combined (V1+V2)")
    print("═" * 70)
    results["V3-combo"] = task_contra_v3()

    print("\n" + "═" * 70)
    print("  VARIANT 5/5: V4 — Capital Rotation (inject+book)")
    print("═" * 70)
    results["V4-rotate"] = task_contra_v4()

    _print_contra_comparison(results)

    # Save results for comparison
    import pickle
    out_path = os.path.join(_CORE_DIR, "data", "contra_regime_results.pkl")
    if IS_KAGGLE:
        out_path = "/kaggle/working/contra_regime_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Results saved to {out_path}")

    return results


# ───────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────

TASK_MAP = {
    "r21a": task_r21a,
    "r22": task_r22,
    "extract": task_extract,
    "optimize": task_optimize,
    "optimize_v28": task_optimize_v28,
    "pipeline": task_pipeline,
    "pipeline_v28": task_pipeline_v28,
    "validate_hybrid": task_validate_hybrid,
    "contra_v0": task_contra_v0,
    "contra_v1": task_contra_v1,
    "contra_v2": task_contra_v2,
    "contra_v3": task_contra_v3,
    "contra_v4": task_contra_v4,
    "contra_all": task_contra_all,
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
