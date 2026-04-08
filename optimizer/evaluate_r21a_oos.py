"""
R21a — Extract best weights from checkpoint & run out-of-sample evaluation.

Usage:
    python -m optimizer.evaluate_r21a_oos
"""
import sys
import os
import pickle
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from optimizer.optimize_weights_r21a import (
    ACTIVE_SIGNALS,
    R19C_WEIGHTS,
    _build_corr_matrix,
    _load_data,
    _prepare_matrices,
    _simulate_equity,
)

_CKPT_PATH = os.path.join(_root, "data", "r21a_optimizer_checkpoint.pkl")


def main():
    # ── 1. Load checkpoint ──────────────────────────────────────
    if not os.path.exists(_CKPT_PATH):
        print(f"ERROR: checkpoint not found at {_CKPT_PATH}")
        sys.exit(1)

    with open(_CKPT_PATH, "rb") as f:
        ckpt = pickle.load(f)

    gen = ckpt["generation"]
    best_weights = ckpt["best_weights"]
    best_score = -ckpt["best_fun"]
    elapsed = ckpt.get("elapsed_min", 0.0)
    train_result = ckpt.get("best_train_result", {})

    print("=" * 70)
    print("  R21a — Checkpoint Weight Extraction & OOS Evaluation")
    print("=" * 70)
    print(f"\n  Checkpoint: generation {gen}/150, score={best_score:.4f}, elapsed={elapsed:.1f}min")
    print(f"\n  Train metrics (from checkpoint):")
    print(f"    Sharpe={train_result.get('sharpe', '?')}  "
          f"CAGR={train_result.get('cagr', '?')}%  "
          f"MaxDD={train_result.get('max_dd', '?')}%  "
          f"Calmar={train_result.get('calmar', '?')}")

    # ── 2. Print extracted weights ──────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  OPTIMIZED WEIGHTS (R21a, gen {gen})")
    print(f"  {'─'*60}")
    for sig in sorted(best_weights, key=lambda s: best_weights[s], reverse=True):
        w = best_weights[sig]
        r19c_w = R19C_WEIGHTS.get(sig, 0.0)
        delta = w - r19c_w
        print(f"    {sig:20s}  {w*100:5.1f}%  (R19c: {r19c_w*100:4.1f}%  Δ{delta*100:+5.1f}%)")

    # ── 3. Load forecasts & build matrices ──────────────────────
    print(f"\n  Loading extracted forecasts...")
    data = _load_data()
    log = data["log"]
    print(f"  Loaded {len(log)} day-snapshots")

    forecasts, prices, vols, dates, symbols, signals = _prepare_matrices(log)
    corr_matrix = _build_corr_matrix(signals)
    print(f"  Shape: {len(dates)} days × {len(symbols)} symbols × {len(signals)} signals")

    # ── 4. Determine train/test split ───────────────────────────
    train_end = 0
    test_start = 0
    for i, d in enumerate(dates):
        if d >= "2020-01-01":
            train_end = i
            test_start = i
            break
    if train_end == 0:
        train_end = int(len(dates) * 0.65)
        test_start = train_end

    print(f"  Train: days 0-{train_end} ({dates[0]} to {dates[train_end - 1]})")
    print(f"  Test:  days {test_start}-{len(dates)} ({dates[test_start]} to {dates[-1]})")

    # ── 5. Evaluate R19c baseline (train / test / full) ─────────
    print(f"\n  Computing R19c baseline...")
    r19c_train = _simulate_equity(R19C_WEIGHTS, forecasts, prices, vols, signals, corr_matrix, 0, train_end, regime_adaptive=False)
    r19c_test = _simulate_equity(R19C_WEIGHTS, forecasts, prices, vols, signals, corr_matrix, test_start, len(dates), regime_adaptive=False)
    r19c_full = _simulate_equity(R19C_WEIGHTS, forecasts, prices, vols, signals, corr_matrix, 0, len(dates), regime_adaptive=False)

    # ── 6. Evaluate R21a optimized (train / test / full) ────────
    print(f"  Computing R21a optimized (regime-adaptive)...")
    r21a_train = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix, 0, train_end, regime_adaptive=True)
    r21a_test = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix, test_start, len(dates), regime_adaptive=True)
    r21a_full = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix, 0, len(dates), regime_adaptive=True)

    # ── 7. Print comparison table ───────────────────────────────
    print(f"\n{'='*70}")
    print(f"  OUT-OF-SAMPLE EVALUATION RESULTS")
    print(f"{'='*70}")

    header = f"  {'Metric':25s} {'Train':>12s} {'Test (OOS)':>12s} {'Full':>12s}"
    sep = f"  {'':25s} {'─'*12} {'─'*12} {'─'*12}"

    print(f"\n  R19c Baseline (no regime):")
    print(header)
    print(sep)
    for m in ["sharpe", "cagr", "max_dd", "calmar", "total_return", "final_equity"]:
        print(f"  {m:25s} {r19c_train[m]:>12.3f} {r19c_test[m]:>12.3f} {r19c_full[m]:>12.3f}")

    print(f"\n  R21a Optimized (regime-adaptive vol):")
    print(header)
    print(sep)
    for m in ["sharpe", "cagr", "max_dd", "calmar", "total_return", "final_equity"]:
        print(f"  {m:25s} {r21a_train[m]:>12.3f} {r21a_test[m]:>12.3f} {r21a_full[m]:>12.3f}")

    # ── 8. Delta analysis ───────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  DELTA (R21a - R19c) on TEST split:")
    print(f"  {'─'*60}")
    for m in ["sharpe", "cagr", "max_dd", "calmar"]:
        d = r21a_test[m] - r19c_test[m]
        sign = "+" if d >= 0 else ""
        better = "✓" if (m == "max_dd" and d < 0) or (m != "max_dd" and d > 0) else "✗"
        print(f"    {m:20s}  {sign}{d:.3f}  {better}")

    # ── 9. Overfit analysis ─────────────────────────────────────
    train_test_gap = r21a_train["sharpe"] - r21a_test["sharpe"]
    print(f"\n  Overfit check:")
    print(f"    Train Sharpe:     {r21a_train['sharpe']:.3f}")
    print(f"    Test Sharpe:      {r21a_test['sharpe']:.3f}")
    print(f"    Gap:              {train_test_gap:.3f}")

    if train_test_gap > 0.5:
        print(f"    ⚠ WARNING: Large gap ({train_test_gap:.3f}) suggests overfitting!")
    elif train_test_gap > 0.3:
        print(f"    ⚠ CAUTION: Moderate gap, validate with full walk-forward backtest.")
    else:
        print(f"    ✓ OK: Gap within acceptable range.")

    if r21a_test["sharpe"] < 0.8:
        print(f"    ⚠ WARNING: Test Sharpe ({r21a_test['sharpe']:.3f}) < 0.8 — likely overfit!")
    elif r21a_test["sharpe"] >= 1.3:
        print(f"    ✓ STRONG: Test Sharpe ({r21a_test['sharpe']:.3f}) ≥ 1.3 — robust OOS performance.")
    else:
        print(f"    ✓ PASS: Test Sharpe ({r21a_test['sharpe']:.3f}) ≥ 0.8.")

    # ── 10. Summary verdict ─────────────────────────────────────
    print(f"\n{'='*70}")
    oos_sharpe = r21a_test["sharpe"]
    oos_calmar = r21a_test["calmar"]
    oos_cagr = r21a_test["cagr"]
    oos_dd = r21a_test["max_dd"]

    if oos_sharpe >= 1.3 and oos_dd <= 35 and train_test_gap <= 0.5:
        verdict = "ACCEPT — Strong OOS, low drawdown, no overfit"
    elif oos_sharpe >= 0.8 and oos_dd <= 50 and train_test_gap <= 0.5:
        verdict = "ACCEPT (MARGINAL) — Passes minimum thresholds"
    elif train_test_gap > 0.5:
        verdict = "REJECT — Likely overfit (train-test gap too large)"
    elif oos_sharpe < 0.8:
        verdict = "REJECT — OOS Sharpe below minimum threshold"
    else:
        verdict = "REVIEW — Mixed signals, needs walk-forward validation"

    print(f"  VERDICT: {verdict}")
    print(f"  OOS: Sharpe={oos_sharpe:.3f}  CAGR={oos_cagr:.1f}%  MaxDD={oos_dd:.1f}%  Calmar={oos_calmar:.3f}")
    print(f"{'='*70}")

    # ── 11. Save results ────────────────────────────────────────
    results_path = os.path.join(_root, "data", "r21a_oos_evaluation.pkl")
    results = {
        "generation": gen,
        "best_weights": best_weights,
        "r21a_train": r21a_train,
        "r21a_test": r21a_test,
        "r21a_full": r21a_full,
        "r19c_train": r19c_train,
        "r19c_test": r19c_test,
        "r19c_full": r19c_full,
        "train_test_gap": train_test_gap,
        "verdict": verdict,
    }
    with open(results_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
