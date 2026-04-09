"""
Contra-Regime Strategy Backtest — Kaggle Notebook Script.

Runs all 4 variants of the contra-regime strategy on Kaggle free tier:
  V0: R21a baseline (control)
  V1: Bear dip-buyer — MR signals get 3.33× vol boost in downtrend
  V2: Bull profit-taker — tighter stops (6σ) in uptrend
  V3: Combined V1+V2

Paper trading is NOT affected — backtest uses historical Yahoo Finance data only.

═══════════════════════════════════════════════════════════════════
KAGGLE SETUP (one-time):

1. Create Kaggle Notebook → Settings → Internet ON, GPU OFF (CPU-only)
2. Upload centurion_core as Kaggle Dataset:
   - kaggle.com → Your Profile → Datasets → New Dataset
   - Name: "centurion-core"
   - Upload centurion_core/ folder as zip (include data/ folder)
3. In notebook, Add Data → Your Datasets → centurion-core

═══════════════════════════════════════════════════════════════════
NOTEBOOK CELLS (copy each block into a Kaggle cell):

CELL 1 — pip installs (run once)
CELL 2 — Copy code to working dir
CELL 3 — Run all 4 variants (~4-6 hours total on Kaggle 4-CPU)
CELL 4 — Download results

ESTIMATED TIME: ~60-90 min per variant × 4 = 4-6 hours total
SPECS: 4 CPU, 29 GB RAM, 12-hr session limit
═══════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════
# %% CELL 1 — Install dependencies (Kaggle has many pre-installed)
# ═══════════════════════════════════════════════════════════════════

# !pip install -q yfinance hmmlearn scipy ta arch statsmodels \
#     scikit-learn matplotlib seaborn pandas numpy

# ═══════════════════════════════════════════════════════════════════
# %% CELL 2 — Setup: copy code to working dir + path config
# ═══════════════════════════════════════════════════════════════════

import shutil
import os
import sys

# Kaggle stores uploaded datasets in /kaggle/input/
DATASET_NAME = "centurion-core"  # <-- change to match your dataset name
SRC = f"/kaggle/input/{DATASET_NAME}"
DST = "/kaggle/working/centurion_core"

# Copy dataset to working directory (writable)
if os.path.exists(DST):
    shutil.rmtree(DST)

# Handle both flat and nested upload structures
if os.path.exists(os.path.join(SRC, "centurion_core")):
    shutil.copytree(os.path.join(SRC, "centurion_core"), DST)
elif os.path.exists(os.path.join(SRC, "services")):
    shutil.copytree(SRC, DST)
else:
    print(f"ERROR: Cannot find centurion_core in {SRC}")
    print(f"Contents: {os.listdir(SRC)}")
    raise FileNotFoundError(f"centurion_core not found in {SRC}")

os.makedirs(f"{DST}/data", exist_ok=True)
print(f"✓ Code copied to {DST}")

# Add to Python path
if DST not in sys.path:
    sys.path.insert(0, DST)
print(f"✓ Python path configured")

# Verify key modules import
try:
    from services.full_pipeline_backtest import (
        _CONTRA_BEAR_DIP_BUYER,
        _CONTRA_BULL_PROFIT_TAKER,
        _R21A_REGIME_VOL,
    )
    print(f"✓ Backtest module loaded")
    print(f"  Contra flags: DIP={_CONTRA_BEAR_DIP_BUYER}, PROFIT={_CONTRA_BULL_PROFIT_TAKER}")
    print(f"  R21A regime vol: {_R21A_REGIME_VOL}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    raise

# ═══════════════════════════════════════════════════════════════════
# %% CELL 3 — Run all 4 contra-regime variants
# ═══════════════════════════════════════════════════════════════════

# Option A: Run all 4 variants via the unified runner (RECOMMENDED)
# This takes ~4-6 hours but produces a clean comparison table at the end.

os.chdir(DST)
os.environ["CENTURION_BT_CHECKPOINT"] = ""  # fresh run (no resume)

from cloud.run_kaggle import task_contra_all

results = task_contra_all()

# ═══════════════════════════════════════════════════════════════════
# %% CELL 3-ALT — Run individual variant (use if hitting time limits)
# ═══════════════════════════════════════════════════════════════════

# Uncomment ONE of these to run a single variant per Kaggle session:
#
# from cloud.run_kaggle import task_contra_v0
# result = task_contra_v0()   # ~60-90 min — R21a baseline
#
# from cloud.run_kaggle import task_contra_v1
# result = task_contra_v1()   # ~60-90 min — Bear dip-buyer
#
# from cloud.run_kaggle import task_contra_v2
# result = task_contra_v2()   # ~60-90 min — Bull profit-taker
#
# from cloud.run_kaggle import task_contra_v3
# result = task_contra_v3()   # ~60-90 min — Combined

# ═══════════════════════════════════════════════════════════════════
# %% CELL 4 — Display results + download
# ═══════════════════════════════════════════════════════════════════

import pickle

results_path = "/kaggle/working/contra_regime_results.pkl"
if os.path.exists(results_path):
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    print("\n" + "=" * 78)
    print("  CONTRA-REGIME FINAL RESULTS")
    print("=" * 78)

    R21A_REF = {"sharpe": 2.093, "cagr": 74.1, "maxdd": 25.2, "calmar": 2.937}

    for name, r in results.items():
        if r is None:
            print(f"\n  {name}: FAILED")
            continue
        s = r.get("sharpe", 0)
        c = r.get("annual_return_pct", 0)
        m = r.get("max_drawdown_pct", 100)
        cal = r.get("calmar", 0)
        ok = s >= 2.0 and m <= 30.0 and cal >= 2.5
        verdict = "✓ ACCEPT" if ok else "✗ REJECT"

        print(f"\n  {name}:")
        print(f"    Sharpe:  {s:.3f}  (R21a: {R21A_REF['sharpe']:.3f}, Δ{s - R21A_REF['sharpe']:+.3f})")
        print(f"    CAGR:    {c:.1f}%  (R21a: {R21A_REF['cagr']:.1f}%, Δ{c - R21A_REF['cagr']:+.1f}%)")
        print(f"    MaxDD:   {m:.1f}%  (R21a: {R21A_REF['maxdd']:.1f}%, Δ{m - R21A_REF['maxdd']:+.1f}%)")
        print(f"    Calmar:  {cal:.3f}  (R21a: {R21A_REF['calmar']:.3f}, Δ{cal - R21A_REF['calmar']:+.3f})")
        print(f"    → {verdict}")

    print(f"\n{'=' * 78}")

    # Recommendation
    best = None
    best_sharpe = 0
    for name, r in results.items():
        if r and r.get("sharpe", 0) > best_sharpe:
            s = r.get("sharpe", 0)
            m = r.get("max_drawdown_pct", 100)
            cal = r.get("calmar", 0)
            if s >= 2.0 and m <= 30.0 and cal >= 2.5:
                best = name
                best_sharpe = s

    if best:
        print(f"\n  RECOMMENDATION: Deploy {best} (highest Sharpe among accepted variants)")
    else:
        print(f"\n  RECOMMENDATION: Keep R21a baseline (no variant met accept criteria)")

    # Download link
    from IPython.display import FileLink
    display(FileLink(results_path))
else:
    print("No results file found. Run Cell 3 first.")
