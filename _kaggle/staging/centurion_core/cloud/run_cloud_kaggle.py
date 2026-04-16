"""
R21a Cloud Pipeline — Kaggle Notebooks (FREE, RECOMMENDED).

Kaggle offers the BEST free-tier specs: 4 CPU, 29 GB RAM, 12-hr sessions.
Data persists via Kaggle Datasets between sessions.

════════════════════════════════════════════════════════════════
SETUP — Kaggle Notebook:

1. Create a new Kaggle Notebook → Settings → Internet ON, GPU OFF (CPU mode)
2. Upload centurion_core as a Kaggle Dataset:
   - kaggle.com → Your Profile → Datasets → New Dataset
   - Name: "centurion-core"
   - Upload centurion_core/ folder as a zip
   - All scripts (runners, cloud, optimizer) are now inside centurion_core/

3. In notebook, add the dataset:
   - Right sidebar → Add Data → Your Datasets → centurion-core

════════════════════════════════════════════════════════════════
NOTEBOOK CELLS:

CELL 1 — Setup:
    !pip install -q scipy hmmlearn stable-baselines3 ta arch anthropic \
        chromadb sentence-transformers psycopg2-binary redis minio \
        kiteconnect pydantic-settings

CELL 2 — Copy code to working directory:
    import shutil, os
    src = "/kaggle/input/centurion-core"
    dst = "/kaggle/working/centurion"
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(src, dst)
    os.makedirs(f"{dst}/centurion_core/data", exist_ok=True)

CELL 3 — Run optimizer:
    %cd /kaggle/working/centurion
    !python centurion_core/cloud/run_cloud_kaggle.py --step optimize

CELL 4 — Download results (after run completes):
    from IPython.display import FileLink
    FileLink("/kaggle/working/centurion/centurion_core/data/r21a_optimization_results.pkl")

════════════════════════════════════════════════════════════════
SPECS: 4 CPU, 29 GB RAM, 12-hr session, 30 hrs/week quota
COST:  Free (with Kaggle account)
BEST FOR: Optimizer (~30-60 min with 4 CPUs!), any single step
════════════════════════════════════════════════════════════════
"""
import sys
import os
import time
import shutil
import argparse

# ── Detect environment ──
IN_KAGGLE = os.path.exists("/kaggle")
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

# ── Paths ──
if IN_KAGGLE:
    _PROJECT_ROOT = "/kaggle/working/centurion"
    _INPUT_DATA = "/kaggle/input/centurion-core"  # Read-only dataset
    _OUTPUT_DIR = "/kaggle/working/results"
elif IN_COLAB:
    _PROJECT_ROOT = "/content/centurion"
    _INPUT_DATA = None
    _OUTPUT_DIR = "/content/drive/MyDrive/centurion_results"
else:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # dev_algo/
    _INPUT_DATA = None
    _OUTPUT_DIR = None

_CORE_ROOT = os.path.join(_PROJECT_ROOT, "centurion_core")
_DATA_DIR = os.path.join(_CORE_ROOT, "data")

os.makedirs(_DATA_DIR, exist_ok=True)
if _OUTPUT_DIR:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CORE_ROOT)

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")


def _restore_from_input(filename: str):
    """Restore file from Kaggle input dataset (read-only)."""
    dst = os.path.join(_DATA_DIR, filename)
    if os.path.exists(dst):
        return  # Already present
    if _INPUT_DATA:
        # Check both direct and nested paths
        candidates = [
            os.path.join(_INPUT_DATA, "centurion_core", "data", filename),
            os.path.join(_INPUT_DATA, "data", filename),
            os.path.join(_INPUT_DATA, filename),
        ]
        for src in candidates:
            if os.path.exists(src):
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                sz = os.path.getsize(dst) / 1e6 if os.path.isfile(dst) else 0
                print(f"  Restored from dataset: {filename} ({sz:.1f} MB)")
                return


def _save_output(filename: str):
    """Copy result file to output directory for download."""
    src = os.path.join(_DATA_DIR, filename)
    if not os.path.exists(src):
        return
    if _OUTPUT_DIR:
        dst = os.path.join(_OUTPUT_DIR, filename)
        shutil.copy2(src, dst)
        sz = os.path.getsize(dst) / 1e6
        print(f"  Saved to output: {dst} ({sz:.1f} MB)")
    # Also copy to /kaggle/working/ for easy download
    if IN_KAGGLE:
        dl_path = os.path.join("/kaggle/working", filename)
        shutil.copy2(src, dl_path)
        print(f"  Download: {dl_path}")


def _env_info():
    """Print environment details."""
    import multiprocessing
    env = "Kaggle" if IN_KAGGLE else "Google Colab" if IN_COLAB else "Local"
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        ram_gb = 0
    print(f"  Environment: {env}")
    print(f"  CPUs: {multiprocessing.cpu_count()}")
    if ram_gb > 0:
        print(f"  RAM: {ram_gb:.1f} GB")
    print(f"  Project: {_PROJECT_ROOT}")
    print(f"  Data: {_DATA_DIR}")


# ══════════════════════════════════════════════════════════════
#  STEP 1: Extract forecasts
# ══════════════════════════════════════════════════════════════
def step_extract():
    print("=" * 70)
    print("  KAGGLE — Step 1/3: Forecast Extraction (~5 hours)")
    _env_info()
    print("=" * 70, flush=True)

    # Restore cached data
    for f in ["earnings_cache.json", "fii_flow_cache.json", "nse_sector_map.json"]:
        _restore_from_input(f)
    _restore_from_input("bhavcopy_cache")

    t0 = time.time()
    from runners.run_extract_forecasts import main as extract_main
    extract_main()
    elapsed = (time.time() - t0) / 3600

    print(f"\n  Step 1 complete in {elapsed:.1f} hours", flush=True)
    _save_output("extracted_forecasts.pkl")
    _save_output("backtest_checkpoint_extract.pkl")
    return elapsed


# ══════════════════════════════════════════════════════════════
#  STEP 2: Optimize weights (~30-60 min on Kaggle's 4 CPUs)
# ══════════════════════════════════════════════════════════════
def step_optimize():
    print("=" * 70)
    print("  KAGGLE — Step 2/3: Weight Optimization (~30-60 min)")
    _env_info()
    print("=" * 70, flush=True)

    _restore_from_input("extracted_forecasts.pkl")

    fc_path = os.path.join(_DATA_DIR, "extracted_forecasts.pkl")
    if not os.path.exists(fc_path):
        print(f"\n  ERROR: extracted_forecasts.pkl not found!")
        print(f"  Upload it to your Kaggle dataset, or run 'extract' step first.")
        print(f"  Expected location: centurion_core/data/extracted_forecasts.pkl")
        return 0

    t0 = time.time()
    from optimizer.optimize_weights_r21a import run_optimization
    run_optimization()
    elapsed = (time.time() - t0) / 60

    print(f"\n  Step 2 complete in {elapsed:.1f} minutes", flush=True)
    _save_output("r21a_optimization_results.pkl")
    return elapsed


# ══════════════════════════════════════════════════════════════
#  STEP 3: Validation backtest
# ══════════════════════════════════════════════════════════════
def step_validate():
    print("=" * 70)
    print("  KAGGLE — Step 3/3: Validation Backtest (~5 hours)")
    _env_info()
    print("=" * 70, flush=True)

    _restore_from_input("r21a_optimization_results.pkl")
    for f in ["earnings_cache.json", "fii_flow_cache.json", "nse_sector_map.json"]:
        _restore_from_input(f)
    _restore_from_input("bhavcopy_cache")

    t0 = time.time()
    from runners.run_r21a import main as validate_main
    validate_main()
    elapsed = (time.time() - t0) / 3600

    print(f"\n  Step 3 complete in {elapsed:.1f} hours", flush=True)
    _save_output("backtest_checkpoint_r21a.pkl")
    return elapsed


# ══════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════
def step_pipeline():
    t_total = time.time()

    print("\n" + "=" * 70)
    print("  KAGGLE — Full R21a Pipeline (~10 hours)")
    print("  NOTE: Kaggle gives 12-hr sessions — this is tight.")
    print("  SAFER: Run extract + optimize + validate as separate sessions.")
    _env_info()
    print("=" * 70 + "\n", flush=True)

    step_extract()
    step_optimize()

    # Step 3 via subprocess for clean module state
    import subprocess
    print("\n" + "=" * 70)
    print("  PIPELINE — Step 3/3: Validation (subprocess)")
    print("=" * 70, flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_PROJECT_ROOT}:{_CORE_ROOT}"
    proc = subprocess.run(
        [sys.executable, os.path.join(_CORE_ROOT, "runners", "run_r21a.py")],
        env=env, cwd=_PROJECT_ROOT,
    )
    _save_output("backtest_checkpoint_r21a.pkl")

    total_hrs = (time.time() - t_total) / 3600
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — {total_hrs:.1f} hours total")
    print(f"{'='*70}", flush=True)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R21a Pipeline — Kaggle/Colab")
    parser.add_argument("--step", default="optimize",
                        choices=["extract", "optimize", "validate", "pipeline"],
                        help="Pipeline step (default: optimize)")
    args = parser.parse_args()

    print(f"\n  Starting: {args.step}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    {"extract": step_extract, "optimize": step_optimize,
     "validate": step_validate, "pipeline": step_pipeline}[args.step]()
