"""
R21a Cloud Pipeline — Google Colab (FREE).

Runs the R21a pipeline on Google Colab's free tier.
Data persists via Google Drive between sessions.

════════════════════════════════════════════════════════════════
SETUP (paste each cell in Colab):

CELL 1 — Mount Drive & clone code:
    from google.colab import drive
    drive.mount('/content/drive')
    !git clone https://your-repo-url.git /content/centurion  # OR upload zip
    # If no git: upload centurion_core/ to Google Drive, then:
    # !cp -r "/content/drive/MyDrive/centurion_core" /content/centurion/centurion_core

CELL 2 — Install dependencies:
    !pip install -q scipy numpy pandas scikit-learn torch hmmlearn \
        stable-baselines3 ta aiohttp requests chromadb sentence-transformers \
        anthropic psycopg2-binary redis minio pydantic python-dotenv arch

CELL 3 — Run (choose one):
    !cd /content/centurion && python run_cloud_colab.py --step optimize
    !cd /content/centurion && python run_cloud_colab.py --step extract
    !cd /content/centurion && python run_cloud_colab.py --step validate
    !cd /content/centurion && python run_cloud_colab.py --step pipeline

════════════════════════════════════════════════════════════════
SPECS: 2 CPU, 12.7 GB RAM, 12-hr max session
COST:  Free
BEST FOR: Optimizer (~1-2 hrs), individual pipeline steps (~5 hrs each)
NOTE: Full pipeline (~10 hrs) fits within 12-hr limit but is tight.
      Run steps separately if worried about disconnect.
════════════════════════════════════════════════════════════════
"""
import sys
import os
import time
import shutil
import argparse

# ── Detect environment ──
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
IN_KAGGLE = os.path.exists("/kaggle")

# ── Paths ──
if IN_COLAB:
    _PROJECT_ROOT = "/content/centurion" if os.path.exists("/content/centurion") else os.getcwd()
    _DRIVE_BACKUP = "/content/drive/MyDrive/centurion_results"
elif IN_KAGGLE:
    _PROJECT_ROOT = "/kaggle/working/centurion" if os.path.exists("/kaggle/working/centurion") else os.getcwd()
    _DRIVE_BACKUP = "/kaggle/working/results"
else:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # dev_algo/
    _DRIVE_BACKUP = None

_CORE_ROOT = os.path.join(_PROJECT_ROOT, "centurion_core")
_DATA_DIR = os.path.join(_CORE_ROOT, "data")

# Ensure paths exist
os.makedirs(_DATA_DIR, exist_ok=True)
os.makedirs(_DRIVE_BACKUP, exist_ok=True) if _DRIVE_BACKUP else None

# Add to Python path
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CORE_ROOT)

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")


def _backup_to_drive(filename: str):
    """Copy result file to Google Drive for persistence."""
    if not _DRIVE_BACKUP:
        return
    src = os.path.join(_DATA_DIR, filename)
    if os.path.exists(src):
        dst = os.path.join(_DRIVE_BACKUP, filename)
        shutil.copy2(src, dst)
        sz = os.path.getsize(dst) / 1e6
        print(f"  📁 Backed up to Drive: {dst} ({sz:.1f} MB)")


def _restore_from_drive(filename: str):
    """Restore file from Google Drive if not present locally."""
    if not _DRIVE_BACKUP:
        return
    dst = os.path.join(_DATA_DIR, filename)
    src = os.path.join(_DRIVE_BACKUP, filename)
    if not os.path.exists(dst) and os.path.exists(src):
        shutil.copy2(src, dst)
        sz = os.path.getsize(dst) / 1e6
        print(f"  📁 Restored from Drive: {filename} ({sz:.1f} MB)")


def _env_info():
    """Print environment details."""
    import multiprocessing
    env = "Google Colab" if IN_COLAB else "Kaggle" if IN_KAGGLE else "Local"
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
    if _DRIVE_BACKUP:
        print(f"  Backup: {_DRIVE_BACKUP}")


# ══════════════════════════════════════════════════════════════
#  STEP 1: Extract per-source forecasts (~3-5 hours)
# ══════════════════════════════════════════════════════════════
def step_extract():
    """Extract per-source forecasts using R19c base config."""
    print("=" * 70)
    print("  COLAB/KAGGLE — Step 1/3: Forecast Extraction")
    _env_info()
    print("=" * 70, flush=True)

    # Restore any cached data from Drive
    for f in ["earnings_cache.json", "fii_flow_cache.json", "nse_sector_map.json"]:
        _restore_from_drive(f)

    # Restore bhavcopy cache directory
    if _DRIVE_BACKUP and not os.path.exists(os.path.join(_DATA_DIR, "bhavcopy_cache")):
        bhav_src = os.path.join(_DRIVE_BACKUP, "bhavcopy_cache")
        if os.path.exists(bhav_src):
            shutil.copytree(bhav_src, os.path.join(_DATA_DIR, "bhavcopy_cache"))
            print(f"  📁 Restored bhavcopy_cache/ from Drive")

    t0 = time.time()
    from runners.run_extract_forecasts import main as extract_main
    extract_main()
    elapsed = (time.time() - t0) / 3600

    print(f"\n  Step 1 complete in {elapsed:.1f} hours", flush=True)

    # Backup results to Drive
    _backup_to_drive("extracted_forecasts.pkl")
    _backup_to_drive("backtest_checkpoint_extract.pkl")

    # Backup bhavcopy cache for next session
    if _DRIVE_BACKUP:
        bhav_local = os.path.join(_DATA_DIR, "bhavcopy_cache")
        bhav_drive = os.path.join(_DRIVE_BACKUP, "bhavcopy_cache")
        if os.path.exists(bhav_local) and not os.path.exists(bhav_drive):
            shutil.copytree(bhav_local, bhav_drive)
            print(f"  📁 Backed up bhavcopy_cache/ to Drive")

    return elapsed


# ══════════════════════════════════════════════════════════════
#  STEP 2: Optimize signal weights (~1-2 hours on 2 CPU)
# ══════════════════════════════════════════════════════════════
def step_optimize():
    """Optimize signal weights using extracted forecasts."""
    print("=" * 70)
    print("  COLAB/KAGGLE — Step 2/3: Weight Optimization")
    _env_info()
    print("=" * 70, flush=True)

    # Restore extracted forecasts from Drive if needed
    _restore_from_drive("extracted_forecasts.pkl")

    fc_path = os.path.join(_DATA_DIR, "extracted_forecasts.pkl")
    if not os.path.exists(fc_path):
        print(f"\n  ERROR: {fc_path} not found!")
        print(f"  Run step 'extract' first, or upload extracted_forecasts.pkl to:")
        print(f"    Local: {_DATA_DIR}")
        if _DRIVE_BACKUP:
            print(f"    Drive: {_DRIVE_BACKUP}")
        return 0

    t0 = time.time()
    from optimizer.optimize_weights_r21a import run_optimization
    run_optimization()
    elapsed = (time.time() - t0) / 60

    print(f"\n  Step 2 complete in {elapsed:.1f} minutes", flush=True)

    # Backup results
    _backup_to_drive("r21a_optimization_results.pkl")
    return elapsed


# ══════════════════════════════════════════════════════════════
#  STEP 3: Full validation backtest (~3-5 hours)
# ══════════════════════════════════════════════════════════════
def step_validate():
    """Run full validation backtest with optimized weights + regime vol."""
    print("=" * 70)
    print("  COLAB/KAGGLE — Step 3/3: Validation Backtest")
    _env_info()
    print("=" * 70, flush=True)

    # Restore optimization results from Drive
    _restore_from_drive("r21a_optimization_results.pkl")

    opt_path = os.path.join(_DATA_DIR, "r21a_optimization_results.pkl")
    if not os.path.exists(opt_path):
        print(f"\n  WARNING: {opt_path} not found!")
        print(f"  Will use R19c weights as fallback.")

    for f in ["earnings_cache.json", "fii_flow_cache.json", "nse_sector_map.json"]:
        _restore_from_drive(f)

    t0 = time.time()
    from runners.run_r21a import main as validate_main
    validate_main()
    elapsed = (time.time() - t0) / 3600

    print(f"\n  Step 3 complete in {elapsed:.1f} hours", flush=True)

    _backup_to_drive("backtest_checkpoint_r21a.pkl")
    return elapsed


# ══════════════════════════════════════════════════════════════
#  FULL PIPELINE: Extract → Optimize → Validate
# ══════════════════════════════════════════════════════════════
def step_pipeline():
    """Run all 3 steps sequentially with Drive backup between steps."""
    t_total = time.time()

    print("\n" + "=" * 70)
    print("  COLAB/KAGGLE — Full R21a Pipeline")
    print("  WARNING: ~10 hours total. Colab free tier has 12-hr sessions.")
    print("  TIP: Run steps separately if session might disconnect.")
    _env_info()
    print("=" * 70 + "\n", flush=True)

    step_extract()
    step_optimize()

    # Step 3 needs fresh module state for regime mode
    # Use subprocess to avoid module caching issues
    import subprocess
    print("\n" + "=" * 70)
    print("  PIPELINE — Step 3/3: Validation (subprocess for clean state)")
    print("=" * 70, flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_PROJECT_ROOT}:{_CORE_ROOT}"
    proc = subprocess.run(
        [sys.executable, os.path.join(_CORE_ROOT, "runners", "run_r21a.py")],
        env=env,
        cwd=_PROJECT_ROOT,
    )
    elapsed = (time.time() - t0) / 3600
    _backup_to_drive("backtest_checkpoint_r21a.pkl")

    total_hrs = (time.time() - t_total) / 3600
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — {total_hrs:.1f} hours total")
    if _DRIVE_BACKUP:
        print(f"  Results backed up to: {_DRIVE_BACKUP}")
    print(f"{'='*70}", flush=True)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R21a Cloud Pipeline (Colab/Kaggle)")
    parser.add_argument("--step", default="optimize",
                        choices=["extract", "optimize", "validate", "pipeline"],
                        help="Pipeline step to run (default: optimize)")
    args = parser.parse_args()

    print(f"\n  Starting step: {args.step}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.step == "extract":
        step_extract()
    elif args.step == "optimize":
        step_optimize()
    elif args.step == "validate":
        step_validate()
    elif args.step == "pipeline":
        step_pipeline()
