"""
R21a Cloud Pipeline — Modal Serverless Compute (PAID).

NOTE: Modal charges ~$0.50 per optimizer run / ~$2 per full pipeline.
      For FREE alternatives, see:
        - run_cloud_colab.py   (Google Colab — free, 12hr sessions)
        - run_cloud_kaggle.py  (Kaggle — free, 4 CPU / 29 GB / 12hr)

────────────────────────────────────────────────────────────────
FIRST-TIME SETUP (one-time, ~5 min):

    pip install modal
    modal token new
    modal volume create centurion-data
    modal volume put centurion-data centurion_core/data/bhavcopy_cache/ bhavcopy_cache/
    modal volume put centurion-data centurion_core/data/extracted_forecasts.pkl extracted_forecasts.pkl
    modal volume put centurion-data centurion_core/data/earnings_cache.json earnings_cache.json
    modal volume put centurion-data centurion_core/data/fii_flow_cache.json fii_flow_cache.json
    modal volume put centurion-data centurion_core/data/nse_sector_map.json nse_sector_map.json

USAGE:
    modal run run_cloud_modal.py --step optimize    # ~15-30 min
    modal run run_cloud_modal.py --step pipeline    # ~7-10 hours
    modal run run_cloud_modal.py --step extract|validate
    modal volume get centurion-data r21a_optimization_results.pkl centurion_core/data/

COST: 8 CPU × $0.024/CPU-hr ≈ $0.50 optimizer / $2.00 pipeline
────────────────────────────────────────────────────────────────
"""
import modal
import os
import sys
import time

# ── Modal App ──
app = modal.App("centurion-r21a")

# Persistent volume for data (survives across runs, stores checkpoints + results)
data_vol = modal.Volume.from_name("centurion-data", create_if_missing=True)

# Secrets from local .env (Neon DB, API keys, etc.)
secrets = modal.Secret.from_dotenv(path=".env")

# ── Compute image with all Python dependencies ──
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++", "build-essential", "curl")
    .pip_install(
        "pandas", "numpy", "scipy", "scikit-learn",
        "torch", "aiohttp", "requests",
        "psycopg2-binary", "redis", "minio",
        "chromadb", "sentence-transformers",
        "hmmlearn", "stable-baselines3",
        "ta", "pytz", "python-dotenv",
        "pydantic", "pydantic-settings",
        "anthropic", "sentry-sdk",
    )
)

# ── Code mounts (read-only, rebuilt each deploy) ──
# Mount centurion_core code excluding data/ and heavy dirs
_CODE_EXCLUDES = {
    "data", "__pycache__", "chroma_store", "chroma_db",
    "node_modules", "rl_models", "rl_uploads", "rag_uploads",
    ".pyc", "frontend", "centurion_core-fe", "myenv",
}

code_mount = modal.Mount.from_local_dir(
    local_path="centurion_core",
    remote_path="/app/centurion_core",
    condition=lambda pth: not any(excl in pth for excl in _CODE_EXCLUDES),
)

# Combined mounts (runner/optimizer scripts now inside centurion_core/)
all_mounts = [code_mount]

# Shared function config
_SHARED_CONFIG = dict(
    image=image,
    mounts=all_mounts,
    volumes={"/app/centurion_core/data": data_vol},
    secrets=[secrets],
)


# ══════════════════════════════════════════════════════════════
#  STEP 1: Extract per-source forecasts (~3-5 hours)
# ══════════════════════════════════════════════════════════════
@app.function(
    **_SHARED_CONFIG,
    cpu=4.0,
    memory=16384,
    timeout=28800,  # 8 hours max
)
def run_extract():
    """Step 1: Extract per-source forecasts (R19c base)."""
    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/centurion_core")

    print("=" * 70)
    print("  CLOUD — Step 1/3: Forecast Extraction")
    print(f"  CPUs: {os.cpu_count()}, RAM: ~16 GB")
    print("=" * 70, flush=True)

    t0 = time.time()
    from run_extract_forecasts import main as extract_main
    extract_main()
    elapsed = (time.time() - t0) / 3600
    print(f"\n  Step 1 complete in {elapsed:.1f} hours", flush=True)

    # Persist volume data
    data_vol.commit()
    return {"status": "ok", "elapsed_hours": round(elapsed, 2)}


# ══════════════════════════════════════════════════════════════
#  STEP 2: Optimize signal weights (~15-30 min with 8 CPUs)
# ══════════════════════════════════════════════════════════════
@app.function(
    **_SHARED_CONFIG,
    cpu=8.0,       # maximize parallel workers
    memory=32768,  # 32 GB for large matrices
    timeout=7200,  # 2 hours max
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0),
)
def run_optimize():
    """Step 2: Optimize signal weights (parallel differential evolution)."""
    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/centurion_core")

    print("=" * 70)
    print("  CLOUD — Step 2/3: Weight Optimization")
    print(f"  CPUs: {os.cpu_count()}, RAM: ~32 GB")
    print("=" * 70, flush=True)

    t0 = time.time()
    from optimizer.optimize_weights_r21a import run_optimization
    run_optimization()
    elapsed = (time.time() - t0) / 60
    print(f"\n  Step 2 complete in {elapsed:.1f} minutes", flush=True)

    data_vol.commit()
    return {"status": "ok", "elapsed_minutes": round(elapsed, 1)}


# ══════════════════════════════════════════════════════════════
#  STEP 3: Full validation backtest (~3-5 hours)
# ══════════════════════════════════════════════════════════════
@app.function(
    **_SHARED_CONFIG,
    cpu=4.0,
    memory=16384,
    timeout=28800,  # 8 hours max
)
def run_validate():
    """Step 3: Full validation backtest with optimized weights + regime vol."""
    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/centurion_core")

    print("=" * 70)
    print("  CLOUD — Step 3/3: Validation Backtest")
    print(f"  CPUs: {os.cpu_count()}, RAM: ~16 GB")
    print("=" * 70, flush=True)

    t0 = time.time()
    from runners.run_r21a import main as validate_main
    validate_main()
    elapsed = (time.time() - t0) / 3600
    print(f"\n  Step 3 complete in {elapsed:.1f} hours", flush=True)

    data_vol.commit()
    return {"status": "ok", "elapsed_hours": round(elapsed, 2)}


# ══════════════════════════════════════════════════════════════
#  FULL PIPELINE: Extract → Optimize → Validate
# ══════════════════════════════════════════════════════════════
@app.function(
    **_SHARED_CONFIG,
    cpu=8.0,
    memory=32768,
    timeout=50400,  # 14 hours max
)
def run_pipeline():
    """Full R21a pipeline: extract → optimize → validate."""
    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/centurion_core")

    t_total = time.time()

    # ── Step 1: Extract ──
    print("=" * 70)
    print("  CLOUD PIPELINE — Step 1/3: Forecast Extraction")
    print("=" * 70, flush=True)
    t0 = time.time()
    from runners.run_extract_forecasts import main as extract_main
    extract_main()
    data_vol.commit()
    print(f"\n  Step 1 done in {(time.time()-t0)/3600:.1f} hrs\n", flush=True)

    # ── Step 2: Optimize ──
    # Need fresh import since step1 loaded bt_mod with extraction flags
    print("=" * 70)
    print("  CLOUD PIPELINE — Step 2/3: Weight Optimization")
    print("=" * 70, flush=True)
    t0 = time.time()
    from optimizer.optimize_weights_r21a import run_optimization
    run_optimization()
    data_vol.commit()
    print(f"\n  Step 2 done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    # ── Step 3: Validate ──
    # Subprocess to get clean module state (regime mode needs fresh bt_mod)
    print("=" * 70)
    print("  CLOUD PIPELINE — Step 3/3: Validation Backtest")
    print("=" * 70, flush=True)
    t0 = time.time()
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app:/app/centurion_core"
    proc = subprocess.run(
        [sys.executable, "/app/centurion_core/runners/run_r21a.py"],
        env=env,
        cwd="/app",
    )
    data_vol.commit()
    print(f"\n  Step 3 done in {(time.time()-t0)/3600:.1f} hrs", flush=True)

    total_hrs = (time.time() - t_total) / 3600
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — {total_hrs:.1f} hours total")
    print(f"  Download results:")
    print(f"    modal volume get centurion-data r21a_optimization_results.pkl centurion_core/data/")
    print(f"    modal volume get centurion-data backtest_checkpoint_r21a.pkl centurion_core/data/")
    print(f"{'='*70}", flush=True)

    return {
        "status": "ok" if proc.returncode == 0 else "step3_failed",
        "elapsed_hours": round(total_hrs, 2),
    }


# ══════════════════════════════════════════════════════════════
#  DATA SYNC UTILITIES
# ══════════════════════════════════════════════════════════════
@app.function(
    image=image,
    volumes={"/data": data_vol},
    timeout=300,
)
def list_volume_contents():
    """List files in the cloud data volume."""
    import pathlib
    results = []
    for p in sorted(pathlib.Path("/data").rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            results.append(f"  {str(p.relative_to('/data')):60s}  {sz/1e6:.1f} MB")
    return "\n".join(results) if results else "  (empty)"


@app.function(
    image=image,
    volumes={"/data": data_vol},
    timeout=300,
)
def check_readiness():
    """Check if all required data files are present in the cloud volume."""
    import pathlib
    checks = {
        "extracted_forecasts.pkl": "Step 2 (optimize) — REQUIRED",
        "bhavcopy_cache": "Steps 1 & 3 (extract/validate) — needed for fresh runs",
        "earnings_cache.json": "Optional — earnings data",
        "fii_flow_cache.json": "Optional — FII flow data",
        "nse_sector_map.json": "Optional — sector mapping",
    }
    print("\n  Cloud Volume Readiness Check:")
    print("  " + "─" * 60)
    all_ok = True
    for name, desc in checks.items():
        path = pathlib.Path("/data") / name
        exists = path.exists()
        if exists:
            if path.is_file():
                sz = path.stat().st_size / 1e6
                print(f"  ✓ {name:40s} {sz:.1f} MB")
            else:
                n_files = len(list(path.rglob("*")))
                print(f"  ✓ {name:40s} ({n_files} files)")
        else:
            print(f"  ✗ {name:40s} MISSING — {desc}")
            if "REQUIRED" in desc:
                all_ok = False
    print("  " + "─" * 60)
    if all_ok:
        print("  Ready for optimization! Run: modal run run_cloud_modal.py --step optimize")
    else:
        print("  Missing required files. Upload with: modal volume put centurion-data <local_path> <remote_name>")
    return all_ok


# ══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════
@app.local_entrypoint()
def main(step: str = "pipeline"):
    """
    Run R21a pipeline steps on Modal cloud.

    Args:
        step: extract | optimize | validate | pipeline | check | list
    """
    if step == "check":
        ready = check_readiness.remote()
        return

    if step == "list":
        contents = list_volume_contents.remote()
        print("\n  Modal Volume 'centurion-data' contents:")
        print(contents)
        return

    if step == "extract":
        print("\n  Launching Step 1 (extract) on Modal cloud...")
        print("  Monitor: modal app logs centurion-r21a\n")
        result = run_extract.remote()
        print(f"\n  Result: {result}")

    elif step == "optimize":
        print("\n  Launching Step 2 (optimize) on Modal cloud...")
        print("  Monitor: modal app logs centurion-r21a\n")
        result = run_optimize.remote()
        print(f"\n  Result: {result}")
        print("\n  Download: modal volume get centurion-data r21a_optimization_results.pkl centurion_core/data/")

    elif step == "validate":
        print("\n  Launching Step 3 (validate) on Modal cloud...")
        print("  Monitor: modal app logs centurion-r21a\n")
        result = run_validate.remote()
        print(f"\n  Result: {result}")
        print("\n  Download: modal volume get centurion-data backtest_checkpoint_r21a.pkl centurion_core/data/")

    elif step == "pipeline":
        print("\n  Launching full pipeline on Modal cloud (~7-10 hours)...")
        print("  Monitor: modal app logs centurion-r21a")
        print("  Safe to close this terminal — pipeline runs in cloud.\n")
        result = run_pipeline.remote()
        print(f"\n  Result: {result}")
        print("\n  Download results:")
        print("    modal volume get centurion-data r21a_optimization_results.pkl centurion_core/data/")
        print("    modal volume get centurion-data backtest_checkpoint_r21a.pkl centurion_core/data/")

    else:
        print(f"  Unknown step: {step}")
        print("  Valid steps: extract | optimize | validate | pipeline | check | list")


if __name__ == "__main__":
    print("Use: modal run run_cloud_modal.py --step <step>")
    print("Steps: extract | optimize | validate | pipeline | check | list")
