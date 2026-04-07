"""
Compute container entrypoint — routes to the correct pipeline step.
Controlled by PIPELINE_STEP env var: extract | optimize | validate | pipeline
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

os.chdir("/app")
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/centurion_core")

step = os.environ.get("PIPELINE_STEP", "pipeline").lower().strip()

print(f"{'='*70}")
print(f"  Centurion R21a Compute Container")
print(f"  Step: {step}")
print(f"  CPUs: {os.cpu_count()}")
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print(f"{'='*70}", flush=True)

t0 = time.time()

if step == "extract":
    from runners.run_extract_forecasts import main
    main()

elif step == "optimize":
    from optimizer.optimize_weights_r21a import run_optimization
    run_optimization()

elif step == "validate":
    from runners.run_r21a import main
    main()

elif step == "pipeline":
    from runners.run_r21a_pipeline import step1_extract, step2_optimize, step3_validate
    step1_extract()
    step2_optimize()
    step3_validate()

else:
    print(f"  ERROR: Unknown step '{step}'")
    print(f"  Valid: extract | optimize | validate | pipeline")
    sys.exit(1)

elapsed = (time.time() - t0) / 3600
print(f"\n{'='*70}")
print(f"  Step '{step}' complete — {elapsed:.1f} hours")
print(f"{'='*70}", flush=True)
