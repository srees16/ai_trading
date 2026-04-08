import sys
import os
import time
sys.stdout.reconfigure(line_buffering=True)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from services.full_pipeline_backtest import run_full_backtest

MAX_RETRIES = 5

for _attempt in range(1, MAX_RETRIES + 1):
    try:
        result = run_full_backtest(
            tickers=None,
            capital=500_000,
            period="13y",
            market="IND",
            verbose=True,
            start_date="2012-01-01",
            end_date="2025-12-31",
        )
        print(result.report)
        break  # success — exit retry loop
    except Exception as e:
        print(f"\n  CRASH on attempt {_attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}", flush=True)
        if _attempt < MAX_RETRIES:
            print(f"  Will retry in 5s (checkpoint will auto-resume)...", flush=True)
            time.sleep(5)
        else:
            print(f"  All {MAX_RETRIES} attempts exhausted.", flush=True)
            raise
