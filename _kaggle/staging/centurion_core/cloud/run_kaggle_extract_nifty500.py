"""
NIFTY500 Forecast Extraction — Kaggle Cloud Runner.

Extracts per-source forecasts for ~500 NSE stocks + gold + BTC
across 13 years (2012-2025). Includes ALL 20+ signals:
  - 11 active (v27 champion)
  - 6 new alpha sources (calendar, fundamental_momentum, insider,
    dispersion, gold_equity_rotation, crypto_correlation)
  - 3 extra (carry, skew_signal, ewmac_32_128)

Designed for Kaggle's 12-hour session limit:
  - Saves checkpoint every 50 days + at 11h mark (1h safety margin)
  - Persists exact ticker list so resume uses identical symbols
  - forecast_log included in checkpoint for cross-session extraction

Usage (Kaggle notebook):
    !python centurion_core/cloud/run_kaggle_extract_nifty500.py

Session workflow:
  Session 1: Run → checkpoint saved at ~Day 1000-1200 (or at 11h timeout)
  Session 2: Upload checkpoint to dataset → re-run → resumes → Day 2200+
  Session 3: Resume → completes → download extracted_forecasts_nifty500.pkl
"""
import sys
import os
import time
import shutil
import pickle
import json

# ── Environment detection ──
IN_KAGGLE = os.path.exists("/kaggle")

# Kaggle gives 12 hours max. Save checkpoint at 11h to leave 1h safety margin.
_KAGGLE_MAX_RUNTIME_SECS = 11 * 3600  # 39600 seconds = 11 hours

if IN_KAGGLE:
    _PROJECT_ROOT = "/kaggle/working/centurion_core"
    _INPUT_DATA = "/kaggle/input/centurion-core"
    _OUTPUT_DIR = "/kaggle/working"
else:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
    _INPUT_DATA = None
    _OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data")

_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
os.makedirs(_DATA_DIR, exist_ok=True)

# Python path setup
_parent = os.path.dirname(_PROJECT_ROOT)
for p in [_parent, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

_CHECKPOINT_PATH = os.path.join(_DATA_DIR, "backtest_checkpoint_nifty500_extract.pkl")
_OUTPUT_PATH = os.path.join(_DATA_DIR, "extracted_forecasts_nifty500.pkl")
_TICKER_LIST_PATH = os.path.join(_DATA_DIR, "nifty500_ticker_list.json")


def _restore_from_input(filename: str):
    """Restore file from Kaggle input dataset (read-only).
    Searches kernel self-reference output first, then code dataset."""
    dst = os.path.join(_DATA_DIR, filename)
    if os.path.exists(dst):
        return

    # Search order: kernel's own previous output > code dataset
    _PREV_OUTPUT = "/kaggle/input/nifty500-extract-all-signals"
    search_roots = [_PREV_OUTPUT]
    if _INPUT_DATA:
        search_roots.append(_INPUT_DATA)

    candidates = []
    for root in search_roots:
        candidates.extend([
            os.path.join(root, "centurion_core", "data", filename),
            os.path.join(root, "data", filename),
            os.path.join(root, filename),
        ])
    for src in candidates:
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            sz = os.path.getsize(dst) / 1e6 if os.path.isfile(dst) else 0
            print(f"  Restored: {filename} ({sz:.1f} MB)")
            return


def main():
    t0 = time.time()

    print("=" * 70)
    print("  NIFTY500 Forecast Extraction — All Signals")
    print("  Universe: NIFTY500 (~500 stocks) + GOLDBEES.NS + BTC-USD")
    print("  Signals: ALL 20+ (including 6 new alpha sources)")
    print("  Period: 2012-01-01 to 2025-12-31")
    print("  Checkpoint: every 50 days + at 11h timeout")
    print("=" * 70)

    import multiprocessing
    print(f"  Environment: {'Kaggle' if IN_KAGGLE else 'Local'}")
    print(f"  CPUs: {multiprocessing.cpu_count()}")
    try:
        import psutil
        print(f"  RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    except ImportError:
        pass
    print(f"  Checkpoint: {_CHECKPOINT_PATH}")
    print(f"  Output: {_OUTPUT_PATH}")
    print(flush=True)

    # Restore caches from dataset
    for f in ["earnings_cache.json", "fii_flow_cache.json", "nse_sector_map.json"]:
        _restore_from_input(f)
    _restore_from_input("bhavcopy_cache")
    # Restore checkpoint + ticker list from previous session
    # V14: CENTURION_FRESH_START env var skips checkpoint (notebook sets it when old code was wrong)
    if os.environ.get("CENTURION_FRESH_START") == "1":
        print("  FRESH START: skipping checkpoint restore (CENTURION_FRESH_START=1)")
        if os.path.exists(_CHECKPOINT_PATH):
            os.remove(_CHECKPOINT_PATH)
            print(f"  Deleted stale checkpoint: {_CHECKPOINT_PATH}")
    else:
        _restore_from_input("backtest_checkpoint_nifty500_extract.pkl")
    _restore_from_input("nifty500_ticker_list.json")

    # ── Configure extraction ──
    os.environ["CENTURION_BT_CHECKPOINT"] = _CHECKPOINT_PATH

    # Set runtime limit so backtest saves checkpoint before Kaggle kills us
    if IN_KAGGLE:
        os.environ["CENTURION_MAX_RUNTIME_SECS"] = str(_KAGGLE_MAX_RUNTIME_SECS)
        print(f"  Kaggle mode: will checkpoint at {_KAGGLE_MAX_RUNTIME_SECS // 3600}h "
              f"{(_KAGGLE_MAX_RUNTIME_SECS % 3600) // 60}m (12h session limit)")

    # Force MULTI_ASSET_ENABLED so gold tickers are added
    from config import Config
    Config.MULTI_ASSET_ENABLED = True
    Config.NSE_UNIVERSE_TIER = "NIFTY500"
    Config.PIT_UNIVERSE_ENABLED = False  # R24v5: PIT loads 697 stocks, not needed for extraction

    import services.full_pipeline_backtest as bt_mod

    # Enable forecast extraction
    bt_mod._SAVE_FORECASTS_MODE = True
    bt_mod._forecast_log.clear()

    # R24 FIX: Set R21A regime flags (were missing — defaulted to module init
    # values which may differ from the calibrated R21A settings)
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25
    bt_mod._R21A_REGIME_DEFEND = 0.55

    # ── Ticker list: reuse exact same list across sessions ──
    # If we have a saved ticker list from a previous session, use it.
    # This ensures n_symbols matches the checkpoint exactly.
    saved_tickers = None
    if os.path.exists(_TICKER_LIST_PATH):
        with open(_TICKER_LIST_PATH) as f:
            saved_tickers = json.load(f)
        print(f"  Restored ticker list: {len(saved_tickers)} symbols (from previous session)")
    else:
        # First session: fetch NIFTY500 and save the list for future sessions
        print("  Fetching NIFTY500 ticker list (first session)...")
        from kite_connect.nse.nse_universe import fetch_nse_symbols_nifty500
        saved_tickers = fetch_nse_symbols_nifty500()
        if not saved_tickers:
            print("  WARNING: NIFTY500 fetch failed, falling back to DEFAULT")
            from kite_connect.nse.nse_universe import get_nse_default_tickers
            saved_tickers = get_nse_default_tickers()
        # Append .NS suffix for yfinance
        saved_tickers = [s + ".NS" if not s.endswith(".NS") else s for s in saved_tickers]
        # Add multi-asset tickers (gold ETFs + crypto)
        for extra in ["GOLDBEES.NS", "GOLDIETF.NS", "CPSEETF.NS", "LIQUIDBEES.NS", "BTC-USD"]:
            if extra not in saved_tickers:
                saved_tickers.append(extra)
        # Save for future sessions
        with open(_TICKER_LIST_PATH, 'w') as f:
            json.dump(saved_tickers, f)
        print(f"  Saved ticker list: {len(saved_tickers)} symbols")

    print("\n  Starting NIFTY500 extraction...\n", flush=True)

    result = bt_mod.run_full_backtest(
        tickers=saved_tickers,  # None = auto-fetch, or reuse saved list
        capital=500_000,
        period="13y",
        market="IND",
        annual_vol_target=0.20,  # R24 FIX: explicit (was missing, now respected)
        verbose=True,
        start_date="2012-01-01",
        end_date="2025-12-31",
    )

    # ── Save ticker list on first run (before any checkpoint exists) ──
    # Already done above — ticker list saved before backtest starts

    # Save extracted forecasts
    log = bt_mod._forecast_log
    elapsed_hrs = (time.time() - t0) / 3600

    if not log:
        print(f"\n  WARNING: No forecast data collected (may have resumed from late checkpoint)")
        print(f"  Checking checkpoint for forecast_log...", flush=True)
        if os.path.exists(_CHECKPOINT_PATH):
            with open(_CHECKPOINT_PATH, 'rb') as f:
                ckpt = pickle.load(f)
            if 'forecast_log' in ckpt:
                log = ckpt['forecast_log']
                print(f"  Found {len(log)} snapshots in checkpoint", flush=True)

    print(f"\n  Extracted {len(log)} day-snapshots in {elapsed_hrs:.1f} hours")

    if log:
        # Analyze signal coverage
        all_signals = set()
        for _, _, fc_snap, _, _ in log:
            for sym_fc in fc_snap.values():
                all_signals.update(sym_fc.keys())
        print(f"  Signals found: {len(all_signals)}")
        for s in sorted(all_signals):
            count = sum(1 for _, _, snap, _, _ in log
                        for sym_fc in snap.values() if s in sym_fc)
            print(f"    {s:30s} {count:>8d} data points")

        payload = {
            "log": log,
            "extraction_result": {
                k: result.get(k)
                for k in ["sharpe", "sortino", "calmar", "max_drawdown_pct",
                           "annual_return_pct", "total_return_pct", "n_trades"]
            },
            "universe": "NIFTY500",
            "n_signals": len(all_signals),
        }
        with open(_OUTPUT_PATH, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        sz_mb = os.path.getsize(_OUTPUT_PATH) / 1e6
        print(f"  Saved: {_OUTPUT_PATH} ({sz_mb:.1f} MB)")

        # Copy to Kaggle download location
        if IN_KAGGLE:
            dl = os.path.join("/kaggle/working", "extracted_forecasts_nifty500.pkl")
            shutil.copy2(_OUTPUT_PATH, dl)
            print(f"  Download: {dl}")

            # Also save checkpoint for dataset update
            if os.path.exists(_CHECKPOINT_PATH):
                dl_ckpt = os.path.join("/kaggle/working", "backtest_checkpoint_nifty500_extract.pkl")
                shutil.copy2(_CHECKPOINT_PATH, dl_ckpt)
                print(f"  Checkpoint: {dl_ckpt}")

    if result:
        print(f"\n  Backtest: Sharpe={result.get('sharpe', 0):.3f}  "
              f"CAGR={result.get('annual_return_pct', 0):.1f}%  "
              f"MaxDD={result.get('max_drawdown_pct', 0):.1f}%")

    print(f"\n  DONE in {elapsed_hrs:.1f} hours.", flush=True)


if __name__ == "__main__":
    main()
