"""
R21a Phase 1 — Extract per-source forecasts for weight optimization.

Runs a full R19c-config backtest with _SAVE_FORECASTS_MODE=True.
Saves daily per-source per-symbol forecasts + close prices + vols
to data/extracted_forecasts.pkl (~5 hours).

The optimizer (optimize_weights_r21a.py) loads this and tests thousands
of weight combos in seconds without re-running the backtest.

Usage:
    python run_extract_forecasts.py
"""
import sys
import os
import pickle

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

_OUTPUT_PATH = os.path.join(_root, "data", "extracted_forecasts.pkl")
_CHECKPOINT = os.path.join(_root, "data", "backtest_checkpoint_extract.pkl")


def main():
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

    # ── R19c signal weights (baseline) ────
    _R19C_WEIGHTS = {
        "ewmac_8_32":       0.07,
        "ewmac_16_64":      0.09,
        "ewmac_32_128":     0.00,
        "ewmac_64_256":     0.08,
        "carry":            0.00,
        "screener":         0.05,
        "momentum":         0.16,
        "pead":             0.00,
        "mean_reversion":   0.13,
        "fii_flow":         0.00,
        "decision_engine":  0.00,
        "oi_signal":        0.00,
        "cross_momentum":   0.00,
        "pairs_arb":        0.00,
        "event_driven":     0.00,
        "penfold_trend":    0.12,
        "ehlers_dsp":       0.12,
        "intermarket":      0.00,
        "acceleration":     0.04,
        "carver_value":     0.07,
        "skew_signal":      0.00,
        "sentiment":        0.00,
        "breakout":         0.07,
        "order_flow":       0.00,
    }
    for fw in DEFAULT_FORECAST_WEIGHTS:
        if fw.name in _R19C_WEIGHTS:
            fw.weight = _R19C_WEIGHTS[fw.name]

    # ── Enable forecast extraction ────
    import services.full_pipeline_backtest as bt_mod

    os.environ["CENTURION_BT_CHECKPOINT"] = _CHECKPOINT
    bt_mod._SAVE_FORECASTS_MODE = True
    # Pure R19c — all R20 modes OFF
    bt_mod._R20A_MAXDD_MODE = False
    bt_mod._R20B_MAXDD_MODE = False
    bt_mod._R20C_MAXDD_MODE = False
    bt_mod._R20D_HYBRID_MODE = False
    bt_mod._R19D_REGIME_MODE = False
    bt_mod._R19E_REGIME_MODE = False
    bt_mod._R19F_REGIME_MODE = False
    bt_mod._R19G_REGIME_MODE = False
    bt_mod._R19H_REGIME_MODE = False
    # Clear any prior log
    bt_mod._forecast_log.clear()

    print("=" * 70)
    print("  R21a Phase 1 — Forecast Extraction")
    print("  Config: R19c weights (baseline)")
    print("  Saving per-source forecasts + prices + vols")
    print(f"  Output: {_OUTPUT_PATH}")
    print("=" * 70)

    result = bt_mod.run_full_backtest(
        tickers=None,
        capital=500_000,
        period="13y",
        market="IND",
        verbose=True,
        start_date="2012-01-01",
        end_date="2025-12-31",
    )

    # Save extracted forecasts
    log = bt_mod._forecast_log
    print(f"\n  Extracted {len(log)} day-snapshots")

    payload = {
        "log": log,  # list of (day_idx, date_str, {sym: {src: val}}, {sym: price}, {sym: vol})
        "r19c_result": {
            k: result.get(k)
            for k in ["sharpe", "sortino", "calmar", "max_drawdown_pct",
                       "annual_return_pct", "total_return_pct", "n_trades"]
        },
    }
    with open(_OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    sz_mb = os.path.getsize(_OUTPUT_PATH) / 1e6
    print(f"  Saved to {_OUTPUT_PATH} ({sz_mb:.1f} MB)")
    print(f"  R19c baseline: Sharpe={result.get('sharpe'):.3f}  CAGR={result.get('annual_return_pct'):.1f}%")


if __name__ == "__main__":
    main()
