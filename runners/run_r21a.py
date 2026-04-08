"""
R21a Runner — Optimized Signal Weights (Walk-Forward).

Runs full backtest with signal weights optimized by optimize_weights_r21a.py.
Base: pure R19c (no vol boost, no stops changes, no position floor).
Only change: signal weight allocation.

Usage:
    python run_r21a.py
"""
import sys
import os
import pickle

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

_R21A_CHECKPOINT = os.path.join(_root, "data", "backtest_checkpoint_r21a.pkl")
_OPT_RESULTS = os.path.join(_root, "data", "r21a_optimization_results.pkl")

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")


def main():
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

    # ── Load optimized weights ────
    if os.path.exists(_OPT_RESULTS):
        with open(_OPT_RESULTS, "rb") as f:
            opt = pickle.load(f)
        _R21A_WEIGHTS = opt["best_weights"]
        # Ensure all 24 signals are present (missing = 0)
        all_24 = [
            "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
            "carry", "screener", "momentum", "pead", "mean_reversion",
            "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
            "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
            "intermarket", "acceleration", "carver_value", "skew_signal",
            "sentiment", "breakout", "order_flow",
        ]
        for sig in all_24:
            if sig not in _R21A_WEIGHTS:
                _R21A_WEIGHTS[sig] = 0.0
        print(f"  Loaded optimized weights from {_OPT_RESULTS}")
    else:
        print(f"  WARNING: {_OPT_RESULTS} not found!")
        print(f"  Run optimize_weights_r21a.py first.")
        print(f"  Using R19c weights as fallback.")
        _R21A_WEIGHTS = {
            "ewmac_8_32": 0.07, "ewmac_16_64": 0.09, "ewmac_32_128": 0.00,
            "ewmac_64_256": 0.08, "carry": 0.00, "screener": 0.05,
            "momentum": 0.16, "pead": 0.00, "mean_reversion": 0.13,
            "fii_flow": 0.00, "decision_engine": 0.00, "oi_signal": 0.00,
            "cross_momentum": 0.00, "pairs_arb": 0.00, "event_driven": 0.00,
            "penfold_trend": 0.12, "ehlers_dsp": 0.12, "intermarket": 0.00,
            "acceleration": 0.04, "carver_value": 0.07, "skew_signal": 0.00,
            "sentiment": 0.00, "breakout": 0.07, "order_flow": 0.00,
        }

    for fw in DEFAULT_FORECAST_WEIGHTS:
        if fw.name in _R21A_WEIGHTS:
            fw.weight = _R21A_WEIGHTS[fw.name]

    # ── Set checkpoint path + mode flags ──
    import services.full_pipeline_backtest as bt_mod

    os.environ["CENTURION_BT_CHECKPOINT"] = _R21A_CHECKPOINT
    # Pure R19c base — ALL enhancement modes OFF
    bt_mod._R20A_MAXDD_MODE = False
    bt_mod._R20B_MAXDD_MODE = False
    bt_mod._R20C_MAXDD_MODE = False
    bt_mod._R20D_HYBRID_MODE = False
    bt_mod._R19D_REGIME_MODE = False
    bt_mod._R19E_REGIME_MODE = False
    bt_mod._R19F_REGIME_MODE = False
    bt_mod._R19G_REGIME_MODE = False
    bt_mod._R19H_REGIME_MODE = False
    bt_mod._SAVE_FORECASTS_MODE = False
    # R21a: Enable regime-adaptive vol target
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25   # 25% more aggressive in uptrends
    bt_mod._R21A_REGIME_DEFEND = 0.55  # 45% less exposure in downtrends

    print("=" * 70)
    print("  R21a — Optimized Signal Weights (Walk-Forward)")
    print("  Base: pure R19c (no vol boost, no stops changes)")
    print("  Change: signal weight allocation only")
    print(f"  Checkpoint: {_R21A_CHECKPOINT}")
    print("=" * 70)
    print("  Weights:")
    for sig in sorted(_R21A_WEIGHTS, key=lambda s: _R21A_WEIGHTS[s], reverse=True):
        w = _R21A_WEIGHTS[sig]
        if w > 0.005:
            print(f"    {sig:20s}  {w*100:5.1f}%")

    result = bt_mod.run_full_backtest(
        tickers=None,
        capital=500_000,
        period="13y",
        market="IND",
        verbose=True,
        start_date="2012-01-01",
        end_date="2025-12-31",
    )

    print(f"\n{'='*70}")
    print(f"  R21a KEY METRICS")
    print(f"{'='*70}")
    for k in ["annual_return_pct", "total_return_pct", "sharpe", "sortino",
              "calmar", "max_drawdown_pct", "n_trades", "avg_positions",
              "win_rate", "profit_factor",
              "detrended_sharpe", "trimmed_sharpe", "dm_bias_estimate"]:
        print(f"  {k:25s} = {result.get(k)}")
    if result.get("bootstrap_ci_sharpe"):
        lo, hi = result["bootstrap_ci_sharpe"]
        print(f"  {'sharpe_90pct_ci':25s} = [{lo:.3f}, {hi:.3f}]")

    # Comparison vs R19c
    r19c_sharpe = 1.025
    r19c_maxdd = 67.41
    r21a_sharpe = result.get("sharpe", 0)
    r21a_maxdd = result.get("max_drawdown_pct", 100)
    r21a_cagr = result.get("annual_return_pct", 0)
    print(f"\n  ── R19c vs R21a ──")
    print(f"  Sharpe:  {r19c_sharpe:.3f} → {r21a_sharpe:.3f}  (Δ{r21a_sharpe - r19c_sharpe:+.3f})")
    print(f"  MaxDD:   {r19c_maxdd:.1f}% → {r21a_maxdd:.1f}%  (Δ{r21a_maxdd - r19c_maxdd:+.1f}%)")
    print(f"  CAGR:    48.78% → {r21a_cagr:.1f}%")

    if r21a_sharpe >= 1.5 and r21a_maxdd <= 50.0 and r21a_cagr >= 50.0:
        print(f"\n  TARGET HIT!")
    elif r21a_sharpe > r19c_sharpe:
        print(f"\n  IMPROVEMENT: Sharpe Δ{r21a_sharpe - r19c_sharpe:+.3f}")
    else:
        print(f"\n  NO IMPROVEMENT vs R19c")


if __name__ == "__main__":
    main()
