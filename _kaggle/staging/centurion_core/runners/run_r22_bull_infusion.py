"""
R22 Runner — Centurion Compounder + Bull-Run Capital Infusion.

Runs R21A backtest with R22 bull-run capital infusion enabled.
Simulates injecting fresh capital (default ₹50,000) when a confirmed
bear→bull regime transition is detected (equity crosses above SMA200+2%
for 5 consecutive days after being in bear territory).

This is an OPTIONAL enhancement — compounding works identically without
infusion. The infusion simply amplifies returns during bull phases.

At the end, prints a side-by-side comparison of R22 vs R21A metrics.

Usage:
    python run_r22_bull_infusion.py
    python run_r22_bull_infusion.py --amount 100000
    python run_r22_bull_infusion.py --no-infuse   # alerts only, no capital added
"""
import sys
import os
import pickle
import argparse

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

_R22_CHECKPOINT = os.path.join(_root, "data", "backtest_checkpoint_r22.pkl")
_OPT_RESULTS = os.path.join(_root, "data", "r21a_optimization_results.pkl")

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")


def main(infusion_amount: float = 50_000.0, infuse: bool = True):
    from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS
    import services.full_pipeline_backtest as bt_mod

    # ── Load optimized weights ────
    if os.path.exists(_OPT_RESULTS):
        with open(_OPT_RESULTS, "rb") as f:
            opt = pickle.load(f)
        weights = opt["best_weights"]
        all_24 = [
            "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
            "carry", "screener", "momentum", "pead", "mean_reversion",
            "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
            "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
            "intermarket", "acceleration", "carver_value", "skew_signal",
            "sentiment", "breakout", "order_flow",
        ]
        for sig in all_24:
            if sig not in weights:
                weights[sig] = 0.0
        print(f"  Loaded optimized weights from {_OPT_RESULTS}")
    else:
        print(f"  WARNING: {_OPT_RESULTS} not found — using DEFAULT_FORECAST_WEIGHTS")
        weights = {fw.name: fw.weight for fw in DEFAULT_FORECAST_WEIGHTS}

    for fw in DEFAULT_FORECAST_WEIGHTS:
        if fw.name in weights:
            fw.weight = weights[fw.name]

    # ── Configure R22 ────
    os.environ["CENTURION_BT_CHECKPOINT"] = _R22_CHECKPOINT
    bt_mod._SAVE_FORECASTS_MODE = False
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25
    bt_mod._R21A_REGIME_DEFEND = 0.55

    # R22-specific flags
    bt_mod._R22_BULL_INFUSION = infuse
    bt_mod._R22_INFUSION_AMOUNT = infusion_amount
    bt_mod._R22_INFUSION_COOLDOWN_DAYS = 200
    bt_mod._R22_BULL_CONFIRM_DAYS = 5

    # Ensure Harvest is OFF (this is Compounder-only)
    bt_mod._HARVEST_ENABLED = False
    bt_mod._HARVEST_DIP_BUYER = False
    bt_mod._HARVEST_PROFIT_TAKER = False

    _mode = f"INFUSE ₹{infusion_amount:,.0f}" if infuse else "ALERTS ONLY"
    print("=" * 70)
    print(f"  R22 — Centurion Compounder + Bull-Run Capital Infusion")
    print(f"  Mode: {_mode}")
    print(f"  Cooldown: {bt_mod._R22_INFUSION_COOLDOWN_DAYS} days")
    print(f"  Bull Confirm: {bt_mod._R22_BULL_CONFIRM_DAYS} consecutive days above SMA200+2%")
    print(f"  Checkpoint: {_R22_CHECKPOINT}")
    print("=" * 70)

    # ── Run R22 backtest ────
    result_r22 = bt_mod.run_full_backtest(
        tickers=None,
        capital=500_000,
        period="13y",
        market="IND",
        verbose=True,
        start_date="2012-01-01",
        end_date="2025-12-31",
    )

    # ── Print R22 key metrics ────
    print(f"\n{'='*70}")
    print(f"  R22 KEY METRICS")
    print(f"{'='*70}")
    for k in ["annual_return_pct", "total_return_pct", "sharpe", "sortino",
              "calmar", "max_drawdown_pct", "n_trades", "avg_positions",
              "win_rate", "profit_factor",
              "detrended_sharpe", "trimmed_sharpe", "dm_bias_estimate"]:
        print(f"  {k:25s} = {result_r22.get(k)}")
    if result_r22.get("bootstrap_ci_sharpe"):
        lo, hi = result_r22["bootstrap_ci_sharpe"]
        print(f"  {'sharpe_90pct_ci':25s} = [{lo:.3f}, {hi:.3f}]")

    # ── R22 infusion summary ────
    r22_data = result_r22.get("r22_bull_infusion")
    if r22_data:
        print(f"\n  {'─'*40}")
        print(f"  R22 Bull-Run Infusion Summary:")
        print(f"  Bull Alerts:     {r22_data['n_alerts']}")
        print(f"  Infusions Made:  {r22_data['n_infusions']}")
        print(f"  Total Infused:   ₹{r22_data['total_infused']:,.0f}")

    # ── R21A OOS benchmark comparison ────
    r21a_oos = {"sharpe": 2.093, "max_drawdown_pct": 25.2, "annual_return_pct": 74.1,
                "sortino": 3.2, "calmar": 2.937}
    r22_sharpe = result_r22.get("sharpe", 0)
    r22_maxdd = result_r22.get("max_drawdown_pct", 100)
    r22_cagr = result_r22.get("annual_return_pct", 0)
    r22_sortino = result_r22.get("sortino", 0)
    r22_calmar = result_r22.get("calmar", 0)

    print(f"\n{'='*70}")
    print(f"  R22 vs R21A COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':20s}  {'R21A OOS':>12s}  {'R22':>12s}  {'Delta':>12s}")
    print(f"  {'─'*60}")
    _metrics = [
        ("Sharpe", r21a_oos["sharpe"], r22_sharpe),
        ("Sortino", r21a_oos["sortino"], r22_sortino),
        ("Calmar", r21a_oos["calmar"], r22_calmar),
        ("CAGR (%)", r21a_oos["annual_return_pct"], r22_cagr),
        ("MaxDD (%)", r21a_oos["max_drawdown_pct"], r22_maxdd),
    ]
    for name, r21a_val, r22_val in _metrics:
        delta = r22_val - r21a_val
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        # For MaxDD, lower is better
        if name == "MaxDD (%)":
            arrow = "▲" if delta < 0 else ("▼" if delta > 0 else "─")
        print(f"  {name:20s}  {r21a_val:>12.3f}  {r22_val:>12.3f}  {arrow} {delta:+.3f}")

    # Verdict
    if r22_sharpe > r21a_oos["sharpe"] and r22_maxdd <= r21a_oos["max_drawdown_pct"] * 1.1:
        print(f"\n  ★ R22 IMPROVES on R21A baseline")
    elif r22_sharpe >= r21a_oos["sharpe"] * 0.95:
        print(f"\n  ≈ R22 ON PAR with R21A (within 5%)")
    else:
        print(f"\n  ✗ R22 BELOW R21A — infusion timing may need tuning")

    return result_r22


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R22: Centurion Compounder + Bull-Run Capital Infusion")
    parser.add_argument("--amount", type=float, default=50_000.0,
                        help="₹ amount to infuse per bull-run event (default: 50000)")
    parser.add_argument("--no-infuse", action="store_true",
                        help="Generate alerts only, do NOT infuse capital")
    args = parser.parse_args()
    main(infusion_amount=args.amount, infuse=not args.no_infuse)
