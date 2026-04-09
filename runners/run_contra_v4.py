"""Run Centurion Harvest backtest locally."""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.dirname(_SCRIPT_DIR)  # centurion_core/
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

import services.full_pipeline_backtest as bt
import pickle, time
from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

# Set all modes off first
bt._R19D_REGIME_MODE = False
bt._R19E_REGIME_MODE = False
bt._R19F_REGIME_MODE = False
bt._R19G_REGIME_MODE = False
bt._R19H_REGIME_MODE = False
bt._R20A_MAXDD_MODE = False
bt._R20B_MAXDD_MODE = False
bt._R20C_MAXDD_MODE = False
bt._R20D_HYBRID_MODE = False
bt._SAVE_FORECASTS_MODE = False

# Enable R21A base
bt._R21A_REGIME_VOL = True
bt._R21A_REGIME_BOOST = 1.25
bt._R21A_REGIME_DEFEND = 0.55

# Enable Centurion Harvest (all flags)
bt._HARVEST_DIP_BUYER = True
bt._HARVEST_PROFIT_TAKER = True
bt._HARVEST_ENABLED = True

# Load optimized weights if available
opt_path = os.path.join("data", "r21a_optimization_results.pkl")
all_24 = [
    "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
    "carry", "screener", "momentum", "pead", "mean_reversion",
    "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
    "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
    "intermarket", "acceleration", "carver_value", "skew_signal",
    "sentiment", "breakout", "order_flow",
]
if os.path.exists(opt_path):
    with open(opt_path, "rb") as f:
        opt = pickle.load(f)
    weights = opt["best_weights"]
    print(f"Loaded optimized weights from {opt_path}")
else:
    weights = {
        "ewmac_8_32": 0.07, "ewmac_16_64": 0.09, "ewmac_64_256": 0.08,
        "screener": 0.05, "momentum": 0.16, "mean_reversion": 0.13,
        "penfold_trend": 0.12, "ehlers_dsp": 0.12, "acceleration": 0.04,
        "carver_value": 0.07, "breakout": 0.07,
    }
    print("Using R19c fallback weights")

for sig in all_24:
    if sig not in weights:
        weights[sig] = 0.0
for fw in DEFAULT_FORECAST_WEIGHTS:
    if fw.name in weights:
        fw.weight = weights[fw.name]

# Delete checkpoint to start fresh
ckpt = os.path.join("data", "backtest_checkpoint_contra_v4.pkl")
os.environ["CENTURION_BT_CHECKPOINT"] = ckpt
if os.path.exists(ckpt):
    os.remove(ckpt)

print("\n=== CENTURION HARVEST: Capital Rotation + Dip-Buyer + Profit-Taker ===")
print(f"  DIP_BUYER={bt._HARVEST_DIP_BUYER}")
print(f"  PROFIT_TAKER={bt._HARVEST_PROFIT_TAKER}")
print(f"  HARVEST_ENABLED={bt._HARVEST_ENABLED}")
print(f"  INJECT_PCT={bt._HARVEST_INJECT_PCT}, BOOK_PCT={bt._HARVEST_BOOK_PCT}")
print()

t0 = time.time()
result = bt.run_full_backtest(
    tickers=None,
    capital=500_000,
    period="13y",
    market="IND",
    verbose=True,
    start_date="2012-01-01",
    end_date="2025-12-31",
)
elapsed = (time.time() - t0) / 60.0
print(f"\nCompleted in {elapsed:.1f} minutes")

# Print key comparison
print("\n" + "=" * 70)
print("  CENTURION HARVEST vs COMPOUNDER (R21A) BASELINE")
print("=" * 70)
R21A = {"sharpe": 2.093, "cagr": 74.1, "maxdd": 25.2, "calmar": 2.937}
s = result.get("sharpe", 0)
c = result.get("annual_return_pct", 0)
m = result.get("max_drawdown_pct", 100)
cal = result.get("calmar", 0)
print(f"  Sharpe:  {R21A['sharpe']:.3f} -> {s:.3f}  (d{s - R21A['sharpe']:+.3f})")
print(f"  CAGR:    {R21A['cagr']:.1f}% -> {c:.1f}%  (d{c - R21A['cagr']:+.1f}%)")
print(f"  MaxDD:   {R21A['maxdd']:.1f}% -> {m:.1f}%  (d{m - R21A['maxdd']:+.1f}%)")
print(f"  Calmar:  {R21A['calmar']:.3f} -> {cal:.3f}  (d{cal - R21A['calmar']:+.3f})")

ok = s >= 2.0 and m <= 30.0 and cal >= 2.5
verdict = "ACCEPT" if ok else "REJECT"
print(f"  Verdict: {verdict}")

cr = result.get("capital_rotation")
if cr:
    print(f"\n  Capital Rotation:")
    print(f"    Total Injected:  Rs {cr['total_injected']:,.0f}  ({len(cr['inject_events'])} events)")
    print(f"    Total Booked:    Rs {cr['total_booked']:,.0f}  ({len(cr['book_events'])} events)")
    print(f"    Net Added:       Rs {cr['net_added']:,.0f}")
    if cr["inject_events"]:
        print(f"    Injection Timeline:")
        for ev in cr["inject_events"]:
            print(f"      Day {ev[0]:4d}: +Rs {ev[1]:,.0f}  (equity Rs {ev[2]:,.0f} -> Rs {ev[3]:,.0f})")
    if cr["book_events"]:
        print(f"    Profit Booking Timeline:")
        for ev in cr["book_events"]:
            print(f"      Day {ev[0]:4d}: -Rs {ev[1]:,.0f}  (equity Rs {ev[2]:,.0f} -> Rs {ev[3]:,.0f})")
print("=" * 70)
