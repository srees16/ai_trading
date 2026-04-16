"""R24: Run full pipeline backtest with R21A-calibrated config."""
import os
import pickle
from services.full_pipeline_backtest import run_full_backtest
import services.full_pipeline_backtest as bt_mod
from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS
from config import Config

# R24 FIX: Use same setup as task_r21a() in run_kaggle.py
# Previously used Config.CARVER_ANNUAL_VOL_TARGET (0.40) and no optimized weights
_VOL_TARGET = 0.20  # R21A calibrated (was 0.40 → 2x oversizing)

# R24: Set regime flags explicitly (matching R21A)
bt_mod._R21A_REGIME_VOL = True
bt_mod._R21A_REGIME_BOOST = 1.25
bt_mod._R21A_REGIME_DEFEND = 0.55

# R24v5-RCA: Match R21A EXACTLY — DEFAULT tier (~95 NIFTY50+NEXT50 stocks)
# R21A achieved Sharpe=1.18, CAGR=32.7% on this universe. NIFTY500/BROAD failed
# because 80%+ of stocks are classified as smallcap (61bps/leg cost), destroying alpha.
Config.NSE_UNIVERSE_TIER = "DEFAULT"
Config.PIT_UNIVERSE_ENABLED = False  # R21A never used PIT universe

# R24: Load optimized weights if available
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_opt_path = os.path.join(_CORE_DIR, "data", "r21a_optimization_results.pkl")
if os.path.exists(_opt_path):
    with open(_opt_path, "rb") as f:
        _opt = pickle.load(f)
    _weights = _opt.get("best_weights", {})
    # R24v3: Only apply optimized weights if the optimization has >= 5 signals.
    # A partial optimization (3 signals) zeros out ALL other active signals,
    # leaving combined forecast near-zero → only 3 positions → concentration death.
    if len(_weights) >= 5:
        for fw in DEFAULT_FORECAST_WEIGHTS:
            if fw.name in _weights:
                fw.weight = _weights[fw.name]
            elif fw.weight > 0:
                fw.weight = 0.0
        print(f"Loaded R21A optimized weights ({len(_weights)} signals) from {_opt_path}")
    else:
        print(f"R21A weights at {_opt_path} only has {len(_weights)} signals — using DEFAULT_FORECAST_WEIGHTS")
else:
    print(f"No R21A weights found at {_opt_path} — using DEFAULT_FORECAST_WEIGHTS")

print("=== R24 Production Config for Backtest ===")
print(f"Vol target: {_VOL_TARGET*100:.0f}%")
print(f"Max leverage: {Config.CARVER_MAX_LEVERAGE}x")
print(f"Capital: Rs{Config.CARVER_INITIAL_CAPITAL:,.0f}")
print()

result = run_full_backtest(
    tickers=None,
    capital=Config.CARVER_INITIAL_CAPITAL,
    period="5y",
    market="IND",
    annual_vol_target=_VOL_TARGET,
    verbose=True,
    start_date=getattr(Config, "BACKTEST_START_DATE", ""),
    end_date=getattr(Config, "BACKTEST_END_DATE", ""),
)

print()
print("=" * 60)
print("  KEY PERFORMANCE METRICS (Production Config)")
print("=" * 60)
metrics = [
    "annual_return_pct", "sharpe", "sortino",
    "max_drawdown_pct", "calmar", "n_trades", "total_return_pct",
]
for k in metrics:
    v = result.get(k, "N/A")
    if isinstance(v, float):
        print(f"  {k:25s}: {v:.2f}")
    else:
        print(f"  {k:25s}: {v}")

cagr = result.get("annual_return_pct", 0)
if cagr >= 50:
    print(f"\n  ✅ CAGR {cagr:.1f}% EXCEEDS 50% TARGET")
else:
    print(f"\n  ⚠️ CAGR {cagr:.1f}% below 50% target — gap: {50 - cagr:.1f}%")
