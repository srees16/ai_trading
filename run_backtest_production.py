"""FIX-6: Run full pipeline backtest with PRODUCTION config."""
from services.full_pipeline_backtest import run_full_backtest
from config import Config

print("=== Production Config for Backtest ===")
print(f"Vol target: {Config.CARVER_ANNUAL_VOL_TARGET*100:.0f}%")
print(f"Max leverage: {Config.CARVER_MAX_LEVERAGE}x")
print(f"IDM: {Config.CARVER_DEFAULT_IDM}")
print(f"Capital: Rs{Config.CARVER_INITIAL_CAPITAL:,.0f}")
print()

result = run_full_backtest(
    tickers=None,
    capital=Config.CARVER_INITIAL_CAPITAL,
    period="5y",
    market="IND",
    annual_vol_target=Config.CARVER_ANNUAL_VOL_TARGET,
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
