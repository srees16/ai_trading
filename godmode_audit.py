"""GODMODE Audit: IND + US backtests with regime breakdown & expanded universe."""
import sys, io, os, warnings, json, time
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')

import numpy as np

sys.stderr = io.StringIO()
from services.full_pipeline_backtest import run_full_backtest
sys.stderr = sys.__stderr__

# ── IND Backtest — Expanded universe across sectors ───────────
# Broad Market (NIFTY50 large-cap) + Sectoral coverage + Midcap alpha
ind_tickers = [
    # Large-cap blue chips (NIFTY 50)
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS', 'TATAMOTORS.NS',
    'AXISBANK.NS', 'WIPRO.NS', 'SUNPHARMA.NS', 'MARUTI.NS', 'ONGC.NS',
    # Sectoral coverage (Auto, Metal, Energy, FMCG, IT, Pharma, Realty, PSU Bank)
    'M&M.NS', 'TATASTEEL.NS', 'HINDALCO.NS', 'NTPC.NS', 'POWERGRID.NS',
    'HINDUNILVR.NS', 'NESTLEIND.NS', 'HCLTECH.NS', 'DRREDDY.NS', 'CIPLA.NS',
    'DLF.NS', 'BANKBARODA.NS', 'COALINDIA.NS', 'BRITANNIA.NS', 'JSWSTEEL.NS',
    # Midcap alpha generators
    'TATAPOWER.NS', 'HAL.NS', 'CHOLAFIN.NS', 'POLYCAB.NS', 'ZOMATO.NS',
    'PIDILITIND.NS', 'ABB.NS', 'SIEMENS.NS', 'GODREJCP.NS', 'JINDALSTEL.NS',
]

print("=" * 80)
print("  GODMODE AUDIT — IND STOCKS (5Y, 40 tickers)")
print("=" * 80)
t0 = time.time()
sys.stderr = io.StringIO()
ind = run_full_backtest(
    tickers=ind_tickers, capital=1_000_000, period='5y', market='IND',
    annual_vol_target=0.85, include_carry=False, include_pairs=True,
    verbose=True,
)
sys.stderr = sys.__stderr__
print(f"  [IND backtest completed in {time.time()-t0:.1f}s]")

# ── Source Hit Rates (IND) ────────────────────────────────────
if 'source_hit_rates' in ind:
    print("\n  Source Hit Rates (IND):")
    for src, rate in sorted(ind['source_hit_rates'].items(), key=lambda x: -x[1]):
        print(f"    {src:25s}: {rate*100:5.1f}%")

# ── US Backtest — Expanded to NASDAQ + S&P diversified ────────
us_tickers = [
    # Tech mega-cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Healthcare & Consumer
    "JNJ", "UNH", "PG", "KO", "PEP", "COST", "HD",
    # Financials & Energy
    "JPM", "V", "MA", "BAC", "GS", "XOM", "CVX",
    # Industrials & Communication
    "BA", "CAT", "HON", "DIS", "NFLX", "CMCSA",
    # Growth / Midcap
    "CRM", "ADBE", "AMD", "AVGO", "ISRG", "PANW", "CRWD",
    "SQ", "SHOP", "ABNB",
]

print("\n\n" + "=" * 80)
print("  GODMODE AUDIT — US STOCKS (5Y, 37 tickers)")
print("=" * 80)
t0 = time.time()
sys.stderr = io.StringIO()
us = run_full_backtest(
    tickers=us_tickers, capital=10_000, period='5y', market='US',
    annual_vol_target=0.20, include_carry=True, include_pairs=True,
    verbose=True,
)
sys.stderr = sys.__stderr__
print(f"  [US backtest completed in {time.time()-t0:.1f}s]")

if 'source_hit_rates' in us:
    print("\n  Source Hit Rates (US):")
    for src, rate in sorted(us['source_hit_rates'].items(), key=lambda x: -x[1]):
        print(f"    {src:25s}: {rate*100:5.1f}%")

# ── Regime Breakdown for IND ──────────────────────────────────
print("\n\n" + "=" * 80)
print("  REGIME BREAKDOWN — IND 5Y")
print("=" * 80)

eq = np.array(ind['daily_equity'])
daily_rets = np.diff(eq) / eq[:-1]
n = len(daily_rets)

# Classify each day by 40-day rolling return of the equity curve itself
regimes = []
for i in range(n):
    if i < 40:
        regimes.append('warmup')
    else:
        ret_40d = (eq[i+1] / eq[i+1-40]) - 1
        if ret_40d > 0.08:
            regimes.append('bull')
        elif ret_40d < -0.08:
            regimes.append('bear')
        else:
            regimes.append('sideways')

for regime in ['bull', 'bear', 'sideways']:
    mask = [r == regime for r in regimes]
    rets = daily_rets[mask]
    if len(rets) < 10:
        print(f"  {regime:10s}  days={len(rets):4d}  (insufficient data)")
        continue
    ann_ret = float(np.mean(rets)) * 252
    ann_vol = float(np.std(rets, ddof=1)) * np.sqrt(252)
    sr = ann_ret / ann_vol if ann_vol > 0 else 0
    win_pct = float(np.mean(rets > 0)) * 100
    worst = float(np.min(rets)) * 100
    best = float(np.max(rets)) * 100
    cum = float(np.prod(1 + rets) - 1) * 100
    print(f"  {regime:10s}  days={len(rets):4d}  cum_ret={cum:+7.1f}%  ann_vol={ann_vol*100:5.1f}%  SR={sr:+.2f}  win={win_pct:.0f}%  worst_day={worst:+.2f}%  best_day={best:+.2f}%")

# ── Drawdown Analysis ─────────────────────────────────────────
print("\n  Drawdown Analysis:")
peak = np.maximum.accumulate(eq)
dd = (peak - eq) / peak * 100
# Find top 3 drawdowns
dd_max = float(np.max(dd))
dd_periods = []
in_dd = False
start = 0
for i in range(len(dd)):
    if dd[i] > 5 and not in_dd:
        in_dd = True
        start = i
    elif dd[i] < 1 and in_dd:
        dd_periods.append((start, i, float(np.max(dd[start:i]))))
        in_dd = False
if in_dd:
    dd_periods.append((start, len(dd)-1, float(np.max(dd[start:]))))
dd_periods.sort(key=lambda x: -x[2])
for j, (s, e, d) in enumerate(dd_periods[:5]):
    dur = e - s
    print(f"  DD#{j+1}: -{d:.1f}%  duration={dur} days  (day {s}-{e})")

# ── Alpha Analysis ────────────────────────────────────────────
print(f"\n  Alpha Metrics IND:")
print(f"    CAGR:          {ind['annual_return_pct']:+.1f}%")
print(f"    Sharpe:        {ind['sharpe']:.3f}")
print(f"    Sortino:       {ind['sortino']:.3f}")
print(f"    Calmar:        {ind['calmar']:.3f}")
# Simple alpha vs NIFTY (assume ~12% annual)
nifty_cagr = 12.0
alpha_vs_nifty = ind['annual_return_pct'] - nifty_cagr
print(f"    Alpha vs NIFTY: +{alpha_vs_nifty:.1f}% (NIFTY ~12%)")

print(f"\n  Alpha Metrics US:")
print(f"    CAGR:          {us['annual_return_pct']:+.1f}%")
print(f"    Sharpe:        {us['sharpe']:.3f}")
print(f"    Sortino:       {us['sortino']:.3f}")
print(f"    Calmar:        {us['calmar']:.3f}")
sp500_cagr = 10.0
alpha_vs_sp = us['annual_return_pct'] - sp500_cagr
print(f"    Alpha vs S&P500: {alpha_vs_sp:+.1f}% (S&P ~10%)")
