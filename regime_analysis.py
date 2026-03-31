"""Regime-specific backtest analysis for GODMODE audit."""
import sys, io, os, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import numpy as np
from services.full_pipeline_backtest import run_full_backtest


def regime_analysis(tickers, capital, period, market, label):
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    r = run_full_backtest(
        tickers=tickers, capital=capital, period=period, market=market,
        include_carry=False, include_pairs=False, verbose=False,
    )
    sys.stderr = old_stderr

    eq = np.array(r['daily_equity'])
    rets = np.diff(eq) / eq[:-1]
    n = len(rets)

    regime_rets = {'BULL': [], 'BEAR': [], 'SIDEWAYS': []}
    for i in range(60, n):
        r60 = eq[i + 1] / eq[max(1, i - 59)] - 1
        if r60 > 0.05:
            regime_rets['BULL'].append(rets[i])
        elif r60 < -0.05:
            regime_rets['BEAR'].append(rets[i])
        else:
            regime_rets['SIDEWAYS'].append(rets[i])

    print("=" * 65)
    print(f"  REGIME-SPECIFIC PERFORMANCE: {label}")
    print("=" * 65)
    for regime, rv in regime_rets.items():
        if not rv:
            print(f"  {regime:10s}  (no days classified)")
            continue
        ra = np.array(rv)
        ann_ret = np.mean(ra) * 252 * 100
        ann_vol = np.std(ra) * np.sqrt(252) * 100
        sr = np.mean(ra) / np.std(ra) * np.sqrt(252) if np.std(ra) > 0 else 0
        worst = np.min(ra) * 100
        win_rate = np.sum(ra > 0) / len(ra) * 100
        print(f"  {regime:10s}  days={len(ra):4d}  ann_ret={ann_ret:+7.1f}%  "
              f"ann_vol={ann_vol:5.1f}%  SR={sr:+.2f}  win%={win_rate:.0f}%  worst_day={worst:.1f}%")

    print("-" * 65)
    sharpe = r['sharpe']
    cagr = r['annual_return_pct']
    mdd = r['max_drawdown_pct']
    trades = r['n_trades']
    avg_pos = r['avg_positions']
    print(f"  OVERALL  SR={sharpe:.3f}  CAGR={cagr:+.1f}%  MaxDD={mdd:.1f}%  "
          f"trades={trades}  avg_pos={avg_pos}")
    print()
    return r


if __name__ == '__main__':
    # US 5Y
    regime_analysis(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V', 'UNH', 'XOM'],
        capital=10000, period='5y', market='US', label='US 5Y (10 tickers)'
    )

    # US 2Y
    regime_analysis(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V', 'UNH', 'XOM'],
        capital=10000, period='2y', market='US', label='US 2Y (10 tickers)'
    )

    # IND 2Y
    regime_analysis(
        tickers=['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
                 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS', 'TATAMOTORS.NS'],
        capital=1000000, period='2y', market='IND', label='IND 2Y (10 tickers)'
    )

    # IND 5Y
    regime_analysis(
        tickers=['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
                 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS', 'TATAMOTORS.NS'],
        capital=1000000, period='5y', market='IND', label='IND 5Y (10 tickers)'
    )
