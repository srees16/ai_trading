"""Analyze R21a checkpoint progress."""
import sys, os, pickle, math
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ckpt = os.path.join(_root, "data", "backtest_checkpoint_r21a.pkl")

if not os.path.exists(_ckpt):
    print(f"No checkpoint found at {_ckpt}")
    sys.exit(1)

with open(_ckpt, "rb") as f:
    ck = pickle.load(f)

eq = ck.get("equity_curve", [])
day = ck.get("day_idx", 0)
total = ck.get("total_days", 3190)
equity = eq[-1] if eq else ck.get("equity", 0)
capital = ck.get("capital", 500_000)

rets = np.diff(eq) / np.array(eq[:-1]) if len(eq) > 1 else np.array([])
sharpe = (np.mean(rets) / np.std(rets, ddof=1) * math.sqrt(252)) if len(rets) > 50 and np.std(rets) > 0 else 0.0
neg = rets[rets < 0]
downside = np.std(neg, ddof=1) if len(neg) > 10 else np.std(rets, ddof=1)
sortino = (np.mean(rets) / downside * math.sqrt(252)) if downside > 0 else 0.0

peak = np.maximum.accumulate(eq)
dd = (peak - eq) / peak * 100
maxdd = dd.max() if len(dd) > 0 else 0.0

years = len(rets) / 252.0
cagr = ((equity / capital) ** (1.0 / years) - 1.0) * 100 if years > 0 and equity > capital * 0.1 else 0.0

trades = ck.get("trades_count", 0)
idm = ck.get("idm", 0)
prev = ck.get("prev_positions", {})
active = sum(1 for v in prev.values() if v != 0)

pos_hist = ck.get("position_count_history", [])
avg_pos = np.mean(pos_hist) if pos_hist else active

trade_pnls = ck.get("trade_pnls", [])
if trade_pnls:
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    wr = len(wins) / len(trade_pnls) * 100
    avg_w = np.mean(wins) if wins else 0
    avg_l = np.mean(losses) if losses else 0
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    pf = gross_w / gross_l if gross_l > 0 else float('inf')
else:
    wr = avg_w = avg_l = pf = 0

daily_wins = np.sum(rets > 0) if len(rets) > 0 else 0
daily_total = len(rets)
daily_wr = daily_wins / daily_total * 100 if daily_total > 0 else 0
pos_rets = rets[rets > 0]
neg_rets = rets[rets < 0]
daily_pf = (pos_rets.sum() / abs(neg_rets.sum())) if len(neg_rets) > 0 and abs(neg_rets.sum()) > 0 else 0

print("=" * 50)
print("  R21a Backtest Progress")
print("=" * 50)
print(f"Day:       {day}/{total}")
print(f"Equity:    {equity:,.0f}")
print(f"CAGR:      {cagr:.1f}%")
print(f"Sharpe:    {sharpe:.3f}")
print(f"Sortino:   {sortino:.3f}")
print(f"MaxDD:     {maxdd:.1f}%")
print(f"Trades:    {trades}")
print(f"IDM:       {idm:.2f}")
print(f"Avg Pos:   {avg_pos:.1f}")
print("-" * 50)
print(f"Win Rate (trade):     {wr:.1f}%  ({len(trade_pnls)} round-trips)")
print(f"Profit Factor (trade):{pf:.2f}")
if trade_pnls:
    print(f"Avg Win:  {avg_w:+.2f}%   Avg Loss: {avg_l:.2f}%")
print(f"Win Rate (daily):     {daily_wr:.1f}%")
print(f"Profit Factor (daily):{daily_pf:.2f}")
print("=" * 50)
