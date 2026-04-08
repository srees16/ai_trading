"""
Paper Trading Validation — 4-Week Analysis Report.

Reads checkpoint data from paper_trades.sqlite3 and produces a
conclusive analysis even if the run was interrupted mid-way.

Usage:
    python analyze_paper_results.py
    python analyze_paper_results.py --compare-backtest
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_DB_PATH = Path(__file__).parent / "data" / "paper_trades.sqlite3"


def _connect():
    if not _DB_PATH.exists():
        print(f"ERROR: {_DB_PATH} not found. No paper trading data.")
        sys.exit(1)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def analyze():
    conn = _connect()

    # ── 1. Daily equity curve ──────────────────────────────────
    snapshots = conn.execute(
        "SELECT * FROM daily_snapshots ORDER BY date"
    ).fetchall()

    if not snapshots:
        print("No daily snapshots found. Paper trading hasn't run or EOD snapshot job didn't fire.")
        print("\nFalling back to trade-level analysis...\n")
        snapshots = []

    # ── 2. Weekly checkpoints ──────────────────────────────────
    weeks = conn.execute(
        "SELECT * FROM weekly_checkpoints ORDER BY week_number"
    ).fetchall()

    # ── 3. All trades ──────────────────────────────────────────
    all_trades = conn.execute(
        "SELECT * FROM paper_positions ORDER BY opened_at"
    ).fetchall()
    open_trades = [t for t in all_trades if t["is_open"]]
    closed_trades = [t for t in all_trades if not t["is_open"]]

    # ── 4. Signal log ──────────────────────────────────────────
    signals = conn.execute(
        "SELECT * FROM signal_log ORDER BY date, symbol"
    ).fetchall()

    # ── 5. State ───────────────────────────────────────────────
    cash_row = conn.execute(
        "SELECT value FROM paper_state WHERE key='cash'"
    ).fetchone()
    cash = float(cash_row["value"]) if cash_row else 0

    conn.close()

    # ═══════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════
    print("=" * 70)
    print("  PAPER TRADING VALIDATION REPORT")
    print("=" * 70)

    # Duration
    if snapshots:
        first_date = snapshots[0]["date"]
        last_date = snapshots[-1]["date"]
        n_days = len(snapshots)
        print(f"\n  Period     : {first_date} → {last_date} ({n_days} trading days)")
    elif closed_trades:
        dates = sorted(set(
            t["opened_at"][:10] for t in all_trades if t["opened_at"]
        ))
        print(f"\n  Period     : {dates[0]} → {dates[-1]} ({len(dates)} unique dates)")
    else:
        print("\n  Period     : No data")

    # Capital
    initial = 100_000  # default
    print(f"  Capital    : ₹{initial:,.0f}")

    # ── Equity curve metrics ───────────────────────────────────
    if snapshots:
        equities = [s["equity"] for s in snapshots]
        final_eq = equities[-1]
        total_ret = (final_eq / initial - 1) * 100

        # CAGR (annualized)
        n_days_val = len(equities)
        if n_days_val > 1:
            cagr = ((final_eq / initial) ** (252 / n_days_val) - 1) * 100
        else:
            cagr = 0

        # Max drawdown
        peak = initial
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe from daily returns
        daily_rets = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                daily_rets.append(equities[i] / equities[i - 1] - 1)
        if len(daily_rets) >= 2:
            sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252))
        else:
            sharpe = 0

        print(f"\n  ── Equity Curve ──")
        print(f"  Final equity  : ₹{final_eq:,.0f}")
        print(f"  Total return  : {total_ret:+.1f}%")
        print(f"  CAGR (ann.)   : {cagr:+.1f}%")
        print(f"  Sharpe (ann.) : {sharpe:.3f}")
        print(f"  Max drawdown  : {max_dd * 100:.1f}%")

    # ── Trade metrics ──────────────────────────────────────────
    print(f"\n  ── Trades ──")
    print(f"  Total opened  : {len(all_trades)}")
    print(f"  Still open    : {len(open_trades)}")
    print(f"  Closed        : {len(closed_trades)}")

    if closed_trades:
        wins = [t for t in closed_trades if t["pnl"] > 0]
        losses = [t for t in closed_trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(closed_trades)
        total_pnl = sum(t["pnl"] for t in closed_trades)
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

        # Exit reason breakdown
        reasons = {}
        for t in closed_trades:
            r = t["exit_reason"] or "UNKNOWN"
            reasons[r] = reasons.get(r, 0) + 1

        print(f"  Win rate      : {win_rate:.0%}")
        print(f"  Avg win       : {avg_win:+.1f}%")
        print(f"  Avg loss      : {avg_loss:+.1f}%")
        print(f"  Total P&L     : ₹{total_pnl:,.0f}")
        print(f"  Exit reasons  : {dict(reasons)}")

        # Holding period
        hold_days = []
        for t in closed_trades:
            try:
                o = datetime.fromisoformat(t["opened_at"])
                c = datetime.fromisoformat(t["closed_at"])
                hold_days.append((c - o).total_seconds() / 86400)
            except Exception:
                pass
        if hold_days:
            print(f"  Avg hold days : {np.mean(hold_days):.1f}")
    else:
        print("  (no closed trades yet)")

    # ── Signal analysis ────────────────────────────────────────
    if signals:
        print(f"\n  ── Signal Audit ──")
        total_signals = len(signals)
        traded = sum(1 for s in signals if s["was_traded"])
        print(f"  Signals generated : {total_signals}")
        print(f"  Signals traded    : {traded} ({traded/total_signals:.0%})")

        # Daily signal counts
        by_date = {}
        for s in signals:
            d = s["date"]
            by_date.setdefault(d, {"total": 0, "traded": 0})
            by_date[d]["total"] += 1
            if s["was_traded"]:
                by_date[d]["traded"] += 1

        avg_daily = np.mean([v["total"] for v in by_date.values()])
        print(f"  Avg signals/day   : {avg_daily:.1f}")

    # ── Weekly checkpoints ─────────────────────────────────────
    if weeks:
        print(f"\n  ── Weekly Checkpoints ──")
        print(f"  {'Wk':>3} {'Start':>10} {'End':>10} {'Return':>8} {'Sharpe':>7} "
              f"{'MaxDD':>6} {'Trades':>7} {'WinRate':>8}")
        print(f"  {'─' * 3} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 7} "
              f"{'─' * 6} {'─' * 7} {'─' * 8}")
        for w in weeks:
            print(f"  {w['week_number']:>3} {w['week_start']:>10} {w['week_end']:>10} "
                  f"{w['week_return_pct']:>+7.1f}% {w['sharpe_ratio']:>6.2f} "
                  f"{w['max_dd_pct']:>5.1f}% {w['trades_closed']:>4}/{w['trades_opened']:<3} "
                  f"{w['win_rate'] * 100:>6.0f}%")
    else:
        print("\n  No weekly checkpoints yet (saved every Saturday)")

    # ── Verdict ────────────────────────────────────────────────
    print(f"\n  {'=' * 60}")
    if snapshots and len(snapshots) >= 15:
        if sharpe >= 0.5 and max_dd < 0.30:
            print("  VERDICT: PASS — Ready for live trading")
        elif sharpe >= 0.2:
            print("  VERDICT: MARGINAL — Consider extending paper period")
        else:
            print("  VERDICT: FAIL — Do not go live, needs investigation")
    elif weeks and len(weeks) >= 2:
        avg_wr = np.mean([w["week_return_pct"] for w in weeks])
        print(f"  PARTIAL DATA: {len(weeks)} weeks, avg weekly return {avg_wr:+.1f}%")
        print("  (Need 4 full weeks for conclusive verdict)")
    else:
        print("  INSUFFICIENT DATA — Continue paper trading")
    print(f"  {'=' * 60}\n")


if __name__ == "__main__":
    analyze()
