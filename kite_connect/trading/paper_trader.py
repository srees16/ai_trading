"""
Paper Trading Engine — Virtual Order Simulation for Zerodha Kite.

Zerodha does not provide a native paper-trading API.  This module
implements a local virtual broker that:

1. Accepts trade plans from the AutoExecutor (same interface)
2. Simulates fills using live Kite LTP (or last yfinance close)
3. Applies realistic slippage (Config.SLIPPAGE_MODEL_IND_BPS)
4. Manages a virtual portfolio with SL/TP handling
5. Persists all trades + P&L to a SQLite journal
6. Produces a performance dashboard (daily P&L, drawdown, win rate)

Usage::

    from kite_connect.trading.paper_trader import PaperTrader

    pt = PaperTrader(kite=kite, initial_capital=100_000)
    pt.execute_plans(trade_plans)       # simulate order fills
    pt.poll()                           # check SL/TP (call periodically)
    print(pt.dashboard())               # P&L summary

The scheduler can invoke paper-trading instead of live orders by
setting ``PAPER_TRADE_MODE=true`` in the environment or config.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))
_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "paper_trades.sqlite3"


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class PaperPosition:
    """A single virtual position."""
    symbol: str
    side: str               # BUY
    quantity: int
    entry_price: float      # fill price after slippage
    stop_loss: float
    target_price: float
    opened_at: str = ""
    closed_at: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""   # SL / TP / MANUAL / TRAILING_SL
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_open: bool = True
    peak_price: float = 0.0  # G5: highest price since entry (for trailing SL)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperDashboard:
    """Paper-trading performance summary."""
    initial_capital: float
    current_capital: float
    open_positions: int
    closed_trades: int
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    # Advanced risk metrics (Phase 0)
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    cvar_95: float = 0.0
    profit_factor: float = 0.0
    positions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# Paper Trader
# ═══════════════════════════════════════════════════════════════

class PaperTrader:
    """Virtual broker for simulated order execution.

    Parameters
    ----------
    kite : KiteConnect | None
        Authenticated Kite session for live LTP.  If ``None``,
        falls back to yfinance last close.
    initial_capital : float
        Starting virtual capital (default: ₹1,00,000).
    slippage_bps : float | None
        Override slippage in basis points.  Defaults to
        ``Config.SLIPPAGE_MODEL_IND_BPS``.
    """

    def __init__(
        self,
        kite=None,
        initial_capital: float = 100_000.0,
        slippage_bps: Optional[float] = None,
    ):
        self.kite = kite
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._positions: List[PaperPosition] = []

        if slippage_bps is not None:
            self._slippage_bps = slippage_bps
            self._tiered_slippage = False
        else:
            try:
                from config import Config
                self._slippage_bps = getattr(Config, "SLIPPAGE_MODEL_IND_BPS", 20.0)
                self._slip_large = getattr(Config, "SLIPPAGE_IND_LARGECAP_BPS", 5.0)
                self._slip_mid = getattr(Config, "SLIPPAGE_IND_MIDCAP_BPS", 20.0)
                self._slip_small = getattr(Config, "SLIPPAGE_IND_SMALLCAP_BPS", 50.0)
                self._tiered_slippage = True
            except Exception:
                self._slippage_bps = 20.0
                self._tiered_slippage = False

        # Build large-cap / mid-cap symbol sets for tiered slippage
        self._largecap_set: set = set()
        self._midcap_set: set = set()
        if getattr(self, "_tiered_slippage", False):
            try:
                from kite_connect.core.config import INDEX_CONSTITUENTS
                self._largecap_set = set(INDEX_CONSTITUENTS.get("NIFTY50", []))
                self._midcap_set = set(INDEX_CONSTITUENTS.get("NIFTY_NEXT50", []))
            except Exception:
                pass

        self._cloud = None  # lazy-init cloud sync
        self._init_db()
        self._load_state()

    # ── DB schema ──────────────────────────────────────────────

    def _init_db(self):
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                side        TEXT NOT NULL,
                quantity    INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss   REAL NOT NULL,
                target_price REAL NOT NULL,
                opened_at   TEXT NOT NULL,
                closed_at   TEXT DEFAULT '',
                exit_price  REAL DEFAULT 0,
                exit_reason TEXT DEFAULT '',
                pnl         REAL DEFAULT 0,
                pnl_pct     REAL DEFAULT 0,
                is_open     INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_state (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # ── Paper validation checkpoint tables ─────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                date        TEXT PRIMARY KEY,
                equity      REAL NOT NULL,
                cash        REAL NOT NULL,
                open_positions INTEGER DEFAULT 0,
                closed_today   INTEGER DEFAULT 0,
                day_pnl     REAL DEFAULT 0,
                cumulative_pnl REAL DEFAULT 0,
                cumulative_pnl_pct REAL DEFAULT 0,
                max_drawdown_pct   REAL DEFAULT 0,
                signals_generated  INTEGER DEFAULT 0,
                signals_traded     INTEGER DEFAULT 0,
                snapshot_json TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                forecast    REAL DEFAULT 0,
                combined_forecast REAL DEFAULT 0,
                action      TEXT DEFAULT '',
                entry_price REAL DEFAULT 0,
                stop_loss   REAL DEFAULT 0,
                target_price REAL DEFAULT 0,
                quantity    INTEGER DEFAULT 0,
                pipeline_sources TEXT DEFAULT '',
                was_traded  INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weekly_checkpoints (
                week_number INTEGER PRIMARY KEY,
                week_start  TEXT NOT NULL,
                week_end    TEXT NOT NULL,
                start_equity REAL DEFAULT 0,
                end_equity   REAL DEFAULT 0,
                week_return_pct REAL DEFAULT 0,
                trades_opened  INTEGER DEFAULT 0,
                trades_closed  INTEGER DEFAULT 0,
                win_rate    REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                max_dd_pct  REAL DEFAULT 0,
                avg_holding_days REAL DEFAULT 0,
                summary_json TEXT DEFAULT '{}'
            )
        """)
        conn.commit()
        conn.close()

    def _load_state(self):
        """Restore positions and cash from DB."""
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Cash
        row = conn.execute(
            "SELECT value FROM paper_state WHERE key='cash'"
        ).fetchone()
        if row:
            self.cash = float(row["value"])

        # Open positions
        rows = conn.execute(
            "SELECT * FROM paper_positions WHERE is_open=1"
        ).fetchall()
        for r in rows:
            self._positions.append(PaperPosition(
                symbol=r["symbol"], side=r["side"],
                quantity=r["quantity"], entry_price=r["entry_price"],
                stop_loss=r["stop_loss"], target_price=r["target_price"],
                opened_at=r["opened_at"], is_open=True,
            ))

        conn.close()
        logger.info(
            "Paper trader loaded: cash=%.2f, %d open positions",
            self.cash, len(self._positions),
        )

    def _save_cash(self):
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value) VALUES ('cash', ?)",
            (str(self.cash),),
        )
        conn.commit()
        conn.close()

    def _get_cloud(self):
        """Lazy-init cloud sync (best-effort, never blocks)."""
        if self._cloud is None:
            try:
                from database.paper_cloud import get_paper_cloud
                self._cloud = get_paper_cloud()
            except Exception:
                self._cloud = False  # sentinel: don't retry
        return self._cloud if self._cloud else None

    def _save_position(self, pos: PaperPosition):
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("""
            INSERT INTO paper_positions
            (symbol, side, quantity, entry_price, stop_loss, target_price,
             opened_at, closed_at, exit_price, exit_reason, pnl, pnl_pct, is_open)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos.symbol, pos.side, pos.quantity, pos.entry_price,
            pos.stop_loss, pos.target_price, pos.opened_at,
            pos.closed_at, pos.exit_price, pos.exit_reason,
            pos.pnl, pos.pnl_pct, 1 if pos.is_open else 0,
        ))
        conn.commit()
        conn.close()
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_position(pos.to_dict())

    def _close_position_db(self, pos: PaperPosition):
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("""
            UPDATE paper_positions SET
                closed_at=?, exit_price=?, exit_reason=?,
                pnl=?, pnl_pct=?, is_open=0
            WHERE symbol=? AND is_open=1 AND opened_at=?
        """, (
            pos.closed_at, pos.exit_price, pos.exit_reason,
            pos.pnl, pos.pnl_pct, pos.symbol, pos.opened_at,
        ))
        conn.commit()
        conn.close()
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_position(pos.to_dict())

    # ── Price helpers ──────────────────────────────────────────

    def _get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price from Kite or yfinance."""
        if self.kite:
            try:
                key = f"NSE:{symbol}"
                data = self.kite.ltp([key])
                ltp = data.get(key, {}).get("last_price")
                if ltp and ltp > 0:
                    return float(ltp)
            except Exception:
                pass

        # Fallback: Bhavcopy → yfinance
        try:
            from utils import download_ind_ohlcv
            df = download_ind_ohlcv(symbol, period="5d")
            if not df.empty:
                close = df["Close"].iloc[-1]
                if hasattr(close, "item"):
                    return float(close.item())
                return float(close)
        except Exception:
            pass

        return None

    def _get_base_slippage_bps(self, symbol: str = "") -> float:
        """Return market-cap tiered slippage for *symbol*.

        Large-cap (NIFTY50):   ~5 bps
        Mid-cap (NIFTY_NEXT50): ~20 bps
        Small-cap (others):     ~50 bps
        """
        if not self._tiered_slippage or not symbol:
            return self._slippage_bps
        clean = symbol.replace(".NS", "").upper()
        if clean in self._largecap_set:
            return self._slip_large
        if clean in self._midcap_set:
            return self._slip_mid
        return self._slip_small

    def _apply_slippage(self, price: float, side: str, order_qty: int = 0, adv: float = 0.0, symbol: str = "") -> float:
        """Apply volume-aware, market-cap-tiered slippage.

        base_bps is determined by symbol market-cap tier, then
        impact_bps = order_pct_of_volume × 300  is added.
        Falls back to flat slippage if ADV unknown.
        """
        base_bps = self._get_base_slippage_bps(symbol)
        if adv > 0 and order_qty > 0:
            order_pct = abs(order_qty) / adv
            impact_bps = order_pct * 300.0  # 300 bps impact per 100% of ADV
            total_bps = base_bps + impact_bps
        else:
            total_bps = base_bps
        slip = price * (total_bps / 10_000.0)
        if side == "BUY":
            return round(price + slip, 2)
        return round(price - slip, 2)

    # ── Execution ──────────────────────────────────────────────

    def execute_plans(self, plans: list) -> List[dict]:
        """Simulate order fills for a list of TradePlan objects.

        Returns a list of result dicts compatible with OrderResult.
        """
        results = []
        for plan in plans:
            symbol = plan.symbol
            ltp = self._get_ltp(symbol)
            if ltp is None:
                results.append({
                    "symbol": symbol, "success": False,
                    "error": "No price available",
                })
                continue

            fill_price = self._apply_slippage(ltp, plan.side, order_qty=plan.quantity, symbol=symbol)
            cost = fill_price * plan.quantity

            if plan.side == "BUY":
                if cost > self.cash:
                    results.append({
                        "symbol": symbol, "success": False,
                        "error": f"Insufficient capital: need {cost:.0f}, have {self.cash:.0f}",
                    })
                    continue
                self.cash -= cost

            pos = PaperPosition(
                symbol=symbol,
                side=plan.side,
                quantity=plan.quantity,
                entry_price=fill_price,
                stop_loss=plan.stop_loss,
                target_price=plan.target_price,
                opened_at=datetime.now(_IST).isoformat(),
                peak_price=fill_price,  # G5: initialize peak at entry
            )
            self._positions.append(pos)
            self._save_position(pos)
            self._save_cash()

            logger.info(
                "PAPER %s: %s × %d @ %.2f (slip=%.1fbps, cost=%.0f)",
                plan.side, symbol, plan.quantity, fill_price,
                self._slippage_bps, cost,
            )
            results.append({
                "symbol": symbol,
                "success": True,
                "fill_price": fill_price,
                "quantity": plan.quantity,
                "side": plan.side,
            })

        return results

    # ── SL/TP check (call periodically) ────────────────────────

    def _trail_stop(self, pos: PaperPosition, ltp: float) -> None:
        """G5: Ratchet stop-loss using vol-based trailing stop.

        Uses services.vol_trailing_stop.compute_trailing_stop() which:
          - Scales stop distance by daily volatility (2.5σ swing, 3.5σ positional)
          - Activates profit-lock after 4σ gain (tightens to 1.5σ)
          - Guarantees break-even once profit-lock activates
          - Clamps stop between 2% (min) and 12% (max) of peak

        Falls back to simple 3% percentage trail if vol module unavailable.
        """
        # Update peak price
        if pos.peak_price <= 0:
            pos.peak_price = pos.entry_price
        pos.peak_price = max(pos.peak_price, ltp)

        try:
            from services.vol_trailing_stop import compute_trailing_stop
            from services.instrument_volatility import daily_price_volatility
            from utils import download_ind_ohlcv

            df = download_ind_ohlcv(pos.symbol, period="3mo")
            if df is not None and len(df) >= 20:
                close_series = df["Close"] if "Close" in df.columns else df["close"]
                daily_vol = daily_price_volatility(close_series)
            else:
                daily_vol = 0.02

            try:
                from config import Config
                trade_horizon = getattr(Config, "CARVER_TRADE_HORIZON", "swing")
            except Exception:
                trade_horizon = "swing"

            state = compute_trailing_stop(
                entry_price=pos.entry_price,
                current_price=ltp,
                peak_price=pos.peak_price,
                daily_price_vol=daily_vol,
                previous_stop=pos.stop_loss,
                trade_horizon=trade_horizon,
            )
            new_sl = state.current_stop
        except Exception:
            # Fallback: simple 3% trail from peak (only activate after 5% profit)
            profit_pct = (ltp - pos.entry_price) / pos.entry_price
            if profit_pct < 0.05:
                return
            new_sl = round(pos.peak_price * 0.97, 2)

        # Only ratchet UP for LONG positions
        if new_sl > pos.stop_loss:
            old_sl = pos.stop_loss
            pos.stop_loss = round(new_sl, 2)
            # Persist updated SL to DB
            self._update_stop_db(pos)
            logger.debug(
                "PAPER TRAIL SL: %s %.2f → %.2f (peak=%.2f, ltp=%.2f)",
                pos.symbol, old_sl, pos.stop_loss, pos.peak_price, ltp,
            )

    def _update_stop_db(self, pos: PaperPosition) -> None:
        """Persist updated stop_loss to the DB for crash recovery."""
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.execute(
                "UPDATE paper_positions SET stop_loss=? WHERE symbol=? AND is_open=1 AND opened_at=?",
                (pos.stop_loss, pos.symbol, pos.opened_at),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_stop_loss(pos.symbol, pos.opened_at, pos.stop_loss)

    def poll(self) -> List[dict]:
        """Check open positions against SL/TP using live prices.

        G5: Also applies vol-based trailing stop ratcheting before
        checking SL/TP triggers. Returns list of close events.
        """
        events = []
        for pos in self._positions:
            if not pos.is_open:
                continue

            ltp = self._get_ltp(pos.symbol)
            if ltp is None:
                continue

            # G5: Trail stop before checking triggers
            self._trail_stop(pos, ltp)

            closed = False
            reason = ""

            if ltp <= pos.stop_loss:
                closed = True
                reason = "TRAILING_SL" if pos.stop_loss > pos.entry_price * 0.97 else "SL"
                exit_price = self._apply_slippage(pos.stop_loss, "SELL", symbol=pos.symbol)
            elif ltp >= pos.target_price:
                closed = True
                reason = "TP"
                exit_price = self._apply_slippage(pos.target_price, "SELL", symbol=pos.symbol)

            if closed:
                pos.is_open = False
                pos.exit_price = exit_price
                pos.exit_reason = reason
                pos.closed_at = datetime.now(_IST).isoformat()
                pos.pnl = (exit_price - pos.entry_price) * pos.quantity
                pos.pnl_pct = (exit_price / pos.entry_price - 1) * 100
                self.cash += exit_price * pos.quantity
                self._close_position_db(pos)
                self._save_cash()

                logger.info(
                    "PAPER CLOSE [%s]: %s @ %.2f → %.2f | P&L=%.2f (%.1f%%)",
                    reason, pos.symbol, pos.entry_price, exit_price,
                    pos.pnl, pos.pnl_pct,
                )
                events.append({
                    "type": f"PAPER_{reason}",
                    "symbol": pos.symbol,
                    "entry": pos.entry_price,
                    "exit": exit_price,
                    "pnl": round(pos.pnl, 2),
                    "pnl_pct": round(pos.pnl_pct, 2),
                })

        return events

    # ── Dashboard ──────────────────────────────────────────────

    def dashboard(self) -> PaperDashboard:
        """Compute performance dashboard from all paper trades."""
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM paper_positions").fetchall()
        conn.close()

        open_positions = []
        closed_trades = []

        for r in rows:
            if r["is_open"]:
                open_positions.append(dict(r))
            else:
                closed_trades.append(dict(r))

        total_pnl = sum(t["pnl"] for t in closed_trades)
        wins = [t for t in closed_trades if t["pnl"] > 0]
        losses = [t for t in closed_trades if t["pnl"] <= 0]

        win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
        avg_win = (
            sum(t["pnl_pct"] for t in wins) / len(wins)
            if wins else 0.0
        )
        avg_loss = (
            sum(t["pnl_pct"] for t in losses) / len(losses)
            if losses else 0.0
        )

        current_capital = self.cash
        # Mark-to-market open positions
        for pos_dict in open_positions:
            ltp = self._get_ltp(pos_dict["symbol"])
            if ltp:
                current_capital += ltp * pos_dict["quantity"]
            else:
                current_capital += pos_dict["entry_price"] * pos_dict["quantity"]

        total_pnl_pct = (
            (current_capital / self.initial_capital - 1) * 100
            if self.initial_capital > 0 else 0.0
        )

        # Approximate max drawdown from closed trade sequence
        equity_curve = [self.initial_capital]
        for t in sorted(closed_trades, key=lambda x: x.get("closed_at", "")):
            equity_curve.append(equity_curve[-1] + t["pnl"])
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Sharpe from closed trade returns
        import numpy as np
        trade_returns = [t["pnl_pct"] / 100 for t in closed_trades]
        if len(trade_returns) >= 2:
            sr = float(
                np.mean(trade_returns) / (np.std(trade_returns) + 1e-10)
                * np.sqrt(min(len(trade_returns), 252))
            )
        else:
            sr = 0.0

        # Advanced risk metrics (Phase 0)
        sortino = calmar = omega = cvar95 = pf = 0.0
        if len(trade_returns) >= 5:
            try:
                from services.risk_metrics import RiskMetrics
                returns_series = pd.Series(trade_returns)
                sortino = RiskMetrics.sortino_ratio(returns_series)
                calmar = RiskMetrics.calmar_ratio(returns_series)
                omega = RiskMetrics.omega_ratio(returns_series)
                cvar95 = RiskMetrics.cvar(returns_series, alpha=0.05)
                pf = RiskMetrics.profit_factor(returns_series)
            except Exception as exc:
                logger.debug("Advanced risk metrics unavailable: %s", exc)

        return PaperDashboard(
            initial_capital=self.initial_capital,
            current_capital=round(current_capital, 2),
            open_positions=len(open_positions),
            closed_trades=len(closed_trades),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sr, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            omega_ratio=round(omega, 3),
            cvar_95=round(cvar95, 4),
            profit_factor=round(pf, 3),
            positions=[p.to_dict() for p in self._positions if p.is_open],
        )

    def reset(self):
        """Reset the paper trading journal (start fresh)."""
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("DELETE FROM paper_positions")
        conn.execute("DELETE FROM paper_state")
        conn.commit()
        conn.close()
        self.cash = self.initial_capital
        self._positions.clear()
        logger.info("Paper trader reset: capital=%.2f", self.initial_capital)

    # ── Checkpoint methods for 4-week paper validation ─────────

    def snapshot_daily(self, signals_generated: int = 0, signals_traded: int = 0) -> dict:
        """Save end-of-day equity snapshot for equity curve reconstruction.

        Call this once daily (EOD scheduler job). Even if something crashes
        mid-week, we'll have daily granularity up to the crash point.
        """
        today = datetime.now(_IST).strftime("%Y-%m-%d")
        dashboard = self.dashboard()

        # Count trades closed today
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM paper_positions WHERE is_open=0 AND closed_at LIKE ?",
            (today + "%",),
        ).fetchall()
        closed_today = len(rows)
        day_pnl = sum(r["pnl"] for r in rows)

        # Compute running max drawdown from daily_snapshots history
        prev_snapshots = conn.execute(
            "SELECT equity FROM daily_snapshots ORDER BY date"
        ).fetchall()
        equities = [r["equity"] for r in prev_snapshots] + [dashboard.current_capital]
        peak = equities[0] if equities else self.initial_capital
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        snapshot = {
            "date": today,
            "equity": dashboard.current_capital,
            "cash": self.cash,
            "open_positions": dashboard.open_positions,
            "closed_today": closed_today,
            "day_pnl": round(day_pnl, 2),
            "cumulative_pnl": dashboard.total_pnl,
            "cumulative_pnl_pct": dashboard.total_pnl_pct,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "signals_generated": signals_generated,
            "signals_traded": signals_traded,
            "snapshot_json": json.dumps(dashboard.to_dict()),
        }

        conn.execute("""
            INSERT OR REPLACE INTO daily_snapshots
            (date, equity, cash, open_positions, closed_today, day_pnl,
             cumulative_pnl, cumulative_pnl_pct, max_drawdown_pct,
             signals_generated, signals_traded, snapshot_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot["date"], snapshot["equity"], snapshot["cash"],
            snapshot["open_positions"], snapshot["closed_today"],
            snapshot["day_pnl"], snapshot["cumulative_pnl"],
            snapshot["cumulative_pnl_pct"], snapshot["max_drawdown_pct"],
            snapshot["signals_generated"], snapshot["signals_traded"],
            snapshot["snapshot_json"],
        ))
        conn.commit()
        conn.close()

        logger.info(
            "Daily snapshot: %s equity=%.0f pnl=%.0f (%.1f%%) dd=%.1f%% open=%d closed=%d",
            today, snapshot["equity"], snapshot["cumulative_pnl"],
            snapshot["cumulative_pnl_pct"], snapshot["max_drawdown_pct"],
            snapshot["open_positions"], closed_today,
        )
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_snapshot(snapshot)
        return snapshot

    def log_signals(self, date_str: str, signals: list) -> None:
        """Persist forecast signals for backtest-vs-live comparison.

        Parameters
        ----------
        date_str : str
            Date string (YYYY-MM-DD).
        signals : list of dict
            Each dict: {symbol, forecast, combined_forecast, action,
                       entry_price, stop_loss, target_price, quantity,
                       pipeline_sources, was_traded}
        """
        if not signals:
            return
        conn = sqlite3.connect(str(_DB_PATH))
        for sig in signals:
            conn.execute("""
                INSERT INTO signal_log
                (date, symbol, forecast, combined_forecast, action,
                 entry_price, stop_loss, target_price, quantity,
                 pipeline_sources, was_traded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                sig.get("symbol", ""),
                sig.get("forecast", 0),
                sig.get("combined_forecast", 0),
                sig.get("action", ""),
                sig.get("entry_price", 0),
                sig.get("stop_loss", 0),
                sig.get("target_price", 0),
                sig.get("quantity", 0),
                sig.get("pipeline_sources", ""),
                1 if sig.get("was_traded") else 0,
            ))
        conn.commit()
        conn.close()
        logger.debug("Signal log: %d signals for %s", len(signals), date_str)
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_signals(date_str, signals)

    def checkpoint_weekly(self) -> Optional[dict]:
        """Save weekly aggregated checkpoint for crash-resilient analysis.

        Call this once per week (e.g., Friday EOD or Saturday).
        Returns the checkpoint dict, or None if no data.
        """
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Determine week number from daily snapshots
        snapshots = conn.execute(
            "SELECT * FROM daily_snapshots ORDER BY date"
        ).fetchall()
        if not snapshots:
            conn.close()
            return None

        # Get existing weekly checkpoints to determine current week
        existing_weeks = conn.execute(
            "SELECT week_number FROM weekly_checkpoints ORDER BY week_number DESC LIMIT 1"
        ).fetchone()
        current_week = (existing_weeks["week_number"] + 1) if existing_weeks else 1

        # Get last checkpoint date to determine this week's range
        last_ckpt = conn.execute(
            "SELECT week_end FROM weekly_checkpoints ORDER BY week_number DESC LIMIT 1"
        ).fetchone()
        if last_ckpt:
            week_snapshots = [s for s in snapshots if s["date"] > last_ckpt["week_end"]]
        else:
            week_snapshots = list(snapshots)

        if not week_snapshots:
            conn.close()
            return None

        week_start = week_snapshots[0]["date"]
        week_end = week_snapshots[-1]["date"]
        start_equity = week_snapshots[0]["equity"]
        end_equity = week_snapshots[-1]["equity"]
        week_return = ((end_equity / start_equity) - 1) * 100 if start_equity > 0 else 0

        # Count trades in this week
        trades_opened = conn.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE opened_at >= ? AND opened_at <= ?",
            (week_start, week_end + "T23:59:59"),
        ).fetchone()[0]
        trades_closed = conn.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE closed_at >= ? AND closed_at <= ? AND is_open=0",
            (week_start, week_end + "T23:59:59"),
        ).fetchone()[0]

        # Win rate for this week's closed trades
        week_closed = conn.execute(
            "SELECT pnl FROM paper_positions WHERE closed_at >= ? AND closed_at <= ? AND is_open=0",
            (week_start, week_end + "T23:59:59"),
        ).fetchall()
        wins = sum(1 for r in week_closed if r["pnl"] > 0)
        win_rate = wins / len(week_closed) if week_closed else 0

        # Sharpe from daily returns this week
        import numpy as np
        daily_returns = []
        for i in range(1, len(week_snapshots)):
            prev_eq = week_snapshots[i - 1]["equity"]
            curr_eq = week_snapshots[i]["equity"]
            if prev_eq > 0:
                daily_returns.append(curr_eq / prev_eq - 1)
        if len(daily_returns) >= 2:
            sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252))
        else:
            sharpe = 0.0

        max_dd = max((s["max_drawdown_pct"] for s in week_snapshots), default=0)

        # Average holding days for closed trades this week
        avg_hold = 0.0
        if week_closed:
            hold_days = []
            for r in conn.execute(
                "SELECT opened_at, closed_at FROM paper_positions WHERE closed_at >= ? AND closed_at <= ? AND is_open=0",
                (week_start, week_end + "T23:59:59"),
            ).fetchall():
                try:
                    opened = datetime.fromisoformat(r["opened_at"].replace("Z", "+00:00"))
                    closed = datetime.fromisoformat(r["closed_at"].replace("Z", "+00:00"))
                    hold_days.append((closed - opened).total_seconds() / 86400)
                except Exception:
                    pass
            if hold_days:
                avg_hold = sum(hold_days) / len(hold_days)

        checkpoint = {
            "week_number": current_week,
            "week_start": week_start,
            "week_end": week_end,
            "start_equity": round(start_equity, 2),
            "end_equity": round(end_equity, 2),
            "week_return_pct": round(week_return, 2),
            "trades_opened": trades_opened,
            "trades_closed": trades_closed,
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_dd_pct": round(max_dd, 2),
            "avg_holding_days": round(avg_hold, 1),
        }

        summary = self.dashboard().to_dict()
        conn.execute("""
            INSERT OR REPLACE INTO weekly_checkpoints
            (week_number, week_start, week_end, start_equity, end_equity,
             week_return_pct, trades_opened, trades_closed, win_rate,
             sharpe_ratio, max_dd_pct, avg_holding_days, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint["week_number"], checkpoint["week_start"],
            checkpoint["week_end"], checkpoint["start_equity"],
            checkpoint["end_equity"], checkpoint["week_return_pct"],
            checkpoint["trades_opened"], checkpoint["trades_closed"],
            checkpoint["win_rate"], checkpoint["sharpe_ratio"],
            checkpoint["max_dd_pct"], checkpoint["avg_holding_days"],
            json.dumps(summary),
        ))
        conn.commit()
        conn.close()

        logger.info(
            "Weekly checkpoint W%d: %s→%s equity=%.0f→%.0f ret=%.1f%% "
            "trades=%d/%d wr=%.0f%% sharpe=%.2f dd=%.1f%%",
            current_week, week_start, week_end,
            start_equity, end_equity, week_return,
            trades_opened, trades_closed, win_rate * 100,
            sharpe, max_dd,
        )
        # Cloud sync (best-effort)
        cloud = self._get_cloud()
        if cloud:
            cloud.sync_weekly(checkpoint)
        return checkpoint
