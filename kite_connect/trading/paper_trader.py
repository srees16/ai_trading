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

    def poll(self) -> List[dict]:
        """Check open positions against SL/TP using live prices.

        Returns list of close events.
        """
        events = []
        for pos in self._positions:
            if not pos.is_open:
                continue

            ltp = self._get_ltp(pos.symbol)
            if ltp is None:
                continue

            closed = False
            reason = ""

            if ltp <= pos.stop_loss:
                closed = True
                reason = "SL"
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
