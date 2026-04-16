"""
Vince Money Management Metrics — Phase 1 Foundation.

Tracks HPR, TWR, Geometric Mean, Optimal f per instrument and portfolio.
Based on Ralph Vince 'Mathematics of Money Management' (1992).

Integration:
  - Called by trade_monitor.py on every trade exit (SL/TP/time)
  - Queried by position_sizer.py for equalized-f weights
  - Exposed via /api/vince/metrics for UI
  - Persisted to data/vince_trades.json across sessions
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_TRADES_FILE = os.path.join(_DATA_DIR, "vince_trades.json")


@dataclass
class FundamentalEquation:
    """Vince Eq 1.19c: A² = G² + SD²."""
    A: float          # arithmetic mean HPR (1 + mean_return)
    G: float          # geometric mean HPR
    SD: float         # standard deviation of returns
    N: int            # trade count
    est_twr: float    # estimated TWR = G^(N/2)


@dataclass
class VinceSnapshot:
    """Point-in-time Vince metrics for a symbol or portfolio."""
    symbol: str
    optimal_f: float
    geometric_mean: float
    twr: float
    kelly_full: float
    kelly_half: float
    f_dollar: float
    biggest_loss_pct: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    math_expectation: float
    n_trades: int
    fundamental_eq: FundamentalEquation


class VinceTracker:
    """Per-instrument and portfolio-level Vince metric tracking.

    Thread-safe. Persists trade history to JSON for crash recovery.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._trades: Dict[str, List[float]] = {}  # symbol → list of pnl_pct
        self._load_state()

    # ── Trade Recording ───────────────────────────────────────

    def record_trade(self, symbol: str, pnl_pct: float) -> None:
        """Record a completed trade's return (as fraction, e.g. 0.02 = +2%).

        Called by trade_monitor on every SL/TP/time exit.
        """
        with self._lock:
            if symbol not in self._trades:
                self._trades[symbol] = []
            self._trades[symbol].append(round(pnl_pct, 6))
            self._trades.setdefault("__portfolio__", []).append(round(pnl_pct, 6))
            self._persist_state()

    # ── Optimal f Computation ─────────────────────────────────

    @staticmethod
    def _find_optimal_f(returns: np.ndarray, steps: int = 500) -> Tuple[float, float, float]:
        """Find optimal f by exhaustive search over (0, 1].

        Returns (optimal_f, geometric_mean_at_f, twr_at_f).
        """
        if len(returns) < 5:
            return 0.0, 1.0, 1.0

        biggest_loss = abs(float(np.min(returns)))
        if biggest_loss < 1e-9:
            return 0.0, 1.0, 1.0

        best_f, best_gm, best_twr = 0.01, 0.0, 1.0

        for step in range(1, steps + 1):
            f = step / steps
            hprs = 1.0 + f * (returns / biggest_loss)
            if np.any(hprs <= 0):
                break
            twr = float(np.prod(hprs))
            gm = twr ** (1.0 / len(returns))
            if gm > best_gm:
                best_gm = gm
                best_f = f
                best_twr = twr

        return best_f, best_gm, best_twr

    # ── Metric Queries ────────────────────────────────────────

    def get_snapshot(self, symbol: str = "__portfolio__") -> Optional[VinceSnapshot]:
        """Compute full Vince metrics for a symbol (or portfolio aggregate)."""
        with self._lock:
            trades = self._trades.get(symbol, [])

        if len(trades) < 10:
            return None

        returns = np.array(trades, dtype=np.float64)
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) < 3 or len(losses) < 3:
            return None

        win_rate = len(wins) / len(returns)
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean(np.abs(losses)))
        biggest_loss = float(abs(np.min(returns)))
        math_exp = float(np.mean(returns))

        # Kelly
        R = avg_win / avg_loss if avg_loss > 0 else 2.0
        kelly_full = max(win_rate - (1 - win_rate) / R, 0.0)
        kelly_half = kelly_full * 0.5

        # Optimal f
        opt_f, gm, twr = self._find_optimal_f(returns)
        f_dollar = biggest_loss / opt_f if opt_f > 0 else float('inf')

        # Fundamental Equation
        A = 1 + np.mean(returns)
        SD = float(np.std(returns, ddof=0))
        G_sq = max(A ** 2 - SD ** 2, 0)
        G = math.sqrt(G_sq)
        N = len(returns)
        est_twr = G ** (N / 2) if G > 0 else 0

        fund_eq = FundamentalEquation(
            A=round(float(A), 6),
            G=round(G, 6),
            SD=round(SD, 6),
            N=N,
            est_twr=round(est_twr, 4),
        )

        return VinceSnapshot(
            symbol=symbol,
            optimal_f=round(opt_f, 4),
            geometric_mean=round(gm, 6),
            twr=round(twr, 4),
            kelly_full=round(kelly_full, 4),
            kelly_half=round(kelly_half, 4),
            f_dollar=round(f_dollar, 2),
            biggest_loss_pct=round(biggest_loss * 100, 2),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win * 100, 2),
            avg_loss_pct=round(avg_loss * 100, 2),
            math_expectation=round(math_exp * 100, 4),
            n_trades=N,
            fundamental_eq=fund_eq,
        )

    def get_optimal_f(self, symbol: str = "__portfolio__") -> float:
        """Quick accessor: optimal f for a symbol."""
        snap = self.get_snapshot(symbol)
        return snap.optimal_f if snap else 0.0

    def get_geometric_mean(self, symbol: str = "__portfolio__") -> float:
        """Quick accessor: geometric mean for a symbol."""
        snap = self.get_snapshot(symbol)
        return snap.geometric_mean if snap else 1.0

    def get_kelly_half(self, symbol: str = "__portfolio__") -> float:
        """Quick accessor: half-Kelly fraction."""
        snap = self.get_snapshot(symbol)
        return snap.kelly_half if snap else 0.02

    def get_equalized_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Compute equalized-f weights: allocate proportional to geometric mean.

        Instruments with higher G get more capital allocation.
        Falls back to equal weight if insufficient trade data.
        """
        g_values = {}
        for sym in symbols:
            gm = self.get_geometric_mean(sym)
            # Only use if we have meaningful data (G > 1 means positive edge)
            g_values[sym] = max(gm - 1.0, 1e-6)  # excess G over 1.0

        total = sum(g_values.values())
        if total < 1e-9:
            # Fallback: equal weight
            n = len(symbols)
            return {sym: 1.0 / n for sym in symbols} if n > 0 else {}

        return {sym: g / total for sym, g in g_values.items()}

    def get_fundamental_equation(self, symbol: str = "__portfolio__") -> Optional[FundamentalEquation]:
        """Get Fundamental Equation analysis for a symbol."""
        snap = self.get_snapshot(symbol)
        return snap.fundamental_eq if snap else None

    def get_all_symbols(self) -> List[str]:
        """Return list of symbols with trade data (excluding __portfolio__)."""
        with self._lock:
            return [s for s in self._trades if s != "__portfolio__"]

    def get_trade_count(self, symbol: str = "__portfolio__") -> int:
        """Trade count for a symbol."""
        with self._lock:
            return len(self._trades.get(symbol, []))

    # ── Persistence ───────────────────────────────────────────

    def _persist_state(self) -> None:
        """Save trade history to JSON (called under lock)."""
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_TRADES_FILE, "w") as f:
                json.dump(self._trades, f)
        except Exception as exc:
            logger.warning("Failed to persist Vince trades: %s", exc)

    def _load_state(self) -> None:
        """Load trade history from JSON."""
        try:
            if os.path.exists(_TRADES_FILE):
                with open(_TRADES_FILE, "r") as f:
                    self._trades = json.load(f)
                logger.info(
                    "Loaded Vince trade history: %d symbols, %d total trades",
                    len(self._trades) - 1,  # exclude __portfolio__
                    len(self._trades.get("__portfolio__", [])),
                )
        except Exception as exc:
            logger.warning("Failed to load Vince trades: %s", exc)
            self._trades = {}

    def to_dict(self, symbol: str = "__portfolio__") -> dict:
        """Serialize snapshot to dict for API response."""
        snap = self.get_snapshot(symbol)
        if not snap:
            return {"symbol": symbol, "status": "insufficient_data", "min_trades": 10}
        return {
            "symbol": snap.symbol,
            "optimal_f": snap.optimal_f,
            "geometric_mean": snap.geometric_mean,
            "twr": snap.twr,
            "kelly_full": snap.kelly_full,
            "kelly_half": snap.kelly_half,
            "f_dollar": snap.f_dollar,
            "biggest_loss_pct": snap.biggest_loss_pct,
            "win_rate": snap.win_rate,
            "avg_win_pct": snap.avg_win_pct,
            "avg_loss_pct": snap.avg_loss_pct,
            "math_expectation": snap.math_expectation,
            "n_trades": snap.n_trades,
            "fundamental_equation": {
                "A": snap.fundamental_eq.A,
                "G": snap.fundamental_eq.G,
                "SD": snap.fundamental_eq.SD,
                "N": snap.fundamental_eq.N,
                "est_twr": snap.fundamental_eq.est_twr,
            },
        }


# ── Module-level singleton ────────────────────────────────────
_instance: Optional[VinceTracker] = None
_instance_lock = threading.Lock()


def get_vince_tracker() -> VinceTracker:
    """Get or create the global VinceTracker singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = VinceTracker()
    return _instance
