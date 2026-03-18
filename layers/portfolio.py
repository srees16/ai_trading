"""
Portfolio Management Layer — Allocation, P&L tracking, rebalancing.

Tracks open positions, computes real-time P&L, and provides
portfolio-level analytics.

Emits:
  - ``portfolio.position_opened``
  - ``portfolio.position_closed``
  - ``portfolio.rebalance``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _next_business_day(dt: datetime) -> datetime:
    """Return the next business day (skip Sat/Sun) after *dt*."""
    nxt = dt + timedelta(days=1)
    while nxt.weekday() >= 5:  # 5=Sat, 6=Sun
        nxt += timedelta(days=1)
    return nxt


@dataclass
class Position:
    ticker: str
    quantity: int
    entry_price: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    target: Optional[float] = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class _PendingSettlement:
    """Proceeds from a sale that are locked until T+1 settlement."""
    amount: float
    available_after: datetime  # UTC timestamp when funds become available


class PortfolioManager:
    """
    Tracks open positions and computes portfolio-level metrics.
    """

    def __init__(self, *, initial_capital: float = 500_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._closed_trades: List[dict] = []
        self._pending_settlements: List[_PendingSettlement] = []

    def open_position(
        self,
        ticker: str,
        quantity: int,
        entry_price: float,
        *,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
    ) -> Position:
        from infrastructure.event_bus import event_bus

        cost = quantity * entry_price
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}")

        self.cash -= cost
        pos = Position(
            ticker=ticker, quantity=quantity, entry_price=entry_price,
            current_price=entry_price, stop_loss=stop_loss, target=target,
        )
        self._positions[ticker] = pos

        event_bus.emit(
            "portfolio.position_opened",
            payload={"ticker": ticker, "qty": quantity, "price": entry_price},
            source="portfolio_manager",
        )
        return pos

    def close_position(self, ticker: str, exit_price: float) -> Optional[dict]:
        from infrastructure.event_bus import event_bus

        pos = self._positions.pop(ticker, None)
        if not pos:
            return None

        pos.current_price = exit_price
        proceeds = pos.quantity * exit_price

        # T+1 settlement: proceeds are locked until the next business day
        settlement_dt = _next_business_day(datetime.now(timezone.utc))
        self._pending_settlements.append(
            _PendingSettlement(amount=proceeds, available_after=settlement_dt)
        )

        trade = {
            "ticker": ticker,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "pnl": pos.pnl,
            "pnl_pct": pos.pnl_pct,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "settlement_date": settlement_dt.isoformat(),
        }
        self._closed_trades.append(trade)

        event_bus.emit(
            "portfolio.position_closed",
            payload=trade,
            source="portfolio_manager",
        )
        return trade

    def settle_pending(self) -> float:
        """Release any pending settlements whose T+1 window has elapsed.

        Returns the total amount released into ``cash``.
        """
        now = datetime.now(timezone.utc)
        released = 0.0
        still_pending = []
        for ps in self._pending_settlements:
            if now >= ps.available_after:
                self.cash += ps.amount
                released += ps.amount
            else:
                still_pending.append(ps)
        self._pending_settlements = still_pending
        return released

    @property
    def pending_settlement_amount(self) -> float:
        """Total proceeds locked in T+1 settlement."""
        return sum(ps.amount for ps in self._pending_settlements)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Batch update current prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self._positions:
                self._positions[ticker].current_price = price

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self._positions.values()) + self.pending_settlement_amount

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def closed_trades(self) -> List[dict]:
        return list(self._closed_trades)

    def get_summary(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "pending_settlement": self.pending_settlement_amount,
            "positions_value": sum(p.market_value for p in self._positions.values()),
            "total_value": self.total_value,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "open_positions": len(self._positions),
            "closed_trades": len(self._closed_trades),
        }
