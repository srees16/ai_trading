"""
Dual-Mode Execution Context.

Ensures the **same code path** serves both live trading and simulation.
In backtest mode, market data comes from historical event replay;
in live mode, it comes from real-time feeds.

Usage::

    from infrastructure.execution_context import execution_ctx

    execution_ctx.set_mode("backtest")
    print(execution_ctx.is_live)  # False

    # Components check mode:
    if execution_ctx.is_live:
        broker.place_order(...)
    else:
        paper_broker.place_order(...)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class ExecutionContext:
    """
    Global execution mode container.

    Modes:
        - ``live``: real orders, real market data
        - ``paper``: real market data, simulated orders
        - ``backtest``: historical replay, simulated orders
    """

    VALID_MODES = ("live", "paper", "backtest")

    def __init__(self):
        self._mode: str = "paper"  # safe default
        self._lock = threading.Lock()
        self._simulation_clock: Optional[datetime] = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_live(self) -> bool:
        return self._mode == "live"

    @property
    def is_backtest(self) -> bool:
        return self._mode == "backtest"

    @property
    def is_paper(self) -> bool:
        return self._mode == "paper"

    def set_mode(self, mode: str) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}")
        with self._lock:
            old = self._mode
            self._mode = mode
        if old != mode:
            logger.info("Execution mode changed: %s → %s", old, mode)

    @property
    def now(self) -> datetime:
        """
        Current timestamp — returns simulation clock in backtest mode,
        real wall clock otherwise.
        """
        if self._mode == "backtest" and self._simulation_clock is not None:
            return self._simulation_clock
        return datetime.now(timezone.utc)

    def set_simulation_clock(self, ts: datetime) -> None:
        self._simulation_clock = ts

    def reset_simulation_clock(self) -> None:
        self._simulation_clock = None


execution_ctx = ExecutionContext()
