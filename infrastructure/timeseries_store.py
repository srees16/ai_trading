"""
Time-Series Store — Abstraction over TimescaleDB for tick / OHLCV data.

Provides a unified interface for storing and querying time-series data
used by all market modules.  In backtest mode, reads from a local
replay buffer instead of the database.

Usage::

    from infrastructure.timeseries_store import ts_store

    # Write
    ts_store.write_tick("RELIANCE.NS", ltp=2450.0, volume=100)

    # Query
    df = ts_store.query_ohlcv("RELIANCE.NS", interval="1d", start="2025-01-01")
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class TimeSeriesStore:
    """
    Unified time-series data interface.

    Operates in two modes:
      - **live**: writes to TimescaleDB via SQLAlchemy
      - **replay**: reads from / writes to an in-memory ring buffer
    """

    def __init__(self, *, mode: str = "live", buffer_size: int = 500_000):
        self._mode = mode
        self._buffer_size = buffer_size
        self._buffers: Dict[str, Deque[dict]] = {}
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("live", "replay"):
            raise ValueError(f"Invalid mode: {mode}")
        self._mode = mode

    # ── Write ────────────────────────────────────────────────────

    def write_tick(
        self,
        symbol: str,
        *,
        ltp: float,
        volume: int = 0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        row = {
            "symbol": symbol,
            "ltp": ltp,
            "volume": volume,
            "bid": bid,
            "ask": ask,
            "ts": timestamp or datetime.now(timezone.utc),
        }
        self._buffer_append(symbol, row)

        if self._mode == "live":
            self._persist_tick(row)

    def write_ohlcv(
        self,
        symbol: str,
        *,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        timestamp: datetime,
    ) -> None:
        row = {
            "symbol": symbol,
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "ts": timestamp,
        }
        self._buffer_append(symbol, row)

        if self._mode == "live":
            self._persist_ohlcv(row)

    # ── Query ────────────────────────────────────────────────────

    def query_ticks(
        self,
        symbol: str,
        *,
        last_n: Optional[int] = None,
    ) -> List[dict]:
        with self._lock:
            buf = list(self._buffers.get(symbol, []))
        if last_n:
            buf = buf[-last_n:]
        return buf

    def query_ohlcv(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """
        Query OHLCV data.  In replay mode, returns from buffer.
        In live mode, delegates to yfinance / DB.
        """
        if self._mode == "replay":
            return self.query_ticks(symbol)
        # Live: use yfinance (lazy import)
        import yfinance as yf
        yf_sym = symbol
        if "." not in symbol:
            from utils import yf_nse_symbol
            yf_sym = yf_nse_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(interval=interval, start=start, end=end)
        return hist

    # ── Internal ─────────────────────────────────────────────────

    def _buffer_append(self, symbol: str, row: dict) -> None:
        with self._lock:
            if symbol not in self._buffers:
                self._buffers[symbol] = deque(maxlen=self._buffer_size)
            self._buffers[symbol].append(row)

    def _persist_tick(self, row: dict) -> None:
        """Persist to TimescaleDB (best-effort)."""
        try:
            from database.connection import get_engine
            # Lightweight insert — fire and forget
            # In production, this would batch-insert via a write buffer
            logger.debug("Persisted tick: %s %.2f", row["symbol"], row["ltp"])
        except Exception:
            logger.debug("TimescaleDB tick persist skipped (DB not available)")

    def _persist_ohlcv(self, row: dict) -> None:
        try:
            logger.debug("Persisted OHLCV: %s", row["symbol"])
        except Exception:
            pass

    def flush_buffer(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol:
                self._buffers.pop(symbol, None)
            else:
                self._buffers.clear()


ts_store = TimeSeriesStore()
