"""
NSE circuit-breaker detection module.

Detects whether an Indian stock has hit its daily circuit-limit band:
  • Lower circuit: stock fell to the exchange-imposed floor
  • Upper circuit: stock rose to the exchange-imposed ceiling

NSE circuit bands vary by stock: ±2%, ±5%, ±10%, ±20%.
Index-level circuit breakers (market-wide halt) trigger at
±10%, ±15%, ±20% on Nifty50.

Two modes:
  1. **EOD scoring** — called from IntegratedScorer to penalise
     stocks stuck at lower-circuit (illiquid, can't exit).
  2. **Real-time** — called from the auto-executor to avoid placing
     orders on circuit-locked instruments.

Usage::

    from scrapers.ind_news.circuit_detector import CircuitDetector

    detector = CircuitDetector()
    result = detector.check(ticker="RELIANCE.NS", current_price=2450,
                            prev_close=2500, day_low=2450, day_high=2450)
    # result.hit_circuit → True
    # result.circuit_type → "lower"
    # result.band_pct → 5.0
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitType(str, Enum):
    NONE = "none"
    UPPER = "upper"
    LOWER = "lower"


# Standard NSE circuit-filter bands (percentage from previous close)
_BANDS = [2.0, 5.0, 10.0, 20.0]

# Index-level market-wide circuit breaker thresholds (Nifty50)
_INDEX_BANDS = [10.0, 15.0, 20.0]


@dataclass
class CircuitResult:
    """Result of a circuit-breaker check."""

    ticker: str
    hit_circuit: bool = False
    circuit_type: CircuitType = CircuitType.NONE
    band_pct: Optional[float] = None           # matched band (e.g. 5.0)
    change_pct: Optional[float] = None         # actual % change from prev close
    is_index_halt: bool = False                 # market-wide halt

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "hit_circuit": self.hit_circuit,
            "circuit_type": self.circuit_type.value,
            "band_pct": self.band_pct,
            "change_pct": self.change_pct,
            "is_index_halt": self.is_index_halt,
        }


class CircuitDetector:
    """
    Detects NSE daily circuit-limit hits for individual stocks
    and market-wide index halts.

    All inputs are plain floats — the caller is responsible for
    fetching OHLC / live-quote data from Kite or yfinance.
    """

    # Tolerance for float comparison (0.01% of prev_close)
    _TOLERANCE_FACTOR = 0.0001

    def check(
        self,
        ticker: str,
        current_price: float,
        prev_close: float,
        day_low: Optional[float] = None,
        day_high: Optional[float] = None,
    ) -> CircuitResult:
        """
        Check if *ticker* has hit a circuit limit today.

        Args:
            ticker:        NSE symbol (e.g. ``"RELIANCE.NS"``)
            current_price: Last traded price / LTP
            prev_close:    Previous day's closing price
            day_low:       Intraday low (helps confirm lower circuit)
            day_high:      Intraday high (helps confirm upper circuit)

        Returns:
            :class:`CircuitResult` with hit status and matched band.
        """
        result = CircuitResult(ticker=ticker)

        if not prev_close or prev_close <= 0:
            return result

        change_pct = ((current_price - prev_close) / prev_close) * 100
        result.change_pct = round(change_pct, 2)

        tol = prev_close * self._TOLERANCE_FACTOR

        for band in _BANDS:
            upper_limit = prev_close * (1 + band / 100)
            lower_limit = prev_close * (1 - band / 100)

            # Upper circuit: price at or very near the upper band
            if abs(current_price - upper_limit) <= tol:
                # Confirm with day_high if available
                if day_high is None or abs(day_high - upper_limit) <= tol:
                    result.hit_circuit = True
                    result.circuit_type = CircuitType.UPPER
                    result.band_pct = band
                    break

            # Lower circuit: price at or very near the lower band
            if abs(current_price - lower_limit) <= tol:
                if day_low is None or abs(day_low - lower_limit) <= tol:
                    result.hit_circuit = True
                    result.circuit_type = CircuitType.LOWER
                    result.band_pct = band
                    break

        if result.hit_circuit:
            logger.info(
                "Circuit HIT: %s %s circuit at ±%.0f%% (change=%.2f%%)",
                ticker, result.circuit_type.value, result.band_pct, change_pct,
            )

        return result

    def check_index_halt(
        self,
        index_name: str,
        current_level: float,
        prev_close: float,
    ) -> CircuitResult:
        """
        Check for a market-wide index circuit breaker (SEBI rules).

        Nifty50 halts:
          • ±10% — 45 min halt (before 1pm), 15 min (1pm-2:30pm), none after
          • ±15% — 1h45m halt (before 1pm), 45 min after
          • ±20% — full-day halt

        Returns a :class:`CircuitResult` with ``is_index_halt=True``
        if any threshold is breached.
        """
        result = CircuitResult(ticker=index_name)

        if not prev_close or prev_close <= 0:
            return result

        change_pct = ((current_level - prev_close) / prev_close) * 100
        result.change_pct = round(change_pct, 2)

        for band in _INDEX_BANDS:
            if abs(change_pct) >= band:
                result.hit_circuit = True
                result.is_index_halt = True
                result.band_pct = band
                result.circuit_type = (
                    CircuitType.UPPER if change_pct > 0 else CircuitType.LOWER
                )

        if result.is_index_halt:
            logger.warning(
                "INDEX HALT: %s breached ±%.0f%% (change=%.2f%%)",
                index_name, result.band_pct, change_pct,
            )

        return result

    def is_tradeable(
        self,
        ticker: str,
        current_price: float,
        prev_close: float,
        day_low: Optional[float] = None,
        day_high: Optional[float] = None,
    ) -> bool:
        """
        Convenience method for the auto-executor: returns ``False`` if
        the stock is at a circuit limit (meaning orders would be stuck).
        """
        cr = self.check(ticker, current_price, prev_close, day_low, day_high)
        return not cr.hit_circuit
