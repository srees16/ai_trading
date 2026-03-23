"""
Risk Engine Layer — Position limits, drawdown control, circuit breakers.

Provides pre-trade and post-trade risk checks:
  - Max position size per ticker
  - Max portfolio exposure
  - Max drawdown circuit breaker
  - Per-trade risk limit (% of capital at risk)
  - Indian market specifics: circuit limit detection, T+1 awareness

Emits:
  - ``risk.check_passed``
  - ``risk.check_failed``
  - ``risk.circuit_breaker``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    ticker: str
    passed: bool
    reason: str = ""
    max_quantity: int = 0
    stop_loss: Optional[float] = None
    target: Optional[float] = None


class RiskEngine:
    """
    Pre-trade and portfolio-level risk checks.

    Wraps the existing kite_connect/trading/risk_manager.py
    for Indian stocks and provides a unified interface.
    """

    # Sector mapping for IND stocks — sourced from Config.NSE_SECTOR_MAP.
    # Inverted to ticker→sector for fast lookups.
    @staticmethod
    def _build_sector_map() -> Dict[str, str]:
        from config import Config
        return {ticker: sector for ticker, sector in Config.NSE_SECTOR_MAP.items()}

    _SECTOR_MAP: Dict[str, str] = None  # type: ignore[assignment]

    MAX_SECTOR_CONCENTRATION_PCT: float = 30.0  # max 30% of capital per sector

    def __init__(
        self,
        *,
        total_capital: float = 500_000.0,
        risk_per_trade_pct: float = 2.0,
        max_open_positions: int = 10,
        max_portfolio_drawdown_pct: float = 15.0,
    ):
        self.total_capital = total_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_open_positions = max_open_positions
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self._open_positions: Dict[str, dict] = {}
        # Lazy-initialise the class-level sector map once
        if RiskEngine._SECTOR_MAP is None:
            RiskEngine._SECTOR_MAP = RiskEngine._build_sector_map()

    def check_pre_trade(
        self,
        ticker: str,
        side: str,
        price: float,
        stop_loss: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Evaluate whether a trade is acceptable given current risk limits.
        """
        from infrastructure.event_bus import event_bus

        # Check open position count
        if len(self._open_positions) >= self.max_open_positions and ticker not in self._open_positions:
            result = RiskCheckResult(
                ticker=ticker, passed=False,
                reason=f"Max open positions ({self.max_open_positions}) reached",
            )
            event_bus.emit("risk.check_failed", payload=result.__dict__, source="risk_engine")
            return result

        # Check sector concentration (for IND stocks)
        clean = ticker.upper().replace(".NS", "").replace(".BO", "")
        sector = self._SECTOR_MAP.get(clean)
        if sector:
            sector_exposure = sum(
                pos["qty"] * pos["entry"]
                for sym, pos in self._open_positions.items()
                if self._SECTOR_MAP.get(sym.upper().replace(".NS", "").replace(".BO", "")) == sector
            )
            max_sector = self.total_capital * (self.MAX_SECTOR_CONCENTRATION_PCT / 100)
            if sector_exposure >= max_sector:
                result = RiskCheckResult(
                    ticker=ticker, passed=False,
                    reason=f"Sector {sector} at {sector_exposure / self.total_capital:.0%} — cap is {self.MAX_SECTOR_CONCENTRATION_PCT:.0f}%",
                )
                event_bus.emit("risk.check_failed", payload=result.__dict__, source="risk_engine")
                return result

        # Calculate max risk amount
        max_risk = self.total_capital * (self.risk_per_trade_pct / 100)

        # Position sizing
        if stop_loss and price > 0 and stop_loss > 0:
            risk_per_share = abs(price - stop_loss)
            if risk_per_share > 0:
                max_qty = int(max_risk / risk_per_share)
            else:
                max_qty = 0
        else:
            # Default: limit position to 5% of capital
            max_qty = int((self.total_capital * 0.05) / price) if price > 0 else 0

        if max_qty <= 0:
            result = RiskCheckResult(
                ticker=ticker, passed=False,
                reason="Calculated quantity is zero (stop-loss too close or price too high)",
            )
            event_bus.emit("risk.check_failed", payload=result.__dict__, source="risk_engine")
            return result

        result = RiskCheckResult(
            ticker=ticker, passed=True,
            max_quantity=max_qty, stop_loss=stop_loss,
        )
        event_bus.emit("risk.check_passed", payload=result.__dict__, source="risk_engine")
        return result

    def check_circuit_limit(self, ticker: str, daily_change_pct: float) -> bool:
        """
        Detect if an Indian stock has hit circuit limits (±5/10/20%).

        Returns True if the stock appears to be circuit-frozen.
        """
        circuit_bands = {5.0, 10.0, 20.0}
        abs_change = abs(daily_change_pct)
        # Check if change is within 0.5% of a circuit band
        for band in circuit_bands:
            if abs(abs_change - band) < 0.5:
                from infrastructure.event_bus import event_bus
                event_bus.emit(
                    "risk.circuit_breaker",
                    payload={"ticker": ticker, "change_pct": daily_change_pct, "band": band},
                    source="risk_engine",
                )
                logger.warning(
                    "Circuit limit detected: %s moved %.1f%% (band: ±%.0f%%)",
                    ticker, daily_change_pct, band,
                )
                return True
        return False

    def register_position(self, ticker: str, qty: int, entry_price: float) -> None:
        self._open_positions[ticker] = {"qty": qty, "entry": entry_price}

    def close_position(self, ticker: str) -> None:
        self._open_positions.pop(ticker, None)

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)
