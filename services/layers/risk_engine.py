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
        max_deployment_cap: float = 20_000.0,
        volatility_target=None,
    ):
        self.total_capital = total_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_open_positions = max_open_positions
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self.max_deployment_cap = max_deployment_cap
        self._open_positions: Dict[str, dict] = {}
        self._realized_pnl: float = 0.0  # Track realized losses for drawdown

        # Carver volatility target — drives capital rolling
        self._vol_target = volatility_target  # VolatilityTarget instance

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

        # ── Portfolio drawdown enforcement ──
        current_dd = self._compute_portfolio_drawdown()
        if current_dd >= self.max_portfolio_drawdown_pct:
            result = RiskCheckResult(
                ticker=ticker, passed=False,
                reason=(
                    f"Portfolio drawdown {current_dd:.1f}% exceeds limit "
                    f"({self.max_portfolio_drawdown_pct:.0f}%) — blocking new trades"
                ),
            )
            event_bus.emit("risk.check_failed", payload=result.__dict__, source="risk_engine")
            logger.warning("Portfolio drawdown breach: %.1f%% — trade blocked for %s", current_dd, ticker)
            return result

        # ── Max deployment cap ──
        total_deployed = sum(
            pos["qty"] * pos["entry"] for pos in self._open_positions.values()
        )
        max_deploy = min(self.max_deployment_cap, self.total_capital * 0.80)
        if total_deployed >= max_deploy and ticker not in self._open_positions:
            result = RiskCheckResult(
                ticker=ticker, passed=False,
                reason=f"Capital deployment ₹{total_deployed:,.0f} >= ₹{max_deploy:,.0f} cap",
            )
            event_bus.emit("risk.check_failed", payload=result.__dict__, source="risk_engine")
            return result

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

        # ── Circuit limit check ──
        # Block trades on stocks at/near circuit limits (order unlikely to fill)
        if price > 0:
            try:
                import yfinance as yf
                ns = ticker if "." in ticker else f"{ticker}.NS"
                hist = yf.download(ns, period="2d", progress=False, auto_adjust=True)
                if hasattr(hist, 'columns') and isinstance(hist.columns, __import__('pandas').MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                if hist is not None and len(hist) >= 2:
                    prev_close = float(hist["Close"].iloc[-2])
                    if prev_close > 0:
                        daily_change_pct = (price - prev_close) / prev_close * 100
                        if self.check_circuit_limit(ticker, daily_change_pct):
                            result = RiskCheckResult(
                                ticker=ticker, passed=False,
                                reason=f"Circuit limit detected: {daily_change_pct:.1f}% change — order unlikely to fill",
                            )
                            event_bus.emit("risk.check_failed", payload=result.__dict__, source="risk_engine")
                            return result
            except Exception as exc:
                logger.debug("Circuit limit check failed for %s: %s", ticker, exc)

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

    def update_position_ltp(self, ticker: str, ltp: float) -> None:
        """Update the last-traded-price for an open position (for drawdown computation)."""
        if ticker in self._open_positions:
            self._open_positions[ticker]["ltp"] = ltp

    def close_position(self, ticker: str, exit_price: Optional[float] = None) -> None:
        pos = self._open_positions.pop(ticker, None)
        if pos and exit_price is not None:
            realized = (exit_price - pos["entry"]) * pos["qty"]
            self._realized_pnl += realized
            # Feed capital rolling into VolatilityTarget
            if self._vol_target is not None:
                self._vol_target.add_realized(realized)

    def _sync_vol_target_unrealized(self) -> None:
        """Push current unrealised P&L into VolatilityTarget for capital rolling."""
        if self._vol_target is None:
            return
        total_unrealized = 0.0
        for pos in self._open_positions.values():
            ltp = pos.get("ltp", pos["entry"])
            total_unrealized += (ltp - pos["entry"]) * pos["qty"]
        self._vol_target.set_unrealized(total_unrealized)

    def _compute_portfolio_drawdown(self) -> float:
        """Compute current portfolio drawdown as a percentage of total capital.

        Includes both unrealised P&L on open positions AND realized losses
        from closed positions in the current session.
        Returns 0.0 when no losses.
        """
        total_pnl = self._realized_pnl  # start with realized losses
        for pos in self._open_positions.values():
            ltp = pos.get("ltp", pos["entry"])  # default to entry if no LTP yet
            total_pnl += (ltp - pos["entry"]) * pos["qty"]
        if total_pnl >= 0:
            return 0.0
        return abs(total_pnl) / self.total_capital * 100

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)
