"""
Volatility Target — Portfolio-level risk budgeting (Carver Ch. 9).

Sets a single portfolio-level *percentage volatility target* that drives
ALL position sizing.  The key insight from Carver: your volatility target
should be at the Half-Kelly level — i.e. half of your realistic expected
Sharpe ratio.

For centurion_core swing/positional with expected SR ≈ 0.30–0.50:
  - Half-Kelly → 15 %–25 % annualised percentage volatility target.
  - Default: **20 %** (conservative for swing equity, no leverage).

The daily cash volatility target is:
  ``daily_cash_vol = capital × annual_pct_vol / 16``
  (16 = √256 ≈ annualisation factor for business days)

Capital rolls daily:  ``capital_today = initial + cumulative_pnl``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

ANNUALISATION_FACTOR = 16  # sqrt(256)


@dataclass
class VolatilityTargetConfig:
    """Tuneable volatility-target parameters."""

    initial_capital: float = 500_000.0
    # Annual percentage volatility target (Carver Half-Kelly).
    # 20 % is suitable for a swing/positional equity system with
    # expected SR ≈ 0.40.  Conservative; no leverage assumed.
    annual_vol_target_pct: float = 0.20

    # Minimum capital floor — if rolling capital drops below this
    # fraction of initial capital, halt new trades entirely.
    capital_halt_fraction: float = 0.50  # halt at 50 % drawdown

    # Maximum leverage factor (for cash equities = 1.0, no leverage).
    max_leverage_factor: float = 1.0


class VolatilityTarget:
    """Portfolio-level volatility target with daily capital rolling.

    Usage::

        vt = VolatilityTarget(VolatilityTargetConfig(initial_capital=500_000))
        vt.update_pnl(realized=1200, unrealized=-500)
        daily_target = vt.daily_cash_vol_target
        # Use daily_target in position sizing: vol_scalar = daily_target / instr_value_vol
    """

    def __init__(self, config: Optional[VolatilityTargetConfig] = None):
        self.cfg = config or VolatilityTargetConfig()
        self._initial_capital = self.cfg.initial_capital
        self._realized_pnl: float = 0.0
        self._unrealized_pnl: float = 0.0

    # ── Capital tracking ──────────────────────────────────────

    @property
    def current_capital(self) -> float:
        """Current trading capital = initial + cumulative P&L.

        Carver (Ch. 9 'Rolling up profits and losses'):
        the Kelly criterion implies you should adjust risk to current
        capital, not initial capital.
        """
        return max(0.0, self._initial_capital + self._realized_pnl + self._unrealized_pnl)

    def update_pnl(
        self,
        realized: float = 0.0,
        unrealized: float = 0.0,
    ) -> None:
        """Update cumulative P&L for capital rolling."""
        self._realized_pnl = realized
        self._unrealized_pnl = unrealized

    def add_realized(self, amount: float) -> None:
        """Incrementally add a realized trade result."""
        self._realized_pnl += amount

    def set_unrealized(self, amount: float) -> None:
        """Set current mark-to-market unrealized P&L."""
        self._unrealized_pnl = amount

    # ── Volatility targets ────────────────────────────────────

    @property
    def annual_cash_vol_target(self) -> float:
        """Annual cash volatility target = capital × % vol target."""
        return self.current_capital * self.cfg.annual_vol_target_pct

    @property
    def daily_cash_vol_target(self) -> float:
        """Daily cash volatility target = annual / 16.

        This is the single number that drives ALL position sizing
        in the Carver framework.
        """
        return self.annual_cash_vol_target / ANNUALISATION_FACTOR

    # ── Safety checks ─────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        """True if capital has fallen below halt threshold."""
        halt_level = self._initial_capital * self.cfg.capital_halt_fraction
        if self.current_capital < halt_level:
            logger.warning(
                "Capital ₹%.0f below halt level ₹%.0f (%.0f%% of initial) — "
                "blocking new trades",
                self.current_capital, halt_level,
                self.cfg.capital_halt_fraction * 100,
            )
            return True
        return False

    @property
    def max_portfolio_value(self) -> float:
        """Maximum total portfolio exposure given leverage limit."""
        return self.current_capital * self.cfg.max_leverage_factor

    # ── Diagnostics ───────────────────────────────────────────

    def summary(self) -> dict:
        """Return a diagnostic snapshot."""
        return {
            "initial_capital": self._initial_capital,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._unrealized_pnl,
            "current_capital": self.current_capital,
            "annual_vol_target_pct": self.cfg.annual_vol_target_pct,
            "annual_cash_vol_target": self.annual_cash_vol_target,
            "daily_cash_vol_target": self.daily_cash_vol_target,
            "is_halted": self.is_halted,
        }
