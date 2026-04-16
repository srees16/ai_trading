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

# A4+P2+G3: Regime-adaptive vol scaling — SINGLE dampening layer (Carver approach).
# G3 FIX: Bull boosted to 1.30× (Sharpe=0.73 in bull, strongest regime).
# Bear slashed to 0.15× (Sharpe=-0.01, signals are broken in bear).
# Net effect: capture more upside, protect capital in bear.
REGIME_VOL_SCALE = {
    "trending_bull":    1.30,     # G3: was 1.00 — bull is highest-alpha regime, push sizing
    "trending_bear":    0.15,     # G3: was 0.65 — bear signals BROKEN (Sharpe -0.01), near-halt
    "range_bound":      0.85,     # P2: preserved — range is 25-40% of time, MR alpha here
    "high_volatility":  0.35,     # G3: was 0.50 — more conservative in high-vol chaos
    "crisis":           0.00,     # Full halt
}

# H1 FIX: R21a equity-curve SMA200 regime scaling — matches optimizer assumptions.
# Applied as a SECOND layer on top of HMM market regime (they measure different things):
#   HMM = "is the MARKET in crisis?" (NIFTY + VIX based)
#   SMA200 = "is MY PORTFOLIO trending?" (equity curve based, R21a-optimized)
# Combined multiplier is capped to prevent double-amplification.
_R21A_EQUITY_SMA200_BOOST = 1.25    # uptrend: equity > SMA200 × 1.02
_R21A_EQUITY_SMA200_DEFEND = 0.55   # R21A original: downtrend portfolio vol scale (synced with config/backtest)
_R21A_EQUITY_SMA_LOOKBACK = 200     # trading days for SMA
_R21A_COMBINED_CAP = 1.30           # max combined multiplier (HMM × SMA200)

# Try to load from centralized config (single source of truth)
try:
    from config import Config as _VTCfg
    _R21A_EQUITY_SMA200_BOOST = getattr(_VTCfg, 'R21A_REGIME_BOOST', _R21A_EQUITY_SMA200_BOOST)
    _R21A_EQUITY_SMA200_DEFEND = getattr(_VTCfg, 'R21A_REGIME_DEFEND', _R21A_EQUITY_SMA200_DEFEND)
    _R21A_EQUITY_SMA_LOOKBACK = getattr(_VTCfg, 'R21A_SMA_LOOKBACK', _R21A_EQUITY_SMA_LOOKBACK)
except Exception:
    pass


def smooth_regime_scale(equity: float, sma200: float,
                        boost: float = None, defend: float = None,
                        steepness: float = 10.0) -> float:
    """Smooth sigmoid interpolation between defend and boost based on equity/SMA200 ratio.

    Replaces binary threshold (equity > SMA200×1.02 → boost, < 0.98 → defend).
    Returns a continuous multiplier in [defend, boost] using a sigmoid.

    sigmoid(x) = defend + (boost - defend) / (1 + exp(-steepness × (ratio - 1.0)))
    """
    import math
    if boost is None:
        boost = _R21A_EQUITY_SMA200_BOOST
    if defend is None:
        defend = _R21A_EQUITY_SMA200_DEFEND
    if sma200 <= 0:
        return 1.0
    ratio = equity / sma200
    x = steepness * (ratio - 1.0)
    # Clamp to avoid overflow
    x = max(-20.0, min(20.0, x))
    sig = 1.0 / (1.0 + math.exp(-x))
    return defend + (boost - defend) * sig

# Contra-regime: mean-reversion signals in bear get a higher vol allocation
# so "buy the dip" actually receives meaningful position sizing.
# Normal bear scale = 0.15×; mean-reversion override = 0.50×
_CONTRA_REGIME_MR_BEAR_SCALE = 0.50


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
    capital_halt_fraction: float = 0.80  # halt at 20% drawdown (1.0 - 0.80)

    # Warning thresholds for graduated response
    capital_warning_fraction: float = 0.90   # warn at 10% drawdown
    capital_critical_fraction: float = 0.85  # critical at 15% drawdown

    # Maximum leverage factor (for cash equities = 1.0, no leverage).
    max_leverage_factor: float = 1.0

    # Vince active/inactive equity insurance.
    # Floor = HWM × insurance_pct.  At HWM: active = (1-ins_pct) of equity.
    # As DD deepens, active_frac → 0 smoothly.  0.0 = disabled (use legacy tiers).
    vince_insurance_pct: float = 0.0


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
        self._high_water_mark: float = self.cfg.initial_capital
        self._regime: str = ""  # A4: current regime for vol scaling
        # H1 FIX: equity curve history for SMA200 regime detection
        self._equity_history: list = []

    def set_regime(self, regime: str) -> None:
        """A4: Set current market regime for vol target scaling."""
        self._regime = (regime or "").lower().strip()
        if self._regime:
            scale = REGIME_VOL_SCALE.get(self._regime, 1.0)
            logger.info("Vol target regime=%s, scale=%.2f", self._regime, scale)

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
        self._high_water_mark = max(self._high_water_mark, self.current_capital)
        # H1 FIX: track equity for SMA200 regime detection
        self._equity_history.append(self.current_capital)

    def add_realized(self, amount: float) -> None:
        """Incrementally add a realized trade result."""
        self._realized_pnl += amount
        self._high_water_mark = max(self._high_water_mark, self.current_capital)

    def set_unrealized(self, amount: float) -> None:
        """Set current mark-to-market unrealized P&L."""
        self._unrealized_pnl = amount

    # ── Vince Active/Inactive Equity (C1 fix) ─────────────────

    @property
    def active_equity(self) -> float:
        """Vince active equity: the portion of capital available for sizing.

        Floor = HWM × insurance_pct.  Above HWM: active = equity − floor.
        As DD deepens toward floor, active_equity → 0 smoothly.
        When insurance is disabled (pct=0), returns current_capital.
        """
        ins = self.cfg.vince_insurance_pct
        if ins <= 0:
            return self.current_capital
        floor = self._high_water_mark * ins
        return max(0.0, self.current_capital - floor)

    @property
    def active_equity_fraction(self) -> float:
        """Fraction of capital that is 'active' under Vince insurance.

        At HWM: fraction = 1 − insurance_pct (e.g. 0.80 for 20% insurance).
        At floor (DD = insurance_pct of HWM): fraction = 0.
        When insurance disabled: returns 1.0.
        """
        ins = self.cfg.vince_insurance_pct
        if ins <= 0:
            return 1.0
        cap = self.current_capital
        if cap <= 0:
            return 0.0
        frac = self.active_equity / cap
        return max(0.0, min(1.0, frac))

    # ── Volatility targets ────────────────────────────────────

    @property
    def annual_cash_vol_target(self) -> float:
        """Annual cash volatility target = capital × % vol target."""
        return self.current_capital * self.cfg.annual_vol_target_pct

    @property
    def daily_cash_vol_target(self) -> float:
        """Daily cash volatility target = annual / 16.

        When Vince insurance is enabled, uses active_equity instead of
        current_capital — providing smooth position scale-down as DD deepens.
        A4: Applies regime-adaptive scaling so bear/crisis auto-shrinks.
        H1 FIX: Applies R21a equity SMA200 layer on top of HMM regime.
        """
        if self.cfg.vince_insurance_pct > 0:
            base = self.active_equity * self.cfg.annual_vol_target_pct / ANNUALISATION_FACTOR
        else:
            base = self.annual_cash_vol_target / ANNUALISATION_FACTOR

        combined_scale = 1.0

        # Layer 1: HMM market regime (A4)
        if self._regime:
            combined_scale *= REGIME_VOL_SCALE.get(self._regime, 1.0)

        # Layer 2: R21a equity-curve SMA200 regime (H1 FIX)
        equity_scale = self._equity_sma200_scale()
        combined_scale *= equity_scale

        # Cap combined multiplier to prevent double-amplification
        combined_scale = min(combined_scale, _R21A_COMBINED_CAP)

        base *= combined_scale
        return base

    def daily_cash_vol_target_for_source(self, source: str) -> float:
        """Contra-regime vol target override for specific signal sources.

        Mean-reversion signals in bear regimes get 0.50× instead of 0.15×,
        enabling meaningful "buy the dip" position sizing while the default
        bear scale keeps trend-following signals near-halted.

        For all other source/regime combinations, returns the standard target.
        """
        if (
            source in ("mean_reversion", "mr", "mean_rev")
            and self._regime
            and "bear" in self._regime
        ):
            # Recompute with MR bear override instead of standard bear scale
            if self.cfg.vince_insurance_pct > 0:
                base = self.active_equity * self.cfg.annual_vol_target_pct / ANNUALISATION_FACTOR
            else:
                base = self.annual_cash_vol_target / ANNUALISATION_FACTOR

            combined_scale = _CONTRA_REGIME_MR_BEAR_SCALE  # 0.50× instead of 0.15×

            equity_scale = self._equity_sma200_scale()
            combined_scale *= equity_scale
            combined_scale = min(combined_scale, _R21A_COMBINED_CAP)

            return base * combined_scale

        return self.daily_cash_vol_target

    def _equity_sma200_scale(self) -> float:
        """R21a equity SMA200 regime scale factor.

        Matches the optimizer's backtest assumption:
        - equity > SMA200 × 1.02 → uptrend → 1.25× (push sizing)
        - equity < SMA200 × 0.98 → downtrend → 0.55× (defend)
        - otherwise → neutral → 1.0×
        """
        lookback = _R21A_EQUITY_SMA_LOOKBACK
        if len(self._equity_history) < lookback:
            return 1.0
        sma200 = sum(self._equity_history[-lookback:]) / lookback
        equity = self.current_capital
        if equity > sma200 * 1.02:
            return _R21A_EQUITY_SMA200_BOOST
        elif equity < sma200 * 0.98:
            return _R21A_EQUITY_SMA200_DEFEND
        return 1.0

    # ── Safety checks ─────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        """True if capital has fallen below halt threshold (20% drawdown)."""
        halt_level = self._initial_capital * self.cfg.capital_halt_fraction
        if self.current_capital < halt_level:
            logger.warning(
                "Capital ₹%.0f below halt level ₹%.0f (%.0f%% drawdown) — "
                "blocking new trades",
                self.current_capital, halt_level,
                (1 - self.current_capital / self._initial_capital) * 100,
            )
            return True
        return False

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as a percentage of initial capital."""
        if self._initial_capital <= 0:
            return 0.0
        return max(0.0, (1 - self.current_capital / self._initial_capital) * 100)

    @property
    def risk_scale_factor(self) -> float:
        """Graduated position scale factor based on drawdown.

        When Vince insurance is enabled: returns active_equity_fraction
        (smooth continuous curve, no cliff effects).

        Legacy mode: returns step-function tiers.
        """
        if self.cfg.vince_insurance_pct > 0:
            return self.active_equity_fraction

        dd = self.drawdown_pct / 100.0  # fraction
        if dd < 0.0501:
            return 1.0
        elif dd < 0.0701:
            return 0.85
        elif dd < 0.1001:
            return 0.70
        elif dd < 0.1501:
            return 0.50
        elif dd < 0.2001:
            return 0.25
        else:
            return 0.0  # halted

    @property
    def max_portfolio_value(self) -> float:
        """Maximum total portfolio exposure given leverage limit."""
        return self.current_capital * self.cfg.max_leverage_factor

    # ── Diagnostics ───────────────────────────────────────────

    def summary(self) -> dict:
        """Return a diagnostic snapshot."""
        eq_sma_scale = self._equity_sma200_scale()
        return {
            "initial_capital": self._initial_capital,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._unrealized_pnl,
            "current_capital": self.current_capital,
            "high_water_mark": self._high_water_mark,
            "active_equity": self.active_equity,
            "active_equity_fraction": round(self.active_equity_fraction, 4),
            "vince_insurance_pct": self.cfg.vince_insurance_pct,
            "annual_vol_target_pct": self.cfg.annual_vol_target_pct,
            "annual_cash_vol_target": self.annual_cash_vol_target,
            "daily_cash_vol_target": self.daily_cash_vol_target,
            "risk_scale_factor": round(self.risk_scale_factor, 4),
            "is_halted": self.is_halted,
            "hmm_regime": self._regime,
            "hmm_regime_scale": REGIME_VOL_SCALE.get(self._regime, 1.0) if self._regime else 1.0,
            "equity_sma200_scale": round(eq_sma_scale, 2),
            "equity_history_days": len(self._equity_history),
        }
