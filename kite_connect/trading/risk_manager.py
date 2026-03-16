"""
Risk Management Module for NSE Auto-Trading.

Computes position sizes, stop-loss levels, and profit targets for
screened stocks before order placement.  Core rules:

• **Stop-Loss**: Below the 50-day MA *or* the recent swing low
  (whichever is tighter).
• **Risk Control**: Max 1–2 % of trading capital risked per trade.
• **Exit Strategy**: Target profit at the next resistance level or
  when momentum (RSI) reverses above 70.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class RiskConfig:
    """Tuneable risk parameters."""

    total_capital: float = 500_000.0      # Total trading capital (₹)
    risk_per_trade_pct: float = 0.02      # Max 2 % of capital per trade
    max_open_trades: int = 10             # Portfolio-level concentration cap
    sl_method: str = "tighter"            # "ma50", "swing_low", "tighter"
    swing_lookback: int = 10              # Days for swing-low computation
    min_rr_ratio: float = 2.0            # Minimum reward-to-risk ratio
    use_kelly_sizing: bool = True         # Scale risk by signal confidence (half-Kelly)
    kelly_floor_pct: float = 0.01         # Minimum risk (1%) for low-confidence
    kelly_cap_pct: float = 0.03           # Maximum risk (3%) for high-confidence


# ═══════════════════════════════════════════════════════════════
# Trade plan produced for each qualifying stock
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradePlan:
    symbol: str
    side: str                # "BUY"
    entry_price: float
    stop_loss: float
    target_price: float
    quantity: int
    risk_amount: float       # ₹ at risk
    reward_amount: float     # ₹ potential gain
    rr_ratio: float          # reward / risk
    score: float             # from screener

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_price": round(self.target_price, 2),
            "quantity": self.quantity,
            "risk_amount": round(self.risk_amount, 2),
            "reward_amount": round(self.reward_amount, 2),
            "rr_ratio": round(self.rr_ratio, 2),
            "score": round(self.score, 2),
        }


# ═══════════════════════════════════════════════════════════════
# Risk Manager
# ═══════════════════════════════════════════════════════════════

class RiskManager:
    """Converts a screened-stock DataFrame into a list of *TradePlan* objects."""

    def __init__(self, config: RiskConfig | None = None, kite=None):
        self.cfg = config or RiskConfig()
        self.kite = kite

    def _available_capital(self) -> float:
        """Return capital remaining after deducting open positions."""
        if self.kite is None:
            return self.cfg.total_capital
        try:
            from kite_connect.trading.order_service import get_positions
            positions = get_positions(self.kite)
            deployed = sum(
                abs(float(p.get("quantity", 0))) * float(p.get("average_price", 0))
                for p in positions.get("net", [])
                if float(p.get("quantity", 0)) != 0
            )
            available = max(0, self.cfg.total_capital - deployed)
            logger.info("Capital: %.0f total, %.0f deployed, %.0f available",
                        self.cfg.total_capital, deployed, available)
            return available
        except Exception as exc:
            logger.warning("Could not fetch positions for capital calc: %s", exc)
            return self.cfg.total_capital

    def plan_trades(
        self,
        screened_df: pd.DataFrame,
        max_trades: Optional[int] = None,
    ) -> List[TradePlan]:
        """
        Generate trade plans for the top-ranked screened stocks.

        Parameters
        ----------
        screened_df : pd.DataFrame
            Output of :meth:`NSEScreener.screen` (must include
            ``symbol, close, score, ma_50, support, resistance``).
        max_trades : int | None
            Override for ``max_open_trades``.

        Returns
        -------
        list[TradePlan]
            Ready-to-execute plans, sorted by score descending.
        """
        if screened_df.empty:
            return []

        limit = max_trades or self.cfg.max_open_trades
        available = self._available_capital()
        plans: List[TradePlan] = []

        for _, row in screened_df.head(limit).iterrows():
            if available <= 0:
                logger.info("No capital remaining — stopping plan generation")
                break
            plan = self._build_plan(row, available)
            if plan is not None:
                plans.append(plan)
                # Deduct allocated capital
                available -= plan.quantity * plan.entry_price

        logger.info("Generated %d trade plans (top-%d)", len(plans), limit)
        return plans

    # ── Internal ───────────────────────────────────────────────

    def _build_plan(self, row: pd.Series, available_capital: float) -> Optional[TradePlan]:
        entry = float(row["close"])
        ma50 = float(row.get("ma_50", 0))
        support = float(row.get("support", 0))
        resistance = float(row.get("resistance", 0))
        score = float(row.get("score", 0))

        # ── Stop-loss ──────────────────────────────────────────
        sl_ma50 = ma50 if ma50 > 0 else entry * 0.95
        sl_swing = support if support > 0 else entry * 0.95

        if self.cfg.sl_method == "ma50":
            sl = sl_ma50
        elif self.cfg.sl_method == "swing_low":
            sl = sl_swing
        else:  # "tighter" — whichever is closer to entry (smaller loss)
            sl = max(sl_ma50, sl_swing)

        # SL must be below entry
        if sl >= entry:
            sl = entry * 0.97  # 3 % hard floor

        # ── Target price ───────────────────────────────────────
        target = resistance if resistance > entry else entry * 1.10

        # ── R:R ratio ──────────────────────────────────────────
        risk_per_share = entry - sl
        reward_per_share = target - entry

        if risk_per_share <= 0:
            return None

        rr_ratio = reward_per_share / risk_per_share
        if rr_ratio < self.cfg.min_rr_ratio:
            return None

        # ── Position sizing (Kelly-scaled or fixed risk) ───────
        if self.cfg.use_kelly_sizing:
            # Half-Kelly: scale risk_pct by confidence (score 0-1)
            # score from screener is typically 0-100, normalize
            confidence = min(abs(score) / 100.0, 1.0) if abs(score) > 1 else abs(score)
            half_kelly_pct = (
                self.cfg.kelly_floor_pct
                + confidence * (self.cfg.kelly_cap_pct - self.cfg.kelly_floor_pct)
            )
            effective_risk_pct = half_kelly_pct
        else:
            effective_risk_pct = self.cfg.risk_per_trade_pct

        max_risk = available_capital * effective_risk_pct
        qty = max(1, math.floor(max_risk / risk_per_share))
        # Also cap by available capital (can't exceed capital / entry)
        max_qty_cap = math.floor(available_capital / entry) if entry > 0 else 0
        qty = min(qty, max_qty_cap)

        risk_amount = qty * risk_per_share
        reward_amount = qty * reward_per_share

        return TradePlan(
            symbol=row["symbol"],
            side="BUY",
            entry_price=entry,
            stop_loss=sl,
            target_price=target,
            quantity=qty,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            rr_ratio=rr_ratio,
            score=score,
        )
