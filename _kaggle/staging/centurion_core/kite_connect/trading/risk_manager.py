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
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class RiskConfig:
    """Tuneable risk parameters."""

    total_capital: float = float(os.getenv("CENTURION_TOTAL_CAPITAL", "500000"))  # from env or ₹5L default
    risk_per_trade_pct: float = 0.02      # Max 2 % of capital per trade
    max_open_trades: int = 12             # C4: Synced with Config.MAX_OPEN_TRADES=12
    sl_method: str = "tighter"            # "ma50", "swing_low", "atr", "tighter"
    swing_lookback: int = 10              # Days for swing-low computation
    atr_multiplier: float = 2.0           # ATR × multiplier for ATR-based SL
    min_rr_ratio: float = 1.5            # Balanced R:R — Carver framework allows tighter targets
    use_kelly_sizing: bool = True         # Scale risk by signal confidence (half-Kelly)
    kelly_floor_pct: float = 0.01         # Minimum risk (1%) for low-confidence
    kelly_cap_pct: float = 0.03           # Maximum risk (3%) for high-confidence
    sl_min_pct: float = 0.05              # SL at least 5 % below entry
    sl_max_pct: float = 0.08              # SL at most 8 % below entry
    # R1: Sector concentration limits
    max_sector_exposure_pct: float = 0.30 # Max 30% capital in one sector (aligned with RiskEngine)
    max_trades_per_sector: int = 3        # Max 3 open trades per sector
    # R3: Trailing stop-loss — T1-4: ATR-scaled instead of fixed %
    trailing_sl_enabled: bool = True      # Enable trailing stop once profit > threshold
    trailing_sl_activation_pct: float = 0.05  # Activate trailing SL after 5% profit
    trailing_sl_distance_pct: float = 0.03    # Trail SL 3% below current price (fallback if ATR unavailable)
    trailing_sl_atr_enabled: bool = True      # T1-4: Use ATR-based trailing SL
    trailing_sl_atr_period: int = 14          # T1-4: ATR lookback period
    trailing_sl_atr_multiplier_bull: float = 2.5  # T1-4: N×ATR distance in bull regime
    trailing_sl_atr_multiplier_bear: float = 1.5  # T1-4: N×ATR distance in bear regime (tighter)
    # T1-5: Daily notional loss limit
    daily_notional_loss_limit_pct: float = 0.03  # T1-5: Halt if intraday P&L < -3% of capital
    # P7: Swing exit timing
    swing_max_hold_days: int = 15         # Force exit after 15 trading days (swing)
    positional_max_hold_days: int = 60    # Force exit after 60 trading days (positional)
    # R4: Market regime (VIX / ADX) scaling — unified with Config
    # GAP-4 FIX: Read from Config to ensure single source of truth
    vix_caution_threshold: float = 20.0   # VIX > 20 → scale position (synced with Config)
    vix_panic_threshold: float = 30.0     # VIX > 30 → block all new BUY orders (synced with Config)
    vix_caution_scale: float = 0.60       # 60% position size during caution
    adx_choppy_threshold: float = 20.0    # ADX < 20 → market choppy, scale down
    adx_choppy_scale: float = 0.50        # 50% position size in choppy market
    # Slippage buffer
    slippage_buffer_pct: float = 0.002    # 0.2% buffer for fill price uncertainty

    def __post_init__(self):
        """GAP-4: Sync VIX thresholds from Config (single source of truth)."""
        try:
            from config import Config
            self.vix_caution_threshold = getattr(Config, 'VIX_CAUTION_THRESHOLD', self.vix_caution_threshold)
            self.vix_panic_threshold = getattr(Config, 'VIX_PANIC_THRESHOLD', self.vix_panic_threshold)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Trade plan produced for each qualifying stock
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradePlan:
    symbol: str
    side: str                # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    target_price: float
    quantity: int
    risk_amount: float       # ₹ at risk
    reward_amount: float     # ₹ potential gain
    rr_ratio: float          # reward / risk
    score: float             # from screener
    direction: str = "LONG"  # "LONG" or "SHORT"
    product: str = "CNC"     # "CNC", "MIS", "NRML"
    execution_algo: Optional[str] = None  # T5-3: "TWAP", "VWAP", or None for direct

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
            "direction": self.direction,
            "product": self.product,
            "execution_algo": self.execution_algo,
        }


# ═══════════════════════════════════════════════════════════════
# Risk Manager
# ═══════════════════════════════════════════════════════════════

class RiskManager:
    """Converts a screened-stock DataFrame into a list of *TradePlan* objects."""

    def __init__(
        self,
        config: RiskConfig | None = None,
        kite=None,
        max_deployment_cap: float = 0,
        volatility_target=None,
    ):
        self.cfg = config or RiskConfig()
        self.kite = kite
        self._max_deployment_cap = max_deployment_cap  # 0 = no cap
        self._vol_target = volatility_target  # Carver VolatilityTarget instance

    def _available_capital(self) -> float:
        """Return capital remaining after deducting open positions.

        Respects max_deployment_cap from RiskEngine when set.
        """
        capital_ceiling = self.cfg.total_capital
        if self._max_deployment_cap > 0:
            capital_ceiling = min(capital_ceiling, self._max_deployment_cap)

        if self.kite is None:
            return capital_ceiling
        try:
            from kite_connect.trading.order_service import get_positions
            positions = get_positions(self.kite)
            deployed = sum(
                abs(float(p.get("quantity", 0))) * float(p.get("average_price", 0))
                for p in positions.get("net", [])
                if float(p.get("quantity", 0)) != 0
            )
            available = max(0, capital_ceiling - deployed)
            logger.info("Capital: %.0f ceiling (%.0f total), %.0f deployed, %.0f available",
                        capital_ceiling, self.cfg.total_capital, deployed, available)
            return available
        except Exception as exc:
            logger.warning("Could not fetch positions for capital calc: %s", exc)
            return capital_ceiling

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
        # R1: Track sector exposure
        sector_trade_count: Dict[str, int] = {}
        sector_capital: Dict[str, float] = {}

        # Portfolio-aware allocation: check existing sector weights
        portfolio_snap = None
        try:
            from services.portfolio_analyzer import PortfolioAnalyzer
            analyzer = PortfolioAnalyzer(self.kite)
            portfolio_snap = analyzer.snapshot()
            # Pre-load sector capital from existing holdings
            for sec, val in portfolio_snap.sector_values.items():
                sector_capital[sec] = val
            if portfolio_snap.concentration_warnings:
                for w in portfolio_snap.concentration_warnings:
                    logger.info("Portfolio warning: %s", w)
        except Exception as exc:
            logger.debug("Portfolio analysis unavailable: %s", exc)

        # R4: Market regime scaling (VIX + ADX)
        regime_scale = self._get_regime_scale()
        if regime_scale <= 0:
            logger.warning("VIX panic regime — no new BUY orders generated")
            return []

        for _, row in screened_df.head(limit).iterrows():
            if available <= 0:
                logger.info("No capital remaining — stopping plan generation")
                break

            # R1: Sector concentration check
            sector = str(row.get("sector_name", "")).strip() or "Unknown"
            if sector_trade_count.get(sector, 0) >= self.cfg.max_trades_per_sector:
                logger.info(
                    "Skipping %s — sector '%s' already has %d trades (max %d)",
                    row.get("symbol", "?"), sector,
                    sector_trade_count[sector], self.cfg.max_trades_per_sector,
                )
                continue
            sector_cap = sector_capital.get(sector, 0.0)
            if sector_cap >= self.cfg.total_capital * self.cfg.max_sector_exposure_pct:
                logger.info(
                    "Skipping %s — sector '%s' exposure ₹%.0f >= %.0f%% of capital",
                    row.get("symbol", "?"), sector,
                    sector_cap, self.cfg.max_sector_exposure_pct * 100,
                )
                continue

            plan = self._build_plan(row, available, regime_scale)
            if plan is not None:
                plans.append(plan)
                allocated = plan.quantity * plan.entry_price
                available -= allocated
                sector_trade_count[sector] = sector_trade_count.get(sector, 0) + 1
                sector_capital[sector] = sector_cap + allocated

        logger.info("Generated %d trade plans (top-%d)", len(plans), limit)
        return plans

    def plan_trades_carver(
        self,
        screened_df: pd.DataFrame,
        combined_forecasts: Dict[str, float],
        instrument_vols: Dict[str, float],
        instrument_weights: Optional[Dict[str, float]] = None,
        idm: float = 1.6,
        max_trades: Optional[int] = None,
    ) -> List[TradePlan]:
        """Carver-framework trade planner using combined forecasts and vol-targeting.

        Uses volatility-targeted continuous position sizing instead of Kelly.
        Falls back to legacy plan_trades() if Carver components unavailable.

        Parameters
        ----------
        screened_df : pd.DataFrame
            Output of NSEScreener.screen().
        combined_forecasts : dict[str, float]
            {symbol: combined_forecast} from forecast_combiner.
        instrument_vols : dict[str, float]
            {symbol: instrument_value_volatility} from instrument_volatility.
        instrument_weights : dict[str, float] | None
            Handcrafted instrument weights.
        idm : float
            Instrument Diversification Multiplier.
        max_trades : int | None
            Override for max_open_trades.

        Returns
        -------
        list[TradePlan]
        """
        if screened_df.empty or not combined_forecasts:
            return self.plan_trades(screened_df, max_trades)

        if self._vol_target is None:
            logger.warning("VolatilityTarget not set — falling back to legacy plan_trades")
            return self.plan_trades(screened_df, max_trades)

        limit = max_trades or self.cfg.max_open_trades
        available = self._available_capital()
        plans: List[TradePlan] = []
        sector_trade_count: Dict[str, int] = {}
        sector_capital: Dict[str, float] = {}

        regime_scale = self._get_regime_scale()
        if regime_scale <= 0:
            logger.warning("VIX panic regime — no new BUY orders generated")
            return []

        # Sort by conviction: positive forecasts first (descending), then negative
        # by abs. Prevents bearish stocks from wasting top slots in long-only mode.
        df_sorted = screened_df.copy()
        df_sorted["_forecast"] = df_sorted["symbol"].map(
            lambda s: combined_forecasts.get(s, 0)
        )
        df_sorted["_rank_key"] = df_sorted["_forecast"].apply(
            lambda f: (f > 0, f if f > 0 else -abs(f))
        )
        df_sorted = df_sorted.sort_values("_rank_key", ascending=False)

        for _, row in df_sorted.head(limit).iterrows():
            if available <= 0:
                break

            sym = row.get("symbol", "")
            forecast = combined_forecasts.get(sym)
            vol = instrument_vols.get(sym)
            if forecast is None or vol is None or vol <= 0:
                continue

            # Only take positive-forecast trades (long-only for NSE)
            if forecast <= 0:
                continue

            sector = str(row.get("sector_name", "")).strip() or "Unknown"
            if sector_trade_count.get(sector, 0) >= self.cfg.max_trades_per_sector:
                continue
            sector_cap = sector_capital.get(sector, 0.0)
            if sector_cap >= self.cfg.total_capital * self.cfg.max_sector_exposure_pct:
                continue

            weight = (instrument_weights or {}).get(sym, 0.10)
            # P1-6: Apply spread-based position reduction
            spread_scale = float(row.get("spread_scale", 1.0)) if "spread_scale" in row.index else 1.0
            plan = self._build_plan(
                row, available, regime_scale,
                carver_forecast=forecast,
                instrument_value_vol=vol,
                instrument_weight=weight,
                idm=idm,
            )
            if plan is not None:
                # P1-6: Apply spread-based position reduction
                if spread_scale < 1.0 and plan.quantity > 0:
                    plan.quantity = max(1, int(plan.quantity * spread_scale))
                plans.append(plan)
                allocated = plan.quantity * plan.entry_price
                available -= allocated
                sector_trade_count[sector] = sector_trade_count.get(sector, 0) + 1
                sector_capital[sector] = sector_cap + allocated

        logger.info("Generated %d Carver trade plans (top-%d)", len(plans), limit)
        return plans

    # ── Internal ───────────────────────────────────────────────

    def _get_regime_scale(self) -> float:
        """Compute position-size scaling factor based on VIX and ADX.

        Returns 0.0 (block orders) to 1.0 (full size).
        """
        scale = 1.0
        try:
            from scrapers.macro.macro_indicators import MacroIndicators
            macro = MacroIndicators()
            snap = macro.fetch(market="IND")
            vix = getattr(snap, "india_vix", None)
            if vix is not None:
                if vix >= self.cfg.vix_panic_threshold:
                    logger.warning("VIX=%.1f >= %.0f (panic) — blocking new BUY orders",
                                   vix, self.cfg.vix_panic_threshold)
                    return 0.0
                elif vix >= self.cfg.vix_caution_threshold:
                    scale = min(scale, self.cfg.vix_caution_scale)
                    logger.info("VIX=%.1f (caution) — scaling positions to %.0f%%",
                                vix, scale * 100)
        except Exception as exc:
            # P1-2 FIX: Do NOT penalize to caution_scale on API failure —
            # that punishes normal-market positions when the data source is simply down.
            # Keep scale=1.0 (unchanged) and log a warning.
            logger.warning("VIX regime check failed — keeping current scale %.0f%% (no penalty): %s",
                           scale * 100, exc)

        try:
            # ADX: fetch Nifty 50 ADX as market-wide trend proxy
            import yfinance as yf
            nifty = yf.download("^NSEI", period="2mo", progress=False, auto_adjust=True)
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
            if nifty is not None and len(nifty) > 28:
                from kite_connect.nse.screener import _adx
                market_adx = _adx(nifty)
                if market_adx is not None and market_adx < self.cfg.adx_choppy_threshold:
                    scale = min(scale, self.cfg.adx_choppy_scale)
                    logger.info("Market ADX=%.1f < %.0f (choppy) — scaling to %.0f%%",
                                market_adx, self.cfg.adx_choppy_threshold, scale * 100)
        except Exception as exc:
            # G9 fail-safe: default to choppy scale when ADX data unavailable
            scale = min(scale, self.cfg.adx_choppy_scale)
            logger.warning("ADX regime check failed — defaulting to choppy scale %.0f%%: %s",
                           scale * 100, exc)

        return scale

    def _build_plan(
        self,
        row: pd.Series,
        available_capital: float,
        regime_scale: float = 1.0,
        carver_forecast: Optional[float] = None,
        instrument_value_vol: Optional[float] = None,
        instrument_weight: float = 0.10,
        idm: float = 1.6,
    ) -> Optional[TradePlan]:
        entry = float(row["close"])
        if entry <= 0:
            return None  # BUG-8 FIX: guard against zero/negative entry price
        ma50 = float(row.get("ma_50", 0))
        support = float(row.get("support", 0))
        resistance = float(row.get("resistance", 0))
        score = float(row.get("score", 0))
        atr = float(row.get("atr", 0))

        # ── Stop-loss ──────────────────────────────────────────
        sl_ma50 = ma50 if ma50 > 0 else entry * 0.95
        sl_swing = support if support > 0 else entry * 0.95
        sl_atr = (entry - atr * self.cfg.atr_multiplier) if atr > 0 else entry * 0.95

        if self.cfg.sl_method == "ma50":
            sl = sl_ma50
        elif self.cfg.sl_method == "swing_low":
            sl = sl_swing
        elif self.cfg.sl_method == "atr":
            sl = sl_atr
        else:  # "tighter" — whichever is closest to entry (smallest loss)
            candidates = [sl_ma50, sl_swing]
            if atr > 0:
                candidates.append(sl_atr)
            sl = min(candidates)

        # Clamp SL to [sl_min_pct, sl_max_pct] below entry
        # P0 fix: Add overnight gap buffer — NSE has 16h overnight exposure
        # (close 3:30 PM → open 9:15 AM), during which RBI policy, global
        # news, FII flows can gap stocks 3-8%.  Buffer = 1.5 × daily_vol.
        gap_buffer_pct = 0.0
        if atr > 0 and entry > 0:
            daily_vol_pct = atr / entry  # ATR as fraction of price
            gap_buffer_pct = 1.5 * daily_vol_pct  # 1.5σ gap buffer
            gap_buffer_pct = min(gap_buffer_pct, 0.04)  # cap at 4% extra

        sl_floor = entry * (1 - self.cfg.sl_max_pct - gap_buffer_pct)   # farthest allowed (with gap buffer)
        sl_ceil  = entry * (1 - self.cfg.sl_min_pct)   # closest allowed
        if sl >= entry or sl > sl_ceil:
            sl = sl_ceil
        sl = max(sl_floor, min(sl, sl_ceil))

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

        # ── Position sizing (Carver vol-target or Kelly-scaled) ───
        # Carver path: uses combined forecast + vol target when available
        if (
            carver_forecast is not None
            and instrument_value_vol is not None
            and instrument_value_vol > 0
            and self._vol_target is not None
        ):
            from services.position_sizer import compute_position_size
            daily_cash_vol = self._vol_target.daily_cash_vol_target
            capital = self._vol_target.current_capital
            ps = compute_position_size(
                symbol=row.get("symbol", ""),
                combined_forecast=carver_forecast,
                instrument_value_vol=instrument_value_vol,
                daily_cash_vol_target=daily_cash_vol,
                price=entry,
                capital=capital,
                instrument_weight=instrument_weight,
                idm=idm,
            )
            qty = ps.target_quantity
            # Apply regime scaling as a dampener
            qty = max(0, int(qty * regime_scale))
            # Cap by available capital
            max_qty_cap = math.floor(available_capital / entry) if entry > 0 else 0
            qty = min(qty, max_qty_cap)
            if qty <= 0:
                return None
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

        # Legacy path: Half-Kelly sizing (fallback when Carver modules unavailable)
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

        # R4: Apply VIX/ADX regime scaling to position size
        effective_risk_pct *= regime_scale

        # Per-stock ADX scaling: halve size in choppy individual stocks
        stock_adx = float(row.get("adx", 0))
        if 0 < stock_adx < self.cfg.adx_choppy_threshold:
            effective_risk_pct *= self.cfg.adx_choppy_scale

        # GAP 8: Dynamic slippage buffer — adapt to market conditions
        slippage_pct = self._estimate_slippage(row)
        slippage_risk = entry * slippage_pct
        adjusted_risk_per_share = risk_per_share + slippage_risk
        if adjusted_risk_per_share <= 0:
            return None  # BUG-8 FIX: prevent zero-division

        max_risk = available_capital * effective_risk_pct
        qty = max(1, math.floor(max_risk / adjusted_risk_per_share))
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

    # ── SELL-side exit planning ────────────────────────────────

    def plan_exits(
        self,
        sell_symbols: List[str],
        holdings: List[dict],
    ) -> List[TradePlan]:
        """Generate SELL trade plans for held stocks with SELL/STRONG_SELL verdicts.

        Parameters
        ----------
        sell_symbols : list[str]
            Symbols flagged as SELL/STRONG_SELL by IntegratedScorer.
        holdings : list[dict]
            Kite holdings response (each dict has ``tradingsymbol``,
            ``quantity``, ``average_price``, ``last_price``).

        Returns
        -------
        list[TradePlan]
            SELL plans for matching held positions (full exit).
        """
        held_map = {}
        for h in holdings:
            sym = h.get("tradingsymbol", "")
            qty = int(h.get("quantity", 0))
            if sym and qty > 0:
                held_map[sym] = h

        plans: List[TradePlan] = []
        for sym in sell_symbols:
            if sym not in held_map:
                continue
            h = held_map[sym]
            qty = int(h.get("quantity", 0))
            avg_price = float(h.get("average_price", 0))
            ltp = float(h.get("last_price", avg_price))

            plans.append(TradePlan(
                symbol=sym,
                side="SELL",
                entry_price=ltp,           # current LTP (exit price)
                stop_loss=0.0,             # not applicable for exits
                target_price=0.0,
                quantity=qty,
                risk_amount=0.0,
                reward_amount=0.0,
                rr_ratio=0.0,
                score=0.0,
            ))
            logger.info("Exit plan: SELL %s × %d @ ~%.2f (avg %.2f)",
                        sym, qty, ltp, avg_price)

        return plans

    # ── Gap 8: Dynamic slippage estimation ─────────────────────

    def _estimate_slippage(self, row: pd.Series) -> float:
        """Estimate slippage dynamically based on market depth and volume.

        Three-tier estimation:
          1. Bid-ask spread from Kite market depth (best estimate)
          2. Volume-adjusted heuristic (mid-tier)
          3. Static fallback (cfg.slippage_buffer_pct)

        Returns slippage as a fraction (e.g. 0.003 for 0.3%).
        """
        symbol = row.get("symbol", "")

        # Tier 1: Real bid-ask spread from Kite
        if self.kite is not None:
            try:
                key = f"NSE:{symbol}"
                quote = self.kite.quote([key])
                depth = quote.get(key, {}).get("depth", {})
                buy_depth = depth.get("buy", [])
                sell_depth = depth.get("sell", [])

                if buy_depth and sell_depth:
                    best_bid = buy_depth[0].get("price", 0)
                    best_ask = sell_depth[0].get("price", 0)
                    if best_bid > 0 and best_ask > 0:
                        spread_pct = (best_ask - best_bid) / best_bid
                        # Slippage ~ half the spread (expected fill midpoint)
                        # plus a small buffer for market impact
                        estimated = spread_pct * 0.5 + 0.0005  # half-spread + 5 bps buffer
                        # Clamp to reasonable range
                        estimated = max(0.001, min(estimated, 0.01))
                        logger.debug(
                            "Dynamic slippage for %s: spread=%.3f%%, est=%.3f%%",
                            symbol, spread_pct * 100, estimated * 100,
                        )
                        return estimated
            except Exception:
                pass  # Fall through to tier 2

        # Tier 2: Volume-adjusted heuristic
        avg_volume = float(row.get("avg_volume", 0) or row.get("volume", 0))
        if avg_volume > 0:
            # Higher volume → lower slippage
            # Base: 0.2% for stocks with 1M+ avg volume
            # Scale up for lower volume stocks
            if avg_volume >= 1_000_000:
                return 0.002  # 0.2% — highly liquid
            elif avg_volume >= 500_000:
                return 0.003  # 0.3%
            elif avg_volume >= 100_000:
                return 0.004  # 0.4%
            else:
                return 0.006  # 0.6% — illiquid

        # Tier 3: Static fallback
        return self.cfg.slippage_buffer_pct

    # ── P7: Time-based swing / positional exit review ──────────

    def review_hold_exits(
        self,
        holdings: List[dict],
        trade_horizon: str = "swing",
    ) -> List[TradePlan]:
        """Flag positions that exceed the max hold period for forced exit.

        Parameters
        ----------
        holdings : list[dict]
            Kite holdings response.  Each dict should include
            ``tradingsymbol``, ``quantity``, ``average_price``,
            ``last_price``, and ``order_timestamp`` (or ``product``
            for inference).
        trade_horizon : str
            ``"swing"`` (3-10 day) or ``"positional"`` (2-6 week).

        Returns
        -------
        list[TradePlan]
            SELL plans for positions that have exceeded the hold window.
        """
        from datetime import datetime, timedelta

        max_days = (
            self.cfg.swing_max_hold_days
            if trade_horizon == "swing"
            else self.cfg.positional_max_hold_days
        )
        # A5: Use regime-adaptive hold days for swing
        if trade_horizon == "swing":
            try:
                from config import Config as _HoldCfg
                if hasattr(_HoldCfg, 'get_regime_hold_days'):
                    from services.regime_detector import get_current_regime
                    _regime = get_current_regime()
                    _regime_str = getattr(_regime, 'regime', '') if _regime else ''
                    max_days = _HoldCfg.get_regime_hold_days(str(_regime_str), "swing")
            except Exception:
                pass

        plans: List[TradePlan] = []
        now = datetime.utcnow()

        for h in holdings:
            sym = h.get("tradingsymbol", "")
            qty = int(h.get("quantity", 0))
            if qty <= 0:
                continue

            # Determine entry date from order timestamp or t1_quantity
            order_ts = h.get("order_timestamp") or h.get("opening_date")
            if not order_ts:
                continue  # can't determine age without timestamp

            if isinstance(order_ts, str):
                try:
                    entry_date = datetime.fromisoformat(order_ts)
                except (ValueError, TypeError):
                    continue
            elif isinstance(order_ts, datetime):
                entry_date = order_ts
            else:
                continue

            hold_days = (now - entry_date).days
            if hold_days <= max_days:
                continue

            avg_price = float(h.get("average_price", 0))
            ltp = float(h.get("last_price", avg_price))

            plans.append(TradePlan(
                symbol=sym,
                side="SELL",
                entry_price=ltp,
                stop_loss=0.0,
                target_price=0.0,
                quantity=qty,
                risk_amount=0.0,
                reward_amount=0.0,
                rr_ratio=0.0,
                score=0.0,
            ))
            logger.info(
                "P7 hold-period exit: %s held %d days (max %d for %s) — recommending SELL × %d",
                sym, hold_days, max_days, trade_horizon, qty,
            )

        return plans

    # ── T1-4: ATR-based trailing stop-loss ──────────────────
    @staticmethod
    def compute_atr_trailing_sl(
        current_price: float,
        ohlcv_df,
        regime: str = "bull",
        atr_period: int = 14,
        atr_mult_bull: float = 2.5,
        atr_mult_bear: float = 1.5,
        fallback_pct: float = 0.03,
    ) -> float:
        """Compute ATR-scaled trailing stop distance.

        Returns the stop-loss price (for long positions).
        For volatile stocks, the trail is wider; for calm stocks, tighter.
        In bear regime, use tighter multiplier to protect capital faster.
        """
        try:
            import pandas as pd
            if ohlcv_df is not None and len(ohlcv_df) >= atr_period and "High" in ohlcv_df.columns:
                high = ohlcv_df["High"].tail(atr_period + 1)
                low = ohlcv_df["Low"].tail(atr_period + 1)
                close = ohlcv_df["Close"].tail(atr_period + 1)
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr = float(tr.tail(atr_period).mean())
                # P1-3 FIX: Validate regime string against known values
                _VALID_REGIMES = {"bull", "trending_bull", "bear", "trending_bear",
                                  "crisis", "high_volatility", "range_bound", "range"}
                regime_lower = (regime or "").lower().strip()
                if regime_lower and regime_lower not in _VALID_REGIMES:
                    logger.warning("ATR trailing SL: unknown regime '%s' — defaulting to bull multiplier", regime)
                if "bear" in regime_lower or "crisis" in regime_lower or "high_volatility" in regime_lower:
                    mult = atr_mult_bear
                else:
                    mult = atr_mult_bull
                sl_distance = atr * mult
                sl_price = current_price - sl_distance
                return max(sl_price, current_price * (1 - 0.10))  # Never wider than 10%
        except Exception:
            pass
        # Fallback: fixed percentage
        return current_price * (1 - fallback_pct)

    # ── T1-5: Daily notional loss limit check ──────────────
    @staticmethod
    def check_daily_loss_limit(
        total_capital: float,
        daily_pnl: float,
        limit_pct: float = 0.03,
    ) -> bool:
        """Return True if daily loss exceeds limit (should halt trading).

        Parameters
        ----------
        total_capital : float
            Total portfolio capital.
        daily_pnl : float
            Today's realized + unrealized P&L (negative = loss).
        limit_pct : float
            Max allowed loss as fraction of capital.

        Returns
        -------
        bool
            True if loss limit breached — caller should halt all new orders.
        """
        if total_capital <= 0:
            return False
        loss_threshold = -abs(total_capital * limit_pct)
        if daily_pnl < loss_threshold:
            logger.critical(
                "T1-5: Daily loss limit breached! P&L=%.0f < threshold=%.0f (%.1f%% of %.0f)",
                daily_pnl, loss_threshold, limit_pct * 100, total_capital,
            )
            return True
        return False
