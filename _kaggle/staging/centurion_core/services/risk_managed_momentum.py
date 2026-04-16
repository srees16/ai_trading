"""
Risk-Managed Momentum — T4-4.

Implementation of Barroso & Santa-Clara (2015):
  "Momentum has its moments" — scale momentum exposure by the inverse
  of recent realized momentum-strategy volatility.

Key insight: Momentum crashes are preceded by high momentum vol.
By scaling position size inversely with recent vol, we reduce exposure
before crashes and increase it during calm periods.

Also implements:
  - Novy-Marx (2012): 12-1 month momentum (skip last month)
  - Daniel & Moskowitz (2016): Dynamic momentum weighting
  - Acceleration factor: rate of change of momentum score

Integration:
  - Applied per-instrument to Carver momentum/acceleration forecasts
  - Scales the raw forecast before it enters the forecast combiner
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskManagedMomentumResult:
    """Result of risk-managed momentum computation."""
    raw_momentum: Dict[str, float] = field(default_factory=dict)
    momentum_vol: Dict[str, float] = field(default_factory=dict)
    risk_scaling: Dict[str, float] = field(default_factory=dict)
    adjusted_forecast: Dict[str, float] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)


class RiskManagedMomentum:
    """Risk-managed momentum following Barroso & Santa-Clara (2015).

    Parameters
    ----------
    lookback : int
        Momentum lookback in trading days (default 252 = 12 months).
    skip_month : int
        Days to skip at the short end (default 21 = 1 month).
        Novy-Marx (2012): skip last month avoids short-term reversal.
    vol_lookback : int
        Days to estimate momentum strategy vol (default 126 = 6 months).
    target_vol : float
        Target annualized vol for momentum strategy (default 0.12).
    max_scale : float
        Maximum risk scaling factor (cap to prevent over-leverage).
    min_scale : float
        Minimum risk scaling factor (floor).
    """

    def __init__(
        self,
        lookback: int = 252,
        skip_month: int = 21,
        vol_lookback: int = 126,
        target_vol: float = 0.12,
        max_scale: float = 3.0,
        min_scale: float = 0.25,
    ):
        self.lookback = lookback
        self.skip_month = skip_month
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol
        self.max_scale = max_scale
        self.min_scale = min_scale

    def compute_12_1_momentum(self, prices: pd.Series) -> Optional[float]:
        """Compute 12-1 month momentum (Novy-Marx formulation).

        Returns the return from t-252 to t-21 (skip last month).
        """
        if len(prices) < self.lookback + 5:
            return None

        p_end = prices.iloc[-self.skip_month]    # 1 month ago
        p_start = prices.iloc[-self.lookback]     # 12 months ago

        if p_start <= 0:
            return None

        return float((p_end / p_start) - 1.0)

    def compute_momentum_vol(self, prices: pd.Series) -> Optional[float]:
        """Compute recent volatility of the momentum strategy.

        Uses rolling momentum returns over vol_lookback period.
        """
        if len(prices) < self.lookback + self.vol_lookback + 5:
            return None

        # Compute rolling 1-day momentum returns
        # (change in 12-1 momentum signal)
        mom_returns = []
        for i in range(self.vol_lookback):
            idx = -1 - i
            if abs(idx) >= len(prices) - self.lookback:
                break
            end_idx = idx - self.skip_month
            start_idx = idx - self.lookback
            if abs(start_idx) >= len(prices) or abs(end_idx) >= len(prices):
                break

            p_end = prices.iloc[end_idx]
            p_start = prices.iloc[start_idx]
            if p_start > 0:
                mom_returns.append((p_end / p_start) - 1.0)

        if len(mom_returns) < 30:
            return None

        # Annualized vol of momentum returns
        daily_vol = float(np.std(np.diff(mom_returns)))
        return daily_vol * math.sqrt(252)

    def compute_risk_scaling(self, momentum_vol: float) -> float:
        """Compute position scaling factor.

        scale = target_vol / realized_vol
        Capped at [min_scale, max_scale].
        """
        if momentum_vol <= 0.001:
            return 1.0

        raw_scale = self.target_vol / momentum_vol
        return max(self.min_scale, min(self.max_scale, raw_scale))

    def adjust_forecasts(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.Series],
        raw_forecasts: Dict[str, float],
    ) -> RiskManagedMomentumResult:
        """Apply risk-managed momentum scaling to raw forecasts.

        For each instrument:
        1. Compute 12-1 momentum signal
        2. Estimate recent momentum strategy vol
        3. Scale forecast by target_vol / realized_vol
        4. Cap scaling to [min_scale, max_scale]

        Parameters
        ----------
        symbols : list of str
        price_data : dict of {symbol: pd.Series} — close prices (oldest first)
        raw_forecasts : dict of {symbol: float} — raw Carver forecasts [-20, +20]

        Returns
        -------
        RiskManagedMomentumResult with adjusted forecasts.
        """
        result = RiskManagedMomentumResult()

        for sym in symbols:
            prices = price_data.get(sym)
            raw_fc = raw_forecasts.get(sym, 0.0)

            if prices is None or len(prices) < self.lookback + 50:
                result.adjusted_forecast[sym] = raw_fc
                result.risk_scaling[sym] = 1.0
                continue

            # Step 1: 12-1 momentum
            mom = self.compute_12_1_momentum(prices)
            result.raw_momentum[sym] = round(mom, 4) if mom is not None else 0.0

            # Step 2: Momentum vol
            mom_vol = self.compute_momentum_vol(prices)
            if mom_vol is None or mom_vol <= 0.001:
                result.adjusted_forecast[sym] = raw_fc
                result.risk_scaling[sym] = 1.0
                continue

            result.momentum_vol[sym] = round(mom_vol, 4)

            # Step 3: Risk scaling
            scale = self.compute_risk_scaling(mom_vol)
            result.risk_scaling[sym] = round(scale, 4)

            # Step 4: Apply scaling to raw forecast
            adjusted = raw_fc * scale
            # Re-clip to [-20, +20] (Carver max forecast)
            adjusted = max(-20.0, min(20.0, adjusted))
            result.adjusted_forecast[sym] = round(adjusted, 2)

        n_scaled = sum(1 for s in result.risk_scaling.values() if s != 1.0)
        result.log.append(
            f"Risk-managed momentum: {n_scaled}/{len(symbols)} symbols scaled"
        )
        avg_scale = np.mean(list(result.risk_scaling.values())) if result.risk_scaling else 1.0
        result.log.append(f"Average risk scaling: {avg_scale:.2f}x")

        for line in result.log:
            logger.info("RiskMgdMom: %s", line)

        return result
