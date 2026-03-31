"""
Post-Earnings Announcement Drift (PEAD) Strategy — Phase 1.3.

Exploits the well-documented tendency for stock prices to drift in the
direction of an earnings surprise for 30-60 trading days after announcement.

Research basis:
  - Ball & Brown (1968): Original discovery
  - Bernard & Thomas (1989): Formal documentation of drift
  - Indian evidence: Sehgal & Jain (2013), Bharath et al. (2009)

Integration:
  - Fed as an additional forecast source into the Carver combiner
  - Weight: 10-15% of combined forecast
  - Only active during earnings season windows (~12 weeks/year total)

Signal:
  - Standardized Unexpected Earnings (SUE):
    SUE = (EPS_actual - EPS_consensus) / std(historical surprises)
  - SUE > +1.0 → positive drift → BUY forecast
  - SUE < -1.0 → negative drift → SELL forecast (avoid for long-only)
  - Forecast decays over hold_period days
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Path for cached earnings data
_EARNINGS_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "earnings_cache.json"


@dataclass
class EarningsSurprise:
    """Earnings surprise data for a single stock."""
    ticker: str
    announcement_date: str
    eps_actual: float
    eps_consensus: float
    sue: float                    # Standardized Unexpected Earnings
    surprise_pct: float           # (actual - consensus) / |consensus|
    direction: str = "NEUTRAL"    # POSITIVE / NEGATIVE / NEUTRAL


@dataclass
class PEADSignal:
    """PEAD trading signal for a single stock."""
    ticker: str
    sue: float
    forecast: float               # Carver-scale (-20 to +20)
    days_since_announcement: int
    decay_factor: float           # remaining signal strength (0..1)
    active: bool = True
    surprise: Optional[EarningsSurprise] = None


class PEADStrategy:
    """Post-Earnings Announcement Drift signal generator.

    Parameters
    ----------
    sue_threshold : float
        Minimum |SUE| to generate a signal (default 1.0 std).
    hold_period : int
        Trading days over which the drift typically completes.
    decay_rate : float
        Daily exponential decay of the forecast signal.
    max_forecast : float
        Maximum absolute forecast value (Carver cap).
    """

    def __init__(
        self,
        sue_threshold: float = 1.0,
        hold_period: int = 45,
        decay_rate: float = 0.97,
        max_forecast: float = 20.0,
    ):
        self.sue_threshold = sue_threshold
        self.hold_period = hold_period
        self.decay_rate = decay_rate
        self.max_forecast = max_forecast
        self._active_signals: Dict[str, PEADSignal] = {}
        self._load_active_signals()

    def process_earnings(
        self,
        earnings_data: List[EarningsSurprise],
    ) -> List[PEADSignal]:
        """Process new earnings announcements and generate signals.

        Parameters
        ----------
        earnings_data : list[EarningsSurprise]
            Fresh earnings surprise data (from NSE/Trendlyne/manual).

        Returns
        -------
        list[PEADSignal]
            New signals generated.
        """
        new_signals: List[PEADSignal] = []

        for surprise in earnings_data:
            if abs(surprise.sue) < self.sue_threshold:
                continue

            forecast = self._sue_to_forecast(surprise.sue)
            signal = PEADSignal(
                ticker=surprise.ticker,
                sue=surprise.sue,
                forecast=forecast,
                days_since_announcement=0,
                decay_factor=1.0,
                active=True,
                surprise=surprise,
            )
            self._active_signals[surprise.ticker] = signal
            new_signals.append(signal)
            logger.info(
                "PEAD signal: %s SUE=%.2f forecast=%.1f direction=%s",
                surprise.ticker, surprise.sue, forecast, surprise.direction,
            )

        self._persist_active_signals()
        return new_signals

    def get_current_forecasts(self) -> Dict[str, float]:
        """Return currently active PEAD forecasts.

        Decays signals over time and returns {symbol: forecast}
        for integration with the Carver forecast combiner.
        """
        forecasts: Dict[str, float] = {}
        expired: List[str] = []

        for ticker, signal in self._active_signals.items():
            if signal.days_since_announcement >= self.hold_period:
                expired.append(ticker)
                continue

            # Apply exponential decay
            decay = self.decay_rate ** signal.days_since_announcement
            decayed_forecast = signal.forecast * decay

            if abs(decayed_forecast) < 1.0:  # below noise threshold
                expired.append(ticker)
                continue

            signal.decay_factor = decay
            forecasts[ticker] = round(decayed_forecast, 2)

        for t in expired:
            del self._active_signals[t]

        return forecasts

    def advance_day(self) -> None:
        """Call once per trading day to advance the decay clock.

        FIX-8: Persist state after advancing so days_since survives restarts.
        """
        for signal in self._active_signals.values():
            signal.days_since_announcement += 1
        self._persist_active_signals()

    def _sue_to_forecast(self, sue: float) -> float:
        """Map SUE to Carver-scale forecast.

        SUE of ±3 maps to ±20. Linear scaling with capping.
        """
        # Scale: ~6.67 per SUE unit gives ±20 at ±3
        raw = sue * 6.67
        return max(-self.max_forecast, min(self.max_forecast, raw))

    def _persist_active_signals(self) -> None:
        """Save active signals to disk for crash recovery."""
        try:
            data = {}
            for ticker, sig in self._active_signals.items():
                data[ticker] = {
                    "sue": sig.sue,
                    "forecast": sig.forecast,
                    "days_since": sig.days_since_announcement,
                    "decay_factor": sig.decay_factor,
                }
            _EARNINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_EARNINGS_CACHE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.debug("PEAD signal persist failed: %s", exc)

    def _load_active_signals(self) -> None:
        """Restore active signals from disk."""
        if not _EARNINGS_CACHE_PATH.exists():
            return
        try:
            with open(_EARNINGS_CACHE_PATH, "r") as f:
                data = json.load(f)
            for ticker, info in data.items():
                self._active_signals[ticker] = PEADSignal(
                    ticker=ticker,
                    sue=info.get("sue", 0),
                    forecast=info.get("forecast", 0),
                    days_since_announcement=info.get("days_since", 0),
                    decay_factor=info.get("decay_factor", 1.0),
                    active=True,
                )
        except Exception as exc:
            logger.debug("PEAD signal load failed: %s", exc)


def compute_sue(
    eps_actual: float,
    eps_consensus: float,
    historical_surprises: Optional[List[float]] = None,
) -> float:
    """Compute Standardized Unexpected Earnings (SUE).

    SUE = (EPS_actual - EPS_consensus) / std(historical surprises)

    If historical surprises not available, uses |consensus| as normalizer.
    """
    surprise = eps_actual - eps_consensus

    if historical_surprises and len(historical_surprises) >= 4:
        std = float(np.std(historical_surprises, ddof=1))
        if std > 0:
            return surprise / std

    # Fallback: normalize by |consensus|
    if abs(eps_consensus) > 0.01:
        return surprise / abs(eps_consensus)

    return 0.0
