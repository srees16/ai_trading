"""
India Fear & Greed Composite Index.

A CNN Fear & Greed-style composite index tailored for the Indian
equity market.  Components and weights:

  ┌──────────────────────────────────┬────────┐
  │ Component                        │ Weight │
  ├──────────────────────────────────┼────────┤
  │ India VIX                        │  25 %  │
  │ FII net flows                    │  25 %  │
  │ Nifty Put-Call Ratio (PCR)       │  20 %  │
  │ Market breadth (Advance/Decline) │  15 %  │
  │ BankNifty vs Nifty divergence    │  15 %  │
  └──────────────────────────────────┴────────┘

Each component is individually mapped to a 0-100 scale
(0 = extreme fear, 100 = extreme greed), then the
weighted average yields the composite score.

Usage::

    from scrapers.macro.india_fear_greed import IndiaFearGreedIndex

    fg = IndiaFearGreedIndex()
    result = await fg.compute()
    # result.score = 42.5
    # result.label = "Fear"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────

@dataclass
class FearGreedResult:
    """India Fear & Greed composite index result."""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Component scores (0-100 each)
    vix_score: Optional[float] = None
    fii_score: Optional[float] = None
    pcr_score: Optional[float] = None
    breadth_score: Optional[float] = None
    divergence_score: Optional[float] = None

    # Composite
    score: Optional[float] = None   # 0 (extreme fear) – 100 (extreme greed)
    label: Optional[str] = None     # "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "vix_score": self.vix_score,
            "fii_score": self.fii_score,
            "pcr_score": self.pcr_score,
            "breadth_score": self.breadth_score,
            "divergence_score": self.divergence_score,
            "score": self.score,
            "label": self.label,
        }


# ── Weights ──────────────────────────────────────────────────────────

_W_VIX = 0.25
_W_FII = 0.25
_W_PCR = 0.20
_W_BREADTH = 0.15
_W_DIVERGENCE = 0.15


# ── Component mappers (raw → 0-100) ─────────────────────────────────

def _vix_to_score(india_vix: float) -> float:
    """
    Map India VIX to 0-100 (inverted — high VIX = fear).

    Typical ranges:
      VIX < 12  → extreme greed (score ~95)
      VIX 12-18 → greed (70-90)
      VIX 18-22 → neutral (40-60)
      VIX 22-30 → fear (15-40)
      VIX > 30  → extreme fear (score ~5)
    """
    if india_vix <= 10:
        return 100.0
    if india_vix >= 35:
        return 0.0
    # Linear interpolation: 10→100, 35→0
    return max(0.0, min(100.0, 100 - (india_vix - 10) * (100 / 25)))


def _fii_net_to_score(fii_net_crore: float) -> float:
    """
    Map FII net flow (₹ crore) to 0-100.

    Heavy buying (+3000+) → greed, heavy selling (-3000-) → fear.
    """
    # Clamp to [-5000, +5000]
    clamped = max(-5000, min(5000, fii_net_crore))
    # Linear: -5000→0, +5000→100
    return max(0.0, min(100.0, (clamped + 5000) / 100))


def _pcr_to_score(pcr: float) -> float:
    """
    Map Nifty Option Put-Call Ratio to 0-100.

    PCR > 1.3  → oversold / extreme bullish sentiment → greed
    PCR 1.0-1.3 → moderately bullish
    PCR 0.7-1.0 → neutral
    PCR < 0.7  → overbought / bearish → fear
    """
    if pcr <= 0.4:
        return 5.0
    if pcr >= 1.5:
        return 95.0
    # Linear: 0.4→5, 1.5→95
    return max(0.0, min(100.0, 5 + (pcr - 0.4) * (90 / 1.1)))


def _breadth_to_score(advance_decline_ratio: float) -> float:
    """
    Map Advance/Decline ratio to 0-100.

    A/D > 3.0 → extreme greed
    A/D 1.5-3.0 → greed
    A/D 0.7-1.5 → neutral
    A/D < 0.7 → fear
    """
    if advance_decline_ratio <= 0.2:
        return 0.0
    if advance_decline_ratio >= 4.0:
        return 100.0
    # Linear: 0.2→0, 4.0→100
    return max(0.0, min(100.0, (advance_decline_ratio - 0.2) * (100 / 3.8)))


def _divergence_to_score(banknifty_chg_pct: float, nifty_chg_pct: float) -> float:
    """
    Map BankNifty vs Nifty divergence to 0-100.

    When BankNifty outperforms Nifty → risk-on → greed.
    When BankNifty underperforms → risk-off → fear.
    Divergence = BankNifty_chg% - Nifty_chg%
    """
    divergence = banknifty_chg_pct - nifty_chg_pct
    # Typical range: -3% to +3%
    clamped = max(-3.0, min(3.0, divergence))
    # Linear: -3→0, +3→100
    return max(0.0, min(100.0, (clamped + 3.0) * (100 / 6.0)))


def _score_to_label(score: float) -> str:
    """Convert composite 0-100 score to human label."""
    if score <= 20:
        return "Extreme Fear"
    if score <= 40:
        return "Fear"
    if score <= 60:
        return "Neutral"
    if score <= 80:
        return "Greed"
    return "Extreme Greed"


# ── Main class ───────────────────────────────────────────────────────

class IndiaFearGreedIndex:
    """
    Computes the India Fear & Greed composite index.

    Pulls component data from:
    - ``MacroIndicators`` (India VIX, Nifty/BankNifty changes)
    - ``FIIDIIFlows`` (FII net flows)
    - ``compute_pcr()`` from option_chain (Nifty PCR)
    - ``MarketBreadthScraper`` (A/D ratio from NSE Bhavcopy)

    Usage::

        fg = IndiaFearGreedIndex()
        result = await fg.compute()
    """

    _cached: Optional[FearGreedResult] = None
    _cache_ts: Optional[datetime] = None
    _CACHE_TTL = timedelta(minutes=15)

    async def compute(
        self,
        india_vix: Optional[float] = None,
        fii_net_crore: Optional[float] = None,
        nifty_pcr: Optional[float] = None,
        advance_decline_ratio: Optional[float] = None,
        banknifty_change_pct: Optional[float] = None,
        nifty_change_pct: Optional[float] = None,
    ) -> FearGreedResult:
        """
        Compute the composite index from pre-fetched component inputs.

        Pass ``None`` for any unavailable component — the weight is
        redistributed proportionally across available components.
        """
        now = datetime.utcnow()
        result = FearGreedResult(timestamp=now)

        weighted_parts = []
        total_weight = 0.0

        # 1. India VIX (25%)
        if india_vix is not None:
            result.vix_score = _vix_to_score(india_vix)
            weighted_parts.append((_W_VIX, result.vix_score))
            total_weight += _W_VIX

        # 2. FII flows (25%)
        if fii_net_crore is not None:
            result.fii_score = _fii_net_to_score(fii_net_crore)
            weighted_parts.append((_W_FII, result.fii_score))
            total_weight += _W_FII

        # 3. Nifty PCR (20%)
        if nifty_pcr is not None:
            result.pcr_score = _pcr_to_score(nifty_pcr)
            weighted_parts.append((_W_PCR, result.pcr_score))
            total_weight += _W_PCR

        # 4. Market breadth (15%)
        if advance_decline_ratio is not None:
            result.breadth_score = _breadth_to_score(advance_decline_ratio)
            weighted_parts.append((_W_BREADTH, result.breadth_score))
            total_weight += _W_BREADTH

        # 5. BankNifty divergence (15%)
        if banknifty_change_pct is not None and nifty_change_pct is not None:
            result.divergence_score = _divergence_to_score(
                banknifty_change_pct, nifty_change_pct,
            )
            weighted_parts.append((_W_DIVERGENCE, result.divergence_score))
            total_weight += _W_DIVERGENCE

        # Composite (proportional re-weighting for missing components)
        if total_weight > 0:
            result.score = round(
                sum(w * s for w, s in weighted_parts) / total_weight, 1
            )
            result.label = _score_to_label(result.score)
        else:
            result.score = None
            result.label = "N/A"

        logger.info(
            "IndiaF&G: VIX=%.0f  FII=%.0f  PCR=%.0f  Breadth=%.0f  "
            "Div=%.0f → Composite=%.1f (%s)",
            result.vix_score or 0, result.fii_score or 0,
            result.pcr_score or 0, result.breadth_score or 0,
            result.divergence_score or 0,
            result.score or 0, result.label or "N/A",
        )
        return result
