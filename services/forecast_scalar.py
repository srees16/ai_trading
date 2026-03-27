"""
Forecast Scalar — Normalise all trading signals to Carver scale.

Every trading rule in the Carver framework must output a *forecast*:
  - Average absolute value ≈ 10
  - Capped at ±20
  - Positive = long, negative = short, 0 = flat

This module converts centurion_core's various signal formats into
this standardised forecast scale so they can be combined via the
Forecast Combiner.

Signal sources and their native scales:
  1. NSEScreener score      :  0 → 100    (higher = stronger buy)
  2. DecisionEngine score   : -1 → +1     (positive = buy)
  3. IntegratedScorer verdict: -1 → +1    (positive = buy)
  4. EWMAC crossover        : raw float   (needs vol-adjustment + scalar)
  5. Carry rule              : raw float   (needs vol-adjustment + scalar)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Carver's recommended forecast limits
FORECAST_CAP = 20.0
FORECAST_FLOOR = -20.0
TARGET_ABS_FORECAST = 10.0


def cap_forecast(raw: float) -> float:
    """Cap a forecast to the ±20 range."""
    return max(FORECAST_FLOOR, min(FORECAST_CAP, raw))


# ═══════════════════════════════════════════════════════════════
# Pre-calibrated scalars for each signal source
# ═══════════════════════════════════════════════════════════════

# These map each source's typical output to an avg abs ≈ 10 forecast.
# They can be overridden by walk-forward calibration in data/wf_params/.

# Screener score (0–100):
#   Typical good stock scores 40–70, midpoint ≈ 50.
#   We want 50 → forecast 10.  So scalar = 10/50 = 0.20.
#   Score 0 → forecast 0, score 100 → forecast 20 (at cap).
SCALAR_SCREENER = 0.20

# DecisionEngine / IntegratedScorer (-1 to +1):
#   Typical strong signal ≈ ±0.5.  We want 0.5 → 10.
#   So scalar = 10/0.5 = 20.
SCALAR_DECISION_ENGINE = 20.0
SCALAR_INTEGRATED_SCORER = 20.0

# EWMAC scalars (from Carver Appendix B, Table 49):
#   These depend on the fast/slow look-back pair.
EWMAC_SCALARS = {
    (2, 8): 10.6,
    (4, 16): 7.5,
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
    (64, 256): 1.87,
}

# Carry scalar: annualised carry / vol → forecast.
# Typical carry for equities ≈ 2–4% yield / 20% vol = 0.1–0.2.
# We want 0.1 → 10, so scalar = 100.  But Carver uses 30 for futures.
# For equities the carry signal is weaker; use 40.
SCALAR_CARRY = 40.0


# ═══════════════════════════════════════════════════════════════
# Conversion functions
# ═══════════════════════════════════════════════════════════════

def screener_to_forecast(score: float) -> float:
    """Convert NSEScreener score (0–100) to Carver forecast (-20 to +20).

    A score of 0 means "no conviction" → forecast 0.
    Score of 50 → forecast 10 (average conviction).
    Score ≥ 100 → capped at 20.

    Since the screener only produces long signals (score ≥ 0),
    the forecast is always ≥ 0.
    """
    raw = score * SCALAR_SCREENER
    return cap_forecast(raw)


def decision_engine_to_forecast(score: float) -> float:
    """Convert DecisionEngine score (-1 to +1) to Carver forecast."""
    raw = score * SCALAR_DECISION_ENGINE
    return cap_forecast(raw)


def integrated_scorer_to_forecast(score: float) -> float:
    """Convert IntegratedScorer verdict (-1 to +1) to Carver forecast."""
    raw = score * SCALAR_INTEGRATED_SCORER
    return cap_forecast(raw)


def ewmac_to_forecast(
    raw_crossover: float,
    daily_price_vol: float,
    fast: int,
    slow: int,
) -> float:
    """Convert EWMAC crossover to Carver forecast.

    Parameters
    ----------
    raw_crossover : float
        fast_ewma - slow_ewma (in price points).
    daily_price_vol : float
        Daily standard deviation of price (in same units as crossover).
    fast, slow : int
        Look-back periods (e.g. 16, 64).

    Returns
    -------
    float
        Capped forecast (-20 to +20).
    """
    if daily_price_vol <= 0:
        return 0.0
    vol_adjusted = raw_crossover / daily_price_vol
    scalar = EWMAC_SCALARS.get((fast, slow), 3.75)  # default to 16/64
    raw = vol_adjusted * scalar
    return cap_forecast(raw)


def carry_to_forecast(
    annualised_carry: float,
    annual_price_vol: float,
) -> float:
    """Convert carry signal to Carver forecast.

    Parameters
    ----------
    annualised_carry : float
        Expected annual return from carry (dividend yield - funding cost),
        as a decimal (e.g. 0.03 = 3 %).
    annual_price_vol : float
        Annualised percentage volatility of price (decimal).

    Returns
    -------
    float
        Capped forecast (-20 to +20).
    """
    if annual_price_vol <= 0:
        return 0.0
    vol_adjusted = annualised_carry / annual_price_vol
    raw = vol_adjusted * SCALAR_CARRY
    return cap_forecast(raw)
