"""
Instrument Weights & IDM — Handcrafted portfolio allocation (Carver Ch. 11).

Two-tier handcrafted weighting:
  1. **Top-down sector allocation** — equal weight across sectors
     (or sector-tilted if macro regime favours certain sectors).
  2. **Within-sector equal weight** — within each sector, divide
     the sector's allocation equally among selected instruments.

Instrument Diversification Multiplier (IDM):
  IDM = 1 / sqrt(w' × C × w)
  where w = instrument weight vector, C = return correlation matrix.

  Pre-calibrated defaults for NSE:
    - 3 instruments, avg ρ ≈ 0.5  →  IDM ≈ 1.34
    - 6 instruments, avg ρ ≈ 0.4  →  IDM ≈ 1.61
    - 10 instruments, avg ρ ≈ 0.35 → IDM ≈ 1.82
  Capped at 2.5 maximum.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Maximum IDM (Carver recommendation: cap at 2.5)
MAX_IDM = 2.5

# Pre-calibrated IDM lookup by portfolio size (NSE equity basket)
IDM_LOOKUP: Dict[int, float] = {
    1: 1.00,
    2: 1.20,
    3: 1.34,
    4: 1.48,
    5: 1.55,
    6: 1.61,
    7: 1.67,
    8: 1.72,
    9: 1.77,
    10: 1.82,
}


def get_default_idm(n_instruments: int) -> float:
    """Return pre-calibrated IDM for N instruments.

    Uses the lookup table for 1-10; extrapolates for larger portfolios.
    """
    if n_instruments <= 0:
        return 1.0
    if n_instruments in IDM_LOOKUP:
        return IDM_LOOKUP[n_instruments]
    if n_instruments > 10:
        # Gradual asymptotic approach: IDM → ~2.0 for very diversified
        return min(MAX_IDM, 1.82 + 0.03 * (n_instruments - 10))
    # Interpolate
    lo = max(k for k in IDM_LOOKUP if k <= n_instruments)
    hi = min(k for k in IDM_LOOKUP if k >= n_instruments)
    if lo == hi:
        return IDM_LOOKUP[lo]
    frac = (n_instruments - lo) / (hi - lo)
    return IDM_LOOKUP[lo] + frac * (IDM_LOOKUP[hi] - IDM_LOOKUP[lo])


def compute_idm(
    weights: Dict[str, float],
    correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    avg_correlation: float = 0.40,
) -> float:
    """Compute IDM from instrument weights and correlations.

    IDM = 1 / sqrt(w' × C × w)

    Parameters
    ----------
    weights : dict[str, float]
        {symbol: weight}, weights should sum to ~1.0.
    correlation_matrix : dict | None
        {(sym_a, sym_b): correlation}.  If None, uses avg_correlation
        for all off-diagonal entries.
    avg_correlation : float
        Default off-diagonal correlation when matrix not provided.

    Returns
    -------
    float
        IDM, capped at MAX_IDM.
    """
    names = sorted(weights.keys())
    n = len(names)
    if n <= 1:
        return 1.0

    w = np.array([weights[name] for name in names])

    # Build correlation matrix
    C = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                if correlation_matrix:
                    key = (names[i], names[j])
                    rev = (names[j], names[i])
                    C[i, j] = correlation_matrix.get(key, correlation_matrix.get(rev, avg_correlation))
                else:
                    C[i, j] = avg_correlation

    port_var = float(w @ C @ w)
    if port_var <= 0:
        return 1.0

    idm = 1.0 / math.sqrt(port_var)
    idm = min(idm, MAX_IDM)
    return round(idm, 3)


def compute_handcrafted_weights(
    symbols: List[str],
    sector_map: Optional[Dict[str, str]] = None,
    sector_tilts: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute two-tier handcrafted instrument weights (Carver Ch. 4).

    Level 1: Equal weight across active sectors.
    Level 2: Within each sector, equal weight across instruments.

    Parameters
    ----------
    symbols : list[str]
        Active instruments this rebalance.
    sector_map : dict | None
        {symbol: sector_name}.  If None, all instruments get equal weight.
    sector_tilts : dict | None
        {sector: tilt_multiplier}.  1.0 = neutral, 1.5 = overweight 50%.

    Returns
    -------
    dict[str, float]
        {symbol: weight}, summing to ~1.0.
    """
    if not symbols:
        return {}

    if sector_map is None:
        # Equal weight if no sector info
        w = 1.0 / len(symbols)
        return {sym: round(w, 6) for sym in symbols}

    # Group by sector
    sector_groups: Dict[str, List[str]] = {}
    for sym in symbols:
        sec = sector_map.get(sym, "Unknown")
        sector_groups.setdefault(sec, []).append(sym)

    n_sectors = len(sector_groups)
    if n_sectors == 0:
        w = 1.0 / len(symbols)
        return {sym: round(w, 6) for sym in symbols}

    # Level 1: sector weights (equal, then tilted)
    base_sector_weight = 1.0 / n_sectors
    sector_weights: Dict[str, float] = {}
    for sec in sector_groups:
        tilt = 1.0
        if sector_tilts and sec in sector_tilts:
            tilt = sector_tilts[sec]
        sector_weights[sec] = base_sector_weight * tilt

    # Renormalise sector weights to sum to 1.0
    total_sw = sum(sector_weights.values())
    if total_sw > 0:
        sector_weights = {k: v / total_sw for k, v in sector_weights.items()}

    # Level 2: within-sector equal weight
    weights: Dict[str, float] = {}
    for sec, syms in sector_groups.items():
        per_sym = sector_weights[sec] / len(syms)
        for sym in syms:
            weights[sym] = round(per_sym, 6)

    return weights
