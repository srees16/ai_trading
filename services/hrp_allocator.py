"""
Hierarchical Risk Parity (HRP) — de Prado AFML Ch.16.

Implements the full HRP pipeline:
  1. Distance matrix from correlation matrix
  2. Hierarchical clustering (single-linkage)
  3. Quasi-diagonalization (matrix seriation)
  4. Recursive bisection → asset weights

Advantages over mean-variance (Markowitz):
  - No matrix inversion (stable with noisy correlations)
  - Accounts for hierarchical structure of correlations
  - More robust out-of-sample than IVP or CLA

Usage:
    from services.hrp_allocator import hrp_weights
    weights = hrp_weights(returns_df)  # returns {ticker: weight}
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

# Minimum instruments for HRP to provide meaningful diversification
MIN_INSTRUMENTS = 3


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix.

    d(i,j) = sqrt(0.5 × (1 - ρ(i,j)))
    Range: [0, 1] where 0 = perfectly correlated, 1 = anti-correlated.
    """
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    return dist


def _quasi_diagonalize(link: np.ndarray) -> List[int]:
    """Quasi-diagonalize correlation matrix via dendrogram leaf ordering.

    Returns sorted leaf indices such that correlated assets are adjacent.
    """
    return list(leaves_list(link).astype(int))


def _get_cluster_variance(cov: np.ndarray, cluster_items: List[int]) -> float:
    """Compute inverse-variance portfolio weight for a cluster.

    Returns the cluster's portfolio variance (IVP allocation within cluster).
    """
    sub_cov = cov[np.ix_(cluster_items, cluster_items)]
    # Inverse-variance portfolio within the cluster
    ivp = 1.0 / np.diag(sub_cov)
    ivp = ivp / ivp.sum()
    cluster_var = float(ivp @ sub_cov @ ivp)
    return cluster_var


def _recursive_bisection(
    cov: np.ndarray,
    sorted_indices: List[int],
) -> np.ndarray:
    """Recursive bisection: allocate weights top-down through the dendrogram.

    Start with weight=1 for the full set. At each split, allocate inversely
    proportional to cluster variance.

    Returns weight array of shape (n_assets,).
    """
    n = cov.shape[0]
    weights = np.ones(n)
    cluster_items = [sorted_indices]

    while len(cluster_items) > 0:
        new_clusters = []
        for items in cluster_items:
            if len(items) <= 1:
                continue
            # Split in half
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]

            var_left = _get_cluster_variance(cov, left)
            var_right = _get_cluster_variance(cov, right)

            # Allocate inversely proportional to variance
            total_var = var_left + var_right
            if total_var < 1e-12:
                alpha = 0.5
            else:
                alpha = 1.0 - var_left / total_var  # more var → less weight

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1.0 - alpha)

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        cluster_items = new_clusters

    return weights


def hrp_weights(
    returns: pd.DataFrame,
    min_history: int = 63,
) -> Dict[str, float]:
    """Compute HRP portfolio weights from a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, columns = tickers, rows = dates.
    min_history : int
        Minimum trading days required (default 63 = ~3 months).

    Returns
    -------
    dict of {ticker: weight} summing to 1.0.
    Empty dict if insufficient data.
    """
    # Drop columns with insufficient data
    valid_cols = [c for c in returns.columns if returns[c].dropna().shape[0] >= min_history]
    if len(valid_cols) < MIN_INSTRUMENTS:
        logger.warning(
            "HRP: only %d instruments with sufficient history (need %d)",
            len(valid_cols), MIN_INSTRUMENTS,
        )
        return {}

    rets = returns[valid_cols].dropna()
    if len(rets) < min_history:
        return {}

    n = len(valid_cols)
    tickers = list(valid_cols)

    # Step 1: Correlation → distance matrix
    corr = rets.corr().values
    # Clamp correlation to [-1, 1] for numerical stability
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    dist = _correlation_distance(corr)

    # Step 2: Hierarchical clustering (single-linkage)
    # Convert to condensed distance matrix
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method='single')

    # Step 3: Quasi-diagonalize
    sorted_idx = _quasi_diagonalize(link)

    # Step 4: Recursive bisection
    cov = rets.cov().values
    raw_weights = _recursive_bisection(cov, sorted_idx)

    # Normalize to sum to 1.0
    total = raw_weights.sum()
    if total < 1e-12:
        return {t: 1.0 / n for t in tickers}

    weights = raw_weights / total

    result = {tickers[i]: float(weights[i]) for i in range(n)}
    logger.info(
        "HRP: %d instruments, weights range [%.3f, %.3f]",
        n, weights.min(), weights.max(),
    )
    return result


def hrp_instrument_weights(
    returns: pd.DataFrame,
    current_weights: Dict[str, float],
    blend_ratio: float = 0.5,
) -> Dict[str, float]:
    """Blend HRP weights with existing handcrafted weights.

    Parameters
    ----------
    returns : DataFrame of daily returns
    current_weights : existing instrument weights
    blend_ratio : 0.0 = all current, 1.0 = all HRP

    Returns
    -------
    Blended weights dict summing to 1.0.
    Falls back to current_weights if HRP fails.
    """
    hrp = hrp_weights(returns)
    if not hrp:
        return current_weights

    all_tickers = set(current_weights.keys()) | set(hrp.keys())
    blended = {}
    for t in all_tickers:
        w_current = current_weights.get(t, 0.0)
        w_hrp = hrp.get(t, 0.0)
        blended[t] = (1.0 - blend_ratio) * w_current + blend_ratio * w_hrp

    # Renormalize
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended
