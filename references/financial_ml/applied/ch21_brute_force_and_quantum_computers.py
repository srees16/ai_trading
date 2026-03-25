"""
Chapter 21: Brute Force and Quantum Computers
===============================================
Advances in Financial Machine Learning – Marcos López de Prado

Reformulates a dynamic portfolio optimisation problem as an integer
optimisation (combinatorial) problem.  The optimal trading trajectory
is found by evaluating all feasible allocations – a brute-force search
that a quantum computer could carry out via linear superposition.

Snippets:
  21.1 – pigeonHole()     – partitions of k objects into n slots
  21.2 – getAllWeights()   – set Omega of all weight vectors
  21.3 – evalTCosts, evalSR, dynOptPort – evaluate trajectories
  21.4 – rndMatWithRank() – produce a random matrix of given rank
  21.5 – genMean / params – generate problem parameters
  21.6 – statOptPortf     – static optimal portfolio
  21.7 – dynamic solution – compute & compare static vs dynamic SR
"""

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement, product


# ============================================================================
# Snippet 21.1 – Partitions of k Objects into n Slots (Pigeonhole)
# ============================================================================
def pigeonHole(k, n):
    """
    Generate all partitions of k objects into n slots (pigeonhole problem).

    Yields each partition as a list of length n summing to k.

    Parameters
    ----------
    k : int
        Number of objects (units of capital).
    n : int
        Number of slots (assets).

    Yields
    ------
    list
        A partition [p_1, p_2, ..., p_n] with sum = k.
    """
    for j in combinations_with_replacement(range(n), k):
        r = [0] * n
        for i in j:
            r[i] += 1
        yield r


# ============================================================================
# Snippet 21.2 – Set Omega of All Weight Vectors
# ============================================================================
def getAllWeights(k, n):
    """
    Generate all feasible (signed) weight vectors for k units of capital
    and n assets.

    For each partition of k into n slots, every combination of signs
    {-1, +1}^n is considered.

    Parameters
    ----------
    k : int
        Units of capital.
    n : int
        Number of assets.

    Returns
    -------
    np.ndarray
        Shape (n, total_number_of_weight_vectors).
    """
    parts = pigeonHole(k, n)
    w = None
    for part_ in parts:
        w_ = np.array(part_) / float(k)  # absolute weight vector
        for prod_ in product([-1, 1], repeat=n):
            w_signed_ = (w_ * np.array(prod_)).reshape(-1, 1)
            if w is None:
                w = w_signed_.copy()
            else:
                w = np.append(w, w_signed_, axis=1)
    return w


# ============================================================================
# Snippet 21.3 – Evaluate Transaction Costs and Sharpe Ratio
# ============================================================================
def evalTCosts(w, params):
    """
    Compute transaction costs for a particular trajectory.

    Parameters
    ----------
    w : np.ndarray
        Weight matrix, shape (n_assets, n_horizons).
    params : list of dict
        Each dict has key 'c' (cost factors per asset).

    Returns
    -------
    np.ndarray
        Transaction cost per horizon.
    """
    tcost = np.zeros(w.shape[1])
    w_ = np.zeros(shape=w.shape[0])
    for i in range(tcost.shape[0]):
        c_ = params[i]['c']
        tcost[i] = (c_ * abs(w[:, i] - w_) ** 0.5).sum()
        w_ = w[:, i].copy()
    return tcost


def evalSR(params, w, tcost):
    """
    Evaluate the Sharpe Ratio over multiple horizons.

    Parameters
    ----------
    params : list of dict
        Each dict has 'mean' (Nx1) and 'cov' (NxN).
    w : np.ndarray
        Weight matrix (N x H).
    tcost : np.ndarray
        Transaction cost per horizon.

    Returns
    -------
    float
        Sharpe Ratio.
    """
    mean, cov = 0, 0
    for h in range(w.shape[1]):
        params_ = params[h]
        mean += np.dot(w[:, h].T, params_['mean'])[0] - tcost[h]
        cov += np.dot(w[:, h].T, np.dot(params_['cov'], w[:, h]))
    sr = mean / cov ** 0.5 if cov > 0 else 0
    return sr


def dynOptPort(params, k=None):
    """
    Dynamic optimal portfolio: find the trajectory that maximises the
    global Sharpe Ratio across all horizons.

    This is a brute-force search over all possible trajectories –
    intractable for large problems, but amenable to quantum computing.

    Parameters
    ----------
    params : list of dict
        Problem parameters per horizon.
    k : int, optional
        Capital units. Defaults to number of assets.

    Returns
    -------
    np.ndarray
        Optimal weight trajectory (N x H).
    """
    if k is None:
        k = params[0]['mean'].shape[0]
    n = params[0]['mean'].shape[0]
    w_all = getAllWeights(k, n)
    sr, w = None, None
    for prod_ in product(range(w_all.shape[1]), repeat=len(params)):
        w_ = w_all[:, list(prod_)]  # trajectory
        tcost_ = evalTCosts(w_, params)
        sr_ = evalSR(params, w_, tcost_)
        if sr is None or sr < sr_:
            sr, w = sr_, w_.copy()
    return w


# ============================================================================
# Snippet 21.4 – Random Matrix with Given Rank
# ============================================================================
def rndMatWithRank(nSamples, nCols, rank, sigma=0, homNoise=True):
    """
    Produce a random matrix X with a known rank.

    Useful for Monte Carlo experiments and scenario analyses.

    Parameters
    ----------
    nSamples : int
        Number of rows.
    nCols : int
        Number of columns.
    rank : int
        Desired rank.
    sigma : float
        Noise level.
    homNoise : bool
        If True, add homoscedastic noise; else heteroscedastic.

    Returns
    -------
    np.ndarray
        Matrix of shape (nSamples, nCols) with the given rank + noise.
    """
    rng = np.random.RandomState()
    U, _, _ = np.linalg.svd(rng.randn(nCols, nCols))
    x = np.dot(rng.randn(nSamples, rank), U[:, :rank].T)
    if homNoise:
        x += sigma * rng.randn(nSamples, nCols)
    else:
        sigmas = sigma * (rng.rand(nCols) + 0.5)
        x += rng.randn(nSamples, nCols) * sigmas
    return x


# ============================================================================
# Snippet 21.5 – Generate Problem Parameters
# ============================================================================
def genMean(size):
    """Generate a random vector of means."""
    return np.random.normal(size=(size, 1))


def genParams(size=3, horizon=2):
    """
    Generate the problem parameters (means, covariances, costs) for each
    horizon.

    Parameters
    ----------
    size : int
        Number of assets (N).
    horizon : int
        Number of horizons (H).

    Returns
    -------
    list of dict
        Each dict has 'mean', 'cov', and 'c'.
    """
    params = []
    for h in range(horizon):
        x = rndMatWithRank(1000, size, size, 0.0)
        mean_ = genMean(size)
        cov_ = np.cov(x, rowvar=False)
        c_ = np.random.uniform(size=cov_.shape[0]) * np.diag(cov_) ** 0.5
        params.append({'mean': mean_, 'cov': cov_, 'c': c_})
    return params


# ============================================================================
# Snippet 21.6 – Static Optimal Portfolio
# ============================================================================
def statOptPortf(cov, a):
    """
    Compute the static optimal portfolio (unconstrained mean-variance).

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    a : np.ndarray
        Expected returns (column vector).

    Returns
    -------
    np.ndarray
        Optimal weights, rescaled for full investment (sum of |w| = 1).
    """
    cov_inv = np.linalg.inv(cov)
    w = np.dot(cov_inv, a)
    w /= np.dot(np.dot(a.T, cov_inv), a)
    w /= abs(w).sum()  # re-scale for full investment
    return w


# ============================================================================
# DEMO
# ============================================================================
def main():
    """
    Numerical example: compare static vs. dynamic optimal portfolio.
    Uses small dimensions (size=2, horizon=2) so brute-force completes quickly.
    """
    print("=" * 60)
    print("Chapter 21 – Brute Force and Quantum Computers")
    print("=" * 60)

    # --- 1) Pigeonhole partitions -----------------------------------------
    print("\n[1] Pigeonhole partitions of k=4 into n=3 slots:")
    partitions = list(pigeonHole(4, 3))
    print(f"    Number of partitions: {len(partitions)}")
    for p in partitions[:5]:
        print(f"      {p}")
    if len(partitions) > 5:
        print(f"      ... ({len(partitions) - 5} more)")

    # --- 2) All weight vectors -------------------------------------------
    print("\n[2] All weight vectors (k=3, n=2):")
    w_all = getAllWeights(3, 2)
    print(f"    Shape: {w_all.shape} (assets x weight_vectors)")
    print(f"    First 6 vectors:\n{w_all[:, :6]}")

    # --- 3) Random matrix with known rank --------------------------------
    print("\n[3] Random matrix with rank 3 (1000 x 5):")
    X = rndMatWithRank(1000, 5, 3, sigma=0.0)
    _, svals, _ = np.linalg.svd(X, full_matrices=False)
    print(f"    Singular values: {np.round(svals, 4)}")
    print(f"    (Last 2 should be ≈0 → rank 3)")

    # --- 4) Generate parameters ------------------------------------------
    np.random.seed(42)
    size, horizon = 2, 2
    params = genParams(size=size, horizon=horizon)
    print(f"\n[4] Problem: {size} assets, {horizon} horizons")
    for h, p in enumerate(params):
        print(f"    Horizon {h}: mean={p['mean'].flatten()}, "
              f"cost_factors={np.round(p['c'], 4)}")

    # --- 5) Static solution -----------------------------------------------
    print("\n[5] Static optimal portfolio:")
    w_stat = None
    for params_ in params:
        w_ = statOptPortf(cov=params_['cov'], a=params_['mean'])
        if w_stat is None:
            w_stat = w_.copy()
        else:
            w_stat = np.append(w_stat, w_, axis=1)
    tcost_stat = evalTCosts(w_stat, params)
    sr_stat = evalSR(params, w_stat, tcost_stat)
    print(f"    Static weights:\n{np.round(w_stat, 4)}")
    print(f"    Static SR: {sr_stat:.4f}")

    # --- 6) Dynamic solution ----------------------------------------------
    print("\n[6] Dynamic optimal portfolio (brute force):")
    w_dyn = dynOptPort(params, k=size)
    tcost_dyn = evalTCosts(w_dyn, params)
    sr_dyn = evalSR(params, w_dyn, tcost_dyn)
    print(f"    Dynamic weights:\n{np.round(w_dyn, 4)}")
    print(f"    Dynamic SR: {sr_dyn:.4f}")

    # --- 7) Comparison ----------------------------------------------------
    print(f"\n[7] Comparison:")
    print(f"    Static  SR = {sr_stat:.4f}")
    print(f"    Dynamic SR = {sr_dyn:.4f}")
    improvement = (sr_dyn - sr_stat) / abs(sr_stat) * 100 if sr_stat != 0 else float('inf')
    print(f"    Improvement = {improvement:.1f}%")

    # --- 8) Exercise 21.2: eigenvalue analysis ----------------------------
    print("\n[8] Exercise 21.2: Eigenvalue analysis of random-rank matrices")
    for rank in [1, 5, 10]:
        X = rndMatWithRank(1000, 10, rank, sigma=1.0)
        cov = np.cov(X, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        print(f"    Rank={rank}: eigenvalues = {np.round(eigvals, 2)}")

    print("\n✓ Chapter 21 complete")


if __name__ == "__main__":
    main()
