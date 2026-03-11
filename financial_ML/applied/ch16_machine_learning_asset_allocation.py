"""
Chapter 16: Machine Learning Asset Allocation
==============================================
Advances in Financial Machine Learning – Marcos López de Prado

Implements the Hierarchical Risk Parity (HRP) algorithm:
  1. Tree clustering of the correlation matrix
  2. Quasi-diagonalization (reorder rows/columns so largest values lie on diagonal)
  3. Recursive bisection to compute portfolio weights

Snippets:
  16.1 – Tree clustering using scipy functionality
  16.2 – Quasi-diagonalization (getQuasiDiag)
  16.3 – Recursive bisection (getRecBipart)
  16.4 – Full HRP implementation + numerical example
  16.5 – Monte Carlo experiment comparing HRP, CLA (mean-variance), and IVP
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import random
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import get_multi_close, generate_returns, SYMBOLS

warnings.filterwarnings("ignore")


# ============================================================================
# SNIPPET 16.1 – Tree Clustering Using Scipy Functionality
# ============================================================================
def correlDist(corr):
    """
    A distance matrix based on correlation, where 0 <= d[i,j] <= 1.
    This is a proper distance metric (see Appendix 16.A.1).
    """
    dist = ((1 - corr) / 2.0) ** 0.5  # distance matrix
    return dist


def treeCluster(x):
    """
    Snippet 16.1: Compute correlation, distance, and linkage matrices.
    
    Parameters
    ----------
    x : pd.DataFrame
        Matrix of observations (rows=time, columns=assets).
    
    Returns
    -------
    cov, corr, dist, link
    """
    cov, corr = x.cov(), x.corr()
    dist = correlDist(corr)
    # Convert distance matrix to condensed form for linkage
    dist_condensed = squareform(dist.values, checks=False)
    # Ensure no negative values from floating-point issues
    dist_condensed = np.clip(dist_condensed, 0, None)
    link = sch.linkage(dist_condensed, method='single')
    return cov, corr, dist, link


# ============================================================================
# SNIPPET 16.2 – Quasi-Diagonalization
# ============================================================================
def getQuasiDiag(link):
    """
    Snippet 16.2: Sort clustered items by distance.
    
    Reorganizes the rows and columns of the covariance matrix so that
    the largest values lie along the diagonal. Similar investments are
    placed together, dissimilar investments are placed far apart.
    
    Parameters
    ----------
    link : np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage.
    
    Returns
    -------
    list : Sorted list of original item indices.
    """
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()


# ============================================================================
# SNIPPET 16.3 – Recursive Bisection
# ============================================================================
def getIVP(cov, **kargs):
    """Compute the inverse-variance portfolio."""
    diag = np.diag(cov)
    # Avoid division by zero for degenerate covariance matrices
    diag = np.where(diag > 1e-16, diag, 1e-16)
    ivp = 1.0 / diag
    ivp /= ivp.sum()
    return ivp


def getClusterVar(cov, cItems):
    """Compute variance per cluster using inverse-variance weights."""
    cov_ = cov.loc[cItems, cItems]  # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return max(cVar, 1e-16)  # avoid zero variance


def getRecBipart(cov, sortIx):
    """
    Snippet 16.3: Recursive bisection for HRP portfolio weights.
    
    Splits allocations between adjacent subsets in inverse proportion
    to their aggregated variances. Guarantees 0 <= w_i <= 1 and sum(w) = 1.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix.
    sortIx : list
        Sorted indices from quasi-diagonalization.
    
    Returns
    -------
    pd.Series : Portfolio weights.
    """
    sortIx = list(sortIx)  # ensure plain list
    w = pd.Series(1.0, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems
                  for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                  if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]      # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w.loc[cItems0] *= alpha        # weight 1
            w.loc[cItems1] *= 1 - alpha    # weight 2
    return w


# ============================================================================
# SNIPPET 16.4 – Full HRP Implementation + Numerical Example
# ============================================================================
def getHRP(cov, corr):
    """
    Construct a hierarchical risk parity portfolio.
    
    Parameters
    ----------
    cov : pd.DataFrame or np.ndarray
        Covariance matrix.
    corr : pd.DataFrame or np.ndarray
        Correlation matrix.
    
    Returns
    -------
    pd.Series : HRP portfolio weights, sorted by index.
    """
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    # Ensure valid correlation values (handle NaN from degenerate data)
    corr = corr.clip(-1, 1).fillna(0)
    np.fill_diagonal(corr.values, 1.0)
    dist = correlDist(corr)
    dist_condensed = squareform(dist.values, checks=False)
    dist_condensed = np.clip(dist_condensed, 0, None)
    # Replace any remaining NaN with max distance
    np.nan_to_num(dist_condensed, copy=False, nan=1.0)
    link = sch.linkage(dist_condensed, method='single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()


def plotCorrMatrix(path, corr, labels=None):
    """Heatmap of the correlation matrix (optional, requires matplotlib)."""
    try:
        import matplotlib.pyplot as mpl
        if labels is None:
            labels = []
        mpl.pcolor(corr)
        mpl.colorbar()
        mpl.yticks(np.arange(0.5, corr.shape[0] + 0.5), labels)
        mpl.xticks(np.arange(0.5, corr.shape[0] + 0.5), labels)
        mpl.savefig(path)
        mpl.clf()
        mpl.close()
    except Exception as e:
        print(f"  [plotCorrMatrix] Skipping plot: {e}")


def generateData(nObs, size0, size1, sigma1):
    """
    Snippet 16.4: Generate time series of correlated variables.
    
    Parameters
    ----------
    nObs : int
        Number of observations.
    size0 : int
        Number of uncorrelated variables.
    size1 : int
        Number of correlated variables (perturbations of uncorrelated ones).
    sigma1 : float
        Noise level for correlated variables.
    
    Returns
    -------
    x : pd.DataFrame
        Matrix of observations.
    cols : list
        Indices mapping correlated variables to their uncorrelated sources.
    """
    np.random.seed(seed=12345)
    random.seed(12345)
    x = np.random.normal(0, 1, size=(nObs, size0))
    # Create correlation between variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols


def numericalExample():
    """Run the numerical example from Section 16.5 / Snippet 16.4."""
    print("\n" + "=" * 70)
    print("SNIPPET 16.4 – Numerical Example (Synthetic Correlated Data)")
    print("=" * 70)

    # 1) Generate correlated data
    nObs, size0, size1, sigma1 = 10000, 5, 5, 0.25
    x, cols = generateData(nObs, size0, size1, sigma1)
    print(f"  Correlated pairs: {[(j+1, size0+i) for i, j in enumerate(cols, 1)]}")

    cov, corr = x.cov(), x.corr()

    # 2) Cluster
    dist = correlDist(corr)
    dist_condensed = squareform(dist.values, checks=False)
    dist_condensed = np.clip(dist_condensed, 0, None)
    link = sch.linkage(dist_condensed, method='single')

    # 3) Quasi-diagonalize
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    df0 = corr.loc[sortIx, sortIx]  # reorder
    print(f"  Quasi-diagonal order: {sortIx}")

    # 4) Capital allocation – HRP
    hrp = getRecBipart(cov, sortIx)
    print("\n  HRP Weights:")
    for k, v in hrp.sort_index().items():
        print(f"    Asset {k}: {v:.4f} ({v*100:.2f}%)")

    # 5) Compare with IVP
    ivp = getIVP(cov)
    ivp = pd.Series(ivp, index=cov.columns)
    print("\n  IVP Weights:")
    for k, v in ivp.items():
        print(f"    Asset {k}: {v:.4f} ({v*100:.2f}%)")

    return hrp, ivp


# ============================================================================
# SNIPPET 16.5 – Monte Carlo Experiment (HRP vs. CLA/MV vs. IVP)
# ============================================================================
def getCLA(cov, **kargs):
    """
    Simplified mean-variance minimum-variance portfolio using scipy.optimize.
    Substitutes the full CLA implementation with a constrained optimizer.
    
    min  w' V w
    s.t. sum(w) = 1, 0 <= w_i <= 1
    """
    n = cov.shape[0]
    cov_arr = np.array(cov)

    def portfolio_var(w):
        return np.dot(w, np.dot(cov_arr, w))

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n  # equal-weight initial guess
    result = minimize(portfolio_var, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-12, 'maxiter': 1000})
    return result.x


def generateDataMC(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    """
    Snippet 16.5: Generate correlated data with random shocks for MC experiment.
    
    Parameters
    ----------
    nObs : int
        Total number of observations.
    sLength : int
        In-sample lookback length.
    size0, size1 : int
        Number of uncorrelated / correlated variables.
    mu0 : float
        Mean of returns.
    sigma0 : float
        Std dev of returns.
    sigma1F : float
        Correlation noise factor.
    
    Returns
    -------
    x : np.ndarray, cols : list
    """
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))
    # Create correlation between variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F,
                                       size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    # Add common random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-0.5, -0.5], [2, 2]])
    # Add specific random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[point, cols[-1]] = np.array([-0.5, 2])
    return x, cols


def hrpMC(numIters=100, nObs=520, size0=5, size1=5, mu0=0, sigma0=1e-2,
          sigma1F=0.25, sLength=260, rebal=22):
    """
    Snippet 16.5: Monte Carlo experiment comparing out-of-sample performance
    of HRP, mean-variance (CLA), and inverse-variance (IVP) portfolios.
    
    Parameters
    ----------
    numIters : int
        Number of MC iterations (book uses 10000; default 100 for speed).
    nObs : int
        Total observations per simulation (520 ≈ 2 years daily).
    size0, size1 : int
        Number of uncorrelated / correlated series.
    mu0, sigma0 : float
        Mean and std dev of base returns.
    sigma1F : float
        Noise factor for correlated series.
    sLength : int
        In-sample lookback window (260 ≈ 1 year daily).
    rebal : int
        Rebalancing frequency in observations (22 ≈ monthly).
    
    Returns
    -------
    stats : pd.DataFrame
        Out-of-sample cumulative-return statistics per method per iteration.
    """
    print("\n" + "=" * 70)
    print("SNIPPET 16.5 – Monte Carlo: HRP vs. Mean-Variance vs. IVP")
    print(f"  Iterations: {numIters}, Obs: {nObs}, Lookback: {sLength}, "
          f"Rebal: {rebal}")
    print("=" * 70)

    methods = [
        ('getIVP', lambda cov, corr: getIVP(cov)),
        ('getHRP', lambda cov, corr: getHRP(cov, corr)),
        ('getCLA', lambda cov, corr: getCLA(cov)),
    ]
    stats = {name: pd.Series(dtype=float) for name, _ in methods}
    pointers = list(range(sLength, nObs, rebal))

    for numIter in range(int(numIters)):
        if (numIter + 1) % 25 == 0 or numIter == 0:
            print(f"  Iteration {numIter + 1}/{int(numIters)} ...")

        # 1) Prepare data for one experiment
        x, cols = generateDataMC(nObs, sLength, size0, size1,
                                  mu0, sigma0, sigma1F)
        r = {name: pd.Series(dtype=float) for name, _ in methods}

        # 2) Compute portfolios in-sample, evaluate out-of-sample
        for pointer in pointers:
            x_ = x[pointer - sLength:pointer]
            cov_ = np.cov(x_, rowvar=0)
            corr_ = np.corrcoef(x_, rowvar=0)

            # Out-of-sample returns
            x_oos = x[pointer:pointer + rebal]
            for name, func in methods:
                w_ = func(cov=cov_, corr=corr_)
                r_ = pd.Series(np.dot(x_oos, w_))
                r[name] = pd.concat([r[name], r_])

        # 3) Evaluate and store results
        for name, _ in methods:
            r_ = r[name].reset_index(drop=True)
            p_ = (1 + r_).cumprod()
            stats[name].loc[numIter] = p_.iloc[-1] - 1

    # 4) Report results
    stats = pd.DataFrame.from_dict(stats, orient='columns')
    df_std = stats.std()
    df_var = stats.var()
    df_ratio = df_var / df_var['getHRP'] - 1

    results = pd.concat([df_std, df_var, df_ratio], axis=1)
    results.columns = ['Std Dev', 'Variance', 'Var Ratio vs HRP']
    print("\n  Out-of-Sample Results:")
    print(results.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\n  CLA variance is {df_ratio['getCLA']*100:.2f}% "
          f"{'higher' if df_ratio['getCLA'] > 0 else 'lower'} than HRP")
    print(f"  IVP variance is {df_ratio['getIVP']*100:.2f}% "
          f"{'higher' if df_ratio['getIVP'] > 0 else 'lower'} than HRP")

    return stats


# ============================================================================
# MAIN – Demos with real and synthetic data
# ============================================================================
if __name__ == '__main__':
    print("Chapter 16: Machine Learning Asset Allocation")
    print("Hierarchical Risk Parity (HRP)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # DEMO 1: HRP on real market data (MSFT, GOOG, NVDA, AMD)
    # ------------------------------------------------------------------
    print(f"\n[DEMO 1] HRP on real data: {SYMBOLS}")
    print("-" * 50)

    closes = get_multi_close()
    print(f"  Close prices: {closes.shape[0]} days x {closes.shape[1]} assets")
    returns = closes.pct_change().dropna()

    cov = returns.cov()
    corr = returns.corr()

    print(f"\n  Correlation matrix:")
    print(corr.to_string(float_format=lambda x: f"{x:.4f}"))

    # Snippet 16.1: Tree clustering
    dist = correlDist(corr)
    dist_condensed = squareform(dist.values, checks=False)
    dist_condensed = np.clip(dist_condensed, 0, None)
    link = sch.linkage(dist_condensed, method='single')
    print(f"\n  Linkage matrix shape: {link.shape}")

    # Snippet 16.2: Quasi-diagonalization
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    print(f"  Quasi-diagonal order: {sortIx}")

    # Snippet 16.3: Recursive bisection → HRP weights
    hrp_weights = getRecBipart(cov, sortIx)
    hrp_weights = hrp_weights.sort_index()

    # IVP weights for comparison
    ivp_weights = getIVP(cov)
    ivp_weights = pd.Series(ivp_weights, index=cov.columns)

    # Mean-variance (CLA substitute) weights
    mv_weights = getCLA(cov)
    mv_weights = pd.Series(mv_weights, index=cov.columns)

    print(f"\n  {'Method':<8} ", end="")
    for sym in cov.columns:
        print(f"{sym:>10}", end="")
    print()
    print("  " + "-" * (8 + 10 * len(cov.columns)))

    print(f"  {'HRP':<8} ", end="")
    for sym in cov.columns:
        print(f"{hrp_weights[sym]:>10.4f}", end="")
    print()

    print(f"  {'IVP':<8} ", end="")
    for sym in cov.columns:
        print(f"{ivp_weights[sym]:>10.4f}", end="")
    print()

    print(f"  {'MV':<8} ", end="")
    for sym in cov.columns:
        print(f"{mv_weights[sym]:>10.4f}", end="")
    print()

    # Portfolio standard deviations
    w_hrp = hrp_weights.values
    w_ivp = ivp_weights.values
    w_mv = mv_weights
    cov_arr = cov.values
    sigma_hrp = np.sqrt(np.dot(w_hrp, np.dot(cov_arr, w_hrp)))
    sigma_ivp = np.sqrt(np.dot(w_ivp, np.dot(cov_arr, w_ivp)))
    sigma_mv = np.sqrt(np.dot(w_mv, np.dot(cov_arr, w_mv)))
    print(f"\n  Portfolio daily σ:  HRP={sigma_hrp:.6f}  "
          f"IVP={sigma_ivp:.6f}  MV={sigma_mv:.6f}")

    # ------------------------------------------------------------------
    # DEMO 2: Snippet 16.4 – Numerical example (synthetic data)
    # ------------------------------------------------------------------
    numericalExample()

    # ------------------------------------------------------------------
    # DEMO 3: Snippet 16.5 – Monte Carlo experiment (100 iterations)
    # ------------------------------------------------------------------
    mc_stats = hrpMC(numIters=100)

    print("\n" + "=" * 70)
    print("Chapter 16 complete.")
