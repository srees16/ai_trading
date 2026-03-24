"""
Chapter 3: Optimization Issues
================================
Testing and Tuning Market Trading Systems — Timothy Masters

Algorithms:
  - Regularized linear model via Coordinate Descent (elastic-net)
    (CoordinateDescent class, pp.50–70, C++ conversion)
  - Lambda path optimization with cross-validation
  - Making a linear model nonlinear (basis expansion)
  - Differential Evolution universal nonlinear optimizer (DIFF_EV.CPP, pp.82–96)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_data import generate_returns, generate_indicator_series, get_close_series


# ======================================================================
# Soft Thresholding Function — core of coordinate descent (Eq 3-6)
# ======================================================================

def _soft_threshold(z, threshold):
    """Soft-thresholding operator S(z, λ).

    S(z, λ) = sign(z) * max(|z| - λ, 0)
    """
    if z > threshold:
        return z - threshold
    elif z < -threshold:
        return z + threshold
    return 0.0


# ======================================================================
# Coordinate Descent — Elastic-Net Regularized Linear Model
# (Converted from CoordinateDescent class, pp.50–70)
# ======================================================================

class CoordinateDescent:
    """Elastic-net regularized linear regression via coordinate descent.

    Minimises:
        (1/2N) ||y - Xβ||² + λ[α||β||₁ + (1-α)/2 ||β||₂²]

    Parameters
    ----------
    alpha   : float – mixing parameter (1=Lasso, 0=Ridge, 0<α<1=elastic-net)
    max_iter: int   – maximum iterations for convergence
    tol     : float – convergence tolerance
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.beta_ = None
        self.intercept_ = 0.0
        self.lambda_ = 0.0

    def fit(self, X, y, lam=0.1, weights=None, warm_start=False):
        """Fit the model via coordinate descent.

        Converted from core_train() in CoordinateDescent,
        pp.55–61 of the book.

        Parameters
        ----------
        X          : np.ndarray (n_cases, n_vars)
        y          : np.ndarray (n_cases,)
        lam        : float – regularization strength λ
        weights    : np.ndarray (n_cases,) or None
        warm_start : bool – start from current betas

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.lambda_ = lam

        # Normalise weights
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
        else:
            w = np.full(n, 1.0 / n)

        # Precompute weighted X'X diagonal and X'y (Eq 3-13, 3-14)
        xss = np.array([np.sum(w * X[:, j] ** 2) for j in range(p)])
        xy_inner = np.array([np.sum(w * X[:, j] * y) for j in range(p)])

        # Initialise betas
        if warm_start and self.beta_ is not None and len(self.beta_) == p:
            beta = self.beta_.copy()
        else:
            beta = np.zeros(p)

        resid = y - X @ beta

        s_threshold = self.alpha * lam
        do_active_only = False

        for iteration in range(self.max_iter):
            max_change = 0.0
            active_changed = False

            for j in range(p):
                if do_active_only and beta[j] == 0.0:
                    continue

                # Partial residual: add back current beta's contribution
                rho = np.sum(w * X[:, j] * resid) + xss[j] * beta[j]

                # Apply soft-thresholding (Eq 3-6)
                new_beta = _soft_threshold(rho, s_threshold)
                denom = xss[j] + lam * (1.0 - self.alpha)
                if denom > 0:
                    new_beta /= denom
                else:
                    new_beta = 0.0

                # Track changes
                change = abs(new_beta - beta[j])
                if change > max_change:
                    max_change = change

                if (beta[j] == 0.0) != (new_beta == 0.0):
                    active_changed = True

                # Update residuals incrementally
                if change > 0:
                    resid += X[:, j] * (beta[j] - new_beta)
                    beta[j] = new_beta

            # Convergence check (pp.53–54)
            converged = max_change < self.tol
            if do_active_only:
                if converged:
                    do_active_only = False  # full pass next
            else:
                if converged and not active_changed:
                    break
                do_active_only = True

        self.beta_ = beta
        self.intercept_ = 0.0  # data assumed centred
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        return np.asarray(X) @ self.beta_ + self.intercept_

    def lambda_path(self, X, y, n_lambdas=50, lambda_ratio=1e-3,
                    weights=None):
        """Descend a lambda path from λ_max to λ_min (p.62–66).

        Returns a DataFrame of betas at each lambda value.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        # λ_max: smallest λ that zeros all coefficients
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
        else:
            w = np.full(n, 1.0 / n)

        lambda_max = np.max(np.abs(np.sum(w[:, None] * X * y[:, None],
                                          axis=0))) / max(self.alpha, 1e-10)
        lambda_min = lambda_max * lambda_ratio
        lambdas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min),
                                     n_lambdas))

        paths = np.zeros((n_lambdas, p))
        for i, lam in enumerate(lambdas):
            self.fit(X, y, lam=lam, weights=weights, warm_start=(i > 0))
            paths[i] = self.beta_.copy()

        return pd.DataFrame(
            paths,
            columns=[f"beta_{j}" for j in range(p)],
            index=pd.Index(lambdas, name="lambda"),
        )


# ======================================================================
# Lambda Selection via Cross-Validation (pp.66–70)
# ======================================================================

def cv_lambda_search(X, y, n_lambdas=30, n_folds=5, alpha=1.0, seed=42):
    """Select optimal λ via k-fold cross-validation.

    Parameters
    ----------
    X         : np.ndarray (n, p)
    y         : np.ndarray (n,)
    n_lambdas : int – number of λ values to test
    n_folds   : int – number of CV folds
    alpha     : float – elastic-net mixing
    seed      : int

    Returns
    -------
    dict with 'lambdas', 'cv_errors', 'best_lambda', 'best_error'
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    fold_size = n // n_folds

    # Determine lambda range
    w = np.full(n, 1.0 / n)
    lambda_max = np.max(np.abs(np.sum(w[:, None] * X * y[:, None],
                                      axis=0))) / max(alpha, 1e-10)
    lambdas = np.exp(np.linspace(np.log(lambda_max),
                                 np.log(lambda_max * 1e-3), n_lambdas))

    cv_errors = np.zeros(n_lambdas)

    for lam_i, lam in enumerate(lambdas):
        fold_errors = []
        for fold in range(n_folds):
            test_idx = idx[fold * fold_size: (fold + 1) * fold_size]
            train_idx = np.setdiff1d(idx, test_idx)

            model = CoordinateDescent(alpha=alpha)
            model.fit(X[train_idx], y[train_idx], lam=lam)
            pred = model.predict(X[test_idx])
            mse = np.mean((y[test_idx] - pred) ** 2)
            fold_errors.append(mse)
        cv_errors[lam_i] = np.mean(fold_errors)

    best_i = np.argmin(cv_errors)
    return {
        "lambdas": lambdas,
        "cv_errors": cv_errors,
        "best_lambda": lambdas[best_i],
        "best_error": cv_errors[best_i],
    }


# ======================================================================
# Differential Evolution — Universal Nonlinear Optimizer
# (Converted from DIFF_EV.CPP, pp.82–96)
# ======================================================================

def differential_evolution(objective, bounds, pop_size=50, max_gen=200,
                           F=0.8, CR=0.9, seed=42):
    """Differential evolution optimizer.

    Converted from DIFF_EV.CPP. Minimises *objective(x)*.

    Parameters
    ----------
    objective : callable – f(x) → float, where x is a 1-d array
    bounds    : list of (lo, hi) – parameter bounds
    pop_size  : int   – population size
    max_gen   : int   – maximum generations
    F         : float – differential weight (mutation factor)
    CR        : float – crossover probability
    seed      : int

    Returns
    -------
    dict with 'best_x', 'best_val', 'history' (best value per generation)
    """
    rng = np.random.default_rng(seed)
    ndim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Initialize population uniformly in bounds
    pop = lo + (hi - lo) * rng.random((pop_size, ndim))
    fitness = np.array([objective(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_x = pop[best_idx].copy()
    history = [best_val]

    for gen in range(max_gen):
        for i in range(pop_size):
            # Select three distinct random indices ≠ i
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(candidates, 3, replace=False)

            # Mutation: donor = pop[a] + F * (pop[b] - pop[c])
            donor = pop[a] + F * (pop[b] - pop[c])

            # Clip to bounds
            donor = np.clip(donor, lo, hi)

            # Crossover
            trial = pop[i].copy()
            j_rand = rng.integers(ndim)
            for j in range(ndim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = donor[j]

            # Selection
            trial_val = objective(trial)
            if trial_val <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_val
                if trial_val < best_val:
                    best_val = trial_val
                    best_x = trial.copy()

        history.append(best_val)

    return {
        "best_x": best_x,
        "best_val": best_val,
        "history": history,
    }


# ======================================================================
# Making a Linear Model Nonlinear — Basis Expansion (p.74)
# ======================================================================

def polynomial_basis(X, degree=2):
    """Expand features into polynomial basis (including interactions).

    Parameters
    ----------
    X      : np.ndarray (n, p)
    degree : int – max polynomial degree

    Returns
    -------
    np.ndarray – expanded features
    """
    n, p = X.shape
    from itertools import combinations_with_replacement
    cols = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(p), d):
            col = np.ones(n)
            for idx in combo:
                col *= X[:, idx]
            cols.append(col)
    return np.column_stack(cols)


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("Chapter 3 – Optimization Issues")
    print("  (Testing and Tuning Market Trading Systems)")
    print("=" * 70)

    # --- 1. Coordinate Descent on synthetic data ---
    print("\n--- Coordinate Descent (Elastic-Net) ---")
    rng = np.random.default_rng(42)
    n, p = 500, 10
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -0.8, 0.5]  # only 3 non-zero
    X = rng.normal(0, 1, (n, p))
    y = X @ true_beta + rng.normal(0, 0.5, n)

    model = CoordinateDescent(alpha=0.8)
    model.fit(X, y, lam=0.05)
    print(f"  True betas:  {true_beta}")
    print(f"  Fitted betas: {np.round(model.beta_, 4)}")
    print(f"  Non-zero:     {np.sum(np.abs(model.beta_) > 1e-6)} / {p}")

    # --- 2. Lambda Path ---
    print("\n--- Lambda Path ---")
    path_df = model.lambda_path(X, y, n_lambdas=40)
    print(f"  Lambda range: [{path_df.index.min():.6f}, {path_df.index.max():.4f}]")
    print(f"  Path shape:   {path_df.shape}")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for col in path_df.columns:
        ax1.plot(np.log(path_df.index), path_df[col], linewidth=1)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("log(λ)")
    ax1.set_ylabel("β coefficient")
    ax1.set_title("Lambda Path — Coefficient Evolution")
    fig1.tight_layout()

    # --- 3. CV Lambda Selection ---
    print("\n--- Cross-Validation Lambda Selection ---")
    cv_result = cv_lambda_search(X, y, n_lambdas=25, n_folds=5, alpha=0.8)
    print(f"  Best λ:     {cv_result['best_lambda']:.6f}")
    print(f"  Best MSE:   {cv_result['best_error']:.6f}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(np.log(cv_result["lambdas"]), cv_result["cv_errors"],
             marker="o", markersize=3, color="steelblue")
    ax2.axvline(np.log(cv_result["best_lambda"]), color="tomato",
                linestyle="--", label=f"Best λ={cv_result['best_lambda']:.4f}")
    ax2.set_xlabel("log(λ)")
    ax2.set_ylabel("CV MSE")
    ax2.set_title("Cross-Validation for Lambda Selection")
    ax2.legend()
    fig2.tight_layout()

    # --- 4. Differential Evolution ---
    print("\n--- Differential Evolution ---")

    def rastrigin(x):
        """Rastrigin function — multimodal test function."""
        A = 10
        return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

    bounds = [(-5.12, 5.12)] * 5
    de_result = differential_evolution(rastrigin, bounds, pop_size=60,
                                       max_gen=150, seed=42)
    print(f"  Best value: {de_result['best_val']:.8f}  (global min = 0.0)")
    print(f"  Best x:     {np.round(de_result['best_x'], 4)}")

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(de_result["history"], color="steelblue", linewidth=1)
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Best Objective Value")
    ax3.set_title("Differential Evolution Convergence (5-D Rastrigin)")
    ax3.set_yscale("log")
    fig3.tight_layout()

    # --- 5. Nonlinear Basis Expansion ---
    print("\n--- Polynomial Basis Expansion ---")
    X_simple = rng.normal(0, 1, (200, 2))
    X_poly = polynomial_basis(X_simple, degree=3)
    print(f"  Original features: {X_simple.shape[1]}")
    print(f"  Expanded features: {X_poly.shape[1]}")

    print("\nChapter 3 complete.")


if __name__ == "__main__":
    main()
