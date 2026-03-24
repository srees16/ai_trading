"""
ch06_ensemble_methods.py – Chapter 6: Ensemble Methods
From "Advances in Financial Machine Learning" by Marcos López de Prado.

Implements Snippets 6.1–6.2:
  6.1   bagging_accuracy           – Accuracy of the bagging classifier (analytical)
  6.2   three_ways_to_setup_rf     – Three ways of setting up a Random Forest
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import comb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Allow imports from the parent directory (financial_ML/)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_classification_data


# ============================================================================
# Snippet 6.1 – Accuracy of the Bagging Classifier
# ============================================================================

def bagging_accuracy(N, p, k):
    """
    Compute the probability that a bagging classifier (majority-voting among
    *N* independent classifiers each with accuracy *p*) makes a correct
    prediction, for *k* classes.

    The probability that the correct class receives more than N/k votes is:
        P[X > N/k] = 1 - sum_{i=0}^{floor(N/k)} C(N,i) * p^i * (1-p)^{N-i}

    Parameters
    ----------
    N : int   – number of independent classifiers (estimators).
    p : float – accuracy of each individual classifier.
    k : int   – number of classes.

    Returns
    -------
    (p, p_ensemble) : tuple of (float, float)
        Individual accuracy and ensemble accuracy.
    """
    p_cum = 0.0
    for i in range(0, int(N / k) + 1):
        p_cum += comb(N, i, exact=True) * p ** i * (1 - p) ** (N - i)
    return p, 1 - p_cum


# ============================================================================
# Snippet 6.2 – Three Ways of Setting Up an RF
# ============================================================================

def three_ways_to_setup_rf(X, y, avg_uniqueness=0.5, n_estimators=1000):
    """
    Demonstrate three alternative ways of setting up an RF using sklearn,
    as described in Snippet 6.2.

    Parameters
    ----------
    X : pd.DataFrame – feature matrix.
    y : pd.Series    – labels.
    avg_uniqueness : float – average uniqueness of observations (avgU).
        Controls max_samples for bagged classifiers.
    n_estimators : int – number of estimators for each approach.

    Returns
    -------
    dict – mapping from method name to fitted classifier.
    """
    avgU = avg_uniqueness

    # Method 0: Standard RandomForest
    clf0 = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced_subsample',
        criterion='entropy',
        random_state=42,
        n_jobs=-1,
    )

    # Method 1: BaggingClassifier wrapping a DecisionTreeClassifier
    dt = DecisionTreeClassifier(
        criterion='entropy',
        max_features='sqrt',
        class_weight='balanced',
    )
    clf1 = BaggingClassifier(
        estimator=dt,
        n_estimators=n_estimators,
        max_samples=avgU,
        random_state=42,
        n_jobs=-1,
    )

    # Method 2: BaggingClassifier wrapping a single-tree RandomForestClassifier
    rf_single = RandomForestClassifier(
        n_estimators=1,
        criterion='entropy',
        bootstrap=False,
        class_weight='balanced_subsample',
    )
    clf2 = BaggingClassifier(
        estimator=rf_single,
        n_estimators=n_estimators,
        max_samples=avgU,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    classifiers = {
        'RF (standard)': clf0,
        'Bagging+DT': clf1,
        'Bagging+RF(1 tree)': clf2,
    }

    for name, clf in classifiers.items():
        clf.fit(X, y)

    return classifiers


# ============================================================================
# Main – demonstrate each snippet
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 6 – Ensemble Methods")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Snippet 6.1 – Analytical accuracy of bagging classifier
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Snippet 6.1 – Accuracy of the Bagging Classifier")
    print("-" * 70)

    N, p, k = 100, 1.0 / 3, 3
    p_ind, p_bag = bagging_accuracy(N, p, k)
    print(f"  N={N}, p={p_ind:.4f}, k={k}")
    print(f"  Individual accuracy : {p_ind:.4f}")
    print(f"  Bagging accuracy    : {p_bag:.4f}")

    print("\n  Varying p for N=100, k=2:")
    print(f"  {'p':>8s}  {'Ensemble':>10s}  {'Improvement':>12s}")
    for p_val in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]:
        _, p_ens = bagging_accuracy(100, p_val, 2)
        print(f"  {p_val:8.2f}  {p_ens:10.4f}  {p_ens - p_val:+12.4f}")

    print("\n  Varying N for p=0.55, k=2:")
    print(f"  {'N':>8s}  {'Ensemble':>10s}")
    for N_val in [10, 50, 100, 500, 1000]:
        _, p_ens = bagging_accuracy(N_val, 0.55, 2)
        print(f"  {N_val:8d}  {p_ens:10.4f}")

    # ------------------------------------------------------------------
    # Snippet 6.2 – Three ways to set up an RF
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Snippet 6.2 – Three Ways of Setting Up an RF")
    print("-" * 70)

    X, y = generate_classification_data(n_samples=2000, n_features=20,
                                        n_informative=5, seed=42)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Use fewer estimators for speed in demo
    n_est = 100
    avg_uniq = 0.5
    classifiers = three_ways_to_setup_rf(X, y, avg_uniqueness=avg_uniq,
                                         n_estimators=n_est)

    print(f"\n  Cross-validated accuracy (5-fold) with n_estimators={n_est}:")
    print(f"  {'Method':<25s}  {'Mean Acc':>10s}  {'Std':>8s}")
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        print(f"  {name:<25s}  {scores.mean():10.4f}  {scores.std():8.4f}")

    print("\n  Training-set accuracy (in-sample):")
    for name, clf in classifiers.items():
        acc = clf.score(X, y)
        print(f"  {name:<25s}  {acc:.4f}")

    print("\n" + "=" * 70)
    print("All Chapter 6 snippets executed successfully.")
    print("=" * 70)
