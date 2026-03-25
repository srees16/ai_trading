"""
Chapter 9 – Hyper-Parameter Tuning with Cross-Validation
=========================================================
Implements Snippets 9.1–9.4 from *Advances in Financial Machine Learning*
by Marcos López de Prado.

Snippets
--------
9.1  clfHyperFit  – Grid search with purged K-fold CV
9.2  MyPipeline   – Enhanced Pipeline class (handles sample_weight)
9.3  clfHyperFit  – Randomized search with purged K-fold CV (extends 9.1)
9.4  logUniform_gen / logUniform – Log-uniform distribution for random
     hyper-parameter sampling
"""

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, kstest

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._split import _BaseKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_classification_data

# ──────────────────────────────────────────────────────────────────────────────
# PurgedKFold – minimal self-contained version (from Chapter 7, Snippet 7.3)
# ──────────────────────────────────────────────────────────────────────────────
class PurgedKFold(_BaseKFold):
    """
    K-fold CV that purges training observations whose labels overlap
    with the test set, and optionally applies an embargo.
    """

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pandas Series")
        super(PurgedKFold, self).__init__(
            n_splits=n_splits, shuffle=False, random_state=None,
        )
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]
        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(
                self.t1.iloc[test_indices].max()
            )
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index
            )
            if maxT1Idx + mbrg < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx + mbrg:])
                )
            yield train_indices, test_indices


# ──────────────────────────────────────────────────────────────────────────────
# Helper: create synthetic t1 (label end-times)
# ──────────────────────────────────────────────────────────────────────────────
def _make_t1(index, horizon=5):
    """For each bar in *index*, assign an end-time *horizon* bars ahead."""
    t1 = pd.Series(index=index, dtype="datetime64[ns]")
    for loc in range(len(index)):
        end_loc = min(loc + horizon, len(index) - 1)
        t1.iloc[loc] = index[end_loc]
    return t1


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 9.2 – MyPipeline: Enhanced Pipeline class
# ══════════════════════════════════════════════════════════════════════════════
class MyPipeline(Pipeline):
    """Pipeline subclass that forwards *sample_weight* to the final estimator."""

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 9.1 – Grid search with purged K-fold CV
# (basic version without randomized search)
# ══════════════════════════════════════════════════════════════════════════════
def clfHyperFitGrid(feat, lbl, t1, pipe_clf, param_grid, cv=3,
                    bagging=(0, None, 1.0), n_jobs=-1, pctEmbargo=0,
                    **fit_params):
    """
    Snippet 9.1 – Purged grid-search cross-validation.

    Parameters
    ----------
    feat      : pd.DataFrame – feature matrix (DatetimeIndex).
    lbl       : pd.Series    – labels (same index as feat).
    t1        : pd.Series    – label end-times (index=bar, value=end).
    pipe_clf  : Pipeline     – sklearn Pipeline wrapping the estimator.
    param_grid: dict         – grid of hyper-parameter values.
    cv        : int          – number of CV folds.
    bagging   : tuple        – (n_estimators, max_samples, max_features).
                               Set max_samples to None or <=0 to skip bagging.
    n_jobs    : int          – parallel jobs (-1 = all cores).
    pctEmbargo: float        – embargo fraction.

    Returns
    -------
    gs : Pipeline – fitted (and optionally bagged) best estimator.
    """
    if set(lbl.values) == {0, 1}:
        scoring = "f1"
    else:
        scoring = "neg_log_loss"

    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    gs = GridSearchCV(
        estimator=pipe_clf, param_grid=param_grid,
        scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
    )
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_

    if bagging[1] is not None and bagging[1] > 0:
        gs = BaggingClassifier(
            estimator=MyPipeline(gs.steps),
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]),
            max_features=float(bagging[2]),
            n_jobs=n_jobs,
        )
        gs = gs.fit(
            feat, lbl,
            sample_weight=fit_params.get(
                gs.estimator.steps[-1][0] + "__sample_weight"
            ),
        )
        gs = Pipeline([("bag", gs)])
    return gs


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 9.3 – Randomized search with purged K-fold CV (extends 9.1)
# ══════════════════════════════════════════════════════════════════════════════
def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3,
                bagging=(0, None, 1.0), rndSearchIter=0,
                n_jobs=-1, pctEmbargo=0, **fit_params):
    """
    Snippet 9.3 – Purged grid / randomized-search cross-validation.

    When *rndSearchIter* == 0 a full grid search is performed (Snippet 9.1).
    When *rndSearchIter* > 0 a randomized search is performed instead, drawing
    *rndSearchIter* parameter combinations from the distributions given in
    *param_grid*.
    """
    if set(lbl.values) == {0, 1}:
        scoring = "f1"
    else:
        scoring = "neg_log_loss"

    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)

    if rndSearchIter == 0:
        gs = GridSearchCV(
            estimator=pipe_clf, param_grid=param_grid,
            scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
        )
    else:
        gs = RandomizedSearchCV(
            estimator=pipe_clf, param_distributions=param_grid,
            scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
            n_iter=rndSearchIter,
        )

    gs = gs.fit(feat, lbl, **fit_params).best_estimator_

    if bagging[1] is not None and bagging[1] > 0:
        gs = BaggingClassifier(
            estimator=MyPipeline(gs.steps),
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]),
            max_features=float(bagging[2]),
            n_jobs=n_jobs,
        )
        gs = gs.fit(
            feat, lbl,
            sample_weight=fit_params.get(
                gs.estimator.steps[-1][0] + "__sample_weight"
            ),
        )
        gs = Pipeline([("bag", gs)])
    return gs


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 9.4 – logUniform_gen class
# ══════════════════════════════════════════════════════════════════════════════
class logUniform_gen(rv_continuous):
    """
    Random variable whose logarithm is uniformly distributed.

    If x ~ logUniform(a, b) then log(x) ~ Uniform(log(a), log(b)).
    Useful for sampling hyper-parameters like C or gamma on a log scale.
    """

    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a=1, b=np.exp(1)):
    """Return a frozen log-uniform distribution on [a, b]."""
    return logUniform_gen(a=a, b=b, name="logUniform")


# ══════════════════════════════════════════════════════════════════════════════
# Adapted cvScore (from Ch7 Snippet 7.4, mentioned in Ch9 §9.4)
# ══════════════════════════════════════════════════════════════════════════════
def cvScore(clf, X, y, sample_weight, scoring="neg_log_loss",
            t1=None, cv=None, cvGen=None, pctEmbargo=None):
    """
    Cross-validated score using PurgedKFold.

    Preferred over sklearn's built-in cross_val_score when using
    neg_log_loss scoring (avoids sklearn bug #9144).
    """
    if scoring not in ("neg_log_loss", "accuracy"):
        raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'.")
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)

    scores = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(
            X.iloc[train, :], y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            sc = -log_loss(
                y.iloc[test], prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_,
            )
        else:
            pred = fit.predict(X.iloc[test, :])
            sc = accuracy_score(
                y.iloc[test], pred,
                sample_weight=sample_weight.iloc[test].values,
            )
        scores.append(sc)
    return np.array(scores)


# ======================================================================
#  MAIN – demonstrate each snippet
# ======================================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("Chapter 9 – Hyper-Parameter Tuning with Cross-Validation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Generate data
    # ------------------------------------------------------------------
    N_SAMPLES, N_FEATURES, N_INFO = 1000, 10, 5
    X, y = generate_classification_data(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFO, seed=42,
    )
    t1 = _make_t1(X.index, horizon=5)
    sample_weight = pd.Series(np.ones(len(y)), index=y.index)

    # Build a simple pipeline: SVC with RBF kernel
    pipe_clf = Pipeline([
        ("svc", SVC(kernel="rbf", probability=True)),
    ])

    # ==================================================================
    # Snippet 9.2 – MyPipeline
    # ==================================================================
    print("\n--- Snippet 9.2: MyPipeline ---")
    mp = MyPipeline([("svc", SVC(kernel="rbf", probability=True))])
    mp.fit(X, y, sample_weight=sample_weight.values)
    print(f"  MyPipeline fitted OK, classes = {mp.classes_}")

    # ==================================================================
    # Snippet 9.1 – Grid search with purged K-fold CV
    # ==================================================================
    print("\n--- Snippet 9.1: clfHyperFitGrid (Grid Search) ---")
    param_grid = {
        "svc__C": [1e-1, 1, 10],
        "svc__gamma": [1e-1, 1, 10],
    }
    gs_model = clfHyperFitGrid(
        feat=X, lbl=y, t1=t1, pipe_clf=pipe_clf,
        param_grid=param_grid, cv=3, n_jobs=1, pctEmbargo=0.01,
    )
    preds_gs = gs_model.predict(X)
    acc_gs = accuracy_score(y, preds_gs)
    print(f"  Best grid-search model: {gs_model}")
    print(f"  In-sample accuracy    : {acc_gs:.4f}")

    # ==================================================================
    # Snippet 9.4 – logUniform distribution
    # ==================================================================
    print("\n--- Snippet 9.4: logUniform_gen ---")
    a_lu, b_lu, size_lu = 1e-3, 1e3, 10000
    vals = logUniform(a=a_lu, b=b_lu).rvs(size=size_lu)
    ks_stat = kstest(
        rvs=np.log(vals), cdf="uniform",
        args=(np.log(a_lu), np.log(b_lu / a_lu)),
    )
    print(f"  KS test on log(vals) ~ Uniform : statistic={ks_stat.statistic:.4f}, "
          f"p-value={ks_stat.pvalue:.4f}")
    print(f"  Summary of sampled values:\n{pd.Series(vals).describe().to_string()}")

    # ==================================================================
    # Snippet 9.3 – Randomized search with purged K-fold CV
    # ==================================================================
    print("\n--- Snippet 9.3: clfHyperFit (Randomized Search) ---")
    param_dist = {
        "svc__C": logUniform(a=1e-2, b=1e2),
        "svc__gamma": logUniform(a=1e-2, b=1e2),
    }
    rs_model = clfHyperFit(
        feat=X, lbl=y, t1=t1, pipe_clf=pipe_clf,
        param_grid=param_dist, cv=3, rndSearchIter=10,
        n_jobs=1, pctEmbargo=0.01,
    )
    preds_rs = rs_model.predict(X)
    acc_rs = accuracy_score(y, preds_rs)
    print(f"  Best randomized-search model: {rs_model}")
    print(f"  In-sample accuracy          : {acc_rs:.4f}")

    # ==================================================================
    # cvScore demo (Ch9 §9.4 – scoring discussion)
    # ==================================================================
    print("\n--- cvScore demo (Ch9 §9.4 scoring discussion) ---")
    svc = SVC(kernel="rbf", C=1.0, gamma=0.1, probability=True)
    scores_ll = cvScore(
        svc, X, y, sample_weight, scoring="neg_log_loss",
        t1=t1, cv=3, pctEmbargo=0.01,
    )
    scores_acc = cvScore(
        svc, X, y, sample_weight, scoring="accuracy",
        t1=t1, cv=3, pctEmbargo=0.01,
    )
    print(f"  neg_log_loss scores : {scores_ll}")
    print(f"  mean neg_log_loss   : {scores_ll.mean():.4f}")
    print(f"  accuracy scores     : {scores_acc}")
    print(f"  mean accuracy       : {scores_acc.mean():.4f}")

    print("\n" + "=" * 70)
    print("All Chapter 9 demos completed.")
    print("=" * 70)
