"""
Chapter 7: Cross-Validation in Finance
From "Advances in Financial Machine Learning" by Marcos Lopez de Prado

Snippets:
  7.1 – getTrainTimes: purge training observations that overlap with test labels
  7.2 – getEmbargoTimes: compute embargo timestamps after each test bar
  7.3 – PurgedKFold: sklearn KFold subclass with purging + embargo
  7.4 – cvScore: cross-validation scoring using PurgedKFold
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_classification_data, get_close_series


# ──────────────────────────────────────────────────────────────────────────────
# SNIPPET 7.1 – Purging observations in the training set
# ──────────────────────────────────────────────────────────────────────────────
def getTrainTimes(t1, testTimes):
    """
    Given testTimes, find the times of the training observations.
      - t1.index : time when the observation started.
      - t1.value : time when the observation ended.
      - testTimes: pd.Series whose index=test-start, value=test-end.
    """
    trn = t1.copy(deep=True)
    for i, j in testTimes.items():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  # train starts within test
        df1 = trn[(i <= trn) & (trn <= j)].index               # train ends within test
        df2 = trn[(trn.index <= i) & (j <= trn)].index          # train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


# ──────────────────────────────────────────────────────────────────────────────
# SNIPPET 7.2 – Embargo on training observations
# ──────────────────────────────────────────────────────────────────────────────
def getEmbargoTimes(times, pctEmbargo):
    """
    Get embargo time for each bar.
      - times     : pd.DatetimeIndex of bar timestamps.
      - pctEmbargo: fraction of total bars to embargo (e.g. 0.01 = 1 %).
    Returns a Series mapping each bar time to its embargo boundary.
    """
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = pd.concat([mbrg, pd.Series(times[-1], index=times[-step:])])
    return mbrg


# ──────────────────────────────────────────────────────────────────────────────
# SNIPPET 7.3 – PurgedKFold cross-validation class
# ──────────────────────────────────────────────────────────────────────────────
class PurgedKFold(_BaseKFold):
    """
    Extend KFold to work with labels that span intervals.
    The training set is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), without training examples
    in between.
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
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(
                self.t1.iloc[test_indices].max()
            )
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index
            )
            if maxT1Idx + mbrg < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx + mbrg :])
                )
            yield train_indices, test_indices


# ──────────────────────────────────────────────────────────────────────────────
# SNIPPET 7.4 – Cross-validation score with PurgedKFold
# ──────────────────────────────────────────────────────────────────────────────
def cvScore(clf, X, y, sample_weight, scoring="neg_log_loss",
            t1=None, cv=None, cvGen=None, pctEmbargo=None):
    """
    Compute cross-validated scores using PurgedKFold.
      - clf          : sklearn classifier (must support predict_proba for log_loss).
      - X, y         : features and labels with matching DatetimeIndex.
      - sample_weight: per-observation weights (pd.Series, same index as y).
      - scoring      : 'neg_log_loss' or 'accuracy'.
      - t1           : pd.Series of label end-times (index=start, value=end).
      - cv           : number of folds (used if cvGen is None).
      - cvGen        : pre-built PurgedKFold generator (overrides cv/t1/pctEmbargo).
      - pctEmbargo   : embargo fraction (used if cvGen is None).
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'.")
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(
                y.iloc[test], prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_,
            )
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(
                y.iloc[test], pred,
                sample_weight=sample_weight.iloc[test].values,
            )
        score.append(score_)
    return np.array(score)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: create synthetic t1 (label end-times) from a DatetimeIndex
# ──────────────────────────────────────────────────────────────────────────────
def _make_t1(index, horizon=5):
    """
    For each bar in *index*, assign an end-time *horizon* bars ahead.
    Returns pd.Series(index=bar_time, value=end_time).
    """
    t1 = pd.Series(index=index, dtype="datetime64[ns]")
    for loc in range(len(index)):
        end_loc = min(loc + horizon, len(index) - 1)
        t1.iloc[loc] = index[end_loc]
    return t1


# ======================================================================
#  MAIN – demonstrate each snippet
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 7 – Cross-Validation in Finance")
    print("=" * 70)

    # ------- generate data ------------------------------------------------
    X, y = generate_classification_data(n_samples=1000, n_features=10,
                                        n_informative=3, seed=42)
    dates = X.index
    t1 = _make_t1(dates, horizon=5)
    sample_weight = pd.Series(np.ones(len(y)), index=y.index)

    # =====================================================================
    # Snippet 7.1 – getTrainTimes (purging)
    # =====================================================================
    print("\n--- Snippet 7.1: getTrainTimes (purging) ---")
    # Define a synthetic test window: middle 10 % of the data
    mid = len(dates) // 2
    span = len(dates) // 10
    test_start = dates[mid]
    test_end = dates[mid + span]
    testTimes = pd.Series([test_end], index=[test_start])

    trn = getTrainTimes(t1, testTimes)
    print(f"  Total observations : {len(t1)}")
    print(f"  Test window        : {test_start.date()} → {test_end.date()}")
    print(f"  Training after purge: {len(trn)}")
    purged = len(t1) - len(trn) - span
    print(f"  Purged (overlap)   : {purged}")

    # =====================================================================
    # Snippet 7.2 – getEmbargoTimes
    # =====================================================================
    print("\n--- Snippet 7.2: getEmbargoTimes ---")
    mbrg = getEmbargoTimes(dates, pctEmbargo=0.01)
    print(f"  Number of bars  : {len(dates)}")
    print(f"  Embargo (1 %)   : {int(len(dates) * 0.01)} bars")
    print(f"  Embargo series  : {len(mbrg)} entries")
    print(f"  First bar embargo: {mbrg.iloc[0].date()}")
    print(f"  Last bar embargo : {mbrg.iloc[-1].date()}")

    # =====================================================================
    # Snippet 7.3 – PurgedKFold
    # =====================================================================
    print("\n--- Snippet 7.3: PurgedKFold ---")
    pkf = PurgedKFold(n_splits=5, t1=t1, pctEmbargo=0.01)
    for fold, (train_idx, test_idx) in enumerate(pkf.split(X)):
        print(f"  Fold {fold}: train={len(train_idx)}, test={len(test_idx)}, "
              f"test range=[{dates[test_idx[0]].date()} – "
              f"{dates[test_idx[-1]].date()}]")

    # =====================================================================
    # Snippet 7.4 – cvScore
    # =====================================================================
    print("\n--- Snippet 7.4: cvScore ---")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
    )

    scores_ll = cvScore(
        clf, X, y, sample_weight,
        scoring="neg_log_loss", t1=t1, cv=5, pctEmbargo=0.01,
    )
    print(f"  neg_log_loss per fold : {np.round(scores_ll, 4)}")
    print(f"  mean neg_log_loss     : {scores_ll.mean():.4f}")

    scores_acc = cvScore(
        clf, X, y, sample_weight,
        scoring="accuracy", t1=t1, cv=5, pctEmbargo=0.01,
    )
    print(f"  accuracy per fold     : {np.round(scores_acc, 4)}")
    print(f"  mean accuracy         : {scores_acc.mean():.4f}")

    print("\n" + "=" * 70)
    print("All Chapter 7 snippets executed successfully.")
    print("=" * 70)
