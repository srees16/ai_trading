"""
ch08_feature_importance.py
==========================
Chapter 8 – Feature Importance
from "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

Implements Snippets 8.1–8.10:
  8.1  (quote)   Marcos' First Law of Backtesting
  8.2  featImpMDI        – Mean Decrease Impurity
  8.3  featImpMDA        – Mean Decrease Accuracy (with sample weights)
  8.4  auxFeatImpSFI     – Single Feature Importance
  8.5  get_eVec / orthoFeats – Orthogonal features via PCA
  8.6  weighted Kendall tau between feature importance and PCA ranking
  8.7  getTestData       – Synthetic dataset (informative / redundant / noise)
  8.8  featImportance    – Master dispatcher for MDI / MDA / SFI
  8.9  testFunc          – Run all three methods on synthetic data
  8.10 plotFeatImportance – Visualisation helper
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
from itertools import product

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection._split import _BaseKFold
from scipy.stats import weightedtau

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_classification_data


# ══════════════════════════════════════════════════════════════════════════════
# Minimal PurgedKFold + cvScore (self-contained stubs from Ch 7)
# ══════════════════════════════════════════════════════════════════════════════
class PurgedKFold(_BaseKFold):
    """
    KFold variant that purges training observations whose labels overlap
    the test set, and optionally applies an embargo.
    """

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series of label end-times")
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must share the same index")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [
            (seg[0], seg[-1] + 1)
            for seg in np.array_split(indices, self.n_splits)
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


def cvScore(clf, X, y, sample_weight, scoring="neg_log_loss",
            t1=None, cv=None, cvGen=None, pctEmbargo=None):
    """Cross-validated score using PurgedKFold."""
    if scoring not in ("neg_log_loss", "accuracy"):
        raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'")
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    scores = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :], y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            sc = -log_loss(y.iloc[test], prob,
                           sample_weight=sample_weight.iloc[test].values,
                           labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            sc = accuracy_score(y.iloc[test], pred,
                                sample_weight=sample_weight.iloc[test].values)
        scores.append(sc)
    return pd.Series(scores)


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.1 – Marcos' First Law of Backtesting
# ══════════════════════════════════════════════════════════════════════════════
# "Backtesting is not a research tool.  Feature importance is."
#   -- Marcos Lopez de Prado, Advances in Financial Machine Learning (2018)


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.2 – MDI Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
def featImpMDI(fit, featNames):
    """Feature importance based on in-sample Mean Decrease Impurity."""
    df0 = {i: tree.feature_importances_
           for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)          # max_features=1 ⇒ 0 means not chosen
    imp = pd.concat(
        {"mean": df0.mean(), "std": df0.std() * df0.shape[0] ** -0.5}, axis=1
    )
    imp /= imp["mean"].sum()
    return imp


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.3 – MDA Feature Importance (with sample weights & purged CV)
# ══════════════════════════════════════════════════════════════════════════════
def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo,
               scoring="neg_log_loss"):
    """Feature importance based on OOS Mean Decrease Accuracy."""
    if scoring not in ("neg_log_loss", "accuracy"):
        raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'")
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values,
                                    labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)       # permute single column
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values,
                                            labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred,
                                                 sample_weight=w1.values)
    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / (-scr1)
    else:
        imp = imp / (1.0 - scr1)
    imp = pd.concat(
        {"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1
    )
    return imp, scr0.mean()


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.4 – Single Feature Importance (SFI)
# ══════════════════════════════════════════════════════════════════════════════
def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    """Compute OOS performance score for each feature in isolation."""
    imp = pd.DataFrame(columns=["mean", "std"])
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont["bin"],
                       sample_weight=cont["w"], scoring=scoring, cvGen=cvGen)
        imp.loc[featName, "mean"] = df0.mean()
        imp.loc[featName, "std"] = df0.std() * df0.shape[0] ** -0.5
    return imp


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.5 – Orthogonal Features via PCA
# ══════════════════════════════════════════════════════════════════════════════
def get_eVec(dot, varThres):
    """Compute eigenvectors from dot-product matrix; reduce dimension."""
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]              # sort descending
    eVal, eVec = eVal[idx], eVec[:, idx]
    eVal = pd.Series(eVal,
                     index=["PC_" + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    # reduce dimension
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[:dim + 1], eVec.iloc[:, :dim + 1]
    return eVal, eVec


def orthoFeats(dfX, varThres=0.95):
    """Compute orthogonal features via PCA from feature DataFrame *dfX*."""
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ),
                       index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return dfP


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.6 – Weighted Kendall's Tau
# ══════════════════════════════════════════════════════════════════════════════
def weightedKendallTau(featImp, pcRank):
    """
    Weighted Kendall tau between feature importances and inverse PCA rank.
    Higher values indicate stronger consistency between PCA ranking and
    feature importance ranking.
    """
    return weightedtau(featImp, pcRank ** -1.0)[0]


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.7 – Synthetic Dataset (informative / redundant / noise)
# ══════════════════════════════════════════════════════════════════════════════
def getTestData(n_features=40, n_informative=10, n_redundant=10,
                n_samples=10000):
    """Generate a random dataset for a classification problem."""
    trnsX, cont = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        random_state=0, shuffle=False,
    )
    df0 = pd.bdate_range(end=pd.Timestamp.now(), periods=n_samples)
    trnsX = pd.DataFrame(trnsX, index=df0)
    cont = pd.Series(cont, index=df0).to_frame("bin")
    # label columns: I_0..I_k, R_0..R_j, N_0..N_m
    cols = (["I_" + str(i) for i in range(n_informative)]
            + ["R_" + str(i) for i in range(n_redundant)]
            + ["N_" + str(i) for i in range(n_features - n_informative
                                             - n_redundant)])
    trnsX.columns = cols
    cont["w"] = 1.0 / cont.shape[0]
    cont["t1"] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.8 – Master dispatcher: featImportance
# ══════════════════════════════════════════════════════════════════════════════
def featImportance(trnsX, cont, n_estimators=1000, cv=10, max_samples=1.0,
                   pctEmbargo=0, scoring="accuracy", method="SFI",
                   minWLeaf=0.0, **kargs):
    """Compute feature importance using MDI, MDA, or SFI."""
    # 1) Prepare classifier – max_features=1 to prevent masking
    n_obs = trnsX.shape[0]
    _max_samples = int(n_obs * max_samples) if isinstance(max_samples, float) else max_samples
    clf = DecisionTreeClassifier(
        criterion="entropy", max_features=1,
        class_weight="balanced", min_weight_fraction_leaf=minWLeaf,
    )
    clf = BaggingClassifier(
        estimator=clf, n_estimators=n_estimators,
        max_features=1.0, max_samples=_max_samples,
        oob_score=True, n_jobs=1,
    )
    fit = clf.fit(X=trnsX, y=cont["bin"], sample_weight=cont["w"].values)
    oob = fit.oob_score_

    if method == "MDI":
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont["bin"], cv=cv,
                      sample_weight=cont["w"], t1=cont["t1"],
                      pctEmbargo=pctEmbargo, scoring=scoring).mean()
    elif method == "MDA":
        imp, oos = featImpMDA(clf, X=trnsX, y=cont["bin"], cv=cv,
                              sample_weight=cont["w"], t1=cont["t1"],
                              pctEmbargo=pctEmbargo, scoring=scoring)
    elif method == "SFI":
        cvGen = PurgedKFold(n_splits=cv, t1=cont["t1"],
                            pctEmbargo=pctEmbargo)
        oos = cvScore(clf, X=trnsX, y=cont["bin"],
                      sample_weight=cont["w"], scoring=scoring,
                      cvGen=cvGen).mean()
        imp = auxFeatImpSFI(featNames=trnsX.columns, clf=clf,
                            trnsX=trnsX, cont=cont, scoring=scoring,
                            cvGen=cvGen)
    else:
        raise ValueError(f"Unknown method: {method}")
    return imp, oob, oos


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.9 – testFunc: run all three methods on synthetic data
# ══════════════════════════════════════════════════════════════════════════════
def testFunc(n_features=40, n_informative=10, n_redundant=10,
             n_estimators=1000, n_samples=10000, cv=10):
    """Test feature importance methods on synthetic data."""
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    dict0 = {
        "minWLeaf": [0.0],
        "scoring": ["accuracy"],
        "method": ["MDI", "MDA", "SFI"],
        "max_samples": [1.0],
    }
    jobs = [dict(zip(dict0, i)) for i in product(*dict0.values())]
    out = []
    kargs = {"n_estimators": n_estimators, "cv": cv}
    for job in jobs:
        job["simNum"] = (job["method"] + "_" + job["scoring"] + "_"
                         + "%.2f" % job["minWLeaf"] + "_"
                         + str(job["max_samples"]))
        print("  Running:", job["simNum"])
        kargs.update(job)
        imp, oob, oos = featImportance(trnsX=trnsX, cont=cont, **kargs)
        plotFeatImportance(imp=imp, oob=oob, oos=oos, method=job["method"],
                           tag="testFunc", simNum=job["simNum"])
        df0 = imp[["mean"]] / imp["mean"].abs().sum()
        df0["type"] = [i[0] for i in df0.index]
        df0 = df0.groupby("type")["mean"].sum().to_dict()
        df0.update({"oob": oob, "oos": oos})
        df0.update(job)
        out.append(df0)
    out = pd.DataFrame(out).sort_values(
        ["method", "scoring", "minWLeaf", "max_samples"]
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8.10 – Feature importance plotting
# ══════════════════════════════════════════════════════════════════════════════
def plotFeatImportance(imp, oob, oos, method, tag="", simNum="",
                       savePath=None, **kargs):
    """Plot horizontal bar chart of mean feature importance with std."""
    fig, ax = plt.subplots(figsize=(10, max(imp.shape[0] / 5.0, 4)))
    imp = imp.sort_values("mean", ascending=True)
    imp["mean"].plot(kind="barh", color="b", alpha=0.25, xerr=imp["std"],
                     error_kw={"ecolor": "r"}, ax=ax)
    if method == "MDI":
        ax.set_xlim([0, imp.sum(axis=1).max()])
        ax.axvline(1.0 / imp.shape[0], linewidth=1, color="r",
                   linestyle="dotted")
    ax.get_yaxis().set_visible(False)
    for patch, label in zip(ax.patches, imp.index):
        ax.text(patch.get_width() / 2,
                patch.get_y() + patch.get_height() / 2,
                label, ha="center", va="center", color="black")
    ax.set_title(f"tag={tag} | simNum={simNum} | "
                 f"oob={round(oob, 4)} | oos={round(oos, 4)}")
    plt.tight_layout()
    if savePath:
        fig.savefig(savePath, dpi=100)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN – demonstrate key snippets
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 8 - Feature Importance")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) Generate synthetic data (Snippet 8.7) – small for speed
    # ------------------------------------------------------------------
    N_FEATURES = 20
    N_INFORMATIVE = 5
    N_REDUNDANT = 5
    N_SAMPLES = 1000
    N_ESTIMATORS = 200
    CV = 5

    print(f"\n[1] getTestData: {N_SAMPLES} samples, {N_FEATURES} features "
          f"({N_INFORMATIVE} informative, {N_REDUNDANT} redundant, "
          f"{N_FEATURES - N_INFORMATIVE - N_REDUNDANT} noise)")
    trnsX, cont = getTestData(N_FEATURES, N_INFORMATIVE, N_REDUNDANT, N_SAMPLES)
    print(f"    trnsX shape: {trnsX.shape}")
    print(f"    cont columns: {list(cont.columns)}")

    # ------------------------------------------------------------------
    # 2) Build a bagged decision-tree classifier (shared across methods)
    # ------------------------------------------------------------------
    clf = DecisionTreeClassifier(
        criterion="entropy", max_features=1,
        class_weight="balanced", min_weight_fraction_leaf=0.0,
    )
    clf = BaggingClassifier(
        estimator=clf, n_estimators=N_ESTIMATORS,
        max_features=1.0, max_samples=N_SAMPLES, oob_score=True, n_jobs=1,
    )

    # ------------------------------------------------------------------
    # 3) Snippet 8.2 – MDI Feature Importance
    # ------------------------------------------------------------------
    print("\n[2] MDI Feature Importance (Snippet 8.2) ...")
    fit = clf.fit(X=trnsX, y=cont["bin"], sample_weight=cont["w"].values)
    impMDI = featImpMDI(fit, featNames=trnsX.columns)
    print("    Top-5 features by MDI:")
    top5_mdi = impMDI.sort_values("mean", ascending=False).head(5)
    for feat, row in top5_mdi.iterrows():
        print(f"      {feat:6s}  mean={row['mean']:.4f}  std={row['std']:.4f}")

    # ------------------------------------------------------------------
    # 4) Snippet 8.3 – MDA Feature Importance
    # ------------------------------------------------------------------
    print(f"\n[3] MDA Feature Importance (Snippet 8.3, cv={CV}) ...")
    impMDA, oos_mda = featImpMDA(
        clf, X=trnsX, y=cont["bin"], cv=CV,
        sample_weight=cont["w"], t1=cont["t1"],
        pctEmbargo=0.0, scoring="accuracy",
    )
    print(f"    OOS score (accuracy): {oos_mda:.4f}")
    print("    Top-5 features by MDA:")
    top5_mda = impMDA.sort_values("mean", ascending=False).head(5)
    for feat, row in top5_mda.iterrows():
        print(f"      {feat:6s}  mean={row['mean']:.4f}  std={row['std']:.4f}")

    # ------------------------------------------------------------------
    # 5) Snippet 8.4 – SFI Feature Importance
    # ------------------------------------------------------------------
    print(f"\n[4] SFI Feature Importance (Snippet 8.4, cv={CV}) ...")
    cvGen = PurgedKFold(n_splits=CV, t1=cont["t1"], pctEmbargo=0.0)
    impSFI = auxFeatImpSFI(
        featNames=trnsX.columns, clf=clf, trnsX=trnsX,
        cont=cont, scoring="accuracy", cvGen=cvGen,
    )
    print("    Top-5 features by SFI:")
    top5_sfi = impSFI.sort_values("mean", ascending=False).head(5)
    for feat, row in top5_sfi.iterrows():
        print(f"      {feat:6s}  mean={row['mean']:.4f}  std={row['std']:.4f}")

    # ------------------------------------------------------------------
    # 6) Snippet 8.5 – Orthogonal features
    # ------------------------------------------------------------------
    print("\n[5] Orthogonal Features via PCA (Snippet 8.5) ...")
    dfP = orthoFeats(trnsX, varThres=0.95)
    print(f"    Original features: {trnsX.shape[1]}, "
          f"PCA features: {dfP.shape[1]}")

    # ------------------------------------------------------------------
    # 7) Snippet 8.6 – Weighted Kendall Tau demo
    # ------------------------------------------------------------------
    print("\n[6] Weighted Kendall Tau (Snippet 8.6) ...")
    featImpArr = np.array([0.55, 0.33, 0.07, 0.05])
    pcRankArr = np.array([1, 2, 4, 3])
    wkt = weightedKendallTau(featImpArr, pcRankArr)
    print(f"    featImp = {featImpArr}")
    print(f"    pcRank  = {pcRankArr}")
    print(f"    weighted Kendall tau = {wkt:.4f}")

    # ------------------------------------------------------------------
    # 8) Snippet 8.8 – featImportance master function (MDI)
    # ------------------------------------------------------------------
    print(f"\n[7] featImportance master function (Snippet 8.8, method='MDI') ...")
    imp, oob, oos = featImportance(
        trnsX=trnsX, cont=cont, n_estimators=N_ESTIMATORS,
        cv=CV, method="MDI", scoring="accuracy",
    )
    print(f"    OOB={oob:.4f}, OOS={oos:.4f}")
    print("    Top-3 by MDI:")
    for feat, row in imp.sort_values("mean", ascending=False).head(3).iterrows():
        print(f"      {feat:6s}  mean={row['mean']:.4f}")

    # ------------------------------------------------------------------
    # 9) Snippet 8.10 – Plot (saved to file)
    # ------------------------------------------------------------------
    OUTPUT_DIR = __import__('pathlib').Path(__file__).resolve().parent.parent / "_output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = str(OUTPUT_DIR / "ch08_mdi_plot.png")
    print(f"\n[8] plotFeatImportance (Snippet 8.10) -> _output/ch08_mdi_plot.png")
    plotFeatImportance(imp=imp, oob=oob, oos=oos, method="MDI",
                       tag="ch08_demo", simNum="MDI_accuracy",
                       savePath=save_path)
    print("    Plot saved.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary: feature types vs. normalised MDI importance")
    print("-" * 70)
    df_summary = imp[["mean"]].copy()
    df_summary["mean"] = df_summary["mean"] / df_summary["mean"].abs().sum()
    df_summary["type"] = [idx[0] for idx in df_summary.index]
    type_totals = df_summary.groupby("type")["mean"].sum()
    for t, v in type_totals.items():
        label = {"I": "Informative", "R": "Redundant", "N": "Noise"}[t]
        print(f"  {label:12s} ({t}): {v:.4f}")
    print(f"  OOB accuracy:  {oob:.4f}")
    print(f"  OOS accuracy:  {oos:.4f}")
    print("=" * 70)
    print("Chapter 8 demo complete.")
