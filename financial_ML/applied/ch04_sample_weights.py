"""
ch04_sample_weights.py – Chapter 4: Sample Weights
From "Advances in Financial Machine Learning" by Marcos López de Prado.

Implements Snippets 4.1–4.11:
  4.1   mpNumCoEvents   – number of concurrent events per bar
  4.2   mpSampleTW      – sample weight by average uniqueness
  4.3   getIndMatrix    – build indicator matrix from bar index & t1
  4.4   getAvgUniqueness– average uniqueness from indicator matrix
  4.5   seqBootstrap    – sequential bootstrap sampling
  4.6   main            – numerical example for sequential bootstrap
  4.7   getRndT1        – generate random t1 series
  4.8   auxMC           – Monte Carlo helper comparing bootstraps
  4.9   (mainMC)        – Monte Carlo experiment (single-threaded)
  4.10  mpSampleW       – sample weight by absolute return attribution
  4.11  getTimeDecay    – piecewise-linear time-decay factors

All multiprocessing calls (mpPandasObj) are replaced with single-threaded
equivalents so the file runs standalone.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Allow imports from the parent directory (financial_ML/)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import get_close_series, generate_ohlcv_bars


# ============================================================================
# Snippet 4.1 – Estimating the Number of Concurrent Events (Labels)
# ============================================================================

def mpNumCoEvents(closeIdx, t1, molecule):
    """
    Compute the number of concurrent events per bar.

    Parameters
    ----------
    closeIdx : pd.DatetimeIndex – Index of the close-price series.
    t1       : pd.Series        – Maps event start → event end timestamp.
    molecule : index-like       – Subset of event indices to process.

    Returns
    -------
    pd.Series – Count of concurrent events at each bar in the range.
    """
    # 1) find events that span the period [molecule[0], molecule[-1]]
    t1 = t1.fillna(closeIdx[-1])  # unclosed events still impact other weights
    t1 = t1[t1 >= molecule[0]]    # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()]  # events that start at or before t1[molecule].max()

    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])
    for tIn, tOut in t1.items():
        count.loc[tIn:tOut] += 1.0
    return count.loc[molecule[0]:t1[molecule].max()]


# ============================================================================
# Snippet 4.2 – Estimating the Average Uniqueness of a Label
# ============================================================================

def mpSampleTW(t1, numCoEvents, molecule):
    """
    Derive average uniqueness over each event's lifespan.

    Parameters
    ----------
    t1          : pd.Series – Maps event start → event end timestamp.
    numCoEvents : pd.Series – Number of concurrent events at each bar.
    molecule    : index-like– Subset of event indices to process.

    Returns
    -------
    pd.Series – Average uniqueness weight for each event in *molecule*.
    """
    wght = pd.Series(index=molecule, dtype=float)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (1.0 / numCoEvents.loc[tIn:tOut]).mean()
    return wght


# ============================================================================
# Snippet 4.3 – Build an Indicator Matrix
# ============================================================================

def getIndMatrix(barIx, t1):
    """
    Build a binary indicator matrix: rows = bars, columns = events.
    Entry (t, i) = 1 if bar t falls within event i's lifespan.

    Parameters
    ----------
    barIx : range / index – Index of price bars.
    t1    : pd.Series     – Maps event start (index) → event end (value).

    Returns
    -------
    pd.DataFrame – Binary indicator matrix.
    """
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1_) in enumerate(t1.items()):
        indM.loc[t0:t1_, i] = 1.0
    return indM


# ============================================================================
# Snippet 4.4 – Compute Average Uniqueness
# ============================================================================

def getAvgUniqueness(indM):
    """
    Average uniqueness from an indicator matrix.

    Parameters
    ----------
    indM : pd.DataFrame – Indicator matrix (from getIndMatrix or subset).

    Returns
    -------
    pd.Series – Average uniqueness for each column (event).
    """
    c = indM.sum(axis=1)  # concurrency per bar
    u = indM.div(c, axis=0)  # uniqueness per (bar, event)
    avgU = u[u > 0].mean()  # average uniqueness per event
    return avgU


# ============================================================================
# Snippet 4.5 – Sequential Bootstrap
# ============================================================================

def seqBootstrap(indM, sLength=None):
    """
    Generate a sample via sequential bootstrap.

    At each draw the probability of picking event *i* is proportional
    to its average uniqueness given the events already drawn.

    Parameters
    ----------
    indM    : pd.DataFrame – Indicator matrix.
    sLength : int | None   – Number of draws (default = number of events).

    Returns
    -------
    list – Indices of drawn events (may contain repeats).
    """
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series(dtype=float)
        for i in indM.columns:
            indM_ = indM[phi + [i]]  # indicator matrix for existing draws + candidate
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()  # draw probabilities
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


# ============================================================================
# Snippet 4.6 – Numerical Example of Sequential Bootstrap
# ============================================================================

def exampleSequentialBootstrap():
    """Run sequential bootstrap on the textbook example (Section 4.5.3)."""
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4])  # t0, t1 for each feature obs
    barIx = range(t1.max() + 1)  # index of bars
    indM = getIndMatrix(barIx, t1)

    phi = np.random.choice(indM.columns, size=indM.shape[1])
    print("Standard bootstrap sample:", phi)
    print("Standard uniqueness:", getAvgUniqueness(indM[phi]).mean())

    phi = seqBootstrap(indM)
    print("Sequential bootstrap sample:", phi)
    print("Sequential uniqueness:", getAvgUniqueness(indM[phi]).mean())


# ============================================================================
# Snippet 4.7 – Generating a Random t1 Series
# ============================================================================

def getRndT1(numObs, numBars, maxH):
    """
    Generate a random t1 Series for Monte Carlo experiments.

    Parameters
    ----------
    numObs  : int – Number of observations (I).
    numBars : int – Number of bars (T).
    maxH    : int – Maximum horizon for an observation.

    Returns
    -------
    pd.Series – Random t1 series (start → end).
    """
    t1 = pd.Series(dtype=int)
    for i in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()


# ============================================================================
# Snippet 4.8 – Uniqueness from Standard and Sequential Bootstraps
# ============================================================================

def auxMC(numObs, numBars, maxH):
    """
    Single Monte Carlo iteration: compare average uniqueness of a
    standard bootstrap vs. sequential bootstrap.

    Returns
    -------
    dict – {'stdU': float, 'seqU': float}
    """
    t1 = getRndT1(numObs, numBars, maxH)
    barIx = range(t1.max() + 1)
    indM = getIndMatrix(barIx, t1)
    # Standard bootstrap
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    # Sequential bootstrap
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return {'stdU': stdU, 'seqU': seqU}


# ============================================================================
# Snippet 4.9 – Monte Carlo Experiment (single-threaded)
# ============================================================================

def mainMC(numObs=10, numBars=100, maxH=5, numIters=100):
    """
    Run a Monte Carlo comparing standard vs. sequential bootstrap.

    Parameters
    ----------
    numIters : int – Number of iterations (kept small for quick demo).
    """
    out = []
    for i in range(int(numIters)):
        out.append(auxMC(numObs, numBars, maxH))
    print(pd.DataFrame(out).describe())
    return pd.DataFrame(out)


# ============================================================================
# Snippet 4.10 – Sample Weight by Absolute Return Attribution
# ============================================================================

def mpSampleW(t1, numCoEvents, close, molecule):
    """
    Derive sample weight by return attribution.

    Parameters
    ----------
    t1          : pd.Series – Maps event start → event end.
    numCoEvents : pd.Series – Concurrent-event counts per bar.
    close       : pd.Series – Close prices.
    molecule    : index-like– Subset of event indices to process.

    Returns
    -------
    pd.Series – Absolute-return attributed weights.
    """
    ret = np.log(close).diff()  # log-returns (additive)
    wght = pd.Series(index=molecule, dtype=float)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


# ============================================================================
# Snippet 4.11 – Time-Decay Factors
# ============================================================================

def getTimeDecay(tW, clfLastW=1.0):
    """
    Apply piecewise-linear decay to observed uniqueness (tW).

    Newest observation gets weight = 1, oldest gets weight = clfLastW.

    Parameters
    ----------
    tW       : pd.Series – Average-uniqueness weights (from mpSampleTW).
    clfLastW : float     – Decay parameter c.
                           c = 1   → no decay.
                           0 < c < 1 → linear decay, all positive.
                           c = 0   → oldest weight converges to zero.
                           -1 < c < 0 → oldest fraction gets zero weight.

    Returns
    -------
    pd.Series – Decayed cumulative weights.
    """
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    print(f"  getTimeDecay: const={const:.4f}, slope={slope:.6f}")
    return clfW


# ============================================================================
# Helper: compute numCoEvents and tW for a given close + events['t1']
# ============================================================================

def computeSampleWeights(close, t1):
    """
    Convenience wrapper that chains mpNumCoEvents → mpSampleTW → mpSampleW.

    Parameters
    ----------
    close : pd.Series – Close prices.
    t1    : pd.Series – Maps event start → event end timestamp.

    Returns
    -------
    dict with keys: 'numCoEvents', 'tW' (uniqueness weights),
                    'w' (return-attributed weights).
    """
    out = {}

    # Number of concurrent events (single-threaded replacement for mpPandasObj)
    numCoEvents = mpNumCoEvents(close.index, t1, molecule=t1.index)
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['numCoEvents'] = numCoEvents

    # Average uniqueness weights
    tW = mpSampleTW(t1, numCoEvents, molecule=t1.index)
    out['tW'] = tW

    # Return-attributed weights
    w = mpSampleW(t1, numCoEvents, close, molecule=t1.index)
    w *= w.shape[0] / w.sum()  # scale so weights sum to I
    out['w'] = w

    return out


# ============================================================================
# Minimal stubs from Chapter 3 (getDailyVol, getEvents) to make demo
# self-contained without importing all of ch03.
# ============================================================================

def _getDailyVol(close, span0=100):
    """EWMA daily volatility (Snippet 3.1 stub)."""
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1],
        index=close.index[close.shape[0] - df0.shape[0]:]
    )
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0


def _getVerticalBarrier(tEvents, close, numDays=5):
    """Vertical barrier (Snippet 3.4 stub)."""
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])
    return t1


def _makeT1(close, span0=100, numDays=5, ptSl=1.0, minRet=0.0):
    """
    Produce a synthetic t1 Series for demonstration.

    Uses CUSUM filter–style seed events (every ~10 bars) and the vertical
    barrier as the event end.
    """
    vol = _getDailyVol(close, span0=span0)
    vol = vol.dropna()

    # Simple CUSUM-style seeding: sample every ~10 bars
    step = max(1, len(vol) // 200)
    tEvents = vol.index[::step]

    # vertical barrier as event end
    t1 = _getVerticalBarrier(tEvents, close, numDays=numDays)

    # Only keep events where we have both vol and t1
    common = t1.index.intersection(vol.index)
    t1 = t1.loc[common]
    return t1


# ============================================================================
# Main demo
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 65)
    print("Chapter 4 – Sample Weights  (AFML)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    print("\n[1] Generating synthetic close-price series …")
    bars = generate_ohlcv_bars(n_bars=500, seed=42)
    close = bars["Close"]
    print(f"    Bars: {len(close)},  range: {close.index[0].date()} → {close.index[-1].date()}")

    # ------------------------------------------------------------------
    # 2. Create synthetic t1 (event start → event end)
    # ------------------------------------------------------------------
    print("\n[2] Building synthetic event labels (t1) …")
    t1 = _makeT1(close, span0=50, numDays=5, minRet=0.0)
    print(f"    Number of events: {len(t1)}")
    print(f"    Sample t1:\n{t1.head()}")

    # ------------------------------------------------------------------
    # 3. Number of concurrent events (Snippet 4.1)
    # ------------------------------------------------------------------
    print("\n[3] Snippet 4.1 – mpNumCoEvents …")
    numCoEvents = mpNumCoEvents(close.index, t1, molecule=t1.index)
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    print(f"    numCoEvents stats:\n{numCoEvents.describe()}")

    # ------------------------------------------------------------------
    # 4. Average uniqueness weights (Snippet 4.2)
    # ------------------------------------------------------------------
    print("\n[4] Snippet 4.2 – mpSampleTW (average uniqueness) …")
    tW = mpSampleTW(t1, numCoEvents, molecule=t1.index)
    print(f"    Mean uniqueness:  {tW.mean():.4f}")
    print(f"    Sample tW:\n{tW.head()}")

    # ------------------------------------------------------------------
    # 5. Indicator matrix & average uniqueness (Snippets 4.3–4.4)
    # ------------------------------------------------------------------
    print("\n[5] Snippets 4.3–4.4 – Indicator matrix & average uniqueness …")
    # Use textbook toy example
    t1_toy = pd.Series([2, 3, 5], index=[0, 2, 4])
    barIx_toy = range(t1_toy.max() + 1)
    indM = getIndMatrix(barIx_toy, t1_toy)
    print("    Indicator matrix:")
    print(indM.to_string())
    avgU = getAvgUniqueness(indM)
    print(f"    Average uniqueness per event: {avgU.values}")

    # ------------------------------------------------------------------
    # 6. Sequential bootstrap (Snippets 4.5–4.6)
    # ------------------------------------------------------------------
    print("\n[6] Snippets 4.5–4.6 – Sequential bootstrap example …")
    exampleSequentialBootstrap()

    # ------------------------------------------------------------------
    # 7. Monte Carlo comparison (Snippets 4.7–4.9)
    # ------------------------------------------------------------------
    print("\n[7] Snippets 4.7–4.9 – Monte Carlo (20 iterations for speed) …")
    mcResults = mainMC(numObs=10, numBars=50, maxH=5, numIters=20)

    # ------------------------------------------------------------------
    # 8. Return-attributed weights (Snippet 4.10)
    # ------------------------------------------------------------------
    print("\n[8] Snippet 4.10 – mpSampleW (return-attributed weights) …")
    w = mpSampleW(t1, numCoEvents, close, molecule=t1.index)
    w *= w.shape[0] / w.sum()
    print(f"    Mean weight:  {w.mean():.4f}")
    print(f"    Sample w:\n{w.head()}")

    # ------------------------------------------------------------------
    # 9. Time-decay factors (Snippet 4.11)
    # ------------------------------------------------------------------
    print("\n[9] Snippet 4.11 – getTimeDecay …")
    for c in [1.0, 0.5, 0.0, -0.25]:
        print(f"  c = {c:6.2f} →", end=" ")
        decayed = getTimeDecay(tW, clfLastW=c)
        print(f"  min={decayed.min():.4f}, max={decayed.max():.4f}, mean={decayed.mean():.4f}")

    # ------------------------------------------------------------------
    # 10. Full pipeline via computeSampleWeights helper
    # ------------------------------------------------------------------
    print("\n[10] Full pipeline – computeSampleWeights …")
    sw = computeSampleWeights(close, t1)
    print(f"    numCoEvents mean: {sw['numCoEvents'].mean():.4f}")
    print(f"    tW mean:          {sw['tW'].mean():.4f}")
    print(f"    w mean:           {sw['w'].mean():.4f}")

    print("\n" + "=" * 65)
    print("Chapter 4 demo complete.")
    print("=" * 65)
