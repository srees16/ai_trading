"""
ch03_labeling.py – Chapter 3: Labeling
From "Advances in Financial Machine Learning" by Marcos López de Prado.

Implements Snippets 3.1–3.8:
  3.1  getDailyVol          – EWMA daily volatility estimates
  3.2  applyPtSlOnT1        – triple-barrier labeling method
  3.3  getEvents            – time of first barrier touch (side & size)
  3.4  addVerticalBarrier   – vertical barrier timestamps
  3.5  getBins              – label extraction (side & size)
  3.6  getEvents (meta)     – getEvents expanded for meta-labeling
  3.7  getBins  (meta)      – getBins  expanded for meta-labeling
  3.8  dropLabels           – drop under-populated labels

All multiprocessing calls (mpPandasObj from ch20) are replaced with
single-threaded equivalents so the file runs standalone.
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
# Snippet 3.1 – Daily Volatility Estimates
# ============================================================================

def getDailyVol(close, span0=100):
    """
    Compute daily volatility at intraday estimation points using an
    exponentially weighted moving standard deviation of returns.

    Parameters
    ----------
    close : pd.Series  – Close prices with a DatetimeIndex.
    span0 : int         – Span (in days) for the EWMA std.

    Returns
    -------
    pd.Series – Daily volatility series, indexed like *close*.
    """
    # find the index of the bar ~1 day ago for each bar
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1],                           # OCR fix: en-dash → minus
        index=close.index[close.shape[0] - df0.shape[0]:]
    )
    # daily returns
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0


# ============================================================================
# Snippet 3.2 – Triple-Barrier Labeling Method
# ============================================================================

def applyPtSlOnT1(close, events, ptSl, molecule):
    """
    Apply stop-loss / profit-taking barriers, checking whether they are
    hit before t1 (the vertical barrier / expiration).

    Parameters
    ----------
    close    : pd.Series    – Close prices.
    events   : pd.DataFrame – Must contain columns ['t1', 'trgt', 'side'].
    ptSl     : list[float]  – [profit-take factor, stop-loss factor].
                              0 disables the respective barrier.
    molecule : index-like   – Subset of event indices to process.

    Returns
    -------
    pd.DataFrame – Columns ['t1', 'sl', 'pt'] with timestamps of touches.
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index, dtype=float)  # NaNs

    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index, dtype=float)  # NaNs

    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]                                 # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()     # earliest stop-loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()     # earliest profit-take

    return out


# ============================================================================
# Snippet 3.4 – Adding a Vertical Barrier
# ============================================================================

def addVerticalBarrier(tEvents, close, numDays=1):
    """
    For each timestamp in *tEvents*, find the index of the next bar at or
    immediately after *numDays* trading days.

    Parameters
    ----------
    tEvents : pd.DatetimeIndex – Seed event timestamps.
    close   : pd.Series        – Close prices with DatetimeIndex.
    numDays : int              – Number of calendar days for the barrier.

    Returns
    -------
    pd.Series – Vertical barrier timestamps, indexed by *tEvents*.
    """
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])
    return t1


# ============================================================================
# Snippet 3.3 – Getting the Time of First Touch (side & size)
# ============================================================================

def getEvents(close, tEvents, ptSl, trgt, minRet, t1=False, side=None):
    """
    Find the time of the first barrier touch for each event.

    This is the combined Snippet 3.3 + 3.6 implementation that handles
    both standard labeling and meta-labeling (when *side* is supplied).

    Parameters
    ----------
    close      : pd.Series          – Close prices.
    tEvents    : pd.DatetimeIndex   – Seed timestamps.
    ptSl       : list[float, float] – [profit-take factor, stop-loss factor].
                                      When *side* is None the barriers are
                                      forced symmetric: [ptSl[0], ptSl[0]].
    trgt       : pd.Series          – Target (absolute returns).
    minRet     : float               – Minimum target return required.
    t1         : pd.Series | False   – Vertical barrier timestamps.
                                      False disables the vertical barrier.
    side       : pd.Series | None    – Side from an exogenous model.
                                      None ⇒ learn side & size (Snippet 3.3).
                                      Series ⇒ meta-labeling (Snippet 3.6).

    Returns
    -------
    pd.DataFrame – Columns ['t1', 'trgt'] (+ 'side' when meta-labeling).
    """
    # 1) get target
    trgt = trgt.loc[trgt.index.intersection(tEvents)]
    trgt = trgt[trgt > minRet]

    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)

    # 3) form events object, apply stop-loss on t1
    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
        ptSl_ = [ptSl[0], ptSl[0]]
    else:
        side_ = side.loc[side.index.intersection(trgt.index)]
        ptSl_ = ptSl[:2]

    events = (
        pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1)
        .dropna(subset=['trgt'])
    )

    # single-threaded replacement for mpPandasObj(applyPtSlOnT1, ...)
    df0 = applyPtSlOnT1(
        close=close,
        events=events,
        ptSl=ptSl_,
        molecule=events.index,
    )

    events['t1'] = df0.dropna(how='all').min(axis=1)  # first touch

    if side is None:
        events = events.drop('side', axis=1)

    return events


# ============================================================================
# Snippet 3.5 + 3.7 – Label Extraction (getBins) with Meta-Labeling
# ============================================================================

def getBins(events, close):
    """
    Compute each event's outcome label.

    Combines Snippet 3.5 (side & size) with Snippet 3.7 (meta-labeling).

    * If 'side' is NOT in *events*: bin ∈ {-1, 1} — label by price action.
    * If 'side' IS in *events*:     bin ∈ {0, 1}  — label by PnL (meta).

    Parameters
    ----------
    events : pd.DataFrame – Must contain 't1'; optionally 'side'.
    close  : pd.Series    – Close prices.

    Returns
    -------
    pd.DataFrame – Columns ['ret', 'bin'].
    """
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    if 'side' in events_:
        out['ret'] *= events_['side']                     # meta-labeling

    out['bin'] = np.sign(out['ret'])

    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0               # meta-labeling

    return out


# ============================================================================
# Snippet 3.8 – Dropping Under-Populated Labels
# ============================================================================

def dropLabels(events, minPct=0.05):
    """
    Recursively drop observations whose label appears in fewer than
    *minPct* of cases, unless only two classes remain.

    Parameters
    ----------
    events : pd.DataFrame – Must contain a 'bin' column.
    minPct : float        – Minimum fraction for a label to be kept.

    Returns
    -------
    pd.DataFrame – Filtered events.
    """
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print('dropped label', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events


# ============================================================================
# Helper – CUSUM filter (from Ch2, used here to generate tEvents)
# ============================================================================

def cusum_filter(close, h):
    """
    Symmetric CUSUM filter.  Returns a DatetimeIndex of timestamps where
    the cumulative deviation from the running mean exceeds threshold *h*.
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = close.diff().dropna()
    for i in diff.index:
        sPos = max(0, sPos + diff.loc[i])
        sNeg = min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


# ============================================================================
# Main – end-to-end demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 3: Labeling – Snippets 3.1 – 3.8 Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 0) Load data
    # ------------------------------------------------------------------
    print("\n[0] Loading close prices (MSFT) …")
    close = get_close_series("MSFT")
    print(f"    {len(close)} daily bars, {close.index[0].date()} → {close.index[-1].date()}")

    # ------------------------------------------------------------------
    # 1) Snippet 3.1 – Daily volatility
    # ------------------------------------------------------------------
    print("\n[1] Snippet 3.1 – getDailyVol (span=100)")
    vol = getDailyVol(close, span0=100)
    print(f"    vol shape : {vol.shape}")
    print(f"    vol tail  :\n{vol.tail()}")

    # ------------------------------------------------------------------
    # 2) Generate tEvents via CUSUM filter
    # ------------------------------------------------------------------
    print("\n[2] CUSUM filter to sample events")
    h = vol.mean()  # threshold = mean daily vol
    tEvents = cusum_filter(close, h)
    print(f"    threshold h = {h:.6f}")
    print(f"    # events    = {len(tEvents)}")

    # ------------------------------------------------------------------
    # 3) Snippet 3.4 – Vertical barriers
    # ------------------------------------------------------------------
    print("\n[3] Snippet 3.4 – addVerticalBarrier (numDays=5)")
    t1 = addVerticalBarrier(tEvents, close, numDays=5)
    print(f"    # vertical barriers = {len(t1)}")
    print(f"    sample:\n{t1.head()}")

    # ------------------------------------------------------------------
    # 4) Snippet 3.3 – getEvents (learn side & size, symmetric barriers)
    # ------------------------------------------------------------------
    print("\n[4] Snippet 3.3 – getEvents (ptSl=[1,1], learn side & size)")
    trgt = vol  # target = daily vol
    events = getEvents(
        close=close,
        tEvents=tEvents,
        ptSl=[1, 1],
        trgt=trgt,
        minRet=0.0,
        t1=t1,
        side=None,
    )
    print(f"    events shape = {events.shape}")
    print(f"    events head:\n{events.head()}")

    # ------------------------------------------------------------------
    # 5) Snippet 3.5 – getBins (side & size labels)
    # ------------------------------------------------------------------
    print("\n[5] Snippet 3.5 – getBins (side & size)")
    labels = getBins(events, close)
    print(f"    labels shape = {labels.shape}")
    print(f"    label distribution:\n{labels['bin'].value_counts().sort_index()}")
    print(f"    labels head:\n{labels.head()}")

    # ------------------------------------------------------------------
    # 6) Snippet 3.8 – dropLabels
    # ------------------------------------------------------------------
    print("\n[6] Snippet 3.8 – dropLabels (minPct=0.05)")
    labels_clean = dropLabels(labels, minPct=0.05)
    print(f"    after drop: {labels_clean.shape[0]} rows")
    print(f"    distribution:\n{labels_clean['bin'].value_counts().sort_index()}")

    # ------------------------------------------------------------------
    # 7) Snippets 3.6 / 3.7 – Meta-labeling demo
    # ------------------------------------------------------------------
    print("\n[7] Snippets 3.6/3.7 – Meta-labeling demo")
    # Simulate a primary model: simple moving-average crossover → side
    fast = close.ewm(span=10).mean()
    slow = close.ewm(span=50).mean()
    side_signal = pd.Series(np.where(fast > slow, 1.0, -1.0), index=close.index)

    # Use only events where the primary model has a signal
    side_at_events = side_signal.loc[side_signal.index.intersection(tEvents)]

    meta_events = getEvents(
        close=close,
        tEvents=tEvents,
        ptSl=[1, 2],       # asymmetric barriers allowed with meta-labeling
        trgt=trgt,
        minRet=0.0,
        t1=t1,
        side=side_at_events,
    )
    print(f"    meta events shape = {meta_events.shape}")

    meta_labels = getBins(meta_events, close)
    print(f"    meta-label distribution:\n{meta_labels['bin'].value_counts().sort_index()}")
    print(f"    meta-labels head:\n{meta_labels.head()}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("All Chapter 3 snippets executed successfully.")
    print("=" * 70)
