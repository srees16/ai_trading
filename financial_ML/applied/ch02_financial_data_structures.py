"""
Chapter 2: Financial Data Structures
=====================================
Advances in Financial Machine Learning — Marcos López de Prado

Snippets 2.1–2.4 implemented with OCR fixes and synthetic-data demos.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_data import (
    get_close_series,
    get_multi_close,
    generate_tick_data,
    generate_ohlcv_bars,
)

# ---------------------------------------------------------------------------
# Snippet 2.1 – PCA Weights from a Risk Distribution R
# ---------------------------------------------------------------------------

def pcaWeights(cov, riskDist=None, riskTarget=1.):
    """Following the riskAlloc distribution, match riskTarget.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix (must be symmetric / Hermitian).
    riskDist : array-like or None
        Desired risk distribution across principal components.
        If None, all risk is allocated to the component with
        the smallest eigenvalue.
    riskTarget : float
        Target portfolio variance.

    Returns
    -------
    wghts : np.ndarray
        Column vector of portfolio weights.
    """
    eVal, eVec = np.linalg.eigh(cov)  # must be Hermitian
    indices = eVal.argsort()[::-1]     # sort eigenvalues descending
    eVal, eVec = eVal[indices], eVec[:, indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1.              # all risk on smallest eigenvalue
    loads = riskTarget * (riskDist / eVal) ** 0.5
    wghts = np.dot(eVec, np.reshape(loads, (-1, 1)))
    # ctr = (loads / riskTarget)**2 * eVal  # verify riskDist
    return wghts


# ---------------------------------------------------------------------------
# Snippet 2.2 – Form a Gaps Series, Detract It from Prices
# ---------------------------------------------------------------------------

def rollGaps(series, dictio=None, matchEnd=True):
    """Compute cumulative roll gaps between contracts.

    Parameters
    ----------
    series : pd.DataFrame
        Bar data with columns for instrument identifier, open, and close.
    dictio : dict
        Mapping of logical names to column names.  Defaults to
        {'Instrument': 'Symbol', 'Open': 'Open', 'Close': 'Close'}.
    matchEnd : bool
        If True the rolled series matches the raw price at the *end*
        (backward roll).  If False, it matches at the *start* (forward roll).

    Returns
    -------
    gaps : pd.Series
        Cumulative gap series aligned to *series.index*.
    """
    if dictio is None:
        dictio = {'Instrument': 'Symbol', 'Open': 'Open', 'Close': 'Close'}

    # Compute gaps at each roll, between previous close and next open
    rollDates = series[dictio['Instrument']].drop_duplicates(keep='first').index
    gaps = series[dictio['Close']] * 0
    iloc = list(series.index)
    iloc = [iloc.index(i) - 1 for i in rollDates]   # index of days prior to roll
    gaps.loc[rollDates[1:]] = (
        series[dictio['Open']].loc[rollDates[1:]]
        - series[dictio['Close']].iloc[iloc[1:]].values
    )
    gaps = gaps.cumsum()
    if matchEnd:
        gaps -= gaps.iloc[-1]  # roll backward
    return gaps


def getRolledSeries(series, dictio=None):
    """Apply roll-gap adjustment to a futures bar DataFrame.

    Parameters
    ----------
    series : pd.DataFrame
        Must contain columns mapped by *dictio* plus 'Close' and
        (optionally) 'VWAP'.
    dictio : dict or None
        Column-name mapping (see ``rollGaps``).

    Returns
    -------
    series : pd.DataFrame
        The input DataFrame with 'Close' (and 'VWAP' if present)
        adjusted for roll gaps.
    """
    if dictio is None:
        dictio = {'Instrument': 'Symbol', 'Open': 'Open', 'Close': 'Close'}

    gaps = rollGaps(series, dictio=dictio)
    for fld in ['Close', 'VWAP']:
        if fld in series.columns:
            series[fld] -= gaps
    return series


# ---------------------------------------------------------------------------
# Snippet 2.3 – Non-Negative Rolled Price Series
# ---------------------------------------------------------------------------

def getNonNegativeRolledSeries(raw, dictio=None):
    """Return a $1-investment price series from a rolled futures DataFrame.

    Steps:
      1. Compute cumulative roll gaps.
      2. Subtract gaps from Open and Close to get rolled prices.
      3. Compute returns as rolled-price change / previous raw close.
      4. Form a price series via ``(1 + r).cumprod()``.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw futures bar data with at least 'Open', 'Close', and an
        instrument identifier column.
    dictio : dict or None
        Column-name mapping (see ``rollGaps``).

    Returns
    -------
    rolled : pd.DataFrame
        Copy of *raw* with added 'Returns' and 'rPrices' columns,
        and gap-adjusted 'Open' / 'Close'.
    """
    if dictio is None:
        dictio = {'Instrument': 'Symbol', 'Open': 'Open', 'Close': 'Close'}

    gaps = rollGaps(raw, dictio=dictio)
    rolled = raw.copy(deep=True)
    for fld in ['Open', 'Close']:
        rolled[fld] -= gaps
    rolled['Returns'] = rolled['Close'].diff() / raw['Close'].shift(1)
    rolled['rPrices'] = (1 + rolled['Returns']).cumprod()
    return rolled


# ---------------------------------------------------------------------------
# Snippet 2.4 – The Symmetric CUSUM Filter
# ---------------------------------------------------------------------------

def getTEvents(gRaw, h):
    """Symmetric CUSUM filter.

    Samples an event whenever the cumulative positive or negative
    deviation from the previous value exceeds threshold *h*.

    Parameters
    ----------
    gRaw : pd.Series
        Raw time series (e.g. prices or log-prices).
    h : float
        Filter threshold.

    Returns
    -------
    tEvents : pd.DatetimeIndex
        Timestamps where CUSUM events are detected.
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


# ===================================================================
# Main demo
# ===================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 65)
    print("Chapter 2 – Financial Data Structures  (demo)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Demo 1 – Snippet 2.1: PCA Weights
    # ------------------------------------------------------------------
    print("\n--- Snippet 2.1: pcaWeights ---")
    # Build a synthetic covariance matrix from multi-asset returns
    closes = get_multi_close(start="2022-01-01", end="2024-01-01")
    returns = closes.pct_change().dropna()
    cov_matrix = returns.cov().values
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print(f"Assets: {list(closes.columns)}")

    # Default: all risk on smallest eigenvalue component
    w_default = pcaWeights(cov_matrix)
    print(f"\nPCA weights (min-variance component):\n{w_default.flatten()}")

    # Equal risk across all components
    n = cov_matrix.shape[0]
    w_equal = pcaWeights(cov_matrix, riskDist=np.ones(n) / n)
    print(f"\nPCA weights (equal risk distribution):\n{w_equal.flatten()}")

    # ------------------------------------------------------------------
    # Demo 2 – Snippet 2.2: rollGaps / getRolledSeries
    # ------------------------------------------------------------------
    print("\n--- Snippet 2.2: rollGaps / getRolledSeries ---")
    # Synthesise a two-contract futures series
    rng = np.random.default_rng(42)
    n_bars = 500
    dates = pd.bdate_range("2023-01-03", periods=n_bars)
    # First contract for first 250 bars, second contract for rest
    symbols = np.array(["ESH3"] * 250 + ["ESM3"] * 250)
    log_ret = rng.normal(0, 0.005, n_bars)
    raw_close = 4000 + np.cumsum(log_ret * 4000)
    # Introduce a gap at the roll point
    raw_close[250:] += 15  # simulate a 15-point roll gap
    raw_open = raw_close + rng.normal(0, 0.5, n_bars)

    futures_df = pd.DataFrame({
        "Symbol": symbols,
        "Open": raw_open,
        "Close": raw_close,
    }, index=dates)

    gaps = rollGaps(futures_df)
    print(f"Cumulative gaps (last 5):\n{gaps.tail()}")

    rolled = getRolledSeries(futures_df.copy())
    print(f"\nRolled close (last 5):\n{rolled['Close'].tail()}")

    # ------------------------------------------------------------------
    # Demo 3 – Snippet 2.3: Non-Negative Rolled Prices
    # ------------------------------------------------------------------
    print("\n--- Snippet 2.3: Non-Negative Rolled Price Series ---")
    rolled_nn = getNonNegativeRolledSeries(futures_df.copy())
    print(f"rPrices (last 5):\n{rolled_nn['rPrices'].tail()}")
    print(f"rPrices range: [{rolled_nn['rPrices'].min():.4f}, "
          f"{rolled_nn['rPrices'].max():.4f}]")

    # ------------------------------------------------------------------
    # Demo 4 – Snippet 2.4: CUSUM Filter
    # ------------------------------------------------------------------
    print("\n--- Snippet 2.4: getTEvents (CUSUM filter) ---")
    close = get_close_series("MSFT", start="2022-01-01", end="2024-01-01")
    print(f"Close series length: {len(close)}")

    # Pick a threshold roughly equal to a meaningful price move
    h = close.std() * 0.5
    events = getTEvents(close, h)
    print(f"Threshold h = {h:.2f}")
    print(f"CUSUM events detected: {len(events)}")
    if len(events) > 0:
        print(f"First 5 events: {events[:5].tolist()}")

    print("\n" + "=" * 65)
    print("All Chapter 2 demos completed.")
    print("=" * 65)
