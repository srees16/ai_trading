"""
Chapter 10: Bet Sizing
Advances in Financial Machine Learning – Marcos Lopez de Prado

Snippets 10.1–10.4: Translating predictions into position sizes,
averaging active bets, discretising signals, and dynamic bet sizing
with limit-price derivation.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_ohlcv_bars

# ---------------------------------------------------------------------------
# Snippet 10.1 – From Probabilities to Bet Size
# ---------------------------------------------------------------------------

def getSignal(events, stepSize, prob, pred, numClasses, **kargs):
    """Translate predicted probabilities into a discretised bet size in [-1, 1].

    Parameters
    ----------
    events : DataFrame with at least a 't1' column (barrier touch time) and
             optionally a 'side' column when meta-labeling is used.
    stepSize : float – discretisation granularity (e.g. 0.1).
    prob : Series – predicted probability for the chosen class.
    pred : Series – predicted label (encodes side of the bet).
    numClasses : int – number of classes in the classifier.

    Returns
    -------
    Series of discretised bet sizes in [-1, 1].
    """
    if prob.shape[0] == 0:
        return pd.Series(dtype=float)
    # 1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0 = (prob - 1.0 / numClasses) / (prob * (1.0 - prob)) ** 0.5  # t-value
    signal0 = pred * (2 * norm.cdf(signal0) - 1)  # signal = side * size
    if 'side' in events.columns:
        signal0 *= events.loc[signal0.index, 'side']  # meta-labeling
    # 2) compute average signal among those concurrently open
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avgActiveSignals(df0)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


# ---------------------------------------------------------------------------
# Snippet 10.2 – Averaging Active Bets
# ---------------------------------------------------------------------------

def avgActiveSignals(signals):
    """Compute the average signal among those still active at each time point.

    Parameters
    ----------
    signals : DataFrame with columns 'signal' and 't1', indexed by signal time.

    Returns
    -------
    Series of average active signals at each change-point.
    """
    # 1) time points where signals change (either one starts or one ends)
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = sorted(tPnts)
    out = mpAvgActiveSignals(signals, molecule=tPnts)
    return out


def mpAvgActiveSignals(signals, molecule):
    """At each time in *molecule*, average signals still active.

    A signal is active if:
      a) issued before or at loc AND
      b) loc is before the signal's end-time, or end-time is still unknown (NaT).
    """
    out = pd.Series(dtype=float)
    for loc in molecule:
        mask = (signals.index.values <= loc) & (
            (loc < signals['t1']) | pd.isnull(signals['t1'])
        )
        act = signals[mask].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0  # no signals active at this time
    return out


# ---------------------------------------------------------------------------
# Snippet 10.3 – Size Discretisation to Prevent Overtrading
# ---------------------------------------------------------------------------

def discreteSignal(signal0, stepSize):
    """Discretise signal to multiples of *stepSize* and clip to [-1, 1].

    Parameters
    ----------
    signal0 : Series – raw (averaged) signal values.
    stepSize : float – discretisation step (e.g. 0.1).

    Returns
    -------
    Series of discretised signals capped at ±1.
    """
    signal1 = (signal0 / stepSize).round() * stepSize
    signal1[signal1 > 1] = 1
    signal1[signal1 < -1] = -1
    return signal1


# ---------------------------------------------------------------------------
# Snippet 10.4 – Dynamic Position Size and Limit Price
# ---------------------------------------------------------------------------

def betSize(w, x):
    """Sigmoid-like bet size: m(w, x) = x / sqrt(w + x^2)."""
    return x * (w + x ** 2) ** (-0.5)


def getTPos(w, f, mP, maxPos):
    """Target position size given forecast *f* and market price *mP*."""
    return int(betSize(w, f - mP) * maxPos)


def invPrice(f, w, m):
    """Inverse price function: L(f, w, m) = f - m * sqrt(w / (1 - m^2))."""
    return f - m * (w / (1 - m ** 2)) ** 0.5


def limitPrice(tPos, pos, f, w, maxPos):
    """Breakeven limit price for an order of size (tPos - pos).

    Parameters
    ----------
    tPos : int – target position.
    pos : int – current position.
    f : float – forecasted price.
    w : float – sigmoid width parameter.
    maxPos : int – maximum absolute position.

    Returns
    -------
    float – limit price for the order.
    """
    sgn = 1 if tPos >= pos else -1
    lP = 0.0
    for j in range(abs(pos + sgn), abs(tPos) + 1):
        lP += invPrice(f, w, j / float(maxPos))
    lP /= tPos - pos
    return lP


def getW(x, m):
    """Calibrate sigmoid width w so that betSize(w, x) == m.

    w = x^2 * (m^{-2} - 1)
    """
    return x ** 2 * (m ** (-2) - 1)


# ===================================================================
# DEMONSTRATION
# ===================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Chapter 10 – Bet Sizing  (Advances in Financial Machine Learning)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Generate synthetic OHLCV bars and mock prediction data
    # ------------------------------------------------------------------
    bars = generate_ohlcv_bars(n_bars=500, seed=42)
    close = bars['Close']
    rng = np.random.default_rng(123)

    n_signals = 200
    signal_idx = bars.index[:n_signals]
    # Simulate holding periods (barrier touch times) 5-20 days out
    hold = rng.integers(5, 20, size=n_signals)
    t1 = pd.Series(
        [bars.index[min(i + h, len(bars) - 1)] for i, h in enumerate(hold)],
        index=signal_idx,
        name='t1',
    )
    events = pd.DataFrame({'t1': t1})

    # Simulated predicted probabilities and labels
    prob = pd.Series(rng.uniform(0.45, 0.95, n_signals), index=signal_idx)
    pred = pd.Series(rng.choice([-1, 1], n_signals), index=signal_idx)

    # ------------------------------------------------------------------
    # 2. Snippet 10.1 + 10.2 + 10.3: getSignal pipeline
    # ------------------------------------------------------------------
    print("\n--- Snippet 10.1–10.3: getSignal pipeline ---")
    signal = getSignal(events, stepSize=0.1, prob=prob, pred=pred, numClasses=2)
    print(f"  Generated {len(signal)} averaged / discretised signals")
    print(f"  Signal range : [{signal.min():.2f}, {signal.max():.2f}]")
    print(f"  Mean signal  : {signal.mean():.4f}")
    print(f"\n  First 10 signals:\n{signal.head(10).to_string()}")

    # ------------------------------------------------------------------
    # 3. Snippet 10.3 standalone: discreteSignal demo
    # ------------------------------------------------------------------
    print("\n--- Snippet 10.3: discreteSignal standalone ---")
    raw = pd.Series([0.23, -0.67, 0.05, 0.91, -0.44, 1.05, -1.2])
    disc = discreteSignal(raw, stepSize=0.2)
    demo = pd.DataFrame({'raw': raw, 'discretised': disc})
    print(demo.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Snippet 10.4: Dynamic position size & limit price
    # ------------------------------------------------------------------
    print("\n--- Snippet 10.4: Dynamic Position Size & Limit Price ---")
    pos, maxPos, mP, f = 0, 100, 100, 115
    wParams = {'divergence': 10, 'm': 0.95}

    w = getW(wParams['divergence'], wParams['m'])
    print(f"  Calibrated w         : {w:.4f}")

    tPos = getTPos(w, f, mP, maxPos)
    print(f"  Target position      : {tPos}  (maxPos={maxPos})")

    lP = limitPrice(tPos, pos, f, w, maxPos)
    print(f"  Limit price          : {lP:.4f}  (market={mP}, forecast={f})")

    # Verify calibration: for divergence=10, bet size should be ~0.95
    m_check = betSize(w, wParams['divergence'])
    print(f"  betSize(w, {wParams['divergence']})     : {m_check:.4f}  (expected ~{wParams['m']})")

    # Show how target position varies with market price
    print("\n  Market price sweep (f=115, maxPos=100):")
    print(f"  {'mP':>6s}  {'tPos':>5s}  {'betSz':>7s}")
    for mp in [95, 100, 105, 110, 112, 114, 115, 116, 120]:
        bs = betSize(w, f - mp)
        tp = getTPos(w, f, mp, maxPos)
        print(f"  {mp:6d}  {tp:5d}  {bs:7.4f}")

    print("\n" + "=" * 70)
    print("Chapter 10 demo complete.")
