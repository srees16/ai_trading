"""
ch05_fractionally_differentiated_features.py – Chapter 5: Fractionally Differentiated Features
From "Advances in Financial Machine Learning" by Marcos López de Prado.

Implements Snippets 5.1–5.4:
  5.1   getWeights / plotWeights   – fractional differentiation weights & visualization
  5.2   fracDiff                   – expanding-window fractional differentiation
  5.3   getWeights_FFD / fracDiff_FFD – fixed-width window fracdiff (practical)
  5.4   plotMinFFD                 – find minimum d that passes the ADF test
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Allow imports from the parent directory (financial_ML/)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import get_close_series


# ============================================================================
# Snippet 5.1 – Weighting Function
# ============================================================================

def getWeights(d, size):
    """
    Compute the weights for fractional differentiation via the iterative
    formula:  w_k = -w_{k-1} / k * (d - k + 1).

    Parameters
    ----------
    d    : float – fractional differentiation order.
    size : int   – number of weights to compute.

    Returns
    -------
    np.ndarray of shape (size, 1) – weights in chronological order
                                    (oldest weight first).
    """
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plotWeights(dRange, nPlots, size, savePath=None):
    """
    Plot the sequence of weights w_k for various values of d.

    Parameters
    ----------
    dRange   : tuple  – (d_min, d_max) range of d values.
    nPlots   : int    – number of d values to plot.
    size     : int    – number of weights per curve.
    savePath : str    – if provided, save figure to this path instead of showing.
    """
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[round(d, 2)])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left')
    ax.set_xlabel('k (lag)')
    ax.set_ylabel('w_k')
    ax.set_title(f'Fractional Differentiation Weights (d in [{dRange[0]}, {dRange[1]}])')
    if savePath:
        plt.savefig(savePath, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# Snippet 5.2 – Standard FracDiff (Expanding Window)
# ============================================================================

def fracDiff(series, d, thres=0.01):
    """
    Fractionally differentiate a time series using an expanding window.

    Weights whose cumulative absolute share falls below *thres* are skipped
    (initial observations are dropped to control weight-loss).

    Parameters
    ----------
    series : pd.DataFrame – columns are features, index is datetime.
    d      : float        – fractional differentiation order (any positive real).
    thres  : float        – weight-loss threshold in (0, 1]. thres=1 keeps all.

    Returns
    -------
    pd.DataFrame – fractionally differentiated series (fewer rows than input).
    """
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float)
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# ============================================================================
# Snippet 5.3 – Fixed-Width Window FracDiff (FFD)
# ============================================================================

def getWeights_FFD(d, thres):
    """
    Compute weights for a fixed-width window fractional differentiation.

    Weights are generated iteratively and dropped once |w_k| < thres.

    Parameters
    ----------
    d     : float – fractional differentiation order.
    thres : float – minimum absolute weight to keep.

    Returns
    -------
    np.ndarray of shape (width, 1) – weights in chronological order.
    """
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def fracDiff_FFD(series, d, thres=1e-5):
    """
    Fractionally differentiate a time series using a fixed-width window.

    Uses a constant-width weight vector (from getWeights_FFD) across all
    observations, avoiding the negative drift caused by an expanding window.

    Parameters
    ----------
    series : pd.DataFrame – columns are features, index is datetime.
    d      : float        – fractional differentiation order.
    thres  : float        – minimum absolute weight to keep.

    Returns
    -------
    pd.DataFrame – fractionally differentiated series.
    """
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float)
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# ============================================================================
# Snippet 5.4 – Finding the Minimum d Value That Passes the ADF Test
# ============================================================================

def plotMinFFD(close, savePath=None):
    """
    Iterate d from 0 to 1 in increments of 0.1, apply FFD, and run the
    Augmented Dickey-Fuller test to find the minimum *d* that achieves
    stationarity.

    Parameters
    ----------
    close    : pd.Series or pd.DataFrame – close prices (will be log-transformed).
    savePath : str – if provided, save figure to this path.

    Returns
    -------
    pd.DataFrame – results table with columns
                   ['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'].
    """
    from statsmodels.tsa.stattools import adfuller

    # Ensure DataFrame with a 'Close' column
    if isinstance(close, pd.Series):
        close = close.to_frame(name='Close')

    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    df0 = np.log(close[['Close']]).resample('1D').last().dropna()

    for d in np.linspace(0, 1, 11):
        df1 = fracDiff_FFD(df0, d, thres=0.01)
        if df1.dropna().shape[0] == 0:
            continue
        corr = np.corrcoef(df0.loc[df1.index, 'Close'], df1['Close'])[0, 1]
        adf = adfuller(df1['Close'], maxlag=1, regression='c', autolag=None)
        out.loc[round(d, 1)] = list(adf[:4]) + [adf[4]['5%']] + [corr]

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    out['corr'].plot(ax=ax1, color='blue', label='corr (left)')
    out['adfStat'].plot(ax=ax2, color='green', label='adfStat (right)')
    ax2.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted',
                label='95% critical value')
    ax1.set_xlabel('d')
    ax1.set_ylabel('Correlation')
    ax2.set_ylabel('ADF Statistic')
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower right')
    plt.title('ADF Statistic as a Function of d')

    if savePath:
        plt.savefig(savePath, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return out


# ============================================================================
# Main – demonstrate each snippet with real stock data
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Chapter 5: Fractionally Differentiated Features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Download MSFT close prices
    # ------------------------------------------------------------------
    print("\n[1] Downloading MSFT close prices via sample_data ...")
    close = get_close_series("MSFT")
    print(f"    Obtained {len(close)} daily observations "
          f"from {close.index[0].date()} to {close.index[-1].date()}")

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "_output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Snippet 5.1 – getWeights / plotWeights
    # ------------------------------------------------------------------
    print("\n[2] Snippet 5.1 – Fractional differentiation weights")
    print("    Saving weight plots ...")
    plotWeights(dRange=[0, 1], nPlots=11, size=6,
                savePath=str(OUTPUT_DIR / "ch05_weights_d0_1.png"))
    plotWeights(dRange=[1, 2], nPlots=11, size=6,
                savePath=str(OUTPUT_DIR / "ch05_weights_d1_2.png"))
    print(f"    Saved to {OUTPUT_DIR / 'ch05_weights_d0_1.png'}")
    print(f"    Saved to {OUTPUT_DIR / 'ch05_weights_d1_2.png'}")

    # Show a few example weight vectors
    for d in [0.0, 0.3, 0.5, 1.0]:
        w = getWeights(d, size=6)
        print(f"    d={d:.1f} weights (newest→oldest): "
              f"{np.round(w.flatten()[::-1], 4).tolist()}")

    # ------------------------------------------------------------------
    # 3. Snippet 5.2 – fracDiff (expanding window)
    # ------------------------------------------------------------------
    print("\n[3] Snippet 5.2 – Expanding-window fracDiff")
    close_df = close.to_frame(name='Close')

    for d in [0.3, 0.5, 0.7, 1.0]:
        fd = fracDiff(close_df, d=d, thres=0.01)
        print(f"    d={d:.1f}  →  {len(fd)} obs  |  "
              f"mean={fd['Close'].mean():.4f}  std={fd['Close'].std():.4f}")

    # ------------------------------------------------------------------
    # 4. Snippet 5.3 – fracDiff_FFD (fixed-width window)
    # ------------------------------------------------------------------
    print("\n[4] Snippet 5.3 – Fixed-width window fracDiff (FFD)")
    for d in [0.3, 0.5, 0.7, 1.0]:
        ffd = fracDiff_FFD(close_df, d=d, thres=1e-5)
        print(f"    d={d:.1f}  →  {len(ffd)} obs  |  "
              f"mean={ffd['Close'].mean():.4f}  std={ffd['Close'].std():.4f}")

    # ------------------------------------------------------------------
    # 5. Snippet 5.4 – plotMinFFD (ADF test for minimum d)
    # ------------------------------------------------------------------
    print("\n[5] Snippet 5.4 – Minimum d for stationarity (ADF test)")
    results = plotMinFFD(close, savePath=str(OUTPUT_DIR / "ch05_minFFD.png"))
    print(f"    Saved plot to {OUTPUT_DIR / 'ch05_minFFD.png'}")
    print("\n    ADF results by d:")
    print(results.to_string())

    # Find the minimum d where adfStat < 95% critical value
    crit = results['95% conf'].mean()
    stationary = results[results['adfStat'].astype(float) < crit]
    if not stationary.empty:
        min_d = stationary.index[0]
        row = stationary.iloc[0]
        print(f"\n    Minimum d for stationarity: {min_d}")
        print(f"    ADF stat = {row['adfStat']:.4f}  (95% crit = {crit:.4f})")
        print(f"    Correlation with original series: {row['corr']:.4f}")
        print(f"    → Only {min_d:.0%} differentiation needed vs. 100% for returns!")
    else:
        print("\n    No d in [0, 1] achieved stationarity at 95% confidence.")

    print("\n" + "=" * 70)
    print("Chapter 5 demo complete.")
    print("=" * 70)
