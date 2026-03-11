"""
Chapter 18: Entropy Features
==============================
Advances in Financial Machine Learning – Marcos López de Prado

Entropy measures the amount of information (or disorder) contained in a
price series.  Higher entropy means the series is less predictable.
This chapter implements several entropy estimators and encoding schemes.

Snippets:
  18.1 – plugIn()      – Plug-in (Maximum Likelihood) entropy rate estimator
  18.2 – lempelZiv_lib – LZ compression library
  18.3 – matchLength   – Longest match length (Kontoyiannis)
  18.4 – konto()       – Kontoyiannis' LZ entropy estimate
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import get_close_series, SYMBOLS


# ============================================================================
# Snippet 18.1 – Plug-In (ML) Entropy Rate Estimator
# ============================================================================
def pmf1(msg, w):
    """
    Compute the probability mass function for a one-dimensional discrete
    random variable, using a sliding window of width *w*.

    Parameters
    ----------
    msg : str
        Encoded message string.
    w : int
        Word length.

    Returns
    -------
    dict
        {word: probability}
    """
    lib = {}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    for i in range(w, len(msg)):
        msg_ = msg[i - w:i]
        if msg_ not in lib:
            lib[msg_] = [i - w]
        else:
            lib[msg_] = lib[msg_] + [i - w]
    pmf = float(len(msg) - w)
    pmf = {i: len(lib[i]) / pmf for i in lib}
    return pmf


def plugIn(msg, w):
    """
    Compute the plug-in (Maximum Likelihood) entropy rate.

    Parameters
    ----------
    msg : str
        Encoded message string.
    w : int
        Word length.

    Returns
    -------
    tuple
        (entropy_rate, pmf_dict)
    """
    pmf = pmf1(msg, w)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / w
    return out, pmf


# ============================================================================
# Snippet 18.2 – Lempel-Ziv Library
# ============================================================================
def lempelZiv_lib(msg):
    """
    Build a library of non-redundant substrings using the LZ algorithm.

    The LZ algorithm decomposes a message into the smallest set of
    non-redundant substrings. The dictionary size relative to message
    length is an estimator of the compression rate (and hence entropy).

    Parameters
    ----------
    msg : str
        Input message.

    Returns
    -------
    list
        List of non-redundant substrings.
    """
    i, lib = 1, [msg[0]]
    while i < len(msg):
        for j in range(i, len(msg)):
            msg_ = msg[i:j + 1]
            if msg_ not in lib:
                lib.append(msg_)
                break
        i = j + 1
    return lib


# ============================================================================
# Snippet 18.3 – Longest Match Length
# ============================================================================
def matchLength(msg, i, n):
    """
    Maximum matched length + 1, with overlap.

    Finds the longest substring starting at position i that also appears
    in the window [i-n, i).

    Requirements: i >= n and len(msg) >= i + n.

    Parameters
    ----------
    msg : str
        Input message.
    i : int
        Current position.
    n : int
        Window size.

    Returns
    -------
    tuple
        (length_of_longest_match + 1, matched_substring)
    """
    subS = ''
    for l in range(n):
        msg1 = msg[i:i + l + 1]
        for j in range(i - n, i):
            msg0 = msg[j:j + l + 1]
            if msg1 == msg0:
                subS = msg1
                break  # search for higher l
    return len(subS) + 1, subS  # matched length + 1


# ============================================================================
# Snippet 18.4 – Kontoyiannis' LZ Entropy Estimate (konto)
# ============================================================================
def konto(msg, window=None):
    """
    Kontoyiannis' LZ entropy estimate, 2013 version (centered window).

    Computes the inverse of the average length of the shortest non-redundant
    substring. If non-redundant substrings are short, the text is highly
    entropic.

    Parameters
    ----------
    msg : str or list
        Input message.
    window : int or None
        If None, use expanding window; len(msg) must be even.
        If int, use sliding window of that size.

    Returns
    -------
    dict
        {'num': count, 'sum': cumulative value, 'subS': matched substrings,
         'h': entropy estimate, 'r': redundancy}
    """
    out = {'num': 0, 'sum': 0, 'subS': []}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            l, msg_ = matchLength(msg, i, i)
            out['sum'] += np.log2(i + 1) / l  # to avoid Doeblin condition
        else:
            l, msg_ = matchLength(msg, i, window)
            out['sum'] += np.log2(window + 1) / l  # to avoid Doeblin condition
        out['subS'].append(msg_)
        out['num'] += 1
    out['h'] = out['sum'] / out['num'] if out['num'] > 0 else 0
    out['r'] = 1 - out['h'] / np.log2(len(msg)) if len(msg) > 1 else 0  # redundancy
    return out


# ============================================================================
# Encoding schemes for price series
# ============================================================================
def binary_encode(returns):
    """
    Binary encoding of a return series: 1 if positive, 0 if negative.

    Parameters
    ----------
    returns : pd.Series
        Return series.

    Returns
    -------
    str
        Binary string (e.g. '10110...').
    """
    filtered = returns[returns != 0]
    return ''.join(['1' if r > 0 else '0' for r in filtered])


def quantile_encode(returns, n_bins=10):
    """
    Quantile encoding of returns. Assigns each return to a quantile bin
    (in-sample), producing a string of coded characters.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    str
        Encoded string using characters 0-9, a-z for bins.
    """
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    n_bins = min(n_bins, len(alphabet))
    quantiles = pd.qcut(returns, n_bins, labels=False, duplicates='drop')
    return ''.join([alphabet[int(q)] for q in quantiles.dropna()])


def sigma_encode(returns, step=0.01):
    """
    Sigma encoding: quantize returns by standard deviation multiples.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    step : float
        Size of each sigma bucket.

    Returns
    -------
    str
        Encoded string.
    """
    sigma = returns.std()
    if sigma < 1e-10:
        return '0' * len(returns)
    normalised = returns / sigma
    bins = np.floor(normalised / step).astype(int)
    # Map to alphabet range
    min_bin = bins.min()
    shifted = bins - min_bin
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    max_idx = len(alphabet) - 1
    clipped = shifted.clip(0, max_idx)
    return ''.join([alphabet[int(c)] for c in clipped])


# ============================================================================
# DEMO
# ============================================================================
def main():
    """Demonstrate entropy features on real stock data."""
    OUTPUT_DIR = __import__('pathlib').Path(__file__).resolve().parent.parent / "_output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Chapter 18 – Entropy Features")
    print("=" * 60)

    # --- 1) Fetch data ---------------------------------------------------
    symbol = "MSFT"
    print(f"\n[1] Fetching close prices for {symbol} ...")
    close = get_close_series(symbol, start="2020-01-01", end="2024-12-31")
    returns = close.pct_change().dropna()
    print(f"    {len(returns)} daily returns")

    # --- 2) Binary encoding + Plug-in entropy ----------------------------
    print("\n[2] Plug-in (ML) entropy estimator:")
    msg_bin = binary_encode(returns)
    print(f"    Binary-encoded message length: {len(msg_bin)}")

    for w in [1, 2, 3, 5]:
        h, pmf = plugIn(msg_bin, w)
        print(f"    w={w}: entropy rate = {h:.4f}")

    # --- 3) Lempel-Ziv library -------------------------------------------
    print("\n[3] Lempel-Ziv library:")
    lib = lempelZiv_lib(msg_bin)
    compression_ratio = len(lib) / len(msg_bin)
    print(f"    Library size: {len(lib)}")
    print(f"    Message length: {len(msg_bin)}")
    print(f"    Compression ratio: {compression_ratio:.4f}")

    # --- 4) Longest match function demo ----------------------------------
    print("\n[4] matchLength demo:")
    test_msg = '101010101010'
    mid = len(test_msg) // 2
    length, subs = matchLength(test_msg, mid, mid)
    print(f"    msg='{test_msg}', i={mid}, n={mid}")
    print(f"    Longest match: length={length}, substring='{subs}'")

    # --- 5) Kontoyiannis' entropy estimate --------------------------------
    print("\n[5] Kontoyiannis' LZ entropy estimate:")
    # Use a repeated pattern (low entropy) vs random (high entropy)
    low_entropy_msg = '10' * 50
    high_entropy_msg = msg_bin[:100]

    konto_low = konto(low_entropy_msg)
    konto_high = konto(high_entropy_msg)
    print(f"    Periodic ('10'*50): h={konto_low['h']:.4f}, r={konto_low['r']:.4f}")
    print(f"    Stock returns (first 100): h={konto_high['h']:.4f}, r={konto_high['r']:.4f}")

    # --- 6) Quantile encoding + entropy -----------------------------------
    print("\n[6] Quantile encoding + plug-in entropy:")
    msg_q = quantile_encode(returns, n_bins=5)
    for w in [1, 2, 3]:
        h, _ = plugIn(msg_q, w)
        print(f"    Quantile(5 bins), w={w}: entropy rate = {h:.4f}")

    # --- 7) Compare entropy across symbols --------------------------------
    print("\n[7] Entropy comparison across symbols:")
    results = {}
    for sym in SYMBOLS:
        try:
            c = get_close_series(sym, start="2022-01-01", end="2024-12-31")
            r = c.pct_change().dropna()
            msg = binary_encode(r)
            h1, _ = plugIn(msg, 1)
            lib_ = lempelZiv_lib(msg)
            cr = len(lib_) / len(msg)
            results[sym] = {'PlugIn_w1': h1, 'LZ_ratio': cr}
            print(f"    {sym}: PlugIn(w=1)={h1:.4f}, LZ_ratio={cr:.4f}")
        except Exception as e:
            print(f"    {sym}: error – {e}")

    df_results = pd.DataFrame(results).T
    df_results.to_csv(OUTPUT_DIR / "ch18_entropy.csv")
    print(f"\n[8] Results saved to _output/ch18_entropy.csv")

    print("\n✓ Chapter 18 complete")


if __name__ == "__main__":
    main()
