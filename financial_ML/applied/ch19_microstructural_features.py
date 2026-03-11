"""
Chapter 19: Microstructural Features
======================================
Advances in Financial Machine Learning – Marcos López de Prado

Market microstructure studies "the process and outcomes of exchanging assets
under explicit trading rules."  This chapter implements estimators for:
  - Tick rule (trade classification)
  - Roll model (effective bid-ask spread)
  - High-Low volatility estimator
  - Corwin-Schultz bid-ask spread estimator
  - Kyle's Lambda (price impact)
  - Amihud's Lambda (illiquidity)
  - VPIN (Volume-Synchronized Probability of Informed Trading)

Snippets:
  19.1 – Corwin-Schultz spread estimator  (getBeta, getGamma, getAlpha, corwinSchultz)
  19.2 – getSigma  (Beckers-Parkinson volatility from high-low prices)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import get_prices, get_close_series, generate_tick_data, SYMBOLS


# ============================================================================
# Tick Rule – Trade Classification
# ============================================================================
def tick_rule(prices):
    """
    Classify each trade as buy-initiated (+1) or sell-initiated (-1)
    using the tick rule.

    Parameters
    ----------
    prices : pd.Series
        Trade prices indexed by time.

    Returns
    -------
    pd.Series
        Aggressor flags: +1 (buy) or -1 (sell).
    """
    diff = prices.diff()
    bt = pd.Series(np.nan, index=prices.index, name='tick_rule')
    bt[diff > 0] = 1
    bt[diff < 0] = -1
    # When diff == 0, carry forward the previous classification
    bt.iloc[0] = 1  # arbitrary initialisation
    bt = bt.ffill()
    return bt.astype(int)


# ============================================================================
# Roll Model – Effective Bid-Ask Spread
# ============================================================================
def roll_model(prices):
    """
    Roll [1984] model: estimate the effective bid-ask spread from the
    serial covariance of price changes.

    Parameters
    ----------
    prices : pd.Series
        Price series.

    Returns
    -------
    dict
        {'spread': estimated bid-ask spread (2*c),
         'c': half-spread,
         'sigma_u': estimated noise volatility}
    """
    dp = prices.diff().dropna()
    cov = dp.autocorr(lag=1) * dp.var()  # serial covariance
    c = np.sqrt(max(0, -cov))
    sigma2_dp = dp.var()
    sigma2_u = sigma2_dp + 2 * cov
    return {
        'spread': 2 * c,
        'c': c,
        'sigma_u': np.sqrt(max(0, sigma2_u)),
    }


# ============================================================================
# High-Low Volatility Estimator (Parkinson / Beckers)
# ============================================================================
def hl_volatility(ohlc, window=20):
    """
    Parkinson / Beckers high-low volatility estimator.

    sigma_HL is estimated from log(H/L) which is more accurate than
    close-to-close volatility.

    Parameters
    ----------
    ohlc : pd.DataFrame
        DataFrame with 'High' and 'Low' columns.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling volatility estimate.
    """
    k1 = 4 * np.log(2)
    log_hl = np.log(ohlc['High'] / ohlc['Low'])
    sigma2 = log_hl.rolling(window).apply(lambda x: (x ** 2).mean()) / k1
    return np.sqrt(sigma2).rename('HL_volatility')


# ============================================================================
# Snippet 19.1 – Corwin-Schultz Bid-Ask Spread Estimator
# ============================================================================
def getBeta(series, sl):
    """
    Compute the beta component of Corwin-Schultz estimator.

    Parameters
    ----------
    series : pd.DataFrame
        DataFrame with 'High' and 'Low' columns.
    sl : int
        Sample length for rolling mean.

    Returns
    -------
    pd.Series
    """
    hl = series[['High', 'Low']].values
    hl = np.log(hl[:, 0] / hl[:, 1]) ** 2
    hl = pd.Series(hl, index=series.index)
    beta = hl.rolling(window=2).sum()
    beta = beta.rolling(window=sl).mean()
    return beta.dropna()


def getGamma(series):
    """
    Compute the gamma component of Corwin-Schultz estimator.

    Parameters
    ----------
    series : pd.DataFrame
        DataFrame with 'High' and 'Low' columns.

    Returns
    -------
    pd.Series
    """
    h2 = series['High'].rolling(window=2).max()
    l2 = series['Low'].rolling(window=2).min()
    gamma = np.log(h2.values / l2.values) ** 2
    gamma = pd.Series(gamma, index=h2.index)
    return gamma.dropna()


def getAlpha(beta, gamma):
    """
    Compute the alpha component of Corwin-Schultz estimator.

    Parameters
    ----------
    beta : pd.Series
    gamma : pd.Series

    Returns
    -------
    pd.Series
    """
    den = 3 - 2 * 2 ** 0.5
    alpha = (2 ** 0.5 - 1) * (beta ** 0.5) / den
    alpha -= (gamma / den) ** 0.5
    alpha[alpha < 0] = 0  # set negative alphas to 0 (p.727 of paper)
    return alpha.dropna()


def corwinSchultz(series, sl=1):
    """
    Corwin-Schultz [2012] bid-ask spread estimator from high-low prices.

    Parameters
    ----------
    series : pd.DataFrame
        DataFrame with 'High' and 'Low' columns, DatetimeIndex.
    sl : int
        Sample length for beta's rolling mean.

    Returns
    -------
    pd.DataFrame
        Columns: ['Spread', 'Start_Time'].
    """
    beta = getBeta(series, sl)
    gamma = getGamma(series)
    alpha = getAlpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    startTime = pd.Series(series.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, startTime], axis=1)
    spread.columns = ['Spread', 'Start_Time']
    return spread


# ============================================================================
# Snippet 19.2 – Beckers-Parkinson Volatility from High-Low Prices
# ============================================================================
def getSigma(beta, gamma):
    """
    Estimate volatility from high-low prices as a byproduct of the
    Corwin-Schultz model.

    Parameters
    ----------
    beta : pd.Series
    gamma : pd.Series

    Returns
    -------
    pd.Series
    """
    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** 0.5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma


# ============================================================================
# Kyle's Lambda – Price Impact
# ============================================================================
def kyle_lambda(prices, volumes, aggressor, window=50):
    """
    Kyle [1985] lambda: price impact coefficient estimated by regressing
    price changes on signed volume (net order flow).

    Parameters
    ----------
    prices : pd.Series
        Price series.
    volumes : pd.Series
        Traded volume series.
    aggressor : pd.Series
        Aggressor flags (+1 or -1).
    window : int
        Rolling window size for regression.

    Returns
    -------
    pd.Series
        Rolling Kyle's lambda.
    """
    dp = prices.diff()
    signed_vol = aggressor * volumes

    lambdas = pd.Series(np.nan, index=prices.index, name='Kyle_Lambda')
    for i in range(window, len(prices)):
        y = dp.iloc[i - window + 1:i + 1].values
        x = signed_vol.iloc[i - window + 1:i + 1].values
        mask = np.isfinite(y) & np.isfinite(x)
        y, x = y[mask], x[mask]
        if len(y) < 10:
            continue
        x_mat = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(x_mat, y, rcond=None)[0]
            lambdas.iloc[i] = beta[1]
        except np.linalg.LinAlgError:
            continue
    return lambdas


# ============================================================================
# Amihud's Lambda – Illiquidity
# ============================================================================
def amihud_lambda(prices, volumes, window=50):
    """
    Amihud [2002] lambda: ratio of absolute return to dollar volume,
    averaged over a rolling window.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    volumes : pd.Series
        Volume series.
    window : int
        Rolling window.

    Returns
    -------
    pd.Series
        Rolling Amihud illiquidity.
    """
    abs_ret = np.abs(np.log(prices / prices.shift(1)))
    dollar_vol = prices * volumes
    ratio = abs_ret / dollar_vol
    return ratio.rolling(window).mean().rename('Amihud_Lambda')


# ============================================================================
# VPIN – Volume-Synchronized Probability of Informed Trading
# ============================================================================
def vpin(prices, volumes, n_buckets=50, bucket_size=None):
    """
    VPIN (Easley et al. [2011]): estimate the probability of informed
    trading using volume bars.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    volumes : pd.Series
        Volume series.
    n_buckets : int
        Number of volume buckets to use in the VPIN calculation.
    bucket_size : float or None
        Volume per bucket. If None, computed as total_volume / (2 * n_buckets).

    Returns
    -------
    float
        VPIN estimate.
    """
    # Classify trades using tick rule
    bt = tick_rule(prices)

    # Compute buy/sell volume per trade
    v_buy = volumes.where(bt == 1, 0)
    v_sell = volumes.where(bt == -1, 0)

    if bucket_size is None:
        bucket_size = volumes.sum() / (2 * n_buckets)
        bucket_size = max(bucket_size, 1.0)

    # Accumulate into volume buckets
    cum_vol = volumes.cumsum()
    bucket_boundaries = np.arange(bucket_size, cum_vol.iloc[-1], bucket_size)

    vb_list, vs_list = [], []
    prev_idx = 0
    for boundary in bucket_boundaries[:n_buckets]:
        mask = cum_vol <= boundary
        if mask.sum() == 0:
            continue
        idx = mask.values.nonzero()[0][-1] + 1
        vb = v_buy.iloc[prev_idx:idx].sum()
        vs = v_sell.iloc[prev_idx:idx].sum()
        vb_list.append(vb)
        vs_list.append(vs)
        prev_idx = idx

    if len(vb_list) == 0:
        return 0.0

    vb_arr = np.array(vb_list)
    vs_arr = np.array(vs_list)
    order_imbalance = np.abs(vb_arr - vs_arr).sum()
    total_volume = (vb_arr + vs_arr).sum()

    return order_imbalance / total_volume if total_volume > 0 else 0.0


# ============================================================================
# DEMO
# ============================================================================
def main():
    """Demonstrate microstructural features on real and synthetic data."""
    OUTPUT_DIR = __import__('pathlib').Path(__file__).resolve().parent.parent / "_output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Chapter 19 – Microstructural Features")
    print("=" * 60)

    # --- 1) Tick rule on synthetic tick data ------------------------------
    print("\n[1] Tick rule on synthetic tick data:")
    ticks = generate_tick_data(10000)
    bt = tick_rule(ticks['price'])
    n_buy = (bt == 1).sum()
    n_sell = (bt == -1).sum()
    print(f"    Ticks: {len(bt)}, Buys: {n_buy}, Sells: {n_sell}")

    # --- 2) Roll model on real data ---------------------------------------
    print("\n[2] Roll model (effective spread):")
    for sym in SYMBOLS:
        close = get_close_series(sym, start="2023-01-01", end="2024-12-31")
        rm = roll_model(close)
        print(f"    {sym}: spread={rm['spread']:.6f}, "
              f"c={rm['c']:.6f}, sigma_u={rm['sigma_u']:.6f}")

    # --- 3) High-Low volatility -------------------------------------------
    print("\n[3] High-Low volatility estimator:")
    prices_dict = get_prices(["MSFT"], start="2023-01-01", end="2024-12-31")
    ohlc = prices_dict["MSFT"]
    hl_vol = hl_volatility(ohlc, window=20)
    print(f"    MSFT: mean HL vol = {hl_vol.mean():.6f}, "
          f"last = {hl_vol.iloc[-1]:.6f}")

    # --- 4) Corwin-Schultz spread ------------------------------------------
    print("\n[4] Corwin-Schultz spread estimator:")
    cs_spread = corwinSchultz(ohlc, sl=1)
    print(f"    MSFT: mean spread = {cs_spread['Spread'].mean():.6f}, "
          f"max = {cs_spread['Spread'].max():.6f}")

    # --- 5) Beckers-Parkinson volatility -----------------------------------
    print("\n[5] Beckers-Parkinson volatility (from Corwin-Schultz components):")
    beta = getBeta(ohlc, sl=1)
    gamma = getGamma(ohlc)
    sigma = getSigma(beta, gamma)
    print(f"    MSFT: mean sigma = {sigma.mean():.6f}")

    # --- 6) Kyle's lambda on tick data ------------------------------------
    print("\n[6] Kyle's Lambda on synthetic tick data:")
    tick_agg = tick_rule(ticks['price'])
    kl = kyle_lambda(ticks['price'], ticks['volume'], tick_agg, window=100)
    valid_kl = kl.dropna()
    print(f"    Mean lambda = {valid_kl.mean():.8f}")
    print(f"    Std lambda  = {valid_kl.std():.8f}")

    # --- 7) Amihud's lambda -----------------------------------------------
    print("\n[7] Amihud's Lambda:")
    for sym in ["MSFT", "NVDA"]:
        data = get_prices([sym], start="2023-01-01", end="2024-12-31")
        df = data[sym]
        al = amihud_lambda(df['Close'].squeeze(), df['Volume'].squeeze(), window=20)
        print(f"    {sym}: mean Amihud = {al.mean():.2e}")

    # --- 8) VPIN ----------------------------------------------------------
    print("\n[8] VPIN estimate on synthetic tick data:")
    vpin_val = vpin(ticks['price'], ticks['volume'], n_buckets=50)
    print(f"    VPIN = {vpin_val:.4f}")

    # --- 9) Save summary --------------------------------------------------
    summary = {
        'MSFT_Roll_spread': roll_model(get_close_series("MSFT", "2023-01-01", "2024-12-31"))['spread'],
        'MSFT_HL_vol_mean': hl_vol.mean(),
        'MSFT_CS_spread_mean': cs_spread['Spread'].mean(),
        'VPIN_synthetic': vpin_val,
    }
    pd.Series(summary).to_csv(OUTPUT_DIR / "ch19_microstructure.csv")
    print(f"\n[9] Results saved to _output/ch19_microstructure.csv")

    print("\n✓ Chapter 19 complete")


if __name__ == "__main__":
    main()
