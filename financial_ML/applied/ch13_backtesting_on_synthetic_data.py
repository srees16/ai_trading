"""
Chapter 13: Backtesting on Synthetic Data
=========================================
Advances in Financial Machine Learning, Marcos Lopez de Prado

This chapter studies an alternative backtesting method that uses history to
generate synthetic datasets with statistical characteristics estimated from
observed data.  The key idea is to derive optimal trading rules (OTR)
directly from the estimated stochastic process (Ornstein-Uhlenbeck) rather
than by brute-force historical simulation, thereby avoiding backtest
overfitting of the trading rule parameters (profit-taking and stop-loss).

Snippets
--------
13.1  main()  – driver that sweeps (forecast, half-life) pairs and calls batch()
13.2  batch() – computes a mesh of Sharpe ratios for (profit-taking, stop-loss) pairs
"""

import numpy as np
from itertools import product

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample_data import generate_returns, generate_ohlcv_bars

# ============================================================================
# Snippet 13.1 – Main driver for optimal trading rule determination
# ============================================================================
# Generates a Cartesian product of (forecast, half-life) parameters that
# characterise the O-U process.  For each pair it calls batch() to compute
# Sharpe ratios across a mesh of trading rules.

def main(nIter=1e4, maxHP=100, grid_pts=10):
    """
    Sweep over (forecast, half-life) combinations and compute Sharpe-ratio
    surfaces for each.

    Parameters
    ----------
    nIter   : number of Monte-Carlo paths per trading-rule node
    maxHP   : maximum holding period (vertical barrier)
    grid_pts: number of grid points for profit-taking / stop-loss axes
    """
    rPT = rSLm = np.linspace(0.5, 10, grid_pts)
    results = {}
    count = 0
    for prod_ in product([10, 5, 0, -5, -10], [5, 10, 25, 50, 100]):
        count += 1
        coeffs = {"forecast": prod_[0], "hl": prod_[1], "sigma": 1}
        output = batch(coeffs, nIter=nIter, maxHP=maxHP,
                       rPT=rPT, rSLm=rSLm)
        results[(prod_[0], prod_[1])] = output
    return results


# ============================================================================
# Snippet 13.2 – Compute mesh of Sharpe ratios for a given (forecast, hl)
# ============================================================================
# For every node (profit-taking, stop-loss) in the grid, simulate nIter
# O-U paths.  A path exits when profit-taking, stop-loss, or the max
# holding period is hit.  The Sharpe ratio is computed from the resulting
# distribution of exit P&L values.

def batch(coeffs, nIter=1e4, maxHP=100,
          rPT=np.linspace(0.5, 10, 10),
          rSLm=np.linspace(0.5, 10, 10),
          seed=0):
    """
    Compute Sharpe ratios for a mesh of (profit-taking, stop-loss) pairs.

    Parameters
    ----------
    coeffs : dict with keys 'forecast', 'hl', 'sigma'
        forecast = E_0[P_{i,T_i}]  (long-run equilibrium price target)
        hl       = half-life (tau) of the O-U process
        sigma    = volatility of random shocks
    nIter  : number of Monte-Carlo iterations per node
    maxHP  : maximum holding period (vertical barrier)
    rPT    : array of profit-taking thresholds (multiples of sigma)
    rSLm   : array of stop-loss magnitudes (positive; applied as negative)
    seed   : starting price level

    Returns
    -------
    output1 : list of tuples (pt, sl, mean, std, sharpe)
    """
    phi = 2 ** (-1.0 / coeffs["hl"])       # autoregressive coefficient
    nIter = int(nIter)
    rng = np.random.default_rng(42)
    output1 = []

    for comb_ in product(rPT, rSLm):
        output2 = []
        for _ in range(nIter):
            p, hp = seed, 0
            while True:
                p = ((1 - phi) * coeffs["forecast"]
                     + phi * p
                     + coeffs["sigma"] * rng.standard_normal())
                cP = p - seed
                hp += 1
                if cP > comb_[0] or cP < -comb_[1] or hp > maxHP:
                    output2.append(cP)
                    break
        mean = np.mean(output2)
        std = np.std(output2)
        sharpe = mean / std if std > 0 else 0.0
        output1.append((comb_[0], comb_[1], mean, std, sharpe))

    return output1


# ============================================================================
# Helpers – estimation & display
# ============================================================================

def estimate_ou_params(prices):
    """
    Estimate O-U parameters (phi, sigma) from a price series via OLS
    on the linearised form  P_t = (1-phi)*mu + phi*P_{t-1} + sigma*eps.
    """
    X = prices[:-1].values
    Y = prices[1:].values
    phi_hat = np.cov(Y, X)[0, 1] / np.var(X) if np.var(X) > 0 else 0.0
    residuals = Y - phi_hat * X
    mu_hat = np.mean(residuals) / (1 - phi_hat) if abs(1 - phi_hat) > 1e-12 else 0.0
    sigma_hat = np.std(residuals)
    hl = -np.log(2) / np.log(abs(phi_hat)) if 0 < abs(phi_hat) < 1 else np.inf
    return {"phi": phi_hat, "sigma": sigma_hat, "mu": mu_hat, "half_life": hl}


def display_sharpe_surface(output, rPT, rSLm):
    """Pretty-print the Sharpe-ratio mesh as a 2-D grid."""
    n_pt, n_sl = len(rPT), len(rSLm)
    sharpe_grid = np.full((n_pt, n_sl), np.nan)
    for row in output:
        pt, sl, _mean, _std, sr = row
        i = np.argmin(np.abs(rPT - pt))
        j = np.argmin(np.abs(rSLm - sl))
        sharpe_grid[i, j] = sr

    header = "SL\\PT |" + "".join(f"{v:>7.1f}" for v in rPT)
    print(header)
    print("-" * len(header))
    for j, sl_val in enumerate(rSLm):
        row_str = f"{sl_val:5.1f} |"
        for i in range(n_pt):
            val = sharpe_grid[i, j]
            row_str += f"{val:7.3f}" if not np.isnan(val) else "    nan"
        print(row_str)


# ============================================================================
# __main__ – run a demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 13: Backtesting on Synthetic Data")
    print("  Numerical Determination of Optimal Trading Rules")
    print("=" * 70)

    # -- Use sample_data generators to show integration --
    bars = generate_ohlcv_bars(n_bars=500, seed=42)
    rets = generate_returns(n=500, n_assets=1, seed=42)
    print(f"\n[sample_data] OHLCV bars shape : {bars.shape}")
    print(f"[sample_data] Returns shape    : {rets.shape}")

    # -- Estimate O-U parameters from synthetic close prices --
    close = bars["Close"]
    ou_params = estimate_ou_params(close)
    print(f"\nEstimated O-U parameters from synthetic close prices:")
    print(f"  phi (AR coeff) = {ou_params['phi']:.6f}")
    print(f"  sigma          = {ou_params['sigma']:.4f}")
    print(f"  mu (long-run)  = {ou_params['mu']:.4f}")
    print(f"  half-life      = {ou_params['half_life']:.2f}")

    # -- Run a single (forecast, half-life) configuration --
    # Use small nIter and grid for fast demo
    grid_pts = 10
    rPT = rSLm = np.linspace(0.5, 10, grid_pts)
    demo_nIter = 5_000

    test_cases = [
        {"forecast": 0,  "hl": 5,  "sigma": 1},   # zero forecast, fast reversion
        {"forecast": 5,  "hl": 10, "sigma": 1},    # positive forecast
        {"forecast": -5, "hl": 10, "sigma": 1},    # negative forecast
    ]

    for coeffs in test_cases:
        label = (f"Forecast={coeffs['forecast']}, "
                 f"Half-Life={coeffs['hl']}, "
                 f"Sigma={coeffs['sigma']}")
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"  nIter={demo_nIter}, maxHP=100, grid={grid_pts}x{grid_pts}")
        print("=" * 70)

        output = batch(coeffs, nIter=demo_nIter, maxHP=100,
                       rPT=rPT, rSLm=rSLm)

        # Find optimal trading rule
        best = max(output, key=lambda x: x[4])
        print(f"\n  Optimal trading rule:")
        print(f"    Profit-Taking = {best[0]:.1f}*sigma")
        print(f"    Stop-Loss     = {best[1]:.1f}*sigma")
        print(f"    Mean P&L      = {best[2]:.4f}")
        print(f"    Std P&L       = {best[3]:.4f}")
        print(f"    Sharpe Ratio  = {best[4]:.4f}")

        print(f"\n  Sharpe Ratio Surface:")
        display_sharpe_surface(output, rPT, rSLm)

    # -- Quick sweep of main() with reduced iterations --
    print(f"\n{'=' * 70}")
    print("  Running full sweep via main() (reduced iterations) ...")
    print("=" * 70)
    all_results = main(nIter=1_000, maxHP=100, grid_pts=5)
    print(f"\n  Completed {len(all_results)} (forecast, half-life) combinations")
    for key, output in all_results.items():
        best = max(output, key=lambda x: x[4])
        print(f"  Forecast={key[0]:>3}, HL={key[1]:>3}  =>  "
              f"Best SR={best[4]:+.4f}  (PT={best[0]:.1f}, SL={best[1]:.1f})")

    print(f"\nDone.")
