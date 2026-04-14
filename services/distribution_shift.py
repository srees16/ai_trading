"""
Distribution Shift Detector — measures reality gap between backtest and live returns.

Computes Wasserstein distance, KL divergence, and (optionally) Sinkhorn divergence
to detect when live/paper trading returns diverge from historical backtest returns.
Includes a rolling-window variant to pinpoint when drift began.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

logger = logging.getLogger(__name__)

# ── Thresholds (calibrated for raw decimal daily returns) ─────────────
_WASS_DRIFT = 0.01        # Wasserstein: stable < 0.01
_WASS_REGIME = 0.04       # Wasserstein: regime_break > 0.04
_KL_DRIFT = 0.3           # KL: stable < 0.3
_KL_REGIME = 1.0          # KL: regime_break > 1.0
_MIN_SAMPLES = 30         # minimum days before computing


def _histogram_kl(p_samples: np.ndarray, q_samples: np.ndarray,
                  n_bins: int = 50) -> float:
    """KL divergence via histogram binning with Laplace smoothing."""
    combined = np.concatenate([p_samples, q_samples])
    bin_edges = np.linspace(combined.min() - 1e-8, combined.max() + 1e-8, n_bins + 1)

    p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=False)
    q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=False)

    # Laplace smoothing to avoid log(0)
    p_prob = (p_hist + 1) / (p_hist.sum() + n_bins)
    q_prob = (q_hist + 1) / (q_hist.sum() + n_bins)

    return float(np.sum(rel_entr(p_prob, q_prob)))


def _try_sinkhorn(backtest: np.ndarray, live: np.ndarray) -> Optional[float]:
    """Sinkhorn divergence via geomloss (skip if unavailable)."""
    try:
        import torch
        from geomloss import SamplesLoss

        loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
        bt = torch.tensor(backtest.reshape(-1, 1), dtype=torch.float32)
        lv = torch.tensor(live.reshape(-1, 1), dtype=torch.float32)
        return float(loss(bt, lv).item())
    except ImportError:
        return None
    except Exception as e:
        logger.debug("Sinkhorn computation failed: %s", e)
        return None


def _classify(wass: float, kl: float) -> str:
    """Return verdict string based on thresholds."""
    if wass >= _WASS_REGIME or kl >= _KL_REGIME:
        return "regime_break"
    if wass >= _WASS_DRIFT or kl >= _KL_DRIFT:
        return "drifting"
    return "stable"


def detect_distribution_shift(
    backtest_returns: np.ndarray,
    live_returns: np.ndarray,
) -> Dict:
    """Compute distribution shift between backtest and live daily returns.

    Parameters
    ----------
    backtest_returns : np.ndarray
        Daily returns from the backtest period (e.g. 2020-2025 OOS).
    live_returns : np.ndarray
        Daily returns from live/paper trading.

    Returns
    -------
    dict with keys:
        wasserstein, kl_divergence, sinkhorn (or None),
        verdict ("stable" | "drifting" | "regime_break"),
        n_backtest, n_live
    """
    bt = np.asarray(backtest_returns, dtype=np.float64).ravel()
    lv = np.asarray(live_returns, dtype=np.float64).ravel()

    # Remove NaN/Inf
    bt = bt[np.isfinite(bt)]
    lv = lv[np.isfinite(lv)]

    if len(bt) < _MIN_SAMPLES or len(lv) < _MIN_SAMPLES:
        return {
            "wasserstein": None,
            "kl_divergence": None,
            "sinkhorn": None,
            "verdict": "insufficient_data",
            "n_backtest": len(bt),
            "n_live": len(lv),
        }

    wass = float(wasserstein_distance(bt, lv))
    kl = _histogram_kl(bt, lv)
    sinkhorn = _try_sinkhorn(bt, lv)
    verdict = _classify(wass, kl)

    return {
        "wasserstein": round(wass, 6),
        "kl_divergence": round(kl, 6),
        "sinkhorn": round(sinkhorn, 6) if sinkhorn is not None else None,
        "verdict": verdict,
        "n_backtest": len(bt),
        "n_live": len(lv),
    }


def detect_distribution_shift_rolling(
    backtest_returns: np.ndarray,
    live_returns: np.ndarray,
    window: int = 60,
    step: int = 1,
) -> List[Dict]:
    """Rolling-window shift detection over live returns.

    Slides a window across the live returns, comparing each window
    against the backtest distribution to detect when drift began.

    Parameters
    ----------
    backtest_returns : np.ndarray
        Full backtest daily returns.
    live_returns : np.ndarray
        Live/paper daily returns (must be chronologically ordered).
    window : int
        Rolling window size in days (default 60).
    step : int
        Step size between windows (default 1 = every day).

    Returns
    -------
    list of dicts, each with:
        window_start, window_end, wasserstein, kl_divergence, verdict
    """
    bt = np.asarray(backtest_returns, dtype=np.float64).ravel()
    lv = np.asarray(live_returns, dtype=np.float64).ravel()
    bt = bt[np.isfinite(bt)]
    lv = lv[np.isfinite(lv)]

    if len(lv) < window or len(bt) < _MIN_SAMPLES:
        return []

    results = []
    for start in range(0, len(lv) - window + 1, step):
        end = start + window
        lv_window = lv[start:end]

        wass = float(wasserstein_distance(bt, lv_window))
        kl = _histogram_kl(bt, lv_window)
        verdict = _classify(wass, kl)

        results.append({
            "window_start": start,
            "window_end": end,
            "wasserstein": round(wass, 6),
            "kl_divergence": round(kl, 6),
            "verdict": verdict,
        })

    return results
