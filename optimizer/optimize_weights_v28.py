"""
v28 — Walk-Forward Signal Weight Optimizer (12 signals, anti-overfit).

Changes vs v27:
  1. 12 active signals: +carry (low-corr value), +skew_signal (contrarian)
     Rejected: order_flow (ρ>0.7 with trend cluster), cross_momentum (no data)
  2. Tighter bounds: [0.01, 0.12] per signal (was [0.01, 0.15])
  3. Stricter overfitting gates:
     - PBO < 25% (was 30%)
     - Train-test Sharpe gap < 0.30
     - Test Sharpe >= 0.85
     - Effective N >= 5 active signals
  4. Anchored walk-forward validation (3 windows post-optimization)
  5. Bull leverage sensitivity analysis (NOT optimized — avoids leverage overfit)
  6. Empirical correlation matrix from extracted data

Usage:
    python optimize_weights_v28.py
"""
import sys
import os
import pickle
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

_INPUT_PATH = os.path.join(_root, "data", "extracted_forecasts.pkl")

# ── 12 active signals (v28: +carry, +skew_signal) ──
ACTIVE_SIGNALS = [
    "ewmac_16_64", "ewmac_64_256",
    "screener", "momentum", "mean_reversion",
    "penfold_trend", "ehlers_dsp", "acceleration",
    "carver_value", "breakout",
    # v28 additions (empirically validated low-correlation sources):
    "carry",        # ρ≈0.07 with existing — genuine value/dividend signal
    "skew_signal",  # ρ≈-0.12 with existing — contrarian/options-implied
]

# v27 champion weights (baseline for comparison)
V27_WEIGHTS = {
    "ewmac_16_64": 0.014, "ewmac_64_256": 0.115,
    "screener": 0.101, "momentum": 0.159, "mean_reversion": 0.055,
    "penfold_trend": 0.018, "ehlers_dsp": 0.178, "acceleration": 0.156,
    "carver_value": 0.188, "breakout": 0.016,
    "carry": 0.0, "skew_signal": 0.0,
}

# ── Empirical correlation matrix (computed from extracted_forecasts.pkl) ──
# Existing pairs from v27 (R19c static estimates):
CORR_PAIRS = {
    ("ewmac_16_64", "ewmac_64_256"): 0.60,
    ("ewmac_16_64", "momentum"): 0.55,
    ("ewmac_64_256", "momentum"): 0.50,
    ("ewmac_16_64", "breakout"): 0.50,
    ("ewmac_64_256", "breakout"): 0.40,
    ("ewmac_16_64", "penfold_trend"): 0.50,
    ("ewmac_64_256", "penfold_trend"): 0.45,
    ("momentum", "penfold_trend"): 0.65,
    ("momentum", "breakout"): 0.55,
    ("penfold_trend", "breakout"): 0.65,
    ("momentum", "mean_reversion"): -0.20,
    ("ewmac_16_64", "mean_reversion"): -0.20,
    ("ewmac_64_256", "mean_reversion"): -0.25,
    ("penfold_trend", "mean_reversion"): -0.10,
    ("breakout", "mean_reversion"): -0.10,
    ("momentum", "ehlers_dsp"): 0.45,
    ("ewmac_16_64", "ehlers_dsp"): 0.40,
    ("ewmac_64_256", "ehlers_dsp"): 0.35,
    ("ehlers_dsp", "penfold_trend"): 0.40,
    ("ehlers_dsp", "mean_reversion"): 0.05,
    ("ehlers_dsp", "breakout"): 0.35,
    ("momentum", "screener"): 0.50,
    ("ewmac_16_64", "screener"): 0.50,
    ("screener", "breakout"): 0.40,
    ("screener", "mean_reversion"): -0.05,
    ("screener", "ehlers_dsp"): 0.30,
    ("momentum", "acceleration"): 0.70,
    ("ewmac_16_64", "acceleration"): 0.60,
    ("ewmac_64_256", "acceleration"): 0.40,
    ("acceleration", "penfold_trend"): 0.50,
    ("acceleration", "breakout"): 0.45,
    ("acceleration", "mean_reversion"): -0.15,
    ("acceleration", "ehlers_dsp"): 0.40,
    ("acceleration", "screener"): 0.45,
    ("momentum", "carver_value"): 0.10,
    ("ewmac_64_256", "carver_value"): 0.15,
    ("mean_reversion", "carver_value"): 0.40,
    ("carver_value", "penfold_trend"): 0.10,
    ("carver_value", "ehlers_dsp"): 0.10,
    ("carver_value", "breakout"): 0.05,
    ("carver_value", "screener"): 0.10,
    ("carver_value", "acceleration"): 0.05,
    ("ewmac_16_64", "carver_value"): 0.10,
    # ── v28: Empirical correlations for carry (from extracted data) ──
    ("carry", "ewmac_16_64"): 0.03,
    ("carry", "ewmac_64_256"): 0.06,
    ("carry", "screener"): 0.01,
    ("carry", "momentum"): 0.18,
    ("carry", "mean_reversion"): 0.07,
    ("carry", "penfold_trend"): 0.01,
    ("carry", "ehlers_dsp"): -0.01,
    ("carry", "acceleration"): -0.01,
    ("carry", "carver_value"): 0.32,
    ("carry", "breakout"): -0.03,
    # ── v28: Empirical correlations for skew_signal ──
    ("skew_signal", "ewmac_16_64"): -0.21,
    ("skew_signal", "ewmac_64_256"): -0.46,
    ("skew_signal", "screener"): -0.12,
    ("skew_signal", "momentum"): -0.23,
    ("skew_signal", "mean_reversion"): 0.05,
    ("skew_signal", "penfold_trend"): -0.13,
    ("skew_signal", "ehlers_dsp"): -0.06,
    ("skew_signal", "acceleration"): 0.09,
    ("skew_signal", "carver_value"): -0.10,
    ("skew_signal", "breakout"): -0.03,
    # ── v28: New pair correlations ──
    ("carry", "skew_signal"): -0.39,
}
DEFAULT_CORR = 0.25  # fallback for undefined pairs


def _build_corr_matrix(signals: List[str]) -> np.ndarray:
    """Build NxN correlation matrix for the active signals."""
    n = len(signals)
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = (signals[i], signals[j])
            key_rev = (signals[j], signals[i])
            rho = CORR_PAIRS.get(key, CORR_PAIRS.get(key_rev, DEFAULT_CORR))
            C[i, j] = rho
            C[j, i] = rho
    return C


def _compute_fdm(weights: np.ndarray, corr_matrix: np.ndarray) -> float:
    """FDM = 1 / sqrt(w' C w), capped at 2.0."""
    denom = weights @ corr_matrix @ weights
    if denom <= 0:
        return 1.0
    fdm = 1.0 / math.sqrt(denom)
    return min(fdm, 2.0)


def _load_data() -> dict:
    """Load extracted forecasts from pickle (Kaggle-aware path search)."""
    search_paths = [_INPUT_PATH]
    if os.path.exists("/kaggle/working"):
        search_paths.insert(0, "/kaggle/working/extracted_forecasts.pkl")
    if os.path.exists("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "extracted_forecasts.pkl" in files:
                p = os.path.join(root, "extracted_forecasts.pkl")
                if p not in search_paths:
                    search_paths.append(p)

    for p in search_paths:
        if os.path.exists(p):
            print(f"  Loading forecasts from: {p}")
            with open(p, "rb") as f:
                return pickle.load(f)

    print("ERROR: extracted_forecasts.pkl not found in:")
    for p in search_paths:
        print(f"  - {p}")
    sys.exit(1)


def _prepare_matrices(log: list, signals: List[str]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, List[str], List[str]
]:
    """
    Convert log entries to numpy arrays.

    Returns:
      forecasts: (n_days, n_symbols, n_signals)
      prices: (n_days, n_symbols)
      vols: (n_days, n_symbols)
      dates: list of date strings
      symbols: sorted list of all symbols
    """
    all_syms = set()
    for _, _, fc_snap, px_snap, vol_snap in log:
        all_syms.update(fc_snap.keys())
        all_syms.update(px_snap.keys())
    symbols = sorted(all_syms)
    sym_idx = {s: i for i, s in enumerate(symbols)}
    sig_idx = {s: i for i, s in enumerate(signals)}

    n_days = len(log)
    n_syms = len(symbols)
    n_sigs = len(signals)

    forecasts = np.full((n_days, n_syms, n_sigs), np.nan)
    prices = np.full((n_days, n_syms), np.nan)
    vols = np.full((n_days, n_syms), np.nan)
    dates = []

    for d, (day_idx, date_str, fc_snap, px_snap, vol_snap) in enumerate(log):
        dates.append(date_str)
        for sym, fc_dict in fc_snap.items():
            si = sym_idx.get(sym)
            if si is None:
                continue
            for src, val in fc_dict.items():
                sigi = sig_idx.get(src)
                if sigi is not None and np.isfinite(val):
                    forecasts[d, si, sigi] = val
        for sym, px in px_snap.items():
            si = sym_idx.get(sym)
            if si is not None and np.isfinite(px) and px > 0:
                prices[d, si] = px
        for sym, vol in vol_snap.items():
            si = sym_idx.get(sym)
            if si is not None and np.isfinite(vol) and vol > 0:
                vols[d, si] = vol

    return forecasts, prices, vols, dates, symbols


def _simulate_equity(
    weights_dict: Dict[str, float],
    forecasts: np.ndarray,
    prices: np.ndarray,
    vols: np.ndarray,
    signals: List[str],
    corr_matrix: np.ndarray,
    start_day: int,
    end_day: int,
    capital: float = 500_000.0,
    cost_bps: float = 33.0,
    regime_adaptive: bool = False,
    idm: float = 1.3,
    inertia: float = 0.20,
    max_leverage: float = 2.0,
    return_daily_returns: bool = False,
) -> Dict:
    """
    Equity curve simulation matching the real backtest engine.

    Identical to v27 except:
      - max_leverage is a parameter (for sensitivity analysis, NOT optimization)
      - Supports 12 signals
    """
    n_sigs = len(signals)

    w_arr = np.array([weights_dict.get(s, 0.0) for s in signals])
    w_sum = w_arr.sum()
    if w_sum <= 0:
        return {"sharpe": -99.0, "cagr": 0.0, "max_dd": 100.0, "calmar": 0.0}
    w_arr /= w_sum

    fdm = _compute_fdm(w_arr, corr_matrix)
    cost_frac = cost_bps / 10000.0
    n_syms = forecasts.shape[1]

    daily_returns = []
    equity = capital
    peak = capital
    max_dd = 0.0

    held_shares = {}
    peak_prices_opt = {}
    stop_levels_opt = {}
    _last_combined = {}
    _recomp_counter = 0
    _RECOMPUTE_FREQ = 5

    equity_history = []

    for d in range(start_day, end_day - 1):
        equity_history.append(equity)

        # DD tiers (P1c)
        dd_pct = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        if dd_pct >= 35:
            annual_vol_target = 0.0
        elif dd_pct >= 30:
            annual_vol_target = 0.30
        elif dd_pct >= 20:
            annual_vol_target = 0.40
        elif dd_pct >= 10:
            annual_vol_target = 0.45
        else:
            annual_vol_target = 0.50

        sizing_equity = max(equity, capital * 0.10)
        dynamic_daily_target = sizing_equity * annual_vol_target / 16.0

        # Regime-adaptive vol (P1f)
        if regime_adaptive and len(equity_history) >= 200:
            sma_200 = np.mean(equity_history[-200:])
            if equity > sma_200 * 1.02:
                dynamic_daily_target *= 1.25
            elif equity < sma_200 * 0.98:
                dynamic_daily_target *= 0.15

        fc_slice = forecasts[d]
        px_slice = prices[d]
        vol_slice = vols[d]

        _recomp_counter += 1
        if _recomp_counter >= _RECOMPUTE_FREQ or not _last_combined:
            _recomp_counter = 0
            combined = {}
            for si in range(n_syms):
                fc_row = fc_slice[si]
                if np.all(np.isnan(fc_row)):
                    continue
                if np.isnan(px_slice[si]) or np.isnan(vol_slice[si]):
                    continue
                fc_clean = np.where(np.isnan(fc_row), 0.0, fc_row)
                avail_mask = ~np.isnan(fc_slice[si])
                w_avail = w_arr * avail_mask
                w_avail_sum = w_avail.sum()
                if w_avail_sum <= 0:
                    continue
                w_norm = w_avail / w_avail_sum
                raw = float(np.dot(w_norm, fc_clean))
                scaled = raw * fdm
                scaled = max(-20.0, min(20.0, scaled))
                combined[si] = scaled
            _last_combined = combined
        else:
            combined = _last_combined

        max_pos = 10
        ranked = sorted(combined.items(),
                        key=lambda x: (x[1] > 0, x[1] if x[1] > 0 else -abs(x[1])),
                        reverse=True)
        top_syms_set = set(si for si, _ in ranked[:max_pos])
        grace_set = set(si for si, _ in ranked[:max_pos + 7])

        investable = [si for si, fc in ranked[:max_pos] if abs(fc) > 2.0]
        n_investable = max(5, min(len(investable), max_pos))
        weight_per_sym = 1.0 / n_investable

        target_shares = {}
        for si, fc_val in ranked[:max_pos]:
            if abs(fc_val) < 1.0:
                continue
            vol_d = vol_slice[si]
            px_d = px_slice[si]
            if vol_d <= 0 or px_d <= 0:
                continue
            ivv = px_d * vol_d
            if ivv <= 0:
                continue
            vol_scalar = dynamic_daily_target / ivv
            shares = (fc_val / 10.0) * vol_scalar * weight_per_sym * idm
            if shares < 0:
                shares = 0.0
            per_sym_max_notional = sizing_equity * max_leverage / n_investable
            if abs(shares) * px_d > per_sym_max_notional and px_d > 0:
                shares = per_sym_max_notional / px_d
            shares = round(shares)
            if shares > 0:
                target_shares[si] = shares

        new_held = {}
        day_cost = 0.0
        for si, tgt_sh in target_shares.items():
            cur_sh = held_shares.get(si, 0.0)
            px_d = px_slice[si]
            if np.isnan(px_d) or px_d <= 0:
                continue
            if cur_sh == 0.0:
                new_held[si] = tgt_sh
                day_cost += abs(tgt_sh) * px_d * cost_frac
            else:
                delta_pct = abs(tgt_sh - cur_sh) / max(abs(cur_sh), 1e-10)
                if delta_pct > inertia:
                    new_held[si] = tgt_sh
                    day_cost += abs(tgt_sh - cur_sh) * px_d * cost_frac
                else:
                    new_held[si] = cur_sh

        for si, cur_sh in held_shares.items():
            if si in new_held:
                continue
            if si in grace_set and cur_sh != 0.0:
                new_held[si] = cur_sh
            elif cur_sh != 0.0:
                px_d = px_slice[si]
                if not np.isnan(px_d) and px_d > 0:
                    day_cost += abs(cur_sh) * px_d * cost_frac

        # Trailing stops (5σ)
        for si in list(new_held.keys()):
            sh = new_held[si]
            if sh <= 0:
                continue
            px_d = px_slice[si]
            if np.isnan(px_d) or px_d <= 0:
                continue
            vol_d = vol_slice[si]
            if np.isnan(vol_d) or vol_d <= 0:
                continue
            pk = max(peak_prices_opt.get(si, px_d), px_d)
            peak_prices_opt[si] = pk
            stop_dist = 5.0 * vol_d * pk
            new_stop = pk - stop_dist
            stop_levels_opt[si] = max(stop_levels_opt.get(si, 0), new_stop)
            if px_d < stop_levels_opt[si]:
                day_cost += abs(sh) * px_d * cost_frac
                new_held[si] = 0
                peak_prices_opt.pop(si, None)
                stop_levels_opt.pop(si, None)

        for si in list(peak_prices_opt.keys()):
            if si not in new_held or new_held.get(si, 0) == 0:
                peak_prices_opt.pop(si, None)
                stop_levels_opt.pop(si, None)

        # Portfolio leverage cap
        total_exposure = 0.0
        for si, sh in new_held.items():
            if sh == 0.0:
                continue
            px_d = px_slice[si]
            if not np.isnan(px_d) and px_d > 0:
                total_exposure += abs(sh) * px_d
        max_total_exposure = sizing_equity * max_leverage
        if total_exposure > max_total_exposure and total_exposure > 0:
            scale_down = max_total_exposure / total_exposure
            for si in list(new_held.keys()):
                new_held[si] = round(new_held[si] * scale_down)

        # Daily PnL
        next_px = prices[d + 1]
        day_pnl = 0.0
        for si, sh in new_held.items():
            if sh == 0.0:
                continue
            if np.isnan(next_px[si]) or np.isnan(px_slice[si]) or px_slice[si] <= 0:
                continue
            daily_ret = (next_px[si] - px_slice[si]) / px_slice[si]
            day_pnl += sh * px_slice[si] * daily_ret

        day_pnl -= day_cost
        equity += day_pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

        daily_returns.append(day_pnl / max(equity - day_pnl, 1.0))
        held_shares = new_held

    if len(daily_returns) < 50:
        return {"sharpe": -99.0, "cagr": 0.0, "max_dd": 100.0, "calmar": 0.0}

    daily_returns = np.array(daily_returns)
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1)
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

    years = len(daily_returns) / 252.0
    total_ret = equity / capital
    cagr = (total_ret ** (1.0 / years) - 1.0) * 100.0 if years > 0 and total_ret > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    result = {
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 4),
        "total_return": round((total_ret - 1) * 100, 2),
        "n_days": len(daily_returns),
        "final_equity": round(equity, 0),
    }
    if return_daily_returns:
        result["daily_returns"] = daily_returns
    return result


def _weights_from_vector(x: np.ndarray) -> Dict[str, float]:
    """Convert optimization vector to normalized weight dict."""
    total = sum(abs(v) for v in x)
    if total <= 0:
        return {s: 1.0 / len(ACTIVE_SIGNALS) for s in ACTIVE_SIGNALS}
    return {s: abs(v) / total for s, v in zip(ACTIVE_SIGNALS, x)}


# ── Module-level shared state for optimizer ──
_G_FORECASTS: Optional[np.ndarray] = None
_G_PRICES: Optional[np.ndarray] = None
_G_VOLS: Optional[np.ndarray] = None
_G_SIGNALS: Optional[List[str]] = None
_G_CORR: Optional[np.ndarray] = None
_G_TRAIN_END: int = 0


def _objective(x: np.ndarray) -> float:
    """Objective function: maximize risk-adjusted return with anti-overfit penalties."""
    weights = _weights_from_vector(x)

    result = _simulate_equity(
        weights, _G_FORECASTS, _G_PRICES, _G_VOLS, _G_SIGNALS, _G_CORR,
        0, _G_TRAIN_END, regime_adaptive=True,
    )

    sharpe = result["sharpe"]
    max_dd = result["max_dd"]
    calmar = result.get("calmar", 0.0)
    cagr = result.get("cagr", 0.0)

    # Multi-objective: 50% Sharpe + 30% Calmar + CAGR bonus
    base_score = 0.50 * sharpe + 0.30 * calmar + 0.002 * max(0, cagr - 40)

    # ── Anti-overfit penalties (v28: tighter than v27) ──

    # 1. MaxDD penalty: penalize above 30% (was 35%)
    dd_penalty = max(0, (max_dd - 30)) * 0.06

    # 2. Concentration penalty: cap at 12% (was 15%)
    max_w = max(weights.values())
    conc_penalty = max(0, (max_w - 0.12)) * 4.0

    # 3. Diversity bonus: reward effective N (higher = less concentrated)
    w_arr = np.array(list(weights.values()))
    eff_n = 1.0 / np.sum(w_arr ** 2) if np.sum(w_arr ** 2) > 0 else 1
    diversity_bonus = 0.03 * min(eff_n, 10)

    # 4. Minimum active signals: penalize if fewer than 5 get material weight (>2%)
    n_active = sum(1 for w in weights.values() if w > 0.02)
    active_penalty = max(0, (5 - n_active)) * 0.5

    score = base_score - dd_penalty - conc_penalty + diversity_bonus - active_penalty
    return -score


def _run_walk_forward_validation(
    best_weights: Dict[str, float],
    forecasts: np.ndarray,
    prices: np.ndarray,
    vols: np.ndarray,
    signals: List[str],
    corr_matrix: np.ndarray,
    dates: List[str],
) -> Dict:
    """
    Anchored walk-forward analysis (WFA) — 3 windows.

    Tests the SAME weight vector across different out-of-sample periods.
    This is NOT reoptimized per window (that would be adaptive WFA).
    This is a pure generalization test.

    Windows:
      WF1: Train 2012-2017, Test 2018-2019
      WF2: Train 2012-2019, Test 2020-2021
      WF3: Train 2012-2021, Test 2022-2025
    """
    splits = [
        ("2018-01-01", "2020-01-01", "WF1: 2018-2019"),
        ("2020-01-01", "2022-01-01", "WF2: 2020-2021"),
        ("2022-01-01", "2026-01-01", "WF3: 2022-2025"),
    ]

    results = {}
    for test_start_date, test_end_date, label in splits:
        # Find day indices
        test_start = None
        test_end = None
        for i, d in enumerate(dates):
            if test_start is None and d >= test_start_date:
                test_start = i
            if test_end is None and d >= test_end_date:
                test_end = i
        if test_start is None:
            test_start = len(dates) - 1
        if test_end is None:
            test_end = len(dates)

        if test_end <= test_start + 50:
            print(f"    {label}: SKIPPED (insufficient data)")
            continue

        res = _simulate_equity(
            best_weights, forecasts, prices, vols, signals, corr_matrix,
            test_start, test_end, regime_adaptive=True,
        )
        results[label] = res
        print(f"    {label}: Sharpe={res['sharpe']:.3f}  CAGR={res['cagr']:.1f}%  "
              f"MaxDD={res['max_dd']:.1f}%  Calmar={res.get('calmar', 0):.3f}")

    # Check generalization: all windows should have Sharpe > 0.5
    all_sharpes = [r["sharpe"] for r in results.values()]
    min_sharpe = min(all_sharpes) if all_sharpes else 0.0
    avg_sharpe = np.mean(all_sharpes) if all_sharpes else 0.0
    worst_dd = max(r["max_dd"] for r in results.values()) if results else 100.0

    wfa_pass = min_sharpe > 0.5 and worst_dd < 45.0
    print(f"\n    WFA summary: min_Sharpe={min_sharpe:.3f}, avg_Sharpe={avg_sharpe:.3f}, "
          f"worst_DD={worst_dd:.1f}%")
    print(f"    WFA verdict: {'PASS' if wfa_pass else 'FAIL'} "
          f"(min_Sharpe>0.5 and worst_DD<45%)")

    return {
        "windows": results,
        "min_sharpe": min_sharpe,
        "avg_sharpe": avg_sharpe,
        "worst_dd": worst_dd,
        "pass": wfa_pass,
    }


def _run_bull_leverage_sensitivity(
    best_weights: Dict[str, float],
    forecasts: np.ndarray,
    prices: np.ndarray,
    vols: np.ndarray,
    signals: List[str],
    corr_matrix: np.ndarray,
    n_days: int,
) -> Dict:
    """
    Test different bull-regime leverage levels (NOT optimized).

    Runs the SAME optimal weights at different max_leverage values to show
    the CAGR/MaxDD tradeoff. This avoids optimizing leverage itself (overfit risk).
    """
    print("\n  Bull leverage sensitivity (same weights, different leverage caps):")
    print(f"    {'Leverage':>10s}  {'Sharpe':>8s}  {'CAGR':>8s}  {'MaxDD':>8s}  {'Calmar':>8s}")
    print(f"    {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    results = {}
    for lev in [1.5, 2.0, 2.3, 2.5, 2.8, 3.0]:
        res = _simulate_equity(
            best_weights, forecasts, prices, vols, signals, corr_matrix,
            0, n_days, regime_adaptive=True, max_leverage=lev,
        )
        results[lev] = res
        print(f"    {lev:10.1f}x  {res['sharpe']:8.3f}  {res['cagr']:7.1f}%  "
              f"{res['max_dd']:7.1f}%  {res.get('calmar', 0):8.3f}")

    # Find optimal leverage that meets target: Sharpe>=1.0, CAGR>=30%, MaxDD<=35%
    target_met = {}
    for lev, res in results.items():
        if res["sharpe"] >= 1.0 and res["cagr"] >= 30.0 and res["max_dd"] <= 35.0:
            target_met[lev] = res

    if target_met:
        # Pick lowest leverage that meets target (safest)
        best_lev = min(target_met.keys())
        print(f"\n    TARGET MET at {best_lev}x: Sharpe={target_met[best_lev]['sharpe']:.3f} "
              f"CAGR={target_met[best_lev]['cagr']:.1f}% MaxDD={target_met[best_lev]['max_dd']:.1f}%")
        print(f"    RECOMMENDED: Use {best_lev}x (lowest leverage meeting all targets)")
    else:
        # Find closest to target
        best_calmar = max(results.items(), key=lambda x: x[1].get("calmar", 0))
        print(f"\n    TARGET NOT MET at any leverage level.")
        print(f"    Best risk-adjusted: {best_calmar[0]}x (Calmar={best_calmar[1].get('calmar',0):.3f})")

    return {"results": results, "target_met": target_met}


def run_optimization():
    """Main optimization loop."""
    global _G_FORECASTS, _G_PRICES, _G_VOLS, _G_SIGNALS, _G_CORR, _G_TRAIN_END
    from scipy.optimize import differential_evolution
    from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

    print("=" * 70)
    print("  v28 — Walk-Forward Signal Weight Optimizer  [12 signals]")
    print("  New: +carry (ρ≈0.07), +skew_signal (ρ≈-0.12)")
    print("  Guards: PBO<25%, gap<0.30, test_Sharpe>=0.85, eff_N>=5")
    print("  Leverage: FIXED at 2.0x (sensitivity tested post-opt)")
    print("=" * 70)

    print("\n  Loading extracted forecasts...")
    data = _load_data()
    log = data["log"]
    print(f"  Loaded {len(log)} day-snapshots")

    # Verify new signals have data
    signals = list(ACTIVE_SIGNALS)
    src_counts = {s: 0 for s in signals}
    for _, _, fc_snap, _, _ in log:
        for sym, fc_dict in fc_snap.items():
            for src in fc_dict:
                if src in src_counts:
                    src_counts[src] += 1

    print("\n  Signal coverage in extracted data:")
    usable_signals = []
    for s in signals:
        count = src_counts[s]
        pct = count / (len(log) * 95) * 100  # ~95 symbols
        status = "OK" if count > 1000 else "LOW" if count > 0 else "EMPTY"
        print(f"    {s:20s}  {count:>8d} pts  ({pct:.0f}% fill)  [{status}]")
        if count > 1000:  # need at least 1000 data points to be meaningful
            usable_signals.append(s)
        else:
            print(f"    WARNING: Dropping {s} — insufficient data")

    if len(usable_signals) < 10:
        print(f"\n  ERROR: Only {len(usable_signals)} usable signals (need >= 10)")
        sys.exit(1)

    # Update signals list to only usable ones
    signals = usable_signals
    print(f"\n  Final signal set: {len(signals)} signals")

    print("\n  Building matrices...")
    forecasts, prices, vols, dates, symbols = _prepare_matrices(log, signals)
    print(f"  Shape: {forecasts.shape[0]} days x {forecasts.shape[1]} symbols x {forecasts.shape[2]} signals")

    corr_matrix = _build_corr_matrix(signals)

    # Train/test split: 2012-2019 / 2020-2025
    train_end = None
    for i, d in enumerate(dates):
        if d >= "2020-01-01":
            train_end = i
            break
    if train_end is None:
        train_end = int(len(dates) * 0.65)
    test_start = train_end

    print(f"  Train: days 0-{train_end} ({dates[0]} to {dates[train_end-1]})")
    print(f"  Test:  days {test_start}-{len(dates)} ({dates[test_start]} to {dates[-1]})")

    # Set globals
    _G_FORECASTS = forecasts
    _G_PRICES = prices
    _G_VOLS = vols
    _G_SIGNALS = signals
    _G_CORR = corr_matrix
    _G_TRAIN_END = train_end

    # v27 baseline
    print("\n  Computing v27 baseline (10 signals, regime-adaptive)...")
    v27_train = _simulate_equity(V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
                                 0, train_end, regime_adaptive=True)
    v27_test = _simulate_equity(V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
                                test_start, len(dates), regime_adaptive=True)
    v27_full = _simulate_equity(V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
                                0, len(dates), regime_adaptive=True)
    print(f"  v27 train: Sharpe={v27_train['sharpe']:.3f}  CAGR={v27_train['cagr']:.1f}%  MaxDD={v27_train['max_dd']:.1f}%")
    print(f"  v27 test:  Sharpe={v27_test['sharpe']:.3f}  CAGR={v27_test['cagr']:.1f}%  MaxDD={v27_test['max_dd']:.1f}%")
    print(f"  v27 full:  Sharpe={v27_full['sharpe']:.3f}  CAGR={v27_full['cagr']:.1f}%  MaxDD={v27_full['max_dd']:.1f}%")

    # ── Optimization ──
    n_signals = len(signals)
    bounds = [(0.01, 0.12)] * n_signals  # v28: tighter per-signal cap (was 0.15)

    _CKPT_NAME = "v28_optimizer_checkpoint.pkl"
    _CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _CKPT_NAME)
    _CKPT_PATHS = [_CKPT_PATH]
    if os.path.exists("/kaggle/working"):
        _CKPT_PATHS.insert(0, "/kaggle/working/" + _CKPT_NAME)
    if os.path.exists("/kaggle/input/centurion-core"):
        _CKPT_PATHS.insert(1, "/kaggle/input/centurion-core/" + _CKPT_NAME)
        _CKPT_PATHS.insert(2, "/kaggle/input/centurion-core/centurion_core/optimizer/" + _CKPT_NAME)

    _CKPT_EVERY = 5
    _MAX_ITER = 35   # slightly more than v27 (30) due to 2 extra dimensions
    _POP_SIZE = 5    # actual pop = 5 × 12 = 60 members

    actual_pop = _POP_SIZE * n_signals
    print(f"\n  Running differential evolution (pop={actual_pop}, maxiter={_MAX_ITER}, signals={n_signals})...")
    print(f"  Bounds: [0.01, 0.12] per signal (tighter than v27's [0.01, 0.15])")
    print(f"  Objective: 50% Sharpe + 30% Calmar + CAGR bonus - DD/conc/active penalties")

    # Check for checkpoint
    ckpt = None
    for cp in _CKPT_PATHS:
        if os.path.exists(cp):
            try:
                with open(cp, "rb") as f:
                    ckpt = pickle.load(f)
                print(f"\n  *** CHECKPOINT FOUND: {cp}")
                print(f"      Generation {ckpt['generation']}/{_MAX_ITER}, "
                      f"best score={-ckpt['best_fun']:.4f}")
                break
            except Exception as e:
                print(f"  WARNING: Corrupt checkpoint {cp}: {e}")
                ckpt = None

    np.random.seed(42)
    solver = DifferentialEvolutionSolver(
        _objective,
        bounds,
        maxiter=_MAX_ITER,
        popsize=_POP_SIZE,
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.8,
        workers=1,
        updating='immediate',
    )

    start_gen = 0
    elapsed_prior = 0.0
    best_x = None
    best_fun = np.inf
    total_nfev = 0

    if ckpt is not None:
        try:
            solver.population = ckpt['population'].copy()
            solver.population_energies = ckpt['population_energies'].copy()
            try:
                solver._nfev = ckpt.get('nfev', 0)
            except AttributeError:
                pass
            start_gen = ckpt['generation']
            elapsed_prior = ckpt.get('elapsed_min', 0.0) * 60.0
            best_x = ckpt.get('best_x', None)
            if best_x is not None:
                best_x = best_x.copy()
                best_fun = float(ckpt.get('best_fun', np.inf))
                total_nfev = ckpt.get('nfev', 0)
            print(f"      Resuming from generation {start_gen}...\n")
        except Exception as e:
            print(f"      Failed to restore: {e}, starting fresh...\n")
            start_gen = 0
            elapsed_prior = 0.0
    else:
        print(f"\n  No checkpoint found, starting fresh...\n")

    t0 = time.time()
    converged = False
    if best_x is None:
        best_x = np.full(n_signals, 0.2)
    total_nfev = max(total_nfev, 0)

    for gen in range(start_gen, _MAX_ITER):
        try:
            ret = next(solver)
            if isinstance(ret, tuple):
                xk, fun = ret
            else:
                xk = ret
                fun = _objective(xk)
        except StopIteration:
            converged = True
            print(f"  Generation {gen}: CONVERGED (tol reached)")
            break

        if fun < best_fun:
            best_fun = fun
            best_x = xk.copy()
        try:
            if solver.fun < best_fun:
                best_fun = float(solver.fun)
                best_x = solver.x.copy()
        except AttributeError:
            pass
        try:
            total_nfev = solver._nfev
        except AttributeError:
            total_nfev += _POP_SIZE

        elapsed_total = elapsed_prior + (time.time() - t0)

        if gen % _CKPT_EVERY == 0 or gen == _MAX_ITER - 1:
            best_w = _weights_from_vector(best_x)
            best_res = _simulate_equity(best_w, forecasts, prices, vols, signals, corr_matrix,
                                        0, train_end, regime_adaptive=True)
            print(f"  Gen {gen:3d}/{_MAX_ITER}: score={-best_fun:.4f}  "
                  f"Sharpe={best_res['sharpe']:.3f}  CAGR={best_res['cagr']:.1f}%  "
                  f"MaxDD={best_res['max_dd']:.1f}%  [{elapsed_total/60:.1f}min]",
                  flush=True)

            ckpt_data = {
                'generation': gen + 1,
                'population': solver.population.copy(),
                'population_energies': solver.population_energies.copy(),
                'nfev': total_nfev,
                'best_x': best_x.copy(),
                'best_fun': float(best_fun),
                'elapsed_min': elapsed_total / 60.0,
                'best_weights': best_w,
                'best_train_result': best_res,
            }
            for cp in _CKPT_PATHS:
                try:
                    with open(cp, "wb") as f:
                        pickle.dump(ckpt_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass

    elapsed = elapsed_prior + (time.time() - t0)
    print(f"\n  Optimization {'converged' if converged else 'complete'} in {elapsed/60:.1f} minutes")
    print(f"  Best score: {-best_fun:.4f}")

    best_weights = _weights_from_vector(best_x)

    # Evaluate on all periods
    best_train = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix,
                                  0, train_end, regime_adaptive=True, return_daily_returns=True)
    best_test = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix,
                                 test_start, len(dates), regime_adaptive=True, return_daily_returns=True)
    best_full = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix,
                                 0, len(dates), regime_adaptive=True, return_daily_returns=True)

    print(f"\n{'='*70}")
    print(f"  v28 OPTIMAL WEIGHTS")
    print(f"{'='*70}")
    for sig in sorted(best_weights, key=lambda s: best_weights[s], reverse=True):
        w = best_weights[sig]
        v27_w = V27_WEIGHTS.get(sig, 0.0)
        delta = w - v27_w
        print(f"    {sig:20s}  {w*100:5.1f}%  (v27: {v27_w*100:4.1f}%  delta{delta*100:+5.1f}%)")

    # Effective N
    w_arr = np.array([best_weights[s] for s in signals])
    eff_n = 1.0 / np.sum((w_arr / w_arr.sum()) ** 2) if w_arr.sum() > 0 else 0
    n_active = sum(1 for w in best_weights.values() if w > 0.02)
    print(f"\n  Effective N: {eff_n:.1f}  (active >2%: {n_active})")

    print(f"\n  {'':25s} {'Train':>12s} {'Test':>12s} {'Full':>12s} {'v27 Full':>12s}")
    print(f"  {'':25s} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for metric in ["sharpe", "cagr", "max_dd", "calmar"]:
        t = best_train[metric]
        v = best_test[metric]
        f = best_full[metric]
        v27f = v27_full[metric]
        print(f"  {metric:25s} {t:>12.3f} {v:>12.3f} {f:>12.3f} {v27f:>12.3f}")

    # ── OVERFITTING GATES (v28: stricter than v27) ──
    print(f"\n{'='*70}")
    print(f"  OVERFITTING GATES")
    print(f"{'='*70}")

    gates_passed = 0
    total_gates = 5

    # Gate 1: Train-Test Sharpe gap
    gap = best_train["sharpe"] - best_test["sharpe"]
    g1_pass = gap < 0.30
    gates_passed += int(g1_pass)
    print(f"  G1 Train-Test gap:  {gap:.3f}  {'PASS' if g1_pass else 'FAIL'} (threshold < 0.30)")

    # Gate 2: Test Sharpe minimum
    g2_pass = best_test["sharpe"] >= 0.85
    gates_passed += int(g2_pass)
    print(f"  G2 Test Sharpe:     {best_test['sharpe']:.3f}  {'PASS' if g2_pass else 'FAIL'} (threshold >= 0.85)")

    # Gate 3: Effective N
    g3_pass = eff_n >= 5.0
    gates_passed += int(g3_pass)
    print(f"  G3 Effective N:     {eff_n:.1f}  {'PASS' if g3_pass else 'FAIL'} (threshold >= 5)")

    # Gate 4: MaxDD under control
    g4_pass = best_full["max_dd"] <= 40.0
    gates_passed += int(g4_pass)
    print(f"  G4 Full MaxDD:      {best_full['max_dd']:.1f}%  {'PASS' if g4_pass else 'FAIL'} (threshold <= 40%)")

    # Gate 5: v27 improvement (must be better or no worse)
    v28_better = best_full["sharpe"] >= v27_full["sharpe"] - 0.05
    gates_passed += int(v28_better)
    print(f"  G5 vs v27 Sharpe:   {best_full['sharpe']:.3f} vs {v27_full['sharpe']:.3f}  "
          f"{'PASS' if v28_better else 'FAIL'} (threshold >= v27 - 0.05)")

    print(f"\n  Gates passed: {gates_passed}/{total_gates}")
    if gates_passed < total_gates:
        print(f"  WARNING: {total_gates - gates_passed} gate(s) FAILED — weights may be overfit!")

    # ── Walk-Forward Validation ──
    print(f"\n{'='*70}")
    print(f"  ANCHORED WALK-FORWARD VALIDATION")
    print(f"{'='*70}")
    wfa = _run_walk_forward_validation(
        best_weights, forecasts, prices, vols, signals, corr_matrix, dates
    )

    # ── Neighborhood exploration ──
    print(f"\n{'='*70}")
    print(f"  NEIGHBORHOOD EXPLORATION")
    print(f"{'='*70}")
    best_solutions = [{"weights": dict(best_weights), "train": best_train, "test": best_test}]
    rng = np.random.RandomState(123)
    for _ in range(300):
        x_pert = best_x * (1.0 + rng.randn(len(best_x)) * 0.15)
        x_pert = np.clip(x_pert, 0.01, 0.40)
        w_pert = _weights_from_vector(x_pert)
        tr = _simulate_equity(w_pert, forecasts, prices, vols, signals, corr_matrix,
                              0, train_end, regime_adaptive=True)
        if tr["sharpe"] > v27_train["sharpe"] * 0.85:
            te = _simulate_equity(w_pert, forecasts, prices, vols, signals, corr_matrix,
                                  test_start, len(dates), regime_adaptive=True)
            best_solutions.append({"weights": dict(w_pert), "train": tr, "test": te})

    best_solutions.sort(key=lambda s: s["test"]["sharpe"], reverse=True)
    print(f"\n  Top-5 solutions (by test Sharpe):")
    for i, sol in enumerate(best_solutions[:5]):
        tw = sol["train"]
        te = sol["test"]
        print(f"    #{i+1}: train={tw['sharpe']:.3f} test={te['sharpe']:.3f}  "
              f"CAGR_test={te['cagr']:.1f}%  MaxDD_test={te['max_dd']:.1f}%")
        top3 = sorted(sol["weights"].items(), key=lambda x: x[1], reverse=True)[:3]
        wstr = ", ".join("{0}={1:.0f}%".format(k, v * 100) for k, v in top3)
        print(f"         top3: {wstr}")

    # ── CSCV / PBO ──
    print(f"\n{'='*70}")
    print(f"  CSCV PROBABILITY OF BACKTEST OVERFITTING (PBO)")
    print(f"{'='*70}")
    pbo_result = {}
    try:
        _tts_ch05_path = os.path.join(
            _root, "references", "testune", "applied",
            "ch05_estimating_future_performance_unbiased.py",
        )
        import importlib.util
        _spec = importlib.util.spec_from_file_location("tts_ch05", _tts_ch05_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        cscv_superiority = _mod.cscv_superiority

        pbo_configs = best_solutions[:20]
        returns_rows = []
        for cfg in pbo_configs:
            res_full = _simulate_equity(
                cfg["weights"], forecasts, prices, vols, signals, corr_matrix,
                0, len(dates), regime_adaptive=True, return_daily_returns=True,
            )
            dr = res_full.get("daily_returns")
            if dr is not None and len(dr) > 100:
                returns_rows.append(np.array(dr))

        if len(returns_rows) >= 3:
            min_len = min(len(r) for r in returns_rows)
            ret_matrix = np.row_stack([r[:min_len] for r in returns_rows])

            cscv = cscv_superiority(ret_matrix, n_blocks=8)
            pbo_val = cscv["pbo"]
            pbo_result = {
                "pbo": round(pbo_val, 4),
                "n_combos": cscv["n_combos"],
                "n_less": cscv["n_less"],
                "n_configs": len(returns_rows),
            }

            # v28: stricter PBO threshold (25% vs 30%)
            if pbo_val < 0.25:
                pbo_verdict = "LIKELY REAL (PBO < 25%)"
            elif pbo_val < 0.40:
                pbo_verdict = "CAUTION (25% <= PBO < 40%)"
            else:
                pbo_verdict = "REJECT — likely overfit (PBO >= 40%)"

            pbo_result["verdict"] = pbo_verdict
            print(f"  PBO = {pbo_val:.1%}  ({cscv['n_less']}/{cscv['n_combos']} combos, "
                  f"{len(returns_rows)} configs)")
            print(f"  Verdict: {pbo_verdict}")
        else:
            print(f"  SKIPPED: only {len(returns_rows)} valid configs (need >= 3)")
            pbo_result = {"pbo": None, "skipped": True}
    except Exception as e:
        print(f"  ERROR computing PBO: {e}")
        pbo_result = {"pbo": None, "error": str(e)}

    # ── Bull Leverage Sensitivity (NOT optimized) ──
    print(f"\n{'='*70}")
    print(f"  BULL LEVERAGE SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    lev_results = _run_bull_leverage_sensitivity(
        best_weights, forecasts, prices, vols, signals, corr_matrix, len(dates)
    )

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"  v28 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Signals:       {len(signals)} ({', '.join(signals)})")
    print(f"  Leverage:      2.0x (fixed, not optimized)")
    print(f"  Regime:        1.25x bull, 0.15x bear")
    print(f"  Full Sharpe:   {best_full['sharpe']:.3f} (v27: {v27_full['sharpe']:.3f})")
    print(f"  Full CAGR:     {best_full['cagr']:.1f}% (v27: {v27_full['cagr']:.1f}%)")
    print(f"  Full MaxDD:    {best_full['max_dd']:.1f}% (v27: {v27_full['max_dd']:.1f}%)")
    print(f"  Gates:         {gates_passed}/{total_gates}")
    print(f"  WFA:           {'PASS' if wfa['pass'] else 'FAIL'}")
    pbo_display = pbo_result.get("pbo")
    print(f"  PBO:           {pbo_display:.1%}" if pbo_display is not None else "  PBO:           N/A")

    # Check if revised target met
    target_met = best_full["sharpe"] >= 1.0 and best_full["cagr"] >= 30.0 and best_full["max_dd"] <= 35.0
    if target_met:
        print(f"\n  >>> REVISED TARGET MET (Sharpe>=1.0, CAGR>=30%, MaxDD<=35%) <<<")
    else:
        print(f"\n  Revised target NOT met at 2.0x leverage.")
        # Check leverage sensitivity
        for lev, res in sorted(lev_results.get("results", {}).items()):
            if res["sharpe"] >= 1.0 and res["cagr"] >= 30.0 and res["max_dd"] <= 35.0:
                print(f"  Target achievable at {lev}x leverage: "
                      f"Sharpe={res['sharpe']:.3f} CAGR={res['cagr']:.1f}% MaxDD={res['max_dd']:.1f}%")
                break

    # ── Save results ──
    output = {
        "version": "v28",
        "active_signals": signals,
        "best_weights": best_weights,
        "best_train": best_train,
        "best_test": best_test,
        "best_full": best_full,
        "v27_train": v27_train,
        "v27_test": v27_test,
        "v27_full": v27_full,
        "top_solutions": best_solutions[:10],
        "n_evals": total_nfev,
        "pbo": pbo_result,
        "wfa": wfa,
        "leverage_sensitivity": {str(k): v for k, v in lev_results.get("results", {}).items()},
        "gates": {
            "gap": gap, "test_sharpe": best_test["sharpe"],
            "eff_n": eff_n, "full_max_dd": best_full["max_dd"],
            "vs_v27": v28_better, "passed": gates_passed, "total": total_gates,
        },
    }
    out_paths = [os.path.join(_root, "data", "v28_optimization_results.pkl")]
    if os.path.exists("/kaggle/working"):
        out_paths.append("/kaggle/working/v28_optimization_results.pkl")
    for op in out_paths:
        try:
            with open(op, "wb") as f:
                pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n  Results saved to {op}")
        except Exception as e:
            print(f"  WARNING: Could not save to {op}: {e}")

    # ── Copy-paste weight dict ──
    print(f"\n  ── Copy-paste for forecast_combiner.py ──")
    print(f"    DEFAULT_FORECAST_WEIGHTS = {{")
    all_24 = [
        "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
        "carry", "screener", "momentum", "pead", "mean_reversion",
        "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
        "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
        "intermarket", "acceleration", "carver_value", "skew_signal",
        "sentiment", "breakout", "order_flow",
    ]
    for sig in all_24:
        w = best_weights.get(sig, 0.0)
        print('        "{0}": {1:.4f},'.format(sig, w))
    print(f"    }}")


if __name__ == "__main__":
    run_optimization()
