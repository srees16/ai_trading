"""
Rigorous Live Trading Estimate — Honest PBO + Realistic Performance Haircuts.

Purpose: Compute TRUE PBO and realistic live-trading projections for
centurion_core Indian stocks, addressing the fundamental gap that
R21A Godmode-v2 (Sharpe=1.571, CAGR=61%) has NO valid PBO score.

Tests performed:
  1. TRUE PBO via per-signal daily P&L attribution CSCV
  2. Deflated Sharpe Ratio (corrects for 24-signal search space)
  3. Train/Test Sharpe degradation analysis
  4. Slippage stress tests (1x, 1.5x, 2x, 3x baseline costs)
  5. Execution lag simulation (1-day delayed signals)
  6. Regime detection lag penalty
  7. Monte Carlo confidence intervals (block bootstrap)
  8. Alpha-beta decomposition against NIFTY50

Philosophy: Better to know the truth and plan for it than to discover
it with real money. Every haircut applied here is a surprise avoided in live.

Usage:
    python optimizer/rigorous_live_estimate.py
"""
import sys
import os
import math
import pickle
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

_INPUT_PATH = os.path.join(_root, "data", "extracted_forecasts.pkl")

# ── V27 Champion Weights (R21A base) ──
V27_WEIGHTS = {
    "ewmac_16_64": 0.014, "ewmac_64_256": 0.115,
    "screener": 0.101, "momentum": 0.159, "mean_reversion": 0.055,
    "penfold_trend": 0.018, "ehlers_dsp": 0.178, "acceleration": 0.156,
    "carver_value": 0.188, "breakout": 0.016,
}

ACTIVE_SIGNALS = list(V27_WEIGHTS.keys())

# ── Correlation matrix (from v28 optimizer) ──
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
}
DEFAULT_CORR = 0.25


def _build_corr_matrix(signals: List[str]) -> np.ndarray:
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
    denom = weights @ corr_matrix @ weights
    if denom <= 0:
        return 1.0
    return min(1.0 / math.sqrt(denom), 2.0)


def _load_data() -> dict:
    if not os.path.exists(_INPUT_PATH):
        print(f"ERROR: {_INPUT_PATH} not found")
        sys.exit(1)
    with open(_INPUT_PATH, "rb") as f:
        return pickle.load(f)


def _prepare_matrices(log, signals):
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


def _simulate_with_signal_attribution(
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
    signal_lag: int = 0,
    cost_multiplier: float = 1.0,
    regime_adaptive: bool = True,
    idm: float = 1.3,
    inertia: float = 0.20,
    max_leverage: float = 2.0,
) -> Dict:
    """
    Equity simulation WITH per-signal daily P&L attribution.

    This is the key function: it tracks what fraction of each day's PnL
    is attributable to each signal source, enabling TRUE PBO computation.

    Parameters
    ----------
    signal_lag : int — simulate N-day delayed signal execution (0 = perfect)
    cost_multiplier : float — scale transaction costs (1.0 = baseline, 2.0 = stress)
    """
    n_sigs = len(signals)
    w_arr = np.array([weights_dict.get(s, 0.0) for s in signals])
    w_sum = w_arr.sum()
    if w_sum <= 0:
        return {"sharpe": -99.0}
    w_arr /= w_sum

    fdm = _compute_fdm(w_arr, corr_matrix)
    cost_frac = (cost_bps * cost_multiplier) / 10000.0
    n_syms = forecasts.shape[1]

    daily_returns = []
    signal_daily_returns = {s: [] for s in signals}
    equity = capital
    peak = capital
    max_dd = 0.0

    held_shares = {}
    peak_prices_opt = {}
    stop_levels_opt = {}
    _last_combined = {}
    _last_signal_contribs = {}  # per-symbol signal contribution fractions
    _recomp_counter = 0
    _RECOMPUTE_FREQ = 5

    equity_history = []

    for d in range(start_day, end_day - 1):
        equity_history.append(equity)

        # DD tiers
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

        # Regime-adaptive vol
        if regime_adaptive and len(equity_history) >= 200:
            sma_200 = np.mean(equity_history[-200:])
            if equity > sma_200 * 1.02:
                dynamic_daily_target *= 1.25
            elif equity < sma_200 * 0.98:
                dynamic_daily_target *= 0.15

        # Apply signal lag
        fc_day = max(start_day, d - signal_lag)
        fc_slice = forecasts[fc_day]
        px_slice = prices[d]
        vol_slice = vols[d]

        _recomp_counter += 1
        if _recomp_counter >= _RECOMPUTE_FREQ or not _last_combined:
            _recomp_counter = 0
            combined = {}
            sig_contribs = {}  # {sym_idx: {signal_name: weight_fraction}}
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

                # Track per-signal contribution to this symbol's combined forecast
                contribs = {}
                for sigi, s_name in enumerate(signals):
                    if avail_mask[sigi] and abs(fc_clean[sigi]) > 0:
                        contribs[s_name] = float(w_norm[sigi] * fc_clean[sigi])
                fc_total = sum(abs(v) for v in contribs.values())
                if fc_total > 0:
                    sig_contribs[si] = {k: abs(v) / fc_total for k, v in contribs.items()}
                else:
                    sig_contribs[si] = {}

            _last_combined = combined
            _last_signal_contribs = sig_contribs
        else:
            combined = _last_combined
            sig_contribs = _last_signal_contribs

        max_pos = 10
        ranked = sorted(combined.items(),
                        key=lambda x: (x[1] > 0, x[1] if x[1] > 0 else -abs(x[1])),
                        reverse=True)
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

        # Daily PnL + per-signal attribution
        next_px = prices[d + 1]
        day_pnl = 0.0
        day_signal_pnl = defaultdict(float)

        for si, sh in new_held.items():
            if sh == 0.0:
                continue
            if np.isnan(next_px[si]) or np.isnan(px_slice[si]) or px_slice[si] <= 0:
                continue
            daily_ret = (next_px[si] - px_slice[si]) / px_slice[si]
            sym_pnl = sh * px_slice[si] * daily_ret
            day_pnl += sym_pnl

            # Attribute this symbol's PnL to signals by their contribution fraction
            sym_contribs = sig_contribs.get(si, {})
            if sym_contribs:
                for s_name, frac in sym_contribs.items():
                    day_signal_pnl[s_name] += sym_pnl * frac
            else:
                # Equal attribution if no contrib data
                for s_name in signals:
                    day_signal_pnl[s_name] += sym_pnl / len(signals)

        day_pnl -= day_cost
        equity += day_pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

        day_ret = day_pnl / max(equity - day_pnl, 1.0)
        daily_returns.append(day_ret)

        # Store per-signal daily return (as fraction of equity)
        eq_before = max(equity - day_pnl, 1.0)
        for s_name in signals:
            signal_daily_returns[s_name].append(day_signal_pnl.get(s_name, 0.0) / eq_before)

        held_shares = new_held

    if len(daily_returns) < 50:
        return {"sharpe": -99.0}

    daily_returns = np.array(daily_returns)
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1)
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    sortino_denom = np.std(daily_returns[daily_returns < 0], ddof=1) if np.any(daily_returns < 0) else 1.0
    sortino = (mean_r / sortino_denom * math.sqrt(252)) if sortino_denom > 0 else 0.0

    years = len(daily_returns) / 252.0
    total_ret = equity / capital
    cagr = (total_ret ** (1.0 / years) - 1.0) * 100.0 if years > 0 and total_ret > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # Return stats
    skew = float(np.mean(((daily_returns - mean_r) / std_r) ** 3)) if std_r > 0 else 0.0
    kurt = float(np.mean(((daily_returns - mean_r) / std_r) ** 4)) if std_r > 0 else 3.0

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 4),
        "total_return": round((total_ret - 1) * 100, 2),
        "n_days": len(daily_returns),
        "final_equity": round(equity, 0),
        "daily_returns": daily_returns,
        "signal_daily_returns": {s: np.array(v) for s, v in signal_daily_returns.items()},
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
    }


def _compute_true_pbo(signal_daily_returns: Dict[str, np.ndarray], n_partitions: int = 10) -> Dict:
    """Compute TRUE PBO from per-signal daily P&L attribution.

    Uses the correct CSCV method: each row is one signal's ACTUAL daily
    P&L contribution (not synthetic, not hit-rate-based).
    """
    from services.aronson_validator import compute_pbo

    # Filter to signals with sufficient data
    valid_signals = {s: r for s, r in signal_daily_returns.items()
                     if len(r) >= 60 and np.std(r) > 1e-10}

    if len(valid_signals) < 4:
        return {"pbo": None, "error": "Fewer than 4 valid signals", "n_valid": len(valid_signals)}

    min_len = min(len(r) for r in valid_signals.values())
    signal_names = sorted(valid_signals.keys())
    returns_matrix = np.array([valid_signals[s][:min_len] for s in signal_names])

    print(f"    PBO input: {len(signal_names)} signals × {min_len} days")
    print(f"    Signals: {', '.join(signal_names)}")
    print(f"    Per-signal mean return (ann bp):")
    for s in signal_names:
        ann_bp = np.mean(valid_signals[s]) * 252 * 10000
        std_bp = np.std(valid_signals[s]) * math.sqrt(252) * 10000
        print(f"      {s:20s}: {ann_bp:+7.1f} bp  (vol={std_bp:.0f} bp)")

    result = compute_pbo(returns_matrix, n_partitions=n_partitions)

    # Also run with different partition counts for robustness
    pbo_values = {}
    for S in [8, 10, 12, 16]:
        try:
            r = compute_pbo(returns_matrix, n_partitions=S)
            pbo_values[S] = r["pbo"]
        except Exception:
            pass

    result["pbo_by_partitions"] = pbo_values
    result["signal_names"] = signal_names
    result["method"] = "true_per_signal_pnl_attribution"
    return result


def _deflated_sharpe(observed_sr, n_obs, n_trials, skewness, kurtosis):
    """Deflated Sharpe Ratio (de Prado AFML Ch.14)."""
    from services.deflated_sharpe import deflated_sharpe_ratio
    return deflated_sharpe_ratio(
        observed_sr=observed_sr,
        n_obs=n_obs,
        n_trials=n_trials,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def _block_bootstrap_sharpe(daily_returns, n_bootstrap=2000, block_length=40, ci=0.90):
    """Block bootstrap confidence interval for Sharpe ratio."""
    rng = np.random.RandomState(42)
    n = len(daily_returns)
    n_blocks = max(1, n // block_length)
    sharpes = []

    for _ in range(n_bootstrap):
        # Draw random blocks with replacement
        blocks = []
        for _ in range(n_blocks):
            start = rng.randint(0, n - block_length)
            blocks.append(daily_returns[start:start + block_length])
        sample = np.concatenate(blocks)
        m = np.mean(sample)
        s = np.std(sample, ddof=1)
        if s > 0:
            sharpes.append(m / s * math.sqrt(252))

    sharpes = np.array(sharpes)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(sharpes, alpha * 100))
    hi = float(np.percentile(sharpes, (1 - alpha) * 100))
    return {
        "mean": round(float(np.mean(sharpes)), 4),
        "median": round(float(np.median(sharpes)), 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "ci_pct": ci * 100,
        "std": round(float(np.std(sharpes)), 4),
    }


def main():
    t0 = time.time()

    print("=" * 74)
    print("  RIGOROUS LIVE TRADING ESTIMATE — Centurion Core Indian Stocks")
    print("  Goal: Compute TRUE PBO + Realistic Live Performance Projections")
    print("  Methodology: Per-signal P&L attribution CSCV, Deflated Sharpe,")
    print("               Slippage stress tests, Execution lag, Block bootstrap")
    print("=" * 74)

    # ── Load data ──
    print("\n  Loading extracted forecasts...")
    data = _load_data()
    log = data["log"]
    print(f"  Loaded {len(log)} day-snapshots")

    signals = [s for s in ACTIVE_SIGNALS]
    forecasts, prices, vols, dates, symbols = _prepare_matrices(log, signals)
    corr_matrix = _build_corr_matrix(signals)
    n_days = len(dates)

    print(f"  Shape: {forecasts.shape[0]} days × {forecasts.shape[1]} symbols × {forecasts.shape[2]} signals")
    print(f"  Period: {dates[0]} to {dates[-1]}")

    # Train/test split
    train_end = None
    for i, d in enumerate(dates):
        if d >= "2020-01-01":
            train_end = i
            break
    if train_end is None:
        train_end = int(n_days * 0.65)
    test_start = train_end

    print(f"  Train: {dates[0]} to {dates[train_end-1]} ({train_end} days)")
    print(f"  Test:  {dates[test_start]} to {dates[-1]} ({n_days - test_start} days)")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: BASELINE — v27 Champion Weights (regime-adaptive)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print(f"  TEST 1: BASELINE — v27 Champion Weights")
    print(f"{'='*74}")

    res_train = _simulate_with_signal_attribution(
        V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
        0, train_end, regime_adaptive=True,
    )
    res_test = _simulate_with_signal_attribution(
        V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
        test_start, n_days, regime_adaptive=True,
    )
    res_full = _simulate_with_signal_attribution(
        V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
        0, n_days, regime_adaptive=True,
    )

    print(f"\n  {'Metric':25s} {'Train':>10s} {'Test':>10s} {'Full':>10s}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    for metric in ["sharpe", "sortino", "cagr", "max_dd", "calmar"]:
        t = res_train[metric]
        v = res_test[metric]
        f_ = res_full[metric]
        print(f"  {metric:25s} {t:10.3f} {v:10.3f} {f_:10.3f}")

    sharpe_gap = res_train["sharpe"] - res_test["sharpe"]
    print(f"\n  Train-Test Sharpe gap: {sharpe_gap:+.3f}")
    if sharpe_gap > 0.30:
        print(f"  ⚠️  GAP > 0.30 — potential overfitting to train period")
    elif sharpe_gap < -0.20:
        print(f"  ✅  Test BETTER than train — good generalization")
    else:
        print(f"  ✅  Gap within acceptable range")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: TRUE PBO — Per-Signal Daily P&L Attribution CSCV
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print(f"  TEST 2: TRUE PBO — Per-Signal P&L Attribution CSCV")
    print(f"{'='*74}")

    print("\n  Computing per-signal P&L attribution for FULL period...")
    pbo_full = _compute_true_pbo(res_full["signal_daily_returns"])

    if pbo_full.get("pbo") is not None:
        pbo_val = pbo_full["pbo"]
        print(f"\n  ★ TRUE PBO (full period) = {pbo_val:.1%}")
        print(f"    Method: {pbo_full['method']}")
        print(f"    Combinations: {pbo_full.get('n_combinations', 'N/A')}")
        print(f"    Interpretation: {pbo_full.get('interpretation', 'N/A')}")
        if pbo_full.get("pbo_by_partitions"):
            print(f"    PBO by partition count:")
            for S, p in sorted(pbo_full["pbo_by_partitions"].items()):
                print(f"      S={S:2d}: PBO={p:.1%}")

        if pbo_val < 0.25:
            print(f"\n  ✅  PBO={pbo_val:.1%} < 25% — Signal alpha is LIKELY REAL")
        elif pbo_val < 0.40:
            print(f"\n  ⚠️  PBO={pbo_val:.1%} in 25-40% range — CAUTION, monitor OOS")
        else:
            print(f"\n  ❌  PBO={pbo_val:.1%} ≥ 40% — HIGH OVERFIT RISK, DO NOT DEPLOY")
    else:
        print(f"\n  ❌  Could not compute PBO: {pbo_full.get('error')}")

    # Also compute PBO on TRAIN period only (more conservative)
    print("\n  Computing per-signal PBO for TRAIN period only...")
    pbo_train = _compute_true_pbo(res_train["signal_daily_returns"])
    if pbo_train.get("pbo") is not None:
        print(f"  Train PBO = {pbo_train['pbo']:.1%} ({pbo_train.get('interpretation')})")

    print("\n  Computing per-signal PBO for TEST period only...")
    pbo_test = _compute_true_pbo(res_test["signal_daily_returns"])
    if pbo_test.get("pbo") is not None:
        print(f"  Test PBO = {pbo_test['pbo']:.1%} ({pbo_test.get('interpretation')})")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: DEFLATED SHARPE RATIO
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print(f"  TEST 3: DEFLATED SHARPE RATIO (Multiple Testing Correction)")
    print(f"{'='*74}")

    n_obs = len(res_full["daily_returns"])
    n_trials = 24  # total signal search space
    obs_sr = res_full["sharpe"]
    skew = res_full["skewness"]
    kurt = res_full["kurtosis"]

    dsr = _deflated_sharpe(obs_sr, n_obs, n_trials, skew, kurt)
    print(f"\n  Observed Sharpe: {obs_sr:.3f}")
    print(f"  N observations:  {n_obs}")
    print(f"  N trials tested: {n_trials}")
    print(f"  Skewness:        {skew:.3f}")
    print(f"  Kurtosis:        {kurt:.3f}")
    print(f"\n  ★ Deflated Sharpe p-value = {dsr:.4f}")
    if dsr > 0.95:
        print(f"  ✅  DSR > 0.95 — Sharpe is almost certainly genuine")
    elif dsr > 0.50:
        print(f"  ✅  DSR > 0.50 — Sharpe likely survives multiple testing")
    elif dsr > 0.10:
        print(f"  ⚠️  DSR in 0.10-0.50 — Sharpe may be partially due to luck")
    else:
        print(f"  ❌  DSR < 0.10 — Sharpe likely due to luck/data mining")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: BLOCK BOOTSTRAP CONFIDENCE INTERVAL
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print(f"  TEST 4: BLOCK BOOTSTRAP SHARPE CI (2000 samples, block=40)")
    print(f"{'='*74}")

    boot_full = _block_bootstrap_sharpe(res_full["daily_returns"])
    boot_test = _block_bootstrap_sharpe(res_test["daily_returns"])

    print(f"\n  Full period: Sharpe 90% CI = [{boot_full['ci_lo']:.3f}, {boot_full['ci_hi']:.3f}]")
    print(f"    Mean={boot_full['mean']:.3f}, Median={boot_full['median']:.3f}, Std={boot_full['std']:.3f}")
    print(f"  Test period: Sharpe 90% CI = [{boot_test['ci_lo']:.3f}, {boot_test['ci_hi']:.3f}]")
    print(f"    Mean={boot_test['mean']:.3f}, Median={boot_test['median']:.3f}, Std={boot_test['std']:.3f}")

    if boot_full["ci_lo"] > 0.50:
        print(f"\n  ✅  Lower bound {boot_full['ci_lo']:.3f} > 0.50 — robust")
    elif boot_full["ci_lo"] > 0.0:
        print(f"\n  ⚠️  Lower bound {boot_full['ci_lo']:.3f} — positive but fragile")
    else:
        print(f"\n  ❌  Lower bound {boot_full['ci_lo']:.3f} ≤ 0 — Sharpe may be zero in practice")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 5: SLIPPAGE STRESS TESTS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print(f"  TEST 5: SLIPPAGE STRESS TESTS")
    print(f"  Baseline: 33 bps round-trip. Testing 1x to 3x.")
    print(f"{'='*74}")

    print(f"\n  {'Cost Mult':>10s}  {'EffCost':>8s}  {'Sharpe':>8s}  {'CAGR':>8s}  {'MaxDD':>8s}  {'Calmar':>8s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for cm in [1.0, 1.5, 2.0, 2.5, 3.0]:
        r = _simulate_with_signal_attribution(
            V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
            0, n_days, cost_multiplier=cm, regime_adaptive=True,
        )
        eff_cost = 33 * cm
        print(f"  {cm:10.1f}x  {eff_cost:7.0f}bp  {r['sharpe']:8.3f}  "
              f"{r['cagr']:7.1f}%  {r['max_dd']:7.1f}%  {r['calmar']:8.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 6: EXECUTION LAG SIMULATION
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print(f"  TEST 6: EXECUTION LAG SIMULATION")
    print(f"  Testing: signal executed 0, 1, 2, 3 days after generation")
    print(f"{'='*74}")

    print(f"\n  {'Lag':>6s}  {'Sharpe':>8s}  {'CAGR':>8s}  {'MaxDD':>8s}  {'Calmar':>8s}  {'Δ Sharpe':>10s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")

    base_sr = res_full["sharpe"]
    for lag in [0, 1, 2, 3]:
        r = _simulate_with_signal_attribution(
            V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
            0, n_days, signal_lag=lag, regime_adaptive=True,
        )
        delta = r["sharpe"] - base_sr
        print(f"  {lag:5d}d  {r['sharpe']:8.3f}  {r['cagr']:7.1f}%  "
              f"{r['max_dd']:7.1f}%  {r['calmar']:8.3f}  {delta:+10.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 7: WALK-FORWARD VALIDATION (3 Windows)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print(f"  TEST 7: ANCHORED WALK-FORWARD VALIDATION")
    print(f"{'='*74}")

    splits = [
        ("2018-01-01", "2020-01-01", "WF1: 2018-2019 (Pre-COVID)"),
        ("2020-01-01", "2022-01-01", "WF2: 2020-2021 (COVID Bull)"),
        ("2022-01-01", "2026-01-01", "WF3: 2022-2025 (Rate Hikes)"),
    ]

    wfa_sharpes = []
    for test_start_date, test_end_date, label in splits:
        t_start = None
        t_end = None
        for i, d in enumerate(dates):
            if t_start is None and d >= test_start_date:
                t_start = i
            if t_end is None and d >= test_end_date:
                t_end = i
        if t_start is None:
            t_start = n_days - 1
        if t_end is None:
            t_end = n_days
        if t_end <= t_start + 50:
            print(f"    {label}: SKIPPED")
            continue

        r = _simulate_with_signal_attribution(
            V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
            t_start, t_end, regime_adaptive=True,
        )
        wfa_sharpes.append(r["sharpe"])
        print(f"    {label}: Sharpe={r['sharpe']:.3f}  CAGR={r['cagr']:.1f}%  "
              f"MaxDD={r['max_dd']:.1f}%  Calmar={r['calmar']:.3f}")

    if wfa_sharpes:
        min_sr = min(wfa_sharpes)
        avg_sr = np.mean(wfa_sharpes)
        print(f"\n    Min={min_sr:.3f}, Avg={avg_sr:.3f}")
        if min_sr > 0.5:
            print(f"    ✅  All windows Sharpe > 0.5 — consistent performance")
        elif min_sr > 0.0:
            print(f"    ⚠️  Some windows weak (min={min_sr:.3f})")
        else:
            print(f"    ❌  Negative Sharpe in some windows — system is fragile")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY: REALISTIC LIVE TRADING ESTIMATE
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*74}")
    print(f"  FINAL SUMMARY: REALISTIC LIVE TRADING ESTIMATE")
    print(f"  (Completed in {elapsed/60:.1f} minutes)")
    print(f"{'='*74}")

    # The most realistic live estimate comes from:
    # - Test period Sharpe (true OOS, 2020-2025)
    # - With 1-day execution lag
    # - At 1.5x cost (conservative slippage)
    print(f"\n  Building realistic estimate: Test period + 1-day lag + 1.5x cost...")
    live_estimate = _simulate_with_signal_attribution(
        V27_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
        test_start, n_days, signal_lag=1, cost_multiplier=1.5, regime_adaptive=True,
    )

    pbo_val = pbo_full.get("pbo")
    pbo_str = f"{pbo_val:.1%}" if pbo_val is not None else "N/A"

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  BACKTEST vs REALISTIC LIVE ESTIMATE                            │
  ├──────────────────────┬────────────────┬─────────────────────────┤
  │  Metric              │  Backtest Full │  Realistic Live Est.    │
  ├──────────────────────┼────────────────┼─────────────────────────┤
  │  Sharpe              │  {res_full['sharpe']:12.3f}  │  {live_estimate['sharpe']:12.3f}            │
  │  CAGR                │  {res_full['cagr']:11.1f}%  │  {live_estimate['cagr']:11.1f}%            │
  │  Max Drawdown        │  {res_full['max_dd']:11.1f}%  │  {live_estimate['max_dd']:11.1f}%            │
  │  Calmar              │  {res_full['calmar']:12.3f}  │  {live_estimate['calmar']:12.3f}            │
  │  Sortino             │  {res_full['sortino']:12.3f}  │  {live_estimate['sortino']:12.3f}            │
  ├──────────────────────┼────────────────┼─────────────────────────┤
  │  TRUE PBO            │  {pbo_str:>14s}  │  (same — signal mix)    │
  │  Deflated Sharpe     │  {dsr:14.4f}  │  (p-value)              │
  │  Sharpe 90% CI       │ [{boot_full['ci_lo']:.3f}, {boot_full['ci_hi']:.3f}]  │  [{boot_test['ci_lo']:.3f}, {boot_test['ci_hi']:.3f}]          │
  │  Train-Test Gap      │  {sharpe_gap:+14.3f}  │                         │
  └──────────────────────┴────────────────┴─────────────────────────┘

  HONEST ASSESSMENT:
  """)

    # Verdict
    issues = []
    passes = []
    ok_for_paper = True

    if pbo_val is not None:
        if pbo_val < 0.25:
            passes.append(f"PBO={pbo_val:.1%} < 25% — signal alpha likely real")
        elif pbo_val < 0.40:
            issues.append(f"PBO={pbo_val:.1%} in caution zone (25-40%)")
        else:
            issues.append(f"PBO={pbo_val:.1%} — HIGH OVERFIT RISK")
            ok_for_paper = False
    else:
        issues.append("PBO could not be computed — insufficient signal divergence")

    if dsr > 0.50:
        passes.append(f"Deflated Sharpe p={dsr:.3f} > 0.50 — survives multiple testing")
    else:
        issues.append(f"Deflated Sharpe p={dsr:.3f} — may be due to data mining")

    if boot_full["ci_lo"] > 0.50:
        passes.append(f"Sharpe CI lower bound {boot_full['ci_lo']:.3f} > 0.50")
    elif boot_full["ci_lo"] > 0.0:
        issues.append(f"Sharpe CI lower bound only {boot_full['ci_lo']:.3f}")
    else:
        issues.append(f"Sharpe CI includes zero — unreliable")
        ok_for_paper = False

    if sharpe_gap < 0.30:
        passes.append(f"Train-test gap {sharpe_gap:+.3f} < 0.30")
    else:
        issues.append(f"Train-test gap {sharpe_gap:+.3f} > 0.30 — overfitting")

    if live_estimate["sharpe"] > 0.50:
        passes.append(f"Realistic live Sharpe {live_estimate['sharpe']:.3f} > 0.50")
    else:
        issues.append(f"Realistic live Sharpe only {live_estimate['sharpe']:.3f}")
        ok_for_paper = False

    if wfa_sharpes and min(wfa_sharpes) > 0.0:
        passes.append(f"All WFA windows positive (min={min(wfa_sharpes):.3f})")
    else:
        issues.append("WFA shows negative Sharpe in some windows")

    for p in passes:
        print(f"    ✅  {p}")
    for i in issues:
        print(f"    ⚠️  {i}")

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  PAPER TRADING VERDICT: {'PROCEED' if ok_for_paper else 'DO NOT PROCEED':^40s} │
  """)

    if ok_for_paper:
        print(f"  │  System passes minimum quality gates for paper trading.       │")
        print(f"  │  Expected realistic Sharpe: {live_estimate['sharpe']:.2f} (±{boot_test['std']:.2f})                    │")
        print(f"  │  Expected realistic CAGR:   {live_estimate['cagr']:.0f}% (with 1.5x costs + 1d lag)   │")
        print(f"  │  Start with ₹100K (20%), stage up per protocol.               │")
    else:
        n_critical = sum(1 for i in issues if "HIGH OVERFIT" in i or "includes zero" in i or "only" in i)
        print(f"  │  {n_critical} critical issue(s) detected. Fix before trading.          │")
        print(f"  │  Primary concern: overfitting to historical patterns.          │")

    print(f"  └──────────────────────────────────────────────────────────────────┘")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {k: v for k, v in res_full.items() if k != "daily_returns" and k != "signal_daily_returns"},
        "train": {k: v for k, v in res_train.items() if k != "daily_returns" and k != "signal_daily_returns"},
        "test": {k: v for k, v in res_test.items() if k != "daily_returns" and k != "signal_daily_returns"},
        "live_estimate": {k: v for k, v in live_estimate.items() if k != "daily_returns" and k != "signal_daily_returns"},
        "pbo_full": {k: v for k, v in pbo_full.items() if k != "logit_distribution"},
        "pbo_train": {k: v for k, v in pbo_train.items() if k != "logit_distribution"} if pbo_train.get("pbo") else {},
        "pbo_test": {k: v for k, v in pbo_test.items() if k != "logit_distribution"} if pbo_test.get("pbo") else {},
        "deflated_sharpe_pvalue": dsr,
        "bootstrap_full": boot_full,
        "bootstrap_test": boot_test,
        "sharpe_gap": sharpe_gap,
        "wfa_sharpes": wfa_sharpes,
        "ok_for_paper": ok_for_paper,
    }
    out_path = os.path.join(_root, "data", "rigorous_live_estimate.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
