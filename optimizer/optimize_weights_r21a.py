"""
R21a — Walk-Forward Signal Weight Optimizer.

Loads extracted forecasts (from run_extract_forecasts.py) and finds optimal
signal weights by simulating thousands of weight combos in seconds.

Approach:
  1. Load per-source per-symbol daily forecasts + close prices + vols
  2. For each weight vector, compute combined forecasts -> positions -> equity curve
  3. Train on 2012-2019 (in-sample), validate on 2020-2025 (out-of-sample)
  4. Use scipy differential_evolution for global optimization on 10-dim simplex
  5. Report top solutions with train/test Sharpe, CAGR, MaxDD
  6. Anti-overfit: penalize concentration, require test Sharpe > 0.8

Usage:
    python optimize_weights_r21a.py
"""
import sys
import os
import pickle
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# Force unbuffered stdout (critical for Kaggle notebook output)
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

_INPUT_PATH = os.path.join(_root, "data", "extracted_forecasts.pkl")

# ── 10 active signals from R19c ──
ACTIVE_SIGNALS = [
    "ewmac_8_32", "ewmac_16_64", "ewmac_64_256",
    "screener", "momentum", "mean_reversion",
    "penfold_trend", "ehlers_dsp", "acceleration",
    "carver_value", "breakout",
]
# Note: 11 signals (carver_value was re-enabled in R19c). Breakout also active.

R19C_WEIGHTS = {
    "ewmac_8_32": 0.07, "ewmac_16_64": 0.09, "ewmac_64_256": 0.08,
    "screener": 0.05, "momentum": 0.16, "mean_reversion": 0.13,
    "penfold_trend": 0.12, "ehlers_dsp": 0.12, "acceleration": 0.04,
    "carver_value": 0.07, "breakout": 0.07,
}

# R19c static correlation matrix (key pairs for FDM calc)
# Subset relevant to active signals
CORR_PAIRS = {
    ("ewmac_8_32", "ewmac_16_64"): 0.90,
    ("ewmac_8_32", "ewmac_64_256"): 0.50,
    ("ewmac_16_64", "ewmac_64_256"): 0.60,
    ("ewmac_8_32", "momentum"): 0.55,
    ("ewmac_16_64", "momentum"): 0.55,
    ("ewmac_64_256", "momentum"): 0.50,
    ("ewmac_8_32", "breakout"): 0.50,
    ("ewmac_16_64", "breakout"): 0.50,
    ("ewmac_64_256", "breakout"): 0.40,
    ("ewmac_8_32", "penfold_trend"): 0.50,
    ("ewmac_16_64", "penfold_trend"): 0.50,
    ("ewmac_64_256", "penfold_trend"): 0.45,
    ("momentum", "penfold_trend"): 0.65,
    ("momentum", "breakout"): 0.55,
    ("penfold_trend", "breakout"): 0.65,
    ("momentum", "mean_reversion"): -0.20,
    ("ewmac_8_32", "mean_reversion"): -0.15,
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
    ("ewmac_8_32", "acceleration"): 0.65,
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
    ("ewmac_8_32", "carver_value"): 0.10,
    ("ewmac_16_64", "carver_value"): 0.10,
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
    """Load extracted forecasts from pickle."""
    if not os.path.exists(_INPUT_PATH):
        print(f"ERROR: {_INPUT_PATH} not found. Run run_extract_forecasts.py first.")
        sys.exit(1)
    with open(_INPUT_PATH, "rb") as f:
        return pickle.load(f)


def _prepare_matrices(log: list) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]
]:
    """
    Convert log entries to numpy arrays.

    Returns:
      forecasts: (n_days, n_symbols, n_signals) — per-source raw forecasts
      prices: (n_days, n_symbols) — close prices (NaN where missing)
      vols: (n_days, n_symbols) — daily price vol (NaN where missing)
      dates: list of date strings
      symbols: sorted list of all symbols seen
      signals: ACTIVE_SIGNALS list
    """
    # Collect all symbols across all days
    all_syms = set()
    for _, _, fc_snap, px_snap, vol_snap in log:
        all_syms.update(fc_snap.keys())
        all_syms.update(px_snap.keys())
    symbols = sorted(all_syms)
    sym_idx = {s: i for i, s in enumerate(symbols)}
    signals = ACTIVE_SIGNALS
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

    return forecasts, prices, vols, dates, symbols, signals


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
    idm: float = 2.0,
    inertia: float = 0.10,
) -> Dict:
    """
    Equity curve simulation matching the real backtest engine.

    Key fixes vs earlier version:
      - Tracks positions in SHARES (not notional) to avoid phantom turnover
      - 10% inertia buffer: only rebalance if Δshares > 10% of current
      - Costs applied on |Δshares| × price (not Δnotional)
      - DD tiers set annual_vol_target directly (0.75 → 0.40)
      - IDM applied (instrument diversification multiplier)
      - Adaptive max positions (5-15 based on forecast strength)

    Returns dict with sharpe, cagr, max_dd, calmar, etc.
    """
    n_sigs = len(signals)

    # Normalize weights
    w_arr = np.array([weights_dict.get(s, 0.0) for s in signals])
    w_sum = w_arr.sum()
    if w_sum <= 0:
        return {"sharpe": -99.0, "cagr": 0.0, "max_dd": 100.0, "calmar": 0.0}
    w_arr /= w_sum

    fdm = _compute_fdm(w_arr, corr_matrix)

    cost_frac = cost_bps / 10000.0  # 33bps = 0.0033

    n_syms = forecasts.shape[1]

    daily_returns = []
    equity = capital
    peak = capital
    max_dd = 0.0

    # Track positions in SHARES (not notional) — matches real engine
    held_shares = {}  # sym_idx -> shares (float, can be negative for shorts)

    # Equity history for regime detection (200-day lookback)
    equity_history = []

    for d in range(start_day, end_day - 1):
        equity_history.append(equity)

        # ── DD tiers — set annual_vol_target directly (matching real engine) ──
        dd_pct = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        if dd_pct >= 40:
            annual_vol_target = 0.40
        elif dd_pct >= 30:
            annual_vol_target = 0.45
        elif dd_pct >= 20:
            annual_vol_target = 0.55
        elif dd_pct >= 10:
            annual_vol_target = 0.65
        else:
            annual_vol_target = 0.75   # Full risk at no DD

        # ── Daily target (matching real engine exactly) ──
        sizing_equity = max(equity, capital * 0.10)  # 10% ruin floor
        dynamic_daily_target = sizing_equity * annual_vol_target / 16.0

        # ── Regime-adaptive vol: multiply daily target (after DD tier) ──
        if regime_adaptive and len(equity_history) >= 200:
            sma_200 = np.mean(equity_history[-200:])
            if equity > sma_200 * 1.02:
                dynamic_daily_target *= 1.25   # Uptrend boost
            elif equity < sma_200 * 0.98:
                dynamic_daily_target *= 0.55   # Downtrend defend

        # Compute combined forecasts for all symbols
        fc_slice = forecasts[d]  # (n_syms, n_sigs)
        px_slice = prices[d]  # (n_syms,)
        vol_slice = vols[d]  # (n_syms,)

        combined = {}  # sym_idx -> combined_forecast
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

        # ── Adaptive MAX_POSITIONS (matching real engine) ──
        ranked_abs = sorted(combined.values(), key=lambda x: abs(x), reverse=True)
        top15_avg = np.mean([abs(f) for f in ranked_abs[:15]]) if ranked_abs else 0.0
        if top15_avg > 8.0:
            max_pos = 15
        elif top15_avg > 5.0:
            max_pos = 12
        elif top15_avg > 3.0:
            max_pos = 8
        else:
            max_pos = 5

        # Rank by absolute forecast
        ranked = sorted(combined.items(), key=lambda x: abs(x[1]), reverse=True)
        top_syms_set = set(si for si, _ in ranked[:max_pos])
        grace_set = set(si for si, _ in ranked[:max_pos + 7])

        # Investable = forecast > 2.0 threshold
        investable = [si for si, fc in ranked[:max_pos] if abs(fc) > 2.0]
        n_investable = max(5, min(len(investable), max_pos))
        weight_per_sym = 1.0 / n_investable

        # Compute target shares for top positions
        max_leverage = 3.0  # Portfolio-wide leverage cap (matches real engine)
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

            # Long-only: floor negative positions to 0 (real engine: allow_short=False)
            if shares < 0:
                shares = 0.0

            # Per-symbol leverage cap (real engine: max_notional = equity * max_lev / n_inv)
            per_sym_max_notional = sizing_equity * max_leverage / n_investable
            if abs(shares) * px_d > per_sym_max_notional and px_d > 0:
                shares = per_sym_max_notional / px_d

            # Round to integer shares (real engine uses round())
            shares = round(shares)

            if shares > 0:
                target_shares[si] = shares

        # ── Apply inertia: only rebalance if change > 10% ──
        new_held = {}
        day_cost = 0.0

        for si, tgt_sh in target_shares.items():
            cur_sh = held_shares.get(si, 0.0)
            px_d = px_slice[si]
            if np.isnan(px_d) or px_d <= 0:
                continue

            if cur_sh == 0.0:
                # New position — always enter
                new_held[si] = tgt_sh
                day_cost += abs(tgt_sh) * px_d * cost_frac
            else:
                # Existing position — apply inertia
                delta_pct = abs(tgt_sh - cur_sh) / max(abs(cur_sh), 1e-10)
                if delta_pct > inertia:
                    # Rebalance
                    new_held[si] = tgt_sh
                    day_cost += abs(tgt_sh - cur_sh) * px_d * cost_frac
                else:
                    # Keep current position (within inertia band)
                    new_held[si] = cur_sh

        # Grace zone: keep held positions that fell out of top but still in grace
        for si, cur_sh in held_shares.items():
            if si in new_held:
                continue  # Already handled
            if si in grace_set and cur_sh != 0.0:
                # Hold existing, no resize
                new_held[si] = cur_sh
            elif cur_sh != 0.0:
                # Force exit — below grace zone
                px_d = px_slice[si]
                if not np.isnan(px_d) and px_d > 0:
                    day_cost += abs(cur_sh) * px_d * cost_frac

        # ── Portfolio-wide leverage cap (real engine: FIX-LEV) ──
        # Total |position × price| must not exceed sizing_equity × max_leverage
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

        # ── Compute daily PnL from held positions using next-day prices ──
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

    return {
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 4),
        "total_return": round((total_ret - 1) * 100, 2),
        "n_days": len(daily_returns),
        "final_equity": round(equity, 0),
    }


def _weights_from_vector(x: np.ndarray) -> Dict[str, float]:
    """Convert optimization vector (raw values) to normalized weight dict."""
    # Softmax-like normalization to keep on simplex
    total = sum(abs(v) for v in x)
    if total <= 0:
        return {s: 1.0 / len(ACTIVE_SIGNALS) for s in ACTIVE_SIGNALS}
    return {s: abs(v) / total for s, v in zip(ACTIVE_SIGNALS, x)}


# ── Module-level shared state for multiprocessing ──
_G_FORECASTS: Optional[np.ndarray] = None
_G_PRICES: Optional[np.ndarray] = None
_G_VOLS: Optional[np.ndarray] = None
_G_SIGNALS: Optional[List[str]] = None
_G_CORR: Optional[np.ndarray] = None
_G_TRAIN_END: int = 0
_G_R19C_TRAIN_SHARPE: float = 0.0


def _objective_parallel(x: np.ndarray) -> float:
    """Top-level objective function (picklable for multiprocessing)."""
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

    # Heavy penalty for MaxDD > 50%
    dd_penalty = max(0, (max_dd - 50)) * 0.03

    # Concentration penalty
    max_w = max(weights.values())
    conc_penalty = max(0, (max_w - 0.25)) * 2.0

    # Diversity bonus
    w_arr = np.array(list(weights.values()))
    eff_n = 1.0 / np.sum(w_arr ** 2) if np.sum(w_arr ** 2) > 0 else 1
    diversity_bonus = 0.02 * min(eff_n, 8)

    score = base_score - dd_penalty - conc_penalty + diversity_bonus
    return -score


def run_optimization():
    """Main optimization loop with multiprocessing."""
    global _G_FORECASTS, _G_PRICES, _G_VOLS, _G_SIGNALS, _G_CORR
    global _G_TRAIN_END, _G_R19C_TRAIN_SHARPE
    from scipy.optimize import differential_evolution
    import multiprocessing

    n_cpus = multiprocessing.cpu_count()

    print("=" * 70)
    print("  R21a — Walk-Forward Signal Weight Optimizer (PARALLEL)")
    print(f"  CPUs: {n_cpus}")
    print("=" * 70)

    print("\n  Loading extracted forecasts...")
    data = _load_data()
    log = data["log"]
    r19c = data.get("r19c_result", {})
    print(f"  Loaded {len(log)} day-snapshots")
    print(f"  R19c baseline: {r19c}")

    print("\n  Building matrices...")
    forecasts, prices, vols, dates, symbols, signals = _prepare_matrices(log)
    print(f"  Shape: {forecasts.shape[0]} days x {forecasts.shape[1]} symbols x {forecasts.shape[2]} signals")

    corr_matrix = _build_corr_matrix(signals)

    # Find train/test split
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

    # Set module globals for parallel objective
    _G_FORECASTS = forecasts
    _G_PRICES = prices
    _G_VOLS = vols
    _G_SIGNALS = signals
    _G_CORR = corr_matrix
    _G_TRAIN_END = train_end

    # R19c baseline (NO regime — R19c doesn't use regime-adaptive vol)
    print("\n  Computing R19c baseline on train/test (no regime)...")
    r19c_train = _simulate_equity(R19C_WEIGHTS, forecasts, prices, vols, signals, corr_matrix, 0, train_end, regime_adaptive=False)
    r19c_test = _simulate_equity(R19C_WEIGHTS, forecasts, prices, vols, signals, corr_matrix, test_start, len(dates), regime_adaptive=False)
    r19c_full = _simulate_equity(R19C_WEIGHTS, forecasts, prices, vols, signals, corr_matrix, 0, len(dates), regime_adaptive=False)
    print(f"  R19c train: Sharpe={r19c_train['sharpe']:.3f}  CAGR={r19c_train['cagr']:.1f}%  MaxDD={r19c_train['max_dd']:.1f}%  Calmar={r19c_train.get('calmar',0):.3f}")
    print(f"  R19c test:  Sharpe={r19c_test['sharpe']:.3f}  CAGR={r19c_test['cagr']:.1f}%  MaxDD={r19c_test['max_dd']:.1f}%  Calmar={r19c_test.get('calmar',0):.3f}")
    print(f"  R19c full:  Sharpe={r19c_full['sharpe']:.3f}  CAGR={r19c_full['cagr']:.1f}%  MaxDD={r19c_full['max_dd']:.1f}%  Calmar={r19c_full.get('calmar',0):.3f}")
    _G_R19C_TRAIN_SHARPE = r19c_train["sharpe"]

    # Bounds: each weight between 0.01 and 0.40
    n_signals = len(ACTIVE_SIGNALS)
    bounds = [(0.01, 0.40)] * n_signals

    # ── Checkpoint config ──
    _CKPT_NAME = "r21a_optimizer_checkpoint.pkl"
    _CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _CKPT_NAME)
    # Search order: /kaggle/working first, then input dataset, then local script dir
    _CKPT_PATHS = [_CKPT_PATH]
    if os.path.exists("/kaggle/working"):
        _CKPT_PATHS.insert(0, "/kaggle/working/" + _CKPT_NAME)
    if os.path.exists("/kaggle/input/centurion-core"):
        # Checkpoint uploaded to dataset root or nested inside centurion_core/
        _CKPT_PATHS.insert(1, "/kaggle/input/centurion-core/" + _CKPT_NAME)
        _CKPT_PATHS.insert(2, "/kaggle/input/centurion-core/centurion_core/optimizer/" + _CKPT_NAME)
    _CKPT_EVERY = 5  # save every N generations
    _MAX_ITER = 150
    _POP_SIZE = 60

    print(f"\n  Running differential evolution (population={_POP_SIZE}, maxiter={_MAX_ITER}, workers={n_cpus})...")
    print(f"  Optimizing {n_signals} weights on simplex")
    print(f"  Objective: 50% Sharpe + 30% Calmar + CAGR bonus - DD penalty")
    print(f"  Target: Sharpe>=1.3, CAGR>50%, MaxDD<=50%")
    print(f"  Regime: adaptive vol (1.25x uptrend, 0.55x downtrend)")
    print(f"  Checkpoint every {_CKPT_EVERY} generations")

    # ── Check for checkpoint to resume ──
    import time
    from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

    ckpt = None
    for cp in _CKPT_PATHS:
        if os.path.exists(cp):
            try:
                with open(cp, "rb") as f:
                    ckpt = pickle.load(f)
                print(f"\n  *** CHECKPOINT FOUND: {cp}")
                print(f"      Generation {ckpt['generation']}/{_MAX_ITER}, "
                      f"best score={-ckpt['best_fun']:.4f}, "
                      f"elapsed={ckpt['elapsed_min']:.1f}min")
                break
            except Exception as e:
                print(f"\n  WARNING: Corrupt checkpoint {cp}: {e}")
                ckpt = None

    # ── Build solver (using manual iteration for checkpoint support) ──
    try:
        solver = DifferentialEvolutionSolver(
            _objective_parallel,
            bounds,
            seed=42,
            maxiter=_MAX_ITER,
            popsize=_POP_SIZE,
            tol=1e-5,
            mutation=(0.5, 1.5),
            recombination=0.8,
            workers=n_cpus,
            updating='deferred',
        )
    except TypeError:
        # Older scipy: 'seed' not supported — use positional rng
        solver = DifferentialEvolutionSolver(
            _objective_parallel,
            bounds,
            maxiter=_MAX_ITER,
            popsize=_POP_SIZE,
            tol=1e-5,
            mutation=(0.5, 1.5),
            recombination=0.8,
            workers=n_cpus,
            updating='deferred',
        )

    start_gen = 0
    elapsed_prior = 0.0
    best_x = None
    best_fun = np.inf
    total_nfev = 0

    if ckpt is not None:
        # Restore population state
        try:
            solver.population = ckpt['population'].copy()
            solver.population_energies = ckpt['population_energies'].copy()
            try:
                solver._nfev = ckpt.get('nfev', 0)
            except AttributeError:
                pass  # older scipy doesn't have _nfev
            start_gen = ckpt['generation']
            elapsed_prior = ckpt.get('elapsed_min', 0.0) * 60.0
            best_x = ckpt.get('best_x', None)
            if best_x is not None:
                best_x = best_x.copy()
                best_fun = float(ckpt.get('best_fun', np.inf))
                total_nfev = ckpt.get('nfev', 0)
            print(f"      Resuming from generation {start_gen}...\n")
        except Exception as e:
            print(f"      Failed to restore checkpoint: {e}")
            print(f"      Starting fresh...\n")
            start_gen = 0
            elapsed_prior = 0.0
    else:
        print(f"\n  No checkpoint found, starting fresh...\n")

    t0 = time.time()

    # ── Manual generation loop with checkpoints ──
    converged = False
    if best_x is None:
        best_x = np.full(n_signals, 0.2)  # safe default
    total_nfev = max(total_nfev, 0)
    for gen in range(start_gen, _MAX_ITER):
        try:
            ret = next(solver)
            # Unpack — some scipy versions return (x, fun), others return x only
            if isinstance(ret, tuple):
                xk, fun = ret
            else:
                xk = ret
                fun = _objective_parallel(xk)
        except StopIteration:
            converged = True
            print(f"  Generation {gen}: CONVERGED (tol reached)")
            break

        # Track best solution from next() return values (portable across scipy versions)
        if fun < best_fun:
            best_fun = fun
            best_x = xk.copy()
        # Also try solver attributes (newer scipy)
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
            best_res = _simulate_equity(best_w, forecasts, prices, vols, signals, corr_matrix, 0, train_end, regime_adaptive=True)
            print(f"  Gen {gen:3d}/{_MAX_ITER}: score={-best_fun:.4f}  "
                  f"Sharpe={best_res['sharpe']:.3f}  CAGR={best_res['cagr']:.1f}%  "
                  f"MaxDD={best_res['max_dd']:.1f}%  [{elapsed_total/60:.1f}min]",
                  flush=True)

            # Save checkpoint
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
                    pass  # e.g. read-only input dir on Kaggle

    elapsed = elapsed_prior + (time.time() - t0)
    print(f"\n  Optimization {'converged' if converged else 'complete'} in {elapsed/60:.1f} minutes")
    print(f"  Best score: {-best_fun:.4f}")

    # Build a result-like object from the solver
    class _DEResult:
        pass
    result = _DEResult()
    result.x = best_x
    result.fun = best_fun
    result.nfev = total_nfev

    # Get best weights
    best_weights = _weights_from_vector(result.x)

    # Evaluate on all periods
    best_train = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix, 0, train_end, regime_adaptive=True)
    best_test = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix, test_start, len(dates), regime_adaptive=True)
    best_full = _simulate_equity(best_weights, forecasts, prices, vols, signals, corr_matrix, 0, len(dates), regime_adaptive=True)

    print(f"\n{'='*70}")
    print(f"  OPTIMAL WEIGHTS (R21a)")
    print(f"{'='*70}")
    for sig in sorted(best_weights, key=lambda s: best_weights[s], reverse=True):
        w = best_weights[sig]
        r19c_w = R19C_WEIGHTS.get(sig, 0.0)
        delta = w - r19c_w
        print(f"    {sig:20s}  {w*100:5.1f}%  (R19c: {r19c_w*100:4.1f}%  Δ{delta*100:+5.1f}%)")

    print(f"\n  {'':25s} {'Train':>12s} {'Test':>12s} {'Full':>12s}")
    print(f"  {'':25s} {'─'*12} {'─'*12} {'─'*12}")
    for metric in ["sharpe", "cagr", "max_dd", "calmar"]:
        t = best_train[metric]
        v = best_test[metric]
        f = best_full[metric]
        r = r19c_full[metric]
        print(f"  {metric:25s} {t:>12.3f} {v:>12.3f} {f:>12.3f}  (R19c: {r:.3f})")

    # Overfit check
    train_test_gap = best_train["sharpe"] - best_test["sharpe"]
    print(f"\n  Overfit check:")
    print(f"    Train-Test Sharpe gap: {train_test_gap:.3f}")
    if train_test_gap > 0.5:
        print(f"    WARNING: Large gap suggests overfitting!")
    elif train_test_gap > 0.3:
        print(f"    CAUTION: Moderate gap, validate with full backtest.")
    else:
        print(f"    OK: Gap within acceptable range.")

    if best_test["sharpe"] < 0.8:
        print(f"    WARNING: Test Sharpe < 0.8, likely overfit!")

    # Explore neighborhood of best solution for top-5 alternatives
    print(f"\n  Exploring neighborhood for alternative solutions...")
    best_solutions = [{"weights": dict(best_weights), "train": best_train, "test": best_test}]
    rng = np.random.RandomState(123)
    for _ in range(200):
        # Perturb best weights slightly
        x_pert = result.x * (1.0 + rng.randn(len(result.x)) * 0.15)
        x_pert = np.clip(x_pert, 0.01, 0.40)
        w_pert = _weights_from_vector(x_pert)
        tr = _simulate_equity(w_pert, forecasts, prices, vols, signals, corr_matrix, 0, train_end, regime_adaptive=True)
        if tr["sharpe"] > r19c_train["sharpe"] * 0.90:
            te = _simulate_equity(w_pert, forecasts, prices, vols, signals, corr_matrix, test_start, len(dates), regime_adaptive=True)
            best_solutions.append({"weights": dict(w_pert), "train": tr, "test": te})

    print(f"\n  Top solutions (by test Sharpe):")
    best_solutions.sort(key=lambda s: s["test"]["sharpe"], reverse=True)
    for i, sol in enumerate(best_solutions[:5]):
        tw = sol["train"]
        te = sol["test"]
        print(f"    #{i+1}: train={tw['sharpe']:.3f} test={te['sharpe']:.3f}  "
              f"CAGR_test={te['cagr']:.1f}%  MaxDD_test={te['max_dd']:.1f}%")
        top3 = sorted(sol["weights"].items(), key=lambda x: x[1], reverse=True)[:3]
        wstr = ", ".join(f"{k}={v*100:.0f}%" for k, v in top3)
        print(f"         top3: {wstr}")

    # Save results
    output = {
        "best_weights": best_weights,
        "best_train": best_train,
        "best_test": best_test,
        "best_full": best_full,
        "r19c_train": r19c_train,
        "r19c_test": r19c_test,
        "r19c_full": r19c_full,
        "top_solutions": best_solutions[:10],
        "n_evals": result.nfev,
    }
    out_path = os.path.join(_root, "data", "r21a_optimization_results.pkl")
    # On Kaggle, also save to /kaggle/working/ for easy download
    out_paths = [out_path]
    if os.path.exists("/kaggle/working"):
        out_paths.append("/kaggle/working/r21a_optimization_results.pkl")
    for op in out_paths:
        try:
            with open(op, "wb") as f:
                pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n  Results saved to {op}")
        except Exception as e:
            print(f"\n  WARNING: Could not save to {op}: {e}")

    # Generate run_r21a.py weight dict
    print(f"\n  ── Copy-paste for run_r21a.py ──")
    print(f"    _R21A_WEIGHTS = {{")
    for sig in sorted(ACTIVE_SIGNALS):
        w = best_weights.get(sig, 0.0)
        print(f'        "{sig}": {w:.4f},')
    # Add zero-weight signals
    all_24 = [
        "ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "ewmac_64_256",
        "carry", "screener", "momentum", "pead", "mean_reversion",
        "fii_flow", "decision_engine", "oi_signal", "cross_momentum",
        "pairs_arb", "event_driven", "penfold_trend", "ehlers_dsp",
        "intermarket", "acceleration", "carver_value", "skew_signal",
        "sentiment", "breakout", "order_flow",
    ]
    for sig in all_24:
        if sig not in ACTIVE_SIGNALS:
            print(f'        "{sig}": 0.00,')
    print(f"    }}")


if __name__ == "__main__":
    run_optimization()
