"""
R21a-H1 — Hybrid Regime Validation Backtest.

PURPOSE: Validate that the H1 fix (hybrid HMM × SMA200 regime scaling)
does NOT degrade OOS metrics compared to the original R21a (SMA200-only).

This ensures live paper/live trading behavior matches the optimizer's
proven backtest results before going to production.

REGIME MODES TESTED:
  A) R21a Original    — SMA200 equity regime only (1.25× up / 0.55× down)
  B) HMM-Only         — Market regime only (1.30× bull / 0.15× bear / 0.85× range / 0.35× hvol)
  C) Hybrid HMM×SMA200 — Both layers with 1.30× cap (LIVE MODE)
  D) Aggressive Hybrid — More aggressive uptrend (1.35×), same defend (0.55×)

Market regime is approximated from cross-sectional returns + vol since
the extracted_forecasts.pkl does not contain NIFTY/VIX data.

Usage:
    python -m optimizer.validate_r21a_hybrid
    # or on Kaggle:
    !python centurion_core/cloud/run_kaggle.py --task validate_hybrid
"""
import sys
import os
import pickle
import math
import time
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

from optimizer.optimize_weights_r21a import (
    ACTIVE_SIGNALS,
    R19C_WEIGHTS,
    _build_corr_matrix,
    _load_data,
    _prepare_matrices,
    _compute_fdm,
)

# ── Best R21a weights (from optimizer checkpoint) ──
R21A_WEIGHTS = {
    "ewmac_8_32": 0.196, "ewmac_16_64": 0.008, "ewmac_64_256": 0.108,
    "screener": 0.018, "momentum": 0.112, "mean_reversion": 0.027,
    "penfold_trend": 0.012, "ehlers_dsp": 0.188, "acceleration": 0.119,
    "carver_value": 0.196, "breakout": 0.016,
}

# ── HMM Regime Vol Scales (from volatility_target.py REGIME_VOL_SCALE) ──
HMM_REGIME_SCALES = {
    "trending_bull":    1.30,
    "trending_bear":    0.15,
    "range_bound":      0.85,
    "high_volatility":  0.35,
    "crisis":           0.00,
}

# ── Hybrid H1 constants (from volatility_target.py) ──
H1_SMA200_BOOST = 1.25
H1_SMA200_DEFEND = 0.55
H1_COMBINED_CAP = 1.30

# ── Checkpoint config ──
_CKPT_NAME = "r21a_hybrid_validation_checkpoint.pkl"


def _detect_market_regime(
    prices: np.ndarray,
    d: int,
    lookback: int = 60,
) -> str:
    """
    Approximate HMM-like market regime from cross-sectional price data.

    Uses:
      1. Market-wide return: mean daily return of all valid symbols (SMA50)
      2. Market-wide realized vol: cross-sectional vol of daily returns (annualized)
      3. Breadth: fraction of symbols with positive 20-day return

    Classification (matches HMM output categories):
      - trending_bull:   return > 0 AND vol < 25% AND breadth > 60%
      - trending_bear:   return < -0.05% AND (vol > 30% OR breadth < 35%)
      - range_bound:     abs(return) < 0.05% AND vol < 25%
      - high_volatility: vol > 35%
      - crisis:          return < -0.1% AND vol > 40% AND breadth < 25%
    """
    if d < lookback:
        return "range_bound"  # Not enough history

    # Get valid returns for lookback period
    daily_rets = []
    for prev_d in range(max(0, d - lookback), d):
        px_today = prices[prev_d + 1]
        px_yesterday = prices[prev_d]
        valid = ~np.isnan(px_today) & ~np.isnan(px_yesterday) & (px_yesterday > 0)
        if valid.sum() > 5:
            rets = (px_today[valid] - px_yesterday[valid]) / px_yesterday[valid]
            daily_rets.append(np.nanmean(rets))

    if len(daily_rets) < 20:
        return "range_bound"

    daily_rets = np.array(daily_rets)
    mean_ret = np.mean(daily_rets[-20:])  # 20-day average
    ann_vol = np.std(daily_rets) * math.sqrt(252) * 100  # annualized %

    # Breadth: fraction with positive 20-day return
    px_now = prices[d]
    px_20d_ago = prices[max(0, d - 20)]
    valid_20d = ~np.isnan(px_now) & ~np.isnan(px_20d_ago) & (px_20d_ago > 0)
    if valid_20d.sum() > 5:
        rets_20d = (px_now[valid_20d] - px_20d_ago[valid_20d]) / px_20d_ago[valid_20d]
        breadth = np.mean(rets_20d > 0)
    else:
        breadth = 0.5

    # Classification
    if mean_ret < -0.001 and ann_vol > 40 and breadth < 0.25:
        return "crisis"
    if ann_vol > 35:
        return "high_volatility"
    if mean_ret < -0.0005 and (ann_vol > 30 or breadth < 0.35):
        return "trending_bear"
    if mean_ret > 0 and ann_vol < 25 and breadth > 0.60:
        return "trending_bull"
    return "range_bound"


def _simulate_equity_hybrid(
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
    idm: float = 2.0,
    inertia: float = 0.10,
    regime_mode: str = "r21a_original",
    checkpoint_callback=None,
) -> Dict:
    """
    Equity curve simulation with configurable regime modes.

    regime_mode:
      - "none"           : No regime scaling (pure R19c-style)
      - "r21a_original"  : SMA200 equity curve only (1.25× / 0.55×)
      - "hmm_only"       : Market regime only (HMM scale)
      - "hybrid"         : HMM × SMA200, capped at 1.30× (LIVE MODE)
      - "aggressive"     : Higher uptrend boost (1.35×), same defend
    """
    n_sigs = len(signals)

    # Normalize weights
    w_arr = np.array([weights_dict.get(s, 0.0) for s in signals])
    w_sum = w_arr.sum()
    if w_sum <= 0:
        return _empty_result()
    w_arr /= w_sum

    fdm = _compute_fdm(w_arr, corr_matrix)
    cost_frac = cost_bps / 10000.0

    n_syms = forecasts.shape[1]

    daily_returns = []
    equity = capital
    peak = capital
    max_dd = 0.0

    held_shares = {}
    equity_history = []

    # Regime tracking for diagnostics
    regime_log = {"trending_bull": 0, "trending_bear": 0, "range_bound": 0,
                  "high_volatility": 0, "crisis": 0, "unknown": 0}
    regime_scale_sum = 0.0
    regime_count = 0

    for d in range(start_day, end_day - 1):
        equity_history.append(equity)

        # ── DD tiers (matching R21a optimizer exactly) ──
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
            annual_vol_target = 0.75

        sizing_equity = max(equity, capital * 0.10)
        dynamic_daily_target = sizing_equity * annual_vol_target / 16.0

        # ── Regime-adaptive vol scaling (configurable) ──
        regime_multiplier = 1.0

        if regime_mode == "r21a_original":
            # Original: SMA200 equity only
            if len(equity_history) >= 200:
                sma_200 = np.mean(equity_history[-200:])
                if equity > sma_200 * 1.02:
                    regime_multiplier = 1.25
                elif equity < sma_200 * 0.98:
                    regime_multiplier = 0.55

        elif regime_mode == "hmm_only":
            # HMM market regime only (approximated from price data)
            market_regime = _detect_market_regime(prices, d)
            regime_multiplier = HMM_REGIME_SCALES.get(market_regime, 1.0)
            regime_log[market_regime] = regime_log.get(market_regime, 0) + 1

        elif regime_mode == "hybrid":
            # LIVE MODE: HMM × SMA200, capped
            # Layer 1: market regime
            market_regime = _detect_market_regime(prices, d)
            hmm_scale = HMM_REGIME_SCALES.get(market_regime, 1.0)
            regime_log[market_regime] = regime_log.get(market_regime, 0) + 1

            # Layer 2: equity SMA200
            sma_scale = 1.0
            if len(equity_history) >= 200:
                sma_200 = np.mean(equity_history[-200:])
                if equity > sma_200 * 1.02:
                    sma_scale = H1_SMA200_BOOST
                elif equity < sma_200 * 0.98:
                    sma_scale = H1_SMA200_DEFEND

            # Combined with cap
            regime_multiplier = min(hmm_scale * sma_scale, H1_COMBINED_CAP)

        elif regime_mode == "aggressive":
            # Aggressive variant: higher uptrend boost
            market_regime = _detect_market_regime(prices, d)
            hmm_scale = HMM_REGIME_SCALES.get(market_regime, 1.0)
            regime_log[market_regime] = regime_log.get(market_regime, 0) + 1

            sma_scale = 1.0
            if len(equity_history) >= 200:
                sma_200 = np.mean(equity_history[-200:])
                if equity > sma_200 * 1.02:
                    sma_scale = 1.35  # More aggressive
                elif equity < sma_200 * 0.98:
                    sma_scale = 0.55

            regime_multiplier = min(hmm_scale * sma_scale, 1.40)  # Higher cap too

        # "none" → regime_multiplier stays 1.0

        dynamic_daily_target *= regime_multiplier
        regime_scale_sum += regime_multiplier
        regime_count += 1

        # ── Compute combined forecasts ──
        fc_slice = forecasts[d]
        px_slice = prices[d]
        vol_slice = vols[d]

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
            scaled = max(-20.0, min(20.0, raw * fdm))
            combined[si] = scaled

        # ── Adaptive MAX_POSITIONS ──
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

        ranked = sorted(combined.items(), key=lambda x: abs(x[1]), reverse=True)
        top_syms_set = set(si for si, _ in ranked[:max_pos])
        grace_set = set(si for si, _ in ranked[:max_pos + 7])

        investable = [si for si, fc in ranked[:max_pos] if abs(fc) > 2.0]
        n_investable = max(5, min(len(investable), max_pos))
        weight_per_sym = 1.0 / n_investable

        max_leverage = 3.0
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

        # ── Apply inertia ──
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

        # ── Portfolio leverage cap ──
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

        # ── Daily PnL ──
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

        # Periodic checkpoint
        if checkpoint_callback and (d - start_day) % 200 == 0 and d > start_day:
            checkpoint_callback(d, equity, max_dd, len(daily_returns))

    if len(daily_returns) < 50:
        return _empty_result()

    daily_returns = np.array(daily_returns)
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1)
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

    years = len(daily_returns) / 252.0
    total_ret = equity / capital
    cagr = (total_ret ** (1.0 / years) - 1.0) * 100.0 if years > 0 and total_ret > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    avg_regime = regime_scale_sum / max(regime_count, 1)

    return {
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 4),
        "total_return": round((total_ret - 1) * 100, 2),
        "n_days": len(daily_returns),
        "final_equity": round(equity, 0),
        "regime_distribution": regime_log,
        "avg_regime_multiplier": round(avg_regime, 4),
    }


def _empty_result():
    return {"sharpe": -99.0, "cagr": 0.0, "max_dd": 100.0, "calmar": 0.0,
            "total_return": 0.0, "n_days": 0, "final_equity": 0.0,
            "regime_distribution": {}, "avg_regime_multiplier": 0.0}


def _load_r21a_weights() -> Dict[str, float]:
    """Try to load best weights from checkpoint, fallback to hardcoded."""
    ckpt_paths = [
        os.path.join(_root, "data", "r21a_optimizer_checkpoint.pkl"),
    ]
    if os.path.exists("/kaggle/working"):
        ckpt_paths.insert(0, "/kaggle/working/r21a_optimizer_checkpoint.pkl")
    if os.path.exists("/kaggle/input/centurion-core"):
        ckpt_paths.insert(1, "/kaggle/input/centurion-core/r21a_optimizer_checkpoint.pkl")
        ckpt_paths.insert(2, "/kaggle/input/centurion-core/centurion_core/data/r21a_optimizer_checkpoint.pkl")

    for cp in ckpt_paths:
        if os.path.exists(cp):
            try:
                with open(cp, "rb") as f:
                    ckpt = pickle.load(f)
                w = ckpt.get("best_weights", None)
                gen = ckpt.get("generation", "?")
                if w:
                    print(f"  Loaded R21a weights from checkpoint (gen {gen}): {cp}")
                    return w
            except Exception as e:
                print(f"  WARNING: Could not load checkpoint {cp}: {e}")

    print("  Using hardcoded R21a weights")
    return dict(R21A_WEIGHTS)


def main():
    t0 = time.time()

    print("=" * 74)
    print("  R21a-H1 — HYBRID REGIME VALIDATION BACKTEST")
    print("  Goal: Confirm live hybrid HMM×SMA200 does NOT degrade OOS metrics")
    print("=" * 74)

    # ── 1. Load data ──
    print("\n  Loading extracted forecasts...")
    data = _load_data()
    log = data["log"]
    print(f"  Loaded {len(log)} day-snapshots")

    forecasts, prices, vols, dates, symbols, signals = _prepare_matrices(log)
    corr_matrix = _build_corr_matrix(signals)
    print(f"  Shape: {len(dates)} days × {len(symbols)} symbols × {len(signals)} signals")

    # ── 2. Train/test split ──
    train_end = 0
    for i, d in enumerate(dates):
        if d >= "2020-01-01":
            train_end = i
            break
    if train_end == 0:
        train_end = int(len(dates) * 0.65)
    test_start = train_end

    print(f"  Train: days 0-{train_end} ({dates[0]} to {dates[train_end - 1]})")
    print(f"  Test:  days {test_start}-{len(dates)} ({dates[test_start]} to {dates[-1]})")

    # ── 3. Load weights ──
    r21a_weights = _load_r21a_weights()
    total_w = sum(r21a_weights.values())
    print(f"\n  R21a weights ({len([w for w in r21a_weights.values() if w > 0])} active, sum={total_w:.3f}):")
    for sig in sorted(r21a_weights, key=lambda s: r21a_weights[s], reverse=True):
        w = r21a_weights[sig]
        if w > 0.001:
            print(f"    {sig:20s} = {w*100:5.1f}%")

    # ── 4. Checkpoint setup ──
    is_kaggle = os.path.exists("/kaggle/working")
    ckpt_path = "/kaggle/working/" + _CKPT_NAME if is_kaggle else os.path.join(_root, "data", _CKPT_NAME)

    # Try to load existing checkpoint
    completed_modes = {}
    try:
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                completed_modes = pickle.load(f)
            print(f"\n  Checkpoint loaded: {len(completed_modes)} modes completed:")
            for k in completed_modes:
                print(f"    ✓ {k}")
    except Exception:
        completed_modes = {}

    # ── 5. Define regime modes ──
    MODES = [
        ("A_r21a_original", "r21a_original", "R21a Original (SMA200 only) — optimizer baseline"),
        ("B_hmm_only", "hmm_only", "HMM-Only (market regime) — pre-H1 live behavior"),
        ("C_hybrid", "hybrid", "Hybrid HMM×SMA200 (cap=1.30) — POST-H1 LIVE MODE"),
        ("D_aggressive", "aggressive", "Aggressive Hybrid (1.35× boost, cap=1.40)"),
        ("E_no_regime", "none", "No Regime (R19c-style, no scaling)"),
    ]

    results = {}
    for mode_key, mode_param, description in MODES:
        if mode_key in completed_modes:
            results[mode_key] = completed_modes[mode_key]
            print(f"\n  [{mode_key}] SKIPPED (loaded from checkpoint)")
            continue

        print(f"\n{'─'*74}")
        print(f"  [{mode_key}] {description}")
        print(f"{'─'*74}")

        def _ckpt_cb(d, eq, dd, n):
            elapsed_min = (time.time() - t0) / 60
            print(f"    ... day {d}, equity={eq:.0f}, maxDD={dd:.1f}%, "
                  f"days_processed={n}, elapsed={elapsed_min:.1f}min", flush=True)

        # Train period
        print(f"  Running train period ({dates[0]} to {dates[train_end-1]})...")
        train_result = _simulate_equity_hybrid(
            r21a_weights, forecasts, prices, vols, signals, corr_matrix,
            0, train_end, regime_mode=mode_param, checkpoint_callback=_ckpt_cb,
        )
        print(f"    Train: Sharpe={train_result['sharpe']:.3f}  "
              f"CAGR={train_result['cagr']:.1f}%  "
              f"MaxDD={train_result['max_dd']:.1f}%  "
              f"Calmar={train_result['calmar']:.3f}")

        # Test period (OOS)
        print(f"  Running test period ({dates[test_start]} to {dates[-1]})...")
        test_result = _simulate_equity_hybrid(
            r21a_weights, forecasts, prices, vols, signals, corr_matrix,
            test_start, len(dates), regime_mode=mode_param, checkpoint_callback=_ckpt_cb,
        )
        print(f"    Test:  Sharpe={test_result['sharpe']:.3f}  "
              f"CAGR={test_result['cagr']:.1f}%  "
              f"MaxDD={test_result['max_dd']:.1f}%  "
              f"Calmar={test_result['calmar']:.3f}")

        # Full period
        print(f"  Running full period ({dates[0]} to {dates[-1]})...")
        full_result = _simulate_equity_hybrid(
            r21a_weights, forecasts, prices, vols, signals, corr_matrix,
            0, len(dates), regime_mode=mode_param, checkpoint_callback=_ckpt_cb,
        )
        print(f"    Full:  Sharpe={full_result['sharpe']:.3f}  "
              f"CAGR={full_result['cagr']:.1f}%  "
              f"MaxDD={full_result['max_dd']:.1f}%  "
              f"Calmar={full_result['calmar']:.3f}")

        if full_result.get("regime_distribution"):
            rd = full_result["regime_distribution"]
            total_d = sum(rd.values())
            if total_d > 0:
                print(f"    Regime distribution: ", end="")
                for regime, count in sorted(rd.items(), key=lambda x: -x[1]):
                    if count > 0:
                        print(f"{regime}={count/total_d:.0%} ", end="")
                print(f"  avg_scale={full_result['avg_regime_multiplier']:.3f}")

        gap = train_result["sharpe"] - test_result["sharpe"]
        print(f"    Train-Test gap: {gap:.3f}")

        results[mode_key] = {
            "description": description,
            "train": train_result,
            "test": test_result,
            "full": full_result,
            "gap": gap,
        }

        # Save checkpoint after each mode completes
        completed_modes[mode_key] = results[mode_key]
        try:
            with open(ckpt_path, "wb") as f:
                pickle.dump(completed_modes, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"    Checkpoint saved ({len(completed_modes)}/{len(MODES)} complete)")
        except Exception as e:
            print(f"    WARNING: Checkpoint save failed: {e}")

    # ── 6. R19c baseline (for reference) ──
    if "F_r19c_baseline" not in completed_modes:
        print(f"\n{'─'*74}")
        print(f"  [F_r19c_baseline] R19c weights + no regime (baseline comparison)")
        print(f"{'─'*74}")
        r19c_test = _simulate_equity_hybrid(
            R19C_WEIGHTS, forecasts, prices, vols, signals, corr_matrix,
            test_start, len(dates), regime_mode="none",
        )
        results["F_r19c_baseline"] = {
            "description": "R19c weights, no regime scaling",
            "test": r19c_test,
        }
        print(f"    R19c Test: Sharpe={r19c_test['sharpe']:.3f}  "
              f"CAGR={r19c_test['cagr']:.1f}%  MaxDD={r19c_test['max_dd']:.1f}%")
    else:
        results["F_r19c_baseline"] = completed_modes["F_r19c_baseline"]

    # ── 7. Comparison table ──
    elapsed = time.time() - t0
    print(f"\n\n{'='*74}")
    print(f"  HYBRID REGIME VALIDATION — COMPARISON TABLE  ({elapsed/60:.1f} min total)")
    print(f"{'='*74}")

    # Header
    print(f"\n  {'Mode':30s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'Gap':>7s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")

    # Sort by test Sharpe for easy comparison
    test_results = []
    for mode_key in ["E_no_regime", "A_r21a_original", "B_hmm_only", "C_hybrid", "D_aggressive", "F_r19c_baseline"]:
        if mode_key not in results:
            continue
        r = results[mode_key]
        test = r.get("test", {})
        if not test:
            continue
        gap = r.get("gap", 0)
        label = mode_key.split("_", 1)[1] if "_" in mode_key else mode_key
        test_results.append((label, test, gap))

    for label, test, gap in test_results:
        sharpe = test.get("sharpe", 0)
        cagr = test.get("cagr", 0)
        maxdd = test.get("max_dd", 0)
        calmar = test.get("calmar", 0)
        gap_str = f"{gap:+.3f}" if gap != 0 else "  N/A"
        print(f"  {label:30s} {sharpe:>8.3f} {cagr:>7.1f}% {maxdd:>7.1f}% {calmar:>8.3f} {gap_str:>7s}")

    # ── 8. Verdict ──
    r21a_orig = results.get("A_r21a_original", {}).get("test", {})
    hybrid = results.get("C_hybrid", {}).get("test", {})

    if r21a_orig and hybrid:
        sharpe_delta = hybrid.get("sharpe", 0) - r21a_orig.get("sharpe", 0)
        dd_delta = hybrid.get("max_dd", 0) - r21a_orig.get("max_dd", 0)
        cagr_delta = hybrid.get("cagr", 0) - r21a_orig.get("cagr", 0)
        calmar_delta = hybrid.get("calmar", 0) - r21a_orig.get("calmar", 0)

        print(f"\n  {'─'*60}")
        print(f"  DELTA: Hybrid vs R21a Original (OOS test period)")
        print(f"  {'─'*60}")
        for metric, delta in [("Sharpe", sharpe_delta), ("CAGR", cagr_delta),
                               ("MaxDD", dd_delta), ("Calmar", calmar_delta)]:
            better = "✓" if (metric == "MaxDD" and delta < 0) or (metric != "MaxDD" and delta > 0) else "✗"
            print(f"    {metric:12s}  {delta:+.3f}  {better}")

        print(f"\n{'='*74}")
        # Acceptance criteria: Sharpe not degraded by >0.3, MaxDD not increased by >5%
        if sharpe_delta >= -0.3 and dd_delta <= 5.0:
            if dd_delta < 0:
                verdict = "ACCEPT — Hybrid improves risk (lower MaxDD) with acceptable Sharpe trade-off"
            elif sharpe_delta >= 0:
                verdict = "ACCEPT — Hybrid improves or maintains all metrics"
            else:
                verdict = "ACCEPT (MARGINAL) — Minor Sharpe loss offset by risk improvement"
        elif sharpe_delta < -0.3:
            verdict = f"REVIEW — Significant Sharpe degradation ({sharpe_delta:+.3f}). Consider tuning cap."
        elif dd_delta > 5.0:
            verdict = f"REVIEW — MaxDD increased ({dd_delta:+.1f}%). Check regime detection accuracy."
        else:
            verdict = "REVIEW — Mixed results. Run full walk-forward validation."

        print(f"  VERDICT: {verdict}")
        print(f"\n  Hybrid OOS: Sharpe={hybrid.get('sharpe', 0):.3f}  "
              f"CAGR={hybrid.get('cagr', 0):.1f}%  "
              f"MaxDD={hybrid.get('max_dd', 0):.1f}%  "
              f"Calmar={hybrid.get('calmar', 0):.3f}")
        print(f"  R21a  OOS: Sharpe={r21a_orig.get('sharpe', 0):.3f}  "
              f"CAGR={r21a_orig.get('cagr', 0):.1f}%  "
              f"MaxDD={r21a_orig.get('max_dd', 0):.1f}%  "
              f"Calmar={r21a_orig.get('calmar', 0):.3f}")
        print(f"{'='*74}")

        # ── 9. Recommendation ──
        aggressive = results.get("D_aggressive", {}).get("test", {})
        if aggressive:
            agg_sharpe = aggressive.get("sharpe", 0)
            agg_dd = aggressive.get("max_dd", 0)
            if agg_sharpe > hybrid.get("sharpe", 0) + 0.1 and agg_dd <= hybrid.get("max_dd", 0) + 3:
                print(f"\n  NOTE: Aggressive mode outperforms hybrid "
                      f"(Sharpe={agg_sharpe:.3f}, MaxDD={agg_dd:.1f}%). "
                      f"Consider using higher boost constants.")
    else:
        print("\n  WARNING: Could not compare modes — missing results")

    # ── 10. Save full results ──
    results_path = ckpt_path.replace("checkpoint", "results")
    try:
        with open(results_path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n  Full results saved to {results_path}")
    except Exception as e:
        print(f"\n  WARNING: Could not save results: {e}")

    # Also save to local data/ dir
    local_results_path = os.path.join(_root, "data", "r21a_hybrid_validation_results.pkl")
    try:
        with open(local_results_path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Full results saved to {local_results_path}")
    except Exception:
        pass

    print(f"\n  Total elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
