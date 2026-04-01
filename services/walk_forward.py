"""
Rolling Walk-Forward Optimization & Validation.

Implements a rolling walk-forward harness that re-optimizes strategy
parameters quarterly and validates out-of-sample performance.

Architecture:
    Train window: 252 days (1 year)
    Test window:  63 days (1 quarter)
    Re-optimize:  Every quarter (rolling)

The harness wraps any BaseStrategy and produces OOS Sharpe, OOS return,
and a degradation ratio (OOS / IS) to detect overfitting.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Persist winning params so the live pipeline loads the latest optimals
_WF_PARAMS_DIR = Path("data") / "wf_params"
_WF_PARAMS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WalkForwardResult:
    """Result of a single walk-forward fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    in_sample_return: float = 0.0
    out_of_sample_return: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardSummary:
    """Aggregated walk-forward validation results."""
    strategy_name: str
    ticker: str
    folds: List[WalkForwardResult] = field(default_factory=list)
    avg_oos_sharpe: float = 0.0
    avg_is_sharpe: float = 0.0
    degradation_ratio: float = 0.0  # OOS/IS — <0.5 = likely overfit
    avg_oos_return: float = 0.0
    total_folds: int = 0
    wf_perm_p_value: Optional[float] = None       # WF factory permutation p-value
    wf_perm_significant: Optional[bool] = None    # Is the model factory genuinely skilled?

    def to_dict(self) -> dict:
        d = {
            "strategy": self.strategy_name,
            "ticker": self.ticker,
            "total_folds": self.total_folds,
            "avg_oos_sharpe": round(self.avg_oos_sharpe, 4),
            "avg_is_sharpe": round(self.avg_is_sharpe, 4),
            "degradation_ratio": round(self.degradation_ratio, 4),
            "avg_oos_return_pct": round(self.avg_oos_return * 100, 2),
            "folds": [
                {
                    "fold": f.fold,
                    "is_sharpe": round(f.in_sample_sharpe, 4),
                    "oos_sharpe": round(f.out_of_sample_sharpe, 4),
                    "oos_return": round(f.out_of_sample_return * 100, 2),
                    "params": f.best_params,
                }
                for f in self.folds
            ],
        }
        if self.wf_perm_p_value is not None:
            d["wf_permutation"] = {
                "p_value": round(self.wf_perm_p_value, 4),
                "significant": self.wf_perm_significant,
            }
        return d


# ─── Parameter grids for common strategies ──────────────────

_PARAM_GRIDS = {
    "macd oscillator": [
        {"ma1": 8, "ma2": 21},
        {"ma1": 10, "ma2": 21},
        {"ma1": 12, "ma2": 26},
        {"ma1": 10, "ma2": 30},
    ],
    "rsi pattern": [
        {"rsi_period": 10, "oversold": 25, "overbought": 75},
        {"rsi_period": 14, "oversold": 30, "overbought": 70},
        {"rsi_period": 14, "oversold": 35, "overbought": 65},
        {"rsi_period": 21, "oversold": 30, "overbought": 70},
    ],
    "awesome oscillator": [
        {"ao_short": 5, "ao_long": 34},
        {"ao_short": 5, "ao_long": 40},
        {"ao_short": 7, "ao_long": 34},
    ],
    "parabolic sar": [
        {"af_start": 0.02, "af_increment": 0.02, "af_max": 0.2},
        {"af_start": 0.02, "af_increment": 0.025, "af_max": 0.2},
        {"af_start": 0.015, "af_increment": 0.02, "af_max": 0.15},
    ],
    "bollinger bottom w": [
        {"bb_period": 20, "bb_std": 2.0},
        {"bb_period": 20, "bb_std": 1.5},
        {"bb_period": 25, "bb_std": 2.0},
    ],
    "heikin-ashi": [
        {"confirmation_candles": 1},
        {"confirmation_candles": 2},
    ],
    "support resistance": [
        {"n1": 2, "n2": 2, "back_candles": 30},
        {"n1": 3, "n2": 3, "back_candles": 40},
    ],
    "shooting star": [
        {"lower_bound": 0.2, "body_size": 0.5},
        {"lower_bound": 0.15, "body_size": 0.4},
    ],
    "liquidity sweep": [
        {"swing_lookback": 8, "vol_mult": 1.5, "confirmation_bars": 2},
        {"swing_lookback": 10, "vol_mult": 1.5, "confirmation_bars": 2},
        {"swing_lookback": 12, "vol_mult": 1.3, "confirmation_bars": 3},
        {"swing_lookback": 15, "vol_mult": 2.0, "confirmation_bars": 2},
    ],
    "anchored vwap": [
        {"vol_pctile": 95, "touch_pct": 0.005, "rsi_period": 14},
        {"vol_pctile": 90, "touch_pct": 0.005, "rsi_period": 14},
        {"vol_pctile": 95, "touch_pct": 0.003, "rsi_period": 10},
        {"vol_pctile": 90, "touch_pct": 0.008, "rsi_period": 21},
    ],
    "order flow imbalance": [
        {"obv_lookback": 20, "mfi_period": 14, "mfi_oversold": 40, "mfi_overbought": 60, "cvd_smooth": 10},
        {"obv_lookback": 15, "mfi_period": 10, "mfi_oversold": 35, "mfi_overbought": 65, "cvd_smooth": 8},
        {"obv_lookback": 25, "mfi_period": 14, "mfi_oversold": 30, "mfi_overbought": 70, "cvd_smooth": 12},
    ],
    "volume profile": [
        {"profile_lookback": 60, "n_bins": 50, "va_pct": 0.70, "touch_pct": 0.003},
        {"profile_lookback": 40, "n_bins": 40, "va_pct": 0.70, "touch_pct": 0.005},
        {"profile_lookback": 80, "n_bins": 60, "va_pct": 0.68, "touch_pct": 0.003},
    ],
    "swing combo": [
        {"swing_lookback": 10, "vol_mult": 1.5, "obv_lookback": 20, "mfi_period": 14,
         "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65, "min_agreement": 2},
        {"swing_lookback": 8, "vol_mult": 1.5, "obv_lookback": 15, "mfi_period": 10,
         "rsi_period": 10, "rsi_oversold": 30, "rsi_overbought": 70, "min_agreement": 2},
        {"swing_lookback": 12, "vol_mult": 1.3, "obv_lookback": 25, "mfi_period": 14,
         "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60, "min_agreement": 2},
        {"swing_lookback": 10, "vol_mult": 2.0, "obv_lookback": 20, "mfi_period": 14,
         "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65, "min_agreement": 3},
    ],
    "positional combo": [
        {"profile_lookback": 60, "n_bins": 50, "va_pct": 0.70, "vol_pctile": 95,
         "touch_pct": 0.005, "swing_lookback": 12, "vol_mult": 1.5, "min_agreement": 2},
        {"profile_lookback": 40, "n_bins": 40, "va_pct": 0.70, "vol_pctile": 90,
         "touch_pct": 0.005, "swing_lookback": 10, "vol_mult": 1.5, "min_agreement": 2},
        {"profile_lookback": 80, "n_bins": 60, "va_pct": 0.68, "vol_pctile": 95,
         "touch_pct": 0.003, "swing_lookback": 15, "vol_mult": 1.3, "min_agreement": 2},
        {"profile_lookback": 60, "n_bins": 50, "va_pct": 0.70, "vol_pctile": 95,
         "touch_pct": 0.005, "swing_lookback": 12, "vol_mult": 2.0, "min_agreement": 3},
    ],
}


def walk_forward_validate(
    strategy_cls,
    ticker: str,
    capital: float = 100_000,
    train_days: int = 252,
    test_days: int = 63,
    total_days: int = 756,   # 3 years
    param_grid: Optional[List[dict]] = None,
) -> WalkForwardSummary:
    """Run rolling walk-forward optimization for a strategy.

    Parameters
    ----------
    strategy_cls : type
        Strategy class (must be a BaseStrategy subclass).
    ticker : str
        Ticker symbol to validate.
    capital : float
        Start capital for each fold.
    train_days / test_days : int
        Window sizes in trading days.
    total_days : int
        Total lookback days for the rolling window.
    param_grid : list[dict] | None
        Parameter combinations to search. If None, uses the
        default grid for the strategy.

    Returns
    -------
    WalkForwardSummary
    """
    strategy_name = getattr(strategy_cls, "name", strategy_cls.__name__).lower()

    if param_grid is None:
        param_grid = _PARAM_GRIDS.get(strategy_name, [{}])

    end_date = date.today()
    start_date = end_date - timedelta(days=total_days)

    folds: List[WalkForwardResult] = []
    fold_idx = 0
    cursor = start_date

    # ── Build fold windows first, then execute in parallel ──
    fold_windows = []
    while cursor + timedelta(days=train_days + test_days) <= end_date:
        train_start = cursor
        train_end = cursor + timedelta(days=train_days)
        # Embargo: gap between IS and OOS to prevent feature leakage.
        # Set to 5 business days (≈ minimum positional holding period).
        embargo_days = 5
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)
        fold_windows.append((fold_idx, train_start, train_end, test_start, test_end))
        fold_idx += 1
        cursor = test_end + timedelta(days=1)

    def _run_fold(fold_args):
        _fold_idx, _train_start, _train_end, _test_start, _test_end = fold_args

        # ── In-sample: find best params ──
        _best_sharpe = -999.0
        _best_params = {}
        _best_is_return = 0.0

        for params in param_grid:
            try:
                strategy = strategy_cls()
                result = strategy.run(
                    tickers=[ticker],
                    start_date=str(_train_start),
                    end_date=str(_train_end),
                    capital=capital,
                    **params,
                )
                if not result.success:
                    continue
                metrics = result.metrics
                if isinstance(metrics, dict) and ticker in metrics:
                    metrics = metrics[ticker]
                sr = metrics.get("sharpe_ratio") or metrics.get("sharpe") or 0
                if sr > _best_sharpe:
                    _best_sharpe = sr
                    _best_params = params
                    _best_is_return = metrics.get("total_return", 0) / 100
            except Exception:
                continue

        # ── Out-of-sample: test best params ──
        _oos_sharpe = 0.0
        _oos_return = 0.0
        try:
            strategy = strategy_cls()
            result = strategy.run(
                tickers=[ticker],
                start_date=str(_test_start),
                end_date=str(_test_end),
                capital=capital,
                **_best_params,
            )
            if result.success:
                metrics = result.metrics
                if isinstance(metrics, dict) and ticker in metrics:
                    metrics = metrics[ticker]
                _oos_sharpe = metrics.get("sharpe_ratio") or metrics.get("sharpe") or 0
                _oos_return = (metrics.get("total_return", 0)) / 100

                # P0 fix: Deduct realistic transaction costs from OOS results
                # NSE round-trip: ~0.40% (STT + exchange + GST + slippage)
                # Apply per trade, estimate trades from signals count
                _total_trades = metrics.get("total_trades", 0) or metrics.get("trades", 0) or 2
                _round_trip_cost = 0.004  # 0.40% per round-trip (conservative)
                _cost_drag = _total_trades * _round_trip_cost
                _oos_return = _oos_return - _cost_drag
                # Adjust Sharpe by cost drag (annualized)
                _ann_cost = _cost_drag * (252 / max((_test_end - _test_start).days, 1))
                if _oos_sharpe > 0:
                    _oos_sharpe = max(0, _oos_sharpe - _ann_cost * 2)  # rough cost-adjusted SR
        except Exception:
            pass

        return WalkForwardResult(
            fold=_fold_idx,
            train_start=str(_train_start),
            train_end=str(_train_end),
            test_start=str(_test_start),
            test_end=str(_test_end),
            in_sample_sharpe=float(_best_sharpe),
            out_of_sample_sharpe=float(_oos_sharpe),
            in_sample_return=float(_best_is_return),
            out_of_sample_return=float(_oos_return),
            best_params=_best_params,
        )

    # Run folds in parallel (max 4 workers)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(4, len(fold_windows) or 1)) as executor:
        futures = {executor.submit(_run_fold, fw): fw for fw in fold_windows}
        for future in as_completed(futures):
            try:
                folds.append(future.result())
            except Exception:
                pass

    # Sort by fold index to maintain order
    folds.sort(key=lambda f: f.fold)

    # ── Aggregate ──
    if not folds:
        return WalkForwardSummary(
            strategy_name=strategy_name,
            ticker=ticker,
        )

    is_sharpes = [f.in_sample_sharpe for f in folds]
    oos_sharpes = [f.out_of_sample_sharpe for f in folds]
    oos_returns = [f.out_of_sample_return for f in folds]

    avg_is = float(np.mean(is_sharpes))
    avg_oos = float(np.mean(oos_sharpes))
    deg = avg_oos / avg_is if avg_is > 0 else 0.0

    return WalkForwardSummary(
        strategy_name=strategy_name,
        ticker=ticker,
        folds=folds,
        avg_oos_sharpe=avg_oos,
        avg_is_sharpe=avg_is,
        degradation_ratio=deg,
        avg_oos_return=float(np.mean(oos_returns)),
        total_folds=len(folds),
    )


# ── Gap 5: Optimal parameter persistence & loading ──────────────────

def save_optimal_params(summary: WalkForwardSummary) -> None:
    """Persist the best walk-forward params for a given strategy + ticker.

    Only saves if the strategy passed the overfitting check
    (degradation_ratio >= 0.5) and has valid folds.
    The live screener / auto-executor can load these via
    ``load_optimal_params()`` to use the latest re-optimized settings.
    """
    if not summary.folds or summary.degradation_ratio < 0.5:
        logger.info(
            "Skipping param save for %s on %s — overfit (deg=%.2f)",
            summary.strategy_name, summary.ticker, summary.degradation_ratio,
        )
        return

    # Pick the params from the fold with the best OOS Sharpe
    best_fold = max(summary.folds, key=lambda f: f.out_of_sample_sharpe)
    if not best_fold.best_params:
        return

    record = {
        "strategy": summary.strategy_name,
        "ticker": summary.ticker,
        "params": best_fold.best_params,
        "oos_sharpe": round(best_fold.out_of_sample_sharpe, 4),
        "oos_return_pct": round(best_fold.out_of_sample_return * 100, 2),
        "degradation_ratio": round(summary.degradation_ratio, 4),
        "avg_oos_sharpe": round(summary.avg_oos_sharpe, 4),
        "updated_at": datetime.now().isoformat(),
    }

    safe_name = summary.strategy_name.replace(" ", "_").replace("/", "_")
    safe_ticker = summary.ticker.replace(".", "_")
    path = _WF_PARAMS_DIR / f"{safe_name}_{safe_ticker}.json"
    path.write_text(json.dumps(record, indent=2))
    logger.info(
        "Saved optimal params for %s on %s: %s (OOS Sharpe=%.2f)",
        summary.strategy_name, summary.ticker,
        best_fold.best_params, best_fold.out_of_sample_sharpe,
    )


def load_optimal_params(strategy_name: str, ticker: str) -> Optional[Dict[str, Any]]:
    """Load the most recently saved walk-forward optimal params.

    Returns the param dict if found and still fresh (< 14 days old),
    otherwise None (caller should use default params).
    """
    safe_name = strategy_name.lower().replace(" ", "_").replace("/", "_")
    safe_ticker = ticker.replace(".", "_")
    path = _WF_PARAMS_DIR / f"{safe_name}_{safe_ticker}.json"

    if not path.exists():
        return None

    try:
        record = json.loads(path.read_text())
        updated_at = datetime.fromisoformat(record["updated_at"])
        age_days = (datetime.now() - updated_at).days
        if age_days > 14:
            logger.info(
                "Optimal params for %s on %s are %d days old — stale, skipping",
                strategy_name, ticker, age_days,
            )
            return None
        logger.info(
            "Loaded optimal params for %s on %s: %s (age=%dd)",
            strategy_name, ticker, record["params"], age_days,
        )
        return record["params"]
    except Exception as exc:
        logger.warning("Failed to load WF params from %s: %s", path, exc)
        return None


def load_all_optimal_params() -> Dict[str, Dict[str, Any]]:
    """Load all saved optimal params, keyed by 'strategy::ticker'.

    Useful for bulk inspection or the scheduler audit.
    """
    results = {}
    for path in _WF_PARAMS_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text())
            key = f"{record['strategy']}::{record['ticker']}"
            results[key] = record
        except Exception:
            continue
    return results


def wf_permutation_test(
    summary: WalkForwardSummary,
    raw_returns: np.ndarray,
    factory_fn=None,
    n_perms: int = 2000,
    train_days: int = 252,
    test_days: int = 63,
) -> WalkForwardSummary:
    """Run MC permutation test on the walk-forward model factory.

    Tests whether the WF optimization process (train → select params → test)
    produces OOS performance better than random. Shuffles the entire return
    series and repeats the WF process (Timothy Masters pp.291-293).

    Parameters
    ----------
    summary : WalkForwardSummary
        Existing WF summary to augment with permutation results.
    raw_returns : np.ndarray
        Full daily return series used for the walk-forward.
    factory_fn : callable or None
        factory_fn(train_returns) → position_vector for the test period.
        If None, a default simple mean-reversion factory is used.
    n_perms : int
        Number of MC permutation trials.
    train_days / test_days : int
        Window sizes matching the original WF.

    Returns
    -------
    WalkForwardSummary
        Same summary, with wf_perm_p_value and wf_perm_significant set.
    """
    try:
        from services.mc_permutation_test import MCPermutationTest

        raw_returns = np.asarray(raw_returns, dtype=np.float64).ravel()
        if len(raw_returns) < train_days + test_days:
            logger.warning(
                "Not enough data (%d) for WF permutation test (need %d)",
                len(raw_returns), train_days + test_days,
            )
            return summary

        # Default factory: simple momentum sign
        if factory_fn is None:
            def factory_fn(train):
                # Simple: momentum sign from training period
                if len(train) < 20:
                    return np.ones(test_days)
                momentum = np.mean(train[-20:])
                sign = 1.0 if momentum > 0 else -1.0
                return np.full(test_days, sign)

        mc = MCPermutationTest(
            n_perms=n_perms,
            center_returns=True,
            normalize_time=True,
            seed=42,
        )

        result = mc.test_walk_forward_factory(
            raw_returns, factory_fn,
            train_days=train_days,
            test_days=test_days,
        )

        summary.wf_perm_p_value = result.p_value
        summary.wf_perm_significant = result.significant

        logger.info(
            "WF permutation for %s on %s: p=%.4f, significant=%s",
            summary.strategy_name, summary.ticker,
            result.p_value, result.significant,
        )

    except Exception as exc:
        logger.warning("WF permutation test failed: %s", exc)

    return summary
