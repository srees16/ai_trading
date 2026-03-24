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
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict:
        return {
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
        test_start = train_end + timedelta(days=1)
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
