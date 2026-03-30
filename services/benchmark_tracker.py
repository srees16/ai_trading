"""
Benchmark Alpha Tracker for IND Stocks.

Computes portfolio performance metrics relative to NIFTY 50 benchmark:
  - Cumulative returns (portfolio vs benchmark)
  - Jensen's Alpha (CAPM)
  - Information Ratio
  - Tracking Error
  - Portfolio Beta vs NIFTY
  - Rolling Sharpe comparison
  - Max drawdown comparison

Used by the IntegratedScorer, screener, and portfolio dashboard to
quantify whether the system generates alpha over a passive NIFTY 50 ETF.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

RISK_FREE_RATE_ANNUAL = 0.07  # India 10-year G-Sec yield ~7%
TRADING_DAYS = 252


@dataclass
class BenchmarkComparison:
    """Result of comparing a return series against NIFTY 50."""

    ticker: str
    period_days: int
    # Returns
    portfolio_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    # Risk-adjusted
    portfolio_sharpe: float
    benchmark_sharpe: float
    jensens_alpha: float  # annualised CAPM alpha
    information_ratio: float
    tracking_error_ann: float
    # Risk
    portfolio_beta: float  # vs NIFTY
    portfolio_max_dd_pct: float
    benchmark_max_dd_pct: float
    # Advanced risk metrics (Phase 0)
    portfolio_sortino: float = 0.0
    portfolio_calmar: float = 0.0
    portfolio_omega: float = 0.0
    portfolio_cvar_95: float = 0.0
    # Meta
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def _fetch_benchmark(start: str, end: str) -> pd.Series:
    """Fetch NIFTY 50 daily close prices."""
    try:
        import yfinance as yf

        ticker = getattr(Config, "NIFTY_BENCHMARK_TICKER", "^NSEI")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty benchmark data")
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception as e:
        logger.warning("Failed to fetch NIFTY benchmark: %s", e)
        return pd.Series(dtype=float)


def _max_drawdown(cumulative: pd.Series) -> float:
    """Compute max drawdown from a cumulative return series (1-based)."""
    peak = cumulative.expanding().max()
    dd = (cumulative - peak) / peak
    return float(dd.min()) if len(dd) > 0 else 0.0


def _sharpe(returns: pd.Series, rf_annual: float = RISK_FREE_RATE_ANNUAL) -> float:
    """Annualised Sharpe ratio from daily returns."""
    if returns.empty or returns.std() == 0:
        return 0.0
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def compare_to_benchmark(
    portfolio_prices: pd.Series,
    start_date: str,
    end_date: str,
    ticker: str = "portfolio",
) -> Optional[BenchmarkComparison]:
    """Compare a price series against NIFTY 50 over the same period.

    Parameters
    ----------
    portfolio_prices : pd.Series
        Daily portfolio value or stock close prices (date-indexed).
    start_date, end_date : str
        Date range (YYYY-MM-DD).
    ticker : str
        Label for the portfolio/stock.

    Returns
    -------
    BenchmarkComparison or None on failure.
    """
    bench = _fetch_benchmark(start_date, end_date)
    if bench.empty or len(bench) < 20:
        logger.warning("Insufficient benchmark data for comparison")
        return None

    # Align dates
    port = portfolio_prices.copy()
    if port.index.tz is not None:
        port.index = port.index.tz_localize(None)
    if bench.index.tz is not None:
        bench.index = bench.index.tz_localize(None)

    common = port.index.intersection(bench.index)
    if len(common) < 20:
        logger.warning("Fewer than 20 overlapping days")
        return None

    port = port.loc[common]
    bench = bench.loc[common]

    # Daily returns
    port_ret = port.pct_change().dropna()
    bench_ret = bench.pct_change().dropna()
    common_ret = port_ret.index.intersection(bench_ret.index)
    port_ret = port_ret.loc[common_ret]
    bench_ret = bench_ret.loc[common_ret]

    if len(port_ret) < 10:
        return None

    # Cumulative returns
    port_cum = (1 + port_ret).cumprod()
    bench_cum = (1 + bench_ret).cumprod()

    port_total = float(port_cum.iloc[-1] - 1) * 100
    bench_total = float(bench_cum.iloc[-1] - 1) * 100

    # Beta (slope of regression: r_port vs r_bench)
    cov = np.cov(port_ret.values, bench_ret.values)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0

    # Jensen's Alpha (annualised)
    rf_daily = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / TRADING_DAYS) - 1
    alpha_daily = port_ret.mean() - (rf_daily + beta * (bench_ret.mean() - rf_daily))
    alpha_annual = float(alpha_daily * TRADING_DAYS) * 100

    # Tracking error and Information Ratio
    tracking_diff = port_ret - bench_ret
    te_annual = float(tracking_diff.std() * np.sqrt(TRADING_DAYS)) * 100
    ir = float(tracking_diff.mean() / tracking_diff.std() * np.sqrt(TRADING_DAYS)) if tracking_diff.std() > 0 else 0.0

    # Advanced risk metrics (Phase 0)
    sortino = calmar = omega = cvar95 = 0.0
    try:
        from services.risk_metrics import RiskMetrics
        sortino = RiskMetrics.sortino_ratio(port_ret, rf_annual=RISK_FREE_RATE_ANNUAL)
        calmar = RiskMetrics.calmar_ratio(port_ret)
        omega = RiskMetrics.omega_ratio(port_ret)
        cvar95 = RiskMetrics.cvar(port_ret, alpha=0.05)
    except Exception:
        pass

    return BenchmarkComparison(
        ticker=ticker,
        period_days=len(common_ret),
        portfolio_return_pct=round(port_total, 2),
        benchmark_return_pct=round(bench_total, 2),
        excess_return_pct=round(port_total - bench_total, 2),
        portfolio_sharpe=round(_sharpe(port_ret), 3),
        benchmark_sharpe=round(_sharpe(bench_ret), 3),
        jensens_alpha=round(alpha_annual, 3),
        information_ratio=round(ir, 3),
        tracking_error_ann=round(te_annual, 2),
        portfolio_beta=round(beta, 3),
        portfolio_max_dd_pct=round(_max_drawdown(port_cum) * 100, 2),
        benchmark_max_dd_pct=round(_max_drawdown(bench_cum) * 100, 2),
        portfolio_sortino=round(sortino, 3),
        portfolio_calmar=round(calmar, 3),
        portfolio_omega=round(omega, 3),
        portfolio_cvar_95=round(cvar95, 4),
    )


def compare_strategy_to_nifty(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[BenchmarkComparison]:
    """Convenience: fetch a single stock/ETF price and compare vs NIFTY.

    Useful for evaluating individual screener picks against the benchmark.
    """
    try:
        import yfinance as yf

        suffix = ".NS" if not ticker.endswith((".NS", ".BO")) else ""
        df = yf.download(f"{ticker}{suffix}", start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return compare_to_benchmark(close.dropna(), start_date, end_date, ticker=ticker)
    except Exception as e:
        logger.warning("Benchmark comparison failed for %s: %s", ticker, e)
        return None


def batch_alpha_report(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, BenchmarkComparison]:
    """Run benchmark comparison for multiple tickers.

    Returns a dict keyed by ticker with BenchmarkComparison objects.
    Only includes tickers with successful comparisons.
    """
    results = {}
    for t in tickers:
        comp = compare_strategy_to_nifty(t, start_date, end_date)
        if comp is not None:
            results[t] = comp
    return results
