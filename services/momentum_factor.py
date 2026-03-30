"""
Cross-Sectional Momentum Factor — Phase 1.1.

Ranks stocks in a universe by 12-minus-1-month return (Jegadeesh & Titman 1993).
The classic momentum strategy:
  - Formation period: 12 months (252 trading days)
  - Skip period: 1 month (21 days) to avoid short-term reversal
  - Select top N stocks by momentum score

Integration with Carver pipeline:
  - Produces a momentum forecast per symbol (scaled to -20..+20)
  - Fed into forecast_combiner alongside EWMAC, carry, screener
  - Weight: 15-25% of combined forecast

Research basis:
  - Jegadeesh & Titman (1993): 12-2% annual return premium
  - Asness et al. (2013): Momentum works across all asset classes
  - Indian evidence: Sehgal & Balakrishnan (2002), ~15-20% premium on NSE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MomentumScore:
    """Momentum ranking for a single stock."""
    ticker: str
    return_12m_skip1m: float  # 12-1 month return
    rank: int = 0             # 1 = best momentum
    decile: int = 0           # 1 = top decile
    forecast: float = 0.0     # Carver-scale forecast (-20 to +20)


@dataclass
class MomentumFactorResult:
    """Result of the momentum ranking for the full universe."""
    scores: List[MomentumScore] = field(default_factory=list)
    universe_size: int = 0
    top_n: List[str] = field(default_factory=list)
    bottom_n: List[str] = field(default_factory=list)
    computed_at: str = ""


class MomentumFactor:
    """Cross-sectional momentum ranking engine.

    Parameters
    ----------
    formation_period : int
        Lookback in trading days (default 252 = 12 months).
    skip_period : int
        Days to skip at the end to avoid reversal (default 21 = 1 month).
    n_stocks : int
        Number of top-momentum stocks to select (default 20).
    """

    def __init__(
        self,
        formation_period: int = 252,
        skip_period: int = 21,
        n_stocks: int = 20,
    ):
        self.formation = formation_period
        self.skip = skip_period
        self.n_stocks = n_stocks

    def rank_from_cache(
        self,
        ohlcv_cache: Dict[str, "pd.DataFrame"],
    ) -> MomentumFactorResult:
        """Rank stocks from pre-loaded OHLCV cache.

        Parameters
        ----------
        ohlcv_cache : dict[str, pd.DataFrame]
            {symbol: OHLCV DataFrame with 'Close' column}.

        Returns
        -------
        MomentumFactorResult
        """
        from datetime import datetime

        raw_returns: Dict[str, float] = {}
        min_bars = self.formation + 10  # need enough data

        for ticker, df in ohlcv_cache.items():
            if df is None or len(df) < min_bars:
                continue
            try:
                close = df["Close"]
                if hasattr(close, "squeeze"):
                    close = close.squeeze()

                # 12-minus-1 return: price at (t - skip) / price at (t - formation) - 1
                idx_recent = len(close) - self.skip - 1
                idx_formation = len(close) - self.formation - 1

                if idx_formation < 0 or idx_recent < 0:
                    continue

                price_recent = float(close.iloc[idx_recent])
                price_formation = float(close.iloc[idx_formation])

                if price_formation <= 0:
                    continue

                ret = (price_recent / price_formation) - 1.0
                raw_returns[ticker] = ret
            except Exception as exc:
                logger.debug("Momentum calc failed for %s: %s", ticker, exc)

        if not raw_returns:
            return MomentumFactorResult(computed_at=datetime.utcnow().isoformat())

        # Sort by return descending
        sorted_tickers = sorted(raw_returns, key=raw_returns.get, reverse=True)
        n_total = len(sorted_tickers)
        decile_size = max(1, n_total // 10)

        scores: List[MomentumScore] = []
        for rank_idx, ticker in enumerate(sorted_tickers):
            rank = rank_idx + 1
            decile = min(10, rank_idx // decile_size + 1)
            ret = raw_returns[ticker]

            # Convert momentum return to Carver-scale forecast
            forecast = self._return_to_forecast(ret, rank, n_total)

            scores.append(MomentumScore(
                ticker=ticker,
                return_12m_skip1m=round(ret, 4),
                rank=rank,
                decile=decile,
                forecast=round(forecast, 2),
            ))

        top_n = [s.ticker for s in scores[:self.n_stocks]]
        bottom_n = [s.ticker for s in scores[-self.n_stocks:]]

        return MomentumFactorResult(
            scores=scores,
            universe_size=n_total,
            top_n=top_n,
            bottom_n=bottom_n,
            computed_at=datetime.utcnow().isoformat(),
        )

    def rank_from_yfinance(
        self,
        universe: List[str],
    ) -> MomentumFactorResult:
        """Fetch OHLCV from yfinance and rank.

        Parameters
        ----------
        universe : list[str]
            List of tickers (e.g. ['RELIANCE.NS', 'TCS.NS', ...]).
        """
        import yfinance as yf
        import pandas as pd

        period = f"{self.formation + self.skip + 30}d"
        ohlcv_cache: Dict[str, pd.DataFrame] = {}

        for ticker in universe:
            try:
                df = yf.download(ticker, period=period, progress=False, timeout=15, auto_adjust=True)
                if df is not None and len(df) > self.formation:
                    ohlcv_cache[ticker] = df
            except Exception:
                logger.debug("Download failed for %s", ticker)

        return self.rank_from_cache(ohlcv_cache)

    def get_forecasts(
        self,
        ohlcv_cache: Dict[str, "pd.DataFrame"],
    ) -> Dict[str, float]:
        """Return momentum forecasts in {symbol: forecast} format.

        Ready to feed into the Carver forecast combiner.
        """
        result = self.rank_from_cache(ohlcv_cache)
        return {s.ticker: s.forecast for s in result.scores if s.forecast != 0.0}

    @staticmethod
    def _return_to_forecast(
        ret: float,
        rank: int,
        n_total: int,
    ) -> float:
        """Map momentum return + rank to a Carver-scale forecast.

        Top quintile  → positive forecast (up to +20)
        Bottom quintile → negative forecast (down to -20)
        Middle 60%    → near zero

        Uses rank-based z-score to avoid sensitivity to outlier returns.
        """
        if n_total <= 1:
            return 0.0

        # Rank-based percentile (0 = best, 1 = worst)
        pctile = (rank - 1) / (n_total - 1)

        # Map to z-score: pctile 0 → +2, pctile 0.5 → 0, pctile 1 → -2
        z = 2.0 * (1.0 - 2.0 * pctile)

        # Scale to Carver forecast range (z of ±2 maps to ±20)
        forecast = z * 10.0
        return max(-20.0, min(20.0, forecast))


# ── Convenience ──────────────────────────────────────────────

def compute_momentum_forecasts(
    ohlcv_cache: Dict[str, "pd.DataFrame"],
    formation: int = 252,
    skip: int = 21,
) -> Dict[str, float]:
    """Compute momentum forecasts for all symbols in cache.

    Returns dict of {symbol: forecast} for integration with
    the Carver forecast combiner.
    """
    mf = MomentumFactor(formation_period=formation, skip_period=skip)
    return mf.get_forecasts(ohlcv_cache)
