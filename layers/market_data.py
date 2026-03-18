"""
Market Data Layer — Unified data feed interface.

Provides a single entry point for all market data consumption:
  - Historical OHLCV (yfinance, DB)
  - Real-time ticks (Kite WebSocket, event bus)
  - Fundamental data (yfinance info, external APIs)
  - News feeds (scrapers)

All data flows out as events on the bus:
  - ``market_data.tick``
  - ``market_data.ohlcv``
  - ``market_data.news``
  - ``market_data.fundamental``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Unified market data gateway.

    Wraps yfinance / Kite / scrapers behind a consistent interface
    and emits events for downstream consumers.
    """

    def __init__(self, market: str = "US"):
        self.market = market

    def fetch_ohlcv(
        self,
        tickers: List[str],
        *,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch historical OHLCV data for a list of tickers."""
        import yfinance as yf
        from infrastructure.event_bus import event_bus
        from infrastructure.latency_tracker import latency_tracker

        results = {}
        with latency_tracker.measure("market_data.fetch_ohlcv"):
            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period=period, interval=interval, start=start, end=end)
                    if not hist.empty:
                        results[ticker] = hist
                        event_bus.emit(
                            "market_data.ohlcv",
                            payload={"symbol": ticker, "rows": len(hist)},
                            source="market_data_service",
                        )
                except Exception as exc:
                    logger.warning("OHLCV fetch failed for %s: %s", ticker, exc)

        return results

    def fetch_fundamentals(self, tickers: List[str]) -> Dict[str, dict]:
        """Fetch fundamental data (info dict) for tickers."""
        import yfinance as yf

        results = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                results[ticker] = info or {}
            except Exception as exc:
                logger.warning("Fundamental fetch failed for %s: %s", ticker, exc)
                results[ticker] = {}
        return results

    def fetch_news(self, tickers: List[str]) -> List[dict]:
        """Delegate to the appropriate news aggregator."""
        from infrastructure.event_bus import event_bus

        if self.market == "IND":
            from scrapers.ind_aggregator import IndianNewsAggregator
            aggregator = IndianNewsAggregator()
        else:
            from scrapers.us_aggregator import USNewsAggregator
            aggregator = USNewsAggregator()

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    news = pool.submit(
                        asyncio.run,
                        aggregator.fetch_news_for_tickers(tickers)
                    ).result()
            else:
                news = loop.run_until_complete(
                    aggregator.fetch_news_for_tickers(tickers)
                )
        except RuntimeError:
            news = asyncio.run(aggregator.fetch_news_for_tickers(tickers))

        event_bus.emit(
            "market_data.news",
            payload={"market": self.market, "count": len(news)},
            source="market_data_service",
        )
        return news
