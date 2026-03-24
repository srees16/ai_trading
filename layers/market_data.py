"""
Market Data Layer — Unified data feed interface.

Provides a single entry point for all market data consumption:
  - Historical OHLCV (yfinance for US/global, Kite for Indian tickers)
  - Real-time ticks (Kite WebSocket, event bus)
  - Fundamental data (yfinance info, Tijori for Indian stocks)
  - News feeds (scrapers)

Routing logic:
  - Tickers ending in ``.NS`` or ``.BO`` → Kite Connect (primary), yfinance (fallback)
  - All other tickers → yfinance

All data flows out as events on the bus:
  - ``market_data.tick``
  - ``market_data.ohlcv``
  - ``market_data.news``
  - ``market_data.fundamental``
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────

_IND_SUFFIXES = (".NS", ".BO")


def _is_indian(ticker: str) -> bool:
    """Return True if the ticker targets NSE or BSE."""
    return ticker.upper().endswith(_IND_SUFFIXES)


def _strip_suffix(ticker: str) -> str:
    """Strip .NS/.BO suffix to get the raw NSE/BSE trading symbol."""
    upper = ticker.upper()
    for sfx in _IND_SUFFIXES:
        if upper.endswith(sfx):
            return upper[: -len(sfx)]
    return upper


def _period_to_days(period: str) -> int:
    """Convert yfinance-style period string to approximate day count."""
    mapping = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "ytd": 180, "max": 3650,
    }
    return mapping.get(period, 365)


def _kite_interval(interval: str) -> str:
    """Map yfinance interval strings to Kite Connect interval names."""
    mapping = {
        "1m": "minute", "2m": "2minute", "3m": "3minute",
        "5m": "5minute", "10m": "10minute", "15m": "15minute",
        "30m": "30minute", "60m": "60minute", "1h": "60minute",
        "1d": "day", "1wk": "day", "1mo": "day",
    }
    return mapping.get(interval, "day")


class MarketDataService:
    """
    Unified market data gateway.

    Routes requests based on ticker suffix:
      * Indian tickers (``.NS`` / ``.BO``) → Kite Connect, yfinance fallback
      * US / global tickers → yfinance
    """

    def __init__(self, market: str = "US"):
        self.market = market
        self._instrument_cache: Dict[str, int] = {}  # symbol → instrument_token

    # ── Instrument token resolution ────────────────────────────

    def _resolve_tokens(self, symbols: List[str]) -> Dict[str, int]:
        """Resolve NSE trading symbols to Kite instrument tokens.

        Uses kite.quote() for small batches and caches results.
        """
        from api.dependencies import get_kite_session

        kite = get_kite_session()
        if not kite:
            return {}

        unknown = [s for s in symbols if s not in self._instrument_cache]
        if not unknown:
            return {s: self._instrument_cache[s] for s in symbols if s in self._instrument_cache}

        instruments = [f"NSE:{s}" for s in unknown]
        BATCH = 200
        for i in range(0, len(instruments), BATCH):
            batch = instruments[i: i + BATCH]
            try:
                quotes = kite.quote(batch)
                for key, q in quotes.items():
                    sym = key.split(":", 1)[-1]
                    token = q.get("instrument_token")
                    if token:
                        self._instrument_cache[sym] = token
            except Exception as exc:
                logger.warning("kite.quote() batch failed: %s", exc)

        return {s: self._instrument_cache[s] for s in symbols if s in self._instrument_cache}

    # ── Kite OHLCV ─────────────────────────────────────────────

    def _fetch_ohlcv_kite(
        self,
        tickers: List[str],
        *,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch OHLCV from Kite Connect for Indian tickers."""
        from api.dependencies import get_kite_session

        kite = get_kite_session()
        if not kite:
            logger.info("Kite session not active — falling back to yfinance for IND tickers")
            return {}

        raw_symbols = [_strip_suffix(t) for t in tickers]
        token_map = self._resolve_tokens(raw_symbols)
        if not token_map:
            return {}

        if end:
            to_dt = datetime.strptime(end, "%Y-%m-%d")
        else:
            to_dt = datetime.now()
        if start:
            from_dt = datetime.strptime(start, "%Y-%m-%d")
        else:
            from_dt = to_dt - timedelta(days=_period_to_days(period))

        kite_iv = _kite_interval(interval)
        from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
        to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

        results: Dict[str, Any] = {}
        # Map raw symbol back to original ticker (with suffix)
        raw_to_original = {_strip_suffix(t): t for t in tickers}

        for sym, token in token_map.items():
            original_ticker = raw_to_original.get(sym, f"{sym}.NS")
            try:
                candles = kite.historical_data(token, from_str, to_str, kite_iv)
                if candles:
                    df = pd.DataFrame(candles)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df.set_index("date", inplace=True)
                    # Rename to match yfinance convention
                    col_map = {"open": "Open", "high": "High", "low": "Low",
                               "close": "Close", "volume": "Volume"}
                    df.rename(columns=col_map, inplace=True)
                    results[original_ticker] = df
            except Exception as exc:
                logger.warning("Kite historical_data failed for %s: %s", sym, exc)

        return results

    # ── Bhavcopy OHLCV (NSE EOD archive) ─────────────────────

    @staticmethod
    def _fetch_ohlcv_bhavcopy(
        tickers: List[str],
        *,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch OHLCV from NSE Bhavcopy for Indian tickers.

        Only supports daily interval.  Returns empty dict for
        intraday intervals so the caller can fall through to yfinance.
        """
        if interval not in ("1d", "1wk", "1mo"):
            return {}

        from datetime import date as _date
        from services.bhavcopy_fetcher import fetch_ohlcv_batch

        if end:
            end_dt = datetime.strptime(end, "%Y-%m-%d").date()
        else:
            end_dt = _date.today()
        if start:
            start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        else:
            start_dt = end_dt - timedelta(days=_period_to_days(period))

        try:
            results = fetch_ohlcv_batch(tickers, start=start_dt, end=end_dt)
            if results:
                logger.info(
                    "Bhavcopy returned data for %d / %d IND ticker(s)",
                    len(results), len(tickers),
                )
            return results
        except Exception as exc:
            logger.warning("Bhavcopy fetch failed: %s", exc)
            return {}

    # ── yfinance OHLCV ─────────────────────────────────────────

    @staticmethod
    def _fetch_ohlcv_yfinance(
        tickers: List[str],
        *,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch OHLCV from yfinance (US / global tickers)."""
        import yfinance as yf

        results = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period=period, interval=interval, start=start, end=end)
                if not hist.empty:
                    results[ticker] = hist
            except Exception as exc:
                logger.warning("yfinance OHLCV failed for %s: %s", ticker, exc)
        return results

    # ── Public: routed fetch_ohlcv ─────────────────────────────

    def fetch_ohlcv(
        self,
        tickers: List[str],
        *,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch historical OHLCV data — auto-routes by ticker suffix.

        Indian tickers (.NS/.BO) go through Kite Connect first; on failure
        they fall back to yfinance.  All other tickers use yfinance directly.
        """
        from infrastructure.event_bus import event_bus
        from infrastructure.latency_tracker import latency_tracker

        ind_tickers = [t for t in tickers if _is_indian(t)]
        us_tickers = [t for t in tickers if not _is_indian(t)]

        results: Dict[str, Any] = {}
        kw = dict(period=period, interval=interval, start=start, end=end)

        with latency_tracker.measure("market_data.fetch_ohlcv"):
            # ── US / global tickers → yfinance ────────────────
            if us_tickers:
                results.update(self._fetch_ohlcv_yfinance(us_tickers, **kw))

            # ── Indian tickers → Kite, then Bhavcopy, then yfinance ─
            if ind_tickers:
                kite_results = self._fetch_ohlcv_kite(ind_tickers, **kw)
                results.update(kite_results)

                # Fallback 1: Bhavcopy for tickers Kite couldn't serve
                missed = [t for t in ind_tickers if t not in kite_results]
                bhavcopy_results: Dict[str, Any] = {}
                if missed:
                    logger.info(
                        "Kite missed %d IND ticker(s), trying Bhavcopy: %s",
                        len(missed), missed,
                    )
                    bhavcopy_results = self._fetch_ohlcv_bhavcopy(missed, **kw)
                    results.update(bhavcopy_results)

                # Fallback 2: yfinance for anything still missing
                still_missed = [
                    t for t in ind_tickers
                    if t not in kite_results and t not in bhavcopy_results
                ]
                if still_missed:
                    logger.info(
                        "Bhavcopy missed %d IND ticker(s), falling back to yfinance: %s",
                        len(still_missed), still_missed,
                    )
                    results.update(self._fetch_ohlcv_yfinance(still_missed, **kw))

        for ticker, df in results.items():
            if _is_indian(ticker):
                if ind_tickers and ticker in kite_results:
                    src = "kite"
                elif ticker in bhavcopy_results:
                    src = "bhavcopy"
                else:
                    src = "yfinance"
            else:
                src = "yfinance"
            event_bus.emit(
                "market_data.ohlcv",
                payload={"symbol": ticker, "rows": len(df), "source": src},
                source="market_data_service",
            )

        return results

    # ── Public: routed fetch_fundamentals ──────────────────────

    def fetch_fundamentals(self, tickers: List[str]) -> Dict[str, dict]:
        """Fetch fundamental data — routes Indian tickers through Tijori fallback."""
        import yfinance as yf

        results = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                results[ticker] = info or {}
            except Exception as exc:
                logger.warning("yfinance fundamental fetch failed for %s: %s", ticker, exc)
                results[ticker] = {}

        # Back-fill missing Indian fundamentals via Tijori adapter
        ind_missing = [
            t for t in tickers
            if _is_indian(t) and not results.get(t)
        ]
        if ind_missing:
            try:
                from scrapers.ind_fundamentals.tijori_adapter import fetch_tijori_fundamentals
                for ticker in ind_missing:
                    raw_sym = _strip_suffix(ticker)
                    tijori_data = fetch_tijori_fundamentals(raw_sym)
                    if tijori_data:
                        results[ticker] = tijori_data
                        logger.info("Tijori filled fundamentals for %s", ticker)
            except ImportError:
                logger.debug("Tijori adapter not available for fundamental gap-fill")
            except Exception as exc:
                logger.warning("Tijori fundamental fetch failed: %s", exc)

        return results

    # ── Public: fetch_quotes (live prices) ─────────────────────

    def fetch_quotes(self, tickers: List[str]) -> Dict[str, dict]:
        """Fetch current price quotes — Kite for IND, yfinance for US."""
        ind_tickers = [t for t in tickers if _is_indian(t)]
        us_tickers = [t for t in tickers if not _is_indian(t)]

        results: Dict[str, dict] = {}

        # US tickers via yfinance
        if us_tickers:
            import yfinance as yf
            for ticker in us_tickers:
                try:
                    info = yf.Ticker(ticker).fast_info
                    results[ticker] = {
                        "last_price": getattr(info, "last_price", None),
                        "open": getattr(info, "open", None),
                        "day_high": getattr(info, "day_high", None),
                        "day_low": getattr(info, "day_low", None),
                        "volume": getattr(info, "last_volume", None),
                        "source": "yfinance",
                    }
                except Exception as exc:
                    logger.warning("yfinance quote failed for %s: %s", ticker, exc)

        # Indian tickers via Kite
        if ind_tickers:
            from api.dependencies import get_kite_session
            kite = get_kite_session()
            if kite:
                raw_symbols = [_strip_suffix(t) for t in ind_tickers]
                raw_to_orig = {_strip_suffix(t): t for t in ind_tickers}
                instruments = [f"NSE:{s}" for s in raw_symbols]
                BATCH = 200
                for i in range(0, len(instruments), BATCH):
                    batch = instruments[i: i + BATCH]
                    try:
                        quotes = kite.quote(batch)
                        for key, q in quotes.items():
                            sym = key.split(":", 1)[-1]
                            orig = raw_to_orig.get(sym, f"{sym}.NS")
                            ohlc = q.get("ohlc", {})
                            results[orig] = {
                                "last_price": q.get("last_price"),
                                "open": ohlc.get("open"),
                                "day_high": ohlc.get("high"),
                                "day_low": ohlc.get("low"),
                                "volume": q.get("volume"),
                                "source": "kite",
                            }
                    except Exception as exc:
                        logger.warning("Kite quote batch failed: %s", exc)
            else:
                # Fallback to yfinance for IND if Kite not active
                import yfinance as yf
                for ticker in ind_tickers:
                    try:
                        info = yf.Ticker(ticker).fast_info
                        results[ticker] = {
                            "last_price": getattr(info, "last_price", None),
                            "open": getattr(info, "open", None),
                            "day_high": getattr(info, "day_high", None),
                            "day_low": getattr(info, "day_low", None),
                            "volume": getattr(info, "last_volume", None),
                            "source": "yfinance",
                        }
                    except Exception as exc:
                        logger.warning("yfinance quote fallback failed for %s: %s", ticker, exc)

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
