"""
Data Service Module.

Provides centralized data fetching, caching, and preprocessing for all
trading strategies. This eliminates the need for strategies to implement
their own data loading logic.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from strategies.utils import calculate_rsi

logger = logging.getLogger(__name__)


class DataService:
    """
    Centralized data service for fetching and managing market data.
    
    Provides a unified interface for:
    - Fetching OHLCV data from various sources
    - Caching data to minimize API calls
    - Preprocessing and cleaning data
    - Technical indicator calculation
    
    Example:
        ```python
        service = DataService()
        
        # Fetch single ticker
        df = service.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")
        
        # Fetch multiple tickers
        data = service.get_multiple_ohlcv(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
        ```
    """
    
    _instance = None
    _cache: dict[str, pd.DataFrame] = {}
    _cache_expiry: dict[str, datetime] = {}
    _cache_duration = timedelta(hours=1)
    
    def __new__(cls):
        """Singleton pattern to share cache across instances."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize data service."""
        self._yf_available = self._check_yfinance()
    
    def _check_yfinance(self) -> bool:
        """Check if yfinance is available."""
        return True
    
    @staticmethod
    def _ensure_exchange_suffix(ticker: str) -> str:
        """Add .NS suffix for Indian tickers that are missing it.

        yfinance requires NSE tickers to end with ``.NS`` (e.g.
        ``RELIANCE.NS``).  If the ticker already carries an exchange
        suffix (``.NS``, ``.BO``, ``.L``, etc.) it is returned as-is.
        Otherwise we make a quick ``yf.Ticker`` probe: if the bare
        symbol fails but ``{symbol}.NS`` succeeds, we return the
        suffixed version.
        """
        # Already has an exchange suffix nothing to do
        if '.' in ticker:
            return ticker

        # Quick probe: try the bare ticker first
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).fast_info
            if info and getattr(info, 'timezone', None):
                return ticker  # bare symbol works (US stock, etc.)
        except Exception:
            pass

        # Try with .NS suffix (NSE India), applying override map
        from utils import yf_nse_symbol
        ns_ticker = yf_nse_symbol(ticker)
        try:
            import yfinance as yf
            info = yf.Ticker(ns_ticker).fast_info
            if info and getattr(info, 'timezone', None):
                logger.info("DataService: Resolved %s %s", ticker, ns_ticker)
                return ns_ticker
        except Exception:
            pass

        # Return original — let downstream handle the error
        return ticker
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', '5m', etc.)
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        
        Raises:
            ValueError: If ticker is invalid or data unavailable
        """
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                logger.debug(f"Cache hit for {ticker}")
                return self._cache[cache_key].copy()
        
        # Fetch data
        df = self._fetch_from_yfinance(ticker, start_date, end_date, interval)

        # ── Survivorship bias check ────────────────────────────
        # Reject delisted / suspended tickers early so strategies
        # don't backtest on dead stocks.
        try:
            from services.survivorship_filter import check_ticker
            result = check_ticker(ticker, ohlcv=df)
            if not result.is_valid:
                logger.warning(
                    "DataService: rejected %s — %s", ticker, result.reason,
                )
                return pd.DataFrame()  # return empty → strategy sees no data
        except Exception:
            pass  # degrade gracefully

        # Cache result
        if use_cache and df is not None and not df.empty:
            self._cache[cache_key] = df.copy()
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
        
        return df
    
    def _fetch_from_yfinance(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with retry on crumb/auth errors."""
        if not self._yf_available:
            raise ImportError("yfinance is required for data fetching")

        # Ensure Indian tickers have .NS suffix for yfinance
        ticker = self._ensure_exchange_suffix(ticker)

        last_exc = None
        for attempt in range(2):
            try:
                import yfinance as yf
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True
                )
                
                # Handle MultiIndex columns from newer yfinance versions
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Ensure we have required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                
                # Clean data
                df = self._clean_ohlcv(df)
                
                return df
            
            except Exception as e:
                last_exc = e
                exc_str = str(e)
                # Retry once on 401 Invalid Crumb — clear yfinance cookie cache
                if attempt == 0 and ("401" in exc_str or "Invalid Crumb" in exc_str):
                    logger.info("yfinance crumb expired for %s — retrying", ticker)
                    try:
                        from utils import _clear_yfinance_crumb_cache
                        _clear_yfinance_crumb_cache()
                    except ImportError:
                        pass
                    continue
                break

        logger.warning("Failed to fetch %s from yfinance: %s", ticker, last_exc)
        raise ValueError(f"Failed to fetch data for {ticker}: {last_exc}")
    
    def _clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data."""
        # Remove rows with all NaN
        df = df.dropna(how='all')
        
        # Forward fill small gaps
        df = df.ffill(limit=3)
        
        # Remove rows with zero volume (market closed)
        if 'Volume' in df.columns:
            df = df[df['Volume'] > 0]
        
        # Ensure proper data types
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)

        # ── OHLCV consistency validation ──
        # Fix rows where High < Low (data corruption)
        if {'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
            bad_hl = df['High'] < df['Low']
            if bad_hl.any():
                logger.warning("OHLCV: %d rows with High < Low — swapping", bad_hl.sum())
                df.loc[bad_hl, ['High', 'Low']] = df.loc[bad_hl, ['Low', 'High']].values

            # Clamp High to be >= max(Open, Close) and Low <= min(Open, Close)
            oc_max = df[['Open', 'Close']].max(axis=1)
            oc_min = df[['Open', 'Close']].min(axis=1)
            df['High'] = df['High'].clip(lower=oc_max)
            df['Low'] = df['Low'].clip(upper=oc_min)

            # Drop rows where any OHLC value is zero or negative (bad data)
            invalid_price = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
            if invalid_price.any():
                logger.warning("OHLCV: dropping %d rows with zero/negative prices", invalid_price.sum())
                df = df[~invalid_price]

        # Deduplicate index (can occur when merging yfinance + Bhavcopy)
        if df.index.duplicated().any():
            logger.warning("OHLCV: dropping %d duplicate index rows", df.index.duplicated().sum())
            df = df[~df.index.duplicated(keep='last')]

        # Normalize timezone to tz-naive (IST dates only, avoids merge misalignment)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        return df
    
    def get_multiple_ohlcv(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        result = {}
        
        for ticker in tickers:
            try:
                df = self.get_ohlcv(ticker, start_date, end_date, interval)
                if not df.empty:
                    result[ticker] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        
        return result
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: list[str] = None
    ) -> pd.DataFrame:
        """
        Add common technical indicators to OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators to add. Options:
                - 'sma_20', 'sma_50', 'sma_200' (Simple Moving Averages)
                - 'ema_12', 'ema_26' (Exponential Moving Averages)
                - 'rsi' (Relative Strength Index)
                - 'macd' (MACD line and signal)
                - 'bollinger' (Bollinger Bands)
                - 'atr' (Average True Range)
        
        Returns:
            DataFrame with added indicator columns
        """
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd']
        
        df = df.copy()
        
        for indicator in indicators:
            try:
                if indicator.startswith('sma_'):
                    period = int(indicator.split('_')[1])
                    df[indicator] = df['Close'].rolling(window=period).mean()
                
                elif indicator.startswith('ema_'):
                    period = int(indicator.split('_')[1])
                    df[indicator] = df['Close'].ewm(span=period, adjust=False).mean()
                
                elif indicator == 'rsi':
                    df['rsi'] = self._calculate_rsi(df['Close'])
                
                elif indicator == 'macd':
                    macd, signal = self._calculate_macd(df['Close'])
                    df['macd'] = macd
                    df['macd_signal'] = signal
                    df['macd_hist'] = macd - signal
                
                elif indicator == 'bollinger':
                    bb_mid, bb_upper, bb_lower = self._calculate_bollinger(df['Close'])
                    df['bb_mid'] = bb_mid
                    df['bb_upper'] = bb_upper
                    df['bb_lower'] = bb_lower
                
                elif indicator == 'atr':
                    df['atr'] = self._calculate_atr(df)
            
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator}: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index — delegates to shared utility."""
        return calculate_rsi(prices, period)
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return macd_line, signal_line
    
    def _calculate_bollinger(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        mid = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = mid + (std * std_dev)
        lower = mid - (std * std_dev)
        
        return mid, upper, lower
    
    def _calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            ticker: Specific ticker to clear, or None for all
        """
        if ticker:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(ticker)]
            for key in keys_to_remove:
                del self._cache[key]
                del self._cache_expiry[key]
        else:
            self._cache.clear()
            self._cache_expiry.clear()
