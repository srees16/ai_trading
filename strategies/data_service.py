"""
Data Service Module.

Provides centralized data fetching, caching, and preprocessing for all
trading strategies. This eliminates the need for strategies to implement
their own data loading logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union
import logging
from functools import lru_cache

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
        try:
            import yfinance
            return True
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            return False
    
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
        """Fetch data from Yahoo Finance."""
        if not self._yf_available:
            raise ImportError("yfinance is required for data fetching")
        
        import yfinance as yf
        
        try:
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
            logger.error(f"Error fetching {ticker}: {e}")
            raise ValueError(f"Failed to fetch data for {ticker}: {e}")
    
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
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
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
    
    def preload_data(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str
    ) -> None:
        """
        Preload data for multiple tickers into cache.
        
        Useful for batch operations to minimize download time.
        """
        if not self._yf_available:
            return
        
        import yfinance as yf
        
        # Batch download
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker'
            )
            
            # Cache each ticker
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        df = data
                    else:
                        df = data[ticker].copy()
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    
                    df = self._clean_ohlcv(df)
                    
                    cache_key = f"{ticker}_{start_date}_{end_date}_1d"
                    self._cache[cache_key] = df
                    self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
                
                except Exception as e:
                    logger.debug(f"Preload failed for {ticker}: {e}")
        
        except Exception as e:
            logger.warning(f"Batch preload failed: {e}")


# Convenience function for quick access
def get_data(
    ticker: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> pd.DataFrame:
    """
    Quick access function to get OHLCV data.
    
    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        **kwargs: Additional arguments passed to DataService.get_ohlcv
    
    Returns:
        DataFrame with OHLCV data
    """
    service = DataService()
    return service.get_ohlcv(ticker, start_date, end_date, **kwargs)


def get_multiple_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """
    Quick access function to get multiple tickers.
    
    Args:
        tickers: List of tickers
        start_date: Start date
        end_date: End date
        **kwargs: Additional arguments
    
    Returns:
        Dictionary of DataFrames
    """
    service = DataService()
    return service.get_multiple_ohlcv(tickers, start_date, end_date, **kwargs)
