"""
Stock metrics calculator - fundamentals and technicals.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
from models import StockMetrics
from config import Config


class MetricsCalculator:
    """Calculates fundamental and technical metrics for stocks."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def get_stock_metrics(self, ticker: str) -> Optional[StockMetrics]:
        """
        Calculate all metrics for a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            StockMetrics object with all calculated metrics
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get stock info
            info = stock.info
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=Config.HISTORICAL_DAYS)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                print(f"No historical data for {ticker}")
                return None
            
            # Calculate fundamentals
            fundamentals = self._calculate_fundamentals(ticker, stock, info)
            
            # Calculate technicals
            technicals = self._calculate_technicals(hist)
            
            # Create metrics object
            metrics = StockMetrics(
                ticker=ticker,
                timestamp=datetime.now(),
                peg_ratio=fundamentals.get('peg_ratio'),
                roe=fundamentals.get('roe'),
                eps=fundamentals.get('eps'),
                free_cash_flow=fundamentals.get('free_cash_flow'),
                dcf_value=fundamentals.get('dcf_value'),
                intrinsic_value=fundamentals.get('intrinsic_value'),
                rsi=technicals.get('rsi'),
                macd=technicals.get('macd'),
                macd_signal=technicals.get('macd_signal'),
                macd_histogram=technicals.get('macd_histogram'),
                fibonacci_levels=technicals.get('fibonacci_levels'),
                bollinger_upper=technicals.get('bollinger_upper'),
                bollinger_middle=technicals.get('bollinger_middle'),
                bollinger_lower=technicals.get('bollinger_lower'),
                max_drawdown=technicals.get('max_drawdown'),
                current_price=technicals.get('current_price')
            )
            
            return metrics
        
        except Exception as e:
            print(f"Error calculating metrics for {ticker}: {e}")
            return None
    
    def _calculate_fundamentals(
        self, 
        ticker: str, 
        stock: yf.Ticker, 
        info: dict
    ) -> Dict[str, Optional[float]]:
        """Calculate fundamental metrics."""
        fundamentals = {}
        
        try:
            # PEG Ratio
            peg = info.get('pegRatio')
            fundamentals['peg_ratio'] = float(peg) if peg else None
            
            # ROE (Return on Equity)
            roe = info.get('returnOnEquity')
            fundamentals['roe'] = float(roe) * 100 if roe else None
            
            # EPS (Earnings Per Share)
            eps = info.get('trailingEps')
            fundamentals['eps'] = float(eps) if eps else None
            
            # Free Cash Flow
            fcf = info.get('freeCashflow')
            fundamentals['free_cash_flow'] = float(fcf) if fcf else None
            
            # DCF Value (simplified calculation)
            dcf = self._calculate_dcf(info)
            fundamentals['dcf_value'] = dcf
            
            # Intrinsic Value (using Graham's formula)
            intrinsic = self._calculate_intrinsic_value(info)
            fundamentals['intrinsic_value'] = intrinsic
        
        except Exception as e:
            print(f"Error calculating fundamentals: {e}")
        
        return fundamentals
    
    def _calculate_dcf(self, info: dict) -> Optional[float]:
        """
        Calculate simplified DCF (Discounted Cash Flow) value.
        
        This is a simplified DCF calculation using free cash flow.
        """
        try:
            fcf = info.get('freeCashflow')
            shares = info.get('sharesOutstanding')
            
            if not fcf or not shares:
                return None
            
            # Simplified DCF with 10% discount rate and 3% growth
            discount_rate = 0.10
            growth_rate = 0.03
            
            # Terminal value
            terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
            
            # DCF per share
            dcf_value = terminal_value / shares
            
            return float(dcf_value)
        
        except:
            return None
    
    def _calculate_intrinsic_value(self, info: dict) -> Optional[float]:
        """
        Calculate intrinsic value using Benjamin Graham's formula.
        
        IV = EPS × (8.5 + 2g) × 4.4 / Y
        where g is growth rate and Y is current yield on AAA bonds (approx 4.5%)
        """
        try:
            eps = info.get('trailingEps')
            growth = info.get('earningsGrowth')
            
            if not eps:
                return None
            
            # Default growth rate if not available
            g = (growth * 100) if growth else 10
            
            # Graham's formula
            intrinsic_value = eps * (8.5 + 2 * g) * 4.4 / 4.5
            
            return float(intrinsic_value)
        
        except:
            return None
    
    def _calculate_technicals(self, hist: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate technical indicators using native pandas/numpy."""
        technicals = {}
        
        try:
            # Current price
            technicals['current_price'] = float(hist['Close'].iloc[-1])
            
            # RSI (Relative Strength Index)
            rsi = self._calculate_rsi(hist['Close'], Config.RSI_PERIOD)
            technicals['rsi'] = float(rsi) if rsi is not None else None
            
            # MACD
            macd_dict = self._calculate_macd(
                hist['Close'], 
                Config.MACD_FAST, 
                Config.MACD_SLOW, 
                Config.MACD_SIGNAL
            )
            technicals['macd'] = macd_dict.get('macd')
            technicals['macd_signal'] = macd_dict.get('signal')
            technicals['macd_histogram'] = macd_dict.get('histogram')
            
            # Bollinger Bands
            bb_dict = self._calculate_bollinger_bands(
                hist['Close'], 
                Config.BOLLINGER_PERIOD, 
                Config.BOLLINGER_STD
            )
            technicals['bollinger_upper'] = bb_dict.get('upper')
            technicals['bollinger_middle'] = bb_dict.get('middle')
            technicals['bollinger_lower'] = bb_dict.get('lower')
            
            # Fibonacci Retracement Levels
            fib_levels = self._calculate_fibonacci(hist)
            technicals['fibonacci_levels'] = fib_levels
            
            # Maximum Drawdown
            max_dd = self._calculate_max_drawdown(hist)
            technicals['max_drawdown'] = max_dd
        
        except Exception as e:
            print(f"Error calculating technicals: {e}")
        
        return technicals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not rsi.empty else None
        except:
            return None
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, Optional[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            
            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal_line.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        except:
            return {'macd': None, 'signal': None, 'histogram': None}
    
    def _calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: int = 2
    ) -> Dict[str, Optional[float]]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': float(upper_band.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower_band.iloc[-1])
            }
        except:
            return {'upper': None, 'middle': None, 'lower': None}
    
    def _calculate_fibonacci(self, hist: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        try:
            high = hist['High'].max()
            low = hist['Low'].min()
            diff = high - low
            
            levels = {
                '0.0': float(high),
                '0.236': float(high - 0.236 * diff),
                '0.382': float(high - 0.382 * diff),
                '0.500': float(high - 0.500 * diff),
                '0.618': float(high - 0.618 * diff),
                '0.786': float(high - 0.786 * diff),
                '1.0': float(low)
            }
            
            return levels
        
        except:
            return {}
    
    def _calculate_max_drawdown(self, hist: pd.DataFrame) -> Optional[float]:
        """Calculate maximum drawdown percentage."""
        try:
            prices = hist['Close']
            cumulative_max = prices.cummax()
            drawdown = (prices - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()
            
            return float(max_drawdown * 100)  # Return as percentage
        
        except:
            return None
