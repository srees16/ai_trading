"""
Support and Resistance Strategy.

This strategy identifies key support and resistance levels using local
minima and maxima, then trades based on candlestick patterns near these levels.

Strategy Components:
1. Support level detection: Local lows where price has bounced
2. Resistance level detection: Local highs where price has reversed
3. Pattern recognition: Engulfing and star patterns near S/R levels

Strategy Rules:
- Buy when bullish pattern forms near support
- Sell when bearish pattern forms near resistance

Reference: Price action and technical analysis techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.base_strategy import (
    BaseStrategy, 
    StrategyResult, 
    StrategyCategory,
    ChartData,
    TableData,
    RiskParams
)
from strategies.registry import StrategyRegistry
from strategies.data_service import DataService
from strategies.utils import (
    matplotlib_to_base64,
    dataframe_to_table,
    create_metrics_summary
)


@StrategyRegistry.register_decorator
class SupportResistanceStrategy(BaseStrategy):
    """
    Support and Resistance with Candlestick Patterns Strategy.
    
    This strategy combines support/resistance level detection with
    candlestick pattern recognition for entry signals.
    
    Parameters:
        n1 (int): Candles before for S/R detection (default: 2)
        n2 (int): Candles after for S/R detection (default: 2)
        back_candles (int): Lookback period for S/R levels (default: 30)
        level_proximity (float): Price proximity to S/R level (default: 0.02)
    """
    
    name = "Support/Resistance"
    description = "Price action strategy using support/resistance levels with candlestick patterns"
    category = StrategyCategory.PATTERN_RECOGNITION
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 50
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "n1": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 5,
                "description": "Candles before pivot for S/R"
            },
            "n2": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 5,
                "description": "Candles after pivot for S/R"
            },
            "back_candles": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 100,
                "description": "Lookback period for S/R levels"
            },
            "level_proximity": {
                "type": "float",
                "default": 0.02,
                "min": 0.005,
                "max": 0.1,
                "description": "Max distance to S/R level (as % of price)"
            }
        }
    
    def run(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        capital: float,
        sentiment_data: Optional[dict] = None,
        risk_params: Optional[RiskParams | dict] = None,
        **kwargs
    ) -> StrategyResult:
        """
        Execute the Support/Resistance backtest.
        
        Args:
            tickers: List of ticker symbols to analyze
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            capital: Initial capital
            sentiment_data: Optional sentiment scores from news analysis
            risk_params: Risk management parameters
            **kwargs: Additional parameters
        
        Returns:
            StrategyResult with charts, tables, and metrics
        """
        start_time = time.time()
        
        # Validate inputs
        try:
            self.validate_inputs(tickers, start_date, end_date, capital)
        except ValueError as e:
            return StrategyResult(
                success=False,
                error_message=str(e)
            )
        
        # Parse parameters
        n1 = kwargs.get('n1', 2)
        n2 = kwargs.get('n2', 2)
        back_candles = kwargs.get('back_candles', 30)
        level_proximity = kwargs.get('level_proximity', 0.02)
        risk = self.get_risk_params(risk_params)
        
        # Initialize results containers
        all_charts = []
        all_tables = []
        all_metrics = {}
        combined_signals = []
        combined_portfolio = []
        
        # Initialize data service
        data_service = DataService()
        
        # Process each ticker
        for ticker in tickers:
            try:
                # Fetch data
                df = data_service.get_ohlcv(ticker, start_date, end_date)
                
                if df.empty or len(df) < self.min_data_points:
                    continue
                
                # Generate signals
                signals = self._generate_signals(df, n1, n2, back_candles, level_proximity)
                
                # Apply sentiment adjustment if available
                if sentiment_data and ticker in sentiment_data:
                    signals = self._apply_sentiment(signals, sentiment_data[ticker])
                
                # Calculate portfolio
                portfolio = self._calculate_portfolio(signals, capital, risk)
                
                # Create charts for this ticker
                charts = self._create_charts(signals, ticker)
                all_charts.extend(charts)
                
                # Calculate metrics
                metrics = self.calculate_metrics(portfolio, signals, capital)
                metrics['ticker'] = ticker
                metrics['support_levels'] = len(signals[signals['is_support'] == 1])
                metrics['resistance_levels'] = len(signals[signals['is_resistance'] == 1])
                all_metrics[ticker] = metrics
                
                # Store for combined analysis
                signals['ticker'] = ticker
                portfolio['ticker'] = ticker
                combined_signals.append(signals)
                combined_portfolio.append(portfolio)
                
            except Exception as e:
                all_metrics[ticker] = {'error': str(e)}
        
        # Create summary tables
        if combined_signals:
            # Performance summary table
            perf_data = []
            for ticker, metrics in all_metrics.items():
                if 'error' not in metrics:
                    perf_data.append({
                        'Ticker': ticker,
                        'Total Return': f"{metrics.get('total_return', 0):.2f}%",
                        'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                        'Support Levels': metrics.get('support_levels', 0),
                        'Resistance Levels': metrics.get('resistance_levels', 0)
                    })
            
            if perf_data:
                all_tables.append(TableData(
                    title="Performance Summary",
                    data=perf_data,
                    columns=['Ticker', 'Total Return', 'Sharpe Ratio', 'Support Levels', 'Resistance Levels'],
                    description="Summary of backtest performance for each ticker"
                ))
            
            # Recent signals table
            combined_df = pd.concat(combined_signals, ignore_index=True)
            recent_signals = combined_df[combined_df['signals'] != 0].tail(20)
            
            if not recent_signals.empty:
                signals_table = recent_signals[['ticker', 'Close', 'signals', 'pattern_type']].copy()
                signals_table['Signal Type'] = signals_table['signals'].map({1: 'BUY', -1: 'SELL'})
                all_tables.append(TableData(
                    title="Recent Trading Signals",
                    data=signals_table.to_dict(orient='records'),
                    columns=['ticker', 'Close', 'Signal Type', 'pattern_type'],
                    description="Most recent 20 trading signals"
                ))
        
        # Calculate aggregate metrics
        if all_metrics:
            valid_metrics = [m for m in all_metrics.values() if 'error' not in m]
            if valid_metrics:
                all_metrics['aggregate'] = {
                    'avg_return': np.mean([m.get('total_return', 0) for m in valid_metrics]),
                    'avg_sharpe': np.mean([m.get('sharpe_ratio', 0) for m in valid_metrics]),
                    'total_tickers': len(valid_metrics)
                }
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            charts=all_charts,
            tables=all_tables,
            metrics=all_metrics,
            signals=pd.concat(combined_signals) if combined_signals else None,
            portfolio=pd.concat(combined_portfolio) if combined_portfolio else None,
            success=len(combined_signals) > 0,
            error_message="" if combined_signals else "No valid data for any ticker",
            execution_time=execution_time,
            metadata={
                'strategy': self.name,
                'parameters': {
                    'n1': n1,
                    'n2': n2,
                    'back_candles': back_candles,
                    'level_proximity': level_proximity
                },
                'tickers_processed': len(combined_signals)
            }
        )
    
    def _is_support(self, df: pd.DataFrame, idx: int, n1: int, n2: int) -> bool:
        """Check if the candle at idx is a support level."""
        if idx < n1 or idx + n2 >= len(df):
            return False
        
        # Check candles before - each low should be higher than current
        for i in range(idx - n1 + 1, idx + 1):
            if df['Low'].iloc[i] > df['Low'].iloc[i - 1]:
                return False
        
        # Check candles after - each low should be higher than current
        for i in range(idx + 1, idx + n2 + 1):
            if df['Low'].iloc[i] < df['Low'].iloc[i - 1]:
                return False
        
        return True
    
    def _is_resistance(self, df: pd.DataFrame, idx: int, n1: int, n2: int) -> bool:
        """Check if the candle at idx is a resistance level."""
        if idx < n1 or idx + n2 >= len(df):
            return False
        
        # Check candles before - each high should be lower than current
        for i in range(idx - n1 + 1, idx + 1):
            if df['High'].iloc[i] < df['High'].iloc[i - 1]:
                return False
        
        # Check candles after - each high should be lower than current
        for i in range(idx + 1, idx + n2 + 1):
            if df['High'].iloc[i] > df['High'].iloc[i - 1]:
                return False
        
        return True
    
    def _is_engulfing(self, df: pd.DataFrame, idx: int) -> int:
        """
        Check for engulfing pattern.
        Returns: 1 for bullish, 2 for bearish, 0 for none
        """
        if idx < 1:
            return 0
        
        body = abs(df['Open'].iloc[idx] - df['Close'].iloc[idx])
        prev_body = abs(df['Open'].iloc[idx - 1] - df['Close'].iloc[idx - 1])
        
        min_body = df['Close'].mean() * 0.001  # Minimum body size
        
        if body <= min_body or prev_body <= min_body:
            return 0
        
        # Bearish engulfing
        if (df['Open'].iloc[idx - 1] < df['Close'].iloc[idx - 1] and  # Prev green
            df['Open'].iloc[idx] > df['Close'].iloc[idx] and  # Current red
            df['Open'].iloc[idx] >= df['Close'].iloc[idx - 1] and
            df['Close'].iloc[idx] < df['Open'].iloc[idx - 1]):
            return 1
        
        # Bullish engulfing
        if (df['Open'].iloc[idx - 1] > df['Close'].iloc[idx - 1] and  # Prev red
            df['Open'].iloc[idx] < df['Close'].iloc[idx] and  # Current green
            df['Open'].iloc[idx] <= df['Close'].iloc[idx - 1] and
            df['Close'].iloc[idx] > df['Open'].iloc[idx - 1]):
            return 2
        
        return 0
    
    def _is_star(self, df: pd.DataFrame, idx: int) -> int:
        """
        Check for shooting star / hammer pattern.
        Returns: 1 for shooting star (bearish), 2 for hammer (bullish), 0 for none
        """
        body = abs(df['Open'].iloc[idx] - df['Close'].iloc[idx])
        if body < 0.001:
            body = 0.001
        
        high_wick = df['High'].iloc[idx] - max(df['Open'].iloc[idx], df['Close'].iloc[idx])
        low_wick = min(df['Open'].iloc[idx], df['Close'].iloc[idx]) - df['Low'].iloc[idx]
        
        min_body = df['Close'].mean() * 0.001
        
        if body <= min_body:
            return 0
        
        # Shooting star (long upper wick)
        if high_wick / body > 1 and low_wick < 0.2 * high_wick:
            return 1
        
        # Hammer (long lower wick)
        if low_wick / body > 1 and high_wick < 0.2 * low_wick:
            return 2
        
        return 0
    
    def _close_to_resistance(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        resistance_levels: list, 
        proximity: float
    ) -> bool:
        """Check if price is close to resistance."""
        if not resistance_levels:
            return False
        
        price = df['High'].iloc[idx]
        closest = min(resistance_levels, key=lambda x: abs(x - price))
        limit = price * proximity
        
        c1 = abs(price - closest) <= limit
        c2 = abs(max(df['Open'].iloc[idx], df['Close'].iloc[idx]) - closest) <= limit
        c3 = min(df['Open'].iloc[idx], df['Close'].iloc[idx]) < closest
        c4 = df['Low'].iloc[idx] < closest
        
        return (c1 or c2) and c3 and c4
    
    def _close_to_support(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        support_levels: list, 
        proximity: float
    ) -> bool:
        """Check if price is close to support."""
        if not support_levels:
            return False
        
        price = df['Low'].iloc[idx]
        closest = min(support_levels, key=lambda x: abs(x - price))
        limit = price * proximity
        
        c1 = abs(price - closest) <= limit
        c2 = abs(min(df['Open'].iloc[idx], df['Close'].iloc[idx]) - closest) <= limit
        c3 = max(df['Open'].iloc[idx], df['Close'].iloc[idx]) > closest
        c4 = df['High'].iloc[idx] > closest
        
        return (c1 or c2) and c3 and c4
    
    def _generate_signals(
        self,
        df: pd.DataFrame,
        n1: int,
        n2: int,
        back_candles: int,
        level_proximity: float
    ) -> pd.DataFrame:
        """Generate trading signals based on S/R and patterns."""
        signals = df.copy()
        
        # Mark support and resistance levels
        signals['is_support'] = 0
        signals['is_resistance'] = 0
        signals['signals'] = 0
        signals['positions'] = 0
        signals['pattern_type'] = ''
        
        for i in range(n1, len(signals) - n2):
            if self._is_support(signals, i, n1, n2):
                signals.iloc[i, signals.columns.get_loc('is_support')] = 1
            if self._is_resistance(signals, i, n1, n2):
                signals.iloc[i, signals.columns.get_loc('is_resistance')] = 1
        
        # Generate trading signals
        for row in range(back_candles, len(signals) - n2):
            # Collect recent S/R levels
            support_levels = []
            resistance_levels = []
            
            for subrow in range(row - back_candles + n1, row + 1):
                if signals['is_support'].iloc[subrow] == 1:
                    support_levels.append(signals['Low'].iloc[subrow])
                if signals['is_resistance'].iloc[subrow] == 1:
                    resistance_levels.append(signals['High'].iloc[subrow])
            
            engulfing = self._is_engulfing(signals, row)
            star = self._is_star(signals, row)
            
            # Bearish signal near resistance
            if ((engulfing == 1 or star == 1) and 
                self._close_to_resistance(signals, row, resistance_levels, level_proximity)):
                signals.iloc[row, signals.columns.get_loc('signals')] = -1
                signals.iloc[row, signals.columns.get_loc('pattern_type')] = 'Bearish Engulfing' if engulfing == 1 else 'Shooting Star'
            
            # Bullish signal near support
            elif ((engulfing == 2 or star == 2) and 
                  self._close_to_support(signals, row, support_levels, level_proximity)):
                signals.iloc[row, signals.columns.get_loc('signals')] = 1
                signals.iloc[row, signals.columns.get_loc('pattern_type')] = 'Bullish Engulfing' if engulfing == 2 else 'Hammer'
        
        # Calculate positions
        signals['positions'] = signals['signals'].cumsum()
        signals['positions'] = signals['positions'].clip(0, 1)  # Long only, max 1 position
        
        return signals
    
    def _apply_sentiment(
        self,
        signals: pd.DataFrame,
        sentiment: dict
    ) -> pd.DataFrame:
        """Filter buy/sell signals based on sentiment direction."""
        return self._sentiment_filter_signals(
            signals, sentiment,
            neg_threshold=-0.5,
            pos_threshold=0.5,
            recalc_method='cumsum_clip'
        )
    
    def _calculate_portfolio(
        self,
        signals: pd.DataFrame,
        capital: float,
        risk: RiskParams
    ) -> pd.DataFrame:
        """Long-only portfolio calculation for S/R strategy."""
        return self._calculate_portfolio_long_only(signals, capital, risk)
    
    def _create_charts(
        self,
        signals: pd.DataFrame,
        ticker: str
    ) -> list[ChartData]:
        """Create visualization charts."""
        charts = []
        
        # Chart 1: Price with S/R levels and signals
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax1.plot(signals.index, signals['Close'], label='Price', color='blue', alpha=0.7)
        
        # Plot support levels
        support_levels = signals[signals['is_support'] == 1]
        for idx in support_levels.index:
            ax1.axhline(y=signals.loc[idx, 'Low'], color='green', linestyle='--', 
                       alpha=0.3, linewidth=1)
        ax1.scatter(support_levels.index, support_levels['Low'],
                   marker='_', color='green', s=100, label='Support', linewidths=3)
        
        # Plot resistance levels
        resistance_levels = signals[signals['is_resistance'] == 1]
        for idx in resistance_levels.index:
            ax1.axhline(y=signals.loc[idx, 'High'], color='red', linestyle='--', 
                       alpha=0.3, linewidth=1)
        ax1.scatter(resistance_levels.index, resistance_levels['High'],
                   marker='_', color='red', s=100, label='Resistance', linewidths=3)
        
        # Plot buy/sell signals
        buy_signals = signals[signals['signals'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       marker='^', color='lime', s=150, label='BUY', zorder=5,
                       edgecolors='black', linewidths=1)
        
        sell_signals = signals[signals['signals'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                       marker='v', color='salmon', s=150, label='SELL', zorder=5,
                       edgecolors='black', linewidths=1)
        
        ax1.set_title(f'{ticker} - Support/Resistance Strategy')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} S/R Levels & Signals",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Price chart with support/resistance levels and trading signals",
            ticker=ticker
        ))
        
        # Chart 2: S/R level distribution
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Support level histogram
        support_prices = signals[signals['is_support'] == 1]['Low']
        if not support_prices.empty:
            ax2.hist(support_prices, bins=20, color='green', alpha=0.7, edgecolor='black')
        ax2.set_title('Support Level Distribution')
        ax2.set_xlabel('Price')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Resistance level histogram
        resistance_prices = signals[signals['is_resistance'] == 1]['High']
        if not resistance_prices.empty:
            ax3.hist(resistance_prices, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax3.set_title('Resistance Level Distribution')
        ax3.set_xlabel('Price')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} S/R Distribution",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Distribution of support and resistance levels",
            ticker=ticker
        ))
        
        return charts


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = SupportResistanceStrategy()
    
    result = strategy.run(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000,
        n1=2,
        n2=2
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
