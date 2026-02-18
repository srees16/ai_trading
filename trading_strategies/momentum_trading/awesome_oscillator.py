"""
Awesome Oscillator Strategy.

The Awesome Oscillator is a momentum indicator that measures market momentum
by comparing recent market movements to historical average movements.

Formula: AO = SMA(median_price, 5) - SMA(median_price, 34)
where median_price = (High + Low) / 2

Strategy Rules:
- Long when AO crosses above zero (bullish momentum)
- Exit when AO crosses below zero (bearish momentum)

Reference: Bill Williams' technical analysis indicator
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
class AwesomeOscillatorStrategy(BaseStrategy):
    """
    Awesome Oscillator Strategy.
    
    This momentum strategy uses the difference between short and long simple
    moving averages of the median price to identify trend changes.
    
    Parameters:
        ao_short (int): Short period for Awesome Oscillator (default: 5)
        ao_long (int): Long period for Awesome Oscillator (default: 34)
    """
    
    name = "Awesome Oscillator"
    description = "Momentum strategy using Bill Williams' Awesome Oscillator"
    category = StrategyCategory.MOMENTUM
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 50
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "ao_short": {
                "type": "int",
                "default": 5,
                "min": 2,
                "max": 20,
                "description": "Short period for AO calculation"
            },
            "ao_long": {
                "type": "int",
                "default": 34,
                "min": 20,
                "max": 100,
                "description": "Long period for AO calculation"
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
        Execute the Awesome Oscillator backtest.
        
        Args:
            tickers: List of ticker symbols to analyze
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            capital: Initial capital
            sentiment_data: Optional sentiment scores from news analysis
            risk_params: Risk management parameters
            **kwargs: Additional parameters (ao_short, ao_long)
        
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
        ao_short = kwargs.get('ao_short', 5)
        ao_long = kwargs.get('ao_long', 34)
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
                signals = self._generate_signals(df, ao_short, ao_long)
                
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
                        'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2f}%",
                        'Total Trades': metrics.get('total_trades', 0)
                    })
            
            if perf_data:
                all_tables.append(TableData(
                    title="Performance Summary",
                    data=perf_data,
                    columns=['Ticker', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Total Trades'],
                    description="Summary of backtest performance for each ticker"
                ))
            
            # Combined signals table
            combined_df = pd.concat(combined_signals, ignore_index=True)
            recent_signals = combined_df[combined_df['signals'] != 0].tail(20)
            
            if not recent_signals.empty:
                signals_table = recent_signals[['ticker', 'Close', 'signals', 'awesome_oscillator']].copy()
                signals_table['Signal Type'] = signals_table['signals'].map({1: 'BUY', -1: 'SELL'})
                all_tables.append(TableData(
                    title="Recent Trading Signals",
                    data=signals_table.to_dict(orient='records'),
                    columns=['ticker', 'Close', 'Signal Type', 'awesome_oscillator'],
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
                    'ao_short': ao_short,
                    'ao_long': ao_long
                },
                'tickers_processed': len(combined_signals)
            }
        )
    
    def _generate_signals(
        self,
        df: pd.DataFrame,
        ao_short: int,
        ao_long: int
    ) -> pd.DataFrame:
        """Generate trading signals based on Awesome Oscillator logic."""
        signals = df.copy()
        
        # Calculate median price
        signals['median_price'] = (signals['High'] + signals['Low']) / 2
        
        # Calculate Awesome Oscillator
        # AO = SMA(median_price, short) - SMA(median_price, long)
        signals['sma_short'] = signals['median_price'].rolling(window=ao_short, min_periods=1).mean()
        signals['sma_long'] = signals['median_price'].rolling(window=ao_long, min_periods=1).mean()
        signals['awesome_oscillator'] = signals['sma_short'] - signals['sma_long']
        
        # Generate positions (1 = long, 0 = no position)
        signals['positions'] = 0
        signals.loc[signals.index[ao_long:], 'positions'] = np.where(
            signals['awesome_oscillator'].iloc[ao_long:] > 0, 
            1, 
            0
        )
        
        # Generate trading signals (difference shows entry/exit points)
        signals['signals'] = signals['positions'].diff().fillna(0)
        
        return signals
    
    def _apply_sentiment(
        self,
        signals: pd.DataFrame,
        sentiment: dict
    ) -> pd.DataFrame:
        """Scale Awesome Oscillator magnitude based on sentiment strength."""
        return self._sentiment_scale_indicator(
            signals, 'awesome_oscillator', sentiment
        )
    
    # _calculate_portfolio inherited from BaseStrategy
    
    def _create_charts(
        self,
        signals: pd.DataFrame,
        ticker: str
    ) -> list[ChartData]:
        """Create visualization charts."""
        charts = []
        
        # Chart 1: Price with trading signals
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.plot(signals.index, signals['Close'], label=ticker, color='blue', alpha=0.7)
        
        # Plot buy signals
        buy_signals = signals[signals['signals'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       marker='^', color='green', s=100, label='BUY', zorder=5)
        
        # Plot sell signals
        sell_signals = signals[signals['signals'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                       marker='v', color='red', s=100, label='SELL', zorder=5)
        
        ax1.set_title(f'{ticker} - Awesome Oscillator Trading Signals')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        charts.append(ChartData(
            title=f"{ticker} Price & Signals",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Price chart with buy/sell signals",
            ticker=ticker
        ))
        
        # Chart 2: Awesome Oscillator histogram
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Price subplot
        ax2.plot(signals.index, signals['Close'], label='Price', color='blue')
        ax2.set_title(f'{ticker} - Price')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # AO Histogram - color based on increasing/decreasing
        ao = signals['awesome_oscillator'].values
        ao_colors = []
        for i in range(len(ao)):
            if i == 0:
                ao_colors.append('green' if ao[i] >= 0 else 'red')
            else:
                if ao[i] >= 0:
                    ao_colors.append('lime' if ao[i] > ao[i-1] else 'green')
                else:
                    ao_colors.append('salmon' if ao[i] > ao[i-1] else 'red')
        
        ax3.bar(range(len(signals)), ao, color=ao_colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Awesome Oscillator')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Awesome Oscillator",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Price and Awesome Oscillator histogram",
            ticker=ticker
        ))
        
        return charts


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = AwesomeOscillatorStrategy()
    
    result = strategy.run(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000,
        ao_short=5,
        ao_long=34
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
