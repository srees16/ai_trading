"""
MACD Oscillator Strategy.

Moving Average Convergence/Divergence is a momentum indicator that shows
the relationship between two moving averages of a security's price.

Strategy Rules:
- Long when short MA crosses above long MA (positive momentum)
- Exit when short MA crosses below long MA

Reference: Classic technical analysis indicator developed by Gerald Appel
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
class MACDOscillatorStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence/Divergence) Oscillator Strategy.
    
    This momentum strategy compares short and long moving averages to
    identify trend direction and generate trading signals.
    
    Parameters:
        ma_short (int): Short moving average period (default: 10)
        ma_long (int): Long moving average period (default: 21)
        use_ema (bool): Use EMA instead of SMA (default: True)
    """
    
    name = "MACD Oscillator"
    description = "Momentum strategy using moving average crossovers to identify trends"
    category = StrategyCategory.MOMENTUM
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 50
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "ma_short": {
                "type": "int",
                "default": 10,
                "min": 2,
                "max": 50,
                "description": "Short moving average period"
            },
            "ma_long": {
                "type": "int",
                "default": 21,
                "min": 10,
                "max": 200,
                "description": "Long moving average period"
            },
            "use_ema": {
                "type": "bool",
                "default": True,
                "description": "Use Exponential MA instead of Simple MA"
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
        Execute the MACD Oscillator backtest.
        
        Args:
            tickers: List of ticker symbols to analyze
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            capital: Initial capital
            sentiment_data: Optional sentiment scores from news analysis
            risk_params: Risk management parameters
            **kwargs: Additional parameters (ma_short, ma_long, use_ema)
        
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
        ma_short = kwargs.get('ma_short', 10)
        ma_long = kwargs.get('ma_long', 21)
        use_ema = kwargs.get('use_ema', True)
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
                signals = self._generate_signals(df, ma_short, ma_long, use_ema)
                
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
                signals_table = recent_signals[['ticker', 'Close', 'signals', 'oscillator']].copy()
                signals_table['Signal Type'] = signals_table['signals'].map({1: 'BUY', -1: 'SELL'})
                all_tables.append(TableData(
                    title="Recent Trading Signals",
                    data=signals_table.to_dict(orient='records'),
                    columns=['ticker', 'Close', 'Signal Type', 'oscillator'],
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
                    'ma_short': ma_short,
                    'ma_long': ma_long,
                    'use_ema': use_ema
                },
                'tickers_processed': len(combined_signals)
            }
        )
    
    def _generate_signals(
        self,
        df: pd.DataFrame,
        ma_short: int,
        ma_long: int,
        use_ema: bool
    ) -> pd.DataFrame:
        """Generate trading signals based on MACD logic."""
        signals = df.copy()
        
        # Calculate moving averages
        if use_ema:
            signals['ma_short'] = signals['Close'].ewm(span=ma_short, adjust=False).mean()
            signals['ma_long'] = signals['Close'].ewm(span=ma_long, adjust=False).mean()
        else:
            signals['ma_short'] = signals['Close'].rolling(window=ma_short, min_periods=1).mean()
            signals['ma_long'] = signals['Close'].rolling(window=ma_long, min_periods=1).mean()
        
        # Calculate oscillator
        signals['oscillator'] = signals['ma_short'] - signals['ma_long']
        
        # Generate positions (1 = long, 0 = no position)
        signals['positions'] = 0
        signals.loc[signals.index[ma_short:], 'positions'] = np.where(
            signals['ma_short'].iloc[ma_short:] >= signals['ma_long'].iloc[ma_short:], 
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
        """Apply sentiment-based adjustments to trading signals."""
        sentiment_score = sentiment.get('score', 0)
        
        # Boost or reduce signal strength based on sentiment
        if abs(sentiment_score) > 0.5:
            # Strong sentiment - adjust oscillator magnitude
            multiplier = 1 + (sentiment_score * 0.2)
            signals['oscillator'] = signals['oscillator'] * multiplier
        
        return signals
    
    def _calculate_portfolio(
        self,
        signals: pd.DataFrame,
        capital: float,
        risk: RiskParams
    ) -> pd.DataFrame:
        """Calculate portfolio value over time."""
        portfolio = pd.DataFrame(index=signals.index)
        
        # Calculate position sizes (shares we can buy)
        max_position_value = capital * risk.max_position_size
        shares = int(max_position_value / signals['Close'].max()) if signals['Close'].max() > 0 else 0
        
        # Calculate holdings
        portfolio['positions'] = signals['positions']
        portfolio['Close'] = signals['Close']
        portfolio['holdings'] = signals['positions'] * signals['Close'] * shares
        
        # Calculate cash
        portfolio['cash'] = capital - (signals['signals'] * signals['Close'] * shares).cumsum()
        
        # Total portfolio value
        portfolio['total_value'] = portfolio['holdings'] + portfolio['cash']
        
        # Calculate returns
        portfolio['returns'] = portfolio['total_value'].pct_change().fillna(0)
        
        return portfolio
    
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
        
        ax1.set_title(f'{ticker} - MACD Trading Signals')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        charts.append(ChartData(
            title=f"{ticker} Price & Signals",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Price chart with buy/sell signals"
        ))
        
        # Chart 2: MACD Oscillator
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Moving averages
        ax2.plot(signals.index, signals['ma_short'], label='Short MA', color='blue')
        ax2.plot(signals.index, signals['ma_long'], label='Long MA', color='orange', linestyle='--')
        ax2.set_title(f'{ticker} - Moving Averages')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Oscillator
        colors = ['green' if x >= 0 else 'red' for x in signals['oscillator']]
        ax3.bar(range(len(signals)), signals['oscillator'], color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('MACD Oscillator')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} MACD Analysis",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Moving averages and MACD oscillator"
        ))
        
        return charts


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = MACDOscillatorStrategy()
    
    result = strategy.run(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000,
        ma_short=10,
        ma_long=21
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
