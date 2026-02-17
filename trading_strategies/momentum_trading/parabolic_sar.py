"""
Parabolic SAR Strategy.

The Parabolic Stop and Reverse (SAR) is a trend-following indicator that
provides potential entry and exit points based on trailing stop levels.

Formula:
- SAR = Prior SAR + AF × (EP – Prior SAR)
Where:
- AF = Acceleration Factor (starts at 0.02, increases by 0.02 each time EP changes, max 0.2)
- EP = Extreme Point (highest high or lowest low of current trend)

Strategy Rules:
- Long when price crosses above SAR
- Exit when price crosses below SAR

Reference: J. Welles Wilder Jr.'s technical analysis indicator
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
class ParabolicSARStrategy(BaseStrategy):
    """
    Parabolic Stop and Reverse (SAR) Strategy.
    
    This trend-following strategy uses a trailing stop system to identify
    trend direction and generate entry/exit signals.
    
    Parameters:
        af_start (float): Starting acceleration factor (default: 0.02)
        af_increment (float): AF increment step (default: 0.02)
        af_max (float): Maximum acceleration factor (default: 0.2)
    """
    
    name = "Parabolic SAR"
    description = "Trend-following strategy using Parabolic Stop and Reverse"
    category = StrategyCategory.MOMENTUM
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 30
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "af_start": {
                "type": "float",
                "default": 0.02,
                "min": 0.01,
                "max": 0.1,
                "description": "Starting acceleration factor"
            },
            "af_increment": {
                "type": "float",
                "default": 0.02,
                "min": 0.01,
                "max": 0.05,
                "description": "AF increment step"
            },
            "af_max": {
                "type": "float",
                "default": 0.2,
                "min": 0.1,
                "max": 0.5,
                "description": "Maximum acceleration factor"
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
        Execute the Parabolic SAR backtest.
        
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
        af_start = kwargs.get('af_start', 0.02)
        af_increment = kwargs.get('af_increment', 0.02)
        af_max = kwargs.get('af_max', 0.2)
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
                signals = self._generate_signals(df, af_start, af_increment, af_max)
                
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
                signals_table = recent_signals[['ticker', 'Close', 'signals', 'psar', 'trend']].copy()
                signals_table['Signal Type'] = signals_table['signals'].map({1: 'BUY', -1: 'SELL'})
                signals_table['Trend'] = signals_table['trend'].map({1: 'UPTREND', -1: 'DOWNTREND'})
                all_tables.append(TableData(
                    title="Recent Trading Signals",
                    data=signals_table.to_dict(orient='records'),
                    columns=['ticker', 'Close', 'Signal Type', 'psar', 'Trend'],
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
                    'af_start': af_start,
                    'af_increment': af_increment,
                    'af_max': af_max
                },
                'tickers_processed': len(combined_signals)
            }
        )
    
    def _calculate_psar(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        af_start: float,
        af_increment: float,
        af_max: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate Parabolic SAR values.
        
        Returns:
            Tuple of (psar values, trend direction: 1=up, -1=down)
        """
        n = len(close)
        psar = np.zeros(n)
        trend = np.zeros(n)
        af = np.zeros(n)
        ep = np.zeros(n)
        
        # Initialize first value
        # Assume downtrend initially if first close is below open
        psar[0] = high[0]
        trend[0] = -1
        af[0] = af_start
        ep[0] = low[0]
        
        for i in range(1, n):
            # Calculate SAR
            if trend[i-1] == 1:  # Uptrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                # Make sure SAR doesn't go above last two lows
                psar[i] = min(psar[i], low[i-1])
                if i > 1:
                    psar[i] = min(psar[i], low[i-2])
                
                # Check for trend reversal
                if low[i] < psar[i]:
                    trend[i] = -1
                    psar[i] = ep[i-1]  # Use previous EP as new SAR
                    af[i] = af_start
                    ep[i] = low[i]
                else:
                    trend[i] = 1
                    # Update EP and AF
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            
            else:  # Downtrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                # Make sure SAR doesn't go below last two highs
                psar[i] = max(psar[i], high[i-1])
                if i > 1:
                    psar[i] = max(psar[i], high[i-2])
                
                # Check for trend reversal
                if high[i] > psar[i]:
                    trend[i] = 1
                    psar[i] = ep[i-1]  # Use previous EP as new SAR
                    af[i] = af_start
                    ep[i] = high[i]
                else:
                    trend[i] = -1
                    # Update EP and AF
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return psar, trend
    
    def _generate_signals(
        self,
        df: pd.DataFrame,
        af_start: float,
        af_increment: float,
        af_max: float
    ) -> pd.DataFrame:
        """Generate trading signals based on Parabolic SAR."""
        signals = df.copy()
        
        # Calculate Parabolic SAR
        psar, trend = self._calculate_psar(
            signals['High'].values,
            signals['Low'].values,
            signals['Close'].values,
            af_start,
            af_increment,
            af_max
        )
        
        signals['psar'] = psar
        signals['trend'] = trend
        
        # Generate positions (1 = long when uptrend, 0 = no position)
        signals['positions'] = np.where(signals['trend'] == 1, 1, 0)
        
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
        
        # Use sentiment to filter signals
        # Only take long positions if sentiment confirms
        if sentiment_score < -0.7:
            signals['positions'] = 0
            signals['signals'] = signals['positions'].diff().fillna(0)
        
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
        
        # Chart 1: Price with Parabolic SAR
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax1.plot(signals.index, signals['Close'], label='Price', color='blue', alpha=0.7)
        
        # Plot SAR dots
        uptrend = signals[signals['trend'] == 1]
        downtrend = signals[signals['trend'] == -1]
        
        ax1.scatter(uptrend.index, uptrend['psar'], 
                   marker='.', color='green', s=30, label='SAR (Uptrend)', alpha=0.8)
        ax1.scatter(downtrend.index, downtrend['psar'], 
                   marker='.', color='red', s=30, label='SAR (Downtrend)', alpha=0.8)
        
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
        
        ax1.set_title(f'{ticker} - Parabolic SAR')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Price & Parabolic SAR",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Price chart with Parabolic SAR indicator"
        ))
        
        # Chart 2: Trend indicator
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Price
        ax2.plot(signals.index, signals['Close'], label='Price', color='blue')
        ax2.fill_between(
            signals.index,
            signals['Close'],
            where=signals['trend'] == 1,
            alpha=0.3,
            color='green',
            label='Uptrend'
        )
        ax2.fill_between(
            signals.index,
            signals['Close'],
            where=signals['trend'] == -1,
            alpha=0.3,
            color='red',
            label='Downtrend'
        )
        ax2.set_title(f'{ticker} - Trend Zones')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Trend indicator
        ax3.fill_between(
            signals.index,
            signals['trend'],
            0,
            where=signals['trend'] == 1,
            alpha=0.7,
            color='green',
            label='Uptrend'
        )
        ax3.fill_between(
            signals.index,
            signals['trend'],
            0,
            where=signals['trend'] == -1,
            alpha=0.7,
            color='red',
            label='Downtrend'
        )
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Trend Direction')
        ax3.set_xlabel('Date')
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(['Downtrend', 'Neutral', 'Uptrend'])
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Trend Analysis",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Trend zones and direction indicator"
        ))
        
        return charts


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = ParabolicSARStrategy()
    
    result = strategy.run(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000,
        af_start=0.02,
        af_increment=0.02,
        af_max=0.2
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
