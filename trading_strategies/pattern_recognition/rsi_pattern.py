"""
RSI Pattern Recognition Strategy.

The Relative Strength Index (RSI) is a momentum oscillator measuring
the speed and magnitude of price movements on a 0-100 scale.

Strategy Rules:
- RSI < 30 (oversold) → Buy signal
- RSI > 70 (overbought) → Sell signal

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
class RSIPatternStrategy(BaseStrategy):
    """
    RSI Overbought/Oversold Pattern Strategy.
    
    This strategy uses RSI levels to identify overbought and oversold
    conditions for mean-reversion trading.
    
    Parameters:
        rsi_period (int): RSI calculation period (default: 14)
        oversold_threshold (int): Level below which stock is oversold (default: 30)
        overbought_threshold (int): Level above which stock is overbought (default: 70)
    """
    
    name = "RSI Pattern"
    description = "Mean-reversion strategy using RSI overbought/oversold signals"
    category = StrategyCategory.PATTERN_RECOGNITION
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 30
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "rsi_period": {
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 50,
                "description": "RSI calculation period"
            },
            "oversold_threshold": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 50,
                "description": "RSI level for oversold condition"
            },
            "overbought_threshold": {
                "type": "int",
                "default": 70,
                "min": 50,
                "max": 90,
                "description": "RSI level for overbought condition"
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
        Execute the RSI Pattern backtest.
        
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
        rsi_period = kwargs.get('rsi_period', 14)
        oversold = kwargs.get('oversold_threshold', 30)
        overbought = kwargs.get('overbought_threshold', 70)
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
                signals = self._generate_signals(df, rsi_period, oversold, overbought)
                
                # Apply sentiment adjustment if available
                if sentiment_data and ticker in sentiment_data:
                    signals = self._apply_sentiment(signals, sentiment_data[ticker])
                
                # Calculate portfolio
                portfolio = self._calculate_portfolio(signals, capital, risk)
                
                # Create charts for this ticker
                charts = self._create_charts(signals, ticker, oversold, overbought)
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
            
            # RSI extremes table
            combined_df = pd.concat(combined_signals, ignore_index=True)
            recent_signals = combined_df[combined_df['signals'] != 0].tail(20)
            
            if not recent_signals.empty:
                signals_table = recent_signals[['ticker', 'Close', 'signals', 'rsi']].copy()
                signals_table['Signal Type'] = signals_table['signals'].map({1: 'BUY (Oversold)', -1: 'SELL (Overbought)'})
                signals_table['rsi'] = signals_table['rsi'].round(2)
                all_tables.append(TableData(
                    title="Recent Trading Signals",
                    data=signals_table.to_dict(orient='records'),
                    columns=['ticker', 'Close', 'Signal Type', 'rsi'],
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
                    'rsi_period': rsi_period,
                    'oversold_threshold': oversold,
                    'overbought_threshold': overbought
                },
                'tickers_processed': len(combined_signals)
            }
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI using Smoothed Moving Average (authentic method).
        
        Args:
            prices: Price series
            period: RSI calculation period
        
        Returns:
            RSI values as pandas Series
        """
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Use smoothed moving average (Wilder's method)
        avg_gains = gains.ewm(com=period-1, min_periods=period).mean()
        avg_losses = losses.ewm(com=period-1, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rs = rs.replace([np.inf, -np.inf], 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI for insufficient data
    
    def _generate_signals(
        self,
        df: pd.DataFrame,
        rsi_period: int,
        oversold: int,
        overbought: int
    ) -> pd.DataFrame:
        """Generate trading signals based on RSI levels."""
        signals = df.copy()
        
        # Calculate RSI
        signals['rsi'] = self._calculate_rsi(signals['Close'], rsi_period)
        
        # Generate positions
        # 1 = long when oversold, -1 = short when overbought, 0 = neutral
        signals['positions'] = np.select(
            [signals['rsi'] < oversold, signals['rsi'] > overbought],
            [1, -1],
            default=0
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
        
        # Use sentiment to filter signals
        # Don't go long if sentiment is very negative
        if sentiment_score < -0.7:
            signals.loc[signals['positions'] == 1, 'positions'] = 0
            signals['signals'] = signals['positions'].diff().fillna(0)
        
        # Don't go short if sentiment is very positive
        if sentiment_score > 0.7:
            signals.loc[signals['positions'] == -1, 'positions'] = 0
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
        
        # Long-only version for simplicity
        long_positions = signals['positions'].apply(lambda x: max(0, x))
        
        # Calculate position sizes (shares we can buy)
        max_position_value = capital * risk.max_position_size
        shares = int(max_position_value / signals['Close'].max()) if signals['Close'].max() > 0 else 0
        
        # Calculate holdings
        portfolio['positions'] = long_positions
        portfolio['Close'] = signals['Close']
        portfolio['holdings'] = long_positions * signals['Close'] * shares
        
        # Calculate cash (only accounting for long trades)
        long_signals = signals['signals'].apply(lambda x: max(0, x))
        portfolio['cash'] = capital - (long_signals * signals['Close'] * shares).cumsum()
        
        # Total portfolio value
        portfolio['total_value'] = portfolio['holdings'] + portfolio['cash']
        
        # Calculate returns
        portfolio['returns'] = portfolio['total_value'].pct_change().fillna(0)
        
        return portfolio
    
    def _create_charts(
        self,
        signals: pd.DataFrame,
        ticker: str,
        oversold: int,
        overbought: int
    ) -> list[ChartData]:
        """Create visualization charts."""
        charts = []
        
        # Chart 1: Price with trading signals
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                         gridspec_kw={'height_ratios': [2, 1]},
                                         sharex=True)
        
        # Price plot
        ax1.plot(signals.index, signals['Close'], label=ticker, color='blue', alpha=0.7)
        
        # Plot buy signals (oversold)
        buy_signals = signals[signals['signals'] > 0]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       marker='^', color='green', s=100, label='BUY (Oversold)', zorder=5)
        
        # Plot sell signals (overbought)
        sell_signals = signals[signals['signals'] < 0]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                       marker='v', color='red', s=100, label='SELL (Overbought)', zorder=5)
        
        ax1.set_title(f'{ticker} - RSI Trading Signals')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # RSI plot
        ax2.plot(signals.index, signals['rsi'], label='RSI', color='purple')
        ax2.axhline(y=oversold, color='green', linestyle='--', alpha=0.7, label=f'Oversold ({oversold})')
        ax2.axhline(y=overbought, color='red', linestyle='--', alpha=0.7, label=f'Overbought ({overbought})')
        ax2.fill_between(signals.index, oversold, overbought, alpha=0.2, color='gray')
        
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Price & RSI",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Price chart with RSI indicator and signals"
        ))
        
        # Chart 2: RSI histogram/distribution
        fig2, ax3 = plt.subplots(figsize=(10, 6))
        
        ax3.hist(signals['rsi'].dropna(), bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax3.axvline(x=oversold, color='green', linestyle='--', linewidth=2, label=f'Oversold ({oversold})')
        ax3.axvline(x=overbought, color='red', linestyle='--', linewidth=2, label=f'Overbought ({overbought})')
        ax3.axvline(x=50, color='gray', linestyle='-', linewidth=1, label='Neutral (50)')
        
        ax3.set_title(f'{ticker} - RSI Distribution')
        ax3.set_xlabel('RSI Value')
        ax3.set_ylabel('Frequency')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} RSI Distribution",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Distribution of RSI values over the period"
        ))
        
        return charts


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = RSIPatternStrategy()
    
    result = strategy.run(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
