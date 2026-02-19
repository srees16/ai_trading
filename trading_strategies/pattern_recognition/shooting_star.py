"""
Shooting Star Candlestick Pattern Strategy.

The Shooting Star is a bearish reversal candlestick pattern that forms
after an uptrend. It has a small body near the low with a long upper wick.

Pattern Characteristics:
- Small real body near the low
- Long upper shadow (at least 2x the body)
- Little or no lower shadow
- Appears after an uptrend

Strategy Rules:
- Short when shooting star pattern is identified
- Exit based on stop loss or holding period

Reference: Japanese candlestick charting techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
class ShootingStarStrategy(BaseStrategy):
    """
    Shooting Star Candlestick Pattern Strategy.
    
    This bearish reversal strategy identifies shooting star patterns
    and generates short signals.
    
    Parameters:
        lower_bound (float): Max ratio of lower wick to body (default: 0.2)
        body_size (float): Max body size as multiple of average (default: 0.5)
        stop_threshold (float): Stop loss/profit threshold (default: 0.05)
        holding_period (int): Maximum holding period in days (default: 7)
    """
    
    name = "Shooting Star"
    description = "Bearish reversal strategy using shooting star candlestick pattern"
    category = StrategyCategory.PATTERN_RECOGNITION
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 20
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "lower_bound": {
                "type": "float",
                "default": 0.2,
                "min": 0.05,
                "max": 0.5,
                "description": "Max ratio of lower wick to body"
            },
            "body_size": {
                "type": "float",
                "default": 0.5,
                "min": 0.2,
                "max": 1.0,
                "description": "Max body size as multiple of average"
            },
            "stop_threshold": {
                "type": "float",
                "default": 0.05,
                "min": 0.01,
                "max": 0.15,
                "description": "Stop loss/profit threshold (5% = 0.05)"
            },
            "holding_period": {
                "type": "int",
                "default": 7,
                "min": 1,
                "max": 30,
                "description": "Maximum holding period in days"
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
        Execute the Shooting Star backtest.
        
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
        lower_bound = kwargs.get('lower_bound', 0.2)
        body_size = kwargs.get('body_size', 0.5)
        stop_threshold = kwargs.get('stop_threshold', 0.05)
        holding_period = kwargs.get('holding_period', 7)
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
                signals = self._generate_signals(
                    df, lower_bound, body_size, stop_threshold, holding_period
                )
                
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
                metrics['patterns_found'] = len(signals[signals['pattern'] == 1])
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
                        'Patterns Found': metrics.get('patterns_found', 0)
                    })
            
            if perf_data:
                all_tables.append(TableData(
                    title="Performance Summary",
                    data=perf_data,
                    columns=['Ticker', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Patterns Found'],
                    description="Summary of backtest performance for each ticker"
                ))
            
            # Pattern details table
            combined_df = pd.concat(combined_signals, ignore_index=True)
            patterns = combined_df[combined_df['pattern'] == 1].tail(20)
            
            if not patterns.empty:
                pattern_table = patterns[['ticker', 'Close', 'upper_wick_ratio', 'body_ratio']].copy()
                pattern_table['upper_wick_ratio'] = pattern_table['upper_wick_ratio'].round(2)
                pattern_table['body_ratio'] = pattern_table['body_ratio'].round(2)
                all_tables.append(TableData(
                    title="Recent Shooting Star Patterns",
                    data=pattern_table.to_dict(orient='records'),
                    columns=['ticker', 'Close', 'upper_wick_ratio', 'body_ratio'],
                    description="Most recent shooting star patterns detected"
                ))
        
        # Calculate aggregate metrics
        if all_metrics:
            valid_metrics = [m for m in all_metrics.values() if 'error' not in m]
            if valid_metrics:
                all_metrics['aggregate'] = {
                    'avg_return': np.mean([m.get('total_return', 0) for m in valid_metrics]),
                    'avg_sharpe': np.mean([m.get('sharpe_ratio', 0) for m in valid_metrics]),
                    'total_patterns': sum([m.get('patterns_found', 0) for m in valid_metrics]),
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
                    'lower_bound': lower_bound,
                    'body_size': body_size,
                    'stop_threshold': stop_threshold,
                    'holding_period': holding_period
                },
                'tickers_processed': len(combined_signals)
            }
        )
    
    def _identify_shooting_star(
        self,
        df: pd.DataFrame,
        lower_bound: float,
        body_size: float
    ) -> pd.DataFrame:
        """Identify shooting star patterns in candlestick data."""
        signals = df.copy()
        
        # Calculate candle components
        signals['body'] = abs(signals['Close'] - signals['Open'])
        signals['body'] = signals['body'].replace(0, 0.001)  # Avoid division by zero
        
        signals['upper_wick'] = signals['High'] - signals[['Open', 'Close']].max(axis=1)
        signals['lower_wick'] = signals[['Open', 'Close']].min(axis=1) - signals['Low']
        
        # Calculate ratios
        signals['upper_wick_ratio'] = signals['upper_wick'] / signals['body']
        signals['lower_wick_ratio'] = signals['lower_wick'] / signals['body']
        
        avg_body = signals['body'].rolling(window=20, min_periods=1).mean()
        signals['body_ratio'] = signals['body'] / avg_body
        
        # Shooting star conditions
        # 1: Open > Close (red candle) or close to equal
        signals['c1'] = signals['Open'] >= signals['Close']
        
        # 2: Little or no lower wick
        signals['c2'] = signals['lower_wick'] < lower_bound * signals['body']
        
        # 3: Small body
        signals['c3'] = signals['body'] < avg_body * body_size
        
        # 4: Long upper wick (at least 2x body)
        signals['c4'] = signals['upper_wick'] >= 2 * signals['body']
        
        # 5: Price uptrend (higher close than previous 2 days)
        signals['c5'] = signals['Close'] >= signals['Close'].shift(1)
        signals['c6'] = signals['Close'].shift(1) >= signals['Close'].shift(2)
        
        # 6: Confirmation (next candle's high below shooting star high)
        signals['c7'] = signals['High'].shift(-1) <= signals['High']
        
        # 7: Confirmation (next candle's close below shooting star close)
        signals['c8'] = signals['Close'].shift(-1) <= signals['Close']
        
        # All conditions met = shooting star pattern
        signals['pattern'] = (
            signals['c1'] & signals['c2'] & signals['c3'] & signals['c4'] &
            signals['c5'] & signals['c6'] & signals['c7'] & signals['c8']
        ).astype(int)
        
        return signals
    
    def _generate_signals(
        self,
        df: pd.DataFrame,
        lower_bound: float,
        body_size: float,
        stop_threshold: float,
        holding_period: int
    ) -> pd.DataFrame:
        """Generate trading signals based on shooting star patterns."""
        signals = self._identify_shooting_star(df, lower_bound, body_size)
        
        signals['signals'] = 0
        signals['positions'] = 0
        
        # Generate entry signals (short on shooting star)
        pattern_indices = signals[signals['pattern'] == 1].index.tolist()
        
        for entry_idx in pattern_indices:
            try:
                entry_pos = signals.index.get_loc(entry_idx)
                entry_price = signals['Close'].iloc[entry_pos]
                
                if entry_pos + 1 >= len(signals):
                    continue
                
                # Mark entry signal
                signals.iloc[entry_pos, signals.columns.get_loc('signals')] = -1
                
                # Find exit
                for i in range(1, min(holding_period + 1, len(signals) - entry_pos)):
                    exit_pos = entry_pos + i
                    current_price = signals['Close'].iloc[exit_pos]
                    
                    # Check stop loss/profit
                    price_change = (current_price - entry_price) / entry_price
                    
                    if abs(price_change) > stop_threshold or i == holding_period:
                        signals.iloc[exit_pos, signals.columns.get_loc('signals')] = 1
                        
                        # Mark position
                        for j in range(entry_pos, exit_pos + 1):
                            signals.iloc[j, signals.columns.get_loc('positions')] = -1
                        break
                        
            except Exception:
                continue
        
        return signals
    
    def _apply_sentiment(
        self,
        signals: pd.DataFrame,
        sentiment: dict
    ) -> pd.DataFrame:
        """Filter bearish signals when sentiment is positive."""
        return self._sentiment_filter_signals(
            signals, sentiment, pos_threshold=0.5, recalc_method='cumsum'
        )
    
    def _calculate_portfolio(
        self,
        signals: pd.DataFrame,
        capital: float,
        risk: RiskParams
    ) -> pd.DataFrame:
        """Calculate portfolio value over time."""
        portfolio = pd.DataFrame(index=signals.index)
        
        # For a short strategy, we track position value differently
        portfolio['positions'] = signals['positions']
        portfolio['Close'] = signals['Close']
        
        # Calculate position value (short positions profit when price goes down)
        position_changes = signals['positions'].diff().fillna(0)
        
        # Track cumulative P&L
        portfolio['pnl'] = 0.0
        current_entry = 0
        
        for i in range(1, len(portfolio)):
            if signals['positions'].iloc[i] == -1 and signals['positions'].iloc[i-1] != -1:
                # Entry
                current_entry = signals['Close'].iloc[i]
            elif signals['positions'].iloc[i] != -1 and signals['positions'].iloc[i-1] == -1:
                # Exit
                if current_entry > 0:
                    pnl = (current_entry - signals['Close'].iloc[i]) / current_entry
                    portfolio.iloc[i, portfolio.columns.get_loc('pnl')] = pnl * capital * risk.max_position_size
        
        # Cumulative value
        portfolio['total_value'] = capital + portfolio['pnl'].cumsum()
        portfolio['returns'] = portfolio['total_value'].pct_change().fillna(0)
        
        return portfolio
    
    def _create_charts(
        self,
        signals: pd.DataFrame,
        ticker: str
    ) -> list[ChartData]:
        """Create visualization charts."""
        charts = []
        
        # Chart 1: Candlestick chart with patterns
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot candlesticks (simplified)
        for i in range(len(signals)):
            o, h, l, c = signals['Open'].iloc[i], signals['High'].iloc[i], signals['Low'].iloc[i], signals['Close'].iloc[i]
            color = 'green' if c >= o else 'red'
            
            # Wick
            ax1.plot([i, i], [l, h], color='black', linewidth=0.5)
            
            # Body
            body_bottom = min(o, c)
            body_height = abs(c - o)
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                            facecolor=color, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)
        
        # Highlight shooting star patterns
        patterns = signals[signals['pattern'] == 1]
        if not patterns.empty:
            pattern_indices = [signals.index.get_loc(idx) for idx in patterns.index]
            ax1.scatter(pattern_indices, patterns['High'] * 1.02,
                       marker='*', color='orange', s=200, label='Shooting Star', zorder=5)
        
        # Plot entry/exit signals
        entry_signals = signals[signals['signals'] == -1]
        if not entry_signals.empty:
            entry_indices = [signals.index.get_loc(idx) for idx in entry_signals.index]
            ax1.scatter(entry_indices, entry_signals['Close'],
                       marker='v', color='purple', s=100, label='SHORT', zorder=6)
        
        exit_signals = signals[signals['signals'] == 1]
        if not exit_signals.empty:
            exit_indices = [signals.index.get_loc(idx) for idx in exit_signals.index]
            ax1.scatter(exit_indices, exit_signals['Close'],
                       marker='^', color='blue', s=100, label='COVER', zorder=6)
        
        ax1.set_title(f'{ticker} - Shooting Star Pattern Recognition')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis labels
        tick_spacing = max(1, len(signals) // 10)
        ax1.set_xticks(range(0, len(signals), tick_spacing))
        ax1.set_xticklabels(
            [signals.index[i].strftime('%Y-%m-%d') for i in range(0, len(signals), tick_spacing)],
            rotation=45
        )
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Candlestick & Patterns",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Candlestick chart with shooting star patterns highlighted",
            ticker=ticker
        ))
        
        # Chart 2: Pattern characteristics
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Upper wick ratios over time
        ax2.bar(range(len(signals)), signals['upper_wick_ratio'].clip(0, 10), 
               color='blue', alpha=0.5)
        ax2.axhline(y=2, color='red', linestyle='--', label='Min ratio for pattern')
        ax2.set_title('Upper Wick to Body Ratio')
        ax2.set_ylabel('Ratio')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Body ratio over time
        ax3.bar(range(len(signals)), signals['body_ratio'].clip(0, 3), 
               color='green', alpha=0.5)
        ax3.axhline(y=0.5, color='red', linestyle='--', label='Max ratio for pattern')
        ax3.set_title('Body Size Ratio (vs 20-day avg)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Ratio')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Pattern Analysis",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Candlestick pattern characteristic metrics",
            ticker=ticker
        ))
        
        return charts


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = ShootingStarStrategy()
    
    result = strategy.run(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000,
        lower_bound=0.2,
        body_size=0.5
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
