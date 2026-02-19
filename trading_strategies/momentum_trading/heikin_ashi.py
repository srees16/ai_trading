"""
Heikin-Ashi Strategy.

Heikin-Ashi is a Japanese candlestick technique that uses modified
OHLC values to filter out market noise and identify trends more clearly.

Heikin-Ashi Formulas:
- HA Close = (Open + High + Low + Close) / 4
- HA Open = (previous HA Open + previous HA Close) / 2
- HA High = max(High, HA Open, HA Close)
- HA Low = min(Low, HA Open, HA Close)

Strategy Rules:
- Long when HA candle changes from red to green (bullish reversal)
- Exit when HA candle changes from green to red (bearish reversal)

Reference: Classic Japanese candlestick technique
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
class HeikinAshiStrategy(BaseStrategy):
    """
    Heikin-Ashi Candlestick Strategy.
    
    This trend-following strategy uses Heikin-Ashi candlesticks to identify
    trends and reversals with reduced market noise.
    
    Parameters:
        confirmation_candles (int): Number of candles to confirm trend (default: 1)
        use_ma_filter (bool): Use MA filter for additional confirmation (default: False)
        ma_period (int): Moving average period for filter (default: 20)
    """
    
    name = "Heikin-Ashi"
    description = "Trend-following strategy using Heikin-Ashi candlesticks"
    category = StrategyCategory.MOMENTUM
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 30
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "confirmation_candles": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 5,
                "description": "Number of candles to confirm trend change"
            },
            "use_ma_filter": {
                "type": "bool",
                "default": False,
                "description": "Use moving average filter for additional confirmation"
            },
            "ma_period": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "description": "Moving average period for filter"
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
        Execute the Heikin-Ashi backtest.
        
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
        confirmation_candles = kwargs.get('confirmation_candles', 1)
        use_ma_filter = kwargs.get('use_ma_filter', False)
        ma_period = kwargs.get('ma_period', 20)
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
                    df, confirmation_candles, use_ma_filter, ma_period
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
            
            # HA candle analysis table
            combined_df = pd.concat(combined_signals, ignore_index=True)
            recent = combined_df[combined_df['signals'] != 0].tail(20)
            
            if not recent.empty:
                signals_table = recent[['ticker', 'Close', 'signals', 'ha_color']].copy()
                signals_table['Signal Type'] = signals_table['signals'].map({1: 'BUY', -1: 'SELL'})
                all_tables.append(TableData(
                    title="Recent Trading Signals",
                    data=signals_table.to_dict(orient='records'),
                    columns=['ticker', 'Close', 'Signal Type', 'ha_color'],
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
                    'confirmation_candles': confirmation_candles,
                    'use_ma_filter': use_ma_filter,
                    'ma_period': ma_period
                },
                'tickers_processed': len(combined_signals)
            }
        )
    
    def _generate_signals(
        self,
        df: pd.DataFrame,
        confirmation_candles: int,
        use_ma_filter: bool,
        ma_period: int
    ) -> pd.DataFrame:
        """Generate trading signals based on Heikin-Ashi candles."""
        signals = df.copy()
        
        # Calculate Heikin-Ashi candles
        signals['ha_close'] = (signals['Open'] + signals['High'] + 
                               signals['Low'] + signals['Close']) / 4
        
        # For first HA candle, use regular open
        signals['ha_open'] = signals['Open'].iloc[0]
        
        # Calculate HA Open iteratively
        ha_open_list = [signals['Open'].iloc[0]]
        for i in range(1, len(signals)):
            ha_open = (ha_open_list[i-1] + signals['ha_close'].iloc[i-1]) / 2
            ha_open_list.append(ha_open)
        
        signals['ha_open'] = ha_open_list
        
        # HA High and Low
        signals['ha_high'] = signals[['High', 'ha_open', 'ha_close']].max(axis=1)
        signals['ha_low'] = signals[['Low', 'ha_open', 'ha_close']].min(axis=1)
        
        # Determine candle color (1 = green/bullish, -1 = red/bearish)
        signals['ha_color'] = np.where(signals['ha_close'] >= signals['ha_open'], 1, -1)
        
        # Optional MA filter
        if use_ma_filter:
            signals['ma'] = signals['Close'].rolling(window=ma_period, min_periods=1).mean()
            signals['above_ma'] = signals['Close'] > signals['ma']
        
        # Generate positions based on HA color changes
        signals['positions'] = 0
        
        # Simple approach: be long when HA is green
        signals['positions'] = np.where(signals['ha_color'] == 1, 1, 0)
        
        # Apply MA filter if enabled
        if use_ma_filter:
            signals['positions'] = np.where(
                (signals['ha_color'] == 1) & (signals['above_ma']), 
                1, 
                0
            )
        
        # Apply confirmation (require X consecutive candles of same color)
        if confirmation_candles > 1:
            signals['color_count'] = 0
            count = 0
            prev_color = 0
            counts = []
            
            for i, color in enumerate(signals['ha_color']):
                if color == prev_color:
                    count += 1
                else:
                    count = 1
                counts.append(count)
                prev_color = color
            
            signals['color_count'] = counts
            
            # Only enter position after confirmation
            signals['positions'] = np.where(
                (signals['ha_color'] == 1) & (signals['color_count'] >= confirmation_candles),
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
        """Zero positions when sentiment is strongly negative."""
        return self._sentiment_zero_positions(signals, sentiment)
    
    # _calculate_portfolio inherited from BaseStrategy
    
    def _create_charts(
        self,
        signals: pd.DataFrame,
        ticker: str
    ) -> list[ChartData]:
        """Create visualization charts."""
        charts = []
        
        # Chart 1: Heikin-Ashi candlestick chart
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot Heikin-Ashi candles
        width = 0.8
        for i in range(len(signals)):
            idx = i
            o = signals['ha_open'].iloc[i]
            h = signals['ha_high'].iloc[i]
            l = signals['ha_low'].iloc[i]
            c = signals['ha_close'].iloc[i]
            
            # Determine color
            color = 'green' if c >= o else 'red'
            
            # Draw the wick
            ax1.plot([idx, idx], [l, h], color=color, linewidth=1)
            
            # Draw the body
            body_bottom = min(o, c)
            body_height = abs(c - o)
            rect = Rectangle(
                (idx - width/2, body_bottom), width, body_height,
                facecolor=color, edgecolor=color
            )
            ax1.add_patch(rect)
        
        # Plot buy/sell signals
        buy_signals = signals[signals['signals'] == 1]
        if not buy_signals.empty:
            buy_indices = [signals.index.get_loc(idx) for idx in buy_signals.index]
            ax1.scatter(buy_indices, buy_signals['ha_low'] * 0.98, 
                       marker='^', color='blue', s=100, label='BUY', zorder=5)
        
        sell_signals = signals[signals['signals'] == -1]
        if not sell_signals.empty:
            sell_indices = [signals.index.get_loc(idx) for idx in sell_signals.index]
            ax1.scatter(sell_indices, sell_signals['ha_high'] * 1.02,
                       marker='v', color='purple', s=100, label='SELL', zorder=5)
        
        ax1.set_title(f'{ticker} - Heikin-Ashi Candlesticks')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis labels (show every 20th date)
        tick_spacing = max(1, len(signals) // 10)
        ax1.set_xticks(range(0, len(signals), tick_spacing))
        ax1.set_xticklabels(
            [signals.index[i].strftime('%Y-%m-%d') for i in range(0, len(signals), tick_spacing)],
            rotation=45
        )
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Heikin-Ashi Chart",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Heikin-Ashi candlestick chart with signals",
            ticker=ticker
        ))
        
        # Chart 2: Regular price vs HA price comparison
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Regular price
        ax2.plot(range(len(signals)), signals['Close'], label='Regular Close', color='blue')
        ax2.set_title(f'{ticker} - Regular Close Price')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # HA Close price
        ax3.plot(range(len(signals)), signals['ha_close'], label='HA Close', color='orange')
        ax3.set_title('Heikin-Ashi Close Price (Smoothed)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Date')
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Price Comparison",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Comparison of regular vs Heikin-Ashi prices",
            ticker=ticker
        ))
        
        return charts


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = HeikinAshiStrategy()
    
    result = strategy.run(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000,
        confirmation_candles=1
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
