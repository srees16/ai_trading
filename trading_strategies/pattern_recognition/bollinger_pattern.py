"""
Bollinger Bands Pattern Recognition Strategy.

Bollinger Bands are volatility bands placed above and below a moving average.
This strategy looks for the "Bottom W" pattern which indicates a potential
bullish reversal.

Pattern Structure (Bottom W):
- Node L: First top (starting point)
- Node K: First bottom (touches lower band)
- Node J: Middle node (touches middle band)
- Node M: Second bottom (above lower band, lower than K)
- Node I: Breakout above upper band

Strategy Rules:
- Buy when bottom W pattern is confirmed (price breaks upper band)
- Exit when bandwidth contracts (momentum fading)

Reference: John Bollinger's technical analysis indicator
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
class BollingerPatternStrategy(BaseStrategy):
    """
    Bollinger Bands Bottom W Pattern Strategy.
    
    This strategy identifies the classic "Bottom W" reversal pattern
    using Bollinger Bands for entry signals.
    
    Parameters:
        bb_period (int): Bollinger Bands period (default: 20)
        bb_std (float): Standard deviations for bands (default: 2.0)
        pattern_period (int): Lookback period for W pattern (default: 75)
        alpha (float): Price-band proximity threshold (default: 0.01)
    """
    
    name = "Bollinger Pattern"
    description = "Reversal strategy using Bollinger Bands Bottom W pattern"
    category = StrategyCategory.PATTERN_RECOGNITION
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 100
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "bb_period": {
                "type": "int",
                "default": 20,
                "min": 10,
                "max": 50,
                "description": "Bollinger Bands moving average period"
            },
            "bb_std": {
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 3.0,
                "description": "Standard deviations for bands"
            },
            "pattern_period": {
                "type": "int",
                "default": 75,
                "min": 30,
                "max": 150,
                "description": "Lookback period for W pattern detection"
            },
            "alpha": {
                "type": "float",
                "default": 0.01,
                "min": 0.005,
                "max": 0.05,
                "description": "Price proximity threshold to bands"
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
        Execute the Bollinger Pattern backtest.
        
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
        bb_period = kwargs.get('bb_period', 20)
        bb_std = kwargs.get('bb_std', 2.0)
        pattern_period = kwargs.get('pattern_period', 75)
        alpha = kwargs.get('alpha', 0.01)
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
                signals = self._generate_signals(df, bb_period, bb_std, pattern_period, alpha)
                
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
                metrics['patterns_found'] = len(signals[signals['pattern_confirmed'] == 1])
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
                        'W Patterns': metrics.get('patterns_found', 0)
                    })
            
            if perf_data:
                all_tables.append(TableData(
                    title="Performance Summary",
                    data=perf_data,
                    columns=['Ticker', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'W Patterns'],
                    description="Summary of backtest performance for each ticker"
                ))
            
            # Recent signals table
            combined_df = pd.concat(combined_signals, ignore_index=True)
            recent_signals = combined_df[combined_df['signals'] != 0].tail(20)
            
            if not recent_signals.empty:
                signals_table = recent_signals[['ticker', 'Close', 'signals', 'bandwidth']].copy()
                signals_table['Signal Type'] = signals_table['signals'].map({1: 'BUY (W Pattern)', -1: 'SELL (BB Contract)'})
                signals_table['bandwidth'] = signals_table['bandwidth'].round(4)
                all_tables.append(TableData(
                    title="Recent Trading Signals",
                    data=signals_table.to_dict(orient='records'),
                    columns=['ticker', 'Close', 'Signal Type', 'bandwidth'],
                    description="Most recent 20 trading signals"
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
                    'bb_period': bb_period,
                    'bb_std': bb_std,
                    'pattern_period': pattern_period,
                    'alpha': alpha
                },
                'tickers_processed': len(combined_signals)
            }
        )
    
    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int,
        num_std: float
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        signals = df.copy()
        
        # Calculate bands
        signals['std'] = signals['Close'].rolling(window=period, min_periods=period).std()
        signals['mid_band'] = signals['Close'].rolling(window=period, min_periods=period).mean()
        signals['upper_band'] = signals['mid_band'] + num_std * signals['std']
        signals['lower_band'] = signals['mid_band'] - num_std * signals['std']
        
        # Calculate bandwidth (normalized)
        signals['bandwidth'] = (signals['upper_band'] - signals['lower_band']) / signals['mid_band']
        
        # Calculate %B (position within bands)
        signals['percent_b'] = (signals['Close'] - signals['lower_band']) / (
            signals['upper_band'] - signals['lower_band']
        )
        
        return signals
    
    def _generate_signals(
        self,
        df: pd.DataFrame,
        bb_period: int,
        bb_std: float,
        pattern_period: int,
        alpha: float
    ) -> pd.DataFrame:
        """Generate trading signals based on Bottom W pattern."""
        signals = self._calculate_bollinger_bands(df, bb_period, bb_std)
        
        signals['signals'] = 0
        signals['positions'] = 0
        signals['pattern_confirmed'] = 0
        signals['coordinates'] = ''
        
        # Average bandwidth for exit condition
        avg_bandwidth = signals['bandwidth'].mean()
        beta = avg_bandwidth * 0.8  # Exit when bandwidth contracts below this
        
        for i in range(pattern_period, len(signals)):
            # Skip if already in position
            if signals['positions'].iloc[i-1] > 0:
                # Check for exit condition (bandwidth contraction)
                if signals['bandwidth'].iloc[i] < beta:
                    signals.iloc[i, signals.columns.get_loc('signals')] = -1
                    signals.iloc[i, signals.columns.get_loc('positions')] = 0
                else:
                    signals.iloc[i, signals.columns.get_loc('positions')] = 1
                continue
            
            # Look for Bottom W pattern
            # Condition 4: Price breaks above upper band
            if signals['Close'].iloc[i] > signals['upper_band'].iloc[i]:
                pattern_found = False
                
                # Search backward for pattern nodes
                # Find node J (middle - touches mid band)
                for j in range(i, i - pattern_period, -1):
                    price_j = signals['Close'].iloc[j]
                    mid_j = signals['mid_band'].iloc[j]
                    upper_i = signals['upper_band'].iloc[i]
                    
                    # Condition 2: Price near mid band and mid band near current upper
                    if (abs(mid_j - price_j) < alpha * price_j and 
                        abs(mid_j - upper_i) < alpha * upper_i):
                        
                        # Find node K (first bottom - touches lower band)
                        for k in range(j, i - pattern_period, -1):
                            price_k = signals['Close'].iloc[k]
                            lower_k = signals['lower_band'].iloc[k]
                            
                            # Condition 1: Price near lower band
                            if abs(lower_k - price_k) < alpha * price_k:
                                threshold = price_k
                                
                                # Find node L (first top - above mid band)
                                for l in range(k, i - pattern_period, -1):
                                    if signals['mid_band'].iloc[l] < signals['Close'].iloc[l]:
                                        
                                        # Find node M (second bottom)
                                        for m in range(i, j, -1):
                                            price_m = signals['Close'].iloc[m]
                                            lower_m = signals['lower_band'].iloc[m]
                                            
                                            # Condition 3: Price above lower band but below threshold
                                            if (price_m - lower_m < alpha * price_m and
                                                price_m > lower_m and
                                                price_m < threshold):
                                                
                                                # Pattern confirmed!
                                                pattern_found = True
                                                signals.iloc[i, signals.columns.get_loc('signals')] = 1
                                                signals.iloc[i, signals.columns.get_loc('positions')] = 1
                                                signals.iloc[i, signals.columns.get_loc('pattern_confirmed')] = 1
                                                signals.iloc[i, signals.columns.get_loc('coordinates')] = f'{l},{k},{j},{m},{i}'
                                                break
                                        
                                        if pattern_found:
                                            break
                                
                                if pattern_found:
                                    break
                        
                        if pattern_found:
                            break
                
                if pattern_found:
                    continue
        
        # Forward fill positions
        signals['positions'] = signals['positions'].replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def _apply_sentiment(
        self,
        signals: pd.DataFrame,
        sentiment: dict
    ) -> pd.DataFrame:
        """Filter bullish signals when sentiment is strongly negative."""
        return self._sentiment_filter_signals(
            signals, sentiment, neg_threshold=-0.7, recalc_method='cumsum_clip'
        )
    
    def _calculate_portfolio(
        self,
        signals: pd.DataFrame,
        capital: float,
        risk: RiskParams
    ) -> pd.DataFrame:
        """Long-only portfolio calculation for Bollinger pattern."""
        return self._calculate_portfolio_long_only(signals, capital, risk)
    
    def _create_charts(
        self,
        signals: pd.DataFrame,
        ticker: str
    ) -> list[ChartData]:
        """Create visualization charts."""
        charts = []
        
        # Chart 1: Price with Bollinger Bands and signals
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]},
                                         sharex=True)
        
        # Price and bands
        ax1.plot(signals.index, signals['Close'], label='Price', color='blue', alpha=0.7)
        ax1.plot(signals.index, signals['upper_band'], label='Upper Band', 
                color='red', linestyle='--', alpha=0.5)
        ax1.plot(signals.index, signals['mid_band'], label='Middle Band', 
                color='gray', linestyle='-', alpha=0.5)
        ax1.plot(signals.index, signals['lower_band'], label='Lower Band', 
                color='green', linestyle='--', alpha=0.5)
        
        # Fill between bands
        ax1.fill_between(signals.index, signals['upper_band'], signals['lower_band'],
                        alpha=0.1, color='blue')
        
        # Plot buy signals (W pattern)
        buy_signals = signals[signals['signals'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       marker='^', color='lime', s=150, label='BUY (W Pattern)', zorder=5,
                       edgecolors='black', linewidths=1)
        
        # Plot sell signals (contraction)
        sell_signals = signals[signals['signals'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                       marker='v', color='salmon', s=150, label='SELL (Contraction)', zorder=5,
                       edgecolors='black', linewidths=1)
        
        ax1.set_title(f'{ticker} - Bollinger Bands Bottom W Pattern')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Bandwidth plot
        ax2.fill_between(signals.index, 0, signals['bandwidth'], 
                        alpha=0.5, color='purple')
        ax2.axhline(y=signals['bandwidth'].mean() * 0.8, color='red', 
                   linestyle='--', label='Exit threshold')
        ax2.set_title('Bollinger Bandwidth')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Bandwidth')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} Bollinger Bands",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Price with Bollinger Bands and W pattern signals",
            ticker=ticker
        ))
        
        # Chart 2: %B indicator
        fig2, ax3 = plt.subplots(figsize=(12, 6))
        
        ax3.plot(signals.index, signals['percent_b'], label='%B', color='purple')
        ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Lower Band (0)')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Upper Band (1)')
        ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Middle (0.5)')
        
        ax3.fill_between(signals.index, 0, 1, alpha=0.1, color='gray')
        
        # Highlight pattern confirmations
        patterns = signals[signals['pattern_confirmed'] == 1]
        if not patterns.empty:
            ax3.scatter(patterns.index, patterns['percent_b'],
                       marker='*', color='gold', s=200, label='W Pattern Confirmed', zorder=5)
        
        ax3.set_title(f'{ticker} - Bollinger %B Indicator')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('%B Value')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker} %B Indicator",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Bollinger %B showing position within bands",
            ticker=ticker
        ))
        
        return charts


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = BollingerPatternStrategy()
    
    result = strategy.run(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000,
        bb_period=20,
        bb_std=2.0
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
