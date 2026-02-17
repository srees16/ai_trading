"""
Pairs Trading (Statistical Arbitrage) Strategy.

Pairs trading is a market-neutral strategy that exploits the mean-reverting
relationship between two cointegrated securities.

Strategy Components:
1. Cointegration test (Engle-Granger two-step method)
2. Z-score calculation for spread
3. Entry when spread exceeds threshold, exit on mean reversion

Strategy Rules:
- Test for cointegration over rolling window
- When Z-score > upper_threshold: Short Asset2, Long Asset1
- When Z-score < lower_threshold: Long Asset2, Short Asset1
- Exit when spread returns to mean (Z-score near 0)

Reference: Statistical arbitrage and cointegration analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime
import time
import warnings

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

# Suppress statsmodels warnings
warnings.filterwarnings('ignore')


@StrategyRegistry.register_decorator
class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading (Statistical Arbitrage) Strategy.
    
    This market-neutral strategy identifies cointegrated pairs and trades
    the spread when it deviates from its historical mean.
    
    Parameters:
        bandwidth (int): Rolling window for cointegration test (default: 250)
        z_entry (float): Z-score threshold for entry (default: 1.0)
        z_exit (float): Z-score threshold for exit (default: 0.0)
    
    Note: Requires exactly 2 tickers to form a pair.
    """
    
    name = "Pairs Trading"
    description = "Market-neutral strategy trading cointegrated security pairs"
    category = StrategyCategory.STATISTICAL_ARBITRAGE
    version = "2.0.0"
    author = "Centurion Capital"
    requires_sentiment = False
    min_data_points = 100  # Reduced to support shorter periods
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """Get strategy-specific parameters."""
        return {
            "bandwidth": {
                "type": "int",
                "default": 60,
                "min": 30,
                "max": 500,
                "description": "Rolling window for cointegration test"
            },
            "z_entry": {
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 3.0,
                "description": "Z-score threshold for entry"
            },
            "z_exit": {
                "type": "float",
                "default": 0.0,
                "min": -0.5,
                "max": 1.0,
                "description": "Z-score threshold for exit"
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
        Execute the Pairs Trading backtest.
        
        Args:
            tickers: List of exactly 2 ticker symbols forming the pair
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
        
        # Pairs trading requires exactly 2 tickers
        if len(tickers) < 2:
            return StrategyResult(
                success=False,
                error_message="Pairs trading requires exactly 2 tickers"
            )
        
        # Use first two tickers
        ticker1, ticker2 = tickers[0], tickers[1]
        
        # Parse parameters
        bandwidth = kwargs.get('bandwidth', 60)
        z_entry = kwargs.get('z_entry', 1.0)
        z_exit = kwargs.get('z_exit', 0.0)
        risk = self.get_risk_params(risk_params)
        
        # Initialize data service
        data_service = DataService()
        
        try:
            # Fetch data for both assets
            df1 = data_service.get_ohlcv(ticker1, start_date, end_date)
            df2 = data_service.get_ohlcv(ticker2, start_date, end_date)
            
            if df1.empty or df2.empty:
                return StrategyResult(
                    success=False,
                    error_message="Failed to fetch data for one or both tickers"
                )
            
            # Align dates
            common_idx = df1.index.intersection(df2.index)
            df1 = df1.loc[common_idx]
            df2 = df2.loc[common_idx]
            
            # Need at least bandwidth + some extra for signals
            min_required = max(bandwidth + 20, self.min_data_points)
            if len(df1) < min_required:
                return StrategyResult(
                    success=False,
                    error_message=f"Insufficient data points. Need {min_required}, got {len(df1)}. Try a longer period or smaller bandwidth."
                )
            
            # Generate signals
            signals = self._generate_signals(
                df1, df2, ticker1, ticker2, bandwidth, z_entry, z_exit
            )
            
            # Calculate portfolio
            portfolio = self._calculate_portfolio(signals, capital, risk)
            
            # Create charts
            charts = self._create_charts(signals, ticker1, ticker2)
            
            # Create tables
            tables = self._create_tables(signals, ticker1, ticker2)
            
            # Calculate metrics
            metrics = self.calculate_metrics(portfolio, signals, capital)
            metrics['pair'] = f"{ticker1}/{ticker2}"
            metrics['cointegration_periods'] = int(signals['cointegrated'].sum())
            metrics['total_periods'] = len(signals)
            
        except Exception as e:
            return StrategyResult(
                success=False,
                error_message=str(e)
            )
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            charts=charts,
            tables=tables,
            metrics=metrics,
            signals=signals,
            portfolio=portfolio,
            success=True,
            error_message="",
            execution_time=execution_time,
            metadata={
                'strategy': self.name,
                'parameters': {
                    'bandwidth': bandwidth,
                    'z_entry': z_entry,
                    'z_exit': z_exit
                },
                'pair': f"{ticker1}/{ticker2}"
            }
        )
    
    def _engle_granger_test(
        self,
        X: pd.Series,
        Y: pd.Series
    ) -> tuple[bool, Optional[np.ndarray]]:
        """
        Perform Engle-Granger two-step cointegration test.
        
        Args:
            X: First asset price series
            Y: Second asset price series
        
        Returns:
            Tuple of (is_cointegrated, model_params)
        """
        try:
            # Try to import statsmodels
            import statsmodels.api as sm
            
            # Step 1: Estimate long-run equilibrium
            X_const = sm.add_constant(X)
            model = sm.OLS(Y, X_const).fit()
            residuals = model.resid
            
            # ADF test on residuals
            adf_result = sm.tsa.stattools.adfuller(residuals)
            
            # Check if p-value <= 0.05 (stationary residuals = cointegration)
            if adf_result[1] > 0.05:
                return False, None
            
            # Step 2: Check adjustment coefficient
            X_diff = X.diff().dropna()
            Y_diff = Y.diff().dropna()
            resid_lag = residuals.shift(1).dropna()
            
            # Align all series
            common_idx = X_diff.index.intersection(resid_lag.index)
            X_ecm = sm.add_constant(pd.concat([X_diff.loc[common_idx], resid_lag.loc[common_idx]], axis=1))
            Y_ecm = Y_diff.loc[common_idx]
            
            ecm_model = sm.OLS(Y_ecm, X_ecm).fit()
            
            # Adjustment coefficient should be negative
            if ecm_model.params.iloc[-1] > 0:
                return False, None
            
            return True, model.params.values
            
        except Exception:
            # Fallback: simple correlation-based test
            correlation = X.corr(Y)
            if abs(correlation) > 0.7:
                # Simple linear regression params
                slope = np.cov(X, Y)[0, 1] / np.var(X)
                intercept = Y.mean() - slope * X.mean()
                return True, np.array([intercept, slope])
            return False, None
    
    def _generate_signals(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        bandwidth: int,
        z_entry: float,
        z_exit: float
    ) -> pd.DataFrame:
        """Generate trading signals for the pair."""
        signals = pd.DataFrame(index=df1.index)
        
        signals['asset1'] = df1['Close']
        signals['asset2'] = df2['Close']
        
        # Initialize columns
        signals['cointegrated'] = False
        signals['z_score'] = np.nan
        signals['z_upper'] = np.nan
        signals['z_lower'] = np.nan
        signals['fitted'] = np.nan
        signals['spread'] = np.nan
        signals['signals1'] = 0  # Signal for asset1
        signals['signals2'] = 0  # Signal for asset2
        signals['positions1'] = 0
        signals['positions2'] = 0
        
        prev_coint = False
        model_params = None
        resid_mean = 0
        resid_std = 1
        
        for i in range(bandwidth, len(signals)):
            # Get rolling window
            asset1_window = signals['asset1'].iloc[i-bandwidth:i]
            asset2_window = signals['asset2'].iloc[i-bandwidth:i]
            
            # Test cointegration
            is_coint, params = self._engle_granger_test(asset1_window, asset2_window)
            
            signals.iloc[i, signals.columns.get_loc('cointegrated')] = is_coint
            
            # Cointegration breaks - exit positions
            if prev_coint and not is_coint:
                signals.iloc[i, signals.columns.get_loc('signals1')] = 0
                signals.iloc[i, signals.columns.get_loc('signals2')] = 0
                signals.iloc[i, signals.columns.get_loc('positions1')] = 0
                signals.iloc[i, signals.columns.get_loc('positions2')] = 0
            
            # Cointegration exists
            if is_coint and params is not None:
                model_params = params
                
                # Calculate residual statistics from training window
                fitted_window = model_params[0] + model_params[1] * asset1_window
                resid_window = asset2_window - fitted_window
                resid_mean = resid_window.mean()
                resid_std = resid_window.std()
                
                # Calculate current spread and z-score
                current_fitted = model_params[0] + model_params[1] * signals['asset1'].iloc[i]
                current_spread = signals['asset2'].iloc[i] - current_fitted
                current_z = (current_spread - resid_mean) / resid_std if resid_std > 0 else 0
                
                signals.iloc[i, signals.columns.get_loc('fitted')] = current_fitted
                signals.iloc[i, signals.columns.get_loc('spread')] = current_spread
                signals.iloc[i, signals.columns.get_loc('z_score')] = current_z
                signals.iloc[i, signals.columns.get_loc('z_upper')] = z_entry
                signals.iloc[i, signals.columns.get_loc('z_lower')] = -z_entry
                
                # Get previous position
                prev_pos1 = signals['positions1'].iloc[i-1] if i > 0 else 0
                
                # Generate signals
                if current_z > z_entry and prev_pos1 != -1:
                    # Spread too high: Short asset2, Long asset1
                    signals.iloc[i, signals.columns.get_loc('signals1')] = 1
                    signals.iloc[i, signals.columns.get_loc('signals2')] = -1
                    signals.iloc[i, signals.columns.get_loc('positions1')] = 1
                    signals.iloc[i, signals.columns.get_loc('positions2')] = -1
                    
                elif current_z < -z_entry and prev_pos1 != 1:
                    # Spread too low: Long asset2, Short asset1
                    signals.iloc[i, signals.columns.get_loc('signals1')] = -1
                    signals.iloc[i, signals.columns.get_loc('signals2')] = 1
                    signals.iloc[i, signals.columns.get_loc('positions1')] = -1
                    signals.iloc[i, signals.columns.get_loc('positions2')] = 1
                    
                elif abs(current_z) < z_exit and prev_pos1 != 0:
                    # Mean reversion - exit
                    signals.iloc[i, signals.columns.get_loc('signals1')] = 0
                    signals.iloc[i, signals.columns.get_loc('signals2')] = 0
                    signals.iloc[i, signals.columns.get_loc('positions1')] = 0
                    signals.iloc[i, signals.columns.get_loc('positions2')] = 0
                    
                else:
                    # Hold current position
                    signals.iloc[i, signals.columns.get_loc('positions1')] = prev_pos1
                    signals.iloc[i, signals.columns.get_loc('positions2')] = -prev_pos1
            
            prev_coint = is_coint
        
        # Calculate combined signal for metrics
        signals['signals'] = signals['signals1']  # Use asset1 signal as primary
        signals['positions'] = signals['positions1']
        
        return signals
    
    def _calculate_portfolio(
        self,
        signals: pd.DataFrame,
        capital: float,
        risk: RiskParams
    ) -> pd.DataFrame:
        """Calculate portfolio value over time."""
        portfolio = pd.DataFrame(index=signals.index)
        
        # Split capital between two assets
        capital_per_asset = capital * risk.max_position_size / 2
        
        # Calculate returns from spread trading
        portfolio['asset1_returns'] = signals['asset1'].pct_change().fillna(0)
        portfolio['asset2_returns'] = signals['asset2'].pct_change().fillna(0)
        
        # Position returns (positions are held from previous period)
        portfolio['returns'] = (
            signals['positions1'].shift(1) * portfolio['asset1_returns'] +
            signals['positions2'].shift(1) * portfolio['asset2_returns']
        ).fillna(0) * 0.5  # Scale by position size
        
        # Cumulative portfolio value
        portfolio['total_value'] = capital * (1 + portfolio['returns']).cumprod()
        
        # Store positions for reference
        portfolio['positions'] = signals['positions1']
        portfolio['Close'] = signals['asset1']  # Primary asset for metrics
        
        return portfolio
    
    def _create_charts(
        self,
        signals: pd.DataFrame,
        ticker1: str,
        ticker2: str
    ) -> list[ChartData]:
        """Create visualization charts."""
        charts = []
        
        # Chart 1: Normalized prices
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Normalized prices
        norm1 = signals['asset1'] / signals['asset1'].iloc[0] * 100
        norm2 = signals['asset2'] / signals['asset2'].iloc[0] * 100
        
        ax1.plot(signals.index, norm1, label=ticker1, color='blue')
        ax1.plot(signals.index, norm2, label=ticker2, color='orange')
        ax1.set_title(f'Normalized Prices: {ticker1} vs {ticker2}')
        ax1.set_ylabel('Normalized Price (Base=100)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Spread / Z-score
        ax2.plot(signals.index, signals['z_score'], label='Z-Score', color='purple')
        ax2.axhline(y=signals['z_upper'].dropna().iloc[0] if not signals['z_upper'].dropna().empty else 1, 
                   color='red', linestyle='--', label='Entry (+)')
        ax2.axhline(y=signals['z_lower'].dropna().iloc[0] if not signals['z_lower'].dropna().empty else -1, 
                   color='green', linestyle='--', label='Entry (-)')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.fill_between(signals.index, -1, 1, alpha=0.1, color='gray')
        ax2.set_title('Spread Z-Score')
        ax2.set_ylabel('Z-Score')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Position
        ax3.fill_between(signals.index, 0, signals['positions1'], 
                        where=signals['positions1'] > 0, color='green', alpha=0.5, label='Long Asset1')
        ax3.fill_between(signals.index, 0, signals['positions1'], 
                        where=signals['positions1'] < 0, color='red', alpha=0.5, label='Short Asset1')
        ax3.set_title('Position')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Position')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker1}/{ticker2} Pairs Analysis",
            data=matplotlib_to_base64(fig1),
            chart_type="matplotlib",
            description="Normalized prices, Z-score, and position over time"
        ))
        
        # Chart 2: Cointegration analysis
        fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Scatter plot of prices
        ax4.scatter(signals['asset1'], signals['asset2'], alpha=0.3, c='blue', s=10)
        
        # Regression line (if we have valid fitted values)
        if not signals['fitted'].dropna().empty:
            valid_idx = signals['fitted'].dropna().index
            ax4.plot(signals.loc[valid_idx, 'asset1'], signals.loc[valid_idx, 'fitted'], 
                    color='red', linewidth=2, label='Regression Line')
        
        ax4.set_xlabel(f'{ticker1} Price')
        ax4.set_ylabel(f'{ticker2} Price')
        ax4.set_title('Price Relationship')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Cointegration status over time
        ax5.fill_between(signals.index, 0, signals['cointegrated'].astype(int),
                        alpha=0.5, color='green', label='Cointegrated')
        ax5.set_title('Cointegration Status')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Status (1=Cointegrated)')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        charts.append(ChartData(
            title=f"{ticker1}/{ticker2} Cointegration",
            data=matplotlib_to_base64(fig2),
            chart_type="matplotlib",
            description="Price relationship and cointegration status"
        ))
        
        return charts
    
    def _create_tables(
        self,
        signals: pd.DataFrame,
        ticker1: str,
        ticker2: str
    ) -> list[TableData]:
        """Create data tables."""
        tables = []
        
        # Summary statistics
        coint_pct = signals['cointegrated'].mean() * 100
        avg_z = signals['z_score'].dropna().mean()
        std_z = signals['z_score'].dropna().std()
        
        summary_data = [{
            'Metric': 'Cointegration Rate',
            'Value': f'{coint_pct:.1f}%'
        }, {
            'Metric': 'Average Z-Score',
            'Value': f'{avg_z:.3f}'
        }, {
            'Metric': 'Z-Score Std Dev',
            'Value': f'{std_z:.3f}'
        }, {
            'Metric': 'Total Periods',
            'Value': str(len(signals))
        }]
        
        tables.append(TableData(
            title="Pair Statistics",
            data=summary_data,
            columns=['Metric', 'Value'],
            description=f"Statistics for {ticker1}/{ticker2} pair"
        ))
        
        # Recent signals
        signal_changes = signals[signals['signals1'] != signals['signals1'].shift(1)].tail(20)
        
        if not signal_changes.empty:
            signals_table = signal_changes[['asset1', 'asset2', 'z_score', 'signals1']].copy()
            signals_table['Action'] = signals_table['signals1'].map({
                1: f'Long {ticker1}',
                -1: f'Long {ticker2}',
                0: 'Exit'
            })
            signals_table['z_score'] = signals_table['z_score'].round(3)
            
            tables.append(TableData(
                title="Recent Trading Signals",
                data=signals_table.to_dict(orient='records'),
                columns=['asset1', 'asset2', 'z_score', 'Action'],
                description="Recent position changes"
            ))
        
        return tables


# For backward compatibility - can still be used as a standalone script
if __name__ == "__main__":
    # Example usage
    strategy = PairsTradingStrategy()
    
    result = strategy.run(
        tickers=["KO", "PEP"],  # Coca-Cola and PepsiCo
        start_date="2022-01-01",
        end_date="2024-01-01",
        capital=10000,
        bandwidth=60,
        z_entry=1.0
    )
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {result.metrics}")
