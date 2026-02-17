"""
Base Strategy Module.

Provides the abstract base class that all trading strategies must inherit.
Ensures consistent interface across all strategies for UI integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import pandas as pd
from datetime import datetime


class StrategyCategory(Enum):
    """Categories of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    PATTERN_RECOGNITION = "pattern_recognition"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    BREAKOUT = "breakout"
    OPTIONS = "options"
    OTHER = "other"


@dataclass
class ChartData:
    """
    Container for chart data that can be rendered in the UI.
    
    Attributes:
        title: Chart title for display
        data: Base64 string (matplotlib) or JSON dict (plotly)
        chart_type: Type of chart library used ('matplotlib' or 'plotly')
        description: Optional description of what the chart shows
    """
    title: str
    data: str | dict
    chart_type: str = "matplotlib"  # 'matplotlib' or 'plotly'
    description: str = ""


@dataclass
class TableData:
    """
    Container for table data that can be rendered in the UI.
    
    Attributes:
        title: Table title for display
        data: List of dictionaries (JSON serializable)
        columns: Optional column order/names
        description: Optional description of what the table shows
    """
    title: str
    data: list[dict]
    columns: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class StrategyResult:
    """
    Standardized result container returned by all strategies.
    
    This ensures consistent output format for UI rendering and analysis.
    
    Attributes:
        charts: List of chart data for visualization
        tables: List of table data for display
        metrics: Dictionary of performance metrics
        signals: DataFrame of trading signals (optional)
        portfolio: DataFrame of portfolio values over time (optional)
        success: Whether the strategy executed successfully
        error_message: Error message if execution failed
        execution_time: Time taken to execute in seconds
        metadata: Additional strategy-specific information
    """
    charts: list[ChartData] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    signals: Optional[pd.DataFrame] = None
    portfolio: Optional[pd.DataFrame] = None
    success: bool = True
    error_message: str = ""
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """
        Convert result to JSON-serializable dictionary.
        
        Returns:
            Dictionary suitable for JSON serialization and UI consumption.
        """
        return {
            "charts": [
                {
                    "title": c.title,
                    "data": c.data,
                    "chart_type": c.chart_type,
                    "description": c.description
                }
                for c in self.charts
            ],
            "tables": [
                {
                    "title": t.title,
                    "data": t.data,
                    "columns": t.columns,
                    "description": t.description
                }
                for t in self.tables
            ],
            "metrics": self.metrics,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


@dataclass
class RiskParams:
    """
    Risk management parameters for strategy execution.
    
    Attributes:
        stop_loss_pct: Stop loss percentage (e.g., 0.05 for 5%)
        take_profit_pct: Take profit percentage
        max_position_size: Maximum position size as fraction of capital
        max_drawdown_pct: Maximum allowed drawdown before stopping
        trailing_stop: Whether to use trailing stop loss
        position_sizing: Position sizing method ('fixed', 'kelly', 'volatility')
    """
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_position_size: float = 0.25
    max_drawdown_pct: float = 0.20
    trailing_stop: bool = False
    position_sizing: str = "fixed"


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All backtesting strategies must inherit from this class and implement
    the required abstract methods. This ensures consistent interface
    across the platform.
    
    Class Attributes:
        name: Human-readable strategy name
        description: Brief description of the strategy
        category: Strategy category for organization
        version: Strategy version string
        author: Strategy author/maintainer
        requires_sentiment: Whether strategy can use sentiment data
        min_data_points: Minimum data points required for analysis
    
    Example:
        class MyStrategy(BaseStrategy):
            name = "My Custom Strategy"
            category = StrategyCategory.MOMENTUM
            
            def run(self, tickers, start_date, end_date, capital, **kwargs):
                # Implementation here
                return StrategyResult(...)
    """
    
    # Class attributes - override in subclasses
    name: str = "Base Strategy"
    description: str = "Abstract base strategy"
    category: StrategyCategory = StrategyCategory.OTHER
    version: str = "1.0.0"
    author: str = "Centurion Capital"
    requires_sentiment: bool = False
    min_data_points: int = 50
    
    def __init__(self):
        """Initialize the strategy."""
        self._data_cache: dict[str, pd.DataFrame] = {}
        self._last_run_time: Optional[datetime] = None
    
    @abstractmethod
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
        Execute the backtesting strategy.
        
        This is the main entry point for strategy execution. All strategies
        must implement this method.
        
        Args:
            tickers: List of stock ticker symbols to analyze
            start_date: Start date for backtesting (YYYY-MM-DD format)
            end_date: End date for backtesting (YYYY-MM-DD format)
            capital: Initial capital for backtesting
            sentiment_data: Optional sentiment scores from algo_path
                           Format: {ticker: {"score": float, "label": str}}
            risk_params: Optional risk management parameters
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            StrategyResult containing charts, tables, and metrics
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If strategy execution fails
        """
        pass
    
    def validate_inputs(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        capital: float
    ) -> None:
        """
        Validate input parameters before strategy execution.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date string
            end_date: End date string
            capital: Initial capital
        
        Raises:
            ValueError: If any input is invalid
        """
        if not tickers:
            raise ValueError("At least one ticker is required")
        
        if not all(isinstance(t, str) and t.strip() for t in tickers):
            raise ValueError("All tickers must be non-empty strings")
        
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
        
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
        
        if capital <= 0:
            raise ValueError("Capital must be positive")
    
    def get_risk_params(
        self,
        risk_params: Optional[RiskParams | dict] = None
    ) -> RiskParams:
        """
        Parse and return RiskParams object.
        
        Args:
            risk_params: RiskParams object or dict with risk parameters
        
        Returns:
            RiskParams object with validated parameters
        """
        if risk_params is None:
            return RiskParams()
        
        if isinstance(risk_params, RiskParams):
            return risk_params
        
        if isinstance(risk_params, dict):
            return RiskParams(**{
                k: v for k, v in risk_params.items()
                if k in RiskParams.__dataclass_fields__
            })
        
        return RiskParams()
    
    def calculate_metrics(
        self,
        portfolio: pd.DataFrame,
        signals: pd.DataFrame,
        capital: float,
        risk_free_rate: float = 0.02
    ) -> dict[str, Any]:
        """
        Calculate standard performance metrics.
        
        Args:
            portfolio: DataFrame with portfolio values over time
            signals: DataFrame with trading signals
            capital: Initial capital
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        try:
            if 'total_value' in portfolio.columns:
                values = portfolio['total_value']
            elif 'asset' in portfolio.columns:
                values = portfolio['asset']
            else:
                # Try to find any value-like column
                value_cols = [c for c in portfolio.columns if 'value' in c.lower() or 'asset' in c.lower()]
                values = portfolio[value_cols[0]] if value_cols else portfolio.iloc[:, 0]
            
            returns = values.pct_change().dropna()
            
            # Basic metrics
            metrics['total_return'] = float((values.iloc[-1] / capital - 1) * 100)
            metrics['final_value'] = float(values.iloc[-1])
            metrics['initial_capital'] = float(capital)
            
            # Return statistics
            metrics['mean_return'] = float(returns.mean() * 100)
            metrics['std_return'] = float(returns.std() * 100)
            metrics['max_return'] = float(returns.max() * 100)
            metrics['min_return'] = float(returns.min() * 100)
            
            # Risk metrics
            metrics['sharpe_ratio'] = self._calculate_sharpe(returns, risk_free_rate)
            metrics['sortino_ratio'] = self._calculate_sortino(returns, risk_free_rate)
            metrics['max_drawdown'] = self._calculate_max_drawdown(values)
            
            # Trading metrics
            if 'signals' in signals.columns:
                trades = signals[signals['signals'] != 0]
                metrics['total_trades'] = len(trades)
            else:
                metrics['total_trades'] = 0
            
        except Exception as e:
            metrics['calculation_error'] = str(e)
        
        return metrics
    
    def _calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: float
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return float(excess_returns.mean() / returns.std() * (252 ** 0.5))
    
    def _calculate_sortino(
        self,
        returns: pd.Series,
        risk_free_rate: float
    ) -> float:
        """Calculate Sortino ratio (downside risk adjusted)."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return float(excess_returns.mean() / downside_returns.std() * (252 ** 0.5))
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        if len(values) < 2:
            return 0.0
        
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return float(drawdown.min() * 100)
    
    @classmethod
    def get_info(cls) -> dict:
        """
        Get strategy information for display.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            "name": cls.name,
            "description": cls.description,
            "category": cls.category.value,
            "version": cls.version,
            "author": cls.author,
            "requires_sentiment": cls.requires_sentiment,
            "min_data_points": cls.min_data_points
        }
    
    @classmethod
    def get_parameters(cls) -> dict[str, dict]:
        """
        Get strategy-specific parameters and their descriptions.
        
        Override in subclasses to expose configurable parameters.
        
        Returns:
            Dictionary mapping parameter names to their metadata
        """
        return {}
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
