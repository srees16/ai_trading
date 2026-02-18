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
    ticker: str = ""


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
        from strategies.utils import calculate_max_drawdown
        return calculate_max_drawdown(values)
    
    def _calculate_portfolio(
        self,
        signals: pd.DataFrame,
        capital: float,
        risk: 'RiskParams'
    ) -> pd.DataFrame:
        """
        Default portfolio calculation for standard long/short strategies.
        
        Computes holdings, cash, total value, and returns based on positions
        and trading signals. Override in subclasses for custom portfolio logic
        (e.g., short-only, pairs trading).
        
        Args:
            signals: DataFrame with 'Close', 'positions', and 'signals' columns
            capital: Initial capital
            risk: Risk management parameters
        
        Returns:
            DataFrame with portfolio values over time
        """
        portfolio = pd.DataFrame(index=signals.index)
        
        max_position_value = capital * risk.max_position_size
        close_max = signals['Close'].dropna().max()
        shares = int(max_position_value / close_max) if close_max > 0 else 0
        
        portfolio['positions'] = signals['positions']
        portfolio['Close'] = signals['Close']
        portfolio['holdings'] = signals['positions'] * signals['Close'] * shares
        portfolio['cash'] = capital - (signals['signals'] * signals['Close'] * shares).cumsum()
        portfolio['total_value'] = portfolio['holdings'] + portfolio['cash']
        portfolio['returns'] = portfolio['total_value'].pct_change().fillna(0)
        
        return portfolio
    
    def _calculate_portfolio_long_only(
        self,
        signals: pd.DataFrame,
        capital: float,
        risk: 'RiskParams'
    ) -> pd.DataFrame:
        """
        Portfolio calculation for strategies with bidirectional signals
        that should only track long positions.
        
        Filters positions and signals to long-only (>= 0) before computing
        holdings and cash flow. Used by RSI, Support/Resistance, and
        Bollinger strategies.
        
        Args:
            signals: DataFrame with 'Close', 'positions', and 'signals' columns
            capital: Initial capital
            risk: Risk management parameters
        
        Returns:
            DataFrame with portfolio values over time
        """
        portfolio = pd.DataFrame(index=signals.index)
        
        long_positions = signals['positions'].clip(lower=0)
        long_signals = signals['signals'].clip(lower=0)
        
        max_position_value = capital * risk.max_position_size
        close_max = signals['Close'].dropna().max()
        shares = int(max_position_value / close_max) if close_max > 0 else 0
        
        portfolio['positions'] = long_positions
        portfolio['Close'] = signals['Close']
        portfolio['holdings'] = long_positions * signals['Close'] * shares
        portfolio['cash'] = capital - (long_signals * signals['Close'] * shares).cumsum()
        portfolio['total_value'] = portfolio['holdings'] + portfolio['cash']
        portfolio['returns'] = portfolio['total_value'].pct_change().fillna(0)
        
        return portfolio
    
    # --- Sentiment adjustment helpers ---
    
    def _sentiment_scale_indicator(
        self,
        signals: pd.DataFrame,
        column: str,
        sentiment: dict,
        threshold: float = 0.5,
        scale_factor: float = 0.2
    ) -> pd.DataFrame:
        """
        Scale an indicator column based on sentiment strength.
        
        When |sentiment_score| exceeds threshold, the indicator is multiplied
        by (1 + score * scale_factor), amplifying or dampening signals.
        
        Args:
            signals: Trading signals DataFrame
            column: Name of the indicator column to scale
            sentiment: Sentiment dict with 'score' key
            threshold: Minimum |score| to trigger scaling
            scale_factor: Multiplier coefficient for sentiment score
        
        Returns:
            Modified signals DataFrame
        """
        score = sentiment.get('score', 0)
        if abs(score) > threshold:
            multiplier = 1 + (score * scale_factor)
            signals[column] = signals[column] * multiplier
        return signals
    
    def _sentiment_zero_positions(
        self,
        signals: pd.DataFrame,
        sentiment: dict,
        threshold: float = -0.7
    ) -> pd.DataFrame:
        """
        Zero all positions when sentiment is below threshold.
        
        Used by trend-following strategies to avoid long positions
        when sentiment is strongly negative.
        
        Args:
            signals: Trading signals DataFrame
            sentiment: Sentiment dict with 'score' key
            threshold: Score below which positions are zeroed
        
        Returns:
            Modified signals DataFrame
        """
        score = sentiment.get('score', 0)
        if score < threshold:
            signals['positions'] = 0
            signals['signals'] = signals['positions'].diff().fillna(0)
        return signals
    
    def _sentiment_filter_positions(
        self,
        signals: pd.DataFrame,
        sentiment: dict,
        neg_threshold: float = -0.7,
        pos_threshold: float = 0.7,
        neg_target: int = 1,
        pos_target: int = -1
    ) -> pd.DataFrame:
        """
        Filter specific position directions based on sentiment.
        
        Removes positions matching neg_target when sentiment is below
        neg_threshold, and pos_target when above pos_threshold.
        
        Args:
            signals: Trading signals DataFrame
            sentiment: Sentiment dict with 'score' key
            neg_threshold: Score below which neg_target positions are removed
            pos_threshold: Score above which pos_target positions are removed
            neg_target: Position value to zero on negative sentiment (default: 1)
            pos_target: Position value to zero on positive sentiment (default: -1)
        
        Returns:
            Modified signals DataFrame
        """
        score = sentiment.get('score', 0)
        if score < neg_threshold:
            signals.loc[signals['positions'] == neg_target, 'positions'] = 0
            signals['signals'] = signals['positions'].diff().fillna(0)
        if score > pos_threshold:
            signals.loc[signals['positions'] == pos_target, 'positions'] = 0
            signals['signals'] = signals['positions'].diff().fillna(0)
        return signals
    
    def _sentiment_filter_signals(
        self,
        signals: pd.DataFrame,
        sentiment: dict,
        neg_threshold: Optional[float] = None,
        pos_threshold: Optional[float] = None,
        recalc_method: str = 'cumsum_clip'
    ) -> pd.DataFrame:
        """
        Filter buy/sell signals based on sentiment, then recalculate positions.
        
        Removes buy signals (1) when sentiment is below neg_threshold, and
        sell signals (-1) when above pos_threshold.
        
        Args:
            signals: Trading signals DataFrame
            sentiment: Sentiment dict with 'score' key
            neg_threshold: Score below which buy signals are removed
            pos_threshold: Score above which sell signals are removed
            recalc_method: How to recalculate positions:
                'cumsum_clip' - cumsum().clip(0, 1) for long-only bounded
                'cumsum' - cumsum() for unbounded
        
        Returns:
            Modified signals DataFrame
        """
        score = sentiment.get('score', 0)
        if neg_threshold is not None and score < neg_threshold:
            signals.loc[signals['signals'] == 1, 'signals'] = 0
        if pos_threshold is not None and score > pos_threshold:
            signals.loc[signals['signals'] == -1, 'signals'] = 0
        if recalc_method == 'cumsum_clip':
            signals['positions'] = signals['signals'].cumsum().clip(0, 1)
        elif recalc_method == 'cumsum':
            signals['positions'] = signals['signals'].cumsum()
        return signals
    
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
