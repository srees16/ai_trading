"""
Strategy Utilities Module.

Provides utility functions for:
- Converting matplotlib figures to base64 strings
- Converting plotly figures to JSON
- Converting DataFrames to JSON-serializable tables
- Creating standardized metrics summaries
"""

import base64
import io
import json
from typing import Any, Optional
import pandas as pd
import numpy as np


def matplotlib_to_base64(
    fig,
    format: str = "png",
    dpi: int = 100,
    transparent: bool = False,
    close_fig: bool = True
) -> str:
    """
    Convert a matplotlib figure to a base64-encoded string.
    
    This function captures the figure as an image and encodes it
    to base64 for embedding in web UIs without saving to disk.
    
    Args:
        fig: matplotlib.figure.Figure object
        format: Image format ('png', 'jpg', 'svg')
        dpi: Dots per inch for image quality
        transparent: Whether to use transparent background
        close_fig: Whether to close the figure after conversion
    
    Returns:
        Base64-encoded string of the image
    
    Example:
        ```python
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        base64_str = matplotlib_to_base64(fig)
        ```
    """
    import matplotlib.pyplot as plt
    
    # Create buffer
    buffer = io.BytesIO()
    
    # Save figure to buffer
    fig.savefig(
        buffer,
        format=format,
        dpi=dpi,
        transparent=transparent,
        bbox_inches='tight',
        pad_inches=0.1
    )
    
    # Get buffer value and encode
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    
    # Close buffer
    buffer.close()
    
    # Optionally close figure to free memory
    if close_fig:
        plt.close(fig)
    
    # Return with data URI prefix for direct embedding
    mime_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'svg': 'image/svg+xml'
    }
    mime_type = mime_types.get(format.lower(), 'image/png')
    
    return f"data:{mime_type};base64,{base64_str}"


def matplotlib_figures_to_base64(
    figs: list,
    **kwargs
) -> list[str]:
    """
    Convert multiple matplotlib figures to base64 strings.
    
    Args:
        figs: List of matplotlib Figure objects
        **kwargs: Arguments passed to matplotlib_to_base64
    
    Returns:
        List of base64-encoded strings
    """
    return [matplotlib_to_base64(fig, **kwargs) for fig in figs]


def plotly_to_json(fig) -> dict:
    """
    Convert a plotly figure to JSON-serializable dictionary.
    
    Args:
        fig: plotly.graph_objects.Figure object
    
    Returns:
        JSON-serializable dictionary representation of the figure
    
    Example:
        ```python
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        json_data = plotly_to_json(fig)
        ```
    """
    try:
        # Use plotly's built-in JSON serialization
        return json.loads(fig.to_json())
    except Exception:
        # Fallback: convert to dict manually
        return {
            'data': [trace.to_plotly_json() for trace in fig.data],
            'layout': fig.layout.to_plotly_json()
        }


def dataframe_to_table(
    df: pd.DataFrame,
    title: str = "Data Table",
    description: str = "",
    max_rows: Optional[int] = None,
    round_decimals: int = 4,
    date_format: str = "%Y-%m-%d"
) -> dict:
    """
    Convert a pandas DataFrame to a JSON-serializable table structure.
    
    Args:
        df: pandas DataFrame to convert
        title: Title for the table
        description: Description of the table contents
        max_rows: Maximum rows to include (None for all)
        round_decimals: Number of decimal places for floats
        date_format: Format string for datetime columns
    
    Returns:
        Dictionary with 'title', 'data', 'columns', 'description'
    
    Example:
        ```python
        df = pd.DataFrame({'A': [1, 2], 'B': [3.1415, 2.718]})
        table = dataframe_to_table(df, title="My Data")
        ```
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Truncate if needed
    if max_rows is not None and len(df_copy) > max_rows:
        df_copy = df_copy.head(max_rows)
    
    # Process columns
    for col in df_copy.columns:
        # Handle datetime columns
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime(date_format)
        
        # Handle float columns
        elif pd.api.types.is_float_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].round(round_decimals)
        
        # Handle NaN/Inf values
        if df_copy[col].dtype in ['float64', 'float32']:
            df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)
            df_copy[col] = df_copy[col].fillna(0)
    
    # Reset index if it's meaningful
    if df_copy.index.name or not isinstance(df_copy.index, pd.RangeIndex):
        df_copy = df_copy.reset_index()
    
    return {
        "title": title,
        "data": df_copy.to_dict(orient="records"),
        "columns": list(df_copy.columns),
        "description": description
    }


def create_metrics_summary(
    metrics: dict[str, Any],
    title: str = "Performance Metrics"
) -> dict:
    """
    Create a formatted metrics summary table.
    
    Args:
        metrics: Dictionary of metric names to values
        title: Title for the metrics table
    
    Returns:
        Table-structured dictionary for metrics display
    """
    # Format metrics for display
    formatted = []
    
    metric_formatters = {
        'total_return': lambda v: f"{v:.2f}%",
        'mean_return': lambda v: f"{v:.4f}%",
        'std_return': lambda v: f"{v:.4f}%",
        'max_return': lambda v: f"{v:.4f}%",
        'min_return': lambda v: f"{v:.4f}%",
        'sharpe_ratio': lambda v: f"{v:.2f}",
        'sortino_ratio': lambda v: f"{v:.2f}",
        'max_drawdown': lambda v: f"{v:.2f}%",
        'win_rate': lambda v: f"{v:.2f}%",
        'total_trades': lambda v: f"{int(v)}",
        'final_value': lambda v: f"${v:,.2f}",
        'initial_capital': lambda v: f"${v:,.2f}",
    }
    
    # Human-readable names
    metric_names = {
        'total_return': 'Total Return',
        'mean_return': 'Mean Daily Return',
        'std_return': 'Return Std Dev',
        'max_return': 'Best Daily Return',
        'min_return': 'Worst Daily Return',
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'max_drawdown': 'Maximum Drawdown',
        'win_rate': 'Win Rate',
        'total_trades': 'Total Trades',
        'final_value': 'Final Portfolio Value',
        'initial_capital': 'Initial Capital',
    }
    
    for key, value in metrics.items():
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            continue
        
        display_name = metric_names.get(key, key.replace('_', ' ').title())
        formatter = metric_formatters.get(key, lambda v: str(v))
        
        try:
            formatted_value = formatter(value)
        except (TypeError, ValueError):
            formatted_value = str(value)
        
        formatted.append({
            "metric": display_name,
            "value": formatted_value
        })
    
    return {
        "title": title,
        "data": formatted,
        "columns": ["metric", "value"],
        "description": "Summary of key performance metrics"
    }


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index using Wilder's smoothed method (EWM).
    
    Shared implementation used by DataService and RSI strategy to avoid
    duplicate RSI calculation logic.
    
    Args:
        prices: Price series (typically Close prices)
        period: RSI lookback period (default: 14)
    
    Returns:
        RSI values as pandas Series (0-100 scale, NaN filled with 50)
    """
    delta = prices.diff()
    
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses
    rs = rs.replace([np.inf, -np.inf], 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)


def calculate_max_drawdown(values: pd.Series) -> float:
    """
    Calculate maximum drawdown percentage from a value series.
    
    Shared implementation used by BaseStrategy and MetricsCalculator
    to avoid duplicate drawdown logic.
    
    Args:
        values: Series of portfolio values or prices
    
    Returns:
        Maximum drawdown as a percentage (negative number, e.g. -15.2)
    """
    if len(values) < 2:
        return 0.0
    
    peak = values.cummax()
    drawdown = (values - peak) / peak
    return float(drawdown.min() * 100)


def format_currency(value: float, currency: str = "$") -> str:
    """Format a number as currency."""
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage."""
    return f"{value:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    return numerator / denominator


def clean_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame for JSON serialization.
    
    Handles NaN, Inf, and datetime conversion.
    """
    df = df.copy()
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Convert datetime columns
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = df[col].astype(str)
    
    # Fill NaN values
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        df[col] = df[col].fillna(0)
    
    return df


def calculate_trading_statistics(
    signals: pd.DataFrame,
    prices: pd.Series,
    signal_col: str = 'signals'
) -> dict:
    """
    Calculate trading statistics from signals.
    
    Args:
        signals: DataFrame with trading signals
        prices: Series of prices
        signal_col: Name of the signal column
    
    Returns:
        Dictionary of trading statistics
    """
    stats = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'profit_factor': 0.0
    }
    
    if signal_col not in signals.columns:
        return stats
    
    # Find entry and exit points
    trades = signals[signals[signal_col] != 0].copy()
    stats['total_trades'] = len(trades)
    
    if stats['total_trades'] > 0:
        # Calculate individual trade returns
        trade_returns = []
        position = 0
        entry_price = 0
        
        for idx, row in trades.iterrows():
            signal = row[signal_col]
            price = prices.loc[idx] if idx in prices.index else 0
            
            if signal > 0 and position == 0:  # Entry
                position = 1
                entry_price = price
            elif signal < 0 and position > 0:  # Exit
                if entry_price > 0:
                    trade_return = (price - entry_price) / entry_price
                    trade_returns.append(trade_return)
                position = 0
        
        if trade_returns:
            wins = [r for r in trade_returns if r > 0]
            losses = [r for r in trade_returns if r < 0]
            
            stats['winning_trades'] = len(wins)
            stats['losing_trades'] = len(losses)
            stats['win_rate'] = len(wins) / len(trade_returns) * 100
            stats['avg_win'] = np.mean(wins) * 100 if wins else 0
            stats['avg_loss'] = np.mean(losses) * 100 if losses else 0
            
            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            stats['profit_factor'] = safe_divide(total_wins, total_losses)
    
    return stats
