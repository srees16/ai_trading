"""
Strategies Module for Centurion Capital Trading Platform.

This module provides a standardized interface for backtesting strategies,
enabling seamless integration with the algo_path UI and supporting:
- Dynamic strategy discovery and loading
- Unified input/output formats
- Chart and table rendering for UI consumption
- Optional sentiment data integration

Architecture Overview:
---------------------
- BaseStrategy: Abstract base class all strategies must inherit
- StrategyRegistry: Central registry for strategy discovery
- StrategyLoader: Dynamic importer for strategy modules
- DataService: Centralized data fetching and caching
- Utils: Chart/table conversion utilities

Usage:
------
    from strategies import StrategyRegistry, load_all_strategies
    
    # Load all available strategies
    load_all_strategies()
    
    # Get a specific strategy
    strategy_cls = StrategyRegistry.get("macd_oscillator")
    strategy = strategy_cls()
    
    # Run backtest
    results = strategy.run(
        tickers=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        capital=10000
    )
"""

from .base_strategy import BaseStrategy, StrategyResult
from .registry import StrategyRegistry
from .loader import load_all_strategies, discover_strategies
from .data_service import DataService
from .utils import (
    matplotlib_to_base64,
    plotly_to_json,
    dataframe_to_table,
    create_metrics_summary
)

__all__ = [
    # Core classes
    "BaseStrategy",
    "StrategyResult",
    "StrategyRegistry",
    "DataService",
    
    # Loader functions
    "load_all_strategies",
    "discover_strategies",
    
    # Utility functions
    "matplotlib_to_base64",
    "plotly_to_json",
    "dataframe_to_table",
    "create_metrics_summary",
]

__version__ = "1.0.0"
