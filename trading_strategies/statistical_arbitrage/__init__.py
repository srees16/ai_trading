"""
Statistical Arbitrage Trading Strategies.

This module contains strategies based on statistical relationships
between securities, such as pairs trading and mean reversion.
"""

from .pairs_trading import PairsTradingStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    'PairsTradingStrategy',
    'MeanReversionStrategy',
]
