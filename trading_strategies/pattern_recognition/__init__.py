"""
Pattern Recognition Trading Strategies.

This module contains strategies that use candlestick and price patterns
for trade signal generation.
"""

from .rsi_pattern import RSIPatternStrategy
from .shooting_star import ShootingStarStrategy
from .support_resistance import SupportResistanceStrategy
from .bollinger_pattern import BollingerPatternStrategy

__all__ = [
    'RSIPatternStrategy',
    'ShootingStarStrategy',
    'SupportResistanceStrategy',
    'BollingerPatternStrategy'
]
