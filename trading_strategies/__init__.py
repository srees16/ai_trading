"""
Trading Strategies Module.

This module contains all trading strategy implementations organized by category:
- Momentum Trading: MACD, Awesome Oscillator, Heikin-Ashi, Parabolic SAR
- Pattern Recognition: RSI Pattern, Shooting Star, Support/Resistance, Bollinger Pattern
- Statistical Arbitrage: Pairs Trading

All strategies inherit from BaseStrategy and follow a standardized interface.
"""

# Import from submodules
from .momentum_trading import (
    MACDOscillatorStrategy,
    AwesomeOscillatorStrategy,
    HeikinAshiStrategy,
    ParabolicSARStrategy
)

from .pattern_recognition import (
    RSIPatternStrategy,
    ShootingStarStrategy,
    SupportResistanceStrategy,
    BollingerPatternStrategy
)

from .statistical_arbitrage import (
    PairsTradingStrategy
)

__all__ = [
    # Momentum strategies
    'MACDOscillatorStrategy',
    'AwesomeOscillatorStrategy',
    'HeikinAshiStrategy',
    'ParabolicSARStrategy',
    # Pattern recognition strategies
    'RSIPatternStrategy',
    'ShootingStarStrategy',
    'SupportResistanceStrategy',
    'BollingerPatternStrategy',
    # Statistical arbitrage strategies
    'PairsTradingStrategy'
]

# Strategy registry map for easy lookup
STRATEGY_MAP = {
    'macd': MACDOscillatorStrategy,
    'awesome_oscillator': AwesomeOscillatorStrategy,
    'heikin_ashi': HeikinAshiStrategy,
    'parabolic_sar': ParabolicSARStrategy,
    'rsi_pattern': RSIPatternStrategy,
    'shooting_star': ShootingStarStrategy,
    'support_resistance': SupportResistanceStrategy,
    'bollinger_pattern': BollingerPatternStrategy,
    'pairs_trading': PairsTradingStrategy
}


def get_strategy(name: str):
    """
    Get a strategy class by name.
    
    Args:
        name: Strategy identifier (e.g., 'macd', 'pairs_trading')
    
    Returns:
        Strategy class or None if not found
    """
    return STRATEGY_MAP.get(name.lower())


def list_strategies() -> list[dict]:
    """
    List all available strategies with their metadata.
    
    Returns:
        List of strategy info dictionaries
    """
    strategies = []
    for key, cls in STRATEGY_MAP.items():
        strategies.append({
            'id': key,
            'name': cls.name,
            'description': cls.description,
            'category': cls.category.value,
            'requires_sentiment': cls.requires_sentiment
        })
    return strategies
