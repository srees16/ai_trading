"""
Trading Strategies Module.

This module contains all trading strategy implementations organized by category:
- Momentum Trading: MACD, Awesome Oscillator, Heikin-Ashi, Parabolic SAR
- Pattern Recognition: RSI Pattern, Shooting Star, Support/Resistance, Bollinger Pattern
- Statistical Arbitrage: Pairs Trading, Mean Reversion (Stocks)
- Crypto: Crypto Mean Reversion (Z-Score)

All strategies inherit from BaseStrategy and follow a standardized interface.

Imports are **lazy** — strategy classes and their heavy transitive
dependencies (statsmodels, scipy, torch, …) are only loaded when
first accessed, keeping the login page fast.
"""

import importlib

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
    'PairsTradingStrategy',
    'MeanReversionStrategy',
    'CryptoMeanReversionStrategy',
]

# ── Lazy import mapping ──────────────────────────────────────────────
# Maps a public name → (submodule, class_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    'MACDOscillatorStrategy':    ('.momentum_trading',      'MACDOscillatorStrategy'),
    'AwesomeOscillatorStrategy': ('.momentum_trading',      'AwesomeOscillatorStrategy'),
    'HeikinAshiStrategy':        ('.momentum_trading',      'HeikinAshiStrategy'),
    'ParabolicSARStrategy':      ('.momentum_trading',      'ParabolicSARStrategy'),
    'RSIPatternStrategy':        ('.pattern_recognition',   'RSIPatternStrategy'),
    'ShootingStarStrategy':      ('.pattern_recognition',   'ShootingStarStrategy'),
    'SupportResistanceStrategy': ('.pattern_recognition',   'SupportResistanceStrategy'),
    'BollingerPatternStrategy':  ('.pattern_recognition',   'BollingerPatternStrategy'),
    'PairsTradingStrategy':      ('.statistical_arbitrage', 'PairsTradingStrategy'),
    'MeanReversionStrategy':     ('.statistical_arbitrage', 'MeanReversionStrategy'),
    'CryptoMeanReversionStrategy': ('.crypto',                'CryptoMeanReversionStrategy'),
}

# Strategy key → (submodule, class_name) for STRATEGY_MAP
_STRATEGY_KEYS: dict[str, str] = {
    'macd':                  'MACDOscillatorStrategy',
    'awesome_oscillator':    'AwesomeOscillatorStrategy',
    'heikin_ashi':           'HeikinAshiStrategy',
    'parabolic_sar':         'ParabolicSARStrategy',
    'rsi_pattern':           'RSIPatternStrategy',
    'shooting_star':         'ShootingStarStrategy',
    'support_resistance':    'SupportResistanceStrategy',
    'bollinger_pattern':     'BollingerPatternStrategy',
    'pairs_trading':         'PairsTradingStrategy',
    'mean_reversion':        'MeanReversionStrategy',
    'crypto_mean_reversion': 'CryptoMeanReversionStrategy',
}

# Strategy metadata (avoids importing the classes just for listing)
_STRATEGY_META: list[dict] = [
    {'id': 'macd',                  'name': 'MACD Oscillator',        'description': 'MACD crossover strategy with signal line',                             'category': 'momentum',            'requires_sentiment': False},
    {'id': 'awesome_oscillator',    'name': 'Awesome Oscillator',     'description': 'Bill Williams Awesome Oscillator strategy',                             'category': 'momentum',            'requires_sentiment': False},
    {'id': 'heikin_ashi',           'name': 'Heikin-Ashi',            'description': 'Trend-following strategy using Heikin-Ashi candles',                    'category': 'momentum',            'requires_sentiment': False},
    {'id': 'parabolic_sar',         'name': 'Parabolic SAR',          'description': 'Parabolic Stop and Reverse trend strategy',                            'category': 'momentum',            'requires_sentiment': False},
    {'id': 'rsi_pattern',           'name': 'RSI Pattern',            'description': 'RSI-based pattern recognition strategy',                               'category': 'pattern_recognition', 'requires_sentiment': False},
    {'id': 'shooting_star',         'name': 'Shooting Star',          'description': 'Candlestick shooting star reversal pattern',                           'category': 'pattern_recognition', 'requires_sentiment': False},
    {'id': 'support_resistance',    'name': 'Support/Resistance',     'description': 'Price action strategy using support/resistance levels with candlestick patterns', 'category': 'pattern_recognition', 'requires_sentiment': False},
    {'id': 'bollinger_pattern',     'name': 'Bollinger Pattern',      'description': 'Bollinger Band squeeze and breakout strategy',                         'category': 'pattern_recognition', 'requires_sentiment': False},
    {'id': 'pairs_trading',         'name': 'Pairs Trading',                   'description': 'Statistical arbitrage pairs trading strategy',                         'category': 'statistical_arbitrage', 'requires_sentiment': False, 'min_tickers': 2},
    {'id': 'mean_reversion',        'name': 'Mean Reversion (Z-Score)',        'description': 'Mean reversion strategy for equities',                                 'category': 'statistical_arbitrage', 'requires_sentiment': False, 'min_tickers': 2},
    {'id': 'crypto_mean_reversion', 'name': 'Crypto Mean Reversion (Z-Score)', 'description': 'Z-Score mean reversion strategy for crypto via Binance',               'category': 'crypto',                'requires_sentiment': False, 'min_tickers': 2},
]


def __getattr__(name: str):
    """Lazy-load strategy classes on first access."""
    if name in _LAZY_IMPORTS:
        submod_path, cls_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(submod_path, package=__name__)
        cls = getattr(module, cls_name)
        # Cache in module globals so subsequent accesses are instant
        globals()[name] = cls
        return cls
    if name == 'STRATEGY_MAP':
        # Build on first access
        smap = {k: get_strategy(k) for k in _STRATEGY_KEYS}
        globals()['STRATEGY_MAP'] = smap
        return smap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_strategy(name: str):
    """
    Get a strategy class by name.
    
    Args:
        name: Strategy identifier (e.g., 'macd', 'pairs_trading')
    
    Returns:
        Strategy class or None if not found
    """
    cls_name = _STRATEGY_KEYS.get(name.lower())
    if cls_name is None:
        return None
    # Trigger the lazy import via __getattr__
    return getattr(importlib.import_module(__name__), cls_name)


def list_strategies() -> list[dict]:
    """
    List all available strategies with their metadata.
    
    Returns metadata without importing any strategy classes, keeping
    the call lightweight enough for the login page health check.
    """
    return list(_STRATEGY_META)
