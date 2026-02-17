"""
Momentum Trading Strategies Module.

Contains momentum-based trading strategies:
- MACD Oscillator
- Awesome Oscillator  
- Heikin-Ashi
- Parabolic SAR
"""

from .macd_oscillator import MACDOscillatorStrategy
from .awesome_oscillator import AwesomeOscillatorStrategy
from .heikin_ashi import HeikinAshiStrategy
from .parabolic_sar import ParabolicSARStrategy

__all__ = [
    "MACDOscillatorStrategy",
    "AwesomeOscillatorStrategy",
    "HeikinAshiStrategy",
    "ParabolicSARStrategy",
]
