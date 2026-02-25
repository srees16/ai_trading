"""
Crypto Trading Strategies.

This module contains cryptocurrency-specific trading strategies
powered by the Binance public API, including Z-Score mean reversion.

Imports are lazy to avoid loading heavy dependencies (scipy,
statsmodels, seaborn, etc.) until actually needed.
"""

import importlib as _importlib

__all__ = [
    'CryptoMeanReversionStrategy',
]

_LAZY_MAP: dict[str, tuple[str, str]] = {
    'CryptoMeanReversionStrategy': (
        'trading_strategies.crypto.mean_reversion_strategy',
        'CryptoMeanReversionStrategy',
    ),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        mod_path, attr = _LAZY_MAP[name]
        module = _importlib.import_module(mod_path)
        obj = getattr(module, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
