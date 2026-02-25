"""
Pages Module for Centurion Capital LLC.

Contains page rendering logic separated by functionality.
Imports are lazy — page modules (and their heavy deps) are only
loaded when the page is actually navigated to.
"""

import importlib as _importlib

__all__ = [
    'render_main_page',
    'render_analysis_page',
    'render_fundamental_page',
    'render_backtesting_page',
    'render_crypto_page',
    'render_history_page',
]

_LAZY_MAP: dict[str, tuple[str, str]] = {
    'render_main_page':        ('ui.pages.main_page',        'render_main_page'),
    'render_analysis_page':    ('ui.pages.analysis_page',    'render_analysis_page'),
    'render_fundamental_page': ('ui.pages.fundamental_page', 'render_fundamental_page'),
    'render_backtesting_page': ('ui.pages.backtesting_page', 'render_backtesting_page'),
    'render_crypto_page':      ('ui.pages.crypto_page',      'render_crypto_page'),
    'render_history_page':     ('ui.pages.history_page',     'render_history_page'),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        mod_path, attr = _LAZY_MAP[name]
        module = _importlib.import_module(mod_path)
        obj = getattr(module, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
