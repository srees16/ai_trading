"""
Services Module for Centurion Capital LLC.

Contains business logic services for analysis, session management, etc.
Imports are lazy to avoid pulling in heavy ML / data deps at startup.
"""

import importlib as _importlib

__all__ = [
    'run_analysis_async',
    'initialize_session_state',
]

_LAZY_MAP: dict[str, tuple[str, str]] = {
    'run_analysis_async':       ('services.analysis', 'run_analysis_async'),
    'initialize_session_state': ('services.session',  'initialize_session_state'),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        mod_path, attr = _LAZY_MAP[name]
        module = _importlib.import_module(mod_path)
        obj = getattr(module, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
