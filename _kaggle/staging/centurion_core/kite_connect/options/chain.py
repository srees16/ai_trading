"""
Thin re-export shim so ``from kite_connect.options.chain import …`` works.

The actual implementation lives in ``option_chain.py``.  The v1_gateway
router imports ``get_expiry_dates`` and ``get_option_chain`` from here.
"""

from kite_connect.options.option_chain import discover_expiries, fetch_option_chain


def get_expiry_dates(symbol: str) -> list[str]:
    """Return available expiry dates for *symbol* (e.g. ``NIFTY``, ``BANKNIFTY``)."""
    from api.dependencies import get_kite_session
    kite = get_kite_session()
    if kite is None:
        return []
    return discover_expiries(kite, index=symbol.upper())


def get_option_chain(symbol: str, expiry: str) -> dict:
    """Return the full option chain for *symbol* at *expiry* (``YYYY-MM-DD``)."""
    from api.dependencies import get_kite_session
    kite = get_kite_session()
    if kite is None:
        return {}
    return fetch_option_chain(kite, index=symbol.upper(), expiry=expiry)
