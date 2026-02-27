"""
Order placement service for Zerodha Kite Connect.

Provides functions to place, modify, and cancel orders, as well as
retrieve order book and position data.  Used by the Streamlit UI.
"""

from kiteconnect import exceptions as kite_exceptions


# ── Order Placement ────────────────────────────────────────────

def place_order(kite, symbol, exchange, transaction_type, quantity,
                order_type="MARKET", product="CNC", price=None,
                trigger_price=None, validity="DAY"):
    """
    Place an order on Zerodha via Kite Connect.

    Parameters
    ----------
    kite : KiteConnect
        Authenticated Kite instance.
    symbol : str
        Trading symbol (e.g. ``"RELIANCE"``).
    exchange : str
        ``"NSE"`` or ``"BSE"``.
    transaction_type : str
        ``"BUY"`` or ``"SELL"``.
    quantity : int
        Number of shares.
    order_type : str
        ``"MARKET"``, ``"LIMIT"``, ``"SL"``, or ``"SL-M"``.
    product : str
        ``"CNC"`` (delivery), ``"MIS"`` (intraday), or ``"NRML"``.
    price : float | None
        Required for LIMIT / SL orders.
    trigger_price : float | None
        Required for SL / SL-M orders.
    validity : str
        ``"DAY"`` or ``"IOC"``.

    Returns
    -------
    dict
        ``{"success": True, "order_id": "..."}`` on success, or
        ``{"success": False, "error": "..."}`` on failure.
    """
    try:
        params = dict(
            tradingsymbol=symbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=int(quantity),
            order_type=order_type,
            product=product,
            validity=validity,
            variety="regular",
        )
        if order_type in ("LIMIT", "SL") and price is not None:
            params["price"] = float(price)
        if order_type in ("SL", "SL-M") and trigger_price is not None:
            params["trigger_price"] = float(trigger_price)

        order_id = kite.place_order(**params)
        return {"success": True, "order_id": order_id}

    except kite_exceptions.InputException as e:
        return {"success": False, "error": f"Invalid input: {e}"}
    except kite_exceptions.TokenException as e:
        return {"success": False, "error": f"Session expired: {e}"}
    except kite_exceptions.OrderException as e:
        return {"success": False, "error": f"Order rejected: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Order Book & Positions ─────────────────────────────────────

def get_order_book(kite):
    """Return the full order book for the current session."""
    try:
        return kite.orders() or []
    except Exception:
        return []


def get_positions(kite):
    """Return net positions dict with 'net' and 'day' keys."""
    try:
        return kite.positions()
    except Exception:
        return {"net": [], "day": []}


def get_holdings(kite):
    """Return current portfolio holdings."""
    try:
        return kite.holdings() or []
    except Exception:
        return []


def cancel_order(kite, order_id, variety="regular"):
    """Cancel a pending order."""
    try:
        kite.cancel_order(variety=variety, order_id=order_id)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
