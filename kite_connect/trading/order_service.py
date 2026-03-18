"""
Order placement service for Zerodha Kite Connect.

Provides functions to place, modify, and cancel orders, as well as
retrieve order book and position data.  Used by the Streamlit UI.
"""

import logging

from kiteconnect import exceptions as kite_exceptions

logger = logging.getLogger(__name__)


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
        result = {"success": True, "order_id": order_id}

        # For MARKET orders, try to get fill price from order history
        fill_price = price
        status_text = None
        filled_qty = None
        if order_type == "MARKET":
            try:
                import time
                time.sleep(0.5)
                history = kite.order_history(order_id)
                if history:
                    last = history[-1]
                    fill_price = last.get("average_price") or price
                    status_text = last.get("status")
                    filled_qty = last.get("filled_quantity")
            except Exception:
                pass

        _persist_to_db(symbol, exchange, transaction_type, int(quantity),
                       order_type, product, fill_price, order_id=order_id,
                       success=True, fill_price=fill_price, filled_qty=filled_qty,
                       status_text=status_text)
        _send_order_email(symbol, exchange, transaction_type, int(quantity),
                          fill_price or price or 0, str(order_id),
                          status_text or "PLACED")
        return result

    except kite_exceptions.InputException as e:
        result = {"success": False, "error": f"Invalid input: {e}"}
        _persist_to_db(symbol, exchange, transaction_type, int(quantity),
                       order_type, product, price, success=False, error_msg=str(e))
        _send_order_email(symbol, exchange, transaction_type, int(quantity),
                          price or 0, "-", "FAILED", error=str(e))
        return result
    except kite_exceptions.TokenException as e:
        return {"success": False, "error": f"Session expired: {e}"}
    except kite_exceptions.OrderException as e:
        result = {"success": False, "error": f"Order rejected: {e}"}
        _persist_to_db(symbol, exchange, transaction_type, int(quantity),
                       order_type, product, price, success=False, error_msg=str(e))
        _send_order_email(symbol, exchange, transaction_type, int(quantity),
                          price or 0, "-", "REJECTED", error=str(e))
        return result
    except Exception as e:
        result = {"success": False, "error": str(e)}
        _persist_to_db(symbol, exchange, transaction_type, int(quantity),
                       order_type, product, price, success=False, error_msg=str(e))
        _send_order_email(symbol, exchange, transaction_type, int(quantity),
                          price or 0, "-", "FAILED", error=str(e))
        return result


def _persist_to_db(symbol, exchange, side, quantity, order_type, product,
                   price, order_id=None, success=True, error_msg=None,
                   fill_price=None, filled_qty=None, status_text=None):
    """Best-effort persist of every order to the database."""
    try:
        import sys, os
        _root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from database.service import DatabaseService
        db = DatabaseService()
        db.save_single_order(
            symbol=symbol, exchange=exchange, side=side,
            quantity=quantity, order_type=order_type, product=product,
            price=price or 0, order_id=str(order_id) if order_id else None,
            success=success, error_msg=error_msg,
            fill_price=fill_price, filled_qty=filled_qty,
            status_text=status_text,
        )
    except Exception as exc:
        logger.debug("Order DB persist failed (non-fatal): %s", exc)


def _send_order_email(symbol, exchange, side, quantity, price, order_id,
                      status, error=None):
    """Best-effort email notification for every order."""
    try:
        import sys, os
        _root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from notifications.manager import NotificationManager
        NotificationManager().email_order_confirmation(
            symbol=symbol, side=side, quantity=quantity,
            entry_price=price, fill_price=price,
            order_id=order_id, status=status,
            exchange=exchange, error=error,
        )
    except Exception as exc:
        logger.debug("Order email failed (non-fatal): %s", exc)


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
