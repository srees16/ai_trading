"""
Order placement service for Zerodha Kite Connect.

Provides functions to place, modify, and cancel orders, as well as
retrieve order book and position data.  Used by the Streamlit UI.

Features:
  - Idempotent retry with exponential backoff (max 3 attempts)
  - Circuit breaker: halts orders after consecutive failures
  - Slippage tracking: logs expected vs actual fill price
  - Uses Kite's ``tag`` field for order idempotency
"""

import hashlib
import logging
import os
import time

from kiteconnect import exceptions as kite_exceptions

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_BACKOFF_SECONDS = [1, 2, 4]  # 1s, 2s, 4s

# ── Circuit breaker (3 consecutive failures → halt for 10 min) ──
try:
    from infrastructure.fault_isolation import CircuitBreaker, CircuitOpenError
    _order_circuit = CircuitBreaker(
        "kite_orders",
        failure_threshold=3,
        reset_timeout=600.0,  # 10 minutes before half-open test (G16 fix)
    )
except ImportError:
    _order_circuit = None
    CircuitOpenError = RuntimeError


def _is_nse_market_open() -> bool:
    """Check if NSE is within trading hours (9:15 AM – 3:30 PM IST, weekdays)."""
    from datetime import datetime, timezone, timedelta
    _IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(_IST)
    if now.weekday() > 4:  # Saturday=5, Sunday=6
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ── Order Placement ────────────────────────────────────────────

def place_order(kite, symbol, exchange, transaction_type, quantity,
                order_type="MARKET", product="CNC", price=None,
                trigger_price=None, validity="DAY", tag=None):
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
    # ── G2: KILL SWITCH — instant halt of all order placement ──
    kill_switch = os.environ.get("CENTURION_KILL_SWITCH", "").lower() in ("true", "1", "yes")
    if not kill_switch:
        try:
            from config import Config
            kill_switch = getattr(Config, "KILL_SWITCH", False)
        except Exception:
            pass
    if kill_switch:
        logger.critical("KILL SWITCH ACTIVE — rejecting ALL orders for %s", symbol)
        return {"success": False, "error": "KILL SWITCH active: all order placement halted"}

    # ── Circuit breaker check ──
    if _order_circuit:
        state = _order_circuit.state
        if state == "OPEN":
            logger.error("Circuit breaker OPEN — rejecting order for %s", symbol)
            return {"success": False, "error": "Circuit breaker OPEN: Kite API consecutive failures detected. Halting orders for safety."}

    # ── Gap E fix: market hours guard ──
    # Block orders outside NSE hours (9:15 AM – 3:30 PM IST, weekdays).
    # SL and SL-M orders placed by TradeMonitor are exempt (trigger-based).
    if order_type not in ("SL", "SL-M"):
        if not _is_nse_market_open():
            logger.warning("Order blocked for %s — NSE market is closed", symbol)
            return {"success": False, "error": "NSE market closed (9:15 AM – 3:30 PM IST, Mon-Fri)"}

    # Generate idempotency tag from order parameters (caller tag takes precedence)
    if tag:
        idempotency_tag = tag[:20]  # Kite tag max 20 chars
    else:
        tag_seed = f"{symbol}:{exchange}:{transaction_type}:{quantity}:{order_type}:{price}:{int(time.time()//60)}"
        idempotency_tag = hashlib.sha256(tag_seed.encode()).hexdigest()[:20]

    expected_price = price  # Track for slippage measurement

    for attempt in range(_MAX_RETRIES):
        try:
            # Before retry, check if previous attempt silently succeeded
            if attempt > 0:
                try:
                    orders = kite.orders() or []
                    for o in orders:
                        if o.get("tag") == idempotency_tag and o.get("status") in ("OPEN", "COMPLETE", "TRIGGER PENDING"):
                            logger.info("Idempotent duplicate detected for %s (tag=%s) — skipping retry", symbol, idempotency_tag)
                            return {"success": True, "order_id": o.get("order_id")}
                except Exception:
                    pass

            params = dict(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=int(quantity),
                order_type=order_type,
                product=product,
                validity=validity,
                variety="regular",
                tag=idempotency_tag,
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
                    time.sleep(0.5)
                    history = kite.order_history(order_id)
                    if history:
                        last = history[-1]
                        fill_price = last.get("average_price") or price
                        status_text = last.get("status")
                        filled_qty = last.get("filled_quantity")
                except Exception:
                    pass

            # ── Slippage tracking ──
            slippage_bps = 0.0
            if expected_price and fill_price and expected_price > 0:
                slippage_bps = abs(fill_price - expected_price) / expected_price * 10000
                if slippage_bps > 5:  # > 5 bps slippage
                    logger.info(
                        "SLIPPAGE: %s %s expected=%.2f filled=%.2f slip=%.1f bps",
                        transaction_type, symbol, expected_price, fill_price, slippage_bps,
                    )
            result["slippage_bps"] = round(slippage_bps, 2)
            result["expected_price"] = expected_price
            result["fill_price"] = fill_price

            _persist_to_db(symbol, exchange, transaction_type, int(quantity),
                           order_type, product, fill_price, order_id=order_id,
                           success=True, fill_price=fill_price, filled_qty=filled_qty,
                           status_text=status_text, slippage_bps=slippage_bps)
            _send_order_email(symbol, exchange, transaction_type, int(quantity),
                              fill_price or price or 0, str(order_id),
                              status_text or "PLACED")
            if _order_circuit:
                _order_circuit._on_success()
            return result

        except kite_exceptions.InputException as e:
            # Non-retryable: bad input
            result = {"success": False, "error": f"Invalid input: {e}"}
            _persist_to_db(symbol, exchange, transaction_type, int(quantity),
                           order_type, product, price, success=False, error_msg=str(e))
            _send_order_email(symbol, exchange, transaction_type, int(quantity),
                              price or 0, "-", "FAILED", error=str(e))
            return result
        except kite_exceptions.TokenException as e:
            # Non-retryable: session expired
            return {"success": False, "error": f"Session expired: {e}"}
        except kite_exceptions.OrderException as e:
            # Non-retryable: exchange rejection
            result = {"success": False, "error": f"Order rejected: {e}"}
            _persist_to_db(symbol, exchange, transaction_type, int(quantity),
                           order_type, product, price, success=False, error_msg=str(e))
            _send_order_email(symbol, exchange, transaction_type, int(quantity),
                              price or 0, "-", "REJECTED", error=str(e))
            return result
        except Exception as e:
            # Retryable: network/transient errors
            if _order_circuit:
                _order_circuit._on_failure()
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_SECONDS[attempt]
                logger.warning(
                    "Order attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, symbol, e, wait,
                )
                time.sleep(wait)
                continue
            result = {"success": False, "error": str(e)}
            _persist_to_db(symbol, exchange, transaction_type, int(quantity),
                           order_type, product, price, success=False, error_msg=str(e))
            _send_order_email(symbol, exchange, transaction_type, int(quantity),
                              price or 0, "-", "FAILED", error=str(e))
            return result

    return {"success": False, "error": "Max retries exhausted"}


def _persist_to_db(symbol, exchange, side, quantity, order_type, product,
                   price, order_id=None, success=True, error_msg=None,
                   fill_price=None, filled_qty=None, status_text=None,
                   slippage_bps=0.0):
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
        from services.notifications.manager import NotificationManager
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


def modify_order(kite, order_id, trigger_price=None, price=None, variety="regular"):
    """Modify an existing order (SL price update, etc.)."""
    try:
        params = {"variety": variety, "order_id": order_id}
        if trigger_price is not None:
            params["trigger_price"] = float(trigger_price)
        if price is not None:
            params["price"] = float(price)
        kite.modify_order(**params)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_circuit_breaker_status() -> dict:
    """Return current state of the Kite API circuit breaker."""
    if not _order_circuit:
        return {"enabled": False}
    return {
        "enabled": True,
        "state": _order_circuit.state,
        "failures": _order_circuit._failures,
        "threshold": _order_circuit.failure_threshold,
        "reset_timeout_s": _order_circuit.reset_timeout,
    }


def reset_circuit_breaker():
    """Manually reset the circuit breaker to CLOSED state."""
    if _order_circuit:
        _order_circuit.reset()
        logger.info("Kite order circuit breaker manually reset")
