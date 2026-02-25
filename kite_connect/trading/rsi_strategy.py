"""
RSI-based Auto-Order Strategy Service for Zerodha Kite Connect.

Scans a watchlist of stocks, calculates 14-period RSI on 5-minute candles,
and places BUY orders when RSI < 30 (oversold) with a bullish close reversal.
SELL signals fire when RSI > 70 (overbought) with a bearish close reversal.

Uses Cover Orders (CO / MIS) with an auto-calculated stop-loss derived from
a fixed capital-per-trade and max-loss-per-trade.

Designed to be called from the Streamlit UI or run standalone via CLI.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kiteconnect import KiteConnect, exceptions as kite_exceptions

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# RSI Calculation
# ═══════════════════════════════════════════════════════════════

def calculate_rsi(candles: list[dict], period: int = 14) -> float:
    """
    Compute RSI from a list of OHLC candle dicts (must contain ``"close"``).

    Uses the smoothed (Wilder) method: first *period* bars are simple average,
    subsequent bars use exponential smoothing  ``avg = (prev * (period-1) + current) / period``.

    Returns
    -------
    float
        RSI value rounded to 2 decimal places, or ``50.0`` if data is insufficient.
    """
    if len(candles) < period + 1:
        return 50.0  # neutral fallback

    gains = 0.0
    losses = 0.0

    # --- seed with first `period` changes ---
    for i in range(period):
        change = candles[i + 1]["close"] - candles[i]["close"]
        if change > 0:
            gains += (change / candles[i]["close"]) * 100
        else:
            losses += (abs(change) / candles[i + 1]["close"]) * 100

    avg_gain = gains / period
    avg_loss = losses / period

    # --- smooth through remaining bars ---
    rsi = _rsi_from_avgs(avg_gain, avg_loss)
    for i in range(period, len(candles) - 1):
        change = candles[i + 1]["close"] - candles[i]["close"]
        if change > 0:
            cur_gain = (change / candles[i]["close"]) * 100
            cur_loss = 0.0
        else:
            cur_gain = 0.0
            cur_loss = (abs(change) / candles[i + 1]["close"]) * 100

        avg_gain = (avg_gain * (period - 1) + cur_gain) / period
        avg_loss = (avg_loss * (period - 1) + cur_loss) / period
        rsi = _rsi_from_avgs(avg_gain, avg_loss)

    return round(rsi, 2)


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════
# Signal Detection
# ═══════════════════════════════════════════════════════════════

def detect_signal(candles: list[dict], rsi_low: float = 30, rsi_high: float = 70) -> dict:
    """
    Evaluate RSI + close-reversal conditions on the latest candles.

    Returns
    -------
    dict
        ``{"rsi": float, "signal": "BUY" | "SELL" | None, "close": float, "prev_close": float}``
    """
    rsi = calculate_rsi(candles)
    close_now = candles[-1]["close"]
    close_prev = candles[-2]["close"]

    signal = None
    if rsi < rsi_low and close_now > close_prev:
        signal = "BUY"
    elif rsi > rsi_high and close_now < close_prev:
        signal = "SELL"

    return {
        "rsi": rsi,
        "signal": signal,
        "close": close_now,
        "prev_close": close_prev,
    }


# ═══════════════════════════════════════════════════════════════
# Auto-Order Placement (Cover Order with SL)
# ═══════════════════════════════════════════════════════════════

def compute_sl_and_qty(kite: KiteConnect, symbol: str, side: str,
                       capital: float, max_loss: float) -> dict:
    """
    Derive quantity and stop-loss trigger from capital / max-loss constraints.

    Parameters
    ----------
    kite : KiteConnect
    symbol : str
    side : str  ``"BUY"`` or ``"SELL"``
    capital : float  total capital allocated for this trade
    max_loss : float  maximum acceptable loss in ₹ for this trade

    Returns
    -------
    dict
        ``{"qty": int, "trigger_price": float, "last_price": float}``
    """
    # Margin check to determine affordable qty
    margin_params = [{
        "exchange": "NSE",
        "tradingsymbol": symbol,
        "transaction_type": "BUY",
        "variety": "CO",
        "product": "MIS",
        "order_type": "MARKET",
        "quantity": 1,
    }]
    try:
        margin = kite.order_margins(margin_params)
        margin_per_unit = margin[0]["total"]
    except Exception:
        margin_per_unit = capital  # fallback: 1 unit

    qty = max(1, int(capital / margin_per_unit))
    sl_offset = max_loss / qty  # per-share SL offset

    # Get LTP for SL price
    quote = kite.quote([f"NSE:{symbol}"])
    ltp = quote[f"NSE:{symbol}"]["last_price"]

    if side == "BUY":
        trigger = ltp - sl_offset
    else:
        trigger = ltp + sl_offset

    # Round to tick size (0.05)
    trigger = round(trigger, 2)
    trigger = int(trigger * 100)
    trigger = (trigger - trigger % 5) / 100

    return {"qty": qty, "trigger_price": trigger, "last_price": ltp}


def place_strategy_order(kite: KiteConnect, symbol: str, side: str,
                         capital: float, max_loss: float,
                         order_type: str = "MARKET") -> dict:
    """
    Place a Cover Order (CO / MIS) with auto-calculated SL.

    Returns
    -------
    dict
        ``{"success": bool, "order_id": str | None, "error": str | None,
           "qty": int, "trigger_price": float, "last_price": float}``
    """
    try:
        calc = compute_sl_and_qty(kite, symbol, side, capital, max_loss)
        qty = calc["qty"]
        trigger = calc["trigger_price"]
        ltp = calc["last_price"]

        params = dict(
            variety="co",
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type=side,
            quantity=qty,
            product="MIS",
            order_type=order_type,
            validity="DAY",
            trigger_price=trigger,
        )
        if order_type == "LIMIT":
            params["price"] = ltp

        order_id = kite.place_order(**params)
        return {
            "success": True,
            "order_id": order_id,
            "qty": qty,
            "trigger_price": trigger,
            "last_price": ltp,
            "error": None,
        }
    except kite_exceptions.InputException as e:
        return {"success": False, "error": f"Invalid input: {e}", "order_id": None,
                "qty": 0, "trigger_price": 0, "last_price": 0}
    except kite_exceptions.OrderException as e:
        return {"success": False, "error": f"Order rejected: {e}", "order_id": None,
                "qty": 0, "trigger_price": 0, "last_price": 0}
    except Exception as e:
        return {"success": False, "error": str(e), "order_id": None,
                "qty": 0, "trigger_price": 0, "last_price": 0}


# ═══════════════════════════════════════════════════════════════
# Scan Watchlist (single pass — call repeatedly from UI / loop)
# ═══════════════════════════════════════════════════════════════

def scan_watchlist(kite: KiteConnect, symbols: list[str],
                   capital: float = 50000, max_loss: float = 500,
                   order_limit: int = 5, order_type: str = "MARKET",
                   rsi_low: float = 30, rsi_high: float = 70,
                   interval: str = "5minute", lookback_days: int = 30,
                   auto_place: bool = False) -> list[dict]:
    """
    Scan a list of symbols for RSI signals and optionally auto-place orders.

    Parameters
    ----------
    kite : KiteConnect
    symbols : list[str]          e.g. ["RELIANCE", "HDFCBANK", ...]
    capital : float              capital per trade (₹)
    max_loss : float             max loss per trade (₹)
    order_limit : int            max number of orders to place in this scan
    order_type : str             "MARKET" or "LIMIT"
    rsi_low : float              oversold threshold (default 30)
    rsi_high : float             overbought threshold (default 70)
    interval : str               candle interval (default "5minute")
    lookback_days : int          days of historical data to fetch
    auto_place : bool            if True, actually places orders; if False, only scans

    Returns
    -------
    list[dict]
        One entry per symbol with RSI, signal, and order result (if placed).
    """
    results = []
    orders_placed = 0
    to_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    from_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d") + " 09:15:00"

    # Pre-fetch instrument tokens
    quotes = {}
    try:
        instruments = [f"NSE:{s}" for s in symbols]
        for i in range(0, len(instruments), 200):
            batch = instruments[i:i + 200]
            quotes.update(kite.quote(batch))
    except Exception as e:
        log.warning("Quote pre-fetch failed: %s", e)

    for symbol in symbols:
        if orders_placed >= order_limit:
            break

        entry = {"symbol": symbol, "rsi": None, "signal": None, "order": None}

        # Get instrument token for historical data
        key = f"NSE:{symbol}"
        if key not in quotes:
            entry["error"] = "Quote not found"
            results.append(entry)
            continue

        token = quotes[key].get("instrument_token")
        if not token:
            entry["error"] = "No instrument token"
            results.append(entry)
            continue

        # Fetch historical candles
        try:
            candles = kite.historical_data(token, from_date, to_date, interval)
        except Exception as e:
            entry["error"] = f"Historical data error: {e}"
            results.append(entry)
            continue

        if len(candles) < 16:
            entry["error"] = "Insufficient candle data"
            results.append(entry)
            continue

        # Compute RSI & detect signal
        sig = detect_signal(candles, rsi_low=rsi_low, rsi_high=rsi_high)
        entry["rsi"] = sig["rsi"]
        entry["signal"] = sig["signal"]
        entry["close"] = sig["close"]
        entry["prev_close"] = sig["prev_close"]
        entry["ltp"] = quotes[key].get("last_price", 0)

        # Place order if signal is active
        if sig["signal"] and auto_place:
            result = place_strategy_order(
                kite, symbol, sig["signal"],
                capital=capital, max_loss=max_loss,
                order_type=order_type,
            )
            entry["order"] = result
            if result["success"]:
                orders_placed += 1

        results.append(entry)

    return results


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    from auth.kite_session import create_kite_session
    from core.db_service import get_connection

    logging.basicConfig(level=logging.INFO)

    kite = create_kite_session()
    conn = get_connection()

    # Fetch all stock names from DB
    cur = conn.cursor()
    cur.execute("SELECT name FROM stocks ORDER BY name;")
    symbols = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    print(f"Scanning {len(symbols)} stocks for RSI signals...")
    results = scan_watchlist(
        kite, symbols,
        capital=50000, max_loss=500, order_limit=3,
        auto_place=False,  # dry-run by default
    )

    for r in results:
        if r.get("rsi") is not None:
            flag = f"  *** {r['signal']} ***" if r["signal"] else ""
            print(f"  {r['symbol']:>15}  RSI: {r['rsi']:6.2f}  LTP: {r.get('ltp', 0):>10.2f}{flag}")
        elif r.get("error"):
            print(f"  {r['symbol']:>15}  ERROR: {r['error']}")

    print(f"\nDone — {sum(1 for r in results if r.get('signal'))} signals found.")
