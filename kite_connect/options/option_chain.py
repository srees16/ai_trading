"""
Option Chain Service for Zerodha Kite Connect.

Fetches live option chain data (NIFTY / BANKNIFTY) including:
  - CE & PE LTP, OI, OI Change
  - Expiry date discovery
  - ATM strike detection
"""

import sys
import os
import logging
from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta
from kiteconnect import KiteConnect, exceptions as kite_exceptions

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

log = logging.getLogger(__name__)

# ── Index metadata ─────────────────────────────────────────────
INDEX_META = {
    "NIFTY": {
        "quote_key": "NSE:NIFTY 50",
        "prefix": "NIFTY",
        "step": 50,
    },
    "BANKNIFTY": {
        "quote_key": "NSE:NIFTY BANK",
        "prefix": "BANKNIFTY",
        "step": 100,
    },
}


# ═══════════════════════════════════════════════════════════════
# Expiry Discovery
# ═══════════════════════════════════════════════════════════════

def discover_expiries(kite: KiteConnect, index: str = "BANKNIFTY") -> list[str]:
    """
    Probe the next 30 calendar days + monthly expiry for valid NFO expiries.

    Returns a list of expiry strings usable for building NFO symbols,
    e.g. ``["2602D", "260213", "260220", "260227", "26FEB"]``.

    Weekly format: YY + M (single-digit month, no leading zero) + DD
    Monthly format: YY + MMM (3-letter uppercase month)
    """
    meta = INDEX_META.get(index, INDEX_META["BANKNIFTY"])
    prefix = meta["prefix"]
    step = meta["step"]

    # Get ATM price to build a test strike
    try:
        q = kite.quote([meta["quote_key"]])
        spot = q[meta["quote_key"]]["last_price"]
    except Exception as e:
        log.warning("Could not fetch spot price for %s: %s", index, e)
        return []

    atm = int(spot) - int(spot) % step
    expiries = []

    # Probe daily for next 45 days (weekly expiries)
    for i in range(45):
        dt = datetime.today() + timedelta(days=i)
        year = dt.strftime("%y")
        month = str(dt.month)  # no leading zero
        day = dt.strftime("%d")
        code = f"{year}{month}{day}"
        strike_sym = f"{prefix}{code}{atm}CE"
        try:
            kite.quote([f"NFO:{strike_sym}"])
            expiries.append(code)
        except Exception:
            pass

    # Probe monthly (current + next month)
    for months_ahead in (0, 1, 2):
        dt = datetime.today() + relativedelta(months=months_ahead)
        year = dt.strftime("%y")
        mon = dt.strftime("%b").upper()
        code = f"{year}{mon}"
        strike_sym = f"{prefix}{code}{atm}CE"
        try:
            kite.quote([f"NFO:{strike_sym}"])
            if code not in expiries:
                expiries.append(code)
        except Exception:
            pass

    return expiries


# ═══════════════════════════════════════════════════════════════
# Option Chain Data
# ═══════════════════════════════════════════════════════════════

def fetch_option_chain(
    kite: KiteConnect,
    index: str = "BANKNIFTY",
    expiry_code: str = "",
    num_strikes: int = 20,
    timeframe: str = "5minute",
) -> dict:
    """
    Fetch a full option chain centred on ATM for the given index/expiry.

    Returns
    -------
    dict with keys:
        spot        : float   – underlying last price
        atm_strike  : int     – ATM strike
        step        : int     – strike increment
        strikes     : list[dict]  – per-strike rows, each with:
            strike, ce_ltp, ce_oi, ce_oi_chg, pe_ltp, pe_oi, pe_oi_chg
    """
    meta = INDEX_META.get(index, INDEX_META["BANKNIFTY"])
    prefix = meta["prefix"]
    step = meta["step"]

    # Spot price
    try:
        q = kite.quote([meta["quote_key"]])
        spot = q[meta["quote_key"]]["last_price"]
    except Exception as e:
        log.error("Could not fetch spot for %s: %s", index, e)
        return {"spot": 0, "atm_strike": 0, "step": step, "strikes": []}

    atm = int(spot) - int(spot) % step
    start_strike = atm - (num_strikes // 2) * step

    # Build instrument list
    ce_instruments = []
    pe_instruments = []
    strike_prices = []
    for i in range(num_strikes):
        s = start_strike + i * step
        strike_prices.append(s)
        ce_instruments.append(f"NFO:{prefix}{expiry_code}{s}CE")
        pe_instruments.append(f"NFO:{prefix}{expiry_code}{s}PE")

    all_instruments = ce_instruments + pe_instruments

    # Fetch quotes in one batch (max ~500)
    quotes = {}
    try:
        quotes = kite.quote(all_instruments)
    except Exception as e:
        log.warning("Batch quote failed, trying individually: %s", e)
        for inst in all_instruments:
            try:
                q = kite.quote([inst])
                quotes.update(q)
            except Exception:
                pass

    # Fetch OI change using historical data (last 2 candles)
    def _oi_change(instrument_token: int) -> int:
        """Return OI change (current - previous candle)."""
        try:
            to_dt = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
            from_dt = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d") + " 09:15:00"
            data = kite.historical_data(
                instrument_token, from_dt, to_dt, timeframe, False, True,
            )
            if len(data) >= 2:
                return data[-1]["oi"] - data[-2]["oi"]
            elif data:
                return data[-1].get("oi", 0)
        except Exception:
            pass
        return 0

    # Build strike rows
    rows = []
    for i, strike in enumerate(strike_prices):
        ce_key = ce_instruments[i]
        pe_key = pe_instruments[i]

        ce_q = quotes.get(ce_key, {})
        pe_q = quotes.get(pe_key, {})

        ce_ltp = ce_q.get("last_price", 0)
        pe_ltp = pe_q.get("last_price", 0)
        ce_oi = ce_q.get("oi", 0)
        pe_oi = pe_q.get("oi", 0)

        ce_token = ce_q.get("instrument_token", 0)
        pe_token = pe_q.get("instrument_token", 0)

        ce_oi_chg = _oi_change(ce_token) if ce_token else 0
        pe_oi_chg = _oi_change(pe_token) if pe_token else 0

        rows.append({
            "strike": strike,
            "ce_ltp": ce_ltp,
            "ce_oi": ce_oi,
            "ce_oi_chg": ce_oi_chg,
            "pe_ltp": pe_ltp,
            "pe_oi": pe_oi,
            "pe_oi_chg": pe_oi_chg,
            "is_atm": strike == atm,
        })

    return {
        "spot": spot,
        "atm_strike": atm,
        "step": step,
        "strikes": rows,
    }
