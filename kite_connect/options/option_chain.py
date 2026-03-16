"""
Option Chain Service for Zerodha Kite Connect.

Fetches live option chain data (NIFTY / BANKNIFTY) including:
  - CE & PE LTP, OI, OI Change
  - Expiry date discovery
  - ATM strike detection
  - Black-Scholes Greeks (IV, Delta, Gamma, Theta, Vega)
"""

import sys
import os
import math
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from dateutil.relativedelta import relativedelta
from kiteconnect import KiteConnect, exceptions as kite_exceptions

try:
    from scipy.stats import norm
    from scipy.optimize import brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Append kite_connect to path (not insert) to avoid shadowing top-level packages
_kite_root = os.path.dirname(os.path.dirname(__file__))
if _kite_root not in sys.path:
    sys.path.append(_kite_root)

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

    # ── OI change helper (called from threads) ──────────────────
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

    # ── Collect instrument tokens for threaded OI-change fetch ──
    oi_tasks = {} # key (strike_index, "ce"|"pe")
    token_map = {} # same key instrument_token

    for i, strike in enumerate(strike_prices):
        ce_q = quotes.get(ce_instruments[i], {})
        pe_q = quotes.get(pe_instruments[i], {})
        ce_tok = ce_q.get("instrument_token", 0)
        pe_tok = pe_q.get("instrument_token", 0)
        if ce_tok:
            k = f"{i}_ce"
            oi_tasks[k] = (i, "ce")
            token_map[k] = ce_tok
        if pe_tok:
            k = f"{i}_pe"
            oi_tasks[k] = (i, "pe")
            token_map[k] = pe_tok

    # ── Fire all OI-change calls in parallel (thread pool) ──────
    MAX_WORKERS = min(20, len(oi_tasks) or 1)
    oi_results = {} # key int

    log.info("Fetching OI changes for %d instruments with %d threads",
             len(oi_tasks), MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_oi_change, token_map[k]): k
            for k in oi_tasks
        }
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                oi_results[k] = fut.result()
            except Exception:
                oi_results[k] = 0

    # ── Build strike rows ───────────────────────────────────────
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

        # Net price change from previous close
        ce_change = ce_q.get("net_change", 0) or 0
        pe_change = pe_q.get("net_change", 0) or 0

        # Volume
        ce_volume = ce_q.get("volume", 0) or 0
        pe_volume = pe_q.get("volume", 0) or 0

        # OI change (already fetched in parallel)
        ce_oi_chg = oi_results.get(f"{i}_ce", 0)
        pe_oi_chg = oi_results.get(f"{i}_pe", 0)

        rows.append({
            "strike": strike,
            "ce_ltp": ce_ltp,
            "ce_change": ce_change,
            "ce_oi": ce_oi,
            "ce_oi_chg": ce_oi_chg,
            "ce_volume": ce_volume,
            "pe_ltp": pe_ltp,
            "pe_change": pe_change,
            "pe_oi": pe_oi,
            "pe_oi_chg": pe_oi_chg,
            "pe_volume": pe_volume,
            "is_atm": strike == atm,
        })

    return {
        "spot": spot,
        "atm_strike": atm,
        "step": step,
        "strikes": rows,
    }


# ── Put-Call Ratio ───────────────────────────────────────────────────

def compute_pcr(chain: dict) -> dict:
    """
    Compute the Put-Call Ratio from an option chain returned by
    :func:`fetch_option_chain`.

    Returns a dict with::

        {
            "total_ce_oi":  int,
            "total_pe_oi":  int,
            "pcr_oi":       float,   # PE_OI / CE_OI
            "total_ce_vol": int,
            "total_pe_vol": int,
            "pcr_volume":   float,   # PE_vol / CE_vol
        }

    A PCR_OI > 1.0 is generally considered bullish (more puts being
    written → writers expect market to hold / rise).
    """
    total_ce_oi = 0
    total_pe_oi = 0
    total_ce_vol = 0
    total_pe_vol = 0

    for strike in chain.get("strikes", []):
        total_ce_oi += strike.get("ce_oi", 0) or 0
        total_pe_oi += strike.get("pe_oi", 0) or 0
        total_ce_vol += strike.get("ce_volume", 0) or 0
        total_pe_vol += strike.get("pe_volume", 0) or 0

    pcr_oi = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 0.0
    pcr_volume = (total_pe_vol / total_ce_vol) if total_ce_vol > 0 else 0.0

    return {
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "pcr_oi": round(pcr_oi, 4),
        "total_ce_vol": total_ce_vol,
        "total_pe_vol": total_pe_vol,
        "pcr_volume": round(pcr_volume, 4),
    }


# ═══════════════════════════════════════════════════════════════
# Black-Scholes Greeks Calculator
# ═══════════════════════════════════════════════════════════════

# Default risk-free rate: India 10Y G-Sec yield ≈ 7.1%
_DEFAULT_RISK_FREE = 0.071


def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 in the Black-Scholes formula."""
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _bs_d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if not _HAS_SCIPY:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(d1, sigma, T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if not _HAS_SCIPY:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(d1, sigma, T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = _DEFAULT_RISK_FREE,
    option_type: str = "CE",
) -> float:
    """
    Compute implied volatility via Brent's method.

    Returns IV as a decimal (e.g. 0.25 = 25%). Returns 0.0 if
    scipy is unavailable or the solver fails.
    """
    if not _HAS_SCIPY or market_price <= 0 or T <= 0:
        return 0.0
    price_fn = bs_call_price if option_type == "CE" else bs_put_price

    def objective(sigma):
        return price_fn(S, K, T, r, sigma) - market_price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-6, maxiter=200)
    except (ValueError, RuntimeError):
        return 0.0


def compute_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = _DEFAULT_RISK_FREE,
    option_type: str = "CE",
) -> dict:
    """
    Compute the five core Greeks for a European option.

    Parameters
    ----------
    S : float  – Spot price of the underlying
    K : float  – Strike price
    T : float  – Time to expiry in years (e.g. 7 days = 7/365)
    sigma : float  – Implied volatility (decimal)
    r : float  – Risk-free rate (default 7.1% for India G-Sec)
    option_type : str  – ``"CE"`` or ``"PE"``

    Returns
    -------
    dict with keys: delta, gamma, theta, vega, iv
    """
    result = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "iv": sigma}

    if not _HAS_SCIPY or T <= 0 or sigma <= 0:
        return result

    sqrt_T = math.sqrt(T)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(d1, sigma, T)

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)

    # Vega (per 1% move in vol → divide by 100)
    vega = S * norm.pdf(d1) * sqrt_T / 100.0

    if option_type == "CE":
        delta = norm.cdf(d1)
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365.0  # per calendar day
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        ) / 365.0

    result["delta"] = round(delta, 4)
    result["gamma"] = round(gamma, 6)
    result["theta"] = round(theta, 2)
    result["vega"] = round(vega, 4)
    return result


def enrich_chain_with_greeks(
    chain: dict,
    days_to_expiry: int,
    r: float = _DEFAULT_RISK_FREE,
) -> dict:
    """
    Add Greeks to every strike row in a chain dict returned by
    :func:`fetch_option_chain`.

    Modifies the chain **in-place** and returns it.

    Each strike row gains: ``ce_iv, ce_delta, ce_gamma, ce_theta, ce_vega``
    and the corresponding ``pe_*`` keys.
    """
    spot = chain.get("spot", 0)
    if spot <= 0 or days_to_expiry <= 0:
        return chain

    T = days_to_expiry / 365.0

    for row in chain.get("strikes", []):
        strike = row["strike"]

        # CE Greeks
        ce_ltp = row.get("ce_ltp", 0) or 0
        if ce_ltp > 0:
            iv = compute_iv(ce_ltp, spot, strike, T, r, "CE")
            greeks = compute_greeks(spot, strike, T, iv, r, "CE") if iv > 0 else {}
            row["ce_iv"] = round(iv * 100, 2)  # as percentage
            row["ce_delta"] = greeks.get("delta", 0.0)
            row["ce_gamma"] = greeks.get("gamma", 0.0)
            row["ce_theta"] = greeks.get("theta", 0.0)
            row["ce_vega"] = greeks.get("vega", 0.0)
        else:
            row.update({"ce_iv": 0, "ce_delta": 0, "ce_gamma": 0, "ce_theta": 0, "ce_vega": 0})

        # PE Greeks
        pe_ltp = row.get("pe_ltp", 0) or 0
        if pe_ltp > 0:
            iv = compute_iv(pe_ltp, spot, strike, T, r, "PE")
            greeks = compute_greeks(spot, strike, T, iv, r, "PE") if iv > 0 else {}
            row["pe_iv"] = round(iv * 100, 2)
            row["pe_delta"] = greeks.get("delta", 0.0)
            row["pe_gamma"] = greeks.get("gamma", 0.0)
            row["pe_theta"] = greeks.get("theta", 0.0)
            row["pe_vega"] = greeks.get("vega", 0.0)
        else:
            row.update({"pe_iv": 0, "pe_delta": 0, "pe_gamma": 0, "pe_theta": 0, "pe_vega": 0})

    return chain
