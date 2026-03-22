"""
Batch quote fetching via Kite Connect API (no Streamlit dependency).
"""

import logging

logger = logging.getLogger(__name__)

BATCH_SIZE = 200


def _parse_quote(q: dict) -> dict:
    ohlc = q.get("ohlc", {})
    last_price = q.get("last_price", 0)
    close = ohlc.get("close", 0)
    change_pct = round(((last_price - close) / close) * 100, 2) if close else 0
    return {
        "symbol": "",
        "ltp": last_price,
        "open": ohlc.get("open", 0),
        "high": ohlc.get("high") or last_price,
        "low": ohlc.get("low") or last_price,
        "close": close,
        "volume": q.get("volume", 0),
        "change_pct": change_pct,
    }


def get_batch_quotes(kite, symbols: list[str]) -> list[dict]:
    """
    Fetch live quotes for a list of NSE symbols via Kite Connect.

    Args:
        kite: An authenticated KiteConnect instance.
        symbols: List of stock symbols (e.g. ["RELIANCE", "TCS"]).

    Returns:
        List of quote dicts with ltp, change_pct, ohlc, volume.
    """
    if not symbols:
        return []

    instruments = [f"NSE:{sym}" for sym in symbols]
    results = []
    failed = []

    for i in range(0, len(instruments), BATCH_SIZE):
        batch = instruments[i : i + BATCH_SIZE]
        try:
            quotes = kite.quote(batch)
            for inst_key, q in quotes.items():
                sym = inst_key.replace("NSE:", "")
                parsed = _parse_quote(q)
                parsed["symbol"] = sym
                results.append(parsed)
        except Exception as exc:
            logger.warning("Batch quote failed (%d instruments): %s — retrying individually", len(batch), exc)
            for inst in batch:
                try:
                    q = kite.quote([inst])
                    for inst_key, data in q.items():
                        sym = inst_key.replace("NSE:", "")
                        parsed = _parse_quote(data)
                        parsed["symbol"] = sym
                        results.append(parsed)
                except Exception:
                    failed.append(inst.replace("NSE:", ""))

    if failed:
        logger.warning("%d symbols failed: %s", len(failed), ", ".join(failed))

    return results
