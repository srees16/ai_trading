"""
Survivorship Bias Validator.

Ensures delisted, suspended, or dead stocks are detected and excluded
from the screening and scoring pipelines for both IND and US markets.

Detection methods:
  1. **Data recency** — last trading date older than N days → delisted
  2. **Zero-volume streak** — prolonged zero-volume days → suspended/halted
  3. **yfinance metadata** — quoteType != EQUITY or regularMarketPrice absent
  4. **Kite instruments** — symbol absent from live instrument dump

Each method degrades gracefully if a data source is unavailable.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Cache for per-session validated tickers (avoid repeated API calls)
_VALID_CACHE: Dict[str, bool] = {}
_CACHE_TS: float = 0.0
_CACHE_TTL_S: float = 3600.0  # 1 hour


@dataclass
class DelistCheckResult:
    """Result of a single ticker delist / survivorship check."""
    ticker: str
    is_valid: bool
    reason: str = ""  # empty if valid; explains rejection otherwise


def _clear_cache():
    """Reset the validation cache (useful for tests)."""
    global _VALID_CACHE, _CACHE_TS
    _VALID_CACHE.clear()
    _CACHE_TS = 0.0


# ─── Individual checks ─────────────────────────────────────────

def _check_data_recency(
    ticker: str,
    ohlcv: Optional[pd.DataFrame] = None,
    max_stale_days: int = 10,
) -> Optional[str]:
    """Return a reason string if the ticker's last trade is too old.

    Parameters
    ----------
    ticker : str
        Symbol (with or without exchange suffix).
    ohlcv : pd.DataFrame | None
        Pre-downloaded OHLCV.  If ``None``, a fresh 1-month
        download is attempted via yfinance.
    max_stale_days : int
        Maximum days since the last row before we flag the ticker.

    Returns
    -------
    str | None
        Rejection reason, or ``None`` if the ticker is fresh.
    """
    try:
        if ohlcv is None or ohlcv.empty:
            from utils import download_ind_ohlcv
            ohlcv = download_ind_ohlcv(ticker, period="1mo")
        if ohlcv is None or ohlcv.empty:
            return f"No price data available (likely delisted)"

        last_date = pd.Timestamp(ohlcv.index[-1])
        # Make both timezone-naive for comparison
        if last_date.tzinfo is not None:
            last_date = last_date.tz_localize(None)
        stale_days = (datetime.now() - last_date).days
        if stale_days > max_stale_days:
            return (
                f"Last traded {stale_days} days ago "
                f"({last_date.strftime('%Y-%m-%d')}); likely delisted or suspended"
            )
    except Exception as exc:
        logger.debug("Data recency check failed for %s: %s", ticker, exc)
    return None


def _check_zero_volume_streak(
    ohlcv: Optional[pd.DataFrame],
    max_zero_days: int = 15,
) -> Optional[str]:
    """Flag stocks with a long trailing streak of zero volume."""
    if ohlcv is None or ohlcv.empty or "Volume" not in ohlcv.columns:
        return None
    try:
        tail = ohlcv["Volume"].tail(max_zero_days)
        if (tail == 0).all():
            return (
                f"Zero trading volume for last {max_zero_days}+ days; "
                "likely suspended or illiquid shell"
            )
    except Exception:
        pass
    return None


def _check_yf_metadata(ticker: str) -> Optional[str]:
    """Check yfinance metadata for delisted indicators.

    Flags:
    - quoteType not in {'EQUITY', 'ETF'} (for dead tickers Yahoo returns
      'NONE' or the key is missing entirely)
    - regularMarketPrice absent or zero (Yahoo strips pricing for delisted)
    - currency missing
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}

        qt = info.get("quoteType", "")
        if qt and qt.upper() not in ("EQUITY", "ETF"):
            return f"quoteType='{qt}' — not an active equity"

        rmp = info.get("regularMarketPrice")
        if rmp is None or rmp == 0:
            prev = info.get("previousClose")
            if prev is None or prev == 0:
                return "No market price available — likely delisted"

        currency = info.get("currency", "")
        if not currency:
            return "No currency metadata — ticker may be invalid or delisted"

    except Exception as exc:
        # yfinance can raise for truly dead tickers; that's a signal
        logger.debug("yfinance metadata check raised for %s: %s", ticker, exc)
        # Don't fail-open here — if yfinance can't find the ticker at
        # all, it's safest to flag it, UNLESS other checks already
        # confirmed it's okay (handled by the caller).
        return None  # be lenient; other checks will catch it
    return None


def _check_kite_instruments(
    symbol: str, kite=None,
) -> Optional[str]:
    """Verify the symbol exists in the live Kite instruments list."""
    if kite is None:
        return None  # can't check without a Kite session
    try:
        instruments = kite.instruments("NSE")
        eq_symbols = {
            row["tradingsymbol"]
            for row in instruments
            if row.get("segment") == "NSE" and row.get("instrument_type") == "EQ"
        }
        if symbol not in eq_symbols:
            return f"Symbol '{symbol}' not found in Kite NSE EQ instruments (delisted?)"
    except Exception as exc:
        logger.debug("Kite instruments check failed: %s", exc)
    return None


# ─── Unified check ──────────────────────────────────────────────

def check_ticker(
    ticker: str,
    market: str = "IND",
    ohlcv: Optional[pd.DataFrame] = None,
    kite=None,
    max_stale_days: int = 10,
) -> DelistCheckResult:
    """Run all survivorship checks on a single ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.  For IND screener paths, this is a plain
        NSE symbol (e.g. ``RELIANCE``).  For scorer/strategy paths
        it may carry a ``.NS`` suffix.
    market : str
        ``"IND"`` or ``"US"``.
    ohlcv : pd.DataFrame | None
        Pre-downloaded OHLCV data (avoids redundant download).
    kite : KiteConnect | None
        Authenticated Kite instance (optional).
    max_stale_days : int
        Staleness threshold in calendar days.

    Returns
    -------
    DelistCheckResult
    """
    global _VALID_CACHE, _CACHE_TS

    # Cache eviction
    if time.time() - _CACHE_TS > _CACHE_TTL_S:
        _VALID_CACHE.clear()
        _CACHE_TS = time.time()

    if ticker in _VALID_CACHE:
        is_valid = _VALID_CACHE[ticker]
        return DelistCheckResult(ticker=ticker, is_valid=is_valid,
                                 reason="" if is_valid else "cached rejection")

    # Resolve yfinance-compatible symbol for data checks
    yf_ticker = ticker
    plain_symbol = ticker.replace(".NS", "").replace(".BO", "")
    if market == "IND" and "." not in ticker:
        yf_ticker = f"{ticker}.NS"

    # 1. Data recency
    reason = _check_data_recency(yf_ticker, ohlcv, max_stale_days)
    if reason:
        _VALID_CACHE[ticker] = False
        return DelistCheckResult(ticker=ticker, is_valid=False, reason=reason)

    # 2. Zero-volume streak
    reason = _check_zero_volume_streak(ohlcv)
    if reason:
        _VALID_CACHE[ticker] = False
        return DelistCheckResult(ticker=ticker, is_valid=False, reason=reason)

    # 3. yfinance metadata (skip for plain symbols where we already have OHLCV)
    if ohlcv is None or ohlcv.empty:
        reason = _check_yf_metadata(yf_ticker)
        if reason:
            _VALID_CACHE[ticker] = False
            return DelistCheckResult(ticker=ticker, is_valid=False, reason=reason)

    # 4. Kite instruments check (IND only)
    if market == "IND" and kite is not None:
        reason = _check_kite_instruments(plain_symbol, kite)
        if reason:
            _VALID_CACHE[ticker] = False
            return DelistCheckResult(ticker=ticker, is_valid=False, reason=reason)

    _VALID_CACHE[ticker] = True
    return DelistCheckResult(ticker=ticker, is_valid=True)


# ─── Batch helpers ──────────────────────────────────────────────

def filter_valid_tickers(
    tickers: List[str],
    market: str = "IND",
    ohlcv_cache: Optional[Dict[str, pd.DataFrame]] = None,
    kite=None,
    max_stale_days: int = 10,
) -> tuple[List[str], List[DelistCheckResult]]:
    """Filter a list of tickers, returning (valid, rejected).

    Parameters
    ----------
    tickers : list[str]
        Tickers to validate.
    market : str
        ``"IND"`` or ``"US"``.
    ohlcv_cache : dict[str, pd.DataFrame] | None
        Pre-downloaded OHLCV keyed by symbol.
    kite : KiteConnect | None
        Authenticated Kite instance.

    Returns
    -------
    (valid_tickers, rejected_results)
    """
    valid: List[str] = []
    rejected: List[DelistCheckResult] = []

    for ticker in tickers:
        ohlcv = None
        if ohlcv_cache:
            # Try both plain and suffixed keys
            ohlcv = ohlcv_cache.get(ticker)
            if ohlcv is None:
                plain = ticker.replace(".NS", "").replace(".BO", "")
                ohlcv = ohlcv_cache.get(plain)

        result = check_ticker(
            ticker, market=market, ohlcv=ohlcv,
            kite=kite, max_stale_days=max_stale_days,
        )
        if result.is_valid:
            valid.append(ticker)
        else:
            rejected.append(result)
            logger.info(
                "Survivorship filter: rejected %s — %s",
                ticker, result.reason,
            )

    if rejected:
        logger.warning(
            "Survivorship filter: %d / %d tickers rejected (delisted/suspended)",
            len(rejected), len(tickers),
        )

    return valid, rejected
