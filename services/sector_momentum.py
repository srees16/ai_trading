"""
Sector Momentum helper for P9: sector-aware combo weighting.

Thin wrapper around sector_rotation that provides a simple interface
for IntegratedScorer to query a ticker's sector momentum.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SectorMomentumResult:
    """Lightweight result for IntegratedScorer P9."""
    sector_name: str
    return_20d: float  # approximate 1-month return


def get_sector_momentum(ticker: str, market: str = "IND") -> Optional[SectorMomentumResult]:
    """Return sector momentum for the given ticker.

    Attempts to resolve the ticker's sector and fetch the sectoral
    index 20-day return.  Returns ``None`` if data is unavailable.
    """
    if market != "IND":
        return None

    try:
        from services.sector_rotation import get_sector_rotation
        rotation = get_sector_rotation()
        if rotation is None:
            return None

        # Resolve ticker's sector from screener cache or NSE mapping
        sector_name = _resolve_sector(ticker)
        if not sector_name:
            return None

        sm = rotation.sectors.get(sector_name)
        if sm is None:
            # Fuzzy match: find sector containing the keyword
            for name, data in rotation.sectors.items():
                if sector_name.upper() in name.upper() or name.upper() in sector_name.upper():
                    sm = data
                    break

        if sm is None:
            return None

        return SectorMomentumResult(
            sector_name=sm.sector_name,
            return_20d=sm.return_1m,  # 1m return ≈ 20 trading days
        )
    except Exception as exc:
        logger.debug("Sector momentum lookup failed for %s: %s", ticker, exc)
        return None


def _resolve_sector(ticker: str) -> Optional[str]:
    """Try to map a ticker to its NIFTY sector name."""
    # Strip .NS / .BO suffix for lookup
    clean = ticker.upper().replace(".NS", "").replace(".BO", "")

    # Try database lookup first
    try:
        from database.service import get_database_service
        db = get_database_service()
        if db and db.is_available:
            stock = db.get_stock_info(clean)
            if stock and hasattr(stock, "sector"):
                return stock.sector
    except Exception:
        pass

    # Fallback: use NSE sector mapping if available
    try:
        from kite_connect.nse.sector_map import get_sector_for_symbol
        return get_sector_for_symbol(clean)
    except Exception:
        pass

    return None
