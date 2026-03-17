"""
Portfolio Analyzer — sector weights, allocation drift, and rebalance hints.

Fetches current holdings from Kite, computes sector weights, and provides
context to RiskManager for allocation-aware position sizing.

Usage::

    analyzer = PortfolioAnalyzer(kite)
    snapshot = analyzer.snapshot()
    # snapshot.sector_weights  → {"IT": 0.35, "Financials": 0.25, ...}
    # snapshot.total_deployed  → 385_000.0
    # snapshot.concentration_warnings → ["IT sector overweight: 35% > 30% target"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default target sector weights (equal-weight across sectors).
# Override with actual targets if running a tilt strategy.
_DEFAULT_TARGET_SECTOR_PCT = 0.30  # Max 30% target per sector


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    holdings: List[dict] = field(default_factory=list)
    total_deployed: float = 0.0
    sector_weights: Dict[str, float] = field(default_factory=dict)  # sector → %
    sector_values: Dict[str, float] = field(default_factory=dict)   # sector → ₹
    symbol_weights: Dict[str, float] = field(default_factory=dict)  # symbol → %
    stock_count: int = 0
    concentration_warnings: List[str] = field(default_factory=list)


# Sector mapping for NIFTY 50 + Next 50 constituents.
# This is a simplified mapping — in production, pull from NSE index CSVs.
_SECTOR_MAP: Dict[str, str] = {
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT", "COFORGE": "IT", "MPHASIS": "IT",
    "PERSISTENT": "IT",
    # Financials — Banks
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials",
    "KOTAKBANK": "Financials", "AXISBANK": "Financials", "INDUSINDBK": "Financials",
    "BANKBARODA": "Financials", "PNB": "Financials", "IDFCFIRSTB": "Financials",
    "FEDERALBNK": "Financials", "CANBK": "Financials",
    # Financials — NBFC / Insurance
    "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials",
    "HDFCLIFE": "Financials", "SBILIFE": "Financials", "ICICIGI": "Financials",
    "CHOLAFIN": "Financials", "SHRIRAMFIN": "Financials",
    # Energy / O&G
    "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "ADANIGREEN": "Energy", "ADANIENSOL": "Energy",
    "TATAPOWER": "Energy", "BPCL": "Energy", "IOC": "Energy",
    "ADANIENT": "Energy", "COAL": "Energy", "GAIL": "Energy",
    # Auto
    "MARUTI": "Auto", "M&M": "Auto", "TMPV": "Auto",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "BOSCHLTD": "Auto", "TVS MOTOR": "Auto",
    # Consumer
    "HINDUNILVR": "Consumer", "ITC": "Consumer", "NESTLEIND": "Consumer",
    "TITAN": "Consumer", "BRITANNIA": "Consumer", "DABUR": "Consumer",
    "GODREJCP": "Consumer", "COLPAL": "Consumer", "TRENT": "Consumer",
    "MARICO": "Consumer", "UNITDSPR": "Consumer",
    # Pharma / Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Pharma", "MAXHEALTH": "Pharma",
    "TORNTPHARM": "Pharma",
    # Metals / Materials
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "ULTRACEMCO": "Metals", "GRASIM": "Metals",
    "SHREECEM": "Metals", "AMBUJACEM": "Metals", "ADANIPORTS": "Metals",
    # Telecom
    "BHARTIARTL": "Telecom",
    # Infra / Capital Goods
    "LT": "Infra", "SIEMENS": "Infra", "ABB": "Infra",
    "HAL": "Infra", "BEL": "Infra",
    # Others
    "ZOMATO": "Consumer Tech", "PAYTM": "Consumer Tech",
    "DMART": "Retail", "NAUKRI": "Consumer Tech",
    "PIDILITIND": "Chemicals", "SRF": "Chemicals",
    "JSWENERGY": "Energy",
}


class PortfolioAnalyzer:
    """Analyses current Kite holdings for sector allocation and drift."""

    def __init__(self, kite=None, target_sector_pct: float = _DEFAULT_TARGET_SECTOR_PCT):
        self.kite = kite
        self.target_sector_pct = target_sector_pct

    def snapshot(self) -> PortfolioSnapshot:
        """Build a point-in-time portfolio snapshot from Kite holdings."""
        snap = PortfolioSnapshot()

        if self.kite is None:
            return snap

        try:
            from kite_connect.trading.order_service import get_holdings
            holdings = get_holdings(self.kite)
        except Exception as exc:
            logger.warning("Failed to fetch holdings: %s", exc)
            return snap

        if not holdings:
            return snap

        snap.holdings = holdings

        # Compute per-symbol and per-sector values
        for h in holdings:
            sym = h.get("tradingsymbol", "")
            qty = int(h.get("quantity", 0))
            ltp = float(h.get("last_price", 0))
            if qty <= 0 or ltp <= 0:
                continue

            value = qty * ltp
            snap.total_deployed += value
            snap.symbol_weights[sym] = value
            snap.stock_count += 1

            sector = _SECTOR_MAP.get(sym, "Other")
            snap.sector_values[sector] = snap.sector_values.get(sector, 0) + value

        # Convert to percentages
        if snap.total_deployed > 0:
            for sym in snap.symbol_weights:
                snap.symbol_weights[sym] /= snap.total_deployed
            for sec in snap.sector_values:
                snap.sector_weights[sec] = snap.sector_values[sec] / snap.total_deployed

        # Generate concentration warnings
        for sec, weight in snap.sector_weights.items():
            if weight > self.target_sector_pct:
                snap.concentration_warnings.append(
                    f"{sec} overweight: {weight:.0%} > {self.target_sector_pct:.0%} target"
                )

        logger.info(
            "Portfolio: %d stocks, ₹%.0f deployed, %d sectors, %d warnings",
            snap.stock_count, snap.total_deployed,
            len(snap.sector_weights), len(snap.concentration_warnings),
        )
        return snap

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Return the sector for a given symbol."""
        return _SECTOR_MAP.get(symbol, "Other")

    def should_reduce_sector(self, symbol: str, snapshot: Optional[PortfolioSnapshot] = None) -> bool:
        """Check if adding to this symbol's sector would exceed target allocation."""
        snap = snapshot or self.snapshot()
        sector = self.get_sector_for_symbol(symbol)
        current_weight = snap.sector_weights.get(sector, 0)
        return current_weight >= self.target_sector_pct
