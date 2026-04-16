"""
F&O Open Interest Signal — Gap A6.

Uses NSE F&O open interest data to gauge directional conviction.
OI changes combined with price movements reveal institutional positioning.

Signal logic:
  - Rising OI + Rising price → Long buildup (bullish)
  - Rising OI + Falling price → Short buildup (bearish)
  - Falling OI + Rising price → Short covering (mildly bullish)
  - Falling OI + Falling price → Long unwinding (bearish)

Also provides IV rank for options overlay strategy selection.

Integration:
  - Generates per-stock conviction modifier for F&O stocks only
  - Weight: 5% of combined forecast
  - Only active for ~180 F&O-eligible NSE stocks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

FORECAST_CAP = 20.0

# NSE F&O lot sizes for major stocks (Gap D3: expanded from ~50 to ~180 symbols)
# Source: NSE circular on market lot sizes (updated quarterly)
FNO_LOT_SIZES: Dict[str, int] = {
    # NIFTY 50
    "RELIANCE": 250, "TCS": 150, "INFY": 300, "HDFCBANK": 550,
    "ICICIBANK": 700, "SBIN": 750, "KOTAKBANK": 400, "AXISBANK": 600,
    "BAJFINANCE": 125, "BHARTIARTL": 950, "ITC": 1600, "LT": 150,
    "HCLTECH": 350, "WIPRO": 1500, "MARUTI": 100, "TATAMOTORS": 575,
    "TATASTEEL": 550, "SUNPHARMA": 350, "NTPC": 2800, "POWERGRID": 2700,
    "ONGC": 1925, "HINDALCO": 1075, "INDUSINDBK": 400, "M&M": 350,
    "JSWSTEEL": 675, "ADANIENT": 250, "ADANIPORTS": 625, "TITAN": 175,
    "DIVISLAB": 100, "ULTRACEMCO": 50, "DRREDDY": 125, "TECHM": 350,
    "NESTLEIND": 25, "COALINDIA": 700, "APOLLOHOSP": 125, "HDFCLIFE": 500,
    "CIPLA": 325, "GRASIM": 250, "SBILIFE": 375, "EICHERMOT": 75,
    "BAJAJ-AUTO": 125, "BPCL": 1800, "TATACONSUM": 450, "HEROMOTOCO": 75,
    "ASIANPAINT": 200, "BRITANNIA": 100, "VEDL": 1550, "HINDPETRO": 1350,
    # NIFTY NEXT 50
    "BANKBARODA": 1800, "PNB": 4000, "CANBK": 3400, "RECLTD": 2000,
    "PFC": 3000, "BHEL": 3800, "IOC": 3250, "GAIL": 3400,
    "NMDC": 3400, "SAIL": 5700, "NATIONALUM": 4400, "MUTHOOTFIN": 400,
    "BAJAJFINSV": 125, "CHOLAFIN": 500, "SHRIRAMFIN": 250, "GODREJCP": 400,
    "DABUR": 1250, "MARICO": 1200, "COLPAL": 200, "PIDILITIND": 250,
    "BERGEPAINT": 550, "HAVELLS": 350, "VOLTAS": 350, "CROMPTON": 1400,
    "TRENT": 150, "PAGEIND": 15, "ABCAPITAL": 5400, "MFSL": 500,
    "SBICARD": 700, "NAUKRI": 125, "PERSISTENT": 125, "LTIM": 150,
    "MPHASIS": 250, "COFORGE": 100, "ZOMATO": 4000, "PAYTM": 1500,
    # BROADER F&O
    "DLF": 825, "IRCTC": 625, "HAL": 200, "BEL": 3200,
    "TATAPOWER": 2700, "TATAELXSI": 100, "PIIND": 175, "AARTIIND": 750,
    "DEEPAKNTR": 250, "ASTRAL": 275, "POLYCAB": 100, "ABFRL": 1900,
    "OBEROIRLTY": 350, "GODREJPROP": 325, "PRESTIGE": 500, "PHOENIXLTD": 350,
    "MINDTREE": 200, "LALPATHLAB": 250, "METROPOLIS": 400, "AUROPHARMA": 500,
    "BIOCON": 2300, "TORNTPHARM": 250, "LUPIN": 425, "IPCALAB": 500,
    "GRANULES": 1600, "LAURUSLABS": 1000, "ALKEM": 125, "NATCOPHARM": 500,
    "IDEA": 28000, "TATACOMM": 250, "BANDHANBNK": 1800, "IDFCFIRSTB": 7500,
    "FEDERALBNK": 5000, "RBLBANK": 2500, "MANAPPURAM": 3000, "L&TFH": 4350,
    "LICHSGFIN": 1000, "IBULHSGFIN": 4200, "ICICIGI": 350, "ICICIPRULI": 1500,
    "SRTRANSFIN": 350, "M&MFIN": 2000, "PEL": 550, "CANFINHOME": 750,
    "ACC": 250, "AMBUJACEM": 900, "RAMCOCEM": 550, "SHREECEM": 25,
    "DALBHARAT": 250, "INDIACEM": 2700, "JKCEMENT": 200, "STARCEMENT": 5000,
    "MRF": 5, "BALKRISIND": 200, "CEATLTD": 400, "APOLLOTYRE": 2000,
    "ESCORTS": 200, "ASHOKLEY": 4000, "TVSMOTOR": 300, "MOTHERSON": 5000,
    "BOSCHLTD": 25, "EXIDEIND": 1800, "AMARAJABAT": 700, "AMARARAJA": 700,
    "JINDALSTEL": 625, "NATIONALUM": 4400, "APLAPOLLO": 350,
    "CHAMBLFERT": 1500, "COROMANDEL": 500, "UPL": 1300, "RAIN": 3700,
    "INDUSTOWER": 2300, "IRFC": 5000, "NHPC": 10000, "SJVN": 5000,
    "CONCOR": 900, "MGL": 400, "IGL": 1375, "PETRONET": 3000,
    "JUBLFOOD": 1000, "TATACONSUM": 450, "UBL": 350, "MCDOWELL-N": 500,
    "INDHOTEL": 1000, "LEMONTR": 5000, "DIXON": 100, "HONAUT": 15,
    "SIEMENS": 150, "ABB": 125, "CUMMINSIND": 300,
}


@dataclass
class OISignal:
    """Open Interest based signal for one stock."""
    ticker: str
    oi_change_pct: float       # % change in OI
    price_change_pct: float    # % change in price
    buildup_type: str          # LONG_BUILDUP, SHORT_BUILDUP, SHORT_COVERING, LONG_UNWINDING
    forecast: float            # Carver-scale forecast
    iv_rank: float = 50.0      # IV rank (0-100), 50 = median
    is_fno: bool = True


def classify_oi_buildup(
    oi_change_pct: float,
    price_change_pct: float,
) -> tuple:
    """Classify OI buildup type and assign conviction.

    Returns (buildup_type, conviction_multiplier)
    """
    if oi_change_pct > 2.0 and price_change_pct > 0.5:
        return "LONG_BUILDUP", 1.0
    elif oi_change_pct > 2.0 and price_change_pct < -0.5:
        return "SHORT_BUILDUP", -1.0
    elif oi_change_pct < -2.0 and price_change_pct > 0.5:
        return "SHORT_COVERING", 0.5
    elif oi_change_pct < -2.0 and price_change_pct < -0.5:
        return "LONG_UNWINDING", -0.5
    else:
        return "NEUTRAL", 0.0


def compute_oi_forecast(
    oi_change_pct: float,
    price_change_pct: float,
    volume_ratio: float = 1.0,
) -> float:
    """Compute OI-based forecast.

    Parameters
    ----------
    oi_change_pct : float
        % change in open interest (e.g., 5.0 = 5% increase).
    price_change_pct : float
        % change in stock price.
    volume_ratio : float
        Today's volume / 20-day avg volume (amplifier).

    Returns
    -------
    float
        Carver-scale forecast (-20 to +20 if shorts enabled, else 0 to +20).
    """
    buildup_type, conviction = classify_oi_buildup(oi_change_pct, price_change_pct)

    # Check if short selling is enabled
    try:
        from config import Config
        allow_short = getattr(Config, "SHORT_SELLING_ENABLED", False)
    except Exception:
        allow_short = False

    if conviction <= 0 and not allow_short:
        return 0.0  # Long-only: no signal for bearish setups

    if conviction == 0:
        return 0.0  # NEUTRAL: no signal regardless

    # Scale by OI magnitude
    oi_strength = min(1.0, abs(oi_change_pct) / 10.0)  # 10% OI change = max strength

    # Volume confirmation
    vol_boost = min(1.5, max(0.5, volume_ratio))

    raw = conviction * oi_strength * vol_boost * FORECAST_CAP
    floor = -FORECAST_CAP if allow_short else 0.0
    return round(max(floor, min(FORECAST_CAP, raw)), 2)


def compute_iv_rank(
    current_iv: float,
    iv_history: list[float],
) -> float:
    """Compute IV rank (percentile of current IV vs 1-year history).

    IV Rank = (current_IV - min_IV) / (max_IV - min_IV) × 100

    Parameters
    ----------
    current_iv : float
        Current implied volatility.
    iv_history : list[float]
        Last 252 days of IV values.

    Returns
    -------
    float
        IV rank (0-100).
    """
    if not iv_history or len(iv_history) < 30:
        return 50.0

    min_iv = min(iv_history)
    max_iv = max(iv_history)

    if max_iv <= min_iv:
        return 50.0

    rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
    return round(max(0.0, min(100.0, rank)), 1)


def compute_oi_signals_batch(
    oi_data: Dict[str, Dict],
) -> Dict[str, float]:
    """Compute OI-based forecasts for all F&O stocks.

    Parameters
    ----------
    oi_data : dict
        {symbol: {"oi_change_pct": float, "price_change_pct": float,
                   "volume_ratio": float}}

    Returns
    -------
    dict[str, float]
        {symbol: forecast} for Carver combiner.
    """
    forecasts: Dict[str, float] = {}

    for sym, data in oi_data.items():
        # Strip .NS/.BO suffix for FNO lookup
        bare_sym = sym.replace('.NS', '').replace('.BO', '')
        if bare_sym not in FNO_LOT_SIZES:
            continue  # Only F&O stocks

        oi_pct = data.get("oi_change_pct", 0.0)
        price_pct = data.get("price_change_pct", 0.0)
        vol_ratio = data.get("volume_ratio", 1.0)

        forecast = compute_oi_forecast(oi_pct, price_pct, vol_ratio)
        if forecast != 0:  # G3 FIX: allow negative forecasts for SHORT signals
            forecasts[sym] = forecast

    if forecasts:
        logger.info(
            "OI signals: %d/%d F&O stocks with signals (avg forecast=%.1f)",
            len(forecasts), len(oi_data),
            np.mean(list(forecasts.values())),
        )

    return forecasts


def is_fno_eligible(ticker: str) -> bool:
    """Check if a ticker is F&O eligible on NSE."""
    bare = ticker.replace('.NS', '').replace('.BO', '')
    return bare in FNO_LOT_SIZES


def get_lot_size(ticker: str) -> int:
    """Get F&O lot size for a ticker."""
    bare = ticker.replace('.NS', '').replace('.BO', '')
    return FNO_LOT_SIZES.get(bare, 0)
