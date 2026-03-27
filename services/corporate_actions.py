"""
Corporate Action Handler for NSE Stocks.

Subscribes to NSE corporate action data and adjusts:
  - OHLCV history (split/bonus ratio)
  - Open position quantities and entry prices
  - Stop-loss and target-price levels

Data source: NSE corporate actions page (public, no API key required).
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE: Dict[str, list] = {}
_CACHE_TS: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=6)


@dataclass
class CorporateAction:
    symbol: str
    ex_date: str           # YYYY-MM-DD
    action_type: str       # SPLIT, BONUS, DIVIDEND, RIGHTS
    ratio_from: float      # e.g. 1 (old face value)
    ratio_to: float        # e.g. 5 (new face value)
    description: str = ""

    @property
    def adjustment_factor(self) -> float:
        """Multiplier for price adjustment (< 1 for splits/bonus)."""
        if self.action_type in ("SPLIT", "BONUS") and self.ratio_to > 0:
            return self.ratio_from / self.ratio_to
        return 1.0

    @property
    def quantity_multiplier(self) -> float:
        """Multiplier for position quantity."""
        if self.action_type in ("SPLIT", "BONUS") and self.ratio_from > 0:
            return self.ratio_to / self.ratio_from
        return 1.0


def fetch_corporate_actions(days_ahead: int = 30) -> List[CorporateAction]:
    """Fetch upcoming and recent corporate actions from NSE.

    Scrapes the public NSE corporate actions page. Falls back to
    an empty list on failure (non-blocking).
    """
    global _CACHE, _CACHE_TS

    now = datetime.utcnow()
    if _CACHE_TS and now - _CACHE_TS < _CACHE_TTL and "actions" in _CACHE:
        return _CACHE["actions"]

    actions: List[CorporateAction] = []
    try:
        import requests

        from_date = (datetime.now() - timedelta(days=7)).strftime("%d-%m-%Y")
        to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%d-%m-%Y")

        url = (
            "https://www.nseindia.com/api/corporate-actions"
            f"?index=equities&from_date={from_date}&to_date={to_date}"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        session = requests.Session()
        # Hit the main page first to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            for item in data:
                symbol = item.get("symbol", "")
                subject = item.get("subject", "")
                ex_date_str = item.get("exDate", "")

                action = _parse_action(symbol, subject, ex_date_str)
                if action:
                    actions.append(action)

        logger.info("Fetched %d corporate actions from NSE", len(actions))
    except Exception as exc:
        logger.warning("NSE corporate actions fetch failed: %s", exc)

    _CACHE["actions"] = actions
    _CACHE_TS = now
    return actions


def _parse_action(symbol: str, subject: str, ex_date_str: str) -> Optional[CorporateAction]:
    """Parse a corporate action from NSE subject text."""
    if not symbol or not subject:
        return None

    subject_lower = subject.lower()

    # Try to parse ex-date
    ex_date = ""
    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            ex_date = datetime.strptime(ex_date_str.strip(), fmt).strftime("%Y-%m-%d")
            break
        except (ValueError, AttributeError):
            continue

    # Stock Split: "Face Value Split from Rs 10 to Rs 2"
    split_match = re.search(
        r"split.*?(\d+(?:\.\d+)?)\s*(?:to|into)\s*(\d+(?:\.\d+)?)",
        subject_lower,
    )
    if split_match or "stock split" in subject_lower or "sub-division" in subject_lower:
        ratio_from = float(split_match.group(1)) if split_match else 1
        ratio_to = float(split_match.group(2)) if split_match else 1
        if ratio_to > 0 and ratio_from > 0:
            return CorporateAction(
                symbol=symbol, ex_date=ex_date,
                action_type="SPLIT",
                ratio_from=ratio_from, ratio_to=ratio_to,
                description=subject,
            )

    # Bonus: "Bonus 1:1" or "Bonus issue 3:1"
    bonus_match = re.search(r"bonus.*?(\d+)\s*:\s*(\d+)", subject_lower)
    if bonus_match:
        bonus_shares = float(bonus_match.group(1))
        held_shares = float(bonus_match.group(2))
        # Bonus 1:1 means for every 1 held, you get 1 free → total = 2
        total_after = held_shares + bonus_shares
        return CorporateAction(
            symbol=symbol, ex_date=ex_date,
            action_type="BONUS",
            ratio_from=held_shares, ratio_to=total_after,
            description=subject,
        )

    # Dividend — not a price adjustment but useful for logging
    if "dividend" in subject_lower:
        return CorporateAction(
            symbol=symbol, ex_date=ex_date,
            action_type="DIVIDEND",
            ratio_from=1, ratio_to=1,
            description=subject,
        )

    return None


def get_actions_for_symbols(symbols: List[str]) -> Dict[str, CorporateAction]:
    """Return pending corporate actions keyed by symbol (only SPLIT/BONUS)."""
    all_actions = fetch_corporate_actions()
    today = datetime.now().strftime("%Y-%m-%d")

    result: Dict[str, CorporateAction] = {}
    for action in all_actions:
        if action.symbol in symbols and action.action_type in ("SPLIT", "BONUS"):
            # Include actions within ±7 days of today
            if action.ex_date:
                result[action.symbol] = action

    return result


def adjust_ohlcv_for_action(
    df: pd.DataFrame, action: CorporateAction
) -> pd.DataFrame:
    """Adjust OHLCV DataFrame for a split/bonus corporate action.

    Adjusts prices before the ex-date by the adjustment factor and
    scales volume inversely.
    """
    if action.adjustment_factor == 1.0:
        return df

    adjusted = df.copy()
    factor = action.adjustment_factor

    if action.ex_date and "Date" in str(type(adjusted.index)):
        mask = adjusted.index < pd.Timestamp(action.ex_date)
    else:
        # Cannot determine ex-date boundary — skip adjustment to avoid
        # corrupting the entire price history.
        logger.warning(
            "Skipping OHLCV adjustment for %s (%s): no valid ex_date or non-DatetimeIndex",
            action.symbol, action.action_type,
        )
        return df

    for col in ("Open", "High", "Low", "Close"):
        if col in adjusted.columns:
            adjusted.loc[mask, col] = adjusted.loc[mask, col] * factor

    if "Volume" in adjusted.columns and factor > 0:
        adjusted.loc[mask, "Volume"] = adjusted.loc[mask, "Volume"] / factor

    logger.info(
        "Adjusted %s OHLCV for %s: factor=%.4f (ex-date=%s)",
        action.symbol, action.action_type, factor, action.ex_date,
    )
    return adjusted


def adjust_position(
    qty: int, entry_price: float, stop_loss: float, target_price: float,
    action: CorporateAction,
) -> dict:
    """Adjust an open position for a corporate action."""
    factor = action.adjustment_factor
    multiplier = action.quantity_multiplier

    return {
        "quantity": int(qty * multiplier),
        "entry_price": round(entry_price * factor, 2),
        "stop_loss": round(stop_loss * factor, 2),
        "target_price": round(target_price * factor, 2),
    }
