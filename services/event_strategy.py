"""
Event Strategy — Phase A-2.

Generates trading signals from upcoming market events.

Signal logic:
  - EARNINGS: Pre-earnings straddle / vol-expansion play.
    Generate positive forecast if IV is low relative to historical.
  - RBI_POLICY: Reduce exposure pre-RBI if VIX > 18.
  - FNO_EXPIRY: Tighten stops during expiry week. Avoid new entries.
  - REBALANCE: Front-run expected additions/deletions.

Each event generates a forecast contribution (-20 to +20) that feeds
into forecast_combiner as the "event_driven" source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EventForecast:
    """Forecast generated from an event."""
    symbol: str = ""
    event_type: str = ""
    event_date: str = ""
    days_until: int = 0
    forecast: float = 0.0      # -20 to +20
    action: str = "HOLD"       # INCREASE_VOL_EXPOSURE, REDUCE, TIGHTEN_STOPS, FRONT_RUN, HOLD
    reasoning: str = ""


def _earnings_forecast(
    symbol: str,
    days_until: int,
    current_iv: float = 0.0,
    historical_iv_mean: float = 0.0,
) -> EventForecast:
    """Pre-earnings signal.

    If IV is below average, there's potential for vol expansion → buy straddle.
    If IV is elevated, earnings are priced in → slight negative.
    """
    ef = EventForecast(
        symbol=symbol,
        event_type="EARNINGS",
        days_until=days_until,
    )

    if days_until > 5 or days_until < 0:
        return ef

    if current_iv > 0 and historical_iv_mean > 0:
        iv_ratio = current_iv / historical_iv_mean
        if iv_ratio < 0.8:
            # IV low → vol expansion expected
            ef.forecast = round(min(15.0, (1.0 - iv_ratio) * 30.0), 2)
            ef.action = "INCREASE_VOL_EXPOSURE"
            ef.reasoning = f"IV ratio {iv_ratio:.2f} below 0.8 → vol expansion expected"
        elif iv_ratio > 1.3:
            # IV elevated → earnings priced in, slight reduce
            ef.forecast = round(max(-10.0, -(iv_ratio - 1.0) * 10.0), 2)
            ef.action = "REDUCE"
            ef.reasoning = f"IV ratio {iv_ratio:.2f} above 1.3 → earnings priced in"
        else:
            ef.forecast = 0.0
            ef.action = "HOLD"
            ef.reasoning = f"IV ratio {iv_ratio:.2f} neutral"
    else:
        # No IV data → generic caution pre-earnings
        if days_until <= 2:
            ef.forecast = -3.0
            ef.action = "TIGHTEN_STOPS"
            ef.reasoning = "Earnings in ≤2 days, no IV data → tighten"
        else:
            ef.forecast = 2.0
            ef.action = "HOLD"
            ef.reasoning = "Earnings upcoming, no IV data → slight positive"

    return ef


def _rbi_policy_forecast(
    days_until: int,
    current_vix: float = 0.0,
) -> EventForecast:
    """RBI monetary policy signal.

    High VIX + policy date → reduce exposure.
    Low VIX + policy date → hold, potential positive surprise.
    """
    ef = EventForecast(
        symbol="MARKET",
        event_type="RBI_POLICY",
        days_until=days_until,
    )

    if days_until > 3 or days_until < 0:
        return ef

    if current_vix > 18:
        ef.forecast = round(max(-15.0, -current_vix * 0.5), 2)
        ef.action = "REDUCE"
        ef.reasoning = f"RBI policy in {days_until}d, VIX={current_vix:.1f} (>18) → reduce"
    elif current_vix > 0:
        ef.forecast = round(min(5.0, (18 - current_vix) * 0.3), 2)
        ef.action = "HOLD"
        ef.reasoning = f"RBI policy in {days_until}d, VIX={current_vix:.1f} (low) → calm"
    else:
        ef.forecast = -2.0
        ef.action = "TIGHTEN_STOPS"
        ef.reasoning = f"RBI policy in {days_until}d, no VIX data → cautious"

    return ef


def _fno_expiry_forecast(days_until: int) -> EventForecast:
    """F&O expiry week: tighten stops, avoid new entries."""
    ef = EventForecast(
        symbol="MARKET",
        event_type="FNO_EXPIRY",
        days_until=days_until,
    )

    if days_until > 3 or days_until < 0:
        return ef

    if days_until <= 1:
        ef.forecast = -5.0
        ef.action = "TIGHTEN_STOPS"
        ef.reasoning = f"F&O expiry in {days_until}d → tighten all stops"
    else:
        ef.forecast = -2.0
        ef.action = "TIGHTEN_STOPS"
        ef.reasoning = f"F&O expiry week ({days_until}d) → cautious"

    return ef


def _rebalance_forecast(
    symbol: str,
    days_until: int,
    expected_action: str = "",
) -> EventForecast:
    """Index rebalance front-running.

    Stocks expected to be ADDED get a positive forecast.
    Stocks expected to be REMOVED get a negative forecast.
    """
    ef = EventForecast(
        symbol=symbol,
        event_type="REBALANCE",
        days_until=days_until,
    )

    if days_until > 10 or days_until < 0:
        return ef

    if expected_action == "ADD":
        ef.forecast = round(min(10.0, 10.0 * (10 - days_until) / 10.0), 2)
        ef.action = "FRONT_RUN"
        ef.reasoning = f"Expected NIFTY addition in {days_until}d → accumulate"
    elif expected_action == "REMOVE":
        ef.forecast = round(max(-10.0, -10.0 * (10 - days_until) / 10.0), 2)
        ef.action = "REDUCE"
        ef.reasoning = f"Expected NIFTY removal in {days_until}d → reduce"
    else:
        ef.forecast = 0.0
        ef.action = "HOLD"

    return ef


def generate_event_forecasts(
    current_vix: float = 0.0,
    iv_data: Optional[Dict[str, float]] = None,
    historical_iv: Optional[Dict[str, float]] = None,
    rebalance_expectations: Optional[Dict[str, str]] = None,
) -> List[EventForecast]:
    """Generate forecasts for all upcoming events.

    Parameters
    ----------
    current_vix : float
        India VIX level.
    iv_data : dict, optional
        symbol -> current IV percentile.
    historical_iv : dict, optional
        symbol -> historical mean IV.
    rebalance_expectations : dict, optional
        symbol -> "ADD" or "REMOVE" for expected index changes.
    """
    try:
        from config import Config
        if not getattr(Config, "EVENT_DRIVEN_ENABLED", False):
            return []
    except Exception:
        return []

    try:
        from services.event_calendar import get_upcoming_events
        events = get_upcoming_events(days_ahead=7)
    except Exception as exc:
        logger.warning("Event calendar fetch failed: %s", exc)
        return []

    iv_data = iv_data or {}
    historical_iv = historical_iv or {}
    rebalance_expectations = rebalance_expectations or {}

    forecasts: List[EventForecast] = []

    for evt in events:
        if evt.event_type == "EARNINGS":
            ef = _earnings_forecast(
                symbol=evt.symbol,
                days_until=evt.days_until,
                current_iv=iv_data.get(evt.symbol, 0.0),
                historical_iv_mean=historical_iv.get(evt.symbol, 0.0),
            )
        elif evt.event_type == "RBI_POLICY":
            ef = _rbi_policy_forecast(evt.days_until, current_vix)
        elif evt.event_type == "FNO_EXPIRY":
            ef = _fno_expiry_forecast(evt.days_until)
        elif evt.event_type == "REBALANCE":
            ef = _rebalance_forecast(
                symbol=evt.symbol,
                days_until=evt.days_until,
                expected_action=rebalance_expectations.get(evt.symbol, ""),
            )
        else:
            continue

        ef.event_date = evt.event_date
        if ef.forecast != 0.0:
            forecasts.append(ef)

    logger.info(
        "Event strategy: %d forecasts from %d events",
        len(forecasts), len(events),
    )
    return forecasts
