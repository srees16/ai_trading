"""
Real-time streaming router for Centurion Core — Indian Stocks.

Provides five enhancement endpoints:

1. **SSE tick stream**    — ``GET  /stream/sse``
2. **WebSocket proxy**    — ``WS   /stream/ws``
3. **Kite Postback**      — ``POST /stream/postback``
4. **OHLC aggregates**    — ``GET  /stream/ohlc/{symbol}``
5. **Price alerts**       — ``CRUD /stream/alerts``
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse

from api.schemas.streaming import (
    AlertCondition,
    AlertCreateResponse,
    AlertInfo,
    AlertListResponse,
    CreateAlertRequest,
    KitePostbackPayload,
    OHLCBar,
    OHLCResponse,
    PostbackResponse,
    StreamMessage,
    StreamTickData,
    WSSubscribeRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Real-time Streaming"])


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_dispatcher():
    """Return the singleton WebhookDispatcher."""
    from kite_connect.webhooks.dispatcher import WebhookDispatcher
    return WebhookDispatcher()


def _get_alert_engine():
    """Return the singleton PriceAlertEngine."""
    from kite_connect.webhooks.alert_engine import PriceAlertEngine
    return PriceAlertEngine.get_instance()


def _get_timescale_handler():
    """Return the singleton TimescaleTickHandler."""
    from kite_connect.webhooks.timescale_handler import TimescaleTickHandler

    # Lazy singleton — re-use same instance
    if not hasattr(_get_timescale_handler, "_instance"):
        _get_timescale_handler._instance = TimescaleTickHandler()
    return _get_timescale_handler._instance


def _get_webhook_service():
    """Return the singleton WebhookService."""
    from kite_connect.webhooks.service import WebhookService
    return WebhookService.get_instance()


# ═══════════════════════════════════════════════════════════════════════
# 1. Server-Sent Events (SSE) — GET /stream/sse
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/sse",
    summary="Live tick stream via Server-Sent Events",
    description=(
        "Opens a persistent HTTP connection and pushes tick batches as "
        "SSE events.  Each `data:` line is a JSON-encoded tick batch. "
        "Clients can use `EventSource` in the browser or `httpx` / "
        "`requests` with streaming."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE stream",
            "content": {"text/event-stream": {}},
        }
    },
)
async def sse_tick_stream(
    request: Request,
    symbols: Optional[str] = Query(
        None,
        description="Comma-separated symbols to filter (e.g. RELIANCE,TCS). Omit for all.",
    ),
):
    """
    Stream real-time ticks via Server-Sent Events.

    Uses an ``asyncio.Queue`` bridge: the sync WebhookDispatcher
    calls a callback that pushes into the queue, and the async
    generator yields from it as SSE events.
    """
    # Parse optional symbol filter
    symbol_filter = set()
    if symbols:
        symbol_filter = {s.strip().upper() for s in symbols.split(",") if s.strip()}

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    loop = asyncio.get_event_loop()
    subscriber_id = f"sse_{id(queue)}_{int(time.time())}"

    def _on_tick_batch(event):
        """Sync callback from the dispatcher — push into the async queue."""
        ticks = event.payload.get("ticks", [])
        if not ticks:
            return
        if symbol_filter:
            ticks = [t for t in ticks if t.get("name", "").upper() in symbol_filter]
            if not ticks:
                return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ticks)
        except asyncio.QueueFull:
            pass  # Drop oldest if backpressured

    # Subscribe to TICK_BATCH events
    from kite_connect.webhooks.events import EventType
    dispatcher = _get_dispatcher()
    dispatcher.subscribe(
        subscriber_id,
        [EventType.TICK_BATCH],
        _on_tick_batch,
        description=f"SSE client (filter={symbols or 'all'})",
    )

    async def _event_generator():
        """Yield SSE-formatted events."""
        try:
            # Initial connection event
            yield f"event: connected\ndata: {json.dumps({'subscriber_id': subscriber_id})}\n\n"

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    ticks = await asyncio.wait_for(queue.get(), timeout=30.0)
                    msg = StreamMessage(
                        event="tick_batch",
                        data=[
                            StreamTickData(
                                symbol=t.get("name", ""),
                                last_price=t.get("ltp", 0),
                                high=t.get("high", 0),
                                low=t.get("low", 0),
                                open=t.get("open", 0),
                                close=t.get("close", 0),
                                volume=t.get("volume", 0),
                                change_pct=t.get("change_pct", 0),
                                oi=t.get("oi", 0),
                                timestamp=t.get("timestamp", time.time()),
                            ).model_dump()
                            for t in ticks
                        ],
                    )
                    yield f"event: tick_batch\ndata: {msg.model_dump_json()}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive comment every 30s
                    yield f": keepalive {int(time.time())}\n\n"

        finally:
            dispatcher.unsubscribe(subscriber_id)
            logger.info("SSE client disconnected: %s", subscriber_id)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. WebSocket Proxy — WS /stream/ws
# ═══════════════════════════════════════════════════════════════════════

@router.websocket("/ws")
async def websocket_tick_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time tick streaming.

    Protocol:

    1. Client connects and receives ``{"event": "connected", ...}``.
    2. Client sends ``{"action": "subscribe", "symbols": ["RELIANCE", "TCS"]}``.
    3. Server pushes ``{"event": "tick_batch", "data": [...]}`` messages.
    4. Client can send ``{"action": "ping"}`` and receives ``{"event": "pong"}``.
    5. Client sends ``{"action": "unsubscribe", "symbols": [...]}`` to stop.
    """
    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    loop = asyncio.get_event_loop()
    subscriber_id = f"ws_{id(websocket)}_{int(time.time())}"
    symbol_filter: set = set()

    def _on_tick_batch(event):
        """Sync callback → async queue bridge."""
        ticks = event.payload.get("ticks", [])
        if not ticks:
            return
        if symbol_filter:
            ticks = [t for t in ticks if t.get("name", "").upper() in symbol_filter]
            if not ticks:
                return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ticks)
        except asyncio.QueueFull:
            pass

    # Also subscribe to STRATEGY_ALERT for real-time alerts
    from kite_connect.webhooks.events import EventType

    def _on_alert(event):
        """Forward alert events to the WebSocket client."""
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"_alert": True, "payload": event.payload},
            )
        except asyncio.QueueFull:
            pass

    dispatcher = _get_dispatcher()
    dispatcher.subscribe(
        subscriber_id,
        [EventType.TICK_BATCH, EventType.STRATEGY_ALERT],
        lambda evt: (
            _on_alert(evt)
            if evt.event_type == EventType.STRATEGY_ALERT
            else _on_tick_batch(evt)
        ),
        description="WebSocket client",
    )

    try:
        # Send connected message
        await websocket.send_json({
            "event": "connected",
            "subscriber_id": subscriber_id,
            "ts": time.time(),
        })

        async def _send_loop():
            """Push ticks/alerts from queue to websocket."""
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Keepalive ping
                    await websocket.send_json({"event": "ping", "ts": time.time()})
                    continue

                if isinstance(data, dict) and data.get("_alert"):
                    await websocket.send_json({
                        "event": "alert",
                        "data": data["payload"],
                        "ts": time.time(),
                    })
                else:
                    tick_list = [
                        {
                            "symbol": t.get("name", ""),
                            "last_price": t.get("ltp", 0),
                            "high": t.get("high", 0),
                            "low": t.get("low", 0),
                            "open": t.get("open", 0),
                            "close": t.get("close", 0),
                            "volume": t.get("volume", 0),
                            "change_pct": t.get("change_pct", 0),
                            "oi": t.get("oi", 0),
                            "timestamp": t.get("timestamp", time.time()),
                        }
                        for t in data
                    ]
                    await websocket.send_json({
                        "event": "tick_batch",
                        "data": tick_list,
                        "ts": time.time(),
                    })

        async def _recv_loop():
            """Listen for subscribe/unsubscribe/ping commands from client."""
            nonlocal symbol_filter
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"event": "error", "data": "Invalid JSON"})
                    continue

                action = msg.get("action", "").lower()
                symbols = msg.get("symbols", [])

                if action == "subscribe":
                    symbol_filter.update(s.upper() for s in symbols)
                    await websocket.send_json({
                        "event": "subscribed",
                        "symbols": sorted(symbol_filter),
                        "ts": time.time(),
                    })
                elif action == "unsubscribe":
                    symbol_filter -= {s.upper() for s in symbols}
                    await websocket.send_json({
                        "event": "unsubscribed",
                        "symbols": sorted(symbol_filter),
                        "ts": time.time(),
                    })
                elif action == "ping":
                    await websocket.send_json({"event": "pong", "ts": time.time()})
                else:
                    await websocket.send_json({
                        "event": "error",
                        "data": f"Unknown action: {action}",
                    })

        # Run both loops concurrently
        await asyncio.gather(_send_loop(), _recv_loop())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", subscriber_id)
    except Exception:
        logger.exception("WebSocket error: %s", subscriber_id)
    finally:
        dispatcher.unsubscribe(subscriber_id)


# ═══════════════════════════════════════════════════════════════════════
# 3. Kite Postback — POST /stream/postback
# ═══════════════════════════════════════════════════════════════════════

@router.post(
    "/postback",
    response_model=PostbackResponse,
    summary="Receive Kite order postback",
    description=(
        "Zerodha POSTs order status updates to this endpoint when an order "
        "transitions (e.g. OPEN → COMPLETE). Configure the postback URL in "
        "the Kite Developer Console: ``https://your-domain/stream/postback``."
    ),
)
async def kite_postback(payload: KitePostbackPayload, request: Request):
    """
    Process an incoming Kite postback and dispatch the appropriate
    event through the WebhookDispatcher.
    """
    # Optional: verify checksum
    api_secret = os.getenv("ZERODHA_API_SECRET", "")
    if payload.checksum and api_secret:
        # Zerodha checksum: SHA-256(order_id + order_timestamp + api_secret)
        raw = f"{payload.order_id}{payload.order_timestamp}{api_secret}"
        expected = hashlib.sha256(raw.encode()).hexdigest()
        if not hmac.compare_digest(expected, payload.checksum):
            logger.warning("Postback checksum mismatch for order %s", payload.order_id)
            raise HTTPException(status_code=403, detail="Invalid checksum")

    # Map Kite status to our EventType
    from kite_connect.webhooks.events import EventType, WebhookEvent

    status = (payload.status or "").upper()
    event_map = {
        "COMPLETE": EventType.TRADE_EXECUTED,
        "CANCELLED": EventType.ORDER_CANCELLED,
        "REJECTED": EventType.ORDER_REJECTED,
        "OPEN": EventType.ORDER_PLACED,
        "TRIGGER PENDING": EventType.ORDER_PLACED,
    }
    event_type = event_map.get(status, EventType.ORDER_PLACED)

    # Build event payload
    event_payload = {
        "order_id": payload.order_id,
        "status": payload.status,
        "tradingsymbol": payload.tradingsymbol,
        "exchange": payload.exchange,
        "transaction_type": payload.transaction_type,
        "order_type": payload.order_type,
        "quantity": payload.quantity,
        "filled_quantity": payload.filled_quantity,
        "average_price": payload.average_price,
        "price": payload.price,
        "trigger_price": payload.trigger_price,
        "product": payload.product,
        "variety": payload.variety,
        "tag": payload.tag,
        "order_timestamp": payload.order_timestamp,
        "exchange_timestamp": payload.exchange_timestamp,
        "status_message": payload.status_message,
    }

    # Dispatch event
    dispatcher = _get_dispatcher()
    event = WebhookEvent(
        event_type=event_type,
        payload=event_payload,
        source="kite_postback",
    )
    dispatcher.dispatch(event)

    logger.info(
        "Kite postback received: order=%s status=%s symbol=%s",
        payload.order_id, payload.status, payload.tradingsymbol,
    )

    return PostbackResponse(
        success=True,
        message=f"Postback processed: {payload.status}",
        event_type=event_type.value,
        order_id=payload.order_id,
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. OHLC Aggregates — GET /stream/ohlc/{symbol}
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/ohlc/{symbol}",
    response_model=OHLCResponse,
    summary="Get OHLC bars from TimescaleDB",
    description=(
        "Query pre-aggregated OHLC bars from TimescaleDB continuous "
        "aggregates.  Available intervals: 1m, 5m, 15m, 1h."
    ),
)
async def get_ohlc(
    symbol: str,
    interval: str = Query("1m", description="Aggregate interval: 1m | 5m | 15m | 1h"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Fetch OHLC bars for a symbol from the TimescaleDB aggregates."""
    handler = _get_timescale_handler()

    if interval not in handler._OHLC_VIEWS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval '{interval}'. Use: {list(handler._OHLC_VIEWS)}",
        )

    try:
        rows = handler.query_ohlc(symbol.upper(), interval, limit)
        bars = [
            OHLCBar(
                bucket=r["bucket"],
                symbol=r["symbol"],
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r.get("volume", 0),
                trade_count=r.get("trade_count", 0),
            )
            for r in rows
        ]
        return OHLCResponse(
            success=True,
            symbol=symbol.upper(),
            interval=interval,
            bars=bars,
        )
    except Exception as exc:
        logger.exception("OHLC query failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════
# 5. Price Alerts — CRUD /stream/alerts
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="List all configured price alerts",
)
async def list_alerts(
    active_only: bool = Query(False, description="Show only active alerts"),
):
    """Return all configured price alerts."""
    engine = _get_alert_engine()
    alerts = engine.list_alerts(only_active=active_only)
    return AlertListResponse(
        success=True,
        alerts=[AlertInfo(**a) for a in alerts],
        active_count=engine.active_count,
        total_count=engine.total_count,
    )


@router.post(
    "/alerts",
    response_model=AlertCreateResponse,
    summary="Create a new price alert",
    description=(
        "Set a price alert on a symbol. Supported conditions: "
        "``price_above``, ``price_below``, ``change_pct_above``, "
        "``change_pct_below``, ``volume_above``."
    ),
)
async def create_alert(req: CreateAlertRequest):
    """Create a price alert that fires when the condition is met."""
    engine = _get_alert_engine()
    alert_id = engine.create_alert(
        symbol=req.symbol,
        condition=req.condition.value,
        threshold=req.threshold,
        message=req.message or "",
        one_shot=req.one_shot,
    )
    return AlertCreateResponse(
        success=True,
        alert_id=alert_id,
        message=f"Alert created for {req.symbol} {req.condition.value} {req.threshold}",
    )


@router.delete(
    "/alerts/{alert_id}",
    summary="Delete a price alert",
)
async def delete_alert(alert_id: str):
    """Remove a price alert by its ID."""
    engine = _get_alert_engine()
    removed = engine.remove_alert(alert_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"success": True, "message": f"Alert {alert_id} removed"}


@router.delete(
    "/alerts",
    summary="Clear all price alerts",
)
async def clear_all_alerts():
    """Remove all configured price alerts."""
    engine = _get_alert_engine()
    count = engine.clear_all()
    return {"success": True, "message": f"{count} alerts removed"}


# ═══════════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/status",
    summary="Streaming system status",
    description="Returns the status of all real-time streaming components.",
)
async def streaming_status():
    """Full status of the real-time streaming pipeline."""
    result = {
        "success": True,
        "components": {},
    }

    # WebhookService status
    try:
        svc = _get_webhook_service()
        result["components"]["webhook_service"] = svc.get_status()
    except Exception as exc:
        result["components"]["webhook_service"] = {"error": str(exc)}

    # Alert engine stats
    try:
        engine = _get_alert_engine()
        result["components"]["alert_engine"] = {
            "active_alerts": engine.active_count,
            "total_alerts": engine.total_count,
            **engine.stats,
        }
    except Exception as exc:
        result["components"]["alert_engine"] = {"error": str(exc)}

    # TimescaleDB handler stats
    try:
        ts = _get_timescale_handler()
        result["components"]["timescale"] = {
            "schema_ready": ts._schema_ready,
            **ts.stats,
        }
    except Exception as exc:
        result["components"]["timescale"] = {"error": str(exc)}

    # Dispatcher stats
    try:
        d = _get_dispatcher()
        result["components"]["dispatcher"] = {
            "subscriber_count": d.subscriber_count,
            **d.stats,
        }
    except Exception as exc:
        result["components"]["dispatcher"] = {"error": str(exc)}

    return result
