"""
Webhook-based real-time price streaming for Indian stocks.

Replaces the polling-based approach (kite.quote() every N seconds)
with a push-based architecture using Kite Connect WebSocket (KiteTicker)
and an internal webhook dispatcher that propagates tick events to
subscribers (DB updater, UI, alert system, etc.).
"""
