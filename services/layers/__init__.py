"""
Market Layers — Trading terminal system design.

Organises centurion_core into the 6 institutional-grade layers:

1. market_data     — Real-time + historical data feeds
2. alpha_research  — Signal generation (strategies, ML, sentiment)
3. risk_engine     — Position limits, drawdown, circuit breakers
4. execution       — Order routing, fill management
5. portfolio       — Allocation, rebalancing, P&L tracking
6. monitoring      — Latency, health, alerts, audit trail

Each layer is a self-contained package with a well-defined interface
and communicates via the EventBus.
"""
