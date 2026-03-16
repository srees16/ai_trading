"""
Centurion Core — Shared Infrastructure Layer.

Provides cross-cutting concerns used by all market modules
(US stocks, IND stocks, Crypto, etc.):

* **Event Bus** — In-process pub/sub with async support
* **Time-Series Store** — Abstraction over TimescaleDB / in-memory
* **Model Registry** — Versioned ML model catalogue
* **Logging** — Structured JSON logging with correlation IDs
* **Auth** — Shared session tokens (Streamlit ↔ FastAPI)
* **Config** — Centralised configuration
* **Replay** — Deterministic event replay for backtest / audit
"""
