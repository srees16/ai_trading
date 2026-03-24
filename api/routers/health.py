"""
Health check API endpoints.
"""

import logging
from datetime import datetime

from fastapi import APIRouter

from api.schemas.common import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check():
    """
    Check the health of all system components.
    """
    components: dict[str, bool] = {}

    # Database
    db_ok = False
    try:
        from api.dependencies import get_db_service
        db = get_db_service()
        db_ok = db is not None
    except Exception:
        pass
    components["database"] = db_ok

    # RAG / ChromaDB
    rag_ok = False
    try:
        from rag_pipeline.storage.vector_store import VectorStoreManager
        vs = VectorStoreManager()
        stats = vs.get_stats()
        rag_ok = stats is not None
    except Exception:
        pass
    components["rag_vector_store"] = rag_ok

    # Kite session
    kite_ok = False
    try:
        from api.dependencies import get_kite_session
        kite_ok = get_kite_session() is not None
    except Exception:
        pass
    components["kite_session"] = kite_ok

    overall = "healthy" if db_ok else "degraded"

    return HealthResponse(
        status=overall,
        database=db_ok,
        version="1.0.0",
        timestamp=datetime.utcnow(),
        components=components,
    )


@router.get(
    "/health/infra",
    summary="Infrastructure monitoring dashboard",
)
async def infra_health():
    """
    Detailed infrastructure health including event bus stats,
    latency tracking, execution mode, and event throughput.
    """
    from infrastructure.latency_tracker import latency_tracker
    from infrastructure.execution_context import execution_ctx
    from infrastructure.event_bus import event_bus

    latency_data = {}
    for label, stats in latency_tracker.get_all_stats().items():
        latency_data[label] = {
            "count": stats.count,
            "p50_ms": round(stats.p50_ms, 2),
            "p95_ms": round(stats.p95_ms, 2),
            "p99_ms": round(stats.p99_ms, 2),
            "max_ms": round(stats.max_ms, 2),
            "sla_breaches": stats.sla_breaches,
        }

    try:
        from layers.monitoring import monitoring_service
        monitoring_data = monitoring_service.get_health()
    except Exception:
        monitoring_data = {}

    return {
        "execution_mode": execution_ctx.mode,
        "event_bus_topics": event_bus.topics,
        "event_log_size": len(event_bus.get_log()),
        "latency": latency_data,
        "monitoring": monitoring_data,
    }
