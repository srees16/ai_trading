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
