"""
RAG Pipeline API router.

Endpoints for document ingestion, querying, collection management,
evaluation, and configuration.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from api.dependencies import get_rag_engine
from api.schemas.common import ErrorResponse, SuccessResponse
from api.schemas.rag import (
    CollectionStatsResponse,
    DeleteDocumentRequest,
    EvalMetrics,
    EvalQueryItem,
    IngestResponse,
    IngestStatusResponse,
    RAGConfigResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunkResponse,
    RunEvaluationRequest,
    RunEvaluationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _get_vector_store():
    """Get VectorStoreManager from the RAG engine."""
    engine = get_rag_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")
    return engine._vs


def _get_ingestion_service():
    """Build a PDFIngestionService wired to the running engine."""
    engine = get_rag_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")
    try:
        from rag_pipeline.pdf_ingestion import PDFIngestionService

        return PDFIngestionService(
            vector_store=engine._vs,
            config=engine._config,
            embedding_service=engine._embedder,
            on_change_callback=engine.invalidate_cache,
        )
    except Exception as exc:
        logger.exception("Failed to build ingestion service")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Document Ingestion
# -----------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and ingest a PDF",
)
async def ingest_pdf(
    file: UploadFile = File(...),
    force: bool = Query(False, description="Force re-ingestion even if duplicate"),
):
    """
    Upload a PDF file that will be chunked, embedded, and stored in
    the vector database for RAG.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    svc = _get_ingestion_service()
    try:
        file_bytes = await file.read()
        result = svc.ingest_uploaded_bytes(
            file_name=file.filename,
            file_bytes=file_bytes,
            extra_metadata={"upload_source": "api"},
        )
        return IngestResponse(
            success=True,
            filename=file.filename,
            chunks_created=result.get("chunks_stored", result.get("chunks", 0)),
            message=result.get("status", "ingested"),
        )
    except Exception as exc:
        logger.exception("PDF ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/ingest/directory",
    response_model=IngestStatusResponse,
    summary="Ingest all PDFs from a directory",
)
async def ingest_directory(
    directory: Optional[str] = Query(
        None,
        description="Absolute path to the PDF directory. "
                    "Defaults to the configured upload directory.",
    ),
    recursive: bool = Query(False),
):
    """Batch-ingest all PDFs in a directory."""
    svc = _get_ingestion_service()
    target = directory or svc._config.pdf_upload_dir
    if not os.path.isdir(target):
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {target}")

    try:
        results = svc.ingest_directory(target, recursive=recursive)
        total_chunks = sum(r.get("chunks_stored", r.get("chunks", 0)) for r in results)
        return IngestStatusResponse(
            success=True,
            documents=results,
            total_chunks=total_chunks,
        )
    except Exception as exc:
        logger.exception("Directory ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/reingest",
    response_model=IngestStatusResponse,
    summary="Re-ingest all PDFs with latest pipeline",
)
async def reingest_all():
    """Delete all existing chunks and re-ingest every PDF in the upload dir."""
    svc = _get_ingestion_service()
    try:
        results = svc.reingest_all()
        total_chunks = sum(r.get("chunks_stored", r.get("chunks", 0)) for r in results)
        return IngestStatusResponse(
            success=True,
            documents=results,
            total_chunks=total_chunks,
        )
    except Exception as exc:
        logger.exception("Reingest failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Query
# -----------------------------------------------------------------------

@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query the RAG pipeline",
)
async def rag_query(request: RAGQueryRequest):
    """
    Run the full RAG pipeline — semantic cache, FAQ fast-path,
    hybrid search, re-rank, LLM generation.
    """
    engine = get_rag_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")

    try:
        response = engine.query(
            query_text=request.query,
            top_k=request.top_k,
            source_filter=request.source_filter,
            skip_cache=request.skip_cache,
        )

        chunks = [
            RetrievedChunkResponse(
                text=c.text,
                source=c.source,
                chunk_index=c.chunk_index,
                distance=c.distance,
                metadata=c.metadata,
            )
            for c in response.chunks
        ]

        trace_dict = None
        if response.trace is not None:
            try:
                trace_dict = response.trace.to_dict()
            except Exception:
                trace_dict = {"raw": str(response.trace)}

        return RAGQueryResponse(
            success=True,
            query=response.query,
            answer=response.answer,
            chunks=chunks,
            cached=response.cached,
            faq_hit=response.faq_hit,
            trace=trace_dict,
        )
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Collection Management
# -----------------------------------------------------------------------

@router.get(
    "/collection/stats",
    response_model=CollectionStatsResponse,
    summary="Get vector store collection statistics",
)
async def collection_stats():
    """Return statistics about the ChromaDB collection."""
    try:
        vs = _get_vector_store()
        stats = vs.get_collection_stats()
        return CollectionStatsResponse(
            success=True,
            collection_name=stats.get("collection_name", ""),
            total_chunks=stats.get("total_documents", 0),
            sources=stats.get("sources", []),
            embedding_model=stats.get("embedding_model", ""),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get collection stats")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/collection/sources",
    response_model=SuccessResponse,
    summary="List all ingested source documents",
)
async def list_sources():
    """Return a list of all source filenames in the vector store."""
    try:
        vs = _get_vector_store()
        sources = vs.list_sources()
        return SuccessResponse(success=True, data={"sources": sources, "count": len(sources)})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete(
    "/documents",
    response_model=SuccessResponse,
    summary="Delete a document from the vector store",
)
async def delete_document(request: DeleteDocumentRequest):
    """Remove all chunks belonging to a given source PDF."""
    svc = _get_ingestion_service()
    try:
        removed = svc.delete_source(request.source)
        return SuccessResponse(
            success=True,
            data={
                "source": request.source,
                "chunks_removed": removed,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete(
    "/collection/reset",
    response_model=SuccessResponse,
    summary="Reset the entire collection (destructive)",
)
async def reset_collection():
    """Drop and recreate the ChromaDB collection. All data is lost."""
    try:
        vs = _get_vector_store()
        vs.reset_collection()
        return SuccessResponse(success=True, data={"message": "Collection reset successfully"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

@router.post(
    "/evaluate",
    response_model=RunEvaluationResponse,
    summary="Run retrieval evaluation",
)
async def run_evaluation(request: RunEvaluationRequest):
    """
    Run a batch evaluation of the RAG engine against a set of queries
    with known-relevant documents. Returns per-query and aggregate metrics.
    """
    engine = get_rag_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")

    try:
        from rag_pipeline.evaluation import (
            EvalDataset,
            EvalQuery,
            run_evaluation as _run_eval,
        )

        dataset = EvalDataset(name="api_eval")
        for q in request.queries:
            dataset.add(
                EvalQuery(
                    query=q.query,
                    relevant_doc_ids=q.relevant_doc_ids,
                    relevant_sources=q.relevant_sources,
                    tags=q.tags,
                )
            )

        report = _run_eval(engine, dataset, top_k=request.top_k)

        per_query = [
            EvalMetrics(
                query=r.query,
                recall_at_k=r.recall_10,
                precision_at_k=r.precision_10,
                mrr=r.mrr,
                ndcg_at_k=r.ndcg_10,
                hit_rate=r.hit_rate_10,
            )
            for r in report.results
        ]

        aggregate = {
            "avg_recall_5": report.avg_recall_5,
            "avg_recall_10": report.avg_recall_10,
            "avg_precision_5": report.avg_precision_5,
            "avg_precision_10": report.avg_precision_10,
            "avg_mrr": report.avg_mrr,
            "avg_ndcg_5": report.avg_ndcg_5,
            "avg_ndcg_10": report.avg_ndcg_10,
        }

        return RunEvaluationResponse(
            success=True,
            per_query=per_query,
            aggregate=aggregate,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@router.get(
    "/config",
    response_model=RAGConfigResponse,
    summary="Get current RAG configuration",
)
async def get_rag_config():
    """Return the current RAG pipeline configuration parameters."""
    try:
        from rag_pipeline.config import RAGConfig

        config = RAGConfig()
        return RAGConfigResponse(
            success=True,
            embedding_model=config.embedding_model,
            llm_provider=config.llm_provider,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            top_k=config.top_k,
            chroma_dir=config.chroma_persist_dir,
            collection_name=config.chroma_collection_name,
        )
    except Exception as exc:
        logger.exception("Failed to read RAG config")
        raise HTTPException(status_code=500, detail=str(exc))
