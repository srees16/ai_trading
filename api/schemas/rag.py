"""
Pydantic schemas for RAG Pipeline API endpoints.

Covers: document ingestion, querying, collection management,
evaluation, and configuration.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    """Response after ingesting a document."""
    success: bool = True
    filename: str = ""
    chunks_created: int = 0
    message: str = ""


class IngestStatusResponse(BaseModel):
    """Response showing ingestion status for all documents."""
    success: bool = True
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    total_chunks: int = 0


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class RAGQueryRequest(BaseModel):
    """Request body for querying the RAG pipeline."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["What is momentum trading?"],
    )
    top_k: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    source_filter: Optional[str] = Field(
        None,
        description="Filter results to a specific source document",
    )
    skip_cache: bool = Field(False, description="Skip semantic cache lookup")


class RetrievedChunkResponse(BaseModel):
    """A single retrieved document chunk."""
    text: str
    source: str
    chunk_index: int
    distance: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    """Response from a RAG query."""
    success: bool = True
    query: str
    answer: str
    chunks: List[RetrievedChunkResponse] = Field(default_factory=list)
    cached: bool = False
    faq_hit: bool = False
    trace: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Collection Management
# ---------------------------------------------------------------------------

class CollectionStatsResponse(BaseModel):
    """Statistics about the vector store collection."""
    success: bool = True
    collection_name: str = ""
    total_chunks: int = 0
    sources: List[str] = Field(default_factory=list)
    embedding_model: str = ""


class DeleteDocumentRequest(BaseModel):
    """Request to delete a document from the vector store."""
    source: str = Field(..., description="Source identifier (filename) to delete")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvalQueryItem(BaseModel):
    """A single evaluation query with expected relevant docs."""
    query: str
    relevant_doc_ids: List[str] = Field(default_factory=list)
    relevant_sources: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class RunEvaluationRequest(BaseModel):
    """Request to run evaluation against a dataset."""
    queries: List[EvalQueryItem] = Field(
        ...,
        min_length=1,
        description="Evaluation queries with known relevant documents",
    )
    top_k: int = Field(10, ge=1, le=50)


class EvalMetrics(BaseModel):
    """Evaluation metrics for a single query."""
    query: str
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0


class RunEvaluationResponse(BaseModel):
    """Response from evaluation run."""
    success: bool = True
    per_query: List[EvalMetrics] = Field(default_factory=list)
    aggregate: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class RAGConfigResponse(BaseModel):
    """Current RAG pipeline configuration."""
    success: bool = True
    embedding_model: str = ""
    llm_provider: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    top_k: int = 5
    chroma_dir: str = ""
    collection_name: str = ""
