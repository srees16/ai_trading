"""
RAG Pipeline Module for Centurion Capital LLC.

Retrieval-Augmented Generation pipeline for ingesting PDF strategy
documents and providing context-aware responses to user queries.

Components:
    - config: RAG-specific configuration & constants
    - vector_store: ChromaDB vector storage and retrieval
    - pdf_ingestion: PDF parsing, chunking, and embedding
    - query_engine: RAG query orchestration
    - ui: Streamlit UI components for RAG interaction

Usage:
    from rag_pipeline import RAGConfig, VectorStoreManager, PDFIngestionService, RAGQueryEngine

    # Initialize
    config = RAGConfig()
    vector_store = VectorStoreManager(config)
    ingestion = PDFIngestionService(vector_store, config)
    engine = RAGQueryEngine(vector_store, config)

    # Ingest a PDF
    ingestion.ingest_pdf("path/to/strategy.pdf")

    # Query
    results = engine.query("What are the momentum indicators?")
"""

import logging

# ---------------------------------------------------------------------------
# Logging setup for the RAG pipeline
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from rag_pipeline.config import RAGConfig
from rag_pipeline.vector_store import VectorStoreManager
from rag_pipeline.pdf_ingestion import PDFIngestionService
from rag_pipeline.query_engine import RAGQueryEngine
from rag_pipeline.reranker import CrossEncoderReranker
from rag_pipeline.llm_service import OllamaLLMBackend

__all__ = [
    "RAGConfig",
    "VectorStoreManager",
    "PDFIngestionService",
    "RAGQueryEngine",
    "CrossEncoderReranker",
    "OllamaLLMBackend",
]
