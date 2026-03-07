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

# ---------------------------------------------------------------------------
# Lazy public API — heavy submodules are only imported on first access.
# This avoids pulling in chromadb, numpy, sentence-transformers, etc.
# when the package is merely referenced by another import.
# ---------------------------------------------------------------------------

__all__ = [
    "RAGConfig",
    "VectorStoreManager",
    "PDFIngestionService",
    "RAGQueryEngine",
    "CrossEncoderReranker",
    "OllamaLLMBackend",
    "ClaudeLLMBackend",
    "OpenAILLMBackend",
    "create_llm_backend",
    "QueryRewriter",
    "HybridSearcher",
    "BM25Index",
    "EvalDataset",
    "EvalQuery",
    "EvalReport",
    "RetrievalLogger",
    "run_evaluation",
    "TripletExporter",
    "PipelineTrace",
    "Span",
    "count_tokens",
    "truncate_to_budget",
    "budget_chunks",
    "SemanticCache",
    "TieredRetriever",
    "FAQEntry",
    "chunk_structured_blocks",
    "chunk_pdf",
    "EnrichedChunk",
    "DualIndexStore",
    "upsert_chunks",
    "query_index",
    "get_embedding",
    "CODE_COLLECTION",
    "THEORY_COLLECTION",
    "classify_query",
    "classify_intent",
    "classify_pipeline_stages",
    "Retriever",
    "retrieve",
    "retrieve_context",
    "build_prompt_context",
    "RetrievalEvaluator",
    "RetrievalLog",
    "log_retrieval",
    "fetch_logs",
]

_LAZY_IMPORTS = {
    "RAGConfig":              "rag_pipeline.config",
    "VectorStoreManager":     "rag_pipeline.storage.vector_store",
    "PDFIngestionService":    "rag_pipeline.ingestion.pdf_ingestion",
    "RAGQueryEngine":         "rag_pipeline.core.query_engine",
    "CrossEncoderReranker":   "rag_pipeline.core.reranker",
    "OllamaLLMBackend":      "rag_pipeline.llm.llm_service",
    "ClaudeLLMBackend":      "rag_pipeline.llm.llm_service",
    "OpenAILLMBackend":      "rag_pipeline.llm.llm_service",
    "create_llm_backend":    "rag_pipeline.llm.llm_service",
    "QueryRewriter":          "rag_pipeline.core.query_rewriter",
    "HybridSearcher":         "rag_pipeline.core.hybrid_search",
    "BM25Index":              "rag_pipeline.core.hybrid_search",
    "EvalDataset":            "rag_pipeline.llm.evaluation",
    "EvalQuery":              "rag_pipeline.llm.evaluation",
    "EvalReport":             "rag_pipeline.llm.evaluation",
    "RetrievalLogger":        "rag_pipeline.llm.evaluation",
    "run_evaluation":         "rag_pipeline.llm.evaluation",
    "TripletExporter":        "rag_pipeline.storage.triplet_export",
    "PipelineTrace":          "rag_pipeline.utils.perf_trace",
    "Span":                   "rag_pipeline.utils.perf_trace",
    "count_tokens":           "rag_pipeline.utils.token_counter",
    "truncate_to_budget":     "rag_pipeline.utils.token_counter",
    "budget_chunks":          "rag_pipeline.utils.token_counter",
    "SemanticCache":          "rag_pipeline.core.semantic_cache",
    "TieredRetriever":        "rag_pipeline.ingestion.tiered_retrieval",
    "FAQEntry":               "rag_pipeline.ingestion.tiered_retrieval",
    "chunk_structured_blocks": "rag_pipeline.ingestion.chunking",
    "chunk_pdf":              "rag_pipeline.ingestion.chunking",
    "EnrichedChunk":          "rag_pipeline.ingestion.chunking",
    "DualIndexStore":         "rag_pipeline.storage.vector_store",
    "upsert_chunks":          "rag_pipeline.storage.vector_store",
    "query_index":            "rag_pipeline.storage.vector_store",
    "get_embedding":          "rag_pipeline.storage.vector_store",
    "CODE_COLLECTION":        "rag_pipeline.storage.vector_store",
    "THEORY_COLLECTION":      "rag_pipeline.storage.vector_store",
    "classify_query":         "rag_pipeline.core.query_classifier",
    "classify_intent":        "rag_pipeline.core.query_classifier",
    "classify_pipeline_stages": "rag_pipeline.core.query_classifier",
    "Retriever":              "rag_pipeline.core.retriever",
    "retrieve":               "rag_pipeline.core.retriever",
    "retrieve_context":       "rag_pipeline.core.retriever",
    "build_prompt_context":    "rag_pipeline.core.context_builder",
    "RetrievalEvaluator":      "rag_pipeline.utils.retrieval_evaluator",
    "RetrievalLog":            "rag_pipeline.utils.retrieval_evaluator",
    "log_retrieval":           "rag_pipeline.utils.retrieval_evaluator",
    "fetch_logs":              "rag_pipeline.utils.retrieval_evaluator",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        # Cache on the package so subsequent lookups are instant
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rag_pipeline' has no attribute {name!r}")
