"""
RAG Pipeline Configuration for Centurion Capital LLC.

All RAG-specific settings centralised here. Override via environment
variables prefixed with CENTURION_RAG_.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

# Load .env from the rag_pipeline directory (then project root as fallback)
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent

load_dotenv(_MODULE_DIR / ".env", override=False)
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# Default persistent directories
DEFAULT_CHROMA_DIR = str(_PROJECT_ROOT / "data" / "chroma_db")
DEFAULT_PDF_UPLOAD_DIR = str(_PROJECT_ROOT / "data" / "rag_uploads")


@dataclass
class RAGConfig:
    """
    Centralised configuration for the RAG pipeline.

    Every field can be overridden via an environment variable:
        CENTURION_RAG_<FIELD_NAME_UPPER>

    Example:
        CENTURION_RAG_CHUNK_SIZE=1000
        CENTURION_RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
    """

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_CHROMA_DIR", DEFAULT_CHROMA_DIR
        )
    )
    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_COLLECTION", "centurion_strategies"
        )
    )

    # ------------------------------------------------------------------
    # Embedding model (sentence-transformers compatible)
    # ------------------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
        )
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_EMBEDDING_DIM", "768")
        )
    )
    embedding_query_prefix: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_EMBEDDING_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
        )
    )

    # ------------------------------------------------------------------
    # PDF Chunking (token-based)
    # ------------------------------------------------------------------
    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CHUNK_SIZE", "512")
        )
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CHUNK_OVERLAP", "128")
        )
    )
    chunk_unit: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_CHUNK_UNIT", "token"
        ).lower()
    )  # "token" or "char"
    chunk_min_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CHUNK_MIN_TOKENS", "30")
        )
    )
    chunk_max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CHUNK_MAX_TOKENS", "800")
        )
    )

    # ------------------------------------------------------------------
    # Query / Retrieval
    # ------------------------------------------------------------------
    top_k: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_TOP_K", "25")
        )
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_SIMILARITY_THRESHOLD", "0.45")
        )
    )
    query_rewrite_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_QUERY_REWRITE", "true"
        ).lower() == "true"
    )
    query_rewrite_n: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_QUERY_REWRITE_N", "3")
        )
    )
    hyde_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_HYDE_ENABLED", "true"
        ).lower() == "true"
    )

    # ------------------------------------------------------------------
    # Hybrid search (BM25 + vector)
    # ------------------------------------------------------------------
    hybrid_search_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_HYBRID_SEARCH", "true"
        ).lower() == "true"
    )
    hybrid_bm25_weight: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_BM25_WEIGHT", "0.4")
        )
    )
    hybrid_vector_weight: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_VECTOR_WEIGHT", "0.6")
        )
    )

    # ------------------------------------------------------------------
    # Context noise reduction
    # ------------------------------------------------------------------
    max_context_chunks: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_MAX_CONTEXT_CHUNKS", "5")
        )
    )
    dedup_similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_DEDUP_SIM", "0.92")
        )
    )

    # ------------------------------------------------------------------
    # Re-ranking (cross-encoder)
    # ------------------------------------------------------------------
    reranker_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_RERANKER_ENABLED", "true"
        ).lower() == "true"
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
    )
    rerank_top_n: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_RERANK_TOP_N", "5")
        )
    )
    rerank_score_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_RERANK_SCORE_THRESHOLD", "0.25")
        )
    )

    # ------------------------------------------------------------------
    # HNSW index tuning (ChromaDB ANN parameters)
    # ------------------------------------------------------------------
    hnsw_m: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_HNSW_M", "32")
        )
    )
    hnsw_ef_construction: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_HNSW_EF_CONSTRUCTION", "200")
        )
    )
    hnsw_ef_search: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_HNSW_EF_SEARCH", "150")
        )
    )

    # ------------------------------------------------------------------
    # Metadata filtering (restrict retrieval to specific sources/pages)
    # ------------------------------------------------------------------
    enable_metadata_filtering: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_METADATA_FILTERING", "true"
        ).lower() == "true"
    )
    min_chunk_word_count: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_MIN_CHUNK_WORDS", "10")
        )
    )

    # ------------------------------------------------------------------
    # Multi-tenancy / document organisation
    # ------------------------------------------------------------------
    default_space_id: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_SPACE_ID", "default"
        )
    )
    default_doc_type: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_DOC_TYPE", "strategy"
        )
    )
    default_doc_version: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_DOC_VERSION", "1.0"
        )
    )
    enforce_space_isolation: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_SPACE_ISOLATION", "true"
        ).lower() == "true"
    )

    # ------------------------------------------------------------------
    # File uploads
    # ------------------------------------------------------------------
    pdf_upload_dir: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_UPLOAD_DIR", DEFAULT_PDF_UPLOAD_DIR
        )
    )
    max_upload_size_mb: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_MAX_UPLOAD_MB", "50")
        )
    )
    allowed_extensions: List[str] = field(
        default_factory=lambda: ["pdf"]
    )

    # ------------------------------------------------------------------
    # Feature toggle
    # ------------------------------------------------------------------
    rag_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_ENABLED", "true"
        ).lower() == "true"
    )

    # ------------------------------------------------------------------
    # Semantic cache
    # ------------------------------------------------------------------
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_CACHE_ENABLED", "false"
        ).lower() == "true"
    )
    cache_similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_CACHE_SIM_THRESHOLD", "0.95")
        )
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CACHE_TTL", "3600")
        )
    )
    cache_max_entries: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CACHE_MAX_ENTRIES", "256")
        )
    )

    # ------------------------------------------------------------------
    # Tiered retrieval (FAQ fast-path)
    # ------------------------------------------------------------------
    faq_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_FAQ_ENABLED", "false"
        ).lower() == "true"
    )
    faq_collection_name: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_FAQ_COLLECTION", "centurion_faq"
        )
    )
    faq_similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_FAQ_SIM_THRESHOLD", "0.90")
        )
    )

    # ------------------------------------------------------------------
    # Context token budget
    # ------------------------------------------------------------------
    context_token_budget: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CONTEXT_TOKEN_BUDGET", "4000")
        )
    )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    streaming_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_STREAMING", "false"
        ).lower() == "true"
    )

    # ------------------------------------------------------------------
    # Performance / observability
    # ------------------------------------------------------------------
    perf_logging_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_PERF_LOGGING", "true"
        ).lower() == "true"
    )

    # ------------------------------------------------------------------
    # LLM (Ollama local inference)
    # ------------------------------------------------------------------
    llm_provider: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_LLM_PROVIDER", "ollama"
        ).lower()
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_LLM_MODEL", "mistral"
        )
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_LLM_URL", "http://localhost:11434"
        )
    )
    llm_temperature: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_LLM_TEMPERATURE", "0.1")
        )
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_LLM_MAX_TOKENS", "1024")
        )
    )
    llm_timeout: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_LLM_TIMEOUT", "1200")
        )
    )

    # ------------------------------------------------------------------
    # LLM — Anthropic Claude (cloud inference)
    # ------------------------------------------------------------------
    claude_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    claude_model: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_CLAUDE_MODEL", "claude-sonnet-4-20250514"
        )
    )
    claude_max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CLAUDE_MAX_TOKENS", "1024")
        )
    )
    claude_temperature: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_CLAUDE_TEMPERATURE", "0.3")
        )
    )

    # ------------------------------------------------------------------
    # LLM — OpenAI (cloud inference)
    # ------------------------------------------------------------------
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_OPENAI_MODEL", "gpt-4o"
        )
    )
    openai_max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_OPENAI_MAX_TOKENS", "1024")
        )
    )
    openai_temperature: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_OPENAI_TEMPERATURE", "0.3")
        )
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create persistence directories if they don't exist."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.pdf_upload_dir).mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        self.ensure_directories()
