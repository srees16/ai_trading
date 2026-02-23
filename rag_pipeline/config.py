"""
RAG Pipeline Configuration for Centurion Capital LLC.

All RAG-specific settings centralised here. Override via environment
variables prefixed with CENTURION_RAG_.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent

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
            "CENTURION_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_EMBEDDING_DIM", "384")
        )
    )

    # ------------------------------------------------------------------
    # PDF Chunking
    # ------------------------------------------------------------------
    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CHUNK_SIZE", "1000")
        )
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_CHUNK_OVERLAP", "200")
        )
    )

    # ------------------------------------------------------------------
    # Query / Retrieval
    # ------------------------------------------------------------------
    top_k: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_TOP_K", "20")
        )
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CENTURION_RAG_SIMILARITY_THRESHOLD", "1.0")
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
    # LLM (Ollama local inference)
    # ------------------------------------------------------------------
    llm_provider: str = field(
        default_factory=lambda: os.getenv(
            "CENTURION_RAG_LLM_PROVIDER", "ollama"
        )
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
            os.getenv("CENTURION_RAG_LLM_TEMPERATURE", "0.3")
        )
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_LLM_MAX_TOKENS", "1024")
        )
    )
    llm_timeout: int = field(
        default_factory=lambda: int(
            os.getenv("CENTURION_RAG_LLM_TIMEOUT", "600")
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
