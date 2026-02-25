"""
Embedding Service for Centurion Capital LLC RAG Pipeline.

Wraps sentence-transformers to produce dense vector embeddings
from text chunks. Designed so the embedding backend can be swapped
(e.g. OpenAI, Cohere) by implementing the same interface.

Default model: BAAI/bge-base-en-v1.5 (768-dim, instruction-tuned).
"""

import logging
from typing import List, Protocol

import numpy as np

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract protocol – implement this to swap backends
# ---------------------------------------------------------------------------
class EmbeddingBackend(Protocol):
    """Interface every embedding backend must satisfy."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors for the given texts."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (may apply query prefix)."""
        ...


# ---------------------------------------------------------------------------
# Default: sentence-transformers (local, free, fast)
# ---------------------------------------------------------------------------
class SentenceTransformerBackend:
    """
    Local embedding backend using sentence-transformers.

    Models live on HuggingFace Hub and are cached locally after first
    download.  BGE models benefit from a query-instruction prefix for
    asymmetric retrieval tasks.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        query_prefix: str = "",
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model: %s", self._model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts (no prefix)."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with optional instruction prefix."""
        prefixed = f"{self._query_prefix}{text}" if self._query_prefix else text
        embeddings = self.model.encode(
            [prefixed],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings[0].tolist()


# ---------------------------------------------------------------------------
# Factory / convenience
# ---------------------------------------------------------------------------
class EmbeddingService:
    """
    High-level embedding service used by the rest of the pipeline.

    Delegates to a configurable backend (default: sentence-transformers).
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        self._config = config or RAGConfig()
        self._backend = backend or SentenceTransformerBackend(
            model_name=self._config.embedding_model,
            query_prefix=self._config.embedding_query_prefix,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts."""
        return self._backend.embed(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string (with optional instruction prefix)."""
        return self._backend.embed_query(query)
