"""
Embedding Service for Centurion Capital LLC RAG Pipeline.

Wraps sentence-transformers to produce dense vector embeddings
from text chunks. Designed so the embedding backend can be swapped
(e.g. OpenAI, Cohere) by implementing the same interface.
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


# ---------------------------------------------------------------------------
# Default: sentence-transformers (local, free, fast)
# ---------------------------------------------------------------------------
class SentenceTransformerBackend:
    """
    Local embedding backend using sentence-transformers.

    Models live on HuggingFace Hub and are cached locally after first
    download (~90 MB for all-MiniLM-L6-v2).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model: %s", self._model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts and return float lists."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


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
            model_name=self._config.embedding_model
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return self._backend.embed(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string and return one vector."""
        result = self._backend.embed([query])
        return result[0] if result else []
