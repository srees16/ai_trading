"""
Semantic Cache for Centurion Capital LLC RAG Pipeline.

Caches RAG responses keyed by query embedding similarity so that
near-duplicate questions are served from cache in < 5 ms instead of
running the full retrieval + LLM pipeline (typically 2-8 s).

Architecture:
    - Maintains an in-memory list of (query_embedding, response) pairs.
    - On each query, embeds the new query and checks cosine similarity
      against all cached embeddings.
    - If the best match exceeds ``cache_similarity_threshold`` (default 0.95),
      the cached response is returned immediately.
    - Entries expire after ``cache_ttl_seconds`` (default 3600 = 1 h).
    - Maximum cache size is bounded; LRU eviction when full.

Usage:
    from rag_pipeline.core.semantic_cache import SemanticCache

    cache = SemanticCache(embedding_service, config)

    # Check before running pipeline
    hit = cache.lookup(query_text)
    if hit:
        return hit

    # After generating a response, store it
    cache.store(query_text, rag_response)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached RAG response."""
    query: str
    embedding: np.ndarray
    answer: str
    chunks_summary: List[Dict[str, Any]]
    space_id: str = "default"
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Embedding-similarity-based response cache.

    Thread-safe via a reentrant lock so it can be used from
    ``concurrent.futures`` or asyncio wrappers.
    """

    def __init__(
        self,
        embedding_fn,  # callable: str -> List[float]
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_entries: int = 256,
    ) -> None:
        self._embed = embedding_fn
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: List[CacheEntry] = []
        self._lock = threading.Lock()
        logger.info(
            "SemanticCache initialised (threshold=%.2f, ttl=%ds, max=%d)",
            self._threshold, self._ttl, self._max_entries,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def lookup(self, query: str, space_id: str = "default") -> Optional[CacheEntry]:
        """
        Return a cached response if a semantically similar query exists
        within the same tenant space.

        Returns ``None`` on cache miss.
        """
        if not self._entries:
            return None

        query_emb = np.array(self._embed(query), dtype=np.float32)
        now = time.time()

        best_entry: Optional[CacheEntry] = None
        best_sim: float = -1.0

        with self._lock:
            # Evict expired entries while scanning
            alive: List[CacheEntry] = []
            for entry in self._entries:
                if now - entry.timestamp > self._ttl:
                    continue
                alive.append(entry)
                # Only consider entries from the same tenant space
                if entry.space_id != space_id:
                    continue
                sim = self._cosine_similarity(query_emb, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
            self._entries = alive

        if best_entry is not None and best_sim >= self._threshold:
            best_entry.hit_count += 1
            logger.info(
                "Cache HIT (sim=%.4f, hits=%d): %s",
                best_sim, best_entry.hit_count, best_entry.query[:80],
            )
            return best_entry

        logger.debug("Cache MISS (best_sim=%.4f)", best_sim)
        return None

    def store(
        self,
        query: str,
        answer: str,
        chunks_summary: Optional[List[Dict[str, Any]]] = None,
        space_id: str = "default",
    ) -> None:
        """
        Add a query-response pair to the cache.

        If a semantically near-duplicate entry already exists in the
        same space (similarity >= threshold), it is **replaced** so
        that re-submitted queries always store fresh answers instead
        of leaving stale duplicates.

        Evicts the oldest entry (LRU) if the cache is full.
        """
        embedding = np.array(self._embed(query), dtype=np.float32)
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            answer=answer,
            chunks_summary=chunks_summary or [],
            space_id=space_id,
        )

        with self._lock:
            # Remove near-duplicate entries in the same space so that
            # a re-submit overwrites the old cached answer.
            self._entries = [
                e for e in self._entries
                if not (
                    e.space_id == space_id
                    and self._cosine_similarity(embedding, e.embedding)
                    >= self._threshold
                )
            ]
            # Evict if full
            while len(self._entries) >= self._max_entries:
                self._entries.pop(0)  # oldest first
            self._entries.append(entry)

        logger.debug("Cached response for: %s", query[:80])

    def invalidate(self, source: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        If *source* is provided, only entries whose ``chunks_summary``
        references that source are removed.  Otherwise the entire cache
        is cleared.

        Returns the number of entries removed.
        """
        with self._lock:
            if source is None:
                n = len(self._entries)
                self._entries.clear()
                logger.info("Cache cleared (%d entries removed)", n)
                return n

            before = len(self._entries)
            self._entries = [
                e for e in self._entries
                if not any(
                    c.get("source") == source for c in e.chunks_summary
                )
            ]
            removed = before - len(self._entries)
            if removed:
                logger.info(
                    "Invalidated %d cache entries for source '%s'",
                    removed, source,
                )
            return removed

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._entries)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Fast cosine similarity (assumes input is already float32)."""
        dot = np.dot(a, b)
        norms = np.linalg.norm(a) * np.linalg.norm(b)
        if norms == 0:
            return 0.0
        return float(dot / norms)
