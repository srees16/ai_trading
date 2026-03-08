"""
Tiered Retrieval for Centurion Capital LLC RAG Pipeline.

Implements a two-tier retrieval strategy:
    1. **FAQ tier** — A small, high-confidence collection of
       pre-answered questions.  Checked first with a strict similarity
       threshold.  If a match is found the full pipeline is skipped
       entirely (< 50 ms response time).
    2. **Full tier** — Falls through to the standard hybrid
       retrieval + re-ranking + LLM pipeline.

The FAQ collection can be populated manually or auto-populated from
repeat queries that produce high-confidence answers.

Usage:
    from rag_pipeline.ingestion.tiered_retrieval import TieredRetriever

    tiered = TieredRetriever(vector_store, embedding_service, config)

    # Fast path check
    faq_hit = tiered.check_faq(query_text)
    if faq_hit:
        return faq_hit  # Skip full pipeline

    # Otherwise proceed with full retrieval …
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag_pipeline.config import RAGConfig
from rag_pipeline.storage.embeddings import EmbeddingService
from rag_pipeline.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAQ entry
# ---------------------------------------------------------------------------

@dataclass
class FAQEntry:
    """A pre-answered FAQ entry."""
    question: str
    answer: str
    source: str = "faq"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Tiered retriever
# ---------------------------------------------------------------------------

class TieredRetriever:
    """
    Two-tier retrieval: FAQ fast-path → full pipeline fallback.

    The FAQ tier uses a separate ChromaDB collection with a tight
    similarity threshold (default 0.90).  Only queries that are very
    close semantically to an existing FAQ entry are served from this
    tier.  Everything else falls through.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_service: EmbeddingService,
        config: Optional[RAGConfig] = None,
        faq_collection_name: Optional[str] = None,
        faq_similarity_threshold: float = 0.90,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedding_service
        self._config = config or RAGConfig()
        self._faq_collection_name = (
            faq_collection_name
            or getattr(self._config, "faq_collection_name", "centurion_faq")
        )
        self._faq_threshold = faq_similarity_threshold
        self._faq_collection = None

    # ------------------------------------------------------------------ #
    # Lazy-init FAQ collection
    # ------------------------------------------------------------------ #

    def _ensure_faq_collection(self):
        """Get or create the FAQ ChromaDB collection."""
        if self._faq_collection is not None:
            return self._faq_collection
        try:
            import chromadb
            client = self._vs._client  # reuse the same PersistentClient
            self._faq_collection = client.get_or_create_collection(
                name=self._faq_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "FAQ collection '%s' ready (%d entries)",
                self._faq_collection_name,
                self._faq_collection.count(),
            )
        except Exception as e:
            logger.warning("Could not create FAQ collection: %s", e)
        return self._faq_collection

    # ------------------------------------------------------------------ #
    # Fast-path: check FAQ
    # ------------------------------------------------------------------ #

    def check_faq(
        self, query_text: str
    ) -> Optional[FAQEntry]:
        """
        Check the FAQ collection for a high-confidence match.

        Returns an ``FAQEntry`` if a match exceeds the similarity
        threshold, otherwise ``None``.
        """
        coll = self._ensure_faq_collection()
        if coll is None or coll.count() == 0:
            return None

        query_embedding = self._embedder.embed_query(query_text)

        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance  # cosine distance → similarity

        if similarity < self._faq_threshold:
            logger.debug(
                "FAQ miss (sim=%.4f < threshold=%.2f)",
                similarity, self._faq_threshold,
            )
            return None

        doc = results["documents"][0][0]
        meta = results["metadatas"][0][0] if results["metadatas"][0] else {}

        logger.info(
            "FAQ HIT (sim=%.4f): %s",
            similarity, query_text[:80],
        )

        return FAQEntry(
            question=meta.get("question", query_text),
            answer=doc,
            source="faq",
            metadata={**meta, "similarity": similarity},
        )

    # ------------------------------------------------------------------ #
    # Add / manage FAQ entries
    # ------------------------------------------------------------------ #

    def add_faq(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a question-answer pair to the FAQ collection.

        Returns the FAQ entry's ID.
        """
        coll = self._ensure_faq_collection()
        if coll is None:
            raise RuntimeError("FAQ collection is not available")

        import hashlib
        entry_id = hashlib.sha256(question.encode()).hexdigest()[:16]

        embedding = self._embedder.embed_query(question)

        entry_meta = {
            "question": question,
            "source": "faq",
            "created_at": time.time(),
            **(metadata or {}),
        }

        coll.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[entry_meta],
        )
        logger.info("FAQ entry added/updated: %s", question[:80])
        return entry_id

    def remove_faq(self, entry_id: str) -> None:
        """Remove a FAQ entry by ID."""
        coll = self._ensure_faq_collection()
        if coll is not None:
            coll.delete(ids=[entry_id])
            logger.info("FAQ entry removed: %s", entry_id)

    def list_faqs(self) -> List[Dict[str, Any]]:
        """Return all FAQ entries as dicts."""
        coll = self._ensure_faq_collection()
        if coll is None or coll.count() == 0:
            return []
        data = coll.get(include=["documents", "metadatas"])
        entries = []
        for i, doc_id in enumerate(data.get("ids", [])):
            meta = data["metadatas"][i] if data.get("metadatas") else {}
            entries.append({
                "id": doc_id,
                "question": meta.get("question", ""),
                "answer": data["documents"][i] if data.get("documents") else "",
                "metadata": meta,
            })
        return entries

    @property
    def faq_count(self) -> int:
        """Number of entries in the FAQ collection."""
        coll = self._ensure_faq_collection()
        return coll.count() if coll else 0
