"""
ChromaDB Vector Store Manager for Centurion Capital LLC.

Provides a thin, project-aligned wrapper around ChromaDB for:
    - Collection lifecycle management
    - Document insertion with metadata
    - Similarity search with filtering
    - Collection statistics / health
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages a persistent ChromaDB collection for the RAG pipeline.

    Usage:
        config = RAGConfig()
        vs = VectorStoreManager(config)
        vs.add_documents(ids, texts, metadatas, embeddings)
        results = vs.query(query_embeddings, n_results=5)
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self._config = config or RAGConfig()
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------
    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self._config.chroma_persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(
                "ChromaDB client initialised at %s",
                self._config.chroma_persist_dir,
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._config.chroma_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": self._config.hnsw_m,
                    "hnsw:construction_ef": self._config.hnsw_ef_construction,
                    "hnsw:search_ef": self._config.hnsw_ef_search,
                },
            )
            logger.info(
                "Collection '%s' ready (%d documents) "
                "[HNSW M=%d, ef_construction=%d, ef_search=%d]",
                self._config.chroma_collection_name,
                self._collection.count(),
                self._config.hnsw_m,
                self._config.hnsw_ef_construction,
                self._config.hnsw_ef_search,
            )
        return self._collection

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Insert or upsert documents into the collection."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %d documents into collection", len(ids))

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform similarity search.

        Returns ChromaDB-style results dict with keys:
            ids, documents, metadatas, distances
        """
        include = include or ["documents", "metadatas", "distances"]
        kwargs: Dict[str, Any] = {
            "n_results": n_results,
            "include": include,
        }
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        if where is not None:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def delete_by_metadata(
        self, where: Dict[str, Any]
    ) -> None:
        """Delete documents matching a metadata filter."""
        # ChromaDB requires IDs for deletion; fetch them first.
        results = self.collection.get(where=where)
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(
                "Deleted %d documents matching %s", len(results["ids"]), where
            )

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        if ids:
            self.collection.delete(ids=ids)
            logger.info("Deleted %d documents by ID", len(ids))

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return total document count in the collection."""
        return self.collection.count()

    def source_exists(self, source_name: str) -> bool:
        """Check whether any chunks from *source_name* are stored."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=[],  # only need IDs – skip heavy fields
            )
            return bool(results.get("ids"))
        except Exception:
            return False

    def get_source_file_hash(self, source_name: str) -> Optional[str]:
        """Return the ``file_hash`` stored for *source_name*, or None.

        Looks at the first chunk's metadata.  All chunks from the
        same file share the same ``file_hash``.
        """
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=["metadatas"],
            )
            metas = results.get("metadatas") or []
            if metas and metas[0]:
                return metas[0].get("file_hash")
        except Exception:
            pass
        return None

    def get_source_chunk_count(self, source_name: str) -> int:
        """Return the number of chunks stored for *source_name*."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=[],
            )
            return len(results.get("ids") or [])
        except Exception:
            return 0

    def list_sources(self) -> List[str]:
        """Return unique source filenames stored in the collection."""
        all_meta = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in (all_meta.get("metadatas") or []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the collection."""
        count = self.count()
        sources = self.list_sources()
        return {
            "collection_name": self._config.chroma_collection_name,
            "total_documents": count,
            "total_sources": len(sources),
            "sources": sources,
            "persist_directory": self._config.chroma_persist_dir,
            "hnsw_m": self._config.hnsw_m,
            "hnsw_ef_construction": self._config.hnsw_ef_construction,
            "hnsw_ef_search": self._config.hnsw_ef_search,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset_collection(self) -> None:
        """Drop and recreate the collection (destructive)."""
        self.client.delete_collection(self._config.chroma_collection_name)
        self._collection = None
        logger.warning(
            "Collection '%s' has been reset",
            self._config.chroma_collection_name,
        )
        # Re-create so subsequent calls work immediately
        _ = self.collection
