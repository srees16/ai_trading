"""
Fine-tuning Triplet Exporter for Centurion Capital LLC RAG Pipeline.

Generates ``(query, positive, negative)`` triplets from retrieval logs
and evaluation datasets for contrastive embedding fine-tuning
(e.g. training a custom BGE / E5 model with in-domain data).

Export formats:
    - JSONL  — one JSON object per line (HuggingFace-compatible)
    - CSV    — query,positive,negative columns

Usage:
    from rag_pipeline.storage.triplet_export import TripletExporter

    exporter = TripletExporter(vector_store)
    triplets = exporter.generate_from_eval_dataset(dataset)
    exporter.export_jsonl(triplets, "training_triplets.jsonl")
"""

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_pipeline.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """A single training triplet for contrastive learning."""
    query: str
    positive: str  # relevant passage
    negative: str  # irrelevant passage


class TripletExporter:
    """
    Generate and export training triplets for embedding fine-tuning.

    Strategies for negative sampling:
        1. **Random negatives** — random chunks not in the relevant set
        2. **Hard negatives**   — high-similarity but irrelevant chunks
           (from retrieval results that are NOT in the relevant set)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
    ) -> None:
        self._vs = vector_store
        self._all_docs: Optional[List[str]] = None

    def _get_all_docs(self) -> List[str]:
        """Lazily fetch all documents from the vector store."""
        if self._all_docs is None and self._vs is not None:
            results = self._vs.collection.get(include=["documents"])
            self._all_docs = results.get("documents") or []
        return self._all_docs or []

    def generate_from_eval_dataset(
        self,
        dataset,
        query_engine=None,
        n_negatives: int = 3,
        top_k: int = 20,
        hard_negative: bool = True,
    ) -> List[Triplet]:
        """
        Generate triplets from an ``EvalDataset``.

        For each query:
            - Positives: text of chunks with IDs in ``relevant_doc_ids``
            - Hard negatives: top-K retrieved chunks NOT in relevant set
            - Random negatives: random chunks from the corpus

        Args:
            dataset:       EvalDataset with queries and relevant IDs.
            query_engine:  RAGQueryEngine (needed for hard negatives).
            n_negatives:   Number of negative passages per positive.
            top_k:         Number of candidates to retrieve for hard negs.
            hard_negative: If True, use hard negatives; else random.

        Returns:
            List of Triplet objects.
        """
        triplets: List[Triplet] = []
        all_docs = self._get_all_docs()

        for eq in dataset.queries:
            # Get positive passages
            positives = self._get_positive_passages(eq.relevant_doc_ids)
            if not positives:
                logger.warning(
                    "No positive passages found for query: %s", eq.query
                )
                continue

            # Get negative passages
            if hard_negative and query_engine is not None:
                negatives = self._get_hard_negatives(
                    eq.query, eq.relevant_doc_ids, query_engine, top_k
                )
            else:
                negatives = self._get_random_negatives(
                    positives, all_docs, n_negatives * len(positives)
                )

            # Create triplets
            for pos in positives:
                for neg in negatives[:n_negatives]:
                    triplets.append(Triplet(
                        query=eq.query,
                        positive=pos,
                        negative=neg,
                    ))

        logger.info("Generated %d triplets from %d queries",
                     len(triplets), len(dataset.queries))
        return triplets

    def generate_from_retrieval_log(
        self,
        log_entries: List[Dict[str, Any]],
        relevance_threshold: float = 0.3,
        n_negatives: int = 2,
    ) -> List[Triplet]:
        """
        Generate triplets from retrieval log entries.

        Assumes chunks with distance < *relevance_threshold* are positive
        and chunks with distance > 2x threshold are negative.
        """
        triplets: List[Triplet] = []
        all_docs = self._get_all_docs()

        for entry in log_entries:
            query = entry.get("query", "")
            distances = entry.get("distances", [])
            retrieved_ids = entry.get("retrieved_ids", [])

            if not query or not distances:
                continue

            positives_idx = [
                i for i, d in enumerate(distances) if d < relevance_threshold
            ]
            negatives_idx = [
                i for i, d in enumerate(distances) if d > relevance_threshold * 2
            ]

            if not positives_idx:
                continue

            # Look up actual text from vector store
            pos_texts = self._get_passages_by_ids(
                [retrieved_ids[i] for i in positives_idx if i < len(retrieved_ids)]
            )
            neg_texts = self._get_passages_by_ids(
                [retrieved_ids[i] for i in negatives_idx if i < len(retrieved_ids)]
            )

            if not neg_texts:
                neg_texts = self._get_random_negatives(pos_texts, all_docs, n_negatives)

            for pos in pos_texts:
                for neg in neg_texts[:n_negatives]:
                    triplets.append(Triplet(query=query, positive=pos, negative=neg))

        logger.info("Generated %d triplets from %d log entries",
                     len(triplets), len(log_entries))
        return triplets

    def _get_positive_passages(self, doc_ids: List[str]) -> List[str]:
        """Fetch passage text by chunk IDs / hashes from the vector store."""
        if not self._vs or not doc_ids:
            return []
        try:
            results = self._vs.collection.get(
                ids=doc_ids,
                include=["documents"],
            )
            return [d for d in (results.get("documents") or []) if d]
        except Exception:
            # IDs might be chunk_hashes — search by metadata
            return self._get_passages_by_hash(doc_ids)

    def _get_passages_by_hash(self, hashes: List[str]) -> List[str]:
        """Lookup passages by chunk_hash metadata."""
        texts = []
        if not self._vs:
            return texts
        for h in hashes:
            try:
                results = self._vs.collection.get(
                    where={"chunk_hash": h},
                    include=["documents"],
                )
                docs = results.get("documents") or []
                texts.extend(docs)
            except Exception:
                pass
        return texts

    def _get_passages_by_ids(self, ids: List[str]) -> List[str]:
        """Look up texts by document IDs."""
        if not self._vs or not ids:
            return []
        try:
            results = self._vs.collection.get(ids=ids, include=["documents"])
            return [d for d in (results.get("documents") or []) if d]
        except Exception:
            return []

    @staticmethod
    def _get_hard_negatives(
        query: str,
        relevant_ids: List[str],
        query_engine,
        top_k: int,
    ) -> List[str]:
        """Retrieve top-K and take passages NOT in the relevant set."""
        response = query_engine.query(query, top_k=top_k)
        relevant_set = set(relevant_ids)
        return [
            chunk.text for chunk in response.chunks
            if chunk.metadata.get("chunk_hash", "") not in relevant_set
        ]

    @staticmethod
    def _get_random_negatives(
        positives: List[str],
        corpus: List[str],
        n: int,
    ) -> List[str]:
        """Sample random negatives from the corpus, excluding positives."""
        pos_set = set(positives)
        candidates = [d for d in corpus if d not in pos_set]
        if not candidates:
            return []
        return random.sample(candidates, min(n, len(candidates)))

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    @staticmethod
    def export_jsonl(triplets: List[Triplet], path: str) -> None:
        """Export triplets to JSONL (HuggingFace training format)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for t in triplets:
                f.write(json.dumps({
                    "query": t.query,
                    "positive": t.positive,
                    "negative": t.negative,
                }) + "\n")
        logger.info("Exported %d triplets to %s", len(triplets), path)

    @staticmethod
    def export_csv(triplets: List[Triplet], path: str) -> None:
        """Export triplets to CSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "positive", "negative"])
            for t in triplets:
                writer.writerow([t.query, t.positive, t.negative])
        logger.info("Exported %d triplets to %s", len(triplets), path)
