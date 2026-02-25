"""
Retrieval Evaluation Framework for Centurion Capital LLC RAG Pipeline.

Provides offline metrics to measure retrieval quality:
    - Recall@K         — fraction of relevant docs in top-K
    - MRR              — Mean Reciprocal Rank
    - nDCG@K           — normalised Discounted Cumulative Gain
    - Precision@K      — fraction of top-K that are relevant
    - Hit Rate@K       — binary: did any relevant doc appear in top-K?

Also provides:
    - ``RetrievalLogger``  — logs every query + results for later analysis
    - ``EvalDataset``      — stores test queries with expected doc IDs
    - ``run_evaluation()`` — batch-evaluates a query engine against a dataset

Usage:
    from rag_pipeline.evaluation import (
        EvalDataset,
        EvalQuery,
        RetrievalLogger,
        run_evaluation,
    )

    dataset = EvalDataset()
    dataset.add(EvalQuery(
        query="RSI momentum strategy",
        relevant_doc_ids=["abc_0001", "abc_0003"],
    ))

    report = run_evaluation(query_engine, dataset, top_k=10)
    print(report.summary())
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """A single evaluation query with known relevant document IDs."""
    query: str
    relevant_doc_ids: List[str] = field(default_factory=list)
    relevant_sources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class EvalDataset:
    """Collection of evaluation queries."""
    name: str = "default"
    queries: List[EvalQuery] = field(default_factory=list)

    def add(self, eq: EvalQuery) -> None:
        self.queries.append(eq)

    def save(self, path: str) -> None:
        """Save dataset to JSON."""
        data = {
            "name": self.name,
            "queries": [
                {
                    "query": q.query,
                    "relevant_doc_ids": q.relevant_doc_ids,
                    "relevant_sources": q.relevant_sources,
                    "tags": q.tags,
                }
                for q in self.queries
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Saved eval dataset (%d queries) to %s", len(self.queries), path)

    @classmethod
    def load(cls, path: str) -> "EvalDataset":
        """Load dataset from JSON."""
        raw = json.loads(Path(path).read_text())
        ds = cls(name=raw.get("name", "loaded"))
        for q in raw.get("queries", []):
            ds.add(EvalQuery(
                query=q["query"],
                relevant_doc_ids=q.get("relevant_doc_ids", []),
                relevant_sources=q.get("relevant_sources", []),
                tags=q.get("tags", []),
            ))
        return ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """Fraction of relevant documents that appear in the top-K retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """Fraction of top-K retrieved that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for x in top_k if x in relevant) / k


def hit_rate_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """1.0 if any relevant doc appears in top-K, else 0.0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def reciprocal_rank(
    retrieved_ids: List[str], relevant_ids: List[str]
) -> float:
    """Reciprocal rank of the first relevant doc in the retrieved list."""
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Uses binary relevance (1 if in relevant set, 0 otherwise).
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-query evaluation result
# ---------------------------------------------------------------------------

@dataclass
class QueryEvalResult:
    """Evaluation result for one query."""
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    recall_5: float = 0.0
    recall_10: float = 0.0
    precision_5: float = 0.0
    precision_10: float = 0.0
    mrr: float = 0.0
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    hit_rate_5: float = 0.0
    hit_rate_10: float = 0.0


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    dataset_name: str
    n_queries: int
    results: List[QueryEvalResult] = field(default_factory=list)
    avg_recall_5: float = 0.0
    avg_recall_10: float = 0.0
    avg_precision_5: float = 0.0
    avg_precision_10: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg_5: float = 0.0
    avg_ndcg_10: float = 0.0
    avg_hit_rate_5: float = 0.0
    avg_hit_rate_10: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"=== Evaluation Report: {self.dataset_name} ({self.n_queries} queries) ===\n"
            f"  Recall@5:     {self.avg_recall_5:.3f}\n"
            f"  Recall@10:    {self.avg_recall_10:.3f}\n"
            f"  Precision@5:  {self.avg_precision_5:.3f}\n"
            f"  Precision@10: {self.avg_precision_10:.3f}\n"
            f"  MRR:          {self.avg_mrr:.3f}\n"
            f"  nDCG@5:       {self.avg_ndcg_5:.3f}\n"
            f"  nDCG@10:      {self.avg_ndcg_10:.3f}\n"
            f"  Hit Rate@5:   {self.avg_hit_rate_5:.3f}\n"
            f"  Hit Rate@10:  {self.avg_hit_rate_10:.3f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "n_queries": self.n_queries,
            "avg_recall_5": self.avg_recall_5,
            "avg_recall_10": self.avg_recall_10,
            "avg_precision_5": self.avg_precision_5,
            "avg_precision_10": self.avg_precision_10,
            "avg_mrr": self.avg_mrr,
            "avg_ndcg_5": self.avg_ndcg_5,
            "avg_ndcg_10": self.avg_ndcg_10,
            "avg_hit_rate_5": self.avg_hit_rate_5,
            "avg_hit_rate_10": self.avg_hit_rate_10,
        }


# ---------------------------------------------------------------------------
# Retrieval Logger
# ---------------------------------------------------------------------------

class RetrievalLogger:
    """
    Logs every query + retrieved results for offline analysis.

    Appends JSONL (one JSON object per line) to a log file.
    """

    def __init__(self, log_path: Optional[str] = None) -> None:
        self._path = Path(log_path or "data/retrieval_log.jsonl")
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        query: str,
        retrieved_ids: List[str],
        retrieved_sources: List[str],
        distances: List[float],
        rerank_scores: Optional[List[float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a retrieval event to the log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "retrieved_ids": retrieved_ids,
            "retrieved_sources": retrieved_sources,
            "distances": [round(d, 4) for d in distances],
        }
        if rerank_scores:
            entry["rerank_scores"] = [round(s, 4) for s in rerank_scores]
        if extra:
            entry.update(extra)

        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all logged entries."""
        if not self._path.exists():
            return []
        entries = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries


# ---------------------------------------------------------------------------
# Batch evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    query_engine,
    dataset: EvalDataset,
    top_k: int = 20,
) -> EvalReport:
    """
    Run all queries in *dataset* through *query_engine* and compute
    retrieval metrics.

    The query_engine must have a ``.query(query_text, top_k=...)`` method
    that returns a ``RAGResponse`` with ``.chunks``.
    """
    report = EvalReport(
        dataset_name=dataset.name,
        n_queries=len(dataset.queries),
    )

    for eq in dataset.queries:
        response = query_engine.query(eq.query, top_k=top_k)
        # Collect chunk IDs (or source names as fallback)
        retrieved_ids = []
        for chunk in response.chunks:
            chunk_id = chunk.metadata.get("chunk_hash", "") or chunk.source
            retrieved_ids.append(chunk_id)

        # Use relevant_doc_ids if provided, else match by source name
        relevant = eq.relevant_doc_ids or eq.relevant_sources

        result = QueryEvalResult(
            query=eq.query,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
            recall_5=recall_at_k(retrieved_ids, relevant, 5),
            recall_10=recall_at_k(retrieved_ids, relevant, 10),
            precision_5=precision_at_k(retrieved_ids, relevant, 5),
            precision_10=precision_at_k(retrieved_ids, relevant, 10),
            mrr=reciprocal_rank(retrieved_ids, relevant),
            ndcg_5=ndcg_at_k(retrieved_ids, relevant, 5),
            ndcg_10=ndcg_at_k(retrieved_ids, relevant, 10),
            hit_rate_5=hit_rate_at_k(retrieved_ids, relevant, 5),
            hit_rate_10=hit_rate_at_k(retrieved_ids, relevant, 10),
        )
        report.results.append(result)

    # Aggregate
    n = len(report.results) or 1
    report.avg_recall_5 = sum(r.recall_5 for r in report.results) / n
    report.avg_recall_10 = sum(r.recall_10 for r in report.results) / n
    report.avg_precision_5 = sum(r.precision_5 for r in report.results) / n
    report.avg_precision_10 = sum(r.precision_10 for r in report.results) / n
    report.avg_mrr = sum(r.mrr for r in report.results) / n
    report.avg_ndcg_5 = sum(r.ndcg_5 for r in report.results) / n
    report.avg_ndcg_10 = sum(r.ndcg_10 for r in report.results) / n
    report.avg_hit_rate_5 = sum(r.hit_rate_5 for r in report.results) / n
    report.avg_hit_rate_10 = sum(r.hit_rate_10 for r in report.results) / n

    logger.info("Evaluation complete: %s", report.summary())
    return report


# ---------------------------------------------------------------------------
# LLM-as-Judge: Answer Faithfulness & Relevance Scoring
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are an impartial judge evaluating whether an AI answer is faithful to the provided context.

CONTEXT CHUNKS:
{context}

QUESTION:
{query}

AI ANSWER:
{answer}

Evaluate the answer on two dimensions. For each, provide a score from 1-5 and a one-sentence justification.

1. FAITHFULNESS (1-5): Is every claim in the answer supported by the context? \
   5 = fully grounded, 1 = mostly hallucinated.
2. RELEVANCE (1-5): Does the answer address the question? \
   5 = directly answers, 1 = completely off-topic.

Respond in EXACTLY this format (no other text):
FAITHFULNESS: <score>
FAITHFULNESS_REASON: <one sentence>
RELEVANCE: <score>
RELEVANCE_REASON: <one sentence>\
"""


@dataclass
class FaithfulnessResult:
    """Result from LLM-as-judge faithfulness evaluation."""
    query: str
    faithfulness_score: float  # 1-5
    faithfulness_reason: str
    relevance_score: float  # 1-5
    relevance_reason: str


def evaluate_faithfulness(
    query: str,
    answer: str,
    context: str,
    llm_backend=None,
) -> FaithfulnessResult:
    """
    Use an LLM as a judge to score answer faithfulness and relevance.

    Args:
        query: The user's question.
        answer: The RAG-generated answer.
        context: The context chunks that were provided to the LLM.
        llm_backend: An LLM with .generate(query, context) method.

    Returns:
        FaithfulnessResult with scores and justifications.
    """
    import re as _re

    prompt = _FAITHFULNESS_PROMPT.format(
        context=context[:3000],  # cap context for judge
        query=query,
        answer=answer[:2000],
    )

    if llm_backend is None:
        from rag_pipeline.llm_service import create_llm_backend
        from rag_pipeline.config import RAGConfig
        llm_backend = create_llm_backend(RAGConfig())

    raw = llm_backend.generate(prompt, "You are an evaluation judge. Be precise and objective.")

    # Parse scores
    faith_match = _re.search(r"FAITHFULNESS:\s*(\d)", raw)
    rel_match = _re.search(r"RELEVANCE:\s*(\d)", raw)
    faith_reason = _re.search(r"FAITHFULNESS_REASON:\s*(.+)", raw)
    rel_reason = _re.search(r"RELEVANCE_REASON:\s*(.+)", raw)

    return FaithfulnessResult(
        query=query,
        faithfulness_score=float(faith_match.group(1)) if faith_match else 0.0,
        faithfulness_reason=faith_reason.group(1).strip() if faith_reason else "Parse error",
        relevance_score=float(rel_match.group(1)) if rel_match else 0.0,
        relevance_reason=rel_reason.group(1).strip() if rel_reason else "Parse error",
    )


def run_faithfulness_evaluation(
    query_engine,
    queries: List[str],
    llm_judge=None,
) -> List[FaithfulnessResult]:
    """
    Run faithfulness evaluation on a list of queries.

    Executes each query through the RAG engine, then uses an LLM judge
    to score whether the answer is grounded in the retrieved context.

    Returns a list of FaithfulnessResult objects.
    """
    results: List[FaithfulnessResult] = []
    for query in queries:
        response = query_engine.query(query)
        # Rebuild context from chunks for the judge
        context_parts = []
        for i, chunk in enumerate(response.chunks, 1):
            context_parts.append(
                f"[Chunk {i}] Source: {chunk.source}, "
                f"Page: {chunk.metadata.get('page_number', '?')}\n"
                f"{chunk.text}"
            )
        context = "\n---\n".join(context_parts)

        result = evaluate_faithfulness(
            query=query,
            answer=response.answer,
            context=context,
            llm_backend=llm_judge,
        )
        results.append(result)
        logger.info(
            "Faithfulness eval: query='%.50s' faith=%.0f/5 rel=%.0f/5",
            query, result.faithfulness_score, result.relevance_score,
        )

    # Summary
    if results:
        avg_faith = sum(r.faithfulness_score for r in results) / len(results)
        avg_rel = sum(r.relevance_score for r in results) / len(results)
        logger.info(
            "Faithfulness evaluation complete: %d queries, "
            "avg_faithfulness=%.2f/5, avg_relevance=%.2f/5",
            len(results), avg_faith, avg_rel,
        )

    return results


# ---------------------------------------------------------------------------
# Golden Test Dataset — Financial RAG Evaluation Queries
# ---------------------------------------------------------------------------

def create_golden_eval_dataset() -> EvalDataset:
    """
    Create a starter golden evaluation dataset for financial RAG.

    These queries cover common financial strategy document topics and
    are designed to test retrieval quality across different query types:
    factual, procedural, comparative, and analytical.

    Users should customise ``relevant_doc_ids`` and ``relevant_sources``
    after ingesting their own documents.
    """
    ds = EvalDataset(name="centurion_golden_v1")

    # ---- Factual queries ----
    ds.add(EvalQuery(
        query="What is the RSI indicator and how is it calculated?",
        relevant_sources=[],  # populate after ingestion
        tags=["factual", "indicator"],
    ))
    ds.add(EvalQuery(
        query="What are the entry criteria for the momentum strategy?",
        relevant_sources=[],
        tags=["factual", "momentum"],
    ))
    ds.add(EvalQuery(
        query="What is the maximum drawdown limit for the portfolio?",
        relevant_sources=[],
        tags=["factual", "risk"],
    ))

    # ---- Procedural queries ----
    ds.add(EvalQuery(
        query="How do I set up a stop-loss order for the mean reversion strategy?",
        relevant_sources=[],
        tags=["procedural", "risk"],
    ))
    ds.add(EvalQuery(
        query="What are the steps for backtesting a new trading strategy?",
        relevant_sources=[],
        tags=["procedural", "backtest"],
    ))

    # ---- Comparative queries ----
    ds.add(EvalQuery(
        query="How does the momentum strategy compare to the mean reversion strategy in terms of risk?",
        relevant_sources=[],
        tags=["comparative", "strategy"],
    ))
    ds.add(EvalQuery(
        query="What are the differences between RSI and MACD indicators?",
        relevant_sources=[],
        tags=["comparative", "indicator"],
    ))

    # ---- Analytical queries ----
    ds.add(EvalQuery(
        query="What factors drive the performance of the statistical arbitrage strategy?",
        relevant_sources=[],
        tags=["analytical", "strategy"],
    ))
    ds.add(EvalQuery(
        query="How does market volatility affect the pattern recognition strategy?",
        relevant_sources=[],
        tags=["analytical", "volatility"],
    ))
    ds.add(EvalQuery(
        query="What risk management techniques are recommended for derivatives trading?",
        relevant_sources=[],
        tags=["analytical", "derivatives"],
    ))

    # ---- Summary queries ----
    ds.add(EvalQuery(
        query="Summarise the key trading strategies available in the portfolio.",
        relevant_sources=[],
        tags=["summary", "portfolio"],
    ))
    ds.add(EvalQuery(
        query="Give an overview of the position sizing methodology.",
        relevant_sources=[],
        tags=["summary", "position"],
    ))

    logger.info(
        "Created golden eval dataset '%s' with %d queries",
        ds.name, len(ds.queries),
    )
    return ds
