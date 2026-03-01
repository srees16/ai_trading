"""
LLM-powered Query Rewriter for Centurion Capital LLC RAG Pipeline.

Converts short or ambiguous user queries into semantically richer
search queries before embedding.  Supports multi-query expansion
(generates several reformulations, retrieves for each, then merges).

Usage:
    from rag_pipeline.query_rewriter import QueryRewriter
    rewriter = QueryRewriter(config)
    expanded = rewriter.rewrite("RSI strategy")
    # → ["What are the RSI-based trading strategy rules and entry criteria?",
    #    "How is the Relative Strength Index used for momentum signals?", ...]
"""

import logging
import re
from typing import List, Optional, Protocol

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 4 — Deterministic structural query expansion
# ---------------------------------------------------------------------------
# Rules are {pattern: [phrases to append]}.  The original query text is
# always preserved.  This runs BEFORE LLM-based rewriting and is zero-
# latency.

_STRUCTURAL_EXPANSIONS: List[tuple] = [
    # Chapter references
    (re.compile(r"\bfirst\s+chapter\b", re.I),
     ["Chapter 1", "opening chapter", "introduction section"]),
    (re.compile(r"\bsecond\s+chapter\b", re.I),
     ["Chapter 2"]),
    (re.compile(r"\bthird\s+chapter\b", re.I),
     ["Chapter 3"]),
    (re.compile(r"\bfourth\s+chapter\b", re.I),
     ["Chapter 4"]),
    (re.compile(r"\bfifth\s+chapter\b", re.I),
     ["Chapter 5"]),
    (re.compile(r"\bchapter\s+(\d+)\b", re.I),
     []),  # no expansion needed for explicit chapter refs

    # Thematic / abstract concepts
    (re.compile(r"\bcentral\s+theme\b", re.I),
     ["main idea", "primary argument", "core thesis",
      "key concept", "main topic"]),
    (re.compile(r"\bmain\s+(?:idea|point|argument)\b", re.I),
     ["central theme", "core thesis", "primary concept"]),

    # Structure references
    (re.compile(r"\bintroduction\b", re.I),
     ["preface", "opening section", "Chapter 1"]),
    (re.compile(r"\bconclusion\b", re.I),
     ["final chapter", "summary", "closing remarks"]),
    (re.compile(r"\bsummary\b", re.I),
     ["overview", "key takeaways", "recap"]),
]


def expand_query(query: str) -> str:
    """Deterministic structural query expansion (Step 4).

    Appends synonyms and structural references for book-style queries
    so that vector search has broader recall.  Returns the expanded
    query with the original text preserved at the front.

    Rules:
        * "first chapter"  → appends "Chapter 1", "opening chapter",
          "introduction section"
        * "central theme"  → appends "main idea", "primary argument",
          "core thesis"
        * etc.

    If no rule matches, returns the original query unchanged.
    """
    expansions: List[str] = []
    for pattern, phrases in _STRUCTURAL_EXPANSIONS:
        if pattern.search(query):
            expansions.extend(phrases)

    if not expansions:
        return query

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for phrase in expansions:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            unique.append(phrase)

    expanded = query + " " + " ".join(unique)
    logger.info("expand_query: '%s' → '%s'", query, expanded)
    return expanded


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = """\
You are a query-expansion assistant for a financial document retrieval system.

Given a user query, generate {n} different reformulations that:
1. Preserve the original intent
2. Use diverse synonyms and phrasings
3. Add relevant financial/trading context where appropriate
4. Are self-contained (each query should make sense on its own)
5. Cover different angles: definitional, procedural, comparative, and data-oriented
6. When the query asks for code, algorithms, methods, functions, or implementations, \
include at least one variant that uses textbook labelling conventions such as \
"Snippet", "Code Listing", "Algorithm", or "Listing" followed by the topic title \
— for example, if the user asks about PCA weights, one variant should be \
"Snippet PCA Weights from a Risk Distribution"
7. When a specific technique is mentioned (e.g., PCA, RSI, GARCH, Monte Carlo), \
expand the full name in at least one variant and add the likely book or textbook \
name as additional context

Return ONLY the reformulated queries, one per line, numbered 1-{n}.
Do NOT include any other text or explanation.\
"""

_REWRITE_USER = """\
Original query: {query}

Generate {n} reformulated search queries:\
"""

_HYDE_SYSTEM = """\
You are a financial document author. Given a question, write a short paragraph \
(3-5 sentences) that would appear in a financial strategy document and would \
be the ideal answer to the question. Do NOT add disclaimers or caveats. \
Write as if this text exists in an actual document. Be specific and factual.\
"""

_HYDE_USER = """\
Question: {query}

Write a short paragraph that would answer this question in a strategy document:\
"""

_CLASSIFY_SYSTEM = """\
Classify the following query into exactly ONE of these categories:
- FACTUAL: asking for specific facts, numbers, dates, definitions
- PROCEDURAL: asking about steps, processes, rules, criteria
- COMPARATIVE: asking to compare, contrast, or evaluate options
- SUMMARY: asking for an overview or summary of a topic
- ANALYTICAL: asking for analysis, reasoning, or interpretation

Respond with ONLY the category name, nothing else.\
"""


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

class QueryRewriter:
    """
    Expand a short user query into multiple semantically rich variants.

    The rewriter uses the configured LLM to generate reformulations,
    then returns them as a list so the retrieval stage can search for
    each variant and merge results.

    Supports three expansion strategies:
        1. **Multi-query**: LLM-generated paraphrases from diverse angles
        2. **HyDE**: Hypothetical Document Embedding — generates a
           hypothetical answer passage, embeds it, and uses that for
           retrieval (catches chunks the original query phrasing misses)
        3. **Query classification**: Detects query type (factual,
           procedural, comparative, etc.) for downstream use
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm_backend=None,
    ) -> None:
        self._config = config or RAGConfig()
        self._n_expansions = int(
            getattr(self._config, "query_rewrite_n", 3)
        )
        self._hyde_enabled: bool = getattr(
            self._config, "hyde_enabled", True
        )
        # Lazily create LLM backend if not provided
        self._llm = llm_backend

    @property
    def llm(self):
        if self._llm is None:
            from rag_pipeline.llm_service import create_llm_backend
            self._llm = create_llm_backend(self._config)
        return self._llm

    def rewrite(
        self,
        query: str,
        n: Optional[int] = None,
    ) -> List[str]:
        """
        Generate *n* reformulated versions of *query*.

        Always includes the original query as the first element so
        that the original phrasing is never lost.

        When HyDE is enabled, a hypothetical answer passage is added
        as an additional "query" variant for embedding-based retrieval.

        Falls back to ``[query]`` on any LLM error to ensure the
        pipeline never blocks.
        """
        n = n or self._n_expansions
        if n <= 0:
            return [query]

        system_msg = _REWRITE_SYSTEM.format(n=n)
        user_msg = _REWRITE_USER.format(query=query, n=n)

        try:
            raw = self.llm.generate(user_msg, "", system_prompt=system_msg)
            variants = self._parse_variants(raw, query)
            logger.info(
                "Query rewritten: '%s' → %d variants", query, len(variants)
            )
        except Exception as e:
            logger.warning("Query rewrite failed (%s); using original query", e)
            variants = [query]

        # HyDE: generate a hypothetical answer and add it as a search query
        if self._hyde_enabled:
            try:
                hyde_passage = self._generate_hyde(query)
                if hyde_passage:
                    variants.append(hyde_passage)
                    logger.info("HyDE passage generated (%d chars)", len(hyde_passage))
            except Exception as e:
                logger.debug("HyDE generation failed (%s); skipping", e)

        return variants

    def classify_query(self, query: str) -> str:
        """
        Classify *query* into a category: FACTUAL, PROCEDURAL,
        COMPARATIVE, SUMMARY, or ANALYTICAL.

        Returns the category string. Falls back to "FACTUAL" on error.
        """
        try:
            raw = self.llm.generate(query, "", system_prompt=_CLASSIFY_SYSTEM)
            category = raw.strip().upper().split()[0] if raw.strip() else "FACTUAL"
            valid = {"FACTUAL", "PROCEDURAL", "COMPARATIVE", "SUMMARY", "ANALYTICAL"}
            return category if category in valid else "FACTUAL"
        except Exception:
            return "FACTUAL"

    def _generate_hyde(self, query: str) -> Optional[str]:
        """
        Generate a Hypothetical Document Embedding passage.

        Asks the LLM to write a short paragraph that *would* appear
        in a strategy document answering the query. This passage is
        then embedded and used for vector search, often retrieving
        chunks that the original short query misses.
        """
        user_msg = _HYDE_USER.format(query=query)
        raw = self.llm.generate(user_msg, "", system_prompt=_HYDE_SYSTEM)
        passage = raw.strip() if raw else None
        # Sanity: skip if the LLM returned something too short or too long
        if passage and 20 < len(passage) < 2000:
            return passage
        return None

    @staticmethod
    def _parse_variants(raw: str, original: str) -> List[str]:
        """Parse numbered lines from LLM output and prepend original."""
        lines = raw.strip().splitlines()
        variants = [original]  # always keep original first
        seen = {original.lower().strip()}

        for line in lines:
            # Strip numbering: "1. ...", "1) ...", "- ..."
            cleaned = re.sub(r"^\s*[\d]+[.)]\s*", "", line).strip()
            cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in seen:
                variants.append(cleaned)
                seen.add(key)

        return variants
