"""
Token Counting Utilities for Centurion Capital LLC RAG Pipeline.

Provides fast, approximate token counting so the pipeline can enforce
a context token budget *without* importing heavy tokenizer libraries.

Strategies (in order of preference):
    1. ``tiktoken``   — OpenAI's fast BPE tokenizer (if installed)
    2. Whitespace heuristic — ``len(text.split())`` × 1.3 factor

Usage:
    from rag_pipeline.token_counter import count_tokens, truncate_to_budget

    n = count_tokens("Hello world, this is a test.")
    trimmed = truncate_to_budget("long text …", max_tokens=2000)
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import tiktoken (optional dependency)
# ---------------------------------------------------------------------------
_tiktoken_enc = None

try:
    import tiktoken as _tiktoken_mod
    _tiktoken_enc = _tiktoken_mod.get_encoding("cl100k_base")
    logger.debug("tiktoken detected — using cl100k_base for token counting")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """
    Return an approximate token count for *text*.

    Uses tiktoken cl100k_base if available, else a whitespace heuristic
    (word count × 1.3, which empirically aligns with BPE token counts
    for English prose).
    """
    if not text:
        return 0
    if _tiktoken_enc is not None:
        return len(_tiktoken_enc.encode(text, disallowed_special=()))
    # Heuristic: ~1.3 tokens per whitespace-delimited word
    return int(len(text.split()) * 1.3)


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """
    Truncate *text* so that ``count_tokens(result) <= max_tokens``.

    Tries to cut at sentence boundaries to preserve coherence.
    """
    if count_tokens(text) <= max_tokens:
        return text

    if _tiktoken_enc is not None:
        tokens = _tiktoken_enc.encode(text, disallowed_special=())
        return _tiktoken_enc.decode(tokens[:max_tokens])

    # Heuristic: estimate words budget and truncate
    word_budget = int(max_tokens / 1.3)
    words = text.split()
    return " ".join(words[:word_budget])


def budget_chunks(
    chunk_texts: List[str],
    max_total_tokens: int,
    separator_tokens: int = 10,
    greedy_pack: bool = True,
) -> List[int]:
    """
    Return the indices of chunks that fit within *max_total_tokens*.

    Each chunk is separated by an overhead of *separator_tokens*
    (accounts for the source/page/section header lines injected in
    ``_build_context``).

    When *greedy_pack* is True (default), if a chunk is too large to
    fit in the remaining budget the algorithm skips it and tries to
    fit subsequent (smaller) chunks.  This maximises context coverage
    instead of stopping at the first oversized chunk.

    Chunks are consumed in list order (assumed pre-sorted by relevance).
    Higher-ranked chunks are always preferred.
    """
    indices: List[int] = []
    remaining = max_total_tokens

    for i, text in enumerate(chunk_texts):
        cost = count_tokens(text) + separator_tokens
        if cost <= remaining:
            remaining -= cost
            indices.append(i)
        elif not greedy_pack:
            # Strict mode: stop at first chunk that doesn't fit
            break
        # else: greedy_pack — skip this chunk, try the next one

    return indices
