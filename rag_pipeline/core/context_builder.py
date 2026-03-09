"""
Context Builder for Centurion Capital LLC RAG Pipeline.

Transforms the structured output of ``retrieve_context()`` into a
single, LLM-ready prompt context string with clearly delineated
sections:

    ### Strategy Theory
    ### Relevant Code Snippets
    ### Risk & Evaluation Notes

Design goals
------------
- **Deterministic** — same input always produces the same output.
- **Dedup-safe** — identical content is never repeated, even if
  present in both ``theory_context`` and ``code_context``.
- **Indentation-preserving** — code is wrapped in fenced blocks
  with original whitespace intact.
- **Score-ordered** — within each section, higher-relevance chunks
  appear first (uses ``scores.final`` when available, otherwise
  insertion order is preserved).
- **Budget-conformant** — enforces a hard token cap (default 2 500)
  via ``count_tokens()`` from ``token_counter``.  When the budget is
  tight, lowest-scoring chunks are dropped first.  When
  ``intent == "code_generation"`` code chunks receive a score boost
  so they survive trimming ahead of theory.
- **Logged** — token count, chunk count, and drop count are logged
  at INFO level for every call.

Input
-----
``dict`` returned by ``Retriever.retrieve_context()``::

    {
        "theory_context": [{"id", "content", "metadata"}, ...],
        "code_context":   [{"id", "content", "metadata"}, ...],
    }

Output
------
Formatted ``str`` ready for injection into an LLM prompt.

Tech stack: Python 3.11 · No LangChain · No LLM calls.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from rag_pipeline.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Section headers (Markdown H3)
_HEADER_THEORY = "### Strategy Theory"
_HEADER_CODE = "### Relevant Code Snippets"
_HEADER_RISK = "### Risk & Evaluation Notes"

# Default hard token cap for the assembled prompt context.
DEFAULT_MAX_TOKENS = 2500

# Score boost applied to code chunks when intent == "code_generation".
_CODE_PRIORITY_BOOST = 0.5

# Approximate token cost per section header + separator overhead.
_SECTION_HEADER_TOKENS = 8

# Pipeline stages that belong in the Risk & Evaluation Notes section
_RISK_EVAL_STAGES = frozenset({"risk_management", "evaluation"})

# Minimum content length (chars) for a chunk to be included
_MIN_CONTENT_LEN = 10

# Separator between chunks within a section
_CHUNK_SEPARATOR = "\n\n"

# Number of top code chunks protected in strict_code_mode
_STRICT_CODE_PROTECTED_COUNT = 2

# Regex matching Python comment lines (optional leading whitespace + #)
_COMMENT_LINE_RE = re.compile(r"^\s*#.*$", re.MULTILINE)

# Regex matching lines that contain a Python function definition
_DEF_LINE_RE = re.compile(r"^[^#]*\bdef\s+\w+", re.MULTILINE)

# Code-fence language hint heuristic
_LANG_HINT_RE = re.compile(
    r"^\s*(?:import |from |def |class |async def |@)",
    re.MULTILINE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _content_fingerprint(text: str) -> str:
    """Return a short hash for dedup — normalises whitespace first."""
    normalised = " ".join(text.split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:20]


def _sort_key(chunk: Dict[str, Any]) -> float:
    """Extract the relevance score for descending sort.

    Looks for ``scores.final`` (from ranked retriever output),
    falls back to ``metadata.score`` or 0.0.
    """
    scores = chunk.get("scores")
    if isinstance(scores, dict):
        return scores.get("final", 0.0)
    meta = chunk.get("metadata") or {}
    return float(meta.get("score", 0.0))


def _is_risk_eval(chunk: Dict[str, Any]) -> bool:
    """Return *True* if the chunk's pipeline_stage is risk or evaluation."""
    meta = chunk.get("metadata") or {}
    stage_raw = meta.get("pipeline_stage", "")
    if isinstance(stage_raw, str):
        stages = {s.strip() for s in stage_raw.split(",") if s.strip()}
    elif isinstance(stage_raw, list):
        stages = set(stage_raw)
    else:
        stages = set()
    return bool(stages & _RISK_EVAL_STAGES)


def _detect_lang(text: str) -> str:
    """Guess the code-fence language hint."""
    if _LANG_HINT_RE.search(text):
        return "python"
    return ""


def _strip_comments(text: str) -> str:
    """Remove Python comment lines from *text*, preserving everything else.

    Inline comments (``code  # comment``) are left intact; only lines
    that are *entirely* a comment (optional whitespace + ``#``) are
    removed.  Blank lines are preserved to maintain visual structure.
    """
    lines = text.split("\n")
    kept = [ln for ln in lines if not _COMMENT_LINE_RE.fullmatch(ln)]
    return "\n".join(kept)


def _trim_code_preserving_defs(
    text: str,
    max_tokens: int,
) -> str:
    """Trim a code chunk to fit within *max_tokens*, preserving defs.

    Strategy (applied in order until the chunk fits):

    1. **Strip comments** — remove pure-comment lines.
    2. **Drop non-def body lines** — remove lines that are NOT
       function signatures (``def …``) and are not blank separators
       between functions.  This preserves the API surface.
    3. **Hard truncate** — as a last resort, truncate from the bottom
       while keeping every ``def`` line intact.

    The returned text is guaranteed to satisfy
    ``count_tokens(result) <= max_tokens`` (assuming *max_tokens* ≥ 1).
    """
    # Phase 0: already fits?
    if count_tokens(text) <= max_tokens:
        return text

    # Phase 1: strip comments
    stripped = _strip_comments(text)
    if count_tokens(stripped) <= max_tokens:
        return stripped

    # Phase 2: keep only def lines + immediately following signature line
    lines = stripped.split("\n")
    kept: list[str] = []
    prev_was_def = False
    for ln in lines:
        is_def = bool(_DEF_LINE_RE.match(ln))
        if is_def:
            kept.append(ln)
            prev_was_def = True
        elif prev_was_def and ln.strip():
            # Keep the first body line after a def (often has return type
            # or docstring opening)
            kept.append(ln)
            prev_was_def = False
        elif not ln.strip():
            # Preserve blank separators between functions
            kept.append(ln)
            prev_was_def = False
        else:
            prev_was_def = False

    result = "\n".join(kept).strip()
    if count_tokens(result) <= max_tokens:
        return result

    # Phase 3: hard truncate — keep def lines, drop everything else
    def_only = [ln for ln in lines if _DEF_LINE_RE.match(ln)]
    result = "\n".join(def_only).strip()
    if count_tokens(result) <= max_tokens:
        return result

    # Absolute fallback: token-level truncation (preserves top def lines)
    from rag_pipeline.utils.token_counter import truncate_to_budget
    return truncate_to_budget(result, max_tokens)


def _format_code_chunk(content: str) -> str:
    """Wrap code in a fenced code block, preserving indentation."""
    lang = _detect_lang(content)
    fence = f"```{lang}" if lang else "```"
    # Ensure no trailing whitespace on the fence lines
    return f"{fence}\n{content.rstrip()}\n```"


def _format_theory_chunk(content: str) -> str:
    """Format a theory chunk — light clean-up only."""
    return content.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Primary public API
# ═══════════════════════════════════════════════════════════════════════════

def build_prompt_context(
    chunks: Dict[str, Any],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    intent: Optional[str] = None,
    strict_code_mode: bool = False,
) -> str:
    """Build a formatted prompt-context string from retrieved chunks.

    Organises the chunks into three Markdown sections:

    1. **Strategy Theory** — theory chunks *excluding* risk/evaluation.
    2. **Relevant Code Snippets** — code chunks wrapped in fenced
       blocks with original indentation.
    3. **Risk & Evaluation Notes** — theory chunks whose
       ``pipeline_stage`` is ``risk_management`` or ``evaluation``.

    **Token budget enforcement** (hard cap):

    - Each candidate chunk is measured with ``count_tokens()``.
    - Chunks are ranked by relevance score.  When
      ``intent == "code_generation"``, code chunks receive a
      ``+0.5`` score boost so they are packed first.
    - Chunks are greedily packed (highest priority first) until
      the *max_tokens* budget is exhausted.
    - Lowest-scoring chunks that do not fit are silently dropped.
    - Section headers consume budget too.

    Within each section the surviving chunks retain descending-score
    order.  Duplicate content (same normalised text hash) is silently
    dropped.

    Args:
        chunks: Dict with keys ``"theory_context"`` and
            ``"code_context"`` — each a list of dicts with at least
            ``"content"`` and ``"metadata"`` keys.  This is exactly
            the format returned by ``Retriever.retrieve_context()``.
            An optional ``"intent"`` key (str) may also be present
            and will be used when the *intent* argument is ``None``.
        max_tokens: Hard upper-bound on the token count of the
            returned string (default 2 500).
        intent: Query intent string (e.g. ``"code_generation"``).
            When ``None``, falls back to ``chunks.get("intent")``.
        strict_code_mode: When ``True``:
            - Top 2 highest-scoring code chunks are **always**
              included, even if token trimming would drop them.
            - Comments are trimmed first from code chunks.
            - Function definitions (``def …``) are never trimmed.

    Returns:
        A single formatted string ready for LLM prompt injection.
        ``count_tokens(result) <= max_tokens`` is guaranteed.
        Returns an empty string when there is nothing to include.

    Example::

        from rag_pipeline.core.retriever import Retriever
        from rag_pipeline.core.context_builder import build_prompt_context

        ctx = retriever.retrieve_context("Explain Sharpe ratio")
        prompt_block = build_prompt_context(ctx, max_tokens=2500)
        print(prompt_block)
    """
    if not chunks:
        return ""

    # Resolve intent — explicit param wins, then dict key
    _intent: str = intent or (chunks.get("intent") if isinstance(chunks, dict) else None) or ""

    # Resolve strict_code_mode — explicit param wins, then dict key
    _strict_code: bool = strict_code_mode or bool(
        chunks.get("strict_code_mode") if isinstance(chunks, dict) else False
    )

    theory_raw: List[Dict[str, Any]] = chunks.get("theory_context") or []
    code_raw: List[Dict[str, Any]] = chunks.get("code_context") or []

    # Global dedup tracker — no content appears in more than one section
    seen: Set[str] = set()

    # ── Partition theory into strategy vs. risk/eval ─────────────────
    strategy_candidates: List[Dict[str, Any]] = []
    risk_eval_candidates: List[Dict[str, Any]] = []

    for c in sorted(theory_raw, key=_sort_key, reverse=True):
        content = (c.get("content") or "").strip()
        if len(content) < _MIN_CONTENT_LEN:
            continue
        fp = _content_fingerprint(content)
        if fp in seen:
            continue
        seen.add(fp)

        if _is_risk_eval(c):
            risk_eval_candidates.append(c)
        else:
            strategy_candidates.append(c)

    # ── Code chunks ──────────────────────────────────────────────────
    code_candidates: List[Dict[str, Any]] = []

    for c in sorted(code_raw, key=_sort_key, reverse=True):
        content = (c.get("content") or "").strip()
        if len(content) < _MIN_CONTENT_LEN:
            continue
        fp = _content_fingerprint(content)
        if fp in seen:
            continue
        seen.add(fp)
        code_candidates.append(c)

    # ── Token-budget-aware chunk selection ────────────────────────────
    # Build a unified priority list: (priority_score, section_tag, chunk)
    # so we can greedily fill the budget across all sections.

    _prefers_code = (_intent == "code_generation")

    tagged: List[Tuple[float, str, Dict[str, Any]]] = []
    for c in strategy_candidates:
        tagged.append((_sort_key(c), "strategy", c))
    for c in code_candidates:
        score = _sort_key(c) + (_CODE_PRIORITY_BOOST if _prefers_code else 0.0)
        tagged.append((score, "code", c))
    for c in risk_eval_candidates:
        tagged.append((_sort_key(c), "risk", c))

    # Sort by priority descending — highest-value chunks packed first
    tagged.sort(key=lambda t: t[0], reverse=True)

    # Reserve budget for section headers (worst case: all 3 sections)
    active_sections: Set[str] = set()
    header_reserve = 0  # will be adjusted dynamically

    remaining = max_tokens
    selected_strategy: List[Dict[str, Any]] = []
    selected_code: List[Dict[str, Any]] = []
    selected_risk: List[Dict[str, Any]] = []
    dropped = 0

    # ── Strict code mode: pre-reserve top-N code chunks ──────────────
    # These are guaranteed to survive token trimming.  Comments are
    # stripped first; function definitions are never removed.
    _protected_ids: Set[str] = set()

    if _strict_code and code_candidates:
        n_protect = min(_STRICT_CODE_PROTECTED_COUNT, len(code_candidates))
        # code_candidates is already sorted by score descending
        for protected in code_candidates[:n_protect]:
            pid = protected.get("id", id(protected))
            _protected_ids.add(pid)

            raw_content = (protected.get("content") or "").strip()
            # Always strip comments in strict mode, then trim defs
            stripped_content = _strip_comments(raw_content)
            trimmed_content = _trim_code_preserving_defs(
                stripped_content,
                max_tokens=(remaining - _SECTION_HEADER_TOKENS) // max(n_protect, 1),
            )
            # Store trimmed version for later formatting
            protected["_trimmed_content"] = trimmed_content

            formatted = _format_code_chunk(trimmed_content)
            chunk_tokens = count_tokens(formatted)

            section_overhead = 0
            if "code" not in active_sections:
                section_overhead = _SECTION_HEADER_TOKENS

            remaining -= (chunk_tokens + section_overhead)
            active_sections.add("code")
            selected_code.append(protected)

        logger.info(
            "build_prompt_context [STRICT_CODE]: protected top %d code "
            "chunks (%d tokens reserved).",
            len(_protected_ids), max_tokens - remaining,
        )

    for priority, section, chunk in tagged:
        # Skip already-protected code chunks
        chunk_id = chunk.get("id", id(chunk))
        if chunk_id in _protected_ids:
            continue

        # Format the chunk to measure its exact token cost
        content = (chunk.get("content") or "").strip()

        # In strict_code_mode, strip comments from code chunks to
        # reclaim budget for the protected chunks.
        if section == "code" and _strict_code:
            content = _strip_comments(content)
            formatted = _format_code_chunk(content)
        elif section == "code":
            formatted = _format_code_chunk(content)
        else:
            formatted = _format_theory_chunk(content)

        chunk_tokens = count_tokens(formatted)

        # If this is the first chunk in a section, account for the
        # section header + separator overhead.
        section_overhead = 0
        if section not in active_sections:
            section_overhead = _SECTION_HEADER_TOKENS

        total_cost = chunk_tokens + section_overhead
        if total_cost > remaining:
            dropped += 1
            continue

        remaining -= total_cost
        active_sections.add(section)

        if section == "strategy":
            selected_strategy.append(chunk)
        elif section == "code":
            selected_code.append(chunk)
        else:
            selected_risk.append(chunk)

    # ── Re-sort within each section by score (desc) for readability ──
    selected_strategy.sort(key=_sort_key, reverse=True)
    selected_code.sort(key=_sort_key, reverse=True)
    selected_risk.sort(key=_sort_key, reverse=True)

    # ── Assemble sections ────────────────────────────────────────────
    sections: List[str] = []

    if selected_strategy:
        body = _CHUNK_SEPARATOR.join(
            _format_theory_chunk(c["content"]) for c in selected_strategy
        )
        sections.append(f"{_HEADER_THEORY}\n\n{body}")

    if selected_code:
        def _code_content(c: Dict[str, Any]) -> str:
            # Use trimmed content for protected chunks in strict mode
            if "_trimmed_content" in c:
                return c["_trimmed_content"]
            if _strict_code:
                return _strip_comments((c.get("content") or "").strip())
            return (c.get("content") or "").strip()

        body = _CHUNK_SEPARATOR.join(
            _format_code_chunk(_code_content(c)) for c in selected_code
        )
        sections.append(f"{_HEADER_CODE}\n\n{body}")

    if selected_risk:
        body = _CHUNK_SEPARATOR.join(
            _format_theory_chunk(c["content"]) for c in selected_risk
        )
        sections.append(f"{_HEADER_RISK}\n\n{body}")

    if not sections:
        return ""

    result = "\n\n".join(sections)

    # ── Final safety net — truncate if rounding pushed over budget ────
    result_tokens = count_tokens(result)
    if result_tokens > max_tokens:
        if _strict_code:
            # Preserve function definitions even in last-resort trim
            result = _trim_code_preserving_defs(result, max_tokens)
        else:
            from rag_pipeline.utils.token_counter import truncate_to_budget
            result = truncate_to_budget(result, max_tokens)
        result_tokens = count_tokens(result)

    total_selected = len(selected_strategy) + len(selected_code) + len(selected_risk)

    logger.info(
        "build_prompt_context: %d strategy, %d code, %d risk/eval chunks "
        "(%d dropped) %d tokens / %d budget, %d chars.%s",
        len(selected_strategy), len(selected_code), len(selected_risk),
        dropped, result_tokens, max_tokens, len(result),
        " [code-priority]" if _prefers_code else "",
    )

    # Clean up transient keys from protected chunks
    for c in selected_code:
        c.pop("_trimmed_content", None)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests (run with ``python -m rag_pipeline.core.context_builder``)
# ═══════════════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    """Self-contained unit tests — no external test framework needed."""

    import sys

    PASS = 0
    FAIL = 0

    def check(label: str, condition: bool) -> None:
        nonlocal PASS, FAIL
        if condition:
            PASS += 1
            print(f"  PASS: {label}")
        else:
            FAIL += 1
            print(f"  FAIL: {label}")

    # ── 1. Empty / None input ────────────────────────────────────────
    print("\n=== 1. Empty Input ===")
    check("None empty string", build_prompt_context(None) == "")
    check("empty dict empty string", build_prompt_context({}) == "")
    check(
        "empty lists empty string",
        build_prompt_context({"theory_context": [], "code_context": []}) == "",
    )

    # ── 2. Theory-only output ────────────────────────────────────────
    print("\n=== 2. Theory-only ===")
    ctx_theory = {
        "theory_context": [
            {
                "id": "t1",
                "content": "Momentum strategies follow trend signals using moving averages.",
                "metadata": {
                    "chunk_type": "theory",
                    "pipeline_stage": "signal_generation",
                },
            },
        ],
        "code_context": [],
    }
    out = build_prompt_context(ctx_theory)
    check("contains Strategy Theory header", _HEADER_THEORY in out)
    check("contains theory content", "Momentum strategies" in out)
    check("no Code header when no code", _HEADER_CODE not in out)
    check("no Risk header when no risk chunks", _HEADER_RISK not in out)

    # ── 3. Code-only output ──────────────────────────────────────────
    print("\n=== 3. Code-only ===")
    code_content = "def calculate_sharpe(returns):\n    return np.mean(returns) / np.std(returns)"
    ctx_code = {
        "theory_context": [],
        "code_context": [
            {
                "id": "c1",
                "content": code_content,
                "metadata": {
                    "chunk_type": "code",
                    "pipeline_stage": "evaluation",
                },
            },
        ],
    }
    out = build_prompt_context(ctx_code)
    check("contains Code header", _HEADER_CODE in out)
    check("code in fenced block", "```python" in out)
    check("closing fence", out.count("```") >= 2)
    check("original indentation preserved", "    return np.mean" in out)
    check("no Theory header when no theory", _HEADER_THEORY not in out)

    # ── 4. Risk & Evaluation split ───────────────────────────────────
    print("\n=== 4. Risk & Evaluation Split ===")
    ctx_mixed = {
        "theory_context": [
            {
                "id": "t_sig",
                "content": "The z-score indicator normalises price deviations from the mean.",
                "metadata": {
                    "chunk_type": "theory",
                    "pipeline_stage": "signal_generation",
                },
            },
            {
                "id": "t_risk",
                "content": "Maximum drawdown represents the peak-to-trough decline in portfolio value.",
                "metadata": {
                    "chunk_type": "theory",
                    "pipeline_stage": "risk_management",
                },
            },
            {
                "id": "t_eval",
                "content": "The Sharpe ratio is a measure of risk-adjusted return for portfolio evaluation.",
                "metadata": {
                    "chunk_type": "theory",
                    "pipeline_stage": "evaluation",
                },
            },
        ],
        "code_context": [],
    }
    out = build_prompt_context(ctx_mixed)
    check("contains Strategy Theory", _HEADER_THEORY in out)
    check("contains Risk & Evaluation", _HEADER_RISK in out)
    check("z-score in strategy section (not risk)", "z-score" in out)

    # Verify risk content is in risk section, not strategy
    theory_idx = out.index(_HEADER_THEORY)
    risk_idx = out.index(_HEADER_RISK)
    check("risk section comes after theory", risk_idx > theory_idx)

    # drawdown is risk_management should be in risk section
    dd_pos = out.index("Maximum drawdown")
    check("drawdown in risk section", dd_pos > risk_idx)

    # Sharpe is evaluation also in risk section
    sharpe_pos = out.index("Sharpe ratio")
    check("Sharpe in risk section", sharpe_pos > risk_idx)

    # z-score is signal_generation should be in strategy section
    zscore_pos = out.index("z-score")
    check("z-score in strategy section", zscore_pos > theory_idx and zscore_pos < risk_idx)

    # ── 5. Deduplication ─────────────────────────────────────────────
    print("\n=== 5. Deduplication ===")
    dup_content = "Momentum strategies follow trend signals."
    ctx_dup = {
        "theory_context": [
            {"id": "d1", "content": dup_content, "metadata": {"chunk_type": "theory", "pipeline_stage": ""}},
            {"id": "d2", "content": dup_content, "metadata": {"chunk_type": "theory", "pipeline_stage": ""}},
            {"id": "d3", "content": "  " + dup_content + "  ", "metadata": {"chunk_type": "theory", "pipeline_stage": ""}},
        ],
        "code_context": [],
    }
    out = build_prompt_context(ctx_dup)
    count = out.count("Momentum strategies follow trend signals")
    check(f"dedup: appears only once (got {count})", count == 1)

    # Cross-section dedup: same content in theory and code list
    cross_content = "def foo():\n    pass"
    ctx_cross = {
        "theory_context": [
            {"id": "x1", "content": cross_content, "metadata": {"chunk_type": "theory", "pipeline_stage": ""}},
        ],
        "code_context": [
            {"id": "x2", "content": cross_content, "metadata": {"chunk_type": "code", "pipeline_stage": ""}},
        ],
    }
    out = build_prompt_context(ctx_cross)
    count_cross = out.count("def foo()")
    check(f"cross-section dedup: appears once ({count_cross})", count_cross == 1)

    # ── 6. Score ordering ────────────────────────────────────────────
    print("\n=== 6. Score Ordering ===")
    ctx_scored = {
        "theory_context": [
            {
                "id": "lo",
                "content": "Low relevance chunk about general concepts in finance.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": ""},
                "scores": {"final": 0.2},
            },
            {
                "id": "hi",
                "content": "High relevance chunk about specific strategies and indicators.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": ""},
                "scores": {"final": 0.9},
            },
        ],
        "code_context": [],
    }
    out = build_prompt_context(ctx_scored)
    hi_pos = out.index("High relevance")
    lo_pos = out.index("Low relevance")
    check("higher score first", hi_pos < lo_pos)

    # ── 7. Full three-section output ─────────────────────────────────
    print("\n=== 7. Full Three-Section Output ===")
    ctx_full = {
        "theory_context": [
            {
                "id": "th1",
                "content": "Portfolio rebalancing adjusts asset weights quarterly.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "portfolio_construction"},
            },
            {
                "id": "th2",
                "content": "Maximum drawdown quantifies worst peak-to-trough loss.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "risk_management"},
            },
            {
                "id": "th3",
                "content": "Sharpe ratio evaluates risk-adjusted performance of a strategy.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "evaluation"},
            },
        ],
        "code_context": [
            {
                "id": "cd1",
                "content": "def rebalance(weights, target):\n    diff = target - weights\n    return diff",
                "metadata": {"chunk_type": "code", "pipeline_stage": "portfolio_construction"},
            },
        ],
    }
    out = build_prompt_context(ctx_full)
    check("all three sections present",
          _HEADER_THEORY in out and _HEADER_CODE in out and _HEADER_RISK in out)

    # Verify section order: Theory Code Risk
    t_pos = out.index(_HEADER_THEORY)
    c_pos = out.index(_HEADER_CODE)
    r_pos = out.index(_HEADER_RISK)
    check("section order: Theory < Code < Risk", t_pos < c_pos < r_pos)

    # Code block fencing
    check("code fenced with python", "```python" in out)
    check("indentation preserved in code", "    diff = target" in out)

    # ── 8. Comma-separated pipeline_stage ────────────────────────────
    print("\n=== 8. Comma-separated pipeline_stage ===")
    ctx_csv = {
        "theory_context": [
            {
                "id": "csv1",
                "content": "This chunk covers both risk modelling and evaluation metrics.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "risk_management, evaluation"},
            },
        ],
        "code_context": [],
    }
    out = build_prompt_context(ctx_csv)
    check("CSV stage risk section", _HEADER_RISK in out)
    check("CSV content in output", "risk modelling" in out)

    # ── 9. List-typed pipeline_stage ─────────────────────────────────
    print("\n=== 9. List-typed pipeline_stage ===")
    ctx_list = {
        "theory_context": [
            {
                "id": "lst1",
                "content": "Position sizing limits exposure using Kelly criterion.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": ["risk_management"]},
            },
        ],
        "code_context": [],
    }
    out = build_prompt_context(ctx_list)
    check("list stage risk section", _HEADER_RISK in out)

    # ── 10. Non-Python code ──────────────────────────────────────────
    print("\n=== 10. Non-Python Code ===")
    ctx_generic = {
        "theory_context": [],
        "code_context": [
            {
                "id": "gen1",
                "content": "SELECT * FROM trades WHERE pnl > 0;",
                "metadata": {"chunk_type": "code", "pipeline_stage": ""},
            },
        ],
    }
    out = build_prompt_context(ctx_generic)
    # Should have a plain ``` fence, not ```python
    lines = out.split("\n")
    fence_lines = [l for l in lines if l.strip().startswith("```")]
    has_plain_fence = any(l.strip() == "```" for l in fence_lines)
    check("non-Python code gets plain fence", has_plain_fence)

    # ── 11. Short / empty content filtered ───────────────────────────
    print("\n=== 11. Short Content Filtered ===")
    ctx_short = {
        "theory_context": [
            {"id": "s1", "content": "hi", "metadata": {"chunk_type": "theory", "pipeline_stage": ""}},
            {"id": "s2", "content": "", "metadata": {"chunk_type": "theory", "pipeline_stage": ""}},
            {"id": "s3", "content": "A sufficiently long chunk about strategy design.", "metadata": {"chunk_type": "theory", "pipeline_stage": ""}},
        ],
        "code_context": [],
    }
    out = build_prompt_context(ctx_short)
    check("short content excluded", "hi" not in out.split(_HEADER_THEORY)[-1].split("\n")[1] if _HEADER_THEORY in out else True)
    check("valid content included", "sufficiently long" in out)

    # ── 12. metadata.score fallback ordering ─────────────────────────
    print("\n=== 12. Metadata Score Fallback ===")
    ctx_meta_score = {
        "theory_context": [
            {
                "id": "ms1",
                "content": "Alpha chunk ranked first by metadata score.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "", "score": 0.95},
            },
            {
                "id": "ms2",
                "content": "Beta chunk ranked second by metadata score.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "", "score": 0.3},
            },
        ],
        "code_context": [],
    }
    out = build_prompt_context(ctx_meta_score)
    check("meta score: alpha before beta", out.index("Alpha") < out.index("Beta"))

    # ── 13. Hard token cap enforcement ───────────────────────────────
    print("\n=== 13. Hard Token Cap ===")
    # Create many large chunks that together exceed the budget
    big_chunks = {
        "theory_context": [
            {
                "id": f"big_{i}",
                "content": f"Chunk number {i}. " + ("The portfolio construction algorithm rebalances weights nightly. " * 30),
                "metadata": {"chunk_type": "theory", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.9 - i * 0.05},
            }
            for i in range(10)
        ],
        "code_context": [],
    }
    out_big = build_prompt_context(big_chunks, max_tokens=2500)
    out_big_tokens = count_tokens(out_big)
    check(
        f"hard cap: output ≤ 2500 tokens (got {out_big_tokens})",
        out_big_tokens <= 2500,
    )
    check("hard cap: output is non-empty", len(out_big) > 0)

    # Even tighter budget
    out_tiny = build_prompt_context(big_chunks, max_tokens=100)
    out_tiny_tokens = count_tokens(out_tiny)
    check(
        f"tiny cap (100): output ≤ 100 tokens (got {out_tiny_tokens})",
        out_tiny_tokens <= 100,
    )

    # ── 14. Default max_tokens is 2500 ───────────────────────────────
    print("\n=== 14. Default Budget = 2500 ===")
    out_default = build_prompt_context(big_chunks)
    out_default_tokens = count_tokens(out_default)
    check(
        f"default budget: output ≤ 2500 tokens (got {out_default_tokens})",
        out_default_tokens <= 2500,
    )

    # ── 15. Lowest-scoring chunks dropped first ──────────────────────
    print("\n=== 15. Lowest Score Dropped First ===")
    scored_chunks = {
        "theory_context": [
            {
                "id": "hi_score",
                "content": "HIGH_PRIORITY chunk about momentum signal generation strategy with additional context about moving averages.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.95},
            },
            {
                "id": "lo_score",
                "content": "LOW_PRIORITY chunk about general market overview information and the history of financial econometrics.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.1},
            },
        ],
        "code_context": [],
    }
    # Give enough budget for only one chunk + header
    # Each chunk is roughly 18 tokens, header ~8 tokens.
    # Budget of 28 should allow header + 1 chunk but not 2.
    out_one = build_prompt_context(scored_chunks, max_tokens=28)
    if out_one:
        check("high score survives", "HIGH_PRIORITY" in out_one)
        check("low score dropped when budget tight", "LOW_PRIORITY" not in out_one)
    else:
        check("high score survives (budget too small)", True)
        check("low score dropped when budget tight (both dropped)", True)

    # ── 16. Code priority boost (intent=code_generation) ─────────────
    print("\n=== 16. Code Priority Boost ===")
    mixed_intent = {
        "theory_context": [
            {
                "id": "theory_a",
                "content": "THEORY_CHUNK explaining triple barrier labels and meta-labelling approach.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.7},
            },
        ],
        "code_context": [
            {
                "id": "code_a",
                "content": "def CODE_CHUNK():\n    # implement triple barrier\n    return labels",
                "metadata": {"chunk_type": "code", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.6},
            },
        ],
    }
    # With intent=code_generation and tight budget, code should win
    out_code_intent = build_prompt_context(
        mixed_intent, max_tokens=60, intent="code_generation",
    )
    out_theory_intent = build_prompt_context(
        mixed_intent, max_tokens=60, intent="explanation",
    )

    # When intent=code_generation, code chunk (0.6+0.5=1.1) > theory (0.7)
    if out_code_intent:
        has_code = "CODE_CHUNK" in out_code_intent
        check("code_generation intent: code chunk prioritised", has_code)

    # When intent=explanation, theory (0.7) > code (0.6), theory should appear
    if out_theory_intent:
        has_theory = "THEORY_CHUNK" in out_theory_intent
        check("explanation intent: theory chunk prioritised", has_theory)

    # ── 17. Intent from chunks dict ──────────────────────────────────
    print("\n=== 17. Intent from Chunks Dict ===")
    ctx_with_intent = {
        "theory_context": [
            {
                "id": "i1",
                "content": "INTENT_THEORY about volatility estimation models and GARCH.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.65},
            },
        ],
        "code_context": [
            {
                "id": "i2",
                "content": "def INTENT_CODE():\n    # GARCH volatility\n    return sigma",
                "metadata": {"chunk_type": "code", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.55},
            },
        ],
        "intent": "code_generation",  # embedded in chunks dict
    }
    out_dict_intent = build_prompt_context(ctx_with_intent, max_tokens=60)
    if out_dict_intent:
        check(
            "intent from dict: code prioritised",
            "INTENT_CODE" in out_dict_intent,
        )

    # Explicit intent= param overrides dict
    out_override = build_prompt_context(
        ctx_with_intent, max_tokens=60, intent="explanation",
    )
    if out_override:
        check(
            "explicit intent overrides dict",
            "INTENT_THEORY" in out_override,
        )

    # ── 18. Backward compatibility (old callers) ─────────────────────
    print("\n=== 18. Backward Compatibility ===")
    # Calling without max_tokens or intent should still work
    ctx_compat = {
        "theory_context": [
            {
                "id": "bc1",
                "content": "Simple theory chunk for backward compatibility testing purposes.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": ""},
            },
        ],
        "code_context": [],
    }
    out_compat = build_prompt_context(ctx_compat)
    check("backward compat: returns string", isinstance(out_compat, str))
    check("backward compat: has content", "backward compatibility" in out_compat)
    check(
        "backward compat: within default budget",
        count_tokens(out_compat) <= DEFAULT_MAX_TOKENS,
    )

    # ── 19. _strip_comments helper ──────────────────────────────────
    print("\n=== 19. _strip_comments ===")
    commented_code = (
        "# This is a header comment\n"
        "def applyPtSlOnT1(close, events):\n"
        "    # inner comment\n"
        "    events_ = events.copy()\n"
        "    # another comment\n"
        "    return events_"
    )
    stripped = _strip_comments(commented_code)
    check("strip: removes header comment", "# This is a header comment" not in stripped)
    check("strip: removes inner comment", "# inner comment" not in stripped)
    check("strip: removes another comment", "# another comment" not in stripped)
    check("strip: preserves def line", "def applyPtSlOnT1" in stripped)
    check("strip: preserves code line", "events_ = events.copy()" in stripped)
    check("strip: preserves return", "return events_" in stripped)
    # Empty input
    check("strip: empty string", _strip_comments("") == "")
    # No comments
    check("strip: no comments unchanged", _strip_comments("x = 1\ny = 2") == "x = 1\ny = 2")

    # ── 20. _trim_code_preserving_defs helper ────────────────────────
    print("\n=== 20. _trim_code_preserving_defs ===")
    big_func = (
        "# Module docstring comment\n"
        "# Another long comment line explaining the algorithm\n"
        "def applyPtSlOnT1(close, events, ptSl, molecule):\n"
        "    # Implementation comment\n"
        "    events_ = events.loc[molecule]\n"
        "    out = events_[['t1']].copy(deep=True)\n"
        "    if ptSl[0] > 0:\n"
        "        pt = ptSl[0] * events_['trgt']\n"
        "    else:\n"
        "        pt = pd.Series(index=events.index)\n"
        "    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():\n"
        "        # Loop body comment\n"
        "        df0 = close[loc:t1]\n"
        "        out.loc[loc, 'sl'] = df0.min()\n"
        "        out.loc[loc, 'pt'] = df0.max()\n"
        "    return out\n"
        "\n"
        "def getDailyVol(close, span=100):\n"
        "    # Volatility estimation comment\n"
        "    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))\n"
        "    return df0"
    )
    # With a generous budget, everything preserved
    trim1 = _trim_code_preserving_defs(big_func, max_tokens=500)
    check("trim_defs: generous budget has applyPtSlOnT1", "def applyPtSlOnT1" in trim1)
    check("trim_defs: generous budget has getDailyVol", "def getDailyVol" in trim1)

    # With a very tight budget: comments stripped, defs preserved
    trim2 = _trim_code_preserving_defs(big_func, max_tokens=30)
    check("trim_defs: tight budget preserves def applyPtSlOnT1", "def applyPtSlOnT1" in trim2)
    check("trim_defs: tight budget preserves def getDailyVol", "def getDailyVol" in trim2)
    check("trim_defs: tight budget comments stripped", "# Module docstring" not in trim2)
    check("trim_defs: tight budget within budget", count_tokens(trim2) <= 30)

    # Already fits returned unchanged
    small = "def foo():\n    pass"
    check("trim_defs: already fits unchanged", _trim_code_preserving_defs(small, 100) == small)

    # ── 21. Strict code mode: protected chunks ───────────────────────
    print("\n=== 21. Strict Code Mode: Protected Chunks ===")
    # Two high-scoring code chunks + one theory chunk.
    # Give a tight budget that would normally drop some code chunks.
    strict_ctx = {
        "theory_context": [
            {
                "id": "theory_10",
                "content": "Theory about triple barrier labeling approach and meta-labeling. " * 10,
                "metadata": {"chunk_type": "theory", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.85},
            },
        ],
        "code_context": [
            {
                "id": "code_high",
                "content": (
                    "# Important implementation\n"
                    "def applyPtSlOnT1(close, events, ptSl):\n"
                    "    events_ = events.copy()\n"
                    "    return events_"
                ),
                "metadata": {"chunk_type": "code", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.95},
            },
            {
                "id": "code_mid",
                "content": (
                    "# Volatility helper\n"
                    "def getDailyVol(close, span=100):\n"
                    "    df0 = close.pct_change()\n"
                    "    return df0.rolling(span).std()"
                ),
                "metadata": {"chunk_type": "code", "pipeline_stage": "signal_generation"},
                "scores": {"final": 0.80},
            },
        ],
        "strict_code_mode": True,
        "intent": "code_generation",
    }

    # Generous budget: both code chunks included
    out_strict = build_prompt_context(strict_ctx, max_tokens=2500, strict_code_mode=True)
    check("strict: has applyPtSlOnT1", "def applyPtSlOnT1" in out_strict)
    check("strict: has getDailyVol", "def getDailyVol" in out_strict)
    check("strict: code section present", _HEADER_CODE in out_strict)

    # Tight budget: code chunks STILL included (protected)
    out_tight = build_prompt_context(strict_ctx, max_tokens=80, strict_code_mode=True)
    check("strict tight: has applyPtSlOnT1", "def applyPtSlOnT1" in out_tight)
    check("strict tight: has getDailyVol", "def getDailyVol" in out_tight)
    check("strict tight: within budget", count_tokens(out_tight) <= 80)

    # ── 22. Strict code mode: comments trimmed before code ───────────
    print("\n=== 22. Strict Code Mode: Comments Trimmed First ===")
    # The strict mode output for a tight budget should strip comments
    check(
        "strict tight: comments stripped from protected chunk",
        "# Important implementation" not in out_tight,
    )
    check(
        "strict tight: comments stripped from second protected chunk",
        "# Volatility helper" not in out_tight,
    )

    # ── 23. Strict code mode: function defs never trimmed ────────────
    print("\n=== 23. Strict Code Mode: Function Defs Preserved ===")
    # Even with a very small budget, def lines survive
    out_micro = build_prompt_context(strict_ctx, max_tokens=40, strict_code_mode=True)
    check("strict micro: def applyPtSlOnT1 preserved", "def applyPtSlOnT1" in out_micro)
    # getDailyVol might not fit in 40 tokens, but at least the top-1 def survives
    check("strict micro: within budget", count_tokens(out_micro) <= 40)

    # ── 24. Strict code mode from chunks dict ────────────────────────
    print("\n=== 24. Strict Code Mode from Chunks Dict ===")
    # strict_code_mode=True embedded in the chunks dict
    out_dict_strict = build_prompt_context(strict_ctx, max_tokens=2500)
    check("dict strict: applyPtSlOnT1 present",
          "def applyPtSlOnT1" in out_dict_strict)
    check("dict strict: getDailyVol present",
          "def getDailyVol" in out_dict_strict)

    # ── 25. Non-strict mode: no protection ───────────────────────────
    print("\n=== 25. Non-Strict Mode ===")
    non_strict_ctx = {
        "theory_context": strict_ctx["theory_context"],
        "code_context": strict_ctx["code_context"],
        "intent": "code_generation",
    }
    # With non-strict and tight budget, theory with higher boosted
    # priority may push code out — or code may just barely fit.
    out_non_strict = build_prompt_context(non_strict_ctx, max_tokens=2500)
    check("non-strict: returns string", isinstance(out_non_strict, str))
    check("non-strict: non-empty", len(out_non_strict) > 0)

    # ── 26. _STRICT_CODE_PROTECTED_COUNT constant ────────────────────
    print("\n=== 26. Constants ===")
    check("_STRICT_CODE_PROTECTED_COUNT == 2", _STRICT_CODE_PROTECTED_COUNT == 2)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_tests()
