"""
Chunking Module for Centurion Capital LLC RAG Pipeline.

Converts structured blocks (from ``pdf_ingestion.ingest_pdf_structured``)
into enriched, embedding-ready chunks with rich metadata.

Design goals
------------
- **Deterministic** — no LLM calls; every output is reproducible.
- **Split-safe** — code is never split mid-function; text is never
  split mid-paragraph across unrelated sections.
- **Metadata-rich** — every chunk carries ``chunk_type``, inferred
  ``pipeline_stage``, detected ``chapter``, and (for code) parsed
  ``function_names`` / ``class_names`` / ``imports``.
- **Parent-child hierarchy** — every chunk has a deterministic
  ``parent_id`` (SHA-256 of its source block) and sequential
  ``section_id`` within that block, enabling sibling-chunk
  expansion at retrieval time.

Input:  ``List[StructuredBlock]``  (from ``pdf_ingestion.py``)
Output: ``List[EnrichedChunk]``    (ready for embedding & ChromaDB)

Tech stack: Python 3.11 · No LangChain · No LLM calls.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Type definitions
# ═══════════════════════════════════════════════════════════════════════════

# Input type (matches pdf_ingestion.StructuredBlock)
# {"page": int, "block_type": "code"|"text", "content": str}
StructuredBlock = Dict[str, Any]


@dataclass(frozen=True)
class EnrichedChunk:
    """A single chunk ready for embedding with attached metadata.

    Attributes:
        content:  The chunk text (code indentation preserved if code).
        metadata: Dict with keys depending on ``chunk_type``:

            **code** chunks::

                {
                    "chunk_type":      "code",
                    "page":            int,
                    "parent_id":       str,   # SHA-256 hex (16 chars) of source block
                    "section_id":      int,   # 0-based position within parent block
                    "function_names":  List[str],
                    "class_names":     List[str],
                    "imports":         List[str],
                    "pipeline_stage":  str,
                    "chapter":         str,
                }

            **theory** (text) chunks::

                {
                    "chunk_type":      "theory",
                    "page":            int,
                    "parent_id":       str,   # SHA-256 hex (16 chars) of source block
                    "section_id":      int,   # 0-based position within parent block
                    "chapter":         str,
                    "pipeline_stage":  str,
                    "summary":         str,   # deterministic 2-line summary
                }
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience serialisation for downstream consumers
    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "metadata": dict(self.metadata)}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Token counting (lightweight, no heavy dependency)
# ═══════════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"\S+")

# Target token window for text chunks
TEXT_CHUNK_MIN_TOKENS = 400
TEXT_CHUNK_MAX_TOKENS = 700


def _token_count(text: str) -> int:
    """Approximate token count via whitespace splitting.

    Roughly correlates with BPE token counts at ~1.3× word count,
    but for chunking-window purposes raw word count is a safe,
    fast, deterministic proxy.

    Args:
        text: Input text.

    Returns:
        Approximate number of tokens.
    """
    return len(_WORD_RE.findall(text))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pipeline-stage inference (keyword rules, no LLM)
# ═══════════════════════════════════════════════════════════════════════════

# Each stage is defined by a list of trigger keywords / phrases.
# Matching is case-insensitive; a stage is selected when the content
# contains *any* keyword from its trigger list.  If multiple stages
# match, the one with the most keyword hits wins.

_PIPELINE_STAGE_RULES: Dict[str, List[str]] = {
    "risk_management": [
        "drawdown", "max drawdown", "risk", "stop-loss", "stop loss",
        "stoploss", "var ", "value at risk", "position sizing",
        "risk parity", "tail risk", "volatility target",
        "kelly criterion", "risk budget", "exposure limit",
        "margin call", "liquidation", "hedging",
    ],
    "evaluation": [
        "sharpe", "sharpe ratio", "sortino", "calmar", "returns",
        "performance", "benchmark", "alpha", "beta ", "information ratio",
        "treynor", "omega ratio", "annualized", "cumulative return",
        "hit rate", "profit factor", "payoff ratio", "win rate",
        "loss rate", "expectancy",
    ],
    "signal_generation": [
        "signal", "z-score", "zscore", "z score", "indicator",
        "oscillator", "macd", "rsi", "bollinger", "moving average",
        "crossover", "divergence", "stochastic", "atr ",
        "adx", "obv", "volume signal", "buy signal", "sell signal",
        "entry signal", "exit signal", "trigger",
    ],
    "data_processing": [
        "data cleaning", "missing data", "imputation", "outlier",
        "feature engineering", "normalization", "standardization",
        "scaling", "resampling", "ohlcv", "tick data", "bar data",
        "data pipeline", "etl", "data quality", "data schema",
        "time series", "frequency",
    ],
    "backtesting": [
        "backtest", "backtesting", "historical simulation",
        "walk-forward", "walk forward", "out-of-sample",
        "in-sample", "paper trading", "simulation",
        "event-driven", "vectorized backtest", "lookback",
    ],
    "portfolio_construction": [
        "portfolio", "allocation", "rebalancing", "diversification",
        "mean-variance", "mean variance", "efficient frontier",
        "black-litterman", "minimum variance", "equal weight",
        "risk parity portfolio", "correlation matrix", "covariance",
        "asset allocation",
    ],
    "execution": [
        "execution", "order", "slippage", "market impact",
        "transaction cost", "latency", "fill rate",
        "limit order", "market order", "twap", "vwap",
        "smart order", "broker", "api call",
    ],
    "machine_learning": [
        "machine learning", "deep learning", "neural network",
        "random forest", "gradient boosting", "xgboost",
        "lstm", "transformer", "training", "validation",
        "cross-validation", "hyperparameter", "overfitting",
        "regularization", "feature importance", "model selection",
        "ensemble", "classification", "regression",
    ],
    "market_microstructure": [
        "bid-ask", "spread", "order book", "market maker",
        "liquidity", "price impact", "tick size",
        "high-frequency", "hft", "latency arbitrage",
    ],
}

# Pre-compile patterns for fast matching
_STAGE_PATTERNS: Dict[str, re.Pattern] = {}
for _stage, _keywords in _PIPELINE_STAGE_RULES.items():
    # Build alternation; escape special chars; word-boundary where sensible
    parts = []
    for kw in _keywords:
        escaped = re.escape(kw)
        # Add word boundary for short keywords to avoid false positives
        if len(kw) <= 5:
            parts.append(rf"\b{escaped}\b")
        else:
            parts.append(escaped)
    _STAGE_PATTERNS[_stage] = re.compile("|".join(parts), re.IGNORECASE)


def infer_stage(content: str) -> str:
    """Infer the pipeline stage of a chunk using keyword-rule matching.

    Counts how many distinct keywords from each stage appear in
    *content* and returns the stage with the highest count.
    Returns ``"general"`` if no stage matches.

    Args:
        content: The chunk text to classify.

    Returns:
        Pipeline stage string (e.g. ``"risk_management"``).
    """
    best_stage = "general"
    best_count = 0

    for stage, pattern in _STAGE_PATTERNS.items():
        hits = pattern.findall(content)
        count = len(hits)
        if count > best_count:
            best_count = count
            best_stage = stage

    logger.debug("infer_stage → %s (%d keyword hits).", best_stage, best_count)
    return best_stage


# ═══════════════════════════════════════════════════════════════════════════
# 3. Chapter / heading detection
# ═══════════════════════════════════════════════════════════════════════════

# Chapter heading: "Chapter 3 — Risk Management" or "CHAPTER 5: Signals"
_CHAPTER_RE = re.compile(
    r"^\s*(?:chapter|ch\.?)\s+(\d+)[\s.:—\-]*(.*)",
    re.IGNORECASE | re.MULTILINE,
)

# Numbered section heading: "2.3 Denoising Covariance Matrices"
_SECTION_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][a-zA-Z ,'\.\-]{2,100})\s*$",
    re.MULTILINE,
)

# ALL-CAPS heading on its own line
_ALLCAPS_HEADING_RE = re.compile(
    r"^\s*([A-Z][A-Z\s]{3,80})\s*$",
    re.MULTILINE,
)


def detect_chapter(content: str) -> str:
    """Detect the chapter or section heading from chunk content.

    Scans the first ~500 characters for chapter headings, numbered
    sections, or prominent ALL-CAPS titles.

    Args:
        content: Chunk text.

    Returns:
        Detected heading string, or ``""`` if none found.
    """
    # Only inspect the beginning of the chunk
    prefix = content[:500]

    m = _CHAPTER_RE.search(prefix)
    if m:
        num = m.group(1)
        title = m.group(2).strip()
        return f"Chapter {num}: {title}" if title else f"Chapter {num}"

    m = _SECTION_RE.search(prefix)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"

    m = _ALLCAPS_HEADING_RE.search(prefix)
    if m:
        heading = m.group(1).strip()
        # Ignore very short all-caps that might be noise
        if len(heading) > 5:
            return heading.title()

    return ""


# ═══════════════════════════════════════════════════════════════════════════
# 4. Deterministic 2-line summary (no LLM)
# ═══════════════════════════════════════════════════════════════════════════

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"'\(\[])")


def generate_2_line_summary(content: str) -> str:
    """Generate a deterministic 2-sentence extractive summary.

    Strategy:
    1. Split the text into sentences.
    2. Pick the first non-trivial sentence (≥ 8 words) as sentence 1.
    3. Pick the second-longest sentence in the first half as sentence 2
       (heuristic: longer sentences tend to carry more information).
    4. If only one usable sentence exists, return just that one.

    No LLM is used — the summary is purely extractive.

    Args:
        content: The chunk text.

    Returns:
        A 1–2 sentence summary string.
    """
    sentences = _SENTENCE_SPLIT.split(content.strip())
    # Filter to meaningful sentences (≥8 words, not just a heading)
    usable = [
        s.strip() for s in sentences
        if len(s.split()) >= 8 and not s.strip().isupper()
    ]

    if not usable:
        # Fall back to raw truncation
        words = content.split()[:30]
        return " ".join(words) + ("..." if len(content.split()) > 30 else "")

    first = usable[0]
    if len(usable) == 1:
        return first

    # Pick the longest sentence from the first half (excluding the first)
    first_half = usable[1 : max(2, len(usable) // 2 + 1)]
    second = max(first_half, key=lambda s: len(s.split()))

    return f"{first} {second}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Code-block parsing — split into functions / classes / import groups
# ═══════════════════════════════════════════════════════════════════════════

# Regex for top-level def / class / import at zero or minimal indent
_FUNC_START_RE = re.compile(r"^([ \t]*)def\s+(\w+)\s*\(", re.MULTILINE)
_CLASS_START_RE = re.compile(r"^([ \t]*)class\s+(\w+)", re.MULTILINE)
_IMPORT_RE = re.compile(
    r"^(?:import\s+[\w.]+(?:\s*,\s*[\w.]+)*"
    r"|from\s+[\w.]+\s+import\s+.+)$",
    re.MULTILINE,
)
_DECORATOR_RE = re.compile(r"^\s*@\w+", re.MULTILINE)


def _extract_imports(code: str) -> List[str]:
    """Extract all import lines from a code string.

    Args:
        code: Python source code.

    Returns:
        List of import statement strings.
    """
    return [m.group().strip() for m in _IMPORT_RE.finditer(code)]


def _parse_code_units(code: str) -> List[Dict[str, Any]]:
    """Parse a code block into individual logical units.

    A "unit" is one of:
    - A standalone function (``def``)
    - A class (with all its methods)
    - An import group (consecutive import lines)
    - A miscellaneous top-level block

    Uses Python ``ast`` when possible, falls back to regex for
    unparseable fragments.

    Args:
        code: Python source code text.

    Returns:
        List of dicts, each with keys:
        - ``"type"``: ``"function"`` | ``"class"`` | ``"imports"`` | ``"misc"``
        - ``"name"``: Name of the function/class (empty for imports/misc).
        - ``"content"``: The source text of the unit.
        - ``"line_start"``: 0-based starting line index.
        - ``"line_end"``: 0-based ending line index (inclusive).
    """
    lines = code.split("\n")
    units: List[Dict[str, Any]] = []

    # Try AST-based parsing first (most reliable)
    try:
        tree = ast.parse(textwrap.dedent(code))
        return _parse_via_ast(tree, lines, code)
    except SyntaxError:
        logger.debug("AST parse failed — falling back to regex-based splitter.")

    # Regex fallback
    return _parse_via_regex(lines)


def _parse_via_ast(
    tree: ast.Module,
    lines: List[str],
    code: str,
) -> List[Dict[str, Any]]:
    """Extract code units using the AST.

    Handles functions, classes, and import groups. Lines not covered
    by any AST node are grouped as ``"misc"`` blocks.

    Args:
        tree: Parsed AST module.
        lines: Source split into lines.
        code: Original source text.

    Returns:
        Sorted list of code-unit dicts.
    """
    units: List[Dict[str, Any]] = []
    covered: set[int] = set()

    # Gather import blocks
    import_lines: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if hasattr(node, "lineno"):
                import_lines.append(node.lineno - 1)  # 0-based
                covered.add(node.lineno - 1)

    if import_lines:
        import_lines.sort()
        # Group consecutive import lines
        groups = _group_consecutive(import_lines)
        for grp in groups:
            content = "\n".join(lines[i] for i in grp)
            units.append({
                "type": "imports",
                "name": "",
                "content": content,
                "line_start": grp[0],
                "line_end": grp[-1],
            })

    # Functions and classes (top-level nodes only)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno - 1 if node.end_lineno else start
            # Include decorators
            if node.decorator_list:
                dec_start = node.decorator_list[0].lineno - 1
                start = min(start, dec_start)
            content = "\n".join(lines[start : end + 1])
            for i in range(start, end + 1):
                covered.add(i)
            units.append({
                "type": "function",
                "name": node.name,
                "content": content,
                "line_start": start,
                "line_end": end,
            })

        elif isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = node.end_lineno - 1 if node.end_lineno else start
            if node.decorator_list:
                dec_start = node.decorator_list[0].lineno - 1
                start = min(start, dec_start)
            content = "\n".join(lines[start : end + 1])
            for i in range(start, end + 1):
                covered.add(i)
            units.append({
                "type": "class",
                "name": node.name,
                "content": content,
                "line_start": start,
                "line_end": end,
            })

    # Collect uncovered lines as misc blocks
    misc_lines: List[int] = [
        i for i in range(len(lines))
        if i not in covered and lines[i].strip()
    ]
    if misc_lines:
        groups = _group_consecutive(misc_lines, max_gap=2)
        for grp in groups:
            content = "\n".join(lines[i] for i in grp)
            if content.strip():
                units.append({
                    "type": "misc",
                    "name": "",
                    "content": content,
                    "line_start": grp[0],
                    "line_end": grp[-1],
                })

    units.sort(key=lambda u: u["line_start"])
    return units


def _parse_via_regex(lines: List[str]) -> List[Dict[str, Any]]:
    """Regex-based fallback for splitting code that doesn't parse as valid Python.

    Splits on top-level ``def`` / ``class`` boundaries while keeping
    each body intact (by tracking indentation).

    Args:
        lines: Source lines.

    Returns:
        List of code-unit dicts.
    """
    units: List[Dict[str, Any]] = []
    n = len(lines)

    # Find top-level def/class boundaries
    boundaries: List[Tuple[int, str, str]] = []  # (line_idx, type, name)

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(line.lstrip())

        if indent <= 4:  # top-level or first indent
            m = re.match(r"def\s+(\w+)\s*\(", stripped)
            if m:
                boundaries.append((i, "function", m.group(1)))
                continue
            m = re.match(r"class\s+(\w+)", stripped)
            if m:
                boundaries.append((i, "class", m.group(1)))
                continue

    if not boundaries:
        # No recognisable structure — return entire block as misc
        content = "\n".join(lines)
        if content.strip():
            # Try to extract any import lines
            imports = _extract_imports(content)
            unit_type = "imports" if imports and len(imports) == len([l for l in lines if l.strip()]) else "misc"
            units.append({
                "type": unit_type,
                "name": "",
                "content": content,
                "line_start": 0,
                "line_end": n - 1,
            })
        return units

    # Collect import lines before the first boundary
    first_boundary_line = boundaries[0][0]
    # Walk backward from boundary to include decorators
    decorator_start = first_boundary_line
    while decorator_start > 0 and lines[decorator_start - 1].strip().startswith("@"):
        decorator_start -= 1
        boundaries[0] = (decorator_start, boundaries[0][1], boundaries[0][2])
        first_boundary_line = decorator_start

    if first_boundary_line > 0:
        pre_lines = lines[:first_boundary_line]
        pre_content = "\n".join(pre_lines).strip()
        if pre_content:
            imports = _extract_imports(pre_content)
            units.append({
                "type": "imports" if imports else "misc",
                "name": "",
                "content": pre_content,
                "line_start": 0,
                "line_end": first_boundary_line - 1,
            })

    # Split each boundary into its own unit
    for idx, (line_idx, unit_type, name) in enumerate(boundaries):
        # End is the line before the next boundary (or EOF)
        if idx + 1 < len(boundaries):
            next_start = boundaries[idx + 1][0]
            # Walk backward past decorators of the NEXT boundary
            end = next_start - 1
            while end > line_idx and not lines[end].strip():
                end -= 1
        else:
            end = n - 1

        content = "\n".join(lines[line_idx : end + 1])
        units.append({
            "type": unit_type,
            "name": name,
            "content": content,
            "line_start": line_idx,
            "line_end": end,
        })

    return units


def _group_consecutive(
    indices: List[int], max_gap: int = 1
) -> List[List[int]]:
    """Group sorted line indices into consecutive runs.

    Args:
        indices: Sorted list of line indices.
        max_gap: Maximum gap between indices to keep in the same group.

    Returns:
        List of groups (each a list of indices).
    """
    if not indices:
        return []
    groups: List[List[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx <= groups[-1][-1] + max_gap + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


# ═══════════════════════════════════════════════════════════════════════════
# 6. Text-block chunking — semantic boundary splitting
# ═══════════════════════════════════════════════════════════════════════════

# Paragraph break: 2+ newlines
_PARA_BREAK = re.compile(r"\n\s*\n")

# Heading patterns (used as hard split points)
_HEADING_PATTERNS = [
    re.compile(r"^\s*#{1,4}\s+\S", re.MULTILINE),                       # Markdown
    re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z]\w", re.MULTILINE),          # Numbered
    re.compile(r"^\s*[A-Z][A-Z\s]{3,80}$", re.MULTILINE),              # ALL-CAPS
    re.compile(r"^\s*(?:chapter|ch\.?)\s+\d+", re.IGNORECASE | re.MULTILINE),
]


def _is_heading(line: str) -> bool:
    """Return True if *line* looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return any(pat.match(stripped) for pat in _HEADING_PATTERNS)


def _split_into_semantic_sections(text: str) -> List[str]:
    """Split text into semantic sections at heading boundaries.

    Each section starts with a heading (if detected) and contains
    all following paragraphs until the next heading.

    Args:
        text: The text block content.

    Returns:
        List of section strings.
    """
    lines = text.split("\n")
    sections: List[str] = []
    current: List[str] = []

    for line in lines:
        if _is_heading(line) and current:
            section_text = "\n".join(current).strip()
            if section_text:
                sections.append(section_text)
            current = [line]
        else:
            current.append(line)

    if current:
        section_text = "\n".join(current).strip()
        if section_text:
            sections.append(section_text)

    return sections


def _split_section_into_paragraphs(section: str) -> List[str]:
    """Split a section into paragraphs (on blank-line boundaries).

    Args:
        section: A semantic section string.

    Returns:
        List of paragraph strings.
    """
    paras = _PARA_BREAK.split(section)
    return [p.strip() for p in paras if p.strip()]


def _chunk_text_block(
    content: str,
    min_tokens: int = TEXT_CHUNK_MIN_TOKENS,
    max_tokens: int = TEXT_CHUNK_MAX_TOKENS,
) -> List[str]:
    """Chunk a text block into pieces of 400–700 tokens.

    Strategy (never mixes unrelated sections):
    1. Split into semantic sections (at heading boundaries).
    2. Within each section, split into paragraphs.
    3. Accumulate paragraphs into a chunk until the token budget
       is reached, then start a new chunk.
    4. If a single paragraph exceeds *max_tokens*, split it on
       sentence boundaries.
    5. A section boundary always forces a new chunk (even if the
       current chunk is below *min_tokens*).

    Args:
        content: Full text-block content.
        min_tokens: Minimum chunk target (default 400).
        max_tokens: Maximum chunk target (default 700).

    Returns:
        List of chunk strings.
    """
    sections = _split_into_semantic_sections(content)
    chunks: List[str] = []

    for section in sections:
        paragraphs = _split_section_into_paragraphs(section)
        _accumulate_paragraphs(paragraphs, chunks, min_tokens, max_tokens)

    return [c for c in chunks if c.strip()]


def _accumulate_paragraphs(
    paragraphs: List[str],
    chunks: List[str],
    min_tokens: int,
    max_tokens: int,
) -> None:
    """Accumulate paragraphs into chunks within the token budget.

    Modifies *chunks* in place.

    Args:
        paragraphs: Paragraphs to group.
        chunks: Output list (appended to).
        min_tokens: Soft minimum for a chunk.
        max_tokens: Hard maximum for a chunk.
    """
    buffer: List[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = _token_count(para)

        # Over-sized paragraph → split on sentences, then accumulate
        if para_tokens > max_tokens:
            # Flush current buffer first
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer = []
                buffer_tokens = 0
            _split_oversized_paragraph(para, chunks, max_tokens)
            continue

        # Would adding this paragraph exceed the max?
        if buffer_tokens + para_tokens > max_tokens and buffer:
            chunks.append("\n\n".join(buffer))
            buffer = []
            buffer_tokens = 0

        buffer.append(para)
        buffer_tokens += para_tokens

    # Flush remaining
    if buffer:
        chunks.append("\n\n".join(buffer))


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"'\(\[])")


def _split_oversized_paragraph(
    para: str,
    chunks: List[str],
    max_tokens: int,
) -> None:
    """Split an oversized paragraph on sentence boundaries.

    Args:
        para: The oversized paragraph.
        chunks: Output list (appended to).
        max_tokens: Maximum tokens per chunk.
    """
    sentences = _SENT_SPLIT.split(para)
    buf: List[str] = []
    buf_tokens = 0

    for sent in sentences:
        sent_tokens = _token_count(sent)
        if buf_tokens + sent_tokens > max_tokens and buf:
            chunks.append(" ".join(buf))
            buf = []
            buf_tokens = 0
        buf.append(sent.strip())
        buf_tokens += sent_tokens

    if buf:
        chunks.append(" ".join(buf))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Code-block chunking — function / class / import grouping
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_code_block(
    content: str,
) -> List[Dict[str, Any]]:
    """Chunk a code block into logical units.

    Rules:
    - Each function → one chunk.
    - Each class → one chunk.
    - Import group → attached to the following function/class chunk
      as a prefix, so dependent context travels with the code.
    - Misc top-level code → one chunk.
    - A function or class is **never** split.

    Args:
        content: Code block content (with indentation preserved).

    Returns:
        List of dicts with keys ``"content"``, ``"function_names"``,
        ``"class_names"``, ``"imports"``.
    """
    units = _parse_code_units(content)
    if not units:
        return []

    # Separate imports from code units
    import_units: List[Dict[str, Any]] = []
    code_units: List[Dict[str, Any]] = []

    for unit in units:
        if unit["type"] == "imports":
            import_units.append(unit)
        else:
            code_units.append(unit)

    # Build the shared imports prefix
    imports_text = "\n".join(u["content"] for u in import_units)
    all_imports = []
    for u in import_units:
        all_imports.extend(
            line.strip()
            for line in u["content"].split("\n")
            if line.strip()
        )

    results: List[Dict[str, Any]] = []

    for unit in code_units:
        func_names: List[str] = []
        class_names: List[str] = []

        if unit["type"] == "function":
            func_names = [unit["name"]]
        elif unit["type"] == "class":
            class_names = [unit["name"]]
            # Also extract method names from the class body
            for m in _FUNC_START_RE.finditer(unit["content"]):
                method_name = m.group(2)
                if method_name not in class_names:
                    func_names.append(method_name)

        # Prefix imports with the code unit so embeddings have context
        if imports_text and (func_names or class_names):
            chunk_content = imports_text + "\n\n" + unit["content"]
        else:
            chunk_content = unit["content"]

        results.append({
            "content": chunk_content,
            "function_names": func_names,
            "class_names": class_names,
            "imports": all_imports,
        })

    # If there are only import units and no code units, emit them alone
    if import_units and not code_units:
        results.append({
            "content": imports_text,
            "function_names": [],
            "class_names": [],
            "imports": all_imports,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 8. Main entry point — chunk_structured_blocks()
# ═══════════════════════════════════════════════════════════════════════════

def chunk_structured_blocks(
    blocks: List[StructuredBlock],
    *,
    text_min_tokens: int = TEXT_CHUNK_MIN_TOKENS,
    text_max_tokens: int = TEXT_CHUNK_MAX_TOKENS,
) -> List[Dict[str, Any]]:
    """Convert structured blocks into enriched, embedding-ready chunks.

    This is the **primary public API** of the module.

    Args:
        blocks: List of ``StructuredBlock`` dicts from
            ``pdf_ingestion.ingest_pdf_structured()``.
            Each dict has keys ``"page"``, ``"block_type"``, ``"content"``.
        text_min_tokens: Minimum token target for text chunks (default 400).
        text_max_tokens: Maximum token target for text chunks (default 700).

    Returns:
        List of enriched chunk dicts::

            [
                {
                    "content": str,
                    "metadata": {
                        "chunk_type": "code" | "theory",
                        "page": int,
                        ...
                    }
                },
                ...
            ]

    Guarantees:
        - Code indentation is preserved.
        - No function or class is split across chunks.
        - Imports are preserved with dependent code.
        - Text chunks respect section boundaries (400–700 tokens).
        - Output is fully deterministic (no LLM calls).
    """
    if not blocks:
        logger.warning("chunk_structured_blocks called with empty input.")
        return []

    logger.info(
        "Chunking %d structured blocks (text_range=%d–%d tokens).",
        len(blocks), text_min_tokens, text_max_tokens,
    )

    enriched: List[Dict[str, Any]] = []

    for block in blocks:
        page = block.get("page", 0)
        block_type = block.get("block_type", "text")
        content = block.get("content", "").strip()

        if not content:
            continue

        # Deterministic parent_id: 16-char SHA-256 hex of the raw block content
        parent_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        snippet_id = block.get("snippet_id", "")
        snippet_title = block.get("snippet_title", "")

        if block_type == "code":
            _process_code_block(
                content, page, enriched, parent_id,
                snippet_id=snippet_id, snippet_title=snippet_title,
            )
        else:
            _process_text_block(
                content, page, enriched,
                text_min_tokens, text_max_tokens,
                parent_id=parent_id,
                snippet_id=snippet_id, snippet_title=snippet_title,
            )

    logger.info(
        "Chunking complete: %d enriched chunks produced from %d blocks.",
        len(enriched), len(blocks),
    )
    return enriched


def _process_code_block(
    content: str,
    page: int,
    enriched: List[Dict[str, Any]],
    parent_id: str = "",
    *,
    snippet_id: str = "",
    snippet_title: str = "",
) -> None:
    """Process a single code block into enriched chunks.

    Args:
        content: Code block text.
        page: Source page number.
        enriched: Output list (appended to in-place).
        parent_id: Deterministic ID of the parent block (SHA-256 hex, 16 chars).
        snippet_id: Detected snippet number (e.g. "3.1").
        snippet_title: Detected snippet title (e.g. "getDailyVol Function").
    """
    code_chunks = _chunk_code_block(content)

    for section_id, cc in enumerate(code_chunks):
        chunk_content = cc["content"]
        if not chunk_content.strip():
            continue

        metadata = {
            "chunk_type": "code",
            "page": page,
            "parent_id": parent_id,
            "section_id": section_id,
            "function_names": cc["function_names"],
            "class_names": cc["class_names"],
            "imports": cc["imports"],
            "pipeline_stage": infer_stage(chunk_content),
            "chapter": detect_chapter(chunk_content),
        }
        if snippet_id:
            metadata["snippet_id"] = snippet_id
            metadata["snippet_title"] = snippet_title

        enriched.append({
            "content": chunk_content,
            "metadata": metadata,
        })


def _process_text_block(
    content: str,
    page: int,
    enriched: List[Dict[str, Any]],
    min_tokens: int,
    max_tokens: int,
    *,
    parent_id: str = "",
    snippet_id: str = "",
    snippet_title: str = "",
) -> None:
    """Process a single text block into enriched chunks.

    Args:
        content: Text block content.
        page: Source page number.
        enriched: Output list (appended to in-place).
        min_tokens: Minimum token target.
        max_tokens: Maximum token target.
        parent_id: Deterministic ID of the parent block (SHA-256 hex, 16 chars).
        snippet_id: Detected snippet number (e.g. "3.1").
        snippet_title: Detected snippet title (e.g. "getDailyVol Function").
    """
    text_chunks = _chunk_text_block(content, min_tokens, max_tokens)

    for section_id, tc in enumerate(text_chunks):
        if not tc.strip():
            continue

        metadata = {
            "chunk_type": "theory",
            "page": page,
            "parent_id": parent_id,
            "section_id": section_id,
            "chapter": detect_chapter(tc),
            "pipeline_stage": infer_stage(tc),
            "summary": generate_2_line_summary(tc),
        }
        if snippet_id:
            metadata["snippet_id"] = snippet_id
            metadata["snippet_title"] = snippet_title

        enriched.append({
            "content": tc,
            "metadata": metadata,
        })


# ═══════════════════════════════════════════════════════════════════════════
# 9. Convenience helper — full PDF → enriched chunks in one call
# ═══════════════════════════════════════════════════════════════════════════

def chunk_pdf(
    pdf_path: str,
    *,
    text_min_tokens: int = TEXT_CHUNK_MIN_TOKENS,
    text_max_tokens: int = TEXT_CHUNK_MAX_TOKENS,
) -> List[Dict[str, Any]]:
    """End-to-end: extract a PDF and return enriched chunks.

    Combines ``pdf_ingestion.ingest_pdf_structured`` with
    ``chunk_structured_blocks`` for a single-call workflow.

    Args:
        pdf_path: Path to the PDF file.
        text_min_tokens: Minimum token target for text chunks.
        text_max_tokens: Maximum token target for text chunks.

    Returns:
        List of enriched chunk dicts (same format as
        ``chunk_structured_blocks``).
    """
    from rag_pipeline.pdf_ingestion import ingest_pdf_structured

    logger.info("chunk_pdf: extracting structured blocks from %s", pdf_path)
    blocks = ingest_pdf_structured(pdf_path)

    logger.info("chunk_pdf: chunking %d blocks.", len(blocks))
    return chunk_structured_blocks(
        blocks,
        text_min_tokens=text_min_tokens,
        text_max_tokens=text_max_tokens,
    )
