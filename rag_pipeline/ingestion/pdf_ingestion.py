"""
PDF Ingestion Module for Centurion Capital LLC RAG Pipeline.

Production-grade PDF text extraction, cleaning, normalisation, and
code-block detection.  Designed for downstream ChromaDB storage.

Responsibilities
----------------
1. **Extract** text page-by-page via PyMuPDF (``fitz``).
2. **Remove noise** — page numbers, repeated headers / footers,
   table-of-contents pages, index sections, copyright blocks.
3. **Normalise** — fix hyphenated line-breaks, collapse whitespace,
   preserve paragraph structure, translate common Unicode symbols.
4. **Detect code blocks** — triple-backtick fences, consistent
   indentation, Python keyword heuristics.
5. **Structured output** — return ``[{"page": int, "block_type": ..., "content": str}]``.
6. **Full ingestion pipeline** — chunk, embed, and store in ChromaDB
   via ``PDFIngestionService`` (backward-compatible).

Tech stack: Python 3.11 · PyMuPDF (fitz) · No LangChain.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from rag_pipeline.config import RAGConfig
from rag_pipeline.storage.embeddings import EmbeddingService
from rag_pipeline.storage.vector_store import VectorStoreManager, _normalize_for_embedding

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Type definitions
# ═══════════════════════════════════════════════════════════════════════════

class StructuredBlock(TypedDict):
    """A single block in the structured ingestion output."""
    page: int
    block_type: str  # "code" | "text"
    content: str


# ═══════════════════════════════════════════════════════════════════════════
# 1. Unicode symbol normalisation map
# ═══════════════════════════════════════════════════════════════════════════

# Greek letters and common mathematical / technical symbols
_UNICODE_REPLACEMENTS: Dict[str, str] = {
    "σ": "sigma",   "Σ": "Sigma",
    "μ": "mu",      "Μ": "Mu",
    "Δ": "delta",   "δ": "delta",
    "α": "alpha",   "Α": "Alpha",
    "β": "beta",    "Β": "Beta",
    "γ": "gamma",   "Γ": "Gamma",
    "ε": "epsilon", "Ε": "Epsilon",
    "ζ": "zeta",    "Ζ": "Zeta",
    "η": "eta",     "Η": "Eta",
    "θ": "theta",   "Θ": "Theta",
    "ι": "iota",    "Ι": "Iota",
    "κ": "kappa",   "Κ": "Kappa",
    "λ": "lambda",  "Λ": "Lambda",
    "ν": "nu",      "Ν": "Nu",
    "ξ": "xi",      "Ξ": "Xi",
    "π": "pi",      "Π": "Pi",
    "ρ": "rho",     "Ρ": "Rho",
    "τ": "tau",     "Τ": "Tau",
    "υ": "upsilon", "Υ": "Upsilon",
    "φ": "phi",     "Φ": "Phi",
    "χ": "chi",     "Χ": "Chi",
    "ψ": "psi",     "Ψ": "Psi",
    "ω": "omega",   "Ω": "Omega",
    # Common math symbols
    "∞": "infinity",
    "√": "sqrt",
    "∑": "sum",
    "∏": "product",
    "≈": "approx",
    "≠": "!=",
    "≤": "<=",
    "≥": ">=",
    "±": "+/-",
    "×": "x",
    "÷": "/",
    "\u2192": "->",   # → rightwards arrow
    "\u2190": "<-",   # ← leftwards arrow
    "\u2194": "<->",  # ↔ left right arrow
    "⇒": "=>",
    "∈": "in",
    "∉": "not in",
    "⊂": "subset",
    "∅": "empty_set",
    # Typographic
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2013": "-",  # en dash
    "\u2014": "--", # em dash
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",  # non-breaking space
    "\u200b": "",   # zero-width space
    "\ufeff": "",   # BOM
    "\xad": "",     # soft hyphen
}

# Build a single-pass regex from the replacement map
_UNICODE_PATTERN = re.compile(
    "|".join(re.escape(k) for k in _UNICODE_REPLACEMENTS)
)


def normalise_unicode(text: str) -> str:
    """Replace known Unicode symbols with ASCII equivalents.

    Applies NFC normalisation first, then substitutes Greek letters
    and common mathematical symbols with readable ASCII names.

    Args:
        text: Raw text possibly containing Unicode symbols.

    Returns:
        Text with Unicode symbols replaced.
    """
    text = unicodedata.normalize("NFC", text)
    return _UNICODE_PATTERN.sub(lambda m: _UNICODE_REPLACEMENTS[m.group()], text)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise detection and removal
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 2a. Page-number patterns
# ---------------------------------------------------------------------------

_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)

_ROMAN_PAGE_RE = re.compile(
    r"^\s*(?:i{1,4}|iv|vi{0,4}|ix|xi{0,4}|xiv|xv|xvi{0,4})\s*$",
    re.IGNORECASE,
)


def _is_page_number(line: str) -> bool:
    """Return ``True`` if *line* is a standalone page number."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_PAGE_NUMBER_RE.match(stripped) or _ROMAN_PAGE_RE.match(stripped))


def remove_page_numbers(text: str) -> str:
    """Strip standalone page-number lines from extracted text.

    Args:
        text: Page text that may contain page-number noise.

    Returns:
        Cleaned text with page-number lines removed.
    """
    lines = text.split("\n")
    cleaned = [ln for ln in lines if not _is_page_number(ln)]
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# 2b. Repeated headers / footers (>50 % of pages)
# ---------------------------------------------------------------------------

def detect_repeated_headers_footers(
    pages_text: List[str],
    threshold: float = 0.50,
    max_line_len: int = 200,
) -> set[str]:
    """Identify text lines repeated on more than *threshold* of pages.

    Headers, footers, and watermarks are typically short lines that
    appear identically (or near-identically) on most pages.

    Args:
        pages_text: List of per-page raw text strings.
        threshold: Fraction of pages a line must appear on (default 50 %).
        max_line_len: Ignore lines longer than this (unlikely to be
            headers/footers).

    Returns:
        Set of noise-line strings to remove.
    """
    total_pages = len(pages_text)
    if total_pages < 3:
        return set()

    line_page_counts: Counter[str] = Counter()
    for page in pages_text:
        seen: set[str] = set()
        for line in page.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) <= max_line_len and stripped not in seen:
                seen.add(stripped)
                line_page_counts[stripped] += 1

    min_occurrences = max(2, int(total_pages * threshold))
    noise = {
        line for line, count in line_page_counts.items()
        if count >= min_occurrences
    }

    if noise:
        logger.info(
            "Detected %d repeated header/footer lines (appear on >%d%% of %d pages).",
            len(noise), int(threshold * 100), total_pages,
        )
    return noise


def strip_repeated_noise(text: str, noise_lines: set[str]) -> str:
    """Remove lines identified as repeated headers/footers.

    Args:
        text: Full document or page text.
        noise_lines: Set of exact line strings to strip.

    Returns:
        Cleaned text.
    """
    if not noise_lines:
        return text
    return "\n".join(
        ln for ln in text.split("\n") if ln.strip() not in noise_lines
    )


# ---------------------------------------------------------------------------
# 2c. Table-of-contents detection
# ---------------------------------------------------------------------------

# TOC-like lines: "Chapter 3 ..... 42" or "2.1 Introduction ... 15"
_TOC_ENTRY_RE = re.compile(
    r"^.{3,80}\s*[.\u2022\u00b7·]{3,}\s*\d{1,4}\s*$"
)

# Explicit TOC heading
_TOC_HEADING_RE = re.compile(
    r"^\s*(?:table\s+of\s+contents|contents)\s*$", re.IGNORECASE
)


def is_toc_page(page_text: str, entry_threshold: float = 0.40) -> bool:
    """Detect whether a page is a Table of Contents page.

    A page is considered TOC if:
    - It has an explicit "Table of Contents" heading, OR
    - More than *entry_threshold* of its non-blank lines match the
      dotted-leader + page-number TOC pattern.

    Args:
        page_text: Raw text of a single page.
        entry_threshold: Fraction of lines matching TOC pattern.

    Returns:
        ``True`` if the page is likely a TOC page.
    """
    lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
    if not lines:
        return False

    # Explicit heading check
    for line in lines[:5]:
        if _TOC_HEADING_RE.match(line):
            logger.debug("TOC heading detected: %r", line)
            return True

    # Pattern-ratio check
    toc_count = sum(1 for ln in lines if _TOC_ENTRY_RE.match(ln))
    ratio = toc_count / len(lines)
    if ratio >= entry_threshold:
        logger.debug("TOC page detected (%.0f%% TOC-style lines).", ratio * 100)
        return True

    return False


# ---------------------------------------------------------------------------
# 2d. Index section detection
# ---------------------------------------------------------------------------

# Index entries: "word, 12, 34, 56–58" or "algorithm, 5, 12"
_INDEX_ENTRY_RE = re.compile(
    r"^[A-Za-z][\w\s\-']{1,60},\s*\d[\d,\s\-–—]*$"
)

_INDEX_HEADING_RE = re.compile(
    r"^\s*(?:(?:subject\s+)?index|alphabetical\s+index)\s*$", re.IGNORECASE
)


def is_index_page(page_text: str, entry_threshold: float = 0.40) -> bool:
    """Detect whether a page belongs to a book index.

    Args:
        page_text: Raw text of a single page.
        entry_threshold: Fraction of lines matching index-entry pattern.

    Returns:
        ``True`` if the page is likely an index page.
    """
    lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
    if not lines:
        return False

    for line in lines[:5]:
        if _INDEX_HEADING_RE.match(line):
            logger.debug("Index heading detected: %r", line)
            return True

    idx_count = sum(1 for ln in lines if _INDEX_ENTRY_RE.match(ln))
    ratio = idx_count / len(lines)
    if ratio >= entry_threshold:
        logger.debug("Index page detected (%.0f%% index-style lines).", ratio * 100)
        return True

    return False


# ---------------------------------------------------------------------------
# 2e. Copyright block removal
# ---------------------------------------------------------------------------

_COPYRIGHT_PATTERNS = [
    re.compile(r"(?i)©\s*\d{4}"),
    re.compile(r"(?i)copyright\s+(?:©\s*)?\d{4}"),
    re.compile(r"(?i)all\s+rights\s+reserved"),
    re.compile(r"(?i)no\s+part\s+of\s+this\s+(?:book|publication|work)"),
    re.compile(r"(?i)permission\s+(?:of|from)\s+the\s+publisher"),
    re.compile(r"(?i)printed\s+in\s+(?:the\s+)?(?:united\s+states|usa|u\.s\.a)"),
    re.compile(r"(?i)ISBN[\s:-]?\d"),
    re.compile(r"(?i)library\s+of\s+congress"),
]


def remove_copyright_blocks(text: str, max_block_lines: int = 25) -> str:
    """Remove copyright / legal boilerplate blocks from text.

    Scans for clusters of lines matching common copyright patterns.
    Only removes contiguous blocks of up to *max_block_lines* lines
    to avoid stripping real content that happens to mention "copyright".

    Args:
        text: Page or document text.
        max_block_lines: Maximum size of a copyright block to strip.

    Returns:
        Text with copyright blocks removed.
    """
    lines = text.split("\n")
    # Find lines matching any copyright pattern
    flagged: set[int] = set()
    for i, line in enumerate(lines):
        for pat in _COPYRIGHT_PATTERNS:
            if pat.search(line):
                flagged.add(i)
                break

    if not flagged:
        return text

    # Expand flagged lines into contiguous blocks
    blocks_to_remove: list[tuple[int, int]] = []
    sorted_flags = sorted(flagged)

    block_start = sorted_flags[0]
    block_end = sorted_flags[0]
    for idx in sorted_flags[1:]:
        if idx <= block_end + 3:  # allow up to 2 gap lines within a block
            block_end = idx
        else:
            blocks_to_remove.append((block_start, block_end))
            block_start = idx
            block_end = idx
    blocks_to_remove.append((block_start, block_end))

    # Remove blocks (reverse order to preserve indices)
    remove_indices: set[int] = set()
    for start, end in blocks_to_remove:
        block_len = end - start + 1
        if block_len <= max_block_lines:
            for i in range(start, end + 1):
                remove_indices.add(i)
            logger.debug(
                "Removing copyright block: lines %d–%d (%d lines).",
                start, end, block_len,
            )

    if remove_indices:
        lines = [ln for i, ln in enumerate(lines) if i not in remove_indices]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Text normalisation
# ═══════════════════════════════════════════════════════════════════════════

def fix_hyphenated_breaks(text: str) -> str:
    """Rejoin words split by end-of-line hyphenation.

    Patterns handled:
    - ``"strat-\\negy"`` ``"strategy"``
    - ``"opti-\\n misation"`` ``"optimisation"``

    Does NOT rejoin hyphens that are part of compound words
    (e.g., ``"well-known"`` on one line).

    Args:
        text: Text with possible hyphenated line breaks.

    Returns:
        Text with broken words rejoined.
    """
    # Match a lowercase letter followed by a hyphen at end of line,
    # then optional whitespace and a lowercase letter on the next line.
    return re.sub(
        r"([a-zA-Z])-\s*\n\s*([a-z])",
        r"\1\2",
        text,
    )


def normalise_whitespace(text: str) -> str:
    """Collapse multiple inline spaces while preserving indentation and paragraphs.

    - Tabs 4 spaces.
    - Multiple inline spaces single space (preserving leading indent).
    - 3+ consecutive blank lines 2 (paragraph break preserved).

    Args:
        text: Text to normalise.

    Returns:
        Whitespace-normalised text.
    """
    def _collapse_line(line: str) -> str:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        indent = indent.replace("\t", "    ")
        stripped = re.sub(r"  +", " ", stripped)
        return indent + stripped

    text = "\n".join(_collapse_line(ln) for ln in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_page_text(text: str) -> str:
    """Apply the full normalisation pipeline to a single page's text.

    Order of operations:
    1. Unicode normalisation
    2. Fix hyphenated line breaks
    3. Remove page numbers
    4. Remove copyright blocks
    5. Normalise whitespace

    Args:
        text: Raw extracted text from one PDF page.

    Returns:
        Cleaned, normalised text.
    """
    text = normalise_unicode(text)
    text = fix_hyphenated_breaks(text)
    text = remove_page_numbers(text)
    text = remove_copyright_blocks(text)
    text = normalise_whitespace(text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 4. Code block detection
# ═══════════════════════════════════════════════════════════════════════════

# Python keywords that strongly indicate code
_PYTHON_KEYWORDS_RE = re.compile(
    r"^\s*(?:"
    r"def\s+\w+\s*\(|"
    r"class\s+\w+|"
    r"import\s+\w|from\s+\w+\s+import|"
    r"if\s+.*:|elif\s+.*:|else\s*:|"
    r"for\s+\w.*:|while\s+.*:|"
    r"return\s|yield\s|raise\s|"
    r"try\s*:|except\s|finally\s*:|"
    r"with\s+\w|"
    r"@\w+|"
    r"print\s*\(|"
    r"assert\s+|"
    r"pass\s*$"
    r")"
)

# Assignment, function call, attribute access
_CODE_SYNTAX_RE = re.compile(
    r"^\s*(?:"
    r"[a-zA-Z_]\w*\s*=[^=]|"
    r"[a-zA-Z_]\w*\s*\(|"
    r"[a-zA-Z_]\w*\.\w+|"
    r"#\s*\S"
    r")"
)


def _is_code_line(line: str) -> bool:
    """Return ``True`` if *line* resembles a Python source code line.

    Args:
        line: A single line of text.

    Returns:
        Whether the line matches code-like patterns.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Lines with ≥2 spaces of leading indent + non-uppercase start likely code
    indent = len(line) - len(line.lstrip())
    if indent >= 2 and stripped and not stripped[0].isupper():
        return True
    return bool(_PYTHON_KEYWORDS_RE.match(stripped) or _CODE_SYNTAX_RE.match(stripped))


def _detect_fenced_code_blocks(text: str) -> List[Tuple[int, int]]:
    """Find triple-backtick fenced code blocks.

    Returns a list of ``(start_line, end_line)`` index pairs (inclusive).

    Args:
        text: Full page or document text.

    Returns:
        List of (start, end) line-index tuples for fenced blocks.
    """
    lines = text.split("\n")
    blocks: List[Tuple[int, int]] = []
    fence_start: Optional[int] = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            if fence_start is None:
                fence_start = i
            else:
                blocks.append((fence_start, i))
                fence_start = None

    # Unclosed fence — treat everything from start to end as code
    if fence_start is not None:
        blocks.append((fence_start, len(lines) - 1))

    return blocks


def _detect_indented_code_blocks(
    lines: List[str],
    min_indent: int = 2,
    min_consecutive: int = 3,
) -> List[Tuple[int, int]]:
    """Find blocks of consistently indented lines (likely code).

    Requires at least *min_consecutive* consecutive indented lines to
    qualify as a code block.

    Args:
        lines: List of text lines.
        min_indent: Minimum leading-space count to consider "indented".
        min_consecutive: Minimum consecutive indented lines to form a block.

    Returns:
        List of (start, end) line-index tuples.
    """
    blocks: List[Tuple[int, int]] = []
    block_start: Optional[int] = None
    count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if indent >= min_indent and stripped:
            if block_start is None:
                block_start = i
            count += 1
        elif not stripped:
            # Blank lines inside a potential block — keep going
            if block_start is not None:
                count += 0  # don't increment, but don't break
        else:
            # Non-indented, non-blank line — flush block
            if block_start is not None and count >= min_consecutive:
                blocks.append((block_start, i - 1))
            block_start = None
            count = 0

    if block_start is not None and count >= min_consecutive:
        blocks.append((block_start, len(lines) - 1))

    return blocks


def _detect_keyword_code_blocks(
    lines: List[str],
    threshold: float = 0.40,
    window_size: int = 5,
) -> List[Tuple[int, int]]:
    """Find blocks where Python keywords are densely concentrated.

    Slides a window of *window_size* lines; if ≥ *threshold* fraction
    of non-blank lines in the window match Python keyword patterns,
    the window is marked as code.

    Args:
        lines: List of text lines.
        threshold: Fraction of code-like lines required.
        window_size: Sliding window size in lines.

    Returns:
        List of merged (start, end) line-index tuples.
    """
    n = len(lines)
    if n < window_size:
        # Check entire block
        non_empty = [ln for ln in lines if ln.strip()]
        if non_empty:
            code_ratio = sum(1 for ln in non_empty if _is_code_line(ln)) / len(non_empty)
            if code_ratio >= threshold:
                return [(0, n - 1)]
        return []

    is_code_arr = [False] * n
    for i in range(n - window_size + 1):
        window = lines[i : i + window_size]
        non_empty = [ln for ln in window if ln.strip()]
        if not non_empty:
            continue
        code_count = sum(1 for ln in non_empty if _is_code_line(ln))
        if code_count / len(non_empty) >= threshold:
            for j in range(i, i + window_size):
                is_code_arr[j] = True

    # Merge consecutive True ranges
    blocks: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(is_code_arr):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, n - 1))

    return blocks


def _merge_ranges(
    ranges: List[Tuple[int, int]], gap: int = 2
) -> List[Tuple[int, int]]:
    """Merge overlapping or adjacent line-index ranges.

    Args:
        ranges: List of (start, end) tuples (inclusive).
        gap: Maximum gap between ranges to still merge them.

    Returns:
        Merged, sorted list of (start, end) tuples.
    """
    if not ranges:
        return []
    sorted_r = sorted(ranges)
    merged = [sorted_r[0]]
    for start, end in sorted_r[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _snap_code_block_boundaries(
    lines: List[str], code_ranges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Adjust code block boundaries so functions are not split mid-body.

    Expands each range downward until indentation returns to zero or
    the block clearly ends (blank-line gap ≥ 3).

    Args:
        lines: All text lines.
        code_ranges: Initial (start, end) ranges.

    Returns:
        Adjusted (start, end) ranges.
    """
    adjusted: List[Tuple[int, int]] = []
    n = len(lines)

    for start, end in code_ranges:
        # Expand downward to avoid splitting a function
        i = end + 1
        blank_gap = 0
        while i < n:
            stripped = lines[i].strip()
            if not stripped:
                blank_gap += 1
                if blank_gap >= 3:
                    break
                i += 1
                continue
            blank_gap = 0
            indent = len(lines[i]) - len(lines[i].lstrip())
            if indent >= 2 or _is_code_line(lines[i]):
                end = i
                i += 1
            else:
                break
        adjusted.append((start, end))

    return _merge_ranges(adjusted)


def classify_blocks(page_text: str) -> List[dict]:
    """Split page text into code and text blocks.

    Detection strategy (applied in priority order):
    1. Triple-backtick fenced blocks.
    2. Consistently-indented blocks (≥3 consecutive indented lines).
    3. Python-keyword density (sliding window heuristic).

    All three detectors are combined and merged.  Lines not in any
    detected code range are grouped as ``"text"`` blocks.

    Args:
        page_text: Cleaned text of a single PDF page.

    Returns:
        List of dicts with keys ``"block_type"`` and ``"content"``.
    """
    if not page_text.strip():
        return []

    lines = page_text.split("\n")
    n = len(lines)

    # Collect code ranges from all three detectors
    fenced = _detect_fenced_code_blocks(page_text)
    indented = _detect_indented_code_blocks(lines)
    keyword = _detect_keyword_code_blocks(lines)

    all_code_ranges = _merge_ranges(fenced + indented + keyword)
    all_code_ranges = _snap_code_block_boundaries(lines, all_code_ranges)

    # Build set of line indices that are code
    code_lines_set: set[int] = set()
    for start, end in all_code_ranges:
        for i in range(start, min(end + 1, n)):
            code_lines_set.add(i)

    # Group consecutive lines by type
    blocks: List[dict] = []
    current_type: Optional[str] = None
    current_lines: List[str] = []

    for i, line in enumerate(lines):
        line_type = "code" if i in code_lines_set else "text"

        if line_type != current_type:
            if current_lines and current_type is not None:
                content = "\n".join(current_lines)
                # Strip fenced markers from code content
                if current_type == "code":
                    content = _strip_fence_markers(content)
                content = content.strip()
                if content:
                    blocks.append({
                        "block_type": current_type,
                        "content": content,
                    })
            current_type = line_type
            current_lines = [line]
        else:
            current_lines.append(line)

    # Flush last block
    if current_lines and current_type is not None:
        content = "\n".join(current_lines)
        if current_type == "code":
            content = _strip_fence_markers(content)
        content = content.strip()
        if content:
            blocks.append({
                "block_type": current_type,
                "content": content,
            })

    return blocks


def _strip_fence_markers(text: str) -> str:
    """Remove triple-backtick fence lines from code block text.

    Args:
        text: Code block text that may include ``` markers.

    Returns:
        Code text without fence markers.
    """
    lines = text.split("\n")
    cleaned = [
        ln for ln in lines
        if not ln.strip().startswith("```")
    ]
    return "\n".join(cleaned)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Layout-aware PDF extraction (PyMuPDF)
# ═══════════════════════════════════════════════════════════════════════════

_MONO_FONTS = frozenset({
    "courier", "couriernew", "courierstd", "courierstdoblique",
    "courierstdbold", "courierstdboldoblique",
    "consolas", "menlo", "inconsolata", "dejavusansmono",
    "liberationmono", "sourcecodepro", "firacode", "ubuntumono",
    "robotomono", "notomono",
})


def _is_monospace(font_name: str) -> bool:
    """Return ``True`` if *font_name* is a known monospace/code font.

    Args:
        font_name: Font name string from PyMuPDF span metadata.

    Returns:
        Whether the font is monospace.
    """
    normalised = font_name.lower().replace("-", "").replace(" ", "")
    return normalised in _MONO_FONTS or "courier" in font_name.lower() or "mono" in font_name.lower()


def _extract_page_layout(page) -> str:
    """Extract text from a single PyMuPDF page preserving code indentation.

    For monospace-font blocks, indentation is reconstructed from the
    x-coordinate of each span relative to the leftmost code position.
    Prose blocks use plain concatenation.

    Args:
        page: A ``fitz.Page`` object.

    Returns:
        Extracted text with code indentation preserved.
    """
    data = page.get_text("dict", flags=0)
    blocks = data.get("blocks", [])
    output_parts: list[str] = []

    for block in blocks:
        if block.get("type") != 0:  # text blocks only
            continue

        lines = block.get("lines", [])
        if not lines:
            continue

        # Determine dominant font for the block
        font_counts: dict[str, int] = {}
        for line in lines:
            for span in line.get("spans", []):
                fn = span.get("font", "")
                font_counts[fn] = font_counts.get(fn, 0) + len(span.get("text", ""))
        dominant_font = max(font_counts, key=font_counts.get) if font_counts else ""
        is_code_font = _is_monospace(dominant_font)

        if is_code_font:
            # Reconstruct code indentation from x-positions
            min_x = float("inf")
            for line in lines:
                spans = line.get("spans", [])
                if spans:
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if text:
                        min_x = min(min_x, spans[0]["bbox"][0])
            if min_x == float("inf"):
                min_x = 0.0

            # Estimate monospace character width
            char_widths: list[float] = []
            for line in lines:
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if txt and _is_monospace(span.get("font", "")):
                        bbox = span["bbox"]
                        w = bbox[2] - bbox[0]
                        if len(txt) > 0 and w > 0:
                            char_widths.append(w / len(txt))
            char_w = sum(char_widths) / len(char_widths) if char_widths else 5.0

            code_lines: list[str] = []
            for line in lines:
                spans = line.get("spans", [])
                if not spans:
                    code_lines.append("")
                    continue
                x0 = spans[0]["bbox"][0]
                indent_px = max(0.0, x0 - min_x)
                indent_spaces = round(indent_px / char_w)
                indent_spaces = (indent_spaces // 4) * 4  # snap to Python style
                full_text = "".join(s.get("text", "") for s in spans)
                code_lines.append(" " * indent_spaces + full_text.strip())
            output_parts.append("\n".join(code_lines))
        else:
            # Prose block: plain concatenation
            prose_lines: list[str] = []
            for line in lines:
                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans)
                prose_lines.append(line_text)
            output_parts.append("\n".join(prose_lines))

    return "\n".join(output_parts)


def extract_text_by_page(pdf_path: str) -> Tuple[List[str], Dict[str, Any]]:
    """Extract text from each page of a PDF individually.

    Uses layout-aware extraction that preserves code indentation
    by analysing x-coordinates of monospace font spans.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (list_of_page_texts, file_metadata).
    """
    import fitz  # PyMuPDF

    logger.info("Opening PDF: %s", pdf_path)
    doc = fitz.open(pdf_path)
    pages_text: List[str] = []

    for page_num, page in enumerate(doc):
        logger.debug("Extracting page %d/%d.", page_num + 1, len(doc))
        pages_text.append(_extract_page_layout(page))

    metadata: Dict[str, Any] = {
        "source": os.path.basename(pdf_path),
        "full_path": str(Path(pdf_path).resolve()),
        "page_count": len(doc),
        "title": doc.metadata.get("title", "") or os.path.basename(pdf_path),
        "author": doc.metadata.get("author", ""),
        "creation_date": doc.metadata.get("creationDate", ""),
        "subject": doc.metadata.get("subject", ""),
        "keywords": doc.metadata.get("keywords", ""),
    }
    doc.close()
    logger.info(
        "Extracted %d pages from %s.", metadata["page_count"], metadata["source"],
    )
    return pages_text, metadata


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract all text from a PDF file (convenience wrapper).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (full_text, metadata_dict).
    """
    pages_text, metadata = extract_text_by_page(pdf_path)
    return "\n\n".join(pages_text), metadata


# ═══════════════════════════════════════════════════════════════════════════
# 6. Structured ingestion pipeline  (primary public API)
# ═══════════════════════════════════════════════════════════════════════════

def ingest_pdf_structured(pdf_path: str) -> List[StructuredBlock]:
    """Ingest a PDF and return structured, cleaned blocks.

    This is the **primary entry-point** of the module.  It runs the
    full extraction cleaning classification pipeline and produces
    output ready for ChromaDB storage.

    Pipeline steps per page:
    1. Layout-aware text extraction (PyMuPDF).
    2. Skip TOC and index pages.
    3. Strip repeated headers/footers (>50 % page threshold).
    4. Apply full normalisation (unicode, hyphens, whitespace, copyright).
    5. Classify into code vs. text blocks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of ``StructuredBlock`` dicts::

            [
                {"page": 1,  "block_type": "text", "content": "..."},
                {"page": 2,  "block_type": "code", "content": "def foo():..."},
                ...
            ]

        Code indentation is preserved.  No code block is split
        mid-function.  Header/footer noise is removed.
    """
    logger.info("Starting structured ingestion: %s", pdf_path)

    # Step 1: Extract text page-by-page
    pages_text, metadata = extract_text_by_page(pdf_path)
    total_pages = len(pages_text)
    logger.info("Extracted %d pages from '%s'.", total_pages, metadata["source"])

    if not pages_text or all(not p.strip() for p in pages_text):
        logger.warning("PDF contains no extractable text: %s", pdf_path)
        return []

    # Step 2: Detect repeated headers/footers across all pages
    noise_lines = detect_repeated_headers_footers(pages_text, threshold=0.50)

    # Step 3: Process each page
    results: List[StructuredBlock] = []
    toc_done = False  # TOC pages are typically contiguous at the start
    index_started = False  # Index pages are typically at the end

    for page_num, raw_text in enumerate(pages_text, start=1):
        # 3a. Skip TOC pages
        if not toc_done:
            if is_toc_page(raw_text):
                logger.debug("Skipping TOC page %d.", page_num)
                continue
            elif page_num > 1:
                toc_done = True # first non-TOC page after start TOC section is over
        else:
            toc_done = True

        # 3b. Skip index pages (once started, assume rest are index)
        if index_started or is_index_page(raw_text):
            if not index_started:
                logger.debug("Index section begins at page %d.", page_num)
            index_started = True
            continue

        # 3c. Strip repeated header/footer noise
        cleaned = strip_repeated_noise(raw_text, noise_lines)

        # 3d. Full text normalisation
        cleaned = clean_page_text(cleaned)

        if not cleaned.strip():
            logger.debug("Page %d is empty after cleaning — skipped.", page_num)
            continue

        # 3e. Classify into code / text blocks
        blocks = classify_blocks(cleaned)

        for block in blocks:
            results.append(StructuredBlock(
                page=page_num,
                block_type=block["block_type"],
                content=block["content"],
            ))

    logger.info(
        "Structured ingestion complete: %d blocks from %d pages (%s).",
        len(results), total_pages, metadata["source"],
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 7. Text chunking helpers (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════

_WORD_TOKEN_RE = re.compile(r"\S+")


def _approx_token_count(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(_WORD_TOKEN_RE.findall(text))


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"'\(\[])")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*\s+\S|"
    r"[A-Z][A-Z\s]{2,80}$|"
    r"[A-Z][a-zA-Z\s]{2,80}:$"
    r")",
    re.MULTILINE,
)

_SNIPPET_LABEL_RE = re.compile(
    r"(?i)(?:snippet|code|listing|algorithm|example|figure)\s*[\d.]+",
)


def _is_code_block(lines: List[str], threshold: float = 0.4) -> bool:
    """Return ``True`` if ≥ *threshold* fraction of non-empty lines look like code."""
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) < 2:
        return False
    code_count = sum(1 for ln in non_empty if _is_code_line(ln))
    return code_count / len(non_empty) >= threshold


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-ish segments."""
    parts = _SENTENCE_END.split(text)
    return [p for p in parts if p.strip()]


def _split_paragraphs(text: str) -> List[str]:
    """Split text on paragraph boundaries, keeping code blocks intact."""
    lines = text.split("\n")
    segments: List[str] = []
    current_segment: List[str] = []
    in_code_block = False
    blank_count = 0

    for line in lines:
        stripped = line.strip()
        is_blank = not stripped

        if is_blank:
            blank_count += 1
            if in_code_block:
                if blank_count <= 2:
                    current_segment.append(line)
                    continue
                else:
                    in_code_block = False
            if blank_count >= 2 and current_segment:
                segments.append("\n".join(current_segment))
                current_segment = []
                blank_count = 0
            continue

        blank_count = 0
        if _is_code_line(line):
            in_code_block = True
        elif in_code_block and len(line) - len(line.lstrip()) >= 2:
            pass
        elif in_code_block and not _is_code_line(line) and len(line) - len(line.lstrip()) < 2:
            if _SNIPPET_LABEL_RE.search(stripped):
                pass
            else:
                in_code_block = False
        current_segment.append(line)

    if current_segment:
        seg_text = "\n".join(current_segment).strip()
        if seg_text:
            segments.append(seg_text)

    return [s.strip() for s in segments if s.strip()]


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove common PDF noise (legacy helper).

    Preserves leading whitespace (indentation) for code blocks.
    """
    text = fix_hyphenated_breaks(text)
    text = remove_page_numbers(text)
    text = normalise_whitespace(text)
    return text


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    unit: str = "token",
    max_tokens: int = 800,
) -> List[str]:
    """Split *text* into overlapping chunks.

    Supports two modes:
    - ``"token"`` — chunk boundaries measured in approximate whitespace tokens.
    - ``"char"`` — legacy character-count mode.

    Args:
        text: Text to chunk.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between consecutive chunks.
        unit: ``"token"`` or ``"char"``.
        max_tokens: Hard ceiling for token mode.

    Returns:
        List of non-empty chunk strings.
    """
    text = _clean_text(text)
    if not text:
        return []
    if unit == "char":
        return _chunk_by_chars(text, chunk_size, chunk_overlap)
    return _chunk_by_tokens(text, chunk_size, chunk_overlap, max_tokens)


def _chunk_by_tokens(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
) -> List[str]:
    """Token-aware chunking with paragraph- and sentence-boundary snapping."""
    paragraphs = _split_paragraphs(text)
    segments: List[str] = []
    segment_is_code: List[bool] = []

    for para in paragraphs:
        lines = para.split("\n")
        if _is_code_block(lines):
            segments.append(para)
            segment_is_code.append(True)
        else:
            sents = _split_sentences(para)
            if sents:
                segments.extend(sents)
                segment_is_code.extend([False] * len(sents))
            elif para.strip():
                segments.append(para)
                segment_is_code.append(False)

    if not segments:
        return [text] if text.strip() else []

    def _join_segments(segs: List[str], seg_codes: List[bool]) -> str:
        if not segs:
            return ""
        parts: List[str] = [segs[0]]
        for i in range(1, len(segs)):
            if (i - 1 < len(seg_codes) and seg_codes[i - 1]) or \
               (i < len(seg_codes) and seg_codes[i]):
                parts.append("\n" + segs[i])
            else:
                parts.append(" " + segs[i])
        return "".join(parts)

    chunks: List[str] = []
    current: List[str] = []
    current_codes: List[bool] = []
    current_len = 0

    for seg_idx, segment in enumerate(segments):
        is_code = seg_idx < len(segment_is_code) and segment_is_code[seg_idx]
        s_tokens = _approx_token_count(segment)

        if s_tokens > max_tokens:
            if current:
                chunks.append(_join_segments(current, current_codes))
                current, current_codes, current_len = [], [], 0
            if is_code:
                code_lines = segment.split("\n")
                buf_lines: List[str] = []
                buf_len = 0
                for cl in code_lines:
                    cl_tokens = _approx_token_count(cl)
                    if buf_len + cl_tokens > target_tokens and buf_lines:
                        chunks.append("\n".join(buf_lines))
                        avg = max(1, buf_len // max(1, len(buf_lines)))
                        overlap_lines = buf_lines[-(overlap_tokens // avg):]
                        buf_lines = list(overlap_lines)
                        buf_len = sum(_approx_token_count(l) for l in buf_lines)
                    buf_lines.append(cl)
                    buf_len += cl_tokens
                if buf_lines:
                    chunks.append("\n".join(buf_lines))
            else:
                words = segment.split()
                buf: List[str] = []
                buf_len = 0
                for w in words:
                    buf.append(w)
                    buf_len += 1
                    if buf_len >= target_tokens:
                        chunks.append(" ".join(buf))
                        overlap_words = buf[-overlap_tokens:] if overlap_tokens < len(buf) else buf[:]
                        buf = list(overlap_words)
                        buf_len = len(buf)
                if buf:
                    chunks.append(" ".join(buf))
            continue

        if current_len + s_tokens > target_tokens and current:
            chunks.append(_join_segments(current, current_codes))
            overlap_buf: List[str] = []
            overlap_codes_buf: List[bool] = []
            overlap_count = 0
            for oi in range(len(current) - 1, -1, -1):
                t = _approx_token_count(current[oi])
                if overlap_count + t > overlap_tokens:
                    break
                overlap_buf.insert(0, current[oi])
                overlap_codes_buf.insert(0, current_codes[oi] if oi < len(current_codes) else False)
                overlap_count += t
            current = overlap_buf
            current_codes = overlap_codes_buf
            current_len = overlap_count

        current.append(segment)
        current_codes.append(is_code)
        current_len += s_tokens

    if current:
        chunks.append(_join_segments(current, current_codes))

    return [c.strip() for c in chunks if c.strip()]


def _chunk_by_chars(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Legacy character-count chunking (sentence-boundary aware)."""
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            lookback = text[start:end]
            for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
                last = lookback.rfind(sep)
                if last != -1 and last > chunk_size // 4:
                    end = start + last + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end - chunk_overlap)
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 8. Structural document parsing (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StructuralChunk:
    """A chunk with structural metadata from document parsing."""
    text: str
    chapter: str = ""
    chapter_title: str = ""
    section_id: str = ""
    section_title: str = ""
    snippet_id: str = ""
    snippet_title: str = ""
    is_code: bool = False


_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:chapter|ch\.?)\s+(\d+)[\s.:—\-]*(.*)",
    re.IGNORECASE,
)

_NUMBERED_SECTION_RE = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)*)\s+([A-Z][a-zA-Z\s,'\.\-]{2,100})\s*$",
)

_BARE_SECTION_NUM_RE = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)*)\s*$",
)

_TOP_LEVEL_SECTION_RE = re.compile(
    r"^\s*(\d{1,2})\s+([A-Z][a-zA-Z\s,'\-]{3,80})\s*$",
)

_SNIPPET_BLOCK_RE = re.compile(
    r"^\s*(?:snippet|code\s*(?:listing|example)?|listing|algorithm)\s+"
    r"([\d.]+)[\s.:—\-]*(.*)",
    re.IGNORECASE,
)


def _parse_structural_segments(text: str) -> List[StructuralChunk]:
    """Parse document text into structural segments with metadata."""
    lines = text.split("\n")
    segments: List[StructuralChunk] = []

    cur_chapter = ""
    cur_chapter_title = ""
    cur_section_id = ""
    cur_section_title = ""
    cur_snippet_id = ""
    cur_snippet_title = ""
    current_lines: List[str] = []

    def _flush():
        nonlocal cur_snippet_id, cur_snippet_title
        if not current_lines:
            return
        seg_text = "\n".join(current_lines).strip()
        if not seg_text:
            current_lines.clear()
            return
        seg_lines_split = seg_text.split("\n")
        segments.append(StructuralChunk(
            text=seg_text,
            chapter=cur_chapter,
            chapter_title=cur_chapter_title,
            section_id=cur_section_id,
            section_title=cur_section_title,
            snippet_id=cur_snippet_id,
            snippet_title=cur_snippet_title,
            is_code=_is_code_block(seg_lines_split),
        ))
        current_lines.clear()
        cur_snippet_id = ""
        cur_snippet_title = ""

    for line in lines:
        stripped = line.strip()

        m = _CHAPTER_HEADING_RE.match(stripped)
        if m and len(stripped) < 120:
            _flush()
            cur_chapter = m.group(1)
            cur_chapter_title = m.group(2).strip()
            cur_section_id = ""
            cur_section_title = ""
            current_lines.append(line)
            continue

        m = _SNIPPET_BLOCK_RE.match(stripped)
        if m:
            _flush()
            cur_snippet_id = m.group(1)
            cur_snippet_title = m.group(2).strip()
            current_lines.append(line)
            continue

        m = _NUMBERED_SECTION_RE.match(stripped)
        if m and len(stripped) < 120:
            _flush()
            cur_section_id = m.group(1)
            cur_section_title = m.group(2).strip()
            current_lines.append(line)
            continue

        m = _BARE_SECTION_NUM_RE.match(stripped)
        if m:
            _flush()
            cur_section_id = m.group(1)
            cur_section_title = ""
            current_lines.append(line)
            continue

        if not cur_chapter:
            m = _TOP_LEVEL_SECTION_RE.match(stripped)
            if m:
                _flush()
                cur_section_id = m.group(1)
                cur_section_title = m.group(2).strip()
                current_lines.append(line)
                continue

        current_lines.append(line)

    _flush()
    return segments


def structure_aware_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    max_tokens: int = 800,
) -> List[StructuralChunk]:
    """Split text into chunks that respect document structure.

    Args:
        text: Full document text.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Token overlap between consecutive chunks.
        max_tokens: Hard ceiling per chunk.

    Returns:
        List of ``StructuralChunk`` objects with structural metadata.
    """
    text = _clean_text(text)
    if not text:
        return []

    segments = _parse_structural_segments(text)
    if not segments:
        basic_chunks = _chunk_by_tokens(text, chunk_size, chunk_overlap, max_tokens)
        return [StructuralChunk(text=c, is_code=_is_code_block(c.split("\n"))) for c in basic_chunks]

    result: List[StructuralChunk] = []
    buffer_texts: List[str] = []
    buffer_tokens: int = 0
    buffer_meta: Optional[StructuralChunk] = None

    def _make_chunk(combined_text: str, meta: StructuralChunk) -> StructuralChunk:
        lines_split = combined_text.split("\n")
        return StructuralChunk(
            text=combined_text,
            chapter=meta.chapter,
            chapter_title=meta.chapter_title,
            section_id=meta.section_id,
            section_title=meta.section_title,
            snippet_id=meta.snippet_id,
            snippet_title=meta.snippet_title,
            is_code=_is_code_block(lines_split),
        )

    def _flush_buffer():
        nonlocal buffer_texts, buffer_tokens, buffer_meta
        if buffer_texts and buffer_meta:
            combined = "\n\n".join(buffer_texts)
            if combined.strip():
                result.append(_make_chunk(combined, buffer_meta))
        buffer_texts = []
        buffer_tokens = 0
        buffer_meta = None

    for seg in segments:
        seg_tokens = _approx_token_count(seg.text)

        if seg.snippet_id or seg.is_code:
            _flush_buffer()
            if seg_tokens <= max_tokens:
                result.append(seg)
            else:
                code_lines = seg.text.split("\n")
                buf_lines: List[str] = []
                buf_len = 0
                for cl in code_lines:
                    cl_tokens = _approx_token_count(cl)
                    if buf_len + cl_tokens > chunk_size and buf_lines:
                        result.append(StructuralChunk(
                            text="\n".join(buf_lines),
                            chapter=seg.chapter, chapter_title=seg.chapter_title,
                            section_id=seg.section_id, section_title=seg.section_title,
                            snippet_id=seg.snippet_id, snippet_title=seg.snippet_title,
                            is_code=True,
                        ))
                        avg_tpl = max(1, buf_len // max(1, len(buf_lines)))
                        keep = max(1, chunk_overlap // avg_tpl)
                        buf_lines = buf_lines[-keep:]
                        buf_len = sum(_approx_token_count(l) for l in buf_lines)
                    buf_lines.append(cl)
                    buf_len += cl_tokens
                if buf_lines:
                    result.append(StructuralChunk(
                        text="\n".join(buf_lines),
                        chapter=seg.chapter, chapter_title=seg.chapter_title,
                        section_id=seg.section_id, section_title=seg.section_title,
                        snippet_id=seg.snippet_id, snippet_title=seg.snippet_title,
                        is_code=True,
                    ))
            continue

        is_new_boundary = (
            buffer_meta is not None and (
                seg.chapter != buffer_meta.chapter
                or seg.section_id != buffer_meta.section_id
            )
        )
        if is_new_boundary:
            _flush_buffer()

        if seg_tokens > chunk_size:
            _flush_buffer()
            sub_chunks = _chunk_by_tokens(seg.text, chunk_size, chunk_overlap, max_tokens)
            for sub in sub_chunks:
                result.append(StructuralChunk(
                    text=sub,
                    chapter=seg.chapter, chapter_title=seg.chapter_title,
                    section_id=seg.section_id, section_title=seg.section_title,
                    snippet_id="", snippet_title="",
                    is_code=_is_code_block(sub.split("\n")),
                ))
            continue

        if buffer_tokens + seg_tokens > chunk_size and buffer_texts:
            _flush_buffer()

        buffer_texts.append(seg.text)
        buffer_tokens += seg_tokens
        if buffer_meta is None:
            buffer_meta = seg

    _flush_buffer()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 9. Section heading detection
# ═══════════════════════════════════════════════════════════════════════════

def _detect_section_heading(text: str) -> str:
    """Attempt to detect the section heading from the start of a chunk.

    Args:
        text: Chunk text to inspect.

    Returns:
        Detected heading string, or empty string if none found.
    """
    lines = text.strip().split("\n", 5)
    for line in lines[:3]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+(\.\d+)*\s+\S", stripped) and len(stripped) < 120:
            return stripped
        if stripped.isupper() and 3 < len(stripped) < 120:
            return stripped
        if stripped.istitle() and len(stripped) < 100:
            return stripped
        if stripped.endswith(":") and len(stripped) < 100:
            return stripped.rstrip(":")
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# 10. PDFIngestionService — full pipeline (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════

class PDFIngestionService:
    """End-to-end PDF ChromaDB ingestion.

    Usage::

        svc = PDFIngestionService(vector_store, config)
        stats = svc.ingest_pdf("/path/to/file.pdf")
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        config: Optional[RAGConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        on_change_callback: Optional[Any] = None,
    ) -> None:
        self._vs = vector_store
        self._config = config or RAGConfig()
        self._embedder = embedding_service or EmbeddingService(self._config)
        self._on_change = on_change_callback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_file_hash(pdf_path: str) -> str:
        """Compute a SHA-256 hash of the file contents for dedup."""
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 16), b""):
                h.update(block)
        return h.hexdigest()

    def _is_already_ingested(self, source_name: str, file_hash: str) -> bool:
        """Return ``True`` if the same content hash is already stored."""
        stored_hash = self._vs.get_source_file_hash(source_name)
        if stored_hash and stored_hash == file_hash:
            stored_count = self._vs.get_source_chunk_count(source_name)
            if stored_count > 0:
                logger.info(
                    "Document '%s' already ingested (%d chunks, hash=%s). Skipping.",
                    source_name, stored_count, file_hash[:12],
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_pdf(
        self,
        pdf_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """Parse, chunk, embed, and store a single PDF.

        Uses the new cleaning / normalisation pipeline (noise removal,
        unicode normalisation, code-block preservation) before the
        structure-aware chunking and embedding steps.

        Args:
            pdf_path: Path to the PDF file.
            extra_metadata: Additional metadata to attach to all chunks.
            force: Bypass duplicate check and re-ingest.
            progress_callback: Optional callback(stage_name, pct) for UI updates.

        Returns:
            Summary dict with ingestion statistics.
        """
        def _progress(stage: str, pct: float) -> None:
            if progress_callback:
                try:
                    progress_callback(stage, pct)
                except Exception:
                    pass  # UI callback errors should not abort ingestion

        logger.info("Ingesting PDF: %s (force=%s)", pdf_path, force)
        ingest_ts = datetime.now(timezone.utc).isoformat()
        _progress("Checking duplicates…", 0.05)

        # ---- Duplicate detection ----
        source_name = os.path.basename(pdf_path)
        file_hash = self._compute_file_hash(pdf_path)

        if not force and self._is_already_ingested(source_name, file_hash):
            existing_chunks = self._vs.get_source_chunk_count(source_name)
            return {
                "status": "skipped",
                "reason": "already_ingested",
                "source": source_name,
                "chunks": existing_chunks,
                "file_hash": file_hash[:12],
            }

        if self._vs.source_exists(source_name):
            logger.info("Source '%s' exists with different content — replacing.", source_name)
            self.delete_source(source_name)

        # 1. Extract text page-by-page
        _progress("Extracting text…", 0.10)
        pages_text, file_meta = extract_text_by_page(pdf_path)

        # 2. Detect and strip repeated headers/footers
        _progress("Detecting headers/footers…", 0.20)
        noise_lines = detect_repeated_headers_footers(pages_text, threshold=0.50)

        # 3. Clean each page and reassemble
        _progress("Cleaning pages…", 0.30)
        cleaned_pages: List[str] = []
        for pg_num, raw in enumerate(pages_text, start=1):
            # Skip TOC and index pages
            if is_toc_page(raw) or is_index_page(raw):
                logger.debug("Skipping noise page %d (TOC/Index).", pg_num)
                continue
            cleaned = strip_repeated_noise(raw, noise_lines)
            cleaned = clean_page_text(cleaned)
            if cleaned.strip():
                cleaned_pages.append(cleaned)

        full_text = "\n\n".join(cleaned_pages)
        if not full_text.strip():
            logger.warning("PDF contains no extractable text: %s", pdf_path)
            return {"status": "skipped", "reason": "empty", "source": pdf_path}

        # 4. Build page offset index for chunkpage mapping
        _progress("Building page index…", 0.40)
        page_offsets: List[int] = []
        offset = 0
        for page_text in cleaned_pages:
            page_offsets.append(offset)
            offset += len(page_text) + 2

        def _find_page_for_chunk(chunk_text: str) -> int:
            pos = full_text.find(chunk_text[:120])
            if pos < 0:
                return 1
            for i in range(len(page_offsets) - 1, -1, -1):
                if pos >= page_offsets[i]:
                    return i + 1
            return 1

        # 5. Structure-aware chunking
        _progress("Chunking text…", 0.50)
        structural_chunks = structure_aware_chunk(
            full_text,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            max_tokens=self._config.chunk_max_tokens,
        )
        all_chunks = [sc.text for sc in structural_chunks]
        chunk_page_map = [_find_page_for_chunk(c) for c in all_chunks]

        if not all_chunks:
            logger.warning("No chunks produced from PDF: %s", pdf_path)
            return {"status": "skipped", "reason": "no_chunks", "source": pdf_path}

        # 6. Build IDs & enriched metadata
        _progress("Building metadata…", 0.60)
        source_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        filtered_chunks: List[str] = []
        filtered_pages: List[int] = []
        min_words = self._config.chunk_min_tokens

        for idx, chunk in enumerate(all_chunks):
            word_count = len(chunk.split())
            token_count = _approx_token_count(chunk)
            if token_count < min_words:
                logger.debug(
                    "Skipping short chunk %d (%d words) from %s",
                    idx, word_count, file_meta["source"],
                )
                continue

            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
            chunk_id = f"{source_hash}_{idx:04d}"
            section = _detect_section_heading(chunk)
            struct = structural_chunks[idx]

            meta: Dict[str, Any] = {
                "source": file_meta["source"],
                "title": file_meta["title"],
                "author": file_meta["author"],
                "file_hash": file_hash,
                "page_number": chunk_page_map[idx],
                "page_count": file_meta["page_count"],
                "chunk_index": idx,
                "total_chunks": len(all_chunks),
                "chunk_hash": chunk_hash,
                "word_count": word_count,
                "token_count": token_count,
                "char_count": len(chunk),
                "section": section,
                "chapter": struct.chapter,
                "chapter_title": struct.chapter_title,
                "section_id": struct.section_id,
                "section_title": struct.section_title,
                "snippet_id": struct.snippet_id,
                "snippet_title": struct.snippet_title,
                "contains_code": struct.is_code or _is_code_block(chunk.split("\n")),
                "ingested_at": ingest_ts,
                "subject": file_meta.get("subject", ""),
                "keywords": file_meta.get("keywords", ""),
                "space_id": self._config.default_space_id,
                "doc_type": self._config.default_doc_type,
                "doc_version": self._config.default_doc_version,
            }
            if extra_metadata:
                meta.update(extra_metadata)

            ids.append(chunk_id)
            metadatas.append(meta)
            filtered_chunks.append(chunk)
            filtered_pages.append(chunk_page_map[idx])

        if not filtered_chunks:
            logger.warning("All chunks filtered out for PDF: %s", pdf_path)
            return {"status": "skipped", "reason": "all_filtered", "source": pdf_path}

        # 7. Embed (normalise code chunks for embedding, store originals)
        _progress(f"Embedding {len(filtered_chunks)} chunks…", 0.70)

        embed_input: list[str] = []
        for i, chunk_text in enumerate(filtered_chunks):
            if metadatas[i].get("contains_code") or metadatas[i].get("chunk_type") == "code":
                embed_input.append(_normalize_for_embedding(chunk_text))
            else:
                embed_input.append(chunk_text)
        embeddings = self._embedder.embed_texts(embed_input)

        # 8. Store (original text preserved — normalised form used only
        #    for embedding vector computation above)
        _progress("Storing in vector DB…", 0.90)
        self._vs.add_documents(
            ids=ids,
            documents=filtered_chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        stats = {
            "status": "success",
            "source": file_meta["source"],
            "pages": file_meta["page_count"],
            "chunks": len(filtered_chunks),
            "skipped_short": len(all_chunks) - len(filtered_chunks),
            "collection_total": self._vs.count(),
        }
        _progress("Complete", 1.0)
        logger.info("Ingestion complete: %s", stats)
        self._notify_change(file_meta["source"], "ingested")
        return stats

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Ingest all PDFs in a directory.

        Args:
            directory: Path to the directory.
            recursive: Search subdirectories.

        Returns:
            List of per-file ingestion stats.
        """
        pattern = "**/*.pdf" if recursive else "*.pdf"
        results: List[Dict[str, Any]] = []
        for pdf_file in Path(directory).glob(pattern):
            result = self.ingest_pdf(str(pdf_file))
            results.append(result)
        return results

    def reingest_all(self) -> List[Dict[str, Any]]:
        """Delete all existing chunks and re-ingest every PDF.

        Returns:
            List of per-file ingestion stats.
        """
        upload_dir = Path(self._config.pdf_upload_dir)
        if not upload_dir.exists():
            logger.warning("Upload directory does not exist: %s", upload_dir)
            return []

        pdf_files = list(upload_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDFs found in %s", upload_dir)
            return []

        logger.info("Re-ingesting %d PDFs with force=True from %s", len(pdf_files), upload_dir)
        results: List[Dict[str, Any]] = []
        for pdf_path in pdf_files:
            result = self.ingest_pdf(str(pdf_path), force=True)
            results.append(result)
        return results

    def ingest_uploaded_bytes(
        self,
        file_name: str,
        file_bytes: bytes,
        extra_metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """Ingest a PDF from raw bytes (e.g. Streamlit file uploader).

        Args:
            file_name: Original file name.
            file_bytes: Raw PDF bytes.
            extra_metadata: Additional metadata to attach.
            progress_callback: Optional callback(stage_name, pct) for UI updates.

        Returns:
            Ingestion stats dict.
        """
        save_dir = Path(self._config.pdf_upload_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name
        save_path.write_bytes(file_bytes)
        logger.info("Saved uploaded file to %s", save_path)
        return self.ingest_pdf(
            str(save_path),
            extra_metadata=extra_metadata,
            progress_callback=progress_callback,
        )

    def delete_source(self, source_name: str) -> int:
        """Remove all chunks belonging to a given source PDF.

        Args:
            source_name: The source file name.

        Returns:
            Number of chunks deleted.
        """
        results = self._vs.collection.get(where={"source": source_name})
        count = len(results["ids"]) if results["ids"] else 0
        if count:
            self._vs.delete_by_ids(results["ids"])
            self._notify_change(source_name, "deleted")
        return count

    def _notify_change(self, source_name: str, action: str) -> None:
        """Invoke the on-change callback to invalidate downstream caches."""
        if self._on_change is not None:
            try:
                self._on_change(source_name, action)
                logger.info(
                    "Cache invalidation triggered: source='%s', action='%s'",
                    source_name, action,
                )
            except Exception as e:
                logger.warning("Cache invalidation callback failed: %s", e)
