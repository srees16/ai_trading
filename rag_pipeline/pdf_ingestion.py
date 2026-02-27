"""
PDF Ingestion Service for Centurion Capital LLC RAG Pipeline.

Responsibilities:
    1. Parse PDF files to plain text (via PyMuPDF / fitz)
    2. Split text into overlapping chunks (token-based or char-based)
    3. Embed chunks
    4. Store in ChromaDB via VectorStoreManager
"""

import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings import EmbeddingService
from rag_pipeline.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokeniser helper (lightweight, no heavy dependency)
# ---------------------------------------------------------------------------

_WORD_TOKEN_RE = re.compile(r"\S+")


def _approx_token_count(text: str) -> int:
    """Approximate token count using whitespace splitting.

    GPT / sentence-transformer tokenisers track ~75 % of word count,
    but for chunking purposes whitespace words are a safe, fast proxy.
    """
    return len(_WORD_TOKEN_RE.findall(text))


# ---------------------------------------------------------------------------
# Sentence splitter (zero external dependency)
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d\"'\(\[])"  # sentence boundary heuristic
)

# Paragraph boundary: two or more newlines (blank line)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Section heading pattern: numbered ("1.", "2.3"), all-caps, or title-case short line
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*\s+\S|"       # numbered heading
    r"[A-Z][A-Z\s]{2,80}$|"       # ALL-CAPS heading
    r"[A-Z][a-zA-Z\s]{2,80}:$"    # title-case ending with colon
    r")",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Code block detection
# ---------------------------------------------------------------------------

# Heuristic patterns that indicate a line is part of a code block
_CODE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"def\s+\w+\s*\(|"                  # function definition
    r"class\s+\w+|"                      # class definition
    r"import\s+\w|from\s+\w|"           # imports
    r"if\s+.*:|elif\s+.*:|else\s*:|"     # conditionals
    r"for\s+\w.*:|while\s+.*:|"          # loops
    r"return\s|yield\s|raise\s|"         # control flow
    r"try\s*:|except\s|finally\s*:|"     # exception handling
    r"with\s+\w|"                        # context managers
    r"#\s*\S|"                            # comments
    r"[a-zA-Z_]\w*\s*=[^=]|"             # assignments
    r"[a-zA-Z_]\w*\s*\(|"                # function calls
    r"[a-zA-Z_]\w*\.\w+|"                # attribute access
    r"\[.*\]|\{.*\}|"                    # list/dict literals
    r"print\s*\(|"                        # print calls
    r"@\w+"                               # decorators
    r")"
)

# Pattern for snippet/code-block labels like "SNIPPET 2.1" or "Code Example 3"
_SNIPPET_LABEL_RE = re.compile(
    r"(?i)(?:snippet|code|listing|algorithm|example|figure)\s*[\d.]+",
)


def _is_code_line(line: str) -> bool:
    """Return True if *line* looks like a Python code line."""
    stripped = line.strip()
    if not stripped:
        return False
    # Lines with significant leading whitespace (≥2 spaces) that contain
    # alphanumeric content are likely indented code
    if len(line) - len(line.lstrip()) >= 2 and stripped and not stripped[0].isupper():
        return True
    return bool(_CODE_LINE_RE.match(stripped))


def _is_code_block(lines: List[str], threshold: float = 0.4) -> bool:
    """Return True if ≥ *threshold* fraction of non-empty lines look like code."""
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 2:
        return False
    code_count = sum(1 for l in non_empty if _is_code_line(l))
    return code_count / len(non_empty) >= threshold


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-ish segments."""
    parts = _SENTENCE_END.split(text)
    return [p for p in parts if p.strip()]


def _split_paragraphs(text: str) -> List[str]:
    """Split text on paragraph boundaries (blank lines), keeping code blocks intact.

    Code blocks (detected by indentation + Python syntax patterns) are
    never split across paragraphs — the blank lines inside a code block
    are preserved so that complete functions stay in one segment.
    """
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
                # Inside a code block: keep blank lines (they're part of the code)
                if blank_count <= 2:
                    current_segment.append(line)
                    continue
                else:
                    # 3+ consecutive blanks → code block ended
                    in_code_block = False
            if blank_count >= 2 and current_segment:
                # Paragraph break — flush
                segments.append("\n".join(current_segment))
                current_segment = []
                blank_count = 0
            continue

        blank_count = 0

        # Detect code: if this line or recent context looks like code,
        # enter/continue code block mode
        if _is_code_line(line):
            in_code_block = True
        elif in_code_block and len(line) - len(line.lstrip()) >= 2:
            # Still indented — continue code block
            pass
        elif in_code_block and not _is_code_line(line) and len(line) - len(line.lstrip()) < 2:
            # Unindented non-code line → check if the upcoming lines are code
            # If current segment has code and this looks like a snippet label, keep it
            if _SNIPPET_LABEL_RE.search(stripped):
                pass  # labels stay with the code block
            else:
                in_code_block = False

        current_segment.append(line)

    if current_segment:
        seg_text = "\n".join(current_segment).strip()
        if seg_text:
            segments.append(seg_text)

    return [s.strip() for s in segments if s.strip()]


# ---------------------------------------------------------------------------
# Text chunking (token-aware with semantic boundaries)
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace and remove common PDF noise.

    Strips:
    - Repeated headers / footers (lines that appear identically on 3+ pages)
    - Standalone page numbers (e.g. "  12  ", "Page 5 of 20")
    - Excessive whitespace / blank lines
    - Soft hyphens from line-break hyphenation

    IMPORTANT: Leading whitespace (indentation) is PRESERVED because
    it is structural in code blocks — destroying it makes code
    snippets unrecoverable.
    """
    # Remove soft hyphens (word broken across lines)
    text = re.sub(r"-(\r?\n)\s*", "", text)
    # Remove standalone page number lines
    text = re.sub(
        r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Collapse multiple inline spaces/tabs but PRESERVE leading indentation.
    # This is critical for Python code blocks extracted from PDFs.
    def _collapse_inline(line: str) -> str:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        # normalise the indent to spaces (tabs → 4 spaces)
        indent = indent.replace("\t", "    ")
        # collapse multiple inline spaces within the content part
        stripped = re.sub(r"  +", " ", stripped)
        return indent + stripped

    text = "\n".join(_collapse_inline(l) for l in text.split("\n"))
    # Collapse 3+ newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    unit: str = "token",
    max_tokens: int = 800,
) -> List[str]:
    """
    Split *text* into overlapping chunks.

    Supports two modes controlled by *unit*:

    * ``"token"`` (default) — chunk boundaries are measured in
      approximate whitespace tokens.  Target size is *chunk_size*
      tokens with *chunk_overlap* token overlap.  Hard ceiling at
      *max_tokens*.  Tries to land on sentence boundaries.

    * ``"char"`` — legacy character-count mode kept for
      backward-compatibility.

    Returns a list of non-empty chunk strings.
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
    """Token-aware chunking with paragraph- and sentence-boundary snapping.

    Hierarchy of split boundaries (preferred → fallback):
        1. Paragraph breaks (blank lines)
        2. Sentence endings (.!?) — only for prose, NOT for code blocks
        3. Hard word-level split (only for oversized segments)

    Code blocks (detected by indentation + Python syntax patterns) are
    kept intact and joined with newlines instead of spaces so that
    indentation is preserved.
    """
    # First split by paragraphs (code-block-aware), then by sentences
    # within each paragraph ONLY if the paragraph is prose (not code).
    paragraphs = _split_paragraphs(text)
    segments: List[str] = []
    segment_is_code: List[bool] = []

    for para in paragraphs:
        lines = para.split("\n")
        if _is_code_block(lines):
            # Keep code blocks as a single segment — never sentence-split
            segments.append(para)
            segment_is_code.append(True)
        else:
            # Split prose into sentences
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
        """Join segments using newlines for code and spaces for prose."""
        if not segs:
            return ""
        parts: List[str] = [segs[0]]
        for i in range(1, len(segs)):
            # If either the previous or current segment is code, use newlines
            if (i - 1 < len(seg_codes) and seg_codes[i - 1]) or \
               (i < len(seg_codes) and seg_codes[i]):
                parts.append("\n" + segs[i])
            else:
                parts.append(" " + segs[i])
        return "".join(parts)

    chunks: List[str] = []
    current: List[str] = []
    current_codes: List[bool] = []
    current_len = 0  # token count of current chunk

    for seg_idx, segment in enumerate(segments):
        is_code = seg_idx < len(segment_is_code) and segment_is_code[seg_idx]
        s_tokens = _approx_token_count(segment)

        # If a single segment exceeds max_tokens, hard-split it
        if s_tokens > max_tokens:
            # Flush current buffer first
            if current:
                chunks.append(_join_segments(current, current_codes))
                current = []
                current_codes = []
                current_len = 0
            if is_code:
                # For oversized code blocks: split by lines, not words
                code_lines = segment.split("\n")
                buf_lines: List[str] = []
                buf_len = 0
                for cl in code_lines:
                    cl_tokens = _approx_token_count(cl)
                    if buf_len + cl_tokens > target_tokens and buf_lines:
                        chunks.append("\n".join(buf_lines))
                        # keep overlap (last N lines)
                        overlap_lines = buf_lines[-(overlap_tokens // max(1, (buf_len // max(1, len(buf_lines))))):]
                        buf_lines = list(overlap_lines)
                        buf_len = sum(_approx_token_count(l) for l in buf_lines)
                    buf_lines.append(cl)
                    buf_len += cl_tokens
                if buf_lines:
                    chunks.append("\n".join(buf_lines))
            else:
                # Hard split the oversized segment by words
                words = segment.split()
                buf: List[str] = []
                buf_len = 0
                for w in words:
                    buf.append(w)
                    buf_len += 1
                    if buf_len >= target_tokens:
                        chunks.append(" ".join(buf))
                        # keep overlap
                        overlap_words = buf[-overlap_tokens:] if overlap_tokens < len(buf) else buf[:]
                        buf = list(overlap_words)
                        buf_len = len(buf)
                if buf:
                    chunks.append(" ".join(buf))
            continue

        # Would adding this segment exceed the target?
        if current_len + s_tokens > target_tokens and current:
            chunks.append(_join_segments(current, current_codes))
            # Build overlap from the tail of current
            overlap_buf: List[str] = []
            overlap_codes: List[bool] = []
            overlap_count = 0
            for oi in range(len(current) - 1, -1, -1):
                t = _approx_token_count(current[oi])
                if overlap_count + t > overlap_tokens:
                    break
                overlap_buf.insert(0, current[oi])
                overlap_codes.insert(0, current_codes[oi] if oi < len(current_codes) else False)
                overlap_count += t
            current = overlap_buf
            current_codes = overlap_codes
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

        # Try to land on a sentence boundary
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


# ---------------------------------------------------------------------------
# Structural document parsing
# ---------------------------------------------------------------------------

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


# Structural boundary patterns
_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:chapter|ch\.?)\s+(\d+)[\s.:—\-]*(.*)",
    re.IGNORECASE,
)

# Multi-level numbered section: "2.1 Title", "3.2.1 Sub-section"
_NUMBERED_SECTION_RE = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)*)\s+([A-Z][a-zA-Z\s,'\.\-]{2,100})\s*$",
)

# Bare multi-level section number on its own line: "2.4.3" (title is on the next line)
_BARE_SECTION_NUM_RE = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)*)\s*$",
)

# Top-level numbered section (single digit): "2 Denoising and Detoning"
_TOP_LEVEL_SECTION_RE = re.compile(
    r"^\s*(\d{1,2})\s+([A-Z][a-zA-Z\s,'\-]{3,80})\s*$",
)

# Snippet / code block label: "SNIPPET 2.1", "Code Listing 3"
_SNIPPET_BLOCK_RE = re.compile(
    r"^\s*(?:snippet|code\s*(?:listing|example)?|listing|algorithm)\s+"
    r"([\d.]+)[\s.:—\-]*(.*)",
    re.IGNORECASE,
)


def _parse_structural_segments(text: str) -> List[StructuralChunk]:
    """Parse document text into structural segments with metadata.

    Scans line-by-line for chapter headers, section headers, and snippet
    labels. Each structural boundary starts a new segment. All segments
    inherit the current chapter/section context.
    """
    lines = text.split("\n")
    segments: List[StructuralChunk] = []

    # Current structural context (propagated forward)
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
        # Reset snippet after flushing (snippet is specific to one block)
        cur_snippet_id = ""
        cur_snippet_title = ""

    for line in lines:
        stripped = line.strip()

        # Check for chapter heading
        m = _CHAPTER_HEADING_RE.match(stripped)
        if m and len(stripped) < 120:
            _flush()
            cur_chapter = m.group(1)
            cur_chapter_title = m.group(2).strip()
            cur_section_id = ""
            cur_section_title = ""
            current_lines.append(line)
            continue

        # Check for snippet label (before section check to avoid ambiguity)
        m = _SNIPPET_BLOCK_RE.match(stripped)
        if m:
            _flush()
            cur_snippet_id = m.group(1)
            cur_snippet_title = m.group(2).strip()
            current_lines.append(line)
            continue

        # Check for multi-level section heading: "2.3 Title"
        m = _NUMBERED_SECTION_RE.match(stripped)
        if m and len(stripped) < 120:
            _flush()
            cur_section_id = m.group(1)
            cur_section_title = m.group(2).strip()
            current_lines.append(line)
            continue

        # Check for bare section number on its own line: "2.4.3"
        # (common when PDF extraction splits number and title onto separate lines)
        m = _BARE_SECTION_NUM_RE.match(stripped)
        if m:
            _flush()
            cur_section_id = m.group(1)
            cur_section_title = ""
            current_lines.append(line)
            continue

        # Check for top-level section heading: "2 Title"
        # (only if no chapter structure detected yet)
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

    Algorithm:
        1. Clean text (preserve indentation)
        2. Parse structural boundaries (chapters, sections, snippets)
        3. Keep code snippets intact (never split across boundaries)
        4. Merge small adjacent segments under the same section
        5. Sub-split oversized prose with token-based chunking
        6. Propagate structural metadata to all resulting chunks

    Returns ``StructuralChunk`` objects with rich metadata for each chunk.
    """
    text = _clean_text(text)
    if not text:
        return []

    segments = _parse_structural_segments(text)
    if not segments:
        # Fallback: no structural boundaries detected → basic chunking
        basic_chunks = _chunk_by_tokens(text, chunk_size, chunk_overlap, max_tokens)
        return [StructuralChunk(text=c, is_code=_is_code_block(c.split("\n"))) for c in basic_chunks]

    result: List[StructuralChunk] = []

    # Accumulate small segments; flush at structural boundaries or size limit
    buffer_texts: List[str] = []
    buffer_tokens: int = 0
    buffer_meta: Optional[StructuralChunk] = None  # metadata from first segment

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

        # --- Code snippets: always keep intact ---
        if seg.snippet_id or seg.is_code:
            _flush_buffer()
            if seg_tokens <= max_tokens:
                result.append(seg)
            else:
                # Oversized code: split by lines, preserving metadata
                code_lines = seg.text.split("\n")
                buf_lines: List[str] = []
                buf_len = 0
                for cl in code_lines:
                    cl_tokens = _approx_token_count(cl)
                    if buf_len + cl_tokens > chunk_size and buf_lines:
                        result.append(StructuralChunk(
                            text="\n".join(buf_lines),
                            chapter=seg.chapter,
                            chapter_title=seg.chapter_title,
                            section_id=seg.section_id,
                            section_title=seg.section_title,
                            snippet_id=seg.snippet_id,
                            snippet_title=seg.snippet_title,
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
                        chapter=seg.chapter,
                        chapter_title=seg.chapter_title,
                        section_id=seg.section_id,
                        section_title=seg.section_title,
                        snippet_id=seg.snippet_id,
                        snippet_title=seg.snippet_title,
                        is_code=True,
                    ))
            continue

        # --- Structural boundary: new chapter or section → flush ---
        is_new_boundary = (
            buffer_meta is not None and (
                seg.chapter != buffer_meta.chapter
                or seg.section_id != buffer_meta.section_id
            )
        )
        if is_new_boundary:
            _flush_buffer()

        # --- Oversized prose segment: sub-split ---
        if seg_tokens > chunk_size:
            _flush_buffer()
            sub_chunks = _chunk_by_tokens(
                seg.text, chunk_size, chunk_overlap, max_tokens
            )
            for sub in sub_chunks:
                result.append(StructuralChunk(
                    text=sub,
                    chapter=seg.chapter,
                    chapter_title=seg.chapter_title,
                    section_id=seg.section_id,
                    section_title=seg.section_title,
                    snippet_id="",
                    snippet_title="",
                    is_code=_is_code_block(sub.split("\n")),
                ))
            continue

        # --- Would adding this segment exceed target? ---
        if buffer_tokens + seg_tokens > chunk_size and buffer_texts:
            _flush_buffer()

        # Accumulate
        buffer_texts.append(seg.text)
        buffer_tokens += seg_tokens
        if buffer_meta is None:
            buffer_meta = seg

    _flush_buffer()
    return result


# ---------------------------------------------------------------------------
# PDF noise removal — strip repeated headers/footers across pages
# ---------------------------------------------------------------------------

def _strip_repeated_lines(full_text: str, pages_text: List[str], min_occurrences: int = 3) -> str:
    """Remove lines that appear identically across many pages.

    PDF headers, footers, and watermarks are typically repeated on
    every page.  We detect lines that appear on ≥ *min_occurrences*
    different pages and strip them from the full text.
    """
    from collections import Counter

    line_page_counts: Counter = Counter()
    for page_text in pages_text:
        # Unique lines per page (avoid counting duplicates within a page)
        seen: set = set()
        for line in page_text.split("\n"):
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                line_page_counts[stripped] += 1

    # Lines appearing on ≥ min_occurrences pages are likely headers/footers
    noise_lines = {
        line for line, count in line_page_counts.items()
        if count >= min_occurrences and len(line) < 200  # only short repeated lines
    }

    if noise_lines:
        logger.info(
            "Stripping %d repeated header/footer lines (appear on %d+ pages)",
            len(noise_lines), min_occurrences,
        )
        cleaned_lines = []
        for line in full_text.split("\n"):
            if line.strip() not in noise_lines:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    return full_text


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _detect_section_heading(text: str) -> str:
    """
    Attempt to detect the section heading from the start of a chunk.

    Looks for short uppercase lines, numbered headings, or lines ending
    with a colon that typically indicate section titles.
    """
    lines = text.strip().split("\n", 5)  # check first few lines
    for line in lines[:3]:
        stripped = line.strip()
        if not stripped:
            continue
        # Numbered heading:  "1. Introduction", "2.3 Methodology"
        if re.match(r"^\d+(\.\d+)*\s+\S", stripped) and len(stripped) < 120:
            return stripped
        # ALL-CAPS heading
        if stripped.isupper() and 3 < len(stripped) < 120:
            return stripped
        # Title-case heading ending with colon or short standalone line
        if stripped.istitle() and len(stripped) < 100:
            return stripped
        # Line ending with colon (common heading pattern)
        if stripped.endswith(":") and len(stripped) < 100:
            return stripped.rstrip(":")
    return ""



# ---------------------------------------------------------------------------
# Layout-aware text extraction (preserves code indentation)
# ---------------------------------------------------------------------------

# Monospace font families used in academic books for code blocks
_MONO_FONTS = frozenset({
    "courier", "couriernew", "courierstd", "courierstd-oblique",
    "courierstd-bold", "courierstd-boldoblique",
    "consolas", "menlo", "inconsolata", "dejavusansmono",
    "liberationmono", "sourcecodepro", "firacode", "ubuntumono",
    "robotomono", "notomono",
})


def _is_monospace(font_name: str) -> bool:
    """Return True if *font_name* is a known monospace/code font."""
    return font_name.lower().replace("-", "").replace(" ", "") in _MONO_FONTS or \
           "courier" in font_name.lower() or "mono" in font_name.lower()


def _extract_page_with_layout(page) -> str:
    """Extract text from a single PyMuPDF page, preserving code indentation.

    For text blocks using monospace fonts (code), indentation is
    reconstructed from the x-position of each line relative to the
    leftmost code line on the page.  For regular prose blocks, plain
    ``get_text("text")`` output is used.

    This hybrid approach ensures code snippets retain their Python
    indentation (critical for correctness) while prose remains clean.
    """
    data = page.get_text("dict", flags=0)
    blocks = data.get("blocks", [])

    # Separate blocks into code and prose based on dominant font
    output_parts: list = []

    for block in blocks:
        if block.get("type") != 0:  # only text blocks
            continue

        lines = block.get("lines", [])
        if not lines:
            continue

        # Determine if this block uses a monospace (code) font
        font_counts: dict = {}
        for line in lines:
            for span in line.get("spans", []):
                fn = span.get("font", "")
                font_counts[fn] = font_counts.get(fn, 0) + len(span.get("text", ""))
        dominant_font = max(font_counts, key=font_counts.get) if font_counts else ""
        is_code_block_flag = _is_monospace(dominant_font)

        if is_code_block_flag:
            # Reconstruct code with indentation from x-positions
            # Find the minimum x0 across all lines in this code block
            min_x = float("inf")
            for line in lines:
                spans = line.get("spans", [])
                if spans:
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if text:
                        min_x = min(min_x, spans[0]["bbox"][0])
            if min_x == float("inf"):
                min_x = 0.0

            # Estimate character width from the monospace font
            # Use average span width / char count as proxy
            char_widths: list = []
            for line in lines:
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if txt and _is_monospace(span.get("font", "")):
                        bbox = span["bbox"]
                        w = bbox[2] - bbox[0]
                        if len(txt) > 0 and w > 0:
                            char_widths.append(w / len(txt))
            char_w = sum(char_widths) / len(char_widths) if char_widths else 5.0

            code_lines: list = []
            for line in lines:
                spans = line.get("spans", [])
                if not spans:
                    code_lines.append("")
                    continue
                x0 = spans[0]["bbox"][0]
                indent_px = max(0.0, x0 - min_x)
                indent_spaces = round(indent_px / char_w)
                # Round to nearest multiple of 4 for Python style
                indent_spaces = (indent_spaces // 4) * 4
                full_text = "".join(s.get("text", "") for s in spans)
                code_lines.append(" " * indent_spaces + full_text.strip())

            output_parts.append("\n".join(code_lines))
        else:
            # Regular prose: concatenate spans per line
            prose_lines: list = []
            for line in lines:
                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans)
                prose_lines.append(line_text)
            output_parts.append("\n".join(prose_lines))

    return "\n".join(output_parts)


def extract_text_by_page(pdf_path: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract text from each page of a PDF individually.

    Uses layout-aware extraction that preserves code indentation
    by analysing x-coordinates of monospace font spans.

    Returns:
        (list_of_page_texts, file_metadata)
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages_text: List[str] = []
    for page in doc:
        pages_text.append(_extract_page_with_layout(page))

    metadata = {
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
    return pages_text, metadata


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract all text from a PDF file (legacy helper).

    Returns:
        (full_text, metadata_dict)
    """
    pages_text, metadata = extract_text_by_page(pdf_path)
    return "\n\n".join(pages_text), metadata


# ---------------------------------------------------------------------------
# Ingestion service
# ---------------------------------------------------------------------------

class PDFIngestionService:
    """
    End-to-end PDF → ChromaDB ingestion.

    Usage:
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
        # Optional callback invoked after ingestion/deletion to invalidate
        # downstream caches (e.g. SemanticCache, BM25 index).
        # Signature: callback(source_name: str, action: str)
        self._on_change = on_change_callback

    # ------------------------------------------------------------------
    # Public API
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
        """Return True if *source_name* with the same content hash is
        already fully stored in the vector DB."""
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

    def ingest_pdf(
        self,
        pdf_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Parse, chunk, embed, and store a single PDF.

        Uses page-aware chunking so each chunk carries its source page
        number, content hash, word count, section heading, and timestamp
        for efficient metadata-filtered retrieval.

        If the exact same file (by SHA-256 content hash) is already
        persisted in ChromaDB the expensive embedding step is skipped
        and the existing vectors are reused across sessions.

        Set ``force=True`` to bypass the duplicate check and re-chunk /
        re-embed the PDF with the latest chunking pipeline (useful after
        chunking algorithm improvements).

        Returns a summary dict with ingestion statistics.
        """
        logger.info("Ingesting PDF: %s (force=%s)", pdf_path, force)
        ingest_ts = datetime.now(timezone.utc).isoformat()

        # ---- Duplicate detection (fast, no embedding needed) ----
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

        # If a previous version exists (different hash), remove it first
        if self._vs.source_exists(source_name):
            logger.info(
                "Source '%s' exists with different content — replacing.",
                source_name,
            )
            self.delete_source(source_name)

        # 1. Extract text page-by-page
        pages_text, file_meta = extract_text_by_page(pdf_path)
        full_text = "\n\n".join(pages_text)
        if not full_text.strip():
            logger.warning("PDF contains no extractable text: %s", pdf_path)
            return {"status": "skipped", "reason": "empty", "source": pdf_path}

        # 2. Cross-page chunking with page tracking
        #    Instead of chunking each page independently (which breaks
        #    cross-page context), we chunk the full document text and
        #    retroactively map each chunk back to its source page(s).

        # Build a page-offset index: character offset where each page starts
        page_offsets: List[int] = []  # (start_char_offset, page_number)
        offset = 0
        for page_idx, page_text in enumerate(pages_text):
            page_offsets.append(offset)
            offset += len(page_text) + 2  # +2 for the "\n\n" separator

        def _find_page_for_chunk(chunk_text: str) -> int:
            """Find the 1-based page number where this chunk starts."""
            pos = full_text.find(chunk_text[:120])  # match on first 120 chars
            if pos < 0:
                return 1  # fallback to page 1
            # Find which page this offset falls in
            for i in range(len(page_offsets) - 1, -1, -1):
                if pos >= page_offsets[i]:
                    return i + 1  # 1-based
            return 1

        # Remove repeated header/footer lines that appear across pages
        cleaned_full_text = _strip_repeated_lines(full_text, pages_text)

        # 2b. Structure-aware chunking (respects chapter / section / snippet
        #     boundaries and preserves code blocks intact).
        structural_chunks = structure_aware_chunk(
            cleaned_full_text,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            max_tokens=self._config.chunk_max_tokens,
        )
        all_chunks = [sc.text for sc in structural_chunks]
        chunk_page_map = [_find_page_for_chunk(c) for c in all_chunks]

        if not all_chunks:
            logger.warning("No chunks produced from PDF: %s", pdf_path)
            return {"status": "skipped", "reason": "no_chunks", "source": pdf_path}

        # 3. Build IDs & enriched metadata
        source_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        filtered_chunks: List[str] = []
        filtered_pages: List[int] = []
        min_words = self._config.chunk_min_tokens

        for idx, chunk in enumerate(all_chunks):
            word_count = len(chunk.split())
            token_count = _approx_token_count(chunk)
            # Skip very short chunks (likely noise: headers, footers, etc.)
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
                # --- Core identity ---
                "source": file_meta["source"],
                "title": file_meta["title"],
                "author": file_meta["author"],
                # --- File-level fingerprint (same for all chunks) ---
                "file_hash": file_hash,
                # --- Page & position ---
                "page_number": chunk_page_map[idx],
                "page_count": file_meta["page_count"],
                "chunk_index": idx,
                "total_chunks": len(all_chunks),
                # --- Content fingerprint ---
                "chunk_hash": chunk_hash,
                "word_count": word_count,
                "token_count": token_count,
                "char_count": len(chunk),
                # --- Section / heading ---
                "section": section,
                # --- Structural metadata ---
                "chapter": struct.chapter,
                "chapter_title": struct.chapter_title,
                "section_id": struct.section_id,
                "section_title": struct.section_title,
                "snippet_id": struct.snippet_id,
                "snippet_title": struct.snippet_title,
                # --- Code detection ---
                "contains_code": struct.is_code or _is_code_block(chunk.split("\n")),
                # --- Temporal ---
                "ingested_at": ingest_ts,
                # --- PDF metadata ---
                "subject": file_meta.get("subject", ""),
                "keywords": file_meta.get("keywords", ""),
                # --- Multi-tenancy ---
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

        # 4. Embed
        embeddings = self._embedder.embed_texts(filtered_chunks)

        # 5. Store
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
        logger.info("Ingestion complete: %s", stats)

        # Notify downstream caches that content has changed
        self._notify_change(file_meta["source"], "ingested")

        return stats

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Ingest all PDFs in a directory."""
        pattern = "**/*.pdf" if recursive else "*.pdf"
        results: List[Dict[str, Any]] = []
        for pdf_file in Path(directory).glob(pattern):
            result = self.ingest_pdf(str(pdf_file))
            results.append(result)
        return results

    def reingest_all(self) -> List[Dict[str, Any]]:
        """Delete all existing chunks and re-ingest every PDF in the upload directory.

        This is useful after chunking algorithm improvements to ensure
        all documents benefit from the latest pipeline changes.
        """
        upload_dir = Path(self._config.pdf_upload_dir)
        if not upload_dir.exists():
            logger.warning("Upload directory does not exist: %s", upload_dir)
            return []

        pdf_files = list(upload_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDFs found in %s", upload_dir)
            return []

        logger.info(
            "Re-ingesting %d PDFs with force=True from %s",
            len(pdf_files), upload_dir,
        )
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
    ) -> Dict[str, Any]:
        """
        Ingest a PDF from raw bytes (Streamlit file uploader).

        Saves to ``pdf_upload_dir`` then delegates to ``ingest_pdf``.
        """
        save_dir = Path(self._config.pdf_upload_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name

        save_path.write_bytes(file_bytes)
        logger.info("Saved uploaded file to %s", save_path)

        return self.ingest_pdf(str(save_path), extra_metadata=extra_metadata)

    def delete_source(self, source_name: str) -> int:
        """Remove all chunks belonging to a given source PDF."""
        results = self._vs.collection.get(
            where={"source": source_name},
        )
        count = len(results["ids"]) if results["ids"] else 0
        if count:
            self._vs.delete_by_ids(results["ids"])
            # Notify downstream caches that content has changed
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
                logger.warning(
                    "Cache invalidation callback failed: %s", e
                )
