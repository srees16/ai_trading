"""
Code Applier — Extract code snippets from RAG responses and apply them
to strategy files in the centurion_core workspace.

Flow:
    1. ``extract_code_blocks()`` — parse fenced code blocks from LLM output
    2. ``list_strategy_files()`` — discover all editable strategy files
    3. ``generate_patch()`` — use the LLM to produce a modified version
       of the target file that incorporates the code snippet
    4. ``apply_patch()`` — write the modified file (with auto-backup)

All public functions are designed to be called from Streamlit UI widgets.
"""

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag_pipeline.config import RAGConfig
from rag_pipeline.llm_service import create_llm_backend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root — everything is relative to this
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories that contain editable strategy code
_STRATEGY_DIRS = [
    _PROJECT_ROOT / "trading_strategies",
    _PROJECT_ROOT / "strategies",
]

# Backup folder lives beside rag_pipeline/data/
_BACKUP_DIR = _PROJECT_ROOT / "rag_pipeline" / "data" / "code_backups"

# Application log (who applied what, when)
_APPLY_LOG = _PROJECT_ROOT / "rag_pipeline" / "data" / "code_apply_log.jsonl"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CodeBlock:
    """A single fenced code block extracted from an LLM response."""
    language: str          # e.g. "python", "" if unspecified
    code: str              # raw code content (no fences)
    index: int             # 0-based position in the answer


@dataclass
class StrategyFileInfo:
    """Metadata about a discovered strategy file."""
    path: Path             # absolute path
    rel_path: str          # display-friendly relative path
    name: str              # strategy class name (if detectable) or filename
    category: str          # subdirectory / category label


@dataclass
class PatchResult:
    """Result of a code-application attempt."""
    success: bool
    target_file: str
    backup_file: Optional[str] = None
    message: str = ""
    diff_summary: str = ""


# ---------------------------------------------------------------------------
# 1. Extract code blocks from RAG answer
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(
    r"```([\w+#.*-]*)\n?(.*?)```",
    re.DOTALL,
)


def extract_code_blocks(answer: str) -> List[CodeBlock]:
    """Parse all fenced code blocks from an LLM answer string.

    Returns a list of ``CodeBlock`` objects preserving order of appearance.
    Only blocks with non-whitespace content are included.
    """
    blocks: List[CodeBlock] = []
    for idx, m in enumerate(_CODE_BLOCK_RE.finditer(answer)):
        lang = (m.group(1) or "").strip().lower()
        code = m.group(2).strip()
        if code:
            blocks.append(CodeBlock(language=lang, code=code, index=idx))
    return blocks


# ---------------------------------------------------------------------------
# 2. Discover strategy files
# ---------------------------------------------------------------------------

def list_strategy_files() -> List[StrategyFileInfo]:
    """Scan the project for editable strategy / framework files.

    Returns a sorted list of ``StrategyFileInfo`` objects.
    """
    results: List[StrategyFileInfo] = []
    seen: set = set()

    for base_dir in _STRATEGY_DIRS:
        if not base_dir.exists():
            continue
        for py_file in sorted(base_dir.rglob("*.py")):
            # Skip __pycache__, hidden dirs, backtest-only files
            parts = py_file.parts
            if "__pycache__" in parts:
                continue
            if py_file.name.startswith("_"):
                continue

            abs_str = str(py_file.resolve())
            if abs_str in seen:
                continue
            seen.add(abs_str)

            # Derive a human-friendly category from the directory structure
            try:
                rel = py_file.relative_to(_PROJECT_ROOT)
            except ValueError:
                rel = py_file
            rel_str = str(rel).replace("\\", "/")

            # Category = first two path components, e.g. "trading_strategies/momentum_trading"
            rel_parts = rel.parts
            if len(rel_parts) >= 3:
                category = f"{rel_parts[0]}/{rel_parts[1]}"
            elif len(rel_parts) >= 2:
                category = rel_parts[0]
            else:
                category = ""

            # Try to extract the class name from the file (lightweight)
            class_name = _extract_class_name(py_file)
            display_name = class_name if class_name else py_file.stem

            results.append(
                StrategyFileInfo(
                    path=py_file.resolve(),
                    rel_path=rel_str,
                    name=display_name,
                    category=category,
                )
            )

    return results


def _extract_class_name(py_file: Path) -> str:
    """Quick regex scan for the first class declaration inheriting BaseStrategy."""
    try:
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"class\s+(\w+)\s*\(.*?BaseStrategy.*?\):", text)
        return m.group(1) if m else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 3. Generate patch (LLM-assisted merge)
# ---------------------------------------------------------------------------

_CODE_APPLY_SYSTEM = """\
You are a precise Python code editor for Centurion Capital's trading strategy codebase.

TASK: You will receive:
  1. The FULL content of a target Python file (the strategy to improve).
  2. One or more CODE SNIPPETS suggested by a RAG system.

Your job is to MERGE the code snippets into the target file to improve the strategy, \
following these rules STRICTLY:

RULES:
1. Return ONLY the complete modified Python file — no explanations, no markdown, \
no commentary before or after.
2. Do NOT wrap the output in triple backticks or markdown fences.
3. Preserve ALL existing imports, class structure, decorators, and method signatures \
unless the snippet explicitly replaces them.
4. Insert or replace code in the CORRECT location (e.g. new indicator logic goes in \
the `run()` method, new parameters go in `get_parameters()`, new imports go at the top).
5. If the snippet defines a new helper method, add it as a method of the strategy class \
(or a module-level function if appropriate).
6. Maintain consistent indentation (4 spaces) and code style.
7. Do NOT remove, rename, or break existing functionality unless the snippet \
explicitly supersedes it.
8. Add a brief inline comment (# Applied from RAG suggestion) near each modification \
so the developer can review.
9. If the snippet cannot be reasonably applied (incompatible, nonsensical), return \
the ORIGINAL file unchanged with a comment at the top: # NOTE: RAG snippet could \
not be applied — <reason>

IMPORTANT: Output ONLY valid Python source code. Nothing else.\
"""


def generate_patch(
    target_file: Path,
    code_blocks: List[CodeBlock],
    query: str = "",
    config: Optional[RAGConfig] = None,
) -> Tuple[str, str]:
    """Use the LLM to merge code snippets into a target file.

    Args:
        target_file: Absolute path to the strategy file.
        code_blocks: Code snippets to apply.
        query: The original user query (provides intent context).
        config: RAGConfig for LLM settings.

    Returns:
        (modified_source, summary) where *modified_source* is the full
        file content after patching, and *summary* is a short description
        of what changed.
    """
    config = config or RAGConfig()
    llm = create_llm_backend(config)

    original = target_file.read_text(encoding="utf-8")

    # Build the user message
    snippets_text = "\n\n".join(
        f"### Snippet {b.index + 1} ({b.language or 'python'})\n{b.code}"
        for b in code_blocks
    )

    user_msg = (
        f"## Original File: {target_file.name}\n\n"
        f"```python\n{original}\n```\n\n"
        f"---\n\n"
        f"## Code Snippets to Apply\n\n{snippets_text}\n\n"
    )
    if query:
        user_msg += f"## User's Intent\n\n{query}\n\n"

    # Call LLM with the code-apply system prompt
    # We temporarily override by calling the raw backend
    # Build messages manually for Ollama chat API
    try:
        # Use a direct Ollama call with our custom system prompt
        import requests
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})

        response = session.post(
            f"{config.ollama_base_url.rstrip('/')}/api/chat",
            json={
                "model": config.ollama_model,
                "messages": [
                    {"role": "system", "content": _CODE_APPLY_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.05,        # near-deterministic
                    "num_predict": 16384,        # large file output
                    "num_ctx": 16384,
                },
            },
            timeout=config.llm_timeout,
        )
        response.raise_for_status()
        data = response.json()
        modified_source = data.get("message", {}).get("content", "")
    except Exception as e:
        logger.error("LLM code-apply call failed: %s", e, exc_info=True)
        return original, f"LLM error: {e}"

    if not modified_source.strip():
        return original, "LLM returned empty response — file unchanged."

    # Strip any accidental markdown fences the LLM might add
    modified_source = _strip_markdown_fences(modified_source)

    # Generate a brief diff summary
    summary = _diff_summary(original, modified_source)

    return modified_source, summary


def _strip_markdown_fences(text: str) -> str:
    """Remove wrapping ```python ... ``` if the LLM added them."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _diff_summary(original: str, modified: str) -> str:
    """Return a human-readable summary of what changed."""
    orig_lines = original.splitlines()
    mod_lines = modified.splitlines()

    added = 0
    removed = 0
    orig_set = set(orig_lines)
    mod_set = set(mod_lines)

    for line in mod_lines:
        if line not in orig_set:
            added += 1
    for line in orig_lines:
        if line not in mod_set:
            removed += 1

    if added == 0 and removed == 0:
        return "No changes detected — file is identical."

    parts = []
    if added:
        parts.append(f"+{added} lines added")
    if removed:
        parts.append(f"-{removed} lines removed")
    parts.append(f"({len(mod_lines)} total lines)")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# 4. Apply patch (write file + backup)
# ---------------------------------------------------------------------------

def apply_patch(
    target_file: Path,
    modified_source: str,
    query: str = "",
) -> PatchResult:
    """Write *modified_source* to *target_file*, creating a timestamped backup.

    Args:
        target_file: The file to overwrite.
        modified_source: The new file content.
        query: Original query (for audit logging).

    Returns:
        ``PatchResult`` with success/failure info.
    """
    target_file = Path(target_file).resolve()

    if not target_file.exists():
        return PatchResult(
            success=False,
            target_file=str(target_file),
            message=f"Target file does not exist: {target_file}",
        )

    # --- Backup ---
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{target_file.stem}_{timestamp}{target_file.suffix}"
    backup_path = _BACKUP_DIR / backup_name

    try:
        shutil.copy2(str(target_file), str(backup_path))
    except Exception as e:
        return PatchResult(
            success=False,
            target_file=str(target_file),
            message=f"Backup failed: {e}",
        )

    # --- Write modified file ---
    try:
        target_file.write_text(modified_source, encoding="utf-8")
    except Exception as e:
        # Restore from backup
        shutil.copy2(str(backup_path), str(target_file))
        return PatchResult(
            success=False,
            target_file=str(target_file),
            backup_file=str(backup_path),
            message=f"Write failed (restored backup): {e}",
        )

    # --- Syntax check ---
    import py_compile
    try:
        py_compile.compile(str(target_file), doraise=True)
    except py_compile.PyCompileError as e:
        # Syntax error — roll back
        shutil.copy2(str(backup_path), str(target_file))
        return PatchResult(
            success=False,
            target_file=str(target_file),
            backup_file=str(backup_path),
            message=f"Syntax error after patching (rolled back): {e}",
        )

    # --- Audit log ---
    diff_summary = _diff_summary(
        backup_path.read_text(encoding="utf-8"),
        modified_source,
    )
    _log_application(target_file, backup_path, query, diff_summary)

    return PatchResult(
        success=True,
        target_file=str(target_file),
        backup_file=str(backup_path),
        message="Code applied successfully.",
        diff_summary=diff_summary,
    )


def _log_application(
    target: Path,
    backup: Path,
    query: str,
    diff_summary: str,
) -> None:
    """Append a line to the audit log."""
    _APPLY_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target": str(target),
        "backup": str(backup),
        "query": query[:300],
        "diff_summary": diff_summary,
    }
    with _APPLY_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(
        "Code applied: %s → %s (%s)",
        target.name, backup.name, diff_summary,
    )


# ---------------------------------------------------------------------------
# 5. Revert to backup
# ---------------------------------------------------------------------------

def revert_last_patch(target_file: Path) -> PatchResult:
    """Restore the most recent backup for *target_file*.

    Scans ``_BACKUP_DIR`` for the newest backup matching the file stem
    and overwrites the current file with it.
    """
    target_file = Path(target_file).resolve()
    stem = target_file.stem

    if not _BACKUP_DIR.exists():
        return PatchResult(
            success=False,
            target_file=str(target_file),
            message="No backup directory found.",
        )

    # Find backups matching this file's stem (sorted newest first)
    pattern = f"{stem}_*{target_file.suffix}"
    backups = sorted(_BACKUP_DIR.glob(pattern), reverse=True)

    if not backups:
        return PatchResult(
            success=False,
            target_file=str(target_file),
            message=f"No backups found for {target_file.name}.",
        )

    latest = backups[0]
    try:
        shutil.copy2(str(latest), str(target_file))
        return PatchResult(
            success=True,
            target_file=str(target_file),
            backup_file=str(latest),
            message=f"Reverted to backup: {latest.name}",
        )
    except Exception as e:
        return PatchResult(
            success=False,
            target_file=str(target_file),
            message=f"Revert failed: {e}",
        )
