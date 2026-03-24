"""
Global Time Budget Controller for Centurion Capital LLC RAG Pipeline.

Enforces a hard wall-clock budget per query so no single request can
hang indefinitely.  Thresholds:

    ==================  =============  ==============================
    Budget              Default (s)    Action
    ==================  =============  ==============================
    Total query         120            Abort & return partial answer
    Retrieval slow      20             Switch to FAST_MODE
    LLM generation      90             Abort & return collected tokens
    ==================  =============  ==============================

Usage::

    from rag_pipeline.utils.time_budget import TimeBudget

    budget = TimeBudget()        # uses defaults (120 / 20 / 90)
    budget.start()

    # ... after retrieval finishes ...
    if budget.retrieval_exceeded(retrieval_seconds):
        # switch to fast mode for remaining stages
        ...

    # inside LLM streaming loop:
    for token in llm.generate_stream(q, ctx):
        if budget.llm_exceeded(llm_elapsed):
            break  # abort LLM, return partial

    if budget.is_expired():
        answer += budget.cutoff_message()

Tech stack: Python 3.11 · stdlib only · No external deps.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Defaults (overridable via env vars or constructor)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_QUERY_BUDGET_S = 300        # total seconds per query
DEFAULT_RETRIEVAL_SLOW_S = 30 # retrieval threshold FAST_MODE
DEFAULT_LLM_BUDGET_S = 300          # max seconds for LLM generation

# Graceful cutoff messages
_CUTOFF_TOTAL = (
    "\n\n---\n"
    " **Response truncated** — the total processing time exceeded "
    "the allowed budget ({budget}s).  The answer above may be "
    "incomplete.  Please try a more specific query or enable "
    "FAST_MODE for quicker results."
)

_CUTOFF_LLM = (
    "\n\n---\n"
    " **Response truncated** — the language model took longer than "
    "{budget}s.  The partial answer above contains the tokens "
    "generated so far.  Try shortening your query or reducing the "
    "context window."
)


# ═══════════════════════════════════════════════════════════════════════════
# TimeBudget controller
# ═══════════════════════════════════════════════════════════════════════════

class TimeBudget:
    """Track and enforce time limits across the RAG query pipeline.

    Parameters
    ----------
    total_budget_s : float
        Hard wall-clock cap per query (default 120 s).
    retrieval_slow_s : float
        If a retrieval stage takes longer than this, the caller
        should switch to FAST_MODE (default 20 s).
    llm_budget_s : float
        Max wall-clock time for LLM generation.  After this the
        caller should abort and return partial output (default 90 s).
    """

    def __init__(
        self,
        total_budget_s: float = DEFAULT_QUERY_BUDGET_S,
        retrieval_slow_s: float = DEFAULT_RETRIEVAL_SLOW_S,
        llm_budget_s: float = DEFAULT_LLM_BUDGET_S,
    ) -> None:
        self.total_budget_s = total_budget_s
        self.retrieval_slow_s = retrieval_slow_s
        self.llm_budget_s = llm_budget_s

        self._start: float = 0.0
        self._retrieval_switched = False
        self._llm_aborted = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> "TimeBudget":
        """Mark query start and return *self* for chaining."""
        self._start = time.perf_counter()
        return self

    # ── Elapsed / remaining ───────────────────────────────────────────

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since ``start()``."""
        if self._start <= 0:
            return 0.0
        return time.perf_counter() - self._start

    @property
    def remaining(self) -> float:
        """Seconds remaining in the total budget (≥ 0)."""
        return max(0.0, self.total_budget_s - self.elapsed)

    def is_expired(self) -> bool:
        """Return *True* when the total query budget has been exceeded."""
        return self.elapsed >= self.total_budget_s

    # ── Retrieval threshold ───────────────────────────────────────────

    def retrieval_exceeded(self, retrieval_elapsed_s: float) -> bool:
        """Return *True* if retrieval took longer than the slow threshold.

        Also sets an internal flag so that ``was_retrieval_slow`` can
        be queried later.
        """
        exceeded = retrieval_elapsed_s > self.retrieval_slow_s
        if exceeded and not self._retrieval_switched:
            self._retrieval_switched = True
            logger.warning(
                "TimeBudget: retrieval took %.1fs (> %.1fs) — "
                "switching to FAST_MODE for remaining stages.",
                retrieval_elapsed_s, self.retrieval_slow_s,
            )
        return exceeded

    @property
    def was_retrieval_slow(self) -> bool:
        """Whether retrieval exceeded the slow threshold at any point."""
        return self._retrieval_switched

    # ── LLM threshold ─────────────────────────────────────────────────

    def llm_remaining(self) -> float:
        """Max seconds the LLM stage may still run.

        Returns the *lesser* of the LLM budget and the total remaining
        time, so that the LLM cannot exceed the global cap.
        """
        return max(0.0, min(self.llm_budget_s, self.remaining))

    def llm_exceeded(self, llm_elapsed_s: float) -> bool:
        """Return *True* if the LLM has run past its budget.

        Also checks the global budget — if the total query time is
        already exceeded, this returns *True* immediately.
        """
        if self.is_expired():
            self._llm_aborted = True
            return True
        if llm_elapsed_s >= self.llm_budget_s:
            self._llm_aborted = True
            logger.warning(
                "TimeBudget: LLM generation %.1fs exceeded %.1fs budget "
                "— aborting with partial answer.",
                llm_elapsed_s, self.llm_budget_s,
            )
            return True
        return False

    @property
    def was_llm_aborted(self) -> bool:
        """Whether the LLM stage was aborted due to a timeout."""
        return self._llm_aborted

    # ── Cutoff messages ───────────────────────────────────────────────

    def cutoff_message(self) -> str:
        """Return a user-facing message explaining why the response
        was truncated.  Returns an empty string if no cutoff occurred.
        """
        if self._llm_aborted:
            return _CUTOFF_LLM.format(budget=int(self.llm_budget_s))
        if self.is_expired():
            return _CUTOFF_TOTAL.format(budget=int(self.total_budget_s))
        return ""

    # ── Summary dict (for PipelineTrace / logging) ────────────────────

    def as_dict(self) -> dict:
        """Return a JSON-friendly summary of the budget state."""
        return {
            "total_budget_s": self.total_budget_s,
            "retrieval_slow_s": self.retrieval_slow_s,
            "llm_budget_s": self.llm_budget_s,
            "elapsed_s": round(self.elapsed, 3),
            "remaining_s": round(self.remaining, 3),
            "expired": self.is_expired(),
            "retrieval_switched": self._retrieval_switched,
            "llm_aborted": self._llm_aborted,
        }

    def __repr__(self) -> str:
        return (
            f"TimeBudget(elapsed={self.elapsed:.1f}s / "
            f"{self.total_budget_s}s, "
            f"remaining={self.remaining:.1f}s, "
            f"expired={self.is_expired()})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory helper
# ═══════════════════════════════════════════════════════════════════════════

def create_time_budget(
    total_budget_s: Optional[float] = None,
    retrieval_slow_s: Optional[float] = None,
    llm_budget_s: Optional[float] = None,
) -> TimeBudget:
    """Create a ``TimeBudget`` from env vars / explicit overrides.

    Environment variables (override defaults):
        ``CENTURION_RAG_QUERY_BUDGET``   — total seconds (default 120)
        ``CENTURION_RAG_RETRIEVAL_SLOW`` — retrieval threshold (default 20)
        ``CENTURION_RAG_LLM_BUDGET``     — LLM budget (default 90)
    """
    t = total_budget_s or float(
        os.getenv("CENTURION_RAG_QUERY_BUDGET", str(DEFAULT_QUERY_BUDGET_S))
    )
    r = retrieval_slow_s or float(
        os.getenv("CENTURION_RAG_RETRIEVAL_SLOW", str(DEFAULT_RETRIEVAL_SLOW_S))
    )
    l_ = llm_budget_s or float(
        os.getenv("CENTURION_RAG_LLM_BUDGET", str(DEFAULT_LLM_BUDGET_S))
    )
    return TimeBudget(total_budget_s=t, retrieval_slow_s=r, llm_budget_s=l_)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests  (run: python -m rag_pipeline.utils.time_budget)
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

    # ── 1. Basic construction & defaults ─────────────────────────────
    print("\n=== 1. Construction & Defaults ===")
    tb = TimeBudget()
    check("default total_budget_s == 300", tb.total_budget_s == 300)
    check("default retrieval_slow_s == 30", tb.retrieval_slow_s == 30)
    check("default llm_budget_s == 300", tb.llm_budget_s == 300)
    check("elapsed == 0 before start()", tb.elapsed == 0.0)
    check("remaining == total before start()", tb.remaining == 300.0)
    check("not expired before start()", not tb.is_expired())

    # ── 2. Start & elapsed tracking ──────────────────────────────────
    print("\n=== 2. Start & Elapsed ===")
    tb2 = TimeBudget(total_budget_s=5.0).start()
    check("elapsed > 0 after start", tb2.elapsed > 0)
    check("remaining < 5.0 after start", tb2.remaining < 5.0)
    check("not expired (just started 5s budget)", not tb2.is_expired())
    check("start() returns self", tb2.start() is tb2)

    # ── 3. Expiry ────────────────────────────────────────────────────
    print("\n=== 3. Expiry ===")
    tb3 = TimeBudget(total_budget_s=0.0).start()
    check("expired with 0s budget", tb3.is_expired())
    check("remaining == 0 when expired", tb3.remaining == 0.0)

    tb3b = TimeBudget(total_budget_s=0.001).start()
    time.sleep(0.005)
    check("expired after sleep > budget", tb3b.is_expired())

    # ── 4. Retrieval threshold ───────────────────────────────────────
    print("\n=== 4. Retrieval Threshold ===")
    tb4 = TimeBudget(retrieval_slow_s=10.0).start()
    check("retrieval NOT exceeded at 5s", not tb4.retrieval_exceeded(5.0))
    check("was_retrieval_slow=False", not tb4.was_retrieval_slow)
    check("retrieval EXCEEDED at 25s", tb4.retrieval_exceeded(25.0))
    check("was_retrieval_slow=True after exceed", tb4.was_retrieval_slow)
    # Subsequent calls still return True
    check("retrieval still exceeded", tb4.retrieval_exceeded(25.0))

    # ── 5. LLM threshold ────────────────────────────────────────────
    print("\n=== 5. LLM Threshold ===")
    tb5 = TimeBudget(total_budget_s=120.0, llm_budget_s=90.0).start()
    check("llm NOT exceeded at 10s", not tb5.llm_exceeded(10.0))
    check("was_llm_aborted=False", not tb5.was_llm_aborted)
    check("llm EXCEEDED at 91s", tb5.llm_exceeded(91.0))
    check("was_llm_aborted=True", tb5.was_llm_aborted)

    # LLM exceeded by global budget
    tb5b = TimeBudget(total_budget_s=0.001, llm_budget_s=90.0).start()
    time.sleep(0.005)
    check("llm exceeded when global expired", tb5b.llm_exceeded(0.0))

    # ── 6. llm_remaining() ──────────────────────────────────────────
    print("\n=== 6. LLM Remaining ===")
    tb6 = TimeBudget(total_budget_s=50.0, llm_budget_s=90.0).start()
    lr = tb6.llm_remaining()
    # llm_remaining = min(90, remaining) ≈ min(90, ~50) ≈ ~50
    check("llm_remaining capped by global remaining", lr <= 50.0)
    check("llm_remaining > 0", lr > 0)

    tb6b = TimeBudget(total_budget_s=200.0, llm_budget_s=30.0).start()
    lr2 = tb6b.llm_remaining()
    check("llm_remaining capped by llm_budget", lr2 <= 30.0)

    # ── 7. Cutoff messages ──────────────────────────────────────────
    print("\n=== 7. Cutoff Messages ===")
    tb7 = TimeBudget(total_budget_s=120.0, llm_budget_s=90.0).start()
    check("no cutoff message initially", tb7.cutoff_message() == "")

    # LLM abort
    tb7.llm_exceeded(100.0)
    msg_llm = tb7.cutoff_message()
    check("LLM cutoff msg non-empty", len(msg_llm) > 0)
    check("LLM cutoff msg mentions 90s", "90s" in msg_llm)
    check("LLM cutoff msg has truncated", "truncated" in msg_llm.lower())

    # Global expiry (no LLM abort)
    tb7b = TimeBudget(total_budget_s=0.001).start()
    time.sleep(0.005)
    msg_total = tb7b.cutoff_message()
    check("global cutoff msg non-empty", len(msg_total) > 0)
    check("global cutoff msg mentions budget", "budget" in msg_total.lower())

    # ── 8. as_dict() ────────────────────────────────────────────────
    print("\n=== 8. as_dict ===")
    tb8 = TimeBudget(total_budget_s=60.0, retrieval_slow_s=15.0, llm_budget_s=45.0).start()
    d = tb8.as_dict()
    check("as_dict has total_budget_s", d["total_budget_s"] == 60.0)
    check("as_dict has retrieval_slow_s", d["retrieval_slow_s"] == 15.0)
    check("as_dict has llm_budget_s", d["llm_budget_s"] == 45.0)
    check("as_dict has elapsed_s", "elapsed_s" in d)
    check("as_dict has remaining_s", "remaining_s" in d)
    check("as_dict has expired bool", isinstance(d["expired"], bool))
    check("as_dict has retrieval_switched", isinstance(d["retrieval_switched"], bool))
    check("as_dict has llm_aborted", isinstance(d["llm_aborted"], bool))

    # ── 9. __repr__ ─────────────────────────────────────────────────
    print("\n=== 9. __repr__ ===")
    tb9 = TimeBudget(total_budget_s=120.0).start()
    r = repr(tb9)
    check("repr contains 'TimeBudget'", "TimeBudget" in r)
    check("repr contains 'elapsed'", "elapsed" in r)
    check("repr contains 'remaining'", "remaining" in r)

    # ── 10. Custom thresholds ───────────────────────────────────────
    print("\n=== 10. Custom Thresholds ===")
    tb10 = TimeBudget(total_budget_s=60, retrieval_slow_s=5, llm_budget_s=30)
    check("custom total", tb10.total_budget_s == 60)
    check("custom retrieval", tb10.retrieval_slow_s == 5)
    check("custom llm", tb10.llm_budget_s == 30)

    # ── 11. Factory with env vars ───────────────────────────────────
    print("\n=== 11. Factory (create_time_budget) ===")
    old_vals = {}
    env_keys = [
        "CENTURION_RAG_QUERY_BUDGET",
        "CENTURION_RAG_RETRIEVAL_SLOW",
        "CENTURION_RAG_LLM_BUDGET",
    ]
    for k in env_keys:
        old_vals[k] = os.environ.get(k)

    try:
        os.environ["CENTURION_RAG_QUERY_BUDGET"] = "60"
        os.environ["CENTURION_RAG_RETRIEVAL_SLOW"] = "10"
        os.environ["CENTURION_RAG_LLM_BUDGET"] = "45"

        tb11 = create_time_budget()
        check("factory reads QUERY_BUDGET env", tb11.total_budget_s == 60.0)
        check("factory reads RETRIEVAL_SLOW env", tb11.retrieval_slow_s == 10.0)
        check("factory reads LLM_BUDGET env", tb11.llm_budget_s == 45.0)

        # Explicit overrides take precedence
        tb11b = create_time_budget(total_budget_s=99)
        check("explicit override > env var", tb11b.total_budget_s == 99.0)

    finally:
        for k in env_keys:
            if old_vals[k] is not None:
                os.environ[k] = old_vals[k]
            elif k in os.environ:
                del os.environ[k]

    # ── 12. Constants ───────────────────────────────────────────────
    print("\n=== 12. Constants ===")
    check("DEFAULT_QUERY_BUDGET_S == 300", DEFAULT_QUERY_BUDGET_S == 300)
    check("DEFAULT_RETRIEVAL_SLOW_S == 30", DEFAULT_RETRIEVAL_SLOW_S == 30)
    check("DEFAULT_LLM_BUDGET_S == 300", DEFAULT_LLM_BUDGET_S == 300)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_tests()
