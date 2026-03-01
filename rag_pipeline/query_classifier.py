"""
Deterministic Query Classifier for Centurion Capital LLC RAG Pipeline.

Classifies a raw user query string into a structured dict containing:
    - **intent** – ``"code_generation"`` | ``"analysis"`` | ``"conceptual"``
    - **pipeline_stage** – ``list[str]`` of matched pipeline stages
    - **needs_code** – ``bool`` (``True`` when intent is code_generation
      *or* the query explicitly references code artefacts)

All classification is **rule-based**: deterministic keyword matching,
no ML models, no LLM calls.

Pipeline-stage keywords are kept in sync with ``chunking.py`` so the
same vocabulary drives both chunk tagging *and* query routing.

Usage::

    from rag_pipeline.query_classifier import classify_query

    result = classify_query("Write a function to calculate Sharpe ratio")
    # {
    #     "intent": "code_generation",
    #     "pipeline_stage": ["evaluation"],
    #     "needs_code": True,
    # }
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Intent keywords
# ═══════════════════════════════════════════════════════════════════════════
#
# Order matters: rules are evaluated top-to-bottom and the **first**
# matching intent wins.  This gives ``code_generation`` priority over
# ``analysis`` when both could match (e.g. "implement an optimization").
# ═══════════════════════════════════════════════════════════════════════════

_INTENT_RULES: List[Tuple[str, re.Pattern]] = [
    # ── code_generation ──────────────────────────────────────────────
    (
        "code_generation",
        re.compile(
            r"\b(?:"
            r"write|implement|create\s+(?:a\s+)?(?:function|class|module|script)"
            r"|code(?:\s+(?:for|to|that|a|an|the))?"
            r"|function\b"
            r"|def\s+\w+"
            r"|snippet"
            r"|generate\s+(?:a\s+)?(?:function|class|code|script)"
            r"|refactor"
            r"|debug"
            r"|fix\s+(?:the\s+)?(?:bug|error|issue|code)"
            r"|add\s+(?:a\s+)?(?:method|parameter|class|test)"
            r"|build\s+(?:a\s+)?(?:\w+\s+)?(?:function|module|pipeline|class)"
            r"|program"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # ── analysis ─────────────────────────────────────────────────────
    (
        "analysis",
        re.compile(
            r"\b(?:"
            r"optimize|optimise"
            r"|improve"
            r"|reduce\s+(?:the\s+)?(?:drawdown|risk|volatility|loss|exposure)"
            r"|increase\s+(?:the\s+)?(?:sharpe|returns|alpha|hit\s+rate)"
            r"|compare"
            r"|evaluate"
            r"|benchmark"
            r"|backtest"
            r"|analyze|analyse"
            r"|diagnose"
            r"|profile"
            r"|measure"
            r"|performance"
            r"|trade-off|tradeoff"
            r"|sensitivity"
            r"|what\s+(?:is|are)\s+the\s+(?:best|optimal|top)"
            r"|how\s+(?:much|many|well|does|did|do)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # ── conceptual (catch-all explanation / theory queries) ──────────
    (
        "conceptual",
        re.compile(
            r"\b(?:"
            r"why"
            r"|explain"
            r"|what\s+(?:is|are|does|do)\b"
            r"|how\s+(?:does|do|is|are)\b"
            r"|define"
            r"|definition"
            r"|describe"
            r"|overview"
            r"|summary"
            r"|summarize|summarise"
            r"|concept"
            r"|theory"
            r"|meaning"
            r"|difference\s+between"
            r"|when\s+(?:should|to|do)"
            r"|tell\s+me"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]

_DEFAULT_INTENT = "conceptual"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Pipeline-stage keywords
# ═══════════════════════════════════════════════════════════════════════════
#
# Intentionally kept in sync with ``chunking._PIPELINE_STAGE_RULES``.
# Any stage that scores ≥ 1 keyword hit is included in the output list.
# ═══════════════════════════════════════════════════════════════════════════

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

# Pre-compile for fast matching
_STAGE_PATTERNS: Dict[str, re.Pattern] = {}
for _stage, _keywords in _PIPELINE_STAGE_RULES.items():
    _parts = []
    for kw in _keywords:
        escaped = re.escape(kw)
        if len(kw) <= 5:
            _parts.append(rf"\b{escaped}\b")
        else:
            _parts.append(escaped)
    _STAGE_PATTERNS[_stage] = re.compile("|".join(_parts), re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════
# 3. "needs_code" secondary keywords
# ═══════════════════════════════════════════════════════════════════════════
#
# Even if the intent isn't ``code_generation``, some queries implicitly
# need code context (e.g. "explain the `calculate_sharpe` function").
# ═══════════════════════════════════════════════════════════════════════════

_CODE_HINT_RE = re.compile(
    r"\b(?:"
    r"function|class|method|variable|module|import|def |return |lambda"
    r"|script|snippet|source code|codebase|repository|repo\b"
    r"|`.+`"                       # inline code in backticks
    r"|\.py\b"                     # python file reference
    r")\b",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# 3b. Strict code-mode keywords
# ═══════════════════════════════════════════════════════════════════════════
#
# When a query contains any of these strong code-intent signals the
# classifier forces ``intent = "code_generation"``, ``needs_code = True``,
# ``priority = "code_strict"``, and ``strict_code_mode = True``.
# This overrides the normal first-match intent logic so that mixed
# queries like "what is X? how can I implement using python?" are
# correctly routed to code retrieval.
# ═══════════════════════════════════════════════════════════════════════════

_STRICT_CODE_RE = re.compile(
    r"\b(?:"
    r"implement"
    r"|python"
    r"|code"
    r"|function"
    r"|using\s+python"
    r")\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Public API
# ═══════════════════════════════════════════════════════════════════════════

def classify_intent(query: str) -> str:
    """Classify the user's intent from the query string.

    Evaluates intent rules in priority order and returns the first
    match.  Falls back to ``"conceptual"`` if no rule fires.

    Args:
        query: Raw user query string.

    Returns:
        One of ``"code_generation"``, ``"analysis"``, ``"conceptual"``.
    """
    for intent, pattern in _INTENT_RULES:
        if pattern.search(query):
            return intent
    return _DEFAULT_INTENT


def classify_pipeline_stages(query: str) -> List[str]:
    """Detect all matching pipeline stages in the query.

    Every stage whose keyword pattern matches at least once is
    included.  Stages are returned in a deterministic order
    (alphabetically sorted).

    Args:
        query: Raw user query string.

    Returns:
        Sorted list of stage names (may be empty).
    """
    stages: List[str] = []
    for stage, pattern in _STAGE_PATTERNS.items():
        if pattern.search(query):
            stages.append(stage)
    stages.sort()
    return stages


def needs_code(query: str, intent: str) -> bool:
    """Determine whether the query needs code context.

    Returns ``True`` when:
    - intent is ``"code_generation"``, **or**
    - the query contains explicit code-artefact references
      (function names, ``.py`` files, backtick code snippets, etc.)

    Args:
        query: Raw user query string.
        intent: Already-classified intent string.

    Returns:
        Boolean.
    """
    if intent == "code_generation":
        return True
    return bool(_CODE_HINT_RE.search(query))


def classify_query(query: str) -> Dict[str, Any]:
    """Classify a user query into intent, pipeline stages, and code need.

    This is the primary entry point for the module.

    Args:
        query: Raw user query string.

    Returns:
        Dict with keys::

            {
                "intent":           str,   # "code_generation" | "analysis" | "conceptual"
                "pipeline_stage":   list[str],
                "needs_code":       bool,
                "strict_code_mode": bool,   # True when strong code keywords detected
                "priority":         str,    # "code_strict" | "normal"
            }

        When the query contains strong code-intent keywords ("implement",
        "python", "code", "function", "using python") the classifier
        forces ``intent = "code_generation"``, ``needs_code = True``,
        ``priority = "code_strict"``, and ``strict_code_mode = True``
        regardless of which intent pattern matched first.

    Examples::

        >>> classify_query("Write a function to calculate Sharpe ratio")
        {'intent': 'code_generation', 'pipeline_stage': ['evaluation'], 'needs_code': True, 'strict_code_mode': True, 'priority': 'code_strict'}

        >>> classify_query("Why does the z-score signal lag on volatile days?")
        {'intent': 'conceptual', 'pipeline_stage': ['signal_generation'], 'needs_code': False, 'strict_code_mode': False, 'priority': 'normal'}

        >>> classify_query("Optimize the drawdown limit for momentum strategy")
        {'intent': 'analysis', 'pipeline_stage': ['risk_management'], 'needs_code': False, 'strict_code_mode': False, 'priority': 'normal'}
    """
    if not query or not query.strip():
        logger.warning("classify_query called with empty query.")
        return {
            "intent": _DEFAULT_INTENT,
            "pipeline_stage": [],
            "needs_code": False,
            "strict_code_mode": False,
            "priority": "normal",
        }

    query_clean = query.strip()
    intent = classify_intent(query_clean)
    stages = classify_pipeline_stages(query_clean)
    code_needed = needs_code(query_clean, intent)

    # ── Strict code-mode override ────────────────────────────────
    strict = bool(_STRICT_CODE_RE.search(query_clean))
    if strict:
        intent = "code_generation"
        code_needed = True
        priority = "code_strict"
    else:
        priority = "normal"

    logger.debug(
        "classify_query(%r) → intent=%s, stages=%s, needs_code=%s, "
        "strict_code_mode=%s, priority=%s",
        query_clean[:80], intent, stages, code_needed, strict, priority,
    )
    return {
        "intent": intent,
        "pipeline_stage": stages,
        "needs_code": code_needed,
        "strict_code_mode": strict,
        "priority": priority,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Unit tests (run with ``python -m rag_pipeline.query_classifier``)
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

    # ── Intent classification ────────────────────────────────────────
    print("\n=== Intent Classification ===")

    # code_generation
    check(
        '"Write a function" → code_generation',
        classify_intent("Write a function to compute returns") == "code_generation",
    )
    check(
        '"Implement a class" → code_generation',
        classify_intent("Implement a class for portfolio rebalancing") == "code_generation",
    )
    check(
        '"code for" → code_generation',
        classify_intent("Give me code for z-score signal") == "code_generation",
    )
    check(
        '"Refactor the module" → code_generation',
        classify_intent("Refactor the risk module") == "code_generation",
    )
    check(
        '"Debug the error" → code_generation',
        classify_intent("Debug the stop-loss error") == "code_generation",
    )
    check(
        '"Build a pipeline" → code_generation',
        classify_intent("Build a data pipeline for OHLCV") == "code_generation",
    )

    # analysis
    check(
        '"Optimize the strategy" → analysis',
        classify_intent("Optimize the momentum strategy") == "analysis",
    )
    check(
        '"Reduce drawdown" → analysis',
        classify_intent("Reduce the drawdown on this portfolio") == "analysis",
    )
    check(
        '"Improve sharpe" → analysis',
        classify_intent("How can I improve the Sharpe?") == "analysis",
    )
    check(
        '"Compare two strategies" → analysis',
        classify_intent("Compare mean reversion vs momentum") == "analysis",
    )
    check(
        '"Evaluate performance" → analysis',
        classify_intent("Evaluate this strategy performance") == "analysis",
    )
    check(
        '"Backtest this" → analysis',
        classify_intent("Backtest this on 2020 data") == "analysis",
    )
    check(
        '"How much drawdown" → analysis',
        classify_intent("How much drawdown does this strategy have?") == "analysis",
    )

    # conceptual
    check(
        '"Why does…" → conceptual',
        classify_intent("Why does the momentum factor work?") == "conceptual",
    )
    check(
        '"Explain" → conceptual',
        classify_intent("Explain the Kelly criterion") == "conceptual",
    )
    check(
        '"What is Sharpe ratio" → conceptual',
        classify_intent("What is the Sharpe ratio?") == "conceptual",
    )
    check(
        '"Define alpha" → conceptual',
        classify_intent("Define alpha in the context of returns") == "conceptual",
    )
    check(
        '"Describe mean reversion" → conceptual',
        classify_intent("Describe mean reversion strategy") == "conceptual",
    )
    check(
        '"Difference between" → conceptual',
        classify_intent("Difference between Sharpe and Sortino") == "conceptual",
    )

    # default fallback
    check(
        'Ambiguous → conceptual (default)',
        classify_intent("Hello") == "conceptual",
    )
    check(
        'Empty → conceptual (default)',
        classify_intent("") == "conceptual",
    )

    # ── Priority: code_generation > analysis ─────────────────────────
    print("\n=== Intent Priority ===")
    check(
        '"Write code to optimize" → code_generation (priority)',
        classify_intent("Write code to optimize the portfolio") == "code_generation",
    )
    check(
        '"Implement an improvement" → code_generation (priority)',
        classify_intent("Implement an improvement to reduce drawdown") == "code_generation",
    )

    # ── Pipeline stage detection ─────────────────────────────────────
    print("\n=== Pipeline Stage Detection ===")

    stages = classify_pipeline_stages("Calculate the Sharpe ratio for momentum returns")
    check('"sharpe" + "returns" → evaluation', "evaluation" in stages)

    stages = classify_pipeline_stages("Reduce the max drawdown risk on the portfolio")
    check('"drawdown" + "risk" → risk_management', "risk_management" in stages)

    stages = classify_pipeline_stages("How does the z-score signal generate entries?")
    check('"z-score" + "signal" → signal_generation', "signal_generation" in stages)

    stages = classify_pipeline_stages(
        "Backtest the walk-forward strategy with historical simulation"
    )
    check('"backtest" + "walk-forward" → backtesting', "backtesting" in stages)

    stages = classify_pipeline_stages(
        "Optimize portfolio allocation using mean-variance"
    )
    check(
        '"portfolio" + "allocation" + "mean-variance" → portfolio_construction',
        "portfolio_construction" in stages,
    )

    stages = classify_pipeline_stages("Clean missing data and handle outliers")
    check('"missing data" + "outlier" → data_processing', "data_processing" in stages)

    stages = classify_pipeline_stages("Reduce slippage on market orders for TWAP")
    check('"slippage" + "market order" + "twap" → execution', "execution" in stages)

    stages = classify_pipeline_stages(
        "Train an LSTM with cross-validation for feature importance"
    )
    check('"lstm" + "cross-validation" + "feature importance" → machine_learning',
          "machine_learning" in stages)

    stages = classify_pipeline_stages("Improve liquidity in the order book spread")
    check('"liquidity" + "order book" + "spread" → market_microstructure',
          "market_microstructure" in stages)

    # Multi-stage
    stages = classify_pipeline_stages(
        "Optimize the Sharpe ratio while reducing drawdown risk"
    )
    check(
        'Multi-stage: evaluation + risk_management',
        "evaluation" in stages and "risk_management" in stages,
    )

    # No stage
    stages = classify_pipeline_stages("Hello world")
    check('No keywords → empty list', stages == [])

    # Sorted order
    stages = classify_pipeline_stages(
        "The z-score signal has too much drawdown risk and poor Sharpe"
    )
    check(
        'Stages sorted alphabetically',
        stages == sorted(stages),
    )

    # ── needs_code detection ─────────────────────────────────────────
    print("\n=== needs_code Detection ===")

    check(
        'code_generation intent → needs_code=True',
        needs_code("Write a function", "code_generation") is True,
    )
    check(
        'analysis intent, no code hint → needs_code=False',
        needs_code("Optimize returns", "analysis") is False,
    )
    check(
        'conceptual + backtick code → needs_code=True',
        needs_code("Explain the `calculate_sharpe` function", "conceptual") is True,
    )
    check(
        'conceptual + ".py" reference → needs_code=True',
        needs_code("What does strategy.py do?", "conceptual") is True,
    )
    check(
        'analysis + "function" mention → needs_code=True',
        needs_code("Optimize the function for speed", "analysis") is True,
    )

    # ── Full classify_query integration ──────────────────────────────
    print("\n=== Full classify_query ===")

    r = classify_query("Write a function to calculate Sharpe ratio")
    check('Full: intent=code_generation', r["intent"] == "code_generation")
    check('Full: evaluation stage', "evaluation" in r["pipeline_stage"])
    check('Full: needs_code=True', r["needs_code"] is True)
    check('Full: strict_code_mode=True ("function")', r["strict_code_mode"] is True)
    check('Full: priority=code_strict', r["priority"] == "code_strict")

    r = classify_query("Why does the z-score signal lag on volatile days?")
    check('Full: intent=conceptual', r["intent"] == "conceptual")
    check('Full: signal_generation stage', "signal_generation" in r["pipeline_stage"])
    check('Full: needs_code=False', r["needs_code"] is False)
    check('Full: strict_code_mode=False', r["strict_code_mode"] is False)
    check('Full: priority=normal', r["priority"] == "normal")

    r = classify_query("Optimize the drawdown limit for momentum strategy")
    check('Full: intent=analysis', r["intent"] == "analysis")
    check('Full: risk_management stage', "risk_management" in r["pipeline_stage"])
    check('Full: needs_code=False', r["needs_code"] is False)
    check('Full: strict_code_mode=False', r["strict_code_mode"] is False)
    check('Full: priority=normal', r["priority"] == "normal")

    r = classify_query("Explain the `risk_manager.py` module")
    check('Full: conceptual + needs_code (backtick + .py)',
          r["intent"] == "conceptual" and r["needs_code"] is True)

    r = classify_query("")
    check('Empty query → default', r["intent"] == "conceptual")
    check('Empty query → no stages', r["pipeline_stage"] == [])
    check('Empty query → no code', r["needs_code"] is False)
    check('Empty query → strict_code_mode=False', r["strict_code_mode"] is False)
    check('Empty query → priority=normal', r["priority"] == "normal")

    r = classify_query("   ")
    check('Whitespace query → default', r["intent"] == "conceptual")

    # ── Strict code-mode override ────────────────────────────────────
    print("\n=== Strict Code Mode ===")

    # "implement" keyword → forces code_generation even if conceptual matched first
    r = classify_query("what is triple barrier labeling? how can i implement using python?")
    check('"implement" + "python" → code_generation',
          r["intent"] == "code_generation")
    check('"implement" + "python" → needs_code=True',
          r["needs_code"] is True)
    check('"implement" + "python" → strict_code_mode=True',
          r["strict_code_mode"] is True)
    check('"implement" + "python" → priority=code_strict',
          r["priority"] == "code_strict")

    # "code" keyword alone
    r = classify_query("show me the code for CUSUM filter")
    check('"code" → strict_code_mode=True',
          r["strict_code_mode"] is True)
    check('"code" → intent=code_generation',
          r["intent"] == "code_generation")

    # "function" keyword in conceptual context
    r = classify_query("what function calculates daily volatility?")
    check('"function" → strict_code_mode=True',
          r["strict_code_mode"] is True)
    check('"function" → intent=code_generation',
          r["intent"] == "code_generation")

    # "python" keyword alone
    r = classify_query("how to apply meta-labelling in python?")
    check('"python" → strict_code_mode=True',
          r["strict_code_mode"] is True)
    check('"python" → priority=code_strict',
          r["priority"] == "code_strict")

    # "using python" phrase
    r = classify_query("calculate gaps series using python")
    check('"using python" → strict_code_mode=True',
          r["strict_code_mode"] is True)
    check('"using python" → code_generation',
          r["intent"] == "code_generation")

    # Negative: no strict keywords → strict_code_mode=False
    r = classify_query("explain the concept of mean reversion")
    check('No strict keywords → strict_code_mode=False',
          r["strict_code_mode"] is False)
    check('No strict keywords → priority=normal',
          r["priority"] == "normal")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_tests()
