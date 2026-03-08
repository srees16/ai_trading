"""
Fastpath — Lightweight Short-Query Bypass for Centurion RAG Pipeline.

Intercepts simple quant/statistical questions and returns a predefined
knowledge template **instantly**, bypassing the full retrieval + LLM
pipeline entirely.  This eliminates the 60–90 s round-trip through
Chroma, cross-encoder reranking, and Ollama for queries with
well-known, deterministic answers.

Activation criteria (ALL must be true):
    1. Query contains a recognised quant trigger keyword
    2. Query length < 25 words
    3. Intent is NOT ``analysis`` (deep strategy analysis needs context)

When activated, ``try_fastpath()`` returns a concise Markdown answer
with a code snippet.  When it returns ``None``, the caller falls
through to the normal RAG pipeline.

Usage::

    from rag_pipeline.core.fastpath import try_fastpath

    answer = try_fastpath(query_text)
    if answer is not None:
        return answer          # done — skip RAG entirely
    # ... normal pipeline ...

Tech stack: Python 3.11 · stdlib + query_classifier · No LLM calls.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from rag_pipeline.core.query_classifier import classify_query

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

MAX_QUERY_WORDS = 25
"""Short-query ceiling — anything above this falls through to the full
RAG pipeline where richer context retrieval is worthwhile."""

# Intents that should NEVER be fast-pathed.  These require document
# context to produce a meaningful answer.
_BLOCKED_INTENTS = frozenset({"analysis"})

# ═══════════════════════════════════════════════════════════════════════════
# Trigger keywords → template key mapping
#
# Each entry maps one or more regex patterns to a template key.
# Patterns are compiled once at import time.  The FIRST match wins,
# so more specific patterns should appear before broader ones.
# ═══════════════════════════════════════════════════════════════════════════

_TRIGGER_RULES: List[Tuple[str, re.Pattern]] = [
    # --- Volatility / standard deviation ---
    (
        "volatility",
        re.compile(
            r"\b(?:volatil(?:ity|e)|daily\s+vol|std\s*dev"
            r"|standard\s+deviation)\b",
            re.IGNORECASE,
        ),
    ),
    # --- Sharpe ratio ---
    (
        "sharpe",
        re.compile(r"\bsharpe(?:\s+ratio)?\b", re.IGNORECASE),
    ),
    # --- Daily returns ---
    (
        "daily_returns",
        re.compile(
            r"\b(?:daily\s+returns?|log\s+returns?|pct_change)\b",
            re.IGNORECASE,
        ),
    ),
    # --- Rolling calculations ---
    (
        "rolling",
        re.compile(
            r"\b(?:rolling\s+(?:mean|avg|average|std|window|sum))\b",
            re.IGNORECASE,
        ),
    ),
    # --- ATR (Average True Range) ---
    (
        "atr",
        re.compile(
            r"\b(?:atr|average\s+true\s+range)\b",
            re.IGNORECASE,
        ),
    ),
    # --- Beta ---
    (
        "beta",
        re.compile(
            r"\b(?:beta(?:\s+coefficient)?|market\s+beta)\b",
            re.IGNORECASE,
        ),
    ),
    # --- Max drawdown ---
    (
        "max_drawdown",
        re.compile(
            r"\b(?:max(?:imum)?\s+drawdown|mdd)\b",
            re.IGNORECASE,
        ),
    ),
    # --- CAGR ---
    (
        "cagr",
        re.compile(
            r"\b(?:cagr|compound\s+annual\s+growth)\b",
            re.IGNORECASE,
        ),
    ),
    # --- Sortino ratio ---
    (
        "sortino",
        re.compile(
            r"\bsortino(?:\s+ratio)?\b",
            re.IGNORECASE,
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Generic code-snippet bypass
#
# Catches broad "how to calculate X" / "is there a function for X" /
# "formula for X" queries when X is generic quant knowledge (not a
# proprietary strategy).  Returns a direct Python snippet without
# invoking the full RAG pipeline.
# ═══════════════════════════════════════════════════════════════════════════

_GENERIC_CODE_PATTERN = re.compile(
    r"\b(?:"
    r"is\s+there\s+a\s+function"
    r"|how\s+(?:to|do\s+(?:i|you|we))\s+calculate"
    r"|formula\s+for"
    r"|how\s+(?:to|do\s+(?:i|you|we))\s+compute"
    r"|function\s+(?:for|to\s+calculate)"
    r"|code\s+(?:for|to\s+calculate)"
    r"|snippet\s+for"
    r")\b",
    re.IGNORECASE,
)
"""Broad code-request phrasing patterns.  If one matches AND no
strategy keyword is present, the query qualifies for a generic
code-snippet response."""

_STRATEGY_KEYWORDS = re.compile(
    r"\b(?:"
    r"strategy|backtest|signal|crossover|momentum|mean\s*reversion"
    r"|breakout|pairs?\s*trad(?:e|ing)|arbitrage|execution"
    r"|order\s*book|position\s*siz(?:e|ing)|portfolio\s*optim"
    r"|rebalance|regime|factor\s*model|alpha\s*model"
    r"|live\s*trad(?:e|ing)|paper\s*trad(?:e|ing)"
    r")\b",
    re.IGNORECASE,
)
"""Strategy-specific keywords.  When present the query likely needs
document context from the RAG pipeline — do NOT fastpath it."""

# Topic detectors for generic snippets (checked AFTER _GENERIC_CODE_PATTERN)
_GENERIC_TOPIC_RULES: List[Tuple[str, re.Pattern]] = [
    ("generic_volatility", re.compile(
        r"\b(?:volatil(?:ity|e)|daily\s+vol|std\s*dev"
        r"|standard\s+deviation)\b", re.IGNORECASE)),
    ("generic_returns", re.compile(
        r"\b(?:returns?|pct_change|percent(?:age)?\s+change)\b",
        re.IGNORECASE)),
    ("generic_sharpe", re.compile(
        r"\bsharpe(?:\s+ratio)?\b", re.IGNORECASE)),
    ("generic_drawdown", re.compile(
        r"\b(?:drawdown|mdd)\b", re.IGNORECASE)),
    ("generic_beta", re.compile(
        r"\bbeta\b", re.IGNORECASE)),
    ("generic_moving_avg", re.compile(
        r"\b(?:moving\s+average|sma|ema|ewm)\b", re.IGNORECASE)),
    ("generic_correlation", re.compile(
        r"\b(?:correlation|corr)\b", re.IGNORECASE)),
    ("generic_cagr", re.compile(
        r"\b(?:cagr|compound\s+annual\s+growth|annuali[sz]ed\s+return)\b",
        re.IGNORECASE)),
    ("generic_sortino", re.compile(
        r"\bsortino(?:\s+ratio)?\b", re.IGNORECASE)),
    ("generic_atr", re.compile(
        r"\b(?:atr|average\s+true\s+range)\b", re.IGNORECASE)),
]

_GENERIC_TEMPLATES: Dict[str, str] = {
    "generic_volatility": (
        "**Daily Volatility — Python snippet:**\n\n"
        "```python\n"
        "import pandas as pd\n\n"
        "returns = prices.pct_change()\n"
        "daily_vol = returns.std()\n"
        "rolling_vol = returns.rolling(20).std()\n"
        "annualised_vol = daily_vol * (252 ** 0.5)\n"
        "```\n\n"
        "No RAG retrieval needed — this is standard quant knowledge.\n\n"
        "**Confidence: High**"
    ),
    "generic_returns": (
        "**Daily Returns — Python snippet:**\n\n"
        "```python\n"
        "# Simple (percentage) returns\n"
        "returns = prices.pct_change()\n\n"
        "# Log returns\n"
        "import numpy as np\n"
        "log_returns = np.log(prices / prices.shift(1))\n"
        "```\n\n"
        "**Confidence: High**"
    ),
    "generic_sharpe": (
        "**Sharpe Ratio — Python snippet:**\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "rf = 0.05 / 252  # daily risk-free\n"
        "excess = returns - rf\n"
        "sharpe = np.sqrt(252) * excess.mean() / excess.std()\n"
        "```\n\n"
        "**Confidence: High**"
    ),
    "generic_drawdown": (
        "**Max Drawdown — Python snippet:**\n\n"
        "```python\n"
        "cumulative = (1 + returns).cumprod()\n"
        "peak = cumulative.cummax()\n"
        "drawdown = (cumulative - peak) / peak\n"
        "max_dd = drawdown.min()\n"
        "```\n\n"
        "**Confidence: High**"
    ),
    "generic_beta": (
        "**Beta — Python snippet:**\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "cov = np.cov(stock_returns, market_returns)[0, 1]\n"
        "beta = cov / np.var(market_returns)\n"
        "```\n\n"
        "**Confidence: High**"
    ),
    "generic_moving_avg": (
        "**Moving Averages — Python snippet:**\n\n"
        "```python\n"
        "sma_20 = prices.rolling(20).mean()\n"
        "ema_20 = prices.ewm(span=20, adjust=False).mean()\n"
        "```\n\n"
        "**Confidence: High**"
    ),
    "generic_correlation": (
        "**Correlation — Python snippet:**\n\n"
        "```python\n"
        "corr = returns_a.corr(returns_b)       # pairwise\n"
        "corr_matrix = returns_df.corr()        # full matrix\n"
        "rolling_corr = returns_a.rolling(60).corr(returns_b)\n"
        "```\n\n"
        "**Confidence: High**"
    ),
    "generic_cagr": (
        "**CAGR — Python snippet:**\n\n"
        "```python\n"
        "total_return = portfolio[-1] / portfolio[0]\n"
        "n_years = len(portfolio) / 252\n"
        "cagr = total_return ** (1 / n_years) - 1\n"
        "```\n\n"
        "**Confidence: High**"
    ),
    "generic_sortino": (
        "**Sortino Ratio — Python snippet:**\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "rf = 0.05 / 252\n"
        "excess = returns - rf\n"
        "downside_std = np.sqrt(np.mean(excess[excess < 0] ** 2))\n"
        "sortino = np.sqrt(252) * excess.mean() / downside_std\n"
        "```\n\n"
        "**Confidence: High**"
    ),
    "generic_atr": (
        "**ATR — Python snippet:**\n\n"
        "```python\n"
        "import numpy as np, pandas as pd\n\n"
        "tr = pd.concat([\n"
        "    high - low,\n"
        "    np.abs(high - close.shift(1)),\n"
        "    np.abs(low - close.shift(1)),\n"
        "], axis=1).max(axis=1)\n"
        "atr = tr.rolling(14).mean()\n"
        "```\n\n"
        "**Confidence: High**"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge templates
#
# Each template is a self-contained Markdown answer with a code snippet.
# Templates intentionally stay short (~100–200 tokens) so the UI renders
# them almost instantly.
# ═══════════════════════════════════════════════════════════════════════════

_TEMPLATES: Dict[str, str] = {
    # ── Volatility ───────────────────────────────────────────────────
    "volatility": (
        "**Daily volatility** can be estimated using the rolling standard "
        "deviation of daily returns:\n\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "# Daily log returns\n"
        "returns = np.log(close / close.shift(1))\n\n"
        "# Rolling 20-day volatility\n"
        "vol = returns.rolling(window=20).std()\n\n"
        "# Annualised volatility\n"
        "annual_vol = vol * np.sqrt(252)\n"
        "```\n\n"
        "**Key points:**\n"
        "- `window=20` ≈ 1 trading month; use 60 for quarterly.\n"
        "- `np.sqrt(252)` annualises assuming 252 trading days/year.\n"
        "- For **simple returns**, use `close.pct_change()` instead of log.\n\n"
        "**Confidence: High** — standard quantitative finance formula."
    ),

    # ── Sharpe ratio ─────────────────────────────────────────────────
    "sharpe": (
        "**Sharpe Ratio** measures risk-adjusted return:\n\n"
        "$$\\text{Sharpe} = \\frac{\\bar{R} - R_f}{\\sigma}$$\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "risk_free_rate = 0.05 / 252  # daily risk-free rate\n"
        "excess_returns = daily_returns - risk_free_rate\n\n"
        "sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()\n"
        "```\n\n"
        "**Key points:**\n"
        "- Multiply by `√252` to annualise from daily data.\n"
        "- Sharpe > 1.0 is generally considered good, > 2.0 is excellent.\n"
        "- Use **Sortino** if you only care about downside risk.\n\n"
        "**Confidence: High** — standard quantitative finance formula."
    ),

    # ── Daily returns ────────────────────────────────────────────────
    "daily_returns": (
        "**Daily returns** can be computed two ways:\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "# Simple (percentage) returns\n"
        "simple_returns = close.pct_change()\n\n"
        "# Log (continuously compounded) returns\n"
        "log_returns = np.log(close / close.shift(1))\n"
        "```\n\n"
        "**When to use which:**\n"
        "- **Simple returns**: portfolio aggregation (additive across assets).\n"
        "- **Log returns**: time-series analysis (additive across time).\n\n"
        "**Confidence: High** — standard quantitative finance formula."
    ),

    # ── Rolling calculations ─────────────────────────────────────────
    "rolling": (
        "**Rolling window** calculations in pandas:\n\n"
        "```python\n"
        "# Rolling mean (simple moving average)\n"
        "sma = close.rolling(window=20).mean()\n\n"
        "# Rolling standard deviation\n"
        "rolling_std = returns.rolling(window=20).std()\n\n"
        "# Rolling sum\n"
        "rolling_sum = volume.rolling(window=10).sum()\n\n"
        "# Exponential moving average (alternative)\n"
        "ema = close.ewm(span=20, adjust=False).mean()\n"
        "```\n\n"
        "**Key points:**\n"
        "- `min_periods=1` to avoid leading NaNs.\n"
        "- `.ewm()` gives more weight to recent observations.\n\n"
        "**Confidence: High** — standard pandas API."
    ),

    # ── ATR (Average True Range) ─────────────────────────────────────
    "atr": (
        "**ATR (Average True Range)** measures market volatility:\n\n"
        "```python\n"
        "import pandas as pd\n"
        "import numpy as np\n\n"
        "high_low = high - low\n"
        "high_close = np.abs(high - close.shift(1))\n"
        "low_close = np.abs(low - close.shift(1))\n\n"
        "true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)\n"
        "atr = true_range.rolling(window=14).mean()\n"
        "```\n\n"
        "**Key points:**\n"
        "- Default period is 14 (Wilder's original).\n"
        "- ATR is in **price units**, not percentage.\n"
        "- Use `atr / close` for normalised ATR (NATR).\n\n"
        "**Confidence: High** — standard technical analysis indicator."
    ),

    # ── Beta ─────────────────────────────────────────────────────────
    "beta": (
        "**Beta** measures an asset's sensitivity to market movements:\n\n"
        "$$\\beta = \\frac{\\text{Cov}(R_i, R_m)}{\\text{Var}(R_m)}$$\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "# Using covariance / variance\n"
        "cov = np.cov(stock_returns, market_returns)[0, 1]\n"
        "var = np.var(market_returns)\n"
        "beta = cov / var\n\n"
        "# Or with linear regression\n"
        "from numpy.polynomial.polynomial import polyfit\n"
        "beta, alpha = np.polyfit(market_returns, stock_returns, 1)\n"
        "```\n\n"
        "**Key points:**\n"
        "- β > 1 → more volatile than the market.\n"
        "- β < 1 → less volatile (defensive).\n"
        "- Rolling beta: apply the above in a `.rolling()` window.\n\n"
        "**Confidence: High** — standard CAPM metric."
    ),

    # ── Max Drawdown ─────────────────────────────────────────────────
    "max_drawdown": (
        "**Maximum Drawdown** measures the largest peak-to-trough decline:\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "cumulative = (1 + returns).cumprod()\n"
        "peak = cumulative.cummax()\n"
        "drawdown = (cumulative - peak) / peak\n"
        "max_drawdown = drawdown.min()  # most negative value\n"
        "```\n\n"
        "**Key points:**\n"
        "- Result is negative (e.g., -0.25 = 25% drawdown).\n"
        "- Calmar ratio = CAGR / |max drawdown|.\n"
        "- Use `drawdown.idxmin()` to find the date of max drawdown.\n\n"
        "**Confidence: High** — standard risk metric."
    ),

    # ── CAGR ─────────────────────────────────────────────────────────
    "cagr": (
        "**CAGR (Compound Annual Growth Rate):**\n\n"
        "$$\\text{CAGR} = \\left(\\frac{V_{\\text{final}}}{V_{\\text{initial}}}\\right)"
        "^{\\frac{1}{n}} - 1$$\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "total_return = portfolio[-1] / portfolio[0]\n"
        "n_years = len(portfolio) / 252  # trading days\n"
        "cagr = total_return ** (1 / n_years) - 1\n"
        "```\n\n"
        "**Key points:**\n"
        "- Assumes 252 trading days per year.\n"
        "- CAGR smooths out volatility — doesn't capture risk.\n"
        "- Pair with Sharpe or Sortino for risk-adjusted view.\n\n"
        "**Confidence: High** — standard return metric."
    ),

    # ── Sortino ratio ────────────────────────────────────────────────
    "sortino": (
        "**Sortino Ratio** — like Sharpe but only penalises downside risk:\n\n"
        "$$\\text{Sortino} = \\frac{\\bar{R} - R_f}{\\sigma_{\\text{downside}}}$$\n\n"
        "```python\n"
        "import numpy as np\n\n"
        "risk_free_rate = 0.05 / 252\n"
        "excess = daily_returns - risk_free_rate\n"
        "downside = excess[excess < 0]\n"
        "downside_std = np.sqrt(np.mean(downside ** 2))\n"
        "sortino = np.sqrt(252) * excess.mean() / downside_std\n"
        "```\n\n"
        "**Key points:**\n"
        "- More appropriate than Sharpe when return distribution is skewed.\n"
        "- Sortino > Sharpe when upside variance is high (desirable).\n\n"
        "**Confidence: High** — standard risk-adjusted metric."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def try_fastpath(query: str) -> Optional[str]:
    """Attempt to answer *query* via the fastpath bypass.

    Returns a Markdown answer string if the query qualifies, or
    ``None`` if it should fall through to the normal RAG pipeline.

    Qualification rules (all must hold):
        1. Query contains a recognised quant trigger keyword.
        2. Query is short (< ``MAX_QUERY_WORDS`` words).
        3. Classified intent is NOT in ``_BLOCKED_INTENTS``
           (e.g. ``analysis`` requires document context).

    A secondary **generic code-snippet** path also fires when the
    query uses phrasing like *"is there a function"* /
    *"how to calculate"* / *"formula for"* AND contains no
    strategy-specific keywords.  This catches broad knowledge
    requests (e.g. daily volatility) that do not need RAG.
    """
    if not query or not query.strip():
        return None

    query_clean = query.strip()
    word_count = len(query_clean.split())

    # ── Rule 2: length gate ──────────────────────────────────────────
    if word_count >= MAX_QUERY_WORDS:
        logger.debug(
            "Fastpath skip: query too long (%d words >= %d).",
            word_count, MAX_QUERY_WORDS,
        )
        return None

    # ── Rule 1: trigger keyword match ────────────────────────────────
    template_key = _match_trigger(query_clean)
    if template_key is not None:
        # ── Rule 3: intent gate ──────────────────────────────────────
        classification = classify_query(query_clean)
        intent = classification.get("intent", "")
        if intent in _BLOCKED_INTENTS:
            logger.debug(
                "Fastpath skip: intent=%s is blocked (query=%r).",
                intent, query_clean[:80],
            )
            return None

        answer = _TEMPLATES.get(template_key)
        if answer is not None:
            logger.info(
                "Fastpath HIT: trigger=%s, intent=%s, words=%d, query=%r",
                template_key, intent, word_count, query_clean[:80],
            )
            return answer
        logger.warning(
            "Fastpath: matched trigger '%s' but no template found.",
            template_key,
        )

    # ── Generic code-snippet bypass ──────────────────────────────────
    # "is there a function for X" / "how to calculate X" / "formula for X"
    # WITHOUT strategy-specific keywords → return direct snippet.
    if _GENERIC_CODE_PATTERN.search(query_clean):
        if _STRATEGY_KEYWORDS.search(query_clean):
            logger.debug(
                "Fastpath skip: generic code pattern matched but "
                "strategy keyword detected (query=%r).",
                query_clean[:80],
            )
            return None

        # Detect the topic
        for g_key, g_pat in _GENERIC_TOPIC_RULES:
            if g_pat.search(query_clean):
                g_answer = _GENERIC_TEMPLATES.get(g_key)
                if g_answer is not None:
                    logger.info(
                        "Fastpath HIT (generic): topic=%s, words=%d, "
                        "query=%r",
                        g_key, word_count, query_clean[:80],
                    )
                    return g_answer

        logger.debug(
            "Fastpath skip: generic code pattern matched but no "
            "recognised topic (query=%r).",
            query_clean[:80],
        )

    return None


def _match_trigger(query: str) -> Optional[str]:
    """Return the first matching template key, or None."""
    for key, pattern in _TRIGGER_RULES:
        if pattern.search(query):
            return key
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Inline unit tests  (run: python -m rag_pipeline.core.fastpath)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    _pass = _fail = 0

    def _ok(cond: bool, label: str) -> None:
        global _pass, _fail
        if cond:
            _pass += 1
            print(f"  PASS  {label}")
        else:
            _fail += 1
            print(f"  FAIL  {label}")

    # ── Trigger matching ─────────────────────────────────────────────
    print("\n=== Trigger Matching ===")
    _ok(_match_trigger("how to calculate volatility") == "volatility",
        "volatility trigger")
    _ok(_match_trigger("what is standard deviation") == "volatility",
        "std dev → volatility trigger")
    _ok(_match_trigger("daily vol estimate") == "volatility",
        "daily vol → volatility trigger")
    _ok(_match_trigger("sharpe ratio formula") == "sharpe",
        "sharpe trigger")
    _ok(_match_trigger("compute daily returns") == "daily_returns",
        "daily_returns trigger")
    _ok(_match_trigger("log returns calculation") == "daily_returns",
        "log returns trigger")
    _ok(_match_trigger("rolling mean of prices") == "rolling",
        "rolling trigger")
    _ok(_match_trigger("rolling std window") == "rolling",
        "rolling std trigger")
    _ok(_match_trigger("ATR indicator") == "atr",
        "ATR trigger")
    _ok(_match_trigger("average true range") == "atr",
        "average true range trigger")
    _ok(_match_trigger("calculate beta") == "beta",
        "beta trigger")
    _ok(_match_trigger("market beta coefficient") == "beta",
        "market beta trigger")
    _ok(_match_trigger("max drawdown formula") == "max_drawdown",
        "max_drawdown trigger")
    _ok(_match_trigger("CAGR calculation") == "cagr",
        "CAGR trigger")
    _ok(_match_trigger("sortino ratio") == "sortino",
        "sortino trigger")
    _ok(_match_trigger("what is machine learning") is None,
        "no trigger for unrelated query")
    _ok(_match_trigger("explain the momentum strategy") is None,
        "no trigger for strategy query")

    # ── Length gate ──────────────────────────────────────────────────
    print("\n=== Length Gate ===")
    _ok(
        try_fastpath("calculate volatility") is not None,
        "short volatility query → fastpath HIT",
    )
    _ok(
        try_fastpath(
            "I need to understand how to calculate the historical "
            "daily volatility of a portfolio of stocks using rolling "
            "standard deviation over multiple time windows and "
            "compare them across different regimes"
        ) is None,
        "long query (>25 words) → fastpath MISS",
    )
    _ok(try_fastpath("") is None, "empty query → None")
    _ok(try_fastpath("   ") is None, "blank query → None")

    # ── Intent gate (analysis blocked) ───────────────────────────────
    print("\n=== Intent Gate ===")
    # "how much drawdown" → intent=analysis (the 'how much' pattern)
    _ok(
        try_fastpath("how much max drawdown does this have") is None,
        "analysis intent → fastpath MISS",
    )
    # "what is sharpe" → intent=conceptual → allowed
    _ok(
        try_fastpath("what is sharpe ratio") is not None,
        "conceptual intent + sharpe → fastpath HIT",
    )

    # ── Full template content ────────────────────────────────────────
    print("\n=== Template Content ===")
    ans = try_fastpath("how to calculate daily volatility")
    _ok(ans is not None and "rolling" in ans, "volatility answer has 'rolling'")
    _ok(ans is not None and "np.sqrt(252)" in ans, "volatility answer has annualisation")

    ans = try_fastpath("sharpe ratio formula")
    _ok(ans is not None and "excess_returns" in ans, "sharpe answer has code")

    ans = try_fastpath("what is ATR")
    _ok(ans is not None and "true_range" in ans, "ATR answer has true_range")

    ans = try_fastpath("how to estimate beta")
    _ok(ans is not None and "np.cov" in ans, "beta answer has covariance")

    ans = try_fastpath("what is CAGR")
    _ok(ans is not None and "252" in ans, "CAGR answer references trading days")

    ans = try_fastpath("sortino vs sharpe")
    _ok(ans is not None and "downside" in ans, "sortino answer has downside")

    ans = try_fastpath("max drawdown")
    _ok(ans is not None and "cummax" in ans, "max_drawdown answer has cummax")

    ans = try_fastpath("daily returns")
    _ok(ans is not None and "pct_change" in ans, "daily_returns has pct_change")

    ans = try_fastpath("rolling std window size")
    _ok(ans is not None and ".rolling(" in ans, "rolling answer has .rolling()")

    # ── All templates exist ──────────────────────────────────────────
    print("\n=== Template Coverage ===")
    for key, _ in _TRIGGER_RULES:
        _ok(key in _TEMPLATES, f"template exists for trigger '{key}'")

    # ── Generic code-snippet bypass ──────────────────────────────────
    print("\n=== Generic Code-Snippet Bypass ===")

    # Positive: generic "how to calculate" queries
    ans = try_fastpath("is there a function for daily volatility")
    _ok(ans is not None and "pct_change" in ans,
        "'is there a function for volatility' → snippet")

    ans = try_fastpath("how to calculate daily returns")
    _ok(ans is not None and "pct_change" in ans,
        "'how to calculate daily returns' → snippet")

    ans = try_fastpath("formula for sharpe ratio")
    _ok(ans is not None and "excess" in ans,
        "'formula for sharpe ratio' → snippet")

    ans = try_fastpath("how to compute beta")
    _ok(ans is not None and "cov" in ans,
        "'how to compute beta' → snippet")

    ans = try_fastpath("is there a function for max drawdown")
    _ok(ans is not None and "cummax" in ans,
        "'is there a function for drawdown' → snippet")

    ans = try_fastpath("formula for moving average")
    _ok(ans is not None and "rolling" in ans,
        "'formula for moving average' → snippet")

    ans = try_fastpath("code for correlation between two stocks")
    _ok(ans is not None and "corr" in ans,
        "'code for correlation' → snippet")

    ans = try_fastpath("how to calculate CAGR")
    _ok(ans is not None and "252" in ans,
        "'how to calculate CAGR' → snippet")

    ans = try_fastpath("function to calculate sortino ratio")
    _ok(ans is not None and "downside" in ans,
        "'function to calculate sortino' → snippet")

    ans = try_fastpath("snippet for ATR indicator")
    _ok(ans is not None and "rolling" in ans,
        "'snippet for ATR' → snippet")

    # Negative: strategy keyword blocks generic path
    ans = try_fastpath("how to calculate momentum crossover strategy")
    _ok(ans is None,
        "strategy keyword 'crossover strategy' → fastpath MISS")

    ans = try_fastpath("is there a function for backtest signals")
    _ok(ans is None,
        "strategy keyword 'backtest signals' → fastpath MISS")

    ans = try_fastpath("formula for pairs trading arbitrage")
    _ok(ans is None,
        "strategy keyword 'pairs trading' → fastpath MISS")

    # Negative: generic phrase but no recognised topic
    ans = try_fastpath("is there a function for something random")
    _ok(ans is None,
        "generic phrase + unknown topic → fastpath MISS")

    # ── Summary ──────────────────────────────────────────────────────
    total = _pass + _fail
    print(f"\n{'='*60}")
    print(f"Fastpath tests: {_pass}/{total} passed, {_fail} failed")
    print(f"{'='*60}")
    sys.exit(0 if _fail == 0 else 1)
