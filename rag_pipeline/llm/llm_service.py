"""
LLM Service for Centurion Capital LLC RAG Pipeline.

Provides LLM-powered answer generation from RAG-retrieved context.
Supports multiple backends:
    - **Ollama**   — Local inference (Llama 3, Mistral, etc.)
    - **Claude**   — Anthropic Messages API
    - **OpenAI**   — OpenAI Chat Completions API (GPT-4o, GPT-4, etc.)

Each backend supports:
    - ``generate(query, context) -> str``   — blocking, full response
    - ``generate_stream(query, context) -> Generator[str, None, None]``
      — yields tokens as they arrive (reduces perceived latency)

Switching providers:
    Set ``CENTURION_RAG_LLM_PROVIDER`` in ``.env`` to one of:
        ``ollama`` | ``claude`` | ``openai``

Usage:
    from rag_pipeline.llm.llm_service import create_llm_backend
    from rag_pipeline.config import RAGConfig

    llm = create_llm_backend(RAGConfig())
    answer = llm.generate(query="What is RSI?", context="RSI is ...")

    # Streaming:
    for token in llm.generate_stream(query="What is RSI?", context="RSI is ..."):
        print(token, end="", flush=True)
"""

import json
import logging
import time
from typing import Generator, Optional

import requests

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for RAG grounding
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT_FULL = """\
You are a precise document question-answering assistant.

Your task is to answer the user's question using ONLY the context chunks provided below. \
Follow these rules STRICTLY — violations are unacceptable:

GROUNDING RULES (most important):
1. Base your answer EXCLUSIVELY on the provided context chunks. Do NOT use outside knowledge, \
prior training data, or assumptions.
2. NEVER fabricate, infer, or hallucinate ANY information not explicitly stated in the context — \
this includes names, numbers, dates, statistics, company descriptions, product features, or any \
other facts.
3. If the context does not contain enough information to answer the question, you MUST respond \
with EXACTLY: "I could not find sufficient information in the uploaded documents to answer \
this question." — then list what specific information is missing.
4. If you can only partially answer, clearly state which parts are supported by the context \
and which parts you cannot answer. NEVER fill gaps with invented information.
5. Do NOT describe or characterize the document, its authors, or its publisher beyond what \
is explicitly written in the context chunks.

CITATION RULES:
6. ALWAYS cite the source document and page number for EVERY claim, using the format: \
(Source: filename, Page N). Pull these from the chunk headers in the context.
7. When synthesising across multiple chunks, cite each source inline next to the relevant fact.

RESPONSE QUALITY:
8. Synthesise information from multiple context chunks into a single coherent answer. \
Do NOT simply repeat or list chunks verbatim.
9. Use clear, professional language. Use markdown formatting (headers, bullets, bold) \
for readability.
10. If the context contains contradictory information, explicitly note the discrepancy \
and cite both sources so the user can judge.
11. Be concise but thorough. Prefer structured answers (bullet points, numbered lists) \
over long prose.

CODE / FORMULA REPRODUCTION RULES:
12. When the context contains code snippets, source code, formulas, or pseudocode, you MUST \
reproduce them EXACTLY and VERBATIM as they appear in the context. Do NOT rewrite, refactor, \
simplify, translate to another library, or "improve" the code in any way.
13. NEVER generate new code from your training data. The ONLY code you may include in your \
answer is code that appears word-for-word in the provided context chunks. If the user asks \
for code that is not present in the context, respond with: "The requested code was not found \
in the uploaded documents."
14. Preserve the exact indentation, variable names, function names, class names, and comments \
from the source code in the context. Do NOT rename variables or restructure the code.
15. When presenting code from the context, wrap it in a fenced code block with the appropriate \
language tag (e.g., ```python) and indicate its source: (Source: filename, Page N).
16. You MUST reuse the exact function implementation from the provided context if available. \
Do NOT rewrite it unless explicitly asked. If a full implementation exists in context, return \
it verbatim.

CONFIDENCE SIGNAL:
17. End every answer with a confidence indicator on a new line:
    - **High confidence** — answer is fully supported by multiple context chunks
    - **Medium confidence** — answer is supported but from limited context
    - **Low confidence** — answer is partially supported; some gaps exist

OUTPUT LENGTH (mandatory):
18. Provide concise, implementation-focused answers. \
Limit your response to 600 tokens maximum. \
Avoid unnecessary explanations, preambles, or filler text. \
Get straight to the point.\
"""

# ---------------------------------------------------------------------------
# Compact system prompt for local / CPU-bound models (Ollama).
#
# Reduces the system prompt from ~850 tokens to ~150 tokens.  This cuts
# prompt-evaluation time by 5–10× on CPU — the difference between a
# 30 s response and a 200+ s timeout.  All core grounding rules are
# preserved in condensed form.
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT_COMPACT = """\
You are a document QA assistant. Answer ONLY from the provided context chunks.

Rules:
- NEVER use outside knowledge or hallucinate facts.
- If context is insufficient, say: "I could not find sufficient information in the uploaded documents."
- Cite every claim: (Source: filename, Page N).
- Reproduce code/formulas EXACTLY as they appear in the context — never rewrite or invent code.
- If a full function implementation exists in context, return it VERBATIM. Do NOT rewrite.
- Use markdown formatting (bullets, headers, bold). Be concise.
- End with confidence: **High** / **Medium** / **Low**.
- Limit response to 600 tokens.\
"""

# Default: use compact prompt for speed.  Cloud backends (Claude, OpenAI)
# can use the full prompt via CENTURION_RAG_COMPACT_PROMPT=false.
RAG_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT_COMPACT

NO_CONTEXT_SYSTEM_PROMPT = """\
IMPORTANT: No relevant documents were found in the knowledge base for this query.

You MUST respond with EXACTLY this message and nothing else:

"I could not find relevant documents in the knowledge base to answer this question. \
Please upload the relevant PDFs first, or rephrase your question."

Do NOT answer from general knowledge. Do NOT guess. Do NOT add any other text. \
Do NOT attempt to be helpful by providing information from your training data.\
"""


# ---------------------------------------------------------------------------
# Helper: build user message from query + context
# ---------------------------------------------------------------------------

def _build_user_message(query: str, context: str) -> str:
    """Return the user-facing message combining context and query."""
    if context.strip():
        return (
            f"## Context\n\n{context}\n\n"
            f"---\n\n"
            f"## Question\n\n{query}"
        )
    return query


def _pick_system_prompt(context: str, *, compact: bool = True) -> str:
    """Return the appropriate system prompt.

    Parameters
    ----------
    context : str
        The retrieved context text.  If empty, the no-context prompt
        is returned regardless of *compact*.
    compact : bool, default True
        When *True* (the default for Ollama / local inference), use
        the 150-token compact prompt.  Set *False* to use the full
        850-token prompt (recommended for cloud LLMs like Claude or
        OpenAI where prompt-eval is near-instant).
    """
    if not context.strip():
        return NO_CONTEXT_SYSTEM_PROMPT
    return RAG_SYSTEM_PROMPT_COMPACT if compact else RAG_SYSTEM_PROMPT_FULL


# ---------------------------------------------------------------------------
# Ollama backend (local, free, runs Llama 3 / Mistral / etc.)
# ---------------------------------------------------------------------------

# Default flush interval for buffered streaming (seconds).
_STREAM_FLUSH_INTERVAL = 1.5


class OllamaLLMBackend:
    """
    LLM backend using Ollama's local REST API.

    Ollama serves models at http://localhost:11434 by default.
    Uses a persistent ``requests.Session`` to avoid TCP/TLS handshake
    overhead on every call.

    Streaming behaviour
    ~~~~~~~~~~~~~~~~~~~
    Both ``generate()`` and ``generate_stream()`` use Ollama's
    streaming endpoint (``"stream": True``).  Instead of a single
    global *timeout*, each **chunk** (token batch) has its own
    read-timeout (*chunk_timeout*, default 30 s).  If no data
    arrives within that window the request is cancelled — this
    prevents the old 600 s hard-hang.

    ``generate_stream()`` additionally **buffers** tokens and yields
    them every ~1–2 s so the Streamlit UI refreshes smoothly without
    per-token overhead.
    """

    # Hard ceiling — num_predict can never exceed this regardless of
    # what the config or env var says.  Prevents runaway generation.
    _MAX_NUM_PREDICT_CEILING = 800

    # Simple-query threshold: queries shorter than this use a reduced
    # num_predict to save latency.
    _SIMPLE_QUERY_WORD_LIMIT = 30
    _SIMPLE_QUERY_NUM_PREDICT = 300

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_ctx: int = 4096,
        num_predict: int = 400,
        repeat_penalty: float = 1.1,
        max_tokens: int = 400,
        timeout: int = 600,
        chunk_timeout: int = 30,
        first_token_timeout: int = 180,
        warmup: bool = True,
        compact_prompt: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        # Enforce ceiling on num_predict to prevent unbounded generation
        self.num_predict = min(num_predict, self._MAX_NUM_PREDICT_CEILING)
        self.repeat_penalty = repeat_penalty
        self.max_tokens = min(max_tokens, self._MAX_NUM_PREDICT_CEILING)
        self.timeout = timeout
        self.chunk_timeout = chunk_timeout
        self.first_token_timeout = first_token_timeout
        self.compact_prompt = compact_prompt
        # Persistent HTTP session — keeps TCP connection alive
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        logger.info(
            "OllamaLLMBackend initialised: model=%s, temp=%.2f, "
            "top_p=%.2f, num_ctx=%d, num_predict=%d, repeat_penalty=%.2f, "
            "first_token_timeout=%ds, chunk_timeout=%ds",
            self.model, self.temperature, self.top_p,
            self.num_ctx, self.num_predict, self.repeat_penalty,
            self.first_token_timeout, self.chunk_timeout,
        )
        # Optionally warm up the model so subsequent queries skip the
        # multi-second model-load phase.
        if warmup:
            self._warmup()

    # ------------------------------------------------------------------ #
    # Model warm-up (pre-load into memory)
    # ------------------------------------------------------------------ #

    def _warmup(self) -> None:
        """Send a minimal request to pre-load the model into memory.

        Ollama keeps models in RAM/VRAM after the first request, so
        a cheap warm-up eliminates the 5–10 s load delay on the first
        real query.  Uses ``num_predict=1`` and a tiny context window
        to minimise overhead.
        """
        try:
            t0 = time.monotonic()
            resp = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"num_predict": 1, "num_ctx": 128},
                },
                timeout=(10, self.first_token_timeout),
            )
            resp.raise_for_status()
            elapsed = time.monotonic() - t0
            logger.info(
                " Ollama warm-up complete: model=%s loaded in %.1fs",
                self.model, elapsed,
            )
        except requests.ConnectionError:
            logger.warning(
                " Ollama warm-up skipped: cannot connect to %s",
                self.base_url,
            )
        except Exception as e:
            logger.warning(" Ollama warm-up failed: %s", e)

    def _build_prompt(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Build a single prompt string for the Ollama generate API."""
        system_msg = system_prompt if system_prompt is not None else _pick_system_prompt(
            context, compact=self.compact_prompt
        )
        user_msg = _build_user_message(query, context)
        return f"{system_msg}\n\n{user_msg}"

    def _build_messages(self, query: str, context: str, system_prompt: Optional[str] = None) -> list:
        """Build structured messages for the Ollama chat API.

        When *system_prompt* is given it replaces the default RAG/no-context
        system prompt.  This is important for internal pipeline steps
        (query rewriting, HyDE, classification) that need their own
        task-specific instructions.
        """
        sys_content = system_prompt if system_prompt is not None else _pick_system_prompt(
            context, compact=self.compact_prompt
        )
        user_content = query if system_prompt is not None else _build_user_message(query, context)
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

    def generate(self, query: str, context: str, *, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Ollama's chat API.

        Internally uses **streaming** with a per-chunk read timeout
        (``chunk_timeout``) to avoid the old global hard-timeout.
        Tokens are collected and returned as a single string.

        Falls back to a helpful error message if Ollama is unreachable.
        """
        try:
            tokens = list(
                self.generate_stream(query, context, system_prompt=system_prompt)
            )
            answer = "".join(tokens)
            if not answer or answer.startswith("\n"):
                # generate_stream yields error strings starting with 
                return answer or "The LLM returned an empty response. Please try again."
            return answer
        except Exception as e:
            logger.error("Unexpected LLM error: %s", e, exc_info=True)
            return f"Unexpected error: {e}"

    def generate_stream(
        self, query: str, context: str, *, system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream tokens from Ollama's chat API with **two-tier timeout**.

        Tokens are buffered internally and flushed to the caller every
        ~1–2 seconds (``_STREAM_FLUSH_INTERVAL``) so the Streamlit UI
        refreshes smoothly without per-token rerender overhead.

        Two-tier timeout behaviour
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        - **Connection timeout**: 10 s (fast fail if Ollama is down).
        - **Socket read timeout**: ``self.first_token_timeout`` (default
          120 s).  This is the socket-level deadline applied by
          ``requests`` to each ``recv()`` call.  It must be large enough
          to cover model loading + prompt evaluation — Ollama sends
          **zero bytes** during those phases.
        - **Inter-chunk gap timeout**: ``self.chunk_timeout`` (default
          30 s).  After the first token arrives, if no new token
          appears within this window the stream is aborted.  This
          detects mid-generation hangs without penalising slow
          initial start-up.
        """
        messages = self._build_messages(query, context, system_prompt=system_prompt)

        # ── Query-aware num_predict ──────────────────────────────────
        # Short queries (< 30 words) get a smaller generation budget
        # to shave latency.  The hard ceiling is always enforced.
        effective_num_predict = self.num_predict
        query_words = len(query.split())
        if query_words < self._SIMPLE_QUERY_WORD_LIMIT:
            effective_num_predict = min(
                effective_num_predict, self._SIMPLE_QUERY_NUM_PREDICT,
            )
            logger.info(
                "Ollama: simple query (%d words < %d) "
                "num_predict reduced to %d",
                query_words, self._SIMPLE_QUERY_WORD_LIMIT,
                effective_num_predict,
            )
        effective_num_predict = min(
            effective_num_predict, self._MAX_NUM_PREDICT_CEILING,
        )

        # ── Prompt-size safety guard ────────────────────────────────
        # Estimate total input tokens.  If the prompt is too large for
        # the configured num_ctx, truncate the user message so that
        # prompt_eval stays within the first_token_timeout.
        # Rule of thumb: keep input < num_ctx - num_predict - 64 (headroom)
        _max_input = max(self.num_ctx - effective_num_predict - 64, 256)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        _approx_tokens = total_chars // 4   # ~4 chars per token
        if _approx_tokens > _max_input:
            # Trim the longest message (usually the user message)
            trim_target = int(_max_input * 4)  # back to chars
            for m in reversed(messages):
                if len(m.get("content", "")) > 200:
                    m["content"] = m["content"][:trim_target] + (
                        "\n\n[Context truncated to fit model window]"
                    )
                    break
            logger.warning(
                "Prompt truncated: ~%d tokens exceeded num_ctx=%d "
                "(max_input=%d). Trimmed to ~%d chars.",
                _approx_tokens, self.num_ctx, _max_input, trim_target,
            )

        try:
            # Socket timeout = (connect, read).
            # read = first_token_timeout so the socket survives the
            # model-load + prompt-eval phase (can be 30–90 s on CPU).
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": effective_num_predict,
                        "num_ctx": self.num_ctx,
                        "repeat_penalty": self.repeat_penalty,
                    },
                },
                timeout=(10, self.first_token_timeout),
                stream=True,
            )
            response.raise_for_status()

            buffer: list[str] = []
            last_flush = time.monotonic()
            stream_start = time.monotonic()
            last_token_time = time.monotonic()
            token_count = 0
            first_token_received = False

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = data.get("message", {}).get("content", "")
                if token:
                    now = time.monotonic()

                    # ── TTFT (time to first token) logging ───────────
                    if not first_token_received:
                        first_token_received = True
                        ttft = now - stream_start
                        logger.info(
                            "Ollama TTFT: %.2fs (model=%s)",
                            ttft, self.model,
                        )

                    # ── Inter-chunk gap check ────────────────────────
                    # After the first token, abort if the gap between
                    # consecutive tokens exceeds chunk_timeout.
                    if first_token_received and token_count > 0:
                        gap = now - last_token_time
                        if gap > self.chunk_timeout:
                            logger.warning(
                                "Inter-chunk gap %.1fs > %ds — aborting "
                                "stream (model=%s, tokens so far=%d).",
                                gap, self.chunk_timeout,
                                self.model, token_count,
                            )
                            if buffer:
                                yield "".join(buffer)
                                buffer.clear()
                            yield (
                                f"\n\nModel stopped responding for "
                                f"{int(gap)}s mid-generation — stream "
                                f"aborted after {token_count} tokens."
                            )
                            response.close()
                            return

                    last_token_time = now
                    buffer.append(token)
                    token_count += 1

                # Time-based flush: yield accumulated tokens every
                # ~_STREAM_FLUSH_INTERVAL seconds (1–2 s) or when the
                # model signals completion.
                now = time.monotonic()
                is_done = data.get("done", False)
                if buffer and (is_done or now - last_flush >= _STREAM_FLUSH_INTERVAL):
                    yield "".join(buffer)
                    buffer.clear()
                    last_flush = now

                if is_done:
                    # ── Performance logging ──────────────────────────
                    total_s = now - stream_start
                    # Ollama may report eval_count in the final message
                    eval_count = data.get("eval_count", token_count)
                    tps = eval_count / total_s if total_s > 0 else 0.0
                    load_s = data.get("load_duration", 0) / 1e9
                    prompt_eval_s = data.get("prompt_eval_duration", 0) / 1e9
                    # eval_duration is generation time only (nanoseconds)
                    eval_dur_s = data.get("eval_duration", 0) / 1e9
                    gen_tps = (
                        eval_count / eval_dur_s
                        if eval_dur_s > 0 else tps
                    )
                    logger.info(
                        "Ollama perf: model=%s, tokens=%d, "
                        "time=%.2fs, tokens/sec=%.1f (gen=%.1f), "
                        "num_predict=%d, load=%.1fs, prompt_eval=%.1fs",
                        self.model, eval_count, total_s, tps, gen_tps,
                        effective_num_predict, load_s, prompt_eval_s,
                    )
                    break

            # Flush any remaining tokens left in the buffer
            if buffer:
                yield "".join(buffer)

            # Performance log for streams that ended without done=True
            if not token_count:
                logger.warning(
                    "Ollama stream ended with 0 tokens (model=%s).",
                    self.model,
                )

        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            yield (
                "\n**Cannot connect to Ollama.**\n\n"
                "Please ensure Ollama is running:\n"
                "1. Install from https://ollama.com/download\n"
                "2. Run: `ollama pull llama3` (or `mistral`)\n"
                "3. Ollama starts automatically, or run: `ollama serve`"
            )
        except requests.ReadTimeout:
            logger.error(
                "Ollama: no data received in %ds (first_token_timeout) "
                "— cancelling request (model=%s).",
                self.first_token_timeout, self.model,
            )
            yield (
                f"\n**No response from Ollama for "
                f"{self.first_token_timeout}s** "
                "— request cancelled.\n\n"
                "The model may be loading or the context is too large.\n"
                "Try:\n"
                "- Increase `CENTURION_RAG_LLM_FIRST_TOKEN_TIMEOUT` "
                "in `.env`\n"
                f"- Use a smaller model (current: `{self.model}`)\n"
                "- Reduce `CENTURION_RAG_LLM_NUM_CTX`"
            )
        except requests.Timeout:
            logger.error("Ollama connection timed out.")
            yield (
                "\n**Ollama connection timed out.**\n\n"
                "Ensure Ollama is running and reachable."
            )
        except requests.HTTPError as e:
            logger.error("Ollama HTTP error: %s", e)
            if "model" in str(e).lower() or (
                hasattr(response, "status_code") and response.status_code == 404
            ):
                yield (
                    f"\n**Model '{self.model}' not found.**\n\n"
                    f"Pull it first: `ollama pull {self.model}`"
                )
            else:
                yield f"\nLLM error: {e}"
        except Exception as e:
            logger.error("Ollama streaming error: %s", e, exc_info=True)
            yield f"\nStreaming error: {e}"

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Ollama model names may include :latest suffix
            return any(
                self.model in m or m.startswith(self.model)
                for m in models
            )
        except Exception:
            return False

    def list_models(self) -> list:
        """List all models available in the local Ollama instance."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Anthropic Claude backend (cloud API)
# ---------------------------------------------------------------------------

class ClaudeLLMBackend:
    """
    LLM backend using the Anthropic Messages API.

    Requires the ``anthropic`` Python package::

        pip install anthropic

    Set ``ANTHROPIC_API_KEY`` in your ``.env`` file.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for Claude. "
                    "Install it with: pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("Anthropic client initialised (model=%s)", self.model)
        return self._client

    def generate(self, query: str, context: str) -> str:
        """
        Generate a response using the Anthropic Messages API.

        Retries up to 3 times on transient connection errors (e.g.
        ``APIConnectionError``, TLS resets) with exponential backoff.
        """
        system_prompt = _pick_system_prompt(context, compact=False)
        user_message = _build_user_message(query, context)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message},
                    ],
                )

                # Extract text from response content blocks
                answer_parts = []
                for block in message.content:
                    if hasattr(block, "text"):
                        answer_parts.append(block.text)
                answer = "\n".join(answer_parts)

                if not answer:
                    logger.warning("Claude returned empty response")
                    return "The LLM returned an empty response. Please try again."

                logger.info(
                    "Claude response generated (model=%s, input_tokens=%s, output_tokens=%s)",
                    self.model,
                    getattr(message.usage, "input_tokens", "?"),
                    getattr(message.usage, "output_tokens", "?"),
                )
                return answer

            except Exception as e:
                error_type = type(e).__name__
                err_str = str(e).lower()

                # Non-retryable errors — fail immediately
                if "authentication" in err_str or "api_key" in err_str:
                    return (
                        "\u26a0\ufe0f **Claude authentication failed.**\n\n"
                        "Please check your `ANTHROPIC_API_KEY` in the `.env` file."
                    )
                if "rate" in err_str:
                    return "\u26a0\ufe0f **Rate limited.** Please wait a moment and try again."
                if "credit" in err_str or "balance" in err_str or "billing" in err_str:
                    return (
                        "\u26a0\ufe0f **Claude credit balance too low.**\n\n"
                        "Please top up at Plans & Billing on the Anthropic dashboard."
                    )

                # Retryable (connection errors, timeouts, 5xx)
                is_connection = "connection" in error_type.lower() or "connect" in err_str
                is_timeout = "timeout" in err_str
                is_server = "500" in err_str or "502" in err_str or "503" in err_str
                retryable = is_connection or is_timeout or is_server

                if retryable and attempt < max_retries:
                    wait = 2 ** attempt  # 2s, 4s
                    logger.warning(
                        "Claude transient error (attempt %d/%d, retrying in %ds): %s",
                        attempt, max_retries, wait, e,
                    )
                    time.sleep(wait)
                    # Reset client to force a fresh connection
                    self._client = None
                    continue

                logger.error("Claude API error (%s): %s", error_type, e, exc_info=True)
                return f"\u26a0\ufe0f Claude API error: {e}"

        # Should not reach here, but just in case
        return "\u26a0\ufe0f Claude API error after retries."

    def generate_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        """
        Stream tokens from the Anthropic Messages API.

        Uses ``client.messages.stream()`` context manager.
        """
        system_prompt = _pick_system_prompt(context, compact=False)
        user_message = _build_user_message(query, context)

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error("Claude streaming error: %s", e, exc_info=True)
            yield f"\n\u26a0\ufe0f Claude streaming error: {e}"

    def is_available(self) -> bool:
        """Check if the Claude API key is set and client can be created."""
        if not self.api_key:
            return False
        try:
            _ = self.client
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# OpenAI backend (cloud API — GPT-4o, GPT-4, GPT-3.5-turbo, etc.)
# ---------------------------------------------------------------------------

class OpenAILLMBackend:
    """
    LLM backend using the OpenAI Chat Completions API.

    Requires the ``openai`` Python package::

        pip install openai

    Set ``OPENAI_API_KEY`` in your ``.env`` file.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 600,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for OpenAI. "
                    "Install it with: pip install openai"
                )
            self._client = openai.OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            logger.info("OpenAI client initialised (model=%s, timeout=%ds)", self.model, self.timeout)
        return self._client

    def generate(self, query: str, context: str) -> str:
        """
        Generate a response using the OpenAI Chat Completions API.

        Retries up to 3 times on transient connection errors with
        exponential backoff (mirrors Claude retry logic).
        """
        system_prompt = _pick_system_prompt(context, compact=False)
        user_message = _build_user_message(query, context)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )

                answer = response.choices[0].message.content or ""

                if not answer:
                    logger.warning("OpenAI returned empty response")
                    return "The LLM returned an empty response. Please try again."

                usage = response.usage
                logger.info(
                    "OpenAI response generated (model=%s, prompt_tokens=%s, "
                    "completion_tokens=%s)",
                    self.model,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )
                return answer

            except Exception as e:
                error_type = type(e).__name__
                err_str = str(e).lower()

                # Non-retryable errors — fail immediately
                if "authentication" in err_str or "api_key" in err_str or "invalid" in err_str:
                    return (
                        "\u26a0\ufe0f **OpenAI authentication failed.**\n\n"
                        "Please check your `OPENAI_API_KEY` in the `.env` file."
                    )
                if "rate" in err_str:
                    return "\u26a0\ufe0f **Rate limited.** Please wait a moment and try again."
                if "model" in err_str and "not found" in err_str:
                    return (
                        f"\u26a0\ufe0f **Model '{self.model}' not available.**\n\n"
                        "Check your OpenAI plan or try a different model."
                    )
                if "billing" in err_str or "quota" in err_str or "insufficient" in err_str:
                    return (
                        "\u26a0\ufe0f **OpenAI quota/billing error.**\n\n"
                        "Please check your OpenAI billing dashboard."
                    )

                # Retryable (connection errors, timeouts, 5xx)
                is_connection = "connection" in error_type.lower() or "connect" in err_str
                is_timeout = "timeout" in err_str
                is_server = "500" in err_str or "502" in err_str or "503" in err_str
                retryable = is_connection or is_timeout or is_server

                if retryable and attempt < max_retries:
                    wait = 2 ** attempt  # 2s, 4s
                    logger.warning(
                        "OpenAI transient error (attempt %d/%d, retrying in %ds): %s",
                        attempt, max_retries, wait, e,
                    )
                    time.sleep(wait)
                    self._client = None  # reset to force fresh connection
                    continue

                logger.error("OpenAI API error (%s): %s", error_type, e, exc_info=True)
                return f"\u26a0\ufe0f OpenAI API error: {e}"

        return "\u26a0\ufe0f OpenAI API error after retries."

    def generate_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        """
        Stream tokens from the OpenAI Chat Completions API.

        Uses ``stream=True`` on ``chat.completions.create()``.
        """
        system_prompt = _pick_system_prompt(context, compact=False)
        user_message = _build_user_message(query, context)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            logger.error("OpenAI streaming error: %s", e, exc_info=True)
            yield f"\n\u26a0\ufe0f OpenAI streaming error: {e}"

    def is_available(self) -> bool:
        """Check if the OpenAI API key is set and client can be created."""
        if not self.api_key:
            return False
        try:
            _ = self.client
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory: create the right backend from RAGConfig
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "ollama": "OllamaLLMBackend",
    "claude": "ClaudeLLMBackend",
    "openai": "OpenAILLMBackend",
}


def _create_ollama_backend(config: RAGConfig) -> OllamaLLMBackend:
    """Helper to build an Ollama backend from config."""
    return OllamaLLMBackend(
        model=config.llm_model,
        base_url=config.llm_base_url,
        temperature=config.llm_temperature,
        top_p=config.llm_top_p,
        num_ctx=config.llm_num_ctx,
        num_predict=config.llm_num_predict,
        repeat_penalty=config.llm_repeat_penalty,
        max_tokens=config.llm_max_tokens,
        timeout=config.llm_timeout,
        chunk_timeout=config.llm_chunk_timeout,
        first_token_timeout=config.llm_first_token_timeout,
        compact_prompt=config.llm_compact_prompt,
    )


def create_llm_backend(config: Optional[RAGConfig] = None):
    """
    Instantiate the appropriate LLM backend based on ``config.llm_provider``.

    Supported providers:
      - ``ollama``  — local Ollama inference (default)
      - ``claude``  — Anthropic Claude API (with Ollama fallback)
      - ``openai``  — OpenAI API (with Ollama fallback)

    Returns an object with ``.generate(query, context) -> str`` and
    ``.generate_stream(query, context)`` methods.
    """
    config = config or RAGConfig()
    provider = config.llm_provider  # already lowered in config

    # --- Claude ---
    if provider == "claude":
        if not config.claude_api_key:
            logger.warning(
                "Claude selected but ANTHROPIC_API_KEY is not set — "
                "falling back to default backend."
            )
            return _FallbackLLMBackend("claude", "ANTHROPIC_API_KEY")
        primary = ClaudeLLMBackend(
            api_key=config.claude_api_key,
            model=config.claude_model,
            temperature=config.claude_temperature,
            max_tokens=config.claude_max_tokens,
        )
        fallback = _create_ollama_backend(config)
        return _FallbackChainBackend(primary, fallback)

    # --- OpenAI ---
    if provider == "openai":
        if not config.openai_api_key:
            logger.warning(
                "OpenAI selected but OPENAI_API_KEY is not set — "
                "falling back to default backend."
            )
            return _FallbackLLMBackend("openai", "OPENAI_API_KEY")
        primary = OpenAILLMBackend(
            api_key=config.openai_api_key,
            model=config.openai_model,
            temperature=config.openai_temperature,
            max_tokens=config.openai_max_tokens,
            timeout=config.llm_timeout,
        )
        fallback = _create_ollama_backend(config)
        return _FallbackChainBackend(primary, fallback)

    # --- Ollama (default) ---
    if provider != "ollama":
        logger.warning(
            "Unknown LLM provider '%s'. Supported: %s. Falling back to Ollama.",
            provider, ", ".join(_PROVIDERS),
        )
    return _create_ollama_backend(config)


# ---------------------------------------------------------------------------
# Fallback chain: primary Ollama
# ---------------------------------------------------------------------------

class _FallbackChainBackend:
    """
    Wraps a primary LLM backend with an Ollama fallback.

    If ``primary.generate()`` returns an error message (starts with )
    or raises an exception, the request is transparently forwarded to
    the fallback (Ollama) backend.
    """

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_name = type(primary).__name__
        self._fallback_name = type(fallback).__name__
        logger.info(
            "LLM fallback chain: %s %s",
            self._primary_name, self._fallback_name,
        )

    def generate(self, query: str, context: str) -> str:
        try:
            answer = self._primary.generate(query, context)
            # Detect error responses from the primary backend
            if answer.startswith("\u26a0\ufe0f"):
                logger.warning(
                    "%s returned error — falling back to %s",
                    self._primary_name, self._fallback_name,
                )
                return self._fallback.generate(query, context)
            return answer
        except Exception as e:
            logger.error(
                "%s failed (%s) — falling back to %s",
                self._primary_name, e, self._fallback_name,
            )
            return self._fallback.generate(query, context)

    def generate_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        try:
            # Try primary streaming first
            tokens = list(self._primary.generate_stream(query, context))
            full = "".join(tokens)
            if full.startswith("\u26a0\ufe0f"):
                logger.warning(
                    "%s stream returned error — falling back to %s",
                    self._primary_name, self._fallback_name,
                )
                yield from self._fallback.generate_stream(query, context)
            else:
                yield from tokens
        except Exception as e:
            logger.error(
                "%s stream failed (%s) — falling back to %s",
                self._primary_name, e, self._fallback_name,
            )
            yield from self._fallback.generate_stream(query, context)

    def is_available(self) -> bool:
        return self._primary.is_available() or self._fallback.is_available()


class _FallbackLLMBackend:
    """Helpful fallback that tells the user what to configure."""

    def __init__(self, provider: str, missing_key: str = "") -> None:
        self._provider = provider
        self._missing_key = missing_key

    def generate(self, query: str, context: str) -> str:
        if self._missing_key:
            return (
                f"\u26a0\ufe0f **{self._provider.title()} is selected but "
                f"`{self._missing_key}` is not set.**\n\n"
                f"Add it to the project root `.env` and restart the app."
            )
        supported = ", ".join(f"`{p}`" for p in _PROVIDERS)
        return (
            f"\u26a0\ufe0f **Unknown LLM provider `{self._provider}`.**\n\n"
            f"Set `CENTURION_RAG_LLM_PROVIDER` to one of: {supported}"
        )

    def generate_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        """Fallback streaming — yields the full error message at once."""
        yield self.generate(query, context)

    def is_available(self) -> bool:
        return False
