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
    from rag_pipeline.llm_service import create_llm_backend
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

RAG_SYSTEM_PROMPT = """\
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

CONFIDENCE SIGNAL:
16. End every answer with a confidence indicator on a new line:
    - **High confidence** — answer is fully supported by multiple context chunks
    - **Medium confidence** — answer is supported but from limited context
    - **Low confidence** — answer is partially supported; some gaps exist\
"""

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


def _pick_system_prompt(context: str) -> str:
    """Return RAG or no-context system prompt."""
    return RAG_SYSTEM_PROMPT if context.strip() else NO_CONTEXT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Ollama backend (local, free, runs Llama 3 / Mistral / etc.)
# ---------------------------------------------------------------------------

class OllamaLLMBackend:
    """
    LLM backend using Ollama's local REST API.

    Ollama serves models at http://localhost:11434 by default.
    Uses a persistent ``requests.Session`` to avoid TCP/TLS handshake
    overhead on every call.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 600,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Persistent HTTP session — keeps TCP connection alive
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _build_prompt(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Build a single prompt string for the Ollama generate API."""
        system_msg = system_prompt if system_prompt is not None else _pick_system_prompt(context)
        user_msg = _build_user_message(query, context)
        return f"{system_msg}\n\n{user_msg}"

    def _build_messages(self, query: str, context: str, system_prompt: Optional[str] = None) -> list:
        """Build structured messages for the Ollama chat API.

        When *system_prompt* is given it replaces the default RAG/no-context
        system prompt.  This is important for internal pipeline steps
        (query rewriting, HyDE, classification) that need their own
        task-specific instructions.
        """
        sys_content = system_prompt if system_prompt is not None else _pick_system_prompt(context)
        user_content = query if system_prompt is not None else _build_user_message(query, context)
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

    def generate(self, query: str, context: str, *, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Ollama's chat API (non-streaming).

        Uses structured messages so the model properly separates
        system instructions from the user query.
        Falls back to a helpful error message if Ollama is unreachable.

        When *system_prompt* is provided, it overrides the default
        RAG/no-context system prompt.  This is used by internal pipeline
        components (query rewriter, HyDE, classifier) that need their
        own task-specific instructions without the RAG grounding rules.
        """
        messages = self._build_messages(query, context, system_prompt=system_prompt)

        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": 8192,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            answer = data.get("message", {}).get("content", "")

            if not answer:
                logger.warning("Ollama returned empty response: %s", data)
                return "The LLM returned an empty response. Please try again."

            logger.info(
                "LLM response generated (model=%s, tokens=%s)",
                self.model,
                data.get("eval_count", "?"),
            )
            return answer

        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            return (
                "⚠️ **Cannot connect to Ollama.**\n\n"
                "Please ensure Ollama is running:\n"
                "1. Install from https://ollama.com/download\n"
                "2. Run: `ollama pull llama3` (or `mistral`)\n"
                "3. Ollama starts automatically, or run: `ollama serve`"
            )
        except requests.Timeout:
            logger.error("Ollama request timed out after %ds", self.timeout)
            return (
                "⚠️ **Ollama request timed out** after "
                f"{self.timeout}s.\n\nTry a shorter query or increase "
                "`CENTURION_RAG_LLM_TIMEOUT` in your `.env` file."
            )
        except requests.HTTPError as e:
            logger.error("Ollama HTTP error: %s", e)
            if "model" in str(e).lower() or response.status_code == 404:
                return (
                    f"⚠️ **Model '{self.model}' not found.**\n\n"
                    f"Pull it first: `ollama pull {self.model}`"
                )
            return f"⚠️ LLM error: {e}"
        except Exception as e:
            logger.error("Unexpected LLM error: %s", e, exc_info=True)
            return f"⚠️ Unexpected error: {e}"

    def generate_stream(
        self, query: str, context: str, *, system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream tokens from Ollama's chat API.

        Yields individual text chunks as they arrive.
        When *system_prompt* is provided, it overrides the default prompt.
        """
        messages = self._build_messages(query, context, system_prompt=system_prompt)

        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": 8192,
                    },
                },
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error("Ollama streaming error: %s", e, exc_info=True)
            yield f"\n⚠️ Streaming error: {e}"

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
        system_prompt = _pick_system_prompt(context)
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
        system_prompt = _pick_system_prompt(context)
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
        system_prompt = _pick_system_prompt(context)
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
        system_prompt = _pick_system_prompt(context)
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
        max_tokens=config.llm_max_tokens,
        timeout=config.llm_timeout,
    )


def create_llm_backend(config: Optional[RAGConfig] = None):
    """
    Instantiate the appropriate LLM backend based on ``config.llm_provider``.

    Currently hardcoded to Ollama-only. Claude and OpenAI paths are
    commented out.

    Returns an object with ``.generate(query, context) -> str`` and
    ``.generate_stream(query, context)`` methods.
    """
    config = config or RAGConfig()
    provider = config.llm_provider  # already lowered in config

    # --- Always use Ollama regardless of provider setting ---
    return _create_ollama_backend(config)

    # --- Claude (commented out) ---
    # if provider == "claude":
    #     if not config.claude_api_key:
    #         logger.warning(
    #             "Claude selected but ANTHROPIC_API_KEY is not set — "
    #             "falling back to default backend."
    #         )
    #         return _FallbackLLMBackend("claude", "ANTHROPIC_API_KEY")
    #     primary = ClaudeLLMBackend(
    #         api_key=config.claude_api_key,
    #         model=config.claude_model,
    #         temperature=config.claude_temperature,
    #         max_tokens=config.claude_max_tokens,
    #     )
    #     fallback = _create_ollama_backend(config)
    #     return _FallbackChainBackend(primary, fallback)

    # --- OpenAI (commented out) ---
    # if provider == "openai":
    #     if not config.openai_api_key:
    #         logger.warning(
    #             "OpenAI selected but OPENAI_API_KEY is not set — "
    #             "falling back to default backend."
    #         )
    #         return _FallbackLLMBackend("openai", "OPENAI_API_KEY")
    #     primary = OpenAILLMBackend(
    #         api_key=config.openai_api_key,
    #         model=config.openai_model,
    #         temperature=config.openai_temperature,
    #         max_tokens=config.openai_max_tokens,
    #         timeout=config.llm_timeout,
    #     )
    #     fallback = _create_ollama_backend(config)
    #     return _FallbackChainBackend(primary, fallback)

    # logger.warning(
    #     "Unknown LLM provider '%s'. Supported: %s. Using fallback.",
    #     provider, ", ".join(_PROVIDERS),
    # )
    # return _FallbackLLMBackend(provider)


# ---------------------------------------------------------------------------
# Fallback chain: primary → Ollama
# ---------------------------------------------------------------------------

class _FallbackChainBackend:
    """
    Wraps a primary LLM backend with an Ollama fallback.

    If ``primary.generate()`` returns an error message (starts with ⚠️)
    or raises an exception, the request is transparently forwarded to
    the fallback (Ollama) backend.
    """

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_name = type(primary).__name__
        self._fallback_name = type(fallback).__name__
        logger.info(
            "LLM fallback chain: %s → %s",
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
