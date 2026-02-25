"""
LLM Service for Centurion Capital LLC RAG Pipeline.

Provides LLM-powered answer generation from RAG-retrieved context.
Supports local models via Ollama (Llama 3, Mistral, etc.) and is
designed to be easily extended to cloud APIs (OpenAI, Anthropic).

Ollama setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ``ollama pull llama3``  or  ``ollama pull mistral``
    3. Ollama runs as a local server on http://localhost:11434

Usage:
    from rag_pipeline.llm_service import OllamaLLMBackend

    llm = OllamaLLMBackend(model="llama3")
    answer = llm.generate(query="What is RSI?", context="RSI is ...")
"""

import json
import logging
from typing import Optional

import requests

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for RAG grounding
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are Centurion AI, an expert financial analyst and trading strategy assistant \
for Centurion Capital LLC.

Your task is to answer the user's question using ONLY the context provided below. \
Follow these rules strictly:

1. Base your answer exclusively on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say so clearly.
3. Synthesise information from multiple context chunks into a single coherent answer.
4. Use clear, professional language. Format with markdown where helpful.
5. Cite the source document name when referencing specific information.
6. Be concise but thorough — do not repeat chunks verbatim.\
"""

NO_CONTEXT_SYSTEM_PROMPT = """\
You are Centurion AI, an expert financial analyst and trading strategy assistant \
for Centurion Capital LLC.

Answer the user's question using your general knowledge. Be concise, professional, \
and format with markdown where helpful.\
"""


# ---------------------------------------------------------------------------
# Ollama backend (local, free, runs Llama 3 / Mistral / etc.)
# ---------------------------------------------------------------------------

class OllamaLLMBackend:
    """
    LLM backend using Ollama's local REST API.

    Ollama serves models at http://localhost:11434 by default.
    Supports streaming but we use non-streaming for simplicity.
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

    def _build_prompt(self, query: str, context: str) -> str:
        """Build a single prompt string for the Ollama generate API."""
        if context.strip():
            system_msg = RAG_SYSTEM_PROMPT
            user_msg = (
                f"## Context\n\n{context}\n\n"
                f"---\n\n"
                f"## Question\n\n{query}"
            )
        else:
            system_msg = NO_CONTEXT_SYSTEM_PROMPT
            user_msg = query

        return f"{system_msg}\n\n{user_msg}"

    def generate(self, query: str, context: str) -> str:
        """
        Generate a response using Ollama's generate API.

        Falls back to a helpful error message if Ollama is unreachable.
        """
        prompt = self._build_prompt(query, context)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            answer = data.get("response", "")

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
            return "⚠️ LLM request timed out. Try a shorter query or increase the timeout."
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
