"""OpenAI-compatible chat endpoints. One table for the generator and the eval judges.

Every provider here speaks the OpenAI wire format, so a single `ChatOpenAI` with a `base_url`
reaches all of them and no provider SDK is needed. Keeping generation and judging apart is
deliberate: separate budgets, and a model family never grades its own output.

Groq is the default for both. Ollama is the local escape hatch when a quota is spent; Gemini is
the hosted backup. Cerebras and OpenRouter were wired and removed - Cerebras returned 402 with
no active trial, and OpenRouter's 50 requests/day cannot cover a 168-request run.
"""

from __future__ import annotations

from functools import lru_cache

from insurance_rag.config import settings

#: name -> (base_url, the settings field holding the key, where to get one)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "groq": (
        "https://api.groq.com/openai/v1",
        "groq_api_key",
        "console.groq.com/keys",
    ),
    "ollama": (
        "http://localhost:11434/v1",
        "ollama_api_key",
        "a local container - see docs/plan.md",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini_eval_key",
        "aistudio.google.com/apikey",
    ),
}


@lru_cache(maxsize=8)
def chat_model(provider: str, model: str, *, max_tokens: int | None = None):
    """A chat model on `provider`. Cached, so each provider's client is built once."""
    from langchain_openai import ChatOpenAI

    if provider not in PROVIDERS:
        raise SystemExit(f"provider must be one of {sorted(PROVIDERS)}, not {provider!r}")
    base_url, key_field, where = PROVIDERS[provider]

    api_key = getattr(settings, key_field)
    if not api_key:
        raise SystemExit(f"{provider} needs a key in .env - see {where}")

    return ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url, temperature=0, max_tokens=max_tokens
    )
