"""OpenAI-compatible chat endpoints. One table for the generator and the eval judges.

Every provider here speaks the OpenAI wire format, so a single `ChatOpenAI` with a `base_url`
reaches all of them and no provider SDK is needed. Keeping generation and judging on separate
providers is deliberate: they then draw on separate budgets, and a model family never grades
its own output.
"""

from __future__ import annotations

from functools import lru_cache

from insurance_rag.config import settings

#: name -> (base_url, the settings field holding the key, where to get one)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "cerebras": (
        "https://api.cerebras.ai/v1",
        "cerebras_api_key",
        "cloud.cerebras.ai",
    ),
    "groq": (
        "https://api.groq.com/openai/v1",
        "groq_api_key",
        "console.groq.com/keys",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini_eval_key",
        "aistudio.google.com/apikey",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "openrouter_eval_key",
        "openrouter.ai/keys",
    ),
    "ollama": (
        "http://localhost:11434/v1",
        "ollama_api_key",
        "a local container - smoke tests only, never a recorded row",
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
