"""OpenAI-compatible chat endpoints: one table for the generator and the eval judges."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar

from insurance_rag.config import settings
from insurance_rag.ratelimit import is_exhausted

T = TypeVar("T")

#: name -> (base_url, the settings field holding the key, where to get one)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "groq": (
        "https://api.groq.com/openai/v1",
        "groq_api_key",
        "console.groq.com/keys",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "openrouter_key",
        "openrouter.ai/keys",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini_eval_key",
        "aistudio.google.com/apikey",
    ),
    # Free tier covers the ministral sizes only.
    "mistral": (
        "https://api.mistral.ai/v1",
        "mistral_api_key",
        "console.mistral.ai - needs phone verification and a data-training opt-in",
    ),
    # Lists far more models than a free account may call; slow, ~22s a call.
    "nvidia": (
        "https://integrate.api.nvidia.com/v1",
        "nvidia_api_key",
        "build.nvidia.com - free Developer Program, no card",
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


def judge_chain() -> list[tuple[str, str]]:
    """The judge candidates in preference order, split on the first slash only."""
    return [
        (spec.split("/", 1)[0].strip(), spec.split("/", 1)[1].strip())
        for spec in settings.judge_chain.split(",")
        if "/" in spec
    ]


#: Judge candidates whose budget ran out, and those that served; process-lifetime.
_exhausted: set[tuple[str, str]] = set()
_served: dict[tuple[str, str], int] = {}


def judges_used() -> dict[str, int]:
    """Which judges answered how many calls - a mixed run is not one judge's verdict."""
    return {f"{provider}/{model}": n for (provider, model), n in _served.items()}


def with_failover(call: Callable[[str, str], T]) -> T:
    """Run `call` on the first judge with budget left; step down the chain as each runs out."""
    chain = judge_chain()
    if not chain:
        raise SystemExit("IRAG_JUDGE_CHAIN is empty - see insurance_rag/providers.py")

    spent = []
    for provider, model in chain:
        if (provider, model) in _exhausted:
            continue
        try:
            result = call(provider, model)
        except Exception as exc:
            # Anything that is not a budget wall - a bad model id, a broken schema - must surface.
            if not is_exhausted(exc):
                raise
            _exhausted.add((provider, model))
            spent.append(f"{provider}/{model}")
            print(f"  judge {provider}/{model} is out of budget; falling back")
            continue
        _served[(provider, model)] = _served.get((provider, model), 0) + 1
        return result

    raise SystemExit(
        f"every judge in the chain is out of budget ({', '.join(spent) or 'all already spent'}). "
        "Add another provider to IRAG_JUDGE_CHAIN or wait for a quota to reset."
    )
