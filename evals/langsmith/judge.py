"""The grader LLM, shared by all four evaluators.

Deliberately a different provider and family from the generator: Groq serves the answers, so a
judge there would share the rate limit, and a model grading its own family favours its own
phrasing. Both providers below expose OpenAI-compatible endpoints, so neither needs its own SDK.
"""

from __future__ import annotations

import random
import re
import time
from functools import lru_cache

from insurance_rag.config import settings

#: provider -> (base_url, which settings field holds the key, where to get one)
PROVIDERS = {
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
}

__all__ = ["grader", "facts", "grade", "score"]


@lru_cache(maxsize=8)
def grader(schema: type):
    """A judge that must answer in `schema`; cached per schema so the client is built once."""
    from langchain_openai import ChatOpenAI

    if settings.judge_provider not in PROVIDERS:
        raise SystemExit(f"IRAG_JUDGE_PROVIDER must be one of {sorted(PROVIDERS)}")
    base_url, key_field, where = PROVIDERS[settings.judge_provider]

    api_key = getattr(settings, key_field)
    if not api_key:
        raise SystemExit(f"{settings.judge_provider} judge needs a key in .env - see {where}")

    llm = ChatOpenAI(
        model=settings.judge_model, api_key=api_key, base_url=base_url, temperature=0
    )
    # No method="json_schema": that is an OpenAI-only option, and these are compat endpoints.
    return llm.with_structured_output(schema)


def facts(outputs: dict) -> str:
    """The retrieved chunks as one block - the published prompts call this FACTS."""
    return "\n\n".join(outputs.get("retrieved_text", []))


#: Free judge tiers are tight - Gemini 2.5 Flash allows 5 requests a minute - and a generation
#: run is 56 records x 4 judges, so a rate limit is the expected path rather than an error.
MAX_ATTEMPTS = 6
MAX_BACKOFF = 90.0

_RETRY_AFTER_RE = re.compile(r"retry in ([\d.]+)s|retryDelay['\"]?:\s*['\"]?(\d+)s")


def _wait_for(exc: Exception, attempt: int) -> float:
    """Honour the provider's own retry hint when it gives one; otherwise back off and jitter."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        hinted = float(match.group(1) or match.group(2))
        return min(hinted + 1.0, MAX_BACKOFF)
    return min(2.0**attempt + random.uniform(0, 1), MAX_BACKOFF)


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "rate" in type(exc).__name__.lower()


def grade(schema: type, instructions: str, message: str) -> dict:
    """One judge call, retried through rate limits. Concurrency is owned by run_eval."""
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": message}]
    for attempt in range(MAX_ATTEMPTS):
        try:
            return grader(schema).invoke(messages)
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(_wait_for(exc, attempt))
    raise RuntimeError("unreachable")


def score(key: str, value: bool | None) -> dict:
    return {"key": key, "score": bool(value)}
