"""The grader LLM, shared by all four evaluators.

Deliberately a different provider and family from the generator: Groq serves the answers, so a
judge there would share the rate limit, and a model grading its own family favours its own
phrasing. Both providers below expose OpenAI-compatible endpoints, so neither needs its own SDK.
"""

from __future__ import annotations

from functools import lru_cache

from insurance_rag.config import settings
from insurance_rag.ratelimit import with_retry

#: provider -> (base_url, which settings field holds the key, where to get one).
#: Groq quotas are per-model, so judging on Qwen never touches the generator's budget.
PROVIDERS = {
    "groq": (
        "https://api.groq.com/openai/v1",
        "groq_api_key",
        "console.groq.com/keys",
    ),
    "ollama": (
        "http://localhost:11434/v1",
        "ollama_api_key",
        "a local container - smoke tests only, never a recorded row",
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


def grade(schema: type, instructions: str, message: str) -> dict:
    """One judge call, retried through rate limits. Concurrency is owned by run_eval."""
    messages = [{"role": "system", "content": instructions},
                {"role": "user", "content": message}]
    return with_retry(lambda: grader(schema).invoke(messages))


def score(key: str, value: bool | None) -> dict:
    return {"key": key, "score": bool(value)}
