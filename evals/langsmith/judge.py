"""The grader LLM, shared by all four evaluators.

Deliberately a different provider and family from the generator: Groq serves the answers, so a
judge there would share the rate limit, and a model grading its own family favours its own
phrasing. OpenRouter is OpenAI-compatible, so this needs no dependency beyond langchain-openai.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from insurance_rag.config import settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@lru_cache(maxsize=8)
def grader(schema: type):
    """A judge that must answer in `schema`; cached per schema so the client is built once."""
    from langchain_openai import ChatOpenAI

    if not settings.openrouter_eval_key:
        raise SystemExit("set OPEN_ROUTER_EVAL_KEY in .env - get one at openrouter.ai/keys")
    llm = ChatOpenAI(
        model=settings.judge_model,
        api_key=settings.openrouter_eval_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )
    # No method="json_schema": not every OpenRouter model supports it, unlike the OpenAI default.
    return llm.with_structured_output(schema)


def facts(outputs: dict) -> str:
    """The retrieved chunks as one block - the published prompts call this FACTS."""
    return "\n\n".join(outputs.get("retrieved_text", []))


def grade(schema: type, instructions: str, message: str) -> dict:
    """One judge call. Retries live in run_eval, which owns concurrency and backoff."""
    return grader(schema).invoke(
        [{"role": "system", "content": instructions}, {"role": "user", "content": message}]
    )


def score(key: str, value: bool | None) -> dict:
    return {"key": key, "score": bool(value)}


__all__: Sequence[str] = ["grader", "facts", "grade", "score"]
