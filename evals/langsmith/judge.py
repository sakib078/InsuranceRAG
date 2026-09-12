"""The grader LLM, shared by all four evaluators.

Deliberately a different provider and family from the generator: a judge on the generator's
provider would share its rate limit, and a model grading its own family favours its own
phrasing. The endpoints live in `insurance_rag/providers.py`.
"""

from __future__ import annotations

from functools import lru_cache

from insurance_rag.config import settings
from insurance_rag.providers import chat_model
from insurance_rag.ratelimit import with_retry

__all__ = ["grader", "facts", "grade", "score"]


@lru_cache(maxsize=8)
def grader(schema: type):
    """A judge that must answer in `schema`; cached per schema so the client is built once."""
    # No method="json_schema": that is OpenAI-only, and these are compatibility endpoints.
    return chat_model(settings.judge_provider, settings.judge_model).with_structured_output(schema)


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
