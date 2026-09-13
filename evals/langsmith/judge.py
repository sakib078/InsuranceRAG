"""The grader LLM, shared by all four evaluators.

Deliberately a different provider and family from the generator: a judge on the generator's
provider would share its rate limit, and a model grading its own family favours its own
phrasing. `IRAG_JUDGE_CHAIN` lists candidates in preference order and `with_failover` steps
down it as each budget runs out, so one exhausted quota does not end a run. Endpoints and the
routing live in `insurance_rag/providers.py`.
"""

from __future__ import annotations

from functools import lru_cache

from insurance_rag.providers import chat_model, with_failover
from insurance_rag.ratelimit import with_retry

__all__ = ["grader", "facts", "grade", "score"]


@lru_cache(maxsize=16)
def grader(provider: str, model: str, schema: type):
    """A judge that must answer in `schema`; cached per candidate so each client is built once."""
    # No method="json_schema": that is OpenAI-only, and these are compatibility endpoints.
    return chat_model(provider, model).with_structured_output(schema)


def facts(outputs: dict) -> str:
    """The retrieved chunks as one block - the published prompts call this FACTS."""
    return "\n\n".join(outputs.get("retrieved_text", []))


def grade(schema: type, instructions: str, message: str) -> dict:
    """One judge call: sleep through per-minute limits, step down the chain when a budget ends."""
    messages = [{"role": "system", "content": instructions},
                {"role": "user", "content": message}]
    return with_failover(
        lambda provider, model: with_retry(lambda: grader(provider, model, schema).invoke(messages))
    )


def score(key: str, value: bool | None) -> dict:
    return {"key": key, "score": bool(value)}
