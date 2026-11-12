"""The Planner agent: split a question into a few sub-questions, cached on disk."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from insurance_rag.config import DATA_DIR, EVALS_DIR, settings
from insurance_rag.providers import chat_model

__all__ = ["DECOMPOSE_SYSTEM", "decompose", "build_cache"]

log = logging.getLogger(__name__)

CACHE = DATA_DIR / "decompositions.jsonl"
#: Tried when the planner model is down (a 503 under load); a sibling on the same key.
FALLBACK_MODEL = "gemini/models/gemini-3.5-flash-lite"

DECOMPOSE_SYSTEM = """You help search Ontario auto insurance law. A question may need several \
separate provisions to answer. Write one short search query per provision needed, one per line, \
at most {limit}. If one provision is enough, write the question once. No numbering, \
no commentary."""


def _key(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def _load(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    return {r["key"]: r["sub_questions"] for r in map(json.loads, filter(str.strip, lines))}


def _ask(question: str) -> list[str] | None:
    """One model call, repeated on the fallback model; None when both are unavailable."""
    limit = settings.agent_max_sub_questions
    messages = [("system", DECOMPOSE_SYSTEM.format(limit=limit)), ("human", question)]
    for spec in dict.fromkeys((settings.decompose_model, FALLBACK_MODEL)):
        provider, model = spec.split("/", 1)
        try:  # no waiting here: a 503 is cheaper to route around than to sleep through
            reply = chat_model(provider, model, max_tokens=256).invoke(messages)
        except Exception as exc:
            log.warning("planner model %s failed: %s", spec, exc)
            continue
        lines = [line.strip(" -*\t") for line in reply.content.splitlines()]
        return [line for line in lines if line and line != question][:limit]
    return None


def decompose(question: str, *, cache: Path = CACHE) -> list[str]:
    """Sub-questions for `question`; the cache answers every question it has seen."""
    key = _key(question)
    cached = _load(cache)
    if key not in cached:
        subs = _ask(question)
        if subs is None:  # both models are down: search the question alone, ask again next time
            return []
        cached[key] = subs
        record = {"key": key, "question": question, "sub_questions": cached[key]}
        with cache.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    return cached[key]


def build_cache(questions: list[str], path: Path = CACHE) -> int:
    """Decompose every question not yet cached; returns how many model calls that took."""
    before = len(_load(path))
    for question in questions:
        decompose(question, cache=path)
    return len(_load(path)) - before


def main() -> None:
    """Fill the cache for the golden set."""
    lines = (EVALS_DIR / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [r["question"] for r in map(json.loads, filter(str.strip, lines))]
    print(f"{build_cache(questions)} new decompositions cached at {CACHE}")


if __name__ == "__main__":
    main()
