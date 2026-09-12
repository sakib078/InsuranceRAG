"""Custom evaluators: accurate retrieval and accurate citation, scored against hand labels.

"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from functools import lru_cache

from insurance_rag.config import DATA_DIR, settings
from insurance_rag.corpus.manifest import Status, load_manifest

__all__ = [
    "recall", "exclusion_recall", "citation_accuracy",
    "recall_single", "recall_multi", "exclusion_recall_eval", "citation_accuracy_eval",
]

CHUNKS_DIR = DATA_DIR / "chunks"


def _matches(gold: str, locator: str) -> bool:
    """A gold label also matches the ` #2` sub-chunks an oversized provision was split into."""
    return locator == gold or locator.startswith(f"{gold} #")


def _found(gold: str, locators: Iterable[str]) -> bool:
    return any(_matches(gold, locator) for locator in locators)


@lru_cache(maxsize=1)
def _corpus() -> tuple[dict[str, str], frozenset[str]]:
    """locator -> doc_id, and the revoked doc_ids, for judging whether a citation is real."""
    owners: dict[str, str] = {}
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                owners[chunk["locator"]] = chunk["doc_id"]
    revoked = frozenset(r.doc_id for r in load_manifest() if r.status is Status.REVOKED)
    return owners, revoked


# scoring 

def recall(
    retrieved: Sequence[str],
    gold: Sequence[str],
    *,
    k: int = settings.rerank_top_k,
    require_all: bool = False,
) -> float | None:
    """Single-hop needs any gold chunk in the top k; multi-hop needs every one of them."""
    if not gold:
        return None
    hits = sum(1 for g in gold if _found(g, retrieved[:k]))
    return float(hits == len(gold)) if require_all else float(hits > 0)


def exclusion_recall(
    retrieved: Sequence[str],
    exclusions: Sequence[str],
    *,
    k: int = settings.rerank_top_k,
) -> float | None:
    """Did the limiting provision surface at all? `None` on records where none applies."""
    return recall(retrieved, exclusions, k=k, require_all=False)


def citation_accuracy(
    cited: Sequence[str],
    retrieved: Sequence[str],
    gold: Sequence[str],
) -> float:
    """Zero for any fabricated, unretrieved or revoked citation; else the share of gold cited."""
    owners, revoked = _corpus()
    if not gold:  # an unanswerable question must cite nothing at all
        return float(not cited)

    for locator in cited:
        if locator not in owners:
            return 0.0  # cites a provision that does not exist
        if not _found(locator, retrieved) and locator not in retrieved:
            return 0.0  # cites something it was never shown
        if owners[locator] in revoked:
            return 0.0  # cites revoked law as if it were current

    return sum(1 for g in gold if _found(g, cited)) / len(gold)


# --- LangSmith adapters ----------------------------------------------------------------------
#
# `outputs` is what the target returned; `reference_outputs` carries the golden-set labels.
# Returning None makes LangSmith skip the record rather than score it zero.

def _labels(reference_outputs: dict) -> dict:
    return reference_outputs or {}


def recall_single(outputs: dict, reference_outputs: dict) -> dict | None:
    labels = _labels(reference_outputs)
    if labels.get("hop") != "single":
        return None
    score = recall(outputs.get("retrieved_locators", []), labels.get("gold_locators", []))
    return None if score is None else {"key": "recall@5_single", "score": score}


def recall_multi(outputs: dict, reference_outputs: dict) -> dict | None:
    labels = _labels(reference_outputs)
    if labels.get("hop") != "multi":
        return None
    score = recall(
        outputs.get("retrieved_locators", []), labels.get("gold_locators", []), require_all=True
    )
    return None if score is None else {"key": "recall@5_multi", "score": score}


def exclusion_recall_eval(outputs: dict, reference_outputs: dict) -> dict | None:
    labels = _labels(reference_outputs)
    score = exclusion_recall(
        outputs.get("retrieved_locators", []), labels.get("exclusion_locators", [])
    )
    return None if score is None else {"key": "exclusion_recall", "score": score}


def citation_accuracy_eval(outputs: dict, reference_outputs: dict) -> dict:
    labels = _labels(reference_outputs)
    return {
        "key": "citation_accuracy",
        "score": citation_accuracy(
            outputs.get("cited_locators", []),
            outputs.get("retrieved_locators", []),
            labels.get("gold_locators", []),
        ),
    }
