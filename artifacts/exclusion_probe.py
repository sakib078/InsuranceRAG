"""Exclusion checkpoint: does the exclusion pass beat single-shot at the same size? Zero tokens."""

from __future__ import annotations

import json

from evals.evaluators import exclusion_recall
from insurance_rag.agent.tools import WEIGHTS, assemble, exclusion_pass
from insurance_rag.config import EVALS_DIR
from insurance_rag.retrieval.search import search_corpus
from insurance_rag.schema import Chunk


def exclusion_records() -> list[dict]:
    """The golden records that have a limiting provision to find."""
    lines = (EVALS_DIR / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    records = map(json.loads, filter(str.strip, lines))
    return [r for r in records if r["exclusion_locators"]]


def context(question: str, main: list[Chunk], weights: dict[str, float]) -> list[Chunk]:
    """The main pool plus the exclusion pass, assembled to the token limit."""
    found = [hit.chunk for hit in exclusion_pass(question, weights=weights)]
    return assemble({"main": main, "exclusion": found})


def main() -> None:
    """Exclusion recall for each arm, with each pass channel switched off in turn."""
    scores: dict[str, list[float]] = {}
    sizes = []
    for record in exclusion_records():
        question, gold = record["question"], record["exclusion_locators"]
        main_pool = search_corpus(question, k=5)
        full = context(question, main_pool, WEIGHTS)
        arms = {
            "top 5": main_pool,
            "top 20 (reference)": search_corpus(question, k=20),
            "assembled": full,
            "same-size control": search_corpus(question, k=len(full)),
        }
        for channel in WEIGHTS:
            off = {**WEIGHTS, channel: 0}
            arms[f"assembled, no {channel}"] = context(question, main_pool, off)
        sizes.append(len(full))
        for name, pool in arms.items():
            scores.setdefault(name, []).append(
                exclusion_recall([c.locator for c in pool], gold, k=len(pool))
            )
    for name, values in scores.items():
        print(f"{name:<24} {sum(values) / len(values):.3f}")
    print(f"\nrecords: {len(sizes)}, mean assembled size: {sum(sizes) / len(sizes):.1f} chunks")


if __name__ == "__main__":
    main()
