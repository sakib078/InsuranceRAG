"""Neighbours checkpoint: does ordinal expansion rescue multi-hop records? Zero LLM tokens."""

from __future__ import annotations

import json

from evals.evaluators import _found
from insurance_rag.agent.tools import expand
from insurance_rag.config import EVALS_DIR, settings
from insurance_rag.retrieval.search import search_corpus
from insurance_rag.schema import Chunk


def multi_records() -> list[dict]:
    """The multi-hop golden records that have gold locators to find."""
    lines = (EVALS_DIR / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    records = map(json.loads, filter(str.strip, lines))
    return [r for r in records if r["hop"] == "multi" and r["gold_locators"]]


def passes(record: dict, pool: list[Chunk]) -> bool:
    """Multi-hop passes only when every gold locator is in the pool."""
    locators = [c.locator for c in pool]
    return all(_found(gold, locators) for gold in record["gold_locators"])


def main() -> None:
    """Score top 5, top 5 plus neighbours, and the top-25 control; list who each rescues."""
    records = multi_records()
    names = ("top 5", "top 5 + neighbours", "top 25 (control)")
    passed: dict[str, set[str]] = {name: set() for name in names}
    sizes = []
    for record in records:
        seeds = search_corpus(record["question"], k=5)
        pools = {
            "top 5": seeds,
            "top 5 + neighbours": seeds + expand(seeds),
            "top 25 (control)": search_corpus(record["question"], k=settings.ladder_top_k),
        }
        sizes.append(len(pools["top 5 + neighbours"]))
        for name, pool in pools.items():
            if passes(record, pool):
                passed[name].add(record["id"])
    for name, ids in passed.items():
        print(f"{name:<20} {len(ids)}/{len(records)} = {len(ids) / len(records):.3f}")
    print(f"mean pool size with neighbours: {sum(sizes) / len(sizes):.1f} (control: 25)")
    print("rescued by neighbours:", sorted(passed["top 5 + neighbours"] - passed["top 5"]))
    print("rescued by the control:", sorted(passed["top 25 (control)"] - passed["top 5"]))


if __name__ == "__main__":
    main()
