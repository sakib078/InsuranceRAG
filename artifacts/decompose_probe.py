"""Phase 7 kill-check: can a provision's own text retrieve it at k=5? Zero LLM tokens."""

from __future__ import annotations

import json

from evals.evaluators import _matches
from insurance_rag.config import EVALS_DIR, RESULTS_DIR, settings
from insurance_rag.retrieval.search import search_corpus
from insurance_rag.retrieval.store import read_chunks
from insurance_rag.schema import Chunk

__all__ = ["gold_provisions", "self_query", "probe", "GATE"]

POOL = RESULTS_DIR / "pool_k20_retrieval.json"
GATE = 0.68  # break-even: p ** 2.5 equals the 0.381 single-query ceiling


def failing_records() -> list[str]:
    """Multi-hop record ids that miss at k=20."""
    return json.loads(POOL.read_text(encoding="utf-8"))["misses"]["recall@20_multi"]


def _golden() -> dict[str, dict]:
    lines = (EVALS_DIR / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["id"]: r for r in map(json.loads, filter(str.strip, lines))}


def gold_provisions(record_ids: list[str]) -> list[tuple[str, Chunk]]:
    """(record id, first chunk) for every gold locator of the given records."""
    chunks = [
        c
        for path in sorted(settings.chunks_dir.glob("*.jsonl"))
        if not path.name.startswith("_")
        for c in read_chunks(path)
    ]
    golden = _golden()
    found = []
    for rid in record_ids:
        for gold in golden[rid]["gold_locators"]:
            match = next((c for c in chunks if _matches(gold, c.locator)), None)
            if match is None:
                raise ValueError(f"{rid}: no chunk for {gold!r}")
            found.append((rid, match))
    return found


def self_query(chunk: Chunk, *, chars: int = 200) -> str:
    """The provision's own heading plus its first sentence, as a query."""
    body = chunk.text.split("\n", 2)[2].lstrip("# ")
    heading = chunk.ancestor_path[-1] if chunk.ancestor_path else ""
    return " ".join(f"{heading}. {body.split('. ', 1)[0]}".split())[:chars]


def probe(record_ids: list[str], *, k: int = 5) -> dict[str, bool]:
    """Per provision: does its self-query bring it back in the top k?"""
    outcome = {}
    for rid, chunk in gold_provisions(record_ids):
        base = chunk.locator.split(" #")[0]
        hits = search_corpus(self_query(chunk), k=k)
        outcome[f"{rid} {base}"] = any(_matches(base, h.locator) for h in hits)
    return outcome


def main() -> None:
    """Print each provision's result, the self-retrieval rate, and the verdict."""
    outcome = probe(failing_records())
    hits = sum(outcome.values())
    p = hits / len(outcome)
    for key, ok in outcome.items():
        print(f"{'hit ' if ok else 'MISS'}  {key}")
    ceiling = json.loads(POOL.read_text(encoding="utf-8"))["scores"]["recall@20_multi"]
    print(f"\nself-retrieval@5: {hits}/{len(outcome)} = {p:.3f}")
    print(f"oracle decomposition cap p^2.5 = {p ** 2.5:.3f}")
    print(f"single-query ceiling at k=20   = {ceiling:.3f}")
    print("PROCEED" if p >= GATE else "KILL: record Phase 7 as measured-and-dropped")


if __name__ == "__main__":
    main()
