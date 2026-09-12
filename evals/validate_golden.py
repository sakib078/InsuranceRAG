"""Resolve every gold locator against data/chunks/*.jsonl. Run before any metric is computed.

A gold label that matches no chunk scores zero forever and reads as a retrieval failure, so
this is the guard that keeps the golden set honest as the corpus is re-chunked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from insurance_rag.config import DATA_DIR
from insurance_rag.corpus.manifest import Status, load_manifest

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
CHUNKS_DIR = DATA_DIR / "chunks"
HOPS = {"single", "multi", "none"}


def load_records(path: Path = GOLDEN_PATH) -> list[dict]:
    """One JSON object per line; blank lines ignored so the file can be grouped by slice."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def chunk_index() -> tuple[dict[str, list[str]], dict[str, str]]:
    """locator -> chunk_ids, and chunk_id -> doc_id. Built from the chunks, not the manifest."""
    locators: dict[str, list[str]] = {}
    owners: dict[str, str] = {}
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                locators.setdefault(chunk["locator"], []).append(chunk["chunk_id"])
                owners[chunk["chunk_id"]] = chunk["doc_id"]
    return locators, owners


def resolve(gold: str, locators: dict[str, list[str]]) -> list[str]:
    """Exact match, or the ` #2` sub-chunks an oversized provision was split into."""
    hits = list(locators.get(gold, []))
    for locator, ids in locators.items():
        if locator.startswith(f"{gold} #"):
            hits.extend(ids)
    return hits


def check(
    records: list[dict],
    locators: dict[str, list[str]],
    owners: dict[str, str],
    revoked: set[str],
) -> list[str]:
    """Every structural rule the harness later assumes; returns human-readable failures."""
    problems: list[str] = []
    seen: set[str] = set()

    for record in records:
        rid = record.get("id", "<no id>")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)

        if record.get("hop") not in HOPS:
            problems.append(f"{rid}: hop {record.get('hop')!r} not in {sorted(HOPS)}")
        if not (record.get("answer") or "").strip():
            problems.append(f"{rid}: empty answer - Correctness has nothing to judge against")

        gold = record.get("gold_locators", [])
        exclusions = record.get("exclusion_locators", [])

        if record.get("answerable") is False:
            if gold:
                problems.append(f"{rid}: answerable=false but carries gold_locators")
            if record.get("hop") != "none":
                problems.append(f"{rid}: answerable=false should have hop 'none'")
            continue

        if not gold:
            problems.append(f"{rid}: answerable but no gold_locators")
        if record.get("hop") == "multi" and len(gold) < 2:
            problems.append(f"{rid}: hop=multi with {len(gold)} gold locator(s)")
        for locator in exclusions:
            if locator not in gold:
                problems.append(f"{rid}: exclusion locator not in gold_locators - {locator}")

        for locator in gold:
            ids = resolve(locator, locators)
            if not ids:
                problems.append(f"{rid}: UNRESOLVED - {locator}")
                continue
            if len(ids) > 1:
                problems.append(f"{rid}: AMBIGUOUS - {locator} resolves to {len(ids)} chunks")
            for chunk_id in ids:
                if owners[chunk_id] in revoked:
                    problems.append(f"{rid}: REVOKED source - {locator} ({owners[chunk_id]})")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the golden set against the corpus.")
    parser.add_argument("--path", type=Path, default=GOLDEN_PATH)
    args = parser.parse_args()

    records = load_records(args.path)
    locators, owners = chunk_index()
    revoked = {r.doc_id for r in load_manifest() if r.status is Status.REVOKED}
    problems = check(records, locators, owners, revoked)

    slices = Counter(r.get("hop") for r in records)
    print(f"{len(records)} records over {len(locators)} distinct locators")
    print(f"  single-hop {slices['single']:3}   multi-hop {slices['multi']:3}   "
          f"unanswerable {slices['none']:3}")
    gold_total = sum(len(r.get("gold_locators", [])) for r in records)
    excl = sum(1 for r in records if r.get("exclusion_locators"))
    print(f"  {gold_total} gold locators; {excl} records carry an exclusion label")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print("\nall locators resolve")


if __name__ == "__main__":
    main()
