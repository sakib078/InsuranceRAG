"""Sync evals/golden.jsonl to a LangSmith dataset. Idempotent: re-running updates, never doubles.

git holds the labels; LangSmith runs the experiments. The record `id` is the join key, carried
in each example's metadata, so editing a label and re-pushing edits that one example in place.
"""

from __future__ import annotations

import argparse

from langsmith import Client

from insurance_rag.config import settings
from insurance_rag.corpus.manifest import Status, load_manifest
from evals.validate_golden import GOLDEN_PATH, chunk_index, check, load_records

#: Everything except question and answer rides as metadata for the custom evaluators.
LABEL_FIELDS = ("id", "hop", "gold_locators", "exclusion_locators", "answerable")


def to_example(record: dict) -> dict:
    """`question` -> inputs, `answer` -> outputs, the labels -> metadata."""
    return {
        "inputs": {"question": record["question"]},
        "outputs": {"answer": record["answer"]},
        "metadata": {field: record[field] for field in LABEL_FIELDS},
    }


def existing_by_id(client: Client, dataset_id) -> dict[str, str]:
    """Record id -> LangSmith example id, for the examples already in the dataset."""
    found = {}
    for example in client.list_examples(dataset_id=dataset_id):
        record_id = (example.metadata or {}).get("id")
        if record_id:
            found[record_id] = example.id
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=settings.langsmith_dataset)
    parser.add_argument("--dry-run", action="store_true", help="validate and report, push nothing")
    args = parser.parse_args()

    records = load_records(GOLDEN_PATH)
    locators, owners = chunk_index()
    revoked = {r.doc_id for r in load_manifest() if r.status is Status.REVOKED}
    problems = check(records, locators, owners, revoked)
    if problems:
        raise SystemExit(f"golden set has {len(problems)} problem(s); run validate_golden.py")

    if args.dry_run:
        print(f"{len(records)} records ready for dataset {args.dataset!r}")
        return

    client = Client()
    if not client.has_dataset(dataset_name=args.dataset):
        client.create_dataset(args.dataset, description="Ontario auto-insurance golden set")
    dataset = client.read_dataset(dataset_name=args.dataset)

    known = existing_by_id(client, dataset.id)
    updates = [
        {"id": known[r["id"]], **to_example(r)} for r in records if r["id"] in known
    ]
    creates = [to_example(r) for r in records if r["id"] not in known]

    if creates:
        client.create_examples(dataset_id=dataset.id, examples=creates)
    if updates:
        client.update_examples(dataset_id=dataset.id, updates=updates)
    print(f"{args.dataset}: {len(creates)} created, {len(updates)} updated")


if __name__ == "__main__":
    main()
