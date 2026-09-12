"""Push evals/golden.jsonl to a LangSmith dataset.

`python -m evals.push_dataset`

git holds the labels; LangSmith runs the experiments. Re-running replaces the dataset's examples
rather than doubling them.
"""

from __future__ import annotations

import argparse

from langsmith import Client

from insurance_rag.config import settings
from evals.validate_golden import GOLDEN_PATH, load_records

#: The labels the custom evaluators read back.
LABEL_FIELDS = ("id", "hop", "gold_locators", "exclusion_locators", "answerable")


def to_example(record: dict) -> dict:
    """`question` -> inputs; the answer AND the labels -> outputs.

    Labels must live in `outputs`, not `metadata`: LangSmith hands an evaluator
    `example.outputs` as `reference_outputs` and never hands it metadata. Labels put in
    metadata arrive empty, and every evaluator then skips the record without erroring.
    Metadata keeps a copy, which is what the LangSmith UI filters on.
    """
    labels = {field: record[field] for field in LABEL_FIELDS}
    return {
        "inputs": {"question": record["question"]},
        "outputs": {"answer": record["answer"], **labels},
        "metadata": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=settings.langsmith_dataset)
    args = parser.parse_args()

    if not settings.langsmith_api_key:
        raise SystemExit("set LANGSMITH_API_KEY in .env - get one at smith.langchain.com")

    records = load_records(GOLDEN_PATH)
    client = Client(api_key=settings.langsmith_api_key)

    if client.has_dataset(dataset_name=args.dataset):
        # Simpler than reconciling ids, and the golden set is the source of truth either way.
        client.delete_dataset(dataset_name=args.dataset)
    dataset = client.create_dataset(args.dataset, description="Ontario auto-insurance golden set")

    client.create_examples(dataset_id=dataset.id, examples=[to_example(r) for r in records])
    print(f"{args.dataset}: {len(records)} examples")


if __name__ == "__main__":
    main()
