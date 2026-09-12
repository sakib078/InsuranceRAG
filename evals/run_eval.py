"""Score the golden set. Two suites, because they cost very different amounts.

    python -m evals.run_eval --suite retrieval   --config dense_clause_aware
    python -m evals.run_eval --suite generation  --config dense_clause_aware

`retrieval` makes no model calls at all and runs offline: recall@5 by hop and exclusion recall,
scored against the hand labels. `generation` answers all 56 questions and adds the four LangSmith
judges plus citation accuracy, so it runs through `client.evaluate` and needs both keys.

`--config` only names the row; the behaviour comes from whatever `search_corpus` currently does.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from insurance_rag.config import settings
from insurance_rag.retrieval.search import search_with_scores
from evals.evaluators import (
    citation_accuracy_eval, exclusion_recall_eval, recall_multi, recall_single,
)
from evals.validate_golden import GOLDEN_PATH, load_records

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LABEL_FIELDS = ("id", "hop", "gold_locators", "exclusion_locators", "answerable")

#: Citation accuracy needs an answer, so it only joins the generation suite.
RETRIEVAL_EVALUATORS = (recall_single, recall_multi, exclusion_recall_eval)
CUSTOM_EVALUATORS = (*RETRIEVAL_EVALUATORS, citation_accuracy_eval)
METRIC_ORDER = (
    "recall@5_single", "recall@5_multi", "exclusion_recall", "citation_accuracy",
    "correctness", "relevance", "groundedness", "retrieval_relevance",
)


def retrieval_target(question: str, k: int) -> dict:
    """What the pipeline retrieves, in rank order. No generation, so no API call."""
    hits = search_with_scores(question, k=k)
    return {
        "retrieved_locators": [chunk.locator for chunk, _ in hits],
        "retrieved_text": [chunk.text for chunk, _ in hits],
    }


def generation_target(question: str) -> dict:
    """The product as shipped - the retry ladder is left on, so k is not pinned."""
    from insurance_rag.generation.chain import answer

    result = answer(question)
    return {
        "answer": result.text,
        "retrieved_locators": [c.locator for c in result.retrieved],
        "retrieved_text": [c.text for c in result.retrieved],
        "cited_locators": [c.locator for c in result.chunks],
    }


def labels_of(record: dict) -> dict:
    return {field: record[field] for field in LABEL_FIELDS}


def report(totals: dict[str, list[float]], misses: list, show_misses: bool) -> dict:
    """Print the row and return the scores worth recording."""
    print(f"\n{'metric':<22}{'score':>8}{'n':>6}")
    summary = {}
    for key in METRIC_ORDER:
        values = totals.get(key, [])
        if values:
            summary[key] = sum(values) / len(values)
            print(f"{key:<22}{summary[key]:>8.3f}{len(values):>6}")
    if show_misses and misses:
        print("\nscored 0:")
        for key, record_id, question in misses:
            print(f"  {key:<20} {record_id}  {question[:66]}")
    return summary


def run_retrieval(records: list[dict], k: int, show_misses: bool) -> dict:
    """Offline: query pgvector once per record and score the custom retrieval evaluators."""
    totals: dict[str, list[float]] = defaultdict(list)
    misses: list[tuple[str, str, str]] = []
    for record in records:
        outputs = retrieval_target(record["question"], k)
        for evaluator in RETRIEVAL_EVALUATORS:
            verdict = evaluator(outputs, labels_of(record))
            if verdict is None:
                continue
            totals[verdict["key"]].append(verdict["score"])
            if verdict["score"] == 0.0:
                misses.append((verdict["key"], record["id"], record["question"]))
    return report(totals, misses, show_misses)


def run_generation(dataset: str, config: str, concurrency: int) -> dict:
    """Hosted: LangSmith runs the target over the dataset, then all eight evaluators."""
    from langsmith import Client

    from evals.langsmith import BUILT_INS

    if not settings.langsmith_api_key:
        raise SystemExit("set LANGSMITH_API_KEY in .env - get one at smith.langchain.com")

    client = Client(api_key=settings.langsmith_api_key)
    results = client.evaluate(
        lambda inputs: generation_target(inputs["question"]),
        data=dataset,
        evaluators=[*BUILT_INS, *CUSTOM_EVALUATORS],
        experiment_prefix=config,
        max_concurrency=concurrency,  # free tiers rate-limit long before they refuse
        metadata={"config": config, "encoder": str(settings.encoder),
                  "generation_model": settings.generation_model,
                  "judge_model": settings.judge_model},
    )

    totals: dict[str, list[float]] = defaultdict(list)
    for row in results:
        for result in row["evaluation_results"]["results"]:
            if result.score is not None:
                totals[result.key].append(float(result.score))
    return report(totals, [], False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("retrieval", "generation"), default="retrieval")
    parser.add_argument("--config", default="dense_clause_aware", help="name for this row")
    parser.add_argument("-k", type=int, default=settings.rerank_top_k)
    parser.add_argument("--dataset", default=settings.langsmith_dataset)
    # The judge free tier is the bottleneck (Gemini 2.5 Flash: 5 req/min), not the pipeline.
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--misses", action="store_true", help="list the records that scored 0")
    args = parser.parse_args()

    records = load_records(GOLDEN_PATH)
    print(f"{args.config} [{args.suite}]: {len(records)} records", end="")
    print(f" at k={args.k}" if args.suite == "retrieval" else " with the retry ladder on")

    if args.suite == "retrieval":
        summary = run_retrieval(records, args.k, args.misses)
    else:
        summary = run_generation(args.dataset, args.config, args.concurrency)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{args.config}_{args.suite}.json"
    path.write_text(
        json.dumps(
            {
                "config": args.config,
                "suite": args.suite,
                "at": datetime.now(timezone.utc).isoformat(),
                "k": args.k if args.suite == "retrieval" else None,
                "encoder": str(settings.encoder),
                "records": len(records),
                "scores": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {path.parent.name}/{path.name}")


if __name__ == "__main__":
    main()
