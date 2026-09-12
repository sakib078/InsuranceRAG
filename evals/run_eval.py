"""Score the golden set. Two suites, because they cost very different amounts.

    python -m evals.run_eval --suite retrieval   --config dense_clause_aware
    python -m evals.run_eval --suite generation  --config dense_clause_aware

`retrieval` makes no model calls at all and runs offline: recall@5 by hop and exclusion recall,
scored against the hand labels. `generation` answers all 56 questions and adds the LangSmith
judges in `BUILT_INS` plus citation accuracy, so it runs through `client.evaluate` and needs
both keys.

`--config` only names the row; the behaviour comes from whatever `search_corpus` currently does.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from evals.evaluators import (
    citation_accuracy_eval,
    exclusion_recall_eval,
    recall_multi,
    recall_single,
)
from evals.validate_golden import GOLDEN_PATH, load_records
from insurance_rag.config import settings
from insurance_rag.retrieval.search import search_with_scores

RESULTS_DIR = Path(__file__).resolve().parent / "results"
HISTORY_FILE = "history.jsonl"
LABEL_FIELDS = ("id", "hop", "gold_locators", "exclusion_locators", "answerable")

#: Citation accuracy needs an answer, so it only joins the generation suite.
RETRIEVAL_EVALUATORS = (recall_single, recall_multi, exclusion_recall_eval)
CUSTOM_EVALUATORS = (*RETRIEVAL_EVALUATORS, citation_accuracy_eval)
METRIC_ORDER = (
    "recall@5_single", "recall@5_multi", "exclusion_recall", "citation_accuracy",
    "correctness", "relevance", "groundedness", "retrieval_relevance",
)

#: A failed target is skipped rather than scored, so a few of them only shrink n. Past this share
#: the surviving records are no longer the golden set, and the row would be a different experiment.
ERROR_TOLERANCE = 0.10


# --- targets ----------------------------------------------------------------------------------

def retrieval_target(question: str, k: int) -> dict:
    """What the pipeline retrieves, in rank order. No generation, so no API call."""
    hits = search_with_scores(question, k=k)
    return {
        "retrieved_locators": [chunk.locator for chunk, _ in hits],
        "retrieved_text": [chunk.text for chunk, _ in hits],
    }


def generation_target(question: str, k: int | None = None) -> dict:
    """The product as shipped when `k` is None; a pinned width costs far fewer tokens.

    The ladder re-asks at k=20 after every refusal, so the 17 unanswerable records each pay for
    a second, wider prompt to confirm an answer that was already right. Pinning k skips that.
    """
    from insurance_rag.generation.chain import answer

    result = answer(question, k=k)
    return {
        "answer": result.text,
        "retrieved_locators": [c.locator for c in result.retrieved],
        "retrieved_text": [c.text for c in result.retrieved],
        "cited_locators": [c.locator for c in result.chunks],
    }


# --- scoring ----------------------------------------------------------------------------------

def labels_of(record: dict) -> dict:
    return {field: record[field] for field in LABEL_FIELDS}


def report(totals: dict[str, list[float]], misses: list | None = None) -> dict:
    """Print the row and return the scores worth recording."""
    print(f"\n{'metric':<22}{'score':>8}{'n':>6}")
    summary = {}
    for key in METRIC_ORDER:
        values = totals.get(key, [])
        if values:
            summary[key] = sum(values) / len(values)
            print(f"{key:<22}{summary[key]:>8.3f}{len(values):>6}")
    if misses:
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
            if verdict["score"] is None:  # off-slice: not scored, not averaged
                continue
            totals[verdict["key"]].append(verdict["score"])
            if verdict["score"] == 0.0:
                misses.append((verdict["key"], record["id"], record["question"]))
    return report(totals, misses if show_misses else None)


def _errored(row: dict) -> bool:
    """A row whose target raised: LangSmith records the traceback and no outputs at all."""
    run = row.get("run")
    outputs = getattr(run, "outputs", None) or {}
    return bool(getattr(run, "error", None)) or "retrieved_locators" not in outputs


def _tally(results) -> tuple[dict[str, list[float]], int, int]:
    """Collect every non-null score, and count how many targets never ran."""
    totals: dict[str, list[float]] = defaultdict(list)
    rows = errored = 0
    for row in results:
        rows += 1
        errored += _errored(row)
        for result in row["evaluation_results"]["results"]:
            if result.score is not None:
                totals[result.key].append(float(result.score))
    return totals, rows, errored


def _guard(errored: int, rows: int) -> None:
    """Refuse to record a run whose scores are a mean over survivors, not over the golden set."""
    if not errored:
        return
    share = errored / max(rows, 1)
    print(f"\n{errored}/{rows} records ({share:.0%}) errored: the target raised and returned "
          f"nothing, so those records are unscored rather than zero.")
    if share > ERROR_TOLERANCE:
        raise SystemExit(
            f"run DISCARDED - more than {ERROR_TOLERANCE:.0%} of the target calls failed, so "
            "the scores above are a mean over whatever survived, not over the golden set. "
            "Nothing was written to results/ or history.jsonl. Check the provider quota "
            "(a daily token cap fails every call) and re-run."
        )


def run_generation(dataset: str, config: str, concurrency: int, pin_k: int | None) -> dict:
    """Hosted: LangSmith runs the target over the dataset, then every wired evaluator."""
    from langsmith import Client

    from evals.langsmith import BUILT_INS

    if not settings.langsmith_api_key:
        raise SystemExit("set LANGSMITH_API_KEY in .env - get one at smith.langchain.com")

    results = Client(api_key=settings.langsmith_api_key).evaluate(
        lambda inputs: generation_target(inputs["question"], pin_k),
        data=dataset,
        evaluators=[*BUILT_INS, *CUSTOM_EVALUATORS],
        experiment_prefix=config,
        max_concurrency=concurrency,  # free tiers rate-limit long before they refuse
        metadata=describe(config, pin_k),
    )
    totals, rows, errored = _tally(results)
    summary = report(totals)
    _guard(errored, rows)
    return summary


# --- recording --------------------------------------------------------------------------------

def describe(config: str, pin_k: int | None) -> dict:
    """What produced a generation row - the question every later comparison asks."""
    return {
        "config": config,
        "encoder": str(settings.encoder),
        "retrieval": f"pinned k={pin_k}" if pin_k else "ladder",
        "generation": f"{settings.generation_provider}/{settings.generation_model}",
        "judge": f"{settings.judge_provider}/{settings.judge_model}",
    }


def write(record: dict, config: str, suite: str) -> None:
    """The per-config file is the row you cite; history is append-only evidence beside it."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{config}_{suite}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    # Append only after a newline: without this, a file left unterminated by a hand edit gets
    # the next record glued onto its last line, and that line stops being readable JSON.
    history = RESULTS_DIR / HISTORY_FILE
    unterminated = history.exists() and not history.read_text(encoding="utf-8").endswith("\n")
    with history.open("a", encoding="utf-8") as fh:
        fh.write(f"{'\n' if unterminated else ''}{json.dumps(record)}\n")

    print(f"\nwrote {path.parent.name}/{path.name}  (+ appended to {HISTORY_FILE})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("retrieval", "generation"), default="retrieval")
    parser.add_argument("--config", default="dense_clause_aware", help="name for this row")
    parser.add_argument("-k", type=int, default=settings.rerank_top_k)
    parser.add_argument("--dataset", default=settings.langsmith_dataset)
    # The judge's free tier is the bottleneck, not the pipeline.
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--pin-k", type=int, default=None,
                        help="generation: fix k and skip the retry ladder, ~40%% fewer tokens")
    parser.add_argument("--misses", action="store_true", help="list the records that scored 0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(GOLDEN_PATH)
    retrieval = args.suite == "retrieval"

    if retrieval:
        width = f"k={args.k}"
    else:
        width = f"pinned k={args.pin_k}" if args.pin_k else "the retry ladder"
    print(f"{args.config} [{args.suite}]: {len(records)} records at {width}")

    if retrieval:
        summary = run_retrieval(records, args.k, args.misses)
        provenance = {"k": args.k, "encoder": str(settings.encoder)}
    else:
        summary = run_generation(args.dataset, args.config, args.concurrency, args.pin_k)
        provenance = describe(args.config, args.pin_k)

    write(
        {
            "config": args.config,
            "suite": args.suite,
            "at": datetime.now(UTC).isoformat(),
            **provenance,
            "records": len(records),
            "scores": summary,
        },
        args.config,
        args.suite,
    )


if __name__ == "__main__":
    main()
