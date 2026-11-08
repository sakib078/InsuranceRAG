"""Score the agent on the golden set; rows go to evals/agent_results/, apart from single-shot."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from evals.run_eval import _retrieved, run_generation, run_retrieval, write
from evals.validate_golden import GOLDEN_PATH, load_records
from insurance_rag.config import settings
from insurance_rag.providers import judges_used

AGENT_RESULTS_DIR = Path(__file__).resolve().parent / "agent_results"


def describe() -> dict:
    """What produced an agent row - the question every later comparison asks."""
    return {
        "encoder": str(settings.encoder),
        "agent": (
            f"graph, planner {settings.decompose_model}, "
            f"context {settings.agent_context_tokens} tokens, "
            f"guardrails {'on' if settings.enable_guardrails else 'off'}"
        ),
    }


def retrieval_target(question: str, k: int) -> dict:
    """What the agent assembles before it writes, cut to `k` so the row is budget-matched."""
    from insurance_rag.agent.graph import retrieve

    return _retrieved(retrieve(question)[:k])


def generation_target(question: str) -> dict:
    """The agent end to end: one Groq call, two with the retry."""
    from insurance_rag.agent.graph import run_agent

    result = run_agent(question)
    return {
        **_retrieved(result.retrieved),
        "answer": result.text,
        "cited_locators": [c.locator for c in result.chunks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("retrieval", "generation"), default="retrieval")
    parser.add_argument("--config", required=True, help="name for this row")
    parser.add_argument("-k", type=int, default=settings.ladder_top_k,
                        help="retrieval: cut the context to this many chunks before scoring")
    parser.add_argument("--dataset", default=settings.langsmith_dataset)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--misses", action="store_true", help="list the records that scored 0")
    parser.add_argument("--guards", action="store_true",
                        help="generation: run with the policy layer and plain-language prompt on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings.enable_web_fallback = False  # a measured row never includes the web tier
    settings.enable_guardrails = args.guards  # off by default: rows stay comparable
    records = load_records(GOLDEN_PATH)
    print(f"{args.config} [agent {args.suite}]: {len(records)} records")

    misses: dict[str, list[str]] = {}
    if args.suite == "retrieval":
        summary, misses = run_retrieval(records, args.k, args.misses, retrieval_target)
        provenance = {"k": args.k, **describe()}
    else:
        metadata = {
            "config": args.config,
            **describe(),
            "generation": f"{settings.generation_provider}/{settings.generation_model}",
            "judge_chain": settings.judge_chain,
        }
        summary = run_generation(
            args.dataset, args.config, args.concurrency, generation_target, metadata
        )
        provenance = metadata | {"judges_served": judges_used()}

    record = {
        "config": args.config,
        "suite": args.suite,
        "at": datetime.now(UTC).isoformat(),
        **provenance,
        "records": len(records),
        "scores": summary,
        "misses": misses,
    }
    write(record, args.config, args.suite, AGENT_RESULTS_DIR)


if __name__ == "__main__":
    main()
