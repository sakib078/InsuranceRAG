"""Ask the corpus one question: answer, citation blocks, licence line. Traces to traces.jsonl.

`python -m scripts.ask "is physiotherapy covered after a minor injury?"`
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from insurance_rag.config import settings
from insurance_rag.generation.chain import Answer, answer
from insurance_rag.generation.citations import licence_lines, manifest_index, render_citation


def trace(result: Answer) -> None:
    """One JSON line per query - the eval harness reads this later, not the console."""
    path = settings.trace_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "at": datetime.now(timezone.utc).isoformat(),
        "encoder": str(settings.encoder),
        "model": settings.generation_model,
        "question": result.question,
        "answer": result.text,
        "chunk_ids": [c.chunk_id for c in result.chunks],
        "locators": [c.locator for c in result.chunks],
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the indexed corpus one question.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, help="pin the window; default walks the retry ladder")
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    result = answer(args.question, k=args.k)
    print(result.text)

    if result.chunks:
        index = manifest_index()
        print("\nSources")
        for chunk in result.chunks:
            print("\n" + render_citation(chunk, index))
        for note in licence_lines(result.chunks, index):
            print(f"\n{note}")

    if not args.no_trace:
        trace(result)


if __name__ == "__main__":
    main()
