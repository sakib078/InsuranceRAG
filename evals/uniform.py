"""Resolve gold locators against uniform windows and report the oracle ceilings."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache

from evals.validate_golden import GOLDEN_PATH, load_records
from insurance_rag.config import Chunking, settings

GOLD_MAP = "_gold_map.json"
#: The smallest threshold at which the oracle ceiling is 1.000 on every metric - see the report.
DEFAULT_COVERAGE = 0.5


@lru_cache(maxsize=1)
def gold_map() -> dict[str, dict[str, float]]:
    """`gold locator -> {window locator: share of the provision that window holds}`."""
    path = settings.chunks_dir / GOLD_MAP
    if not path.exists():
        raise SystemExit(f"{path} is missing - run `python -m scripts.ingest --uniform` first")
    return json.loads(path.read_text(encoding="utf-8"))


def windows_for(gold: str, coverage: float) -> set[str]:
    """Every window holding enough of this provision to count as retrieved."""
    held = gold_map().get(gold) or gold_map().get(gold.rsplit(" #", 1)[0], {})
    return {w for w, share in held.items() if share >= coverage}


def covered(gold: str, locators: list[str], coverage: float) -> bool:
    """Did the retrieved windows include one that holds enough of this provision?"""
    return bool(windows_for(gold, coverage) & set(locators))


def ceilings(coverage: float) -> dict[str, float]:
    """What a perfect retriever could score at this threshold. Below 1.0, the metric is capped."""
    records = load_records(GOLDEN_PATH)
    scores: dict[str, list[float]] = {"recall@5_single": [], "recall@5_multi": [],
                                      "exclusion_recall": []}
    for record in records:
        if not record["answerable"]:
            continue
        reachable = [bool(windows_for(g, coverage)) for g in record["gold_locators"]]
        key = "recall@5_single" if record["hop"] == "single" else "recall@5_multi"
        scores[key].append(float(all(reachable) if record["hop"] == "multi" else any(reachable)))
        if record["exclusion_locators"]:
            hits = [bool(windows_for(g, coverage)) for g in record["exclusion_locators"]]
            scores["exclusion_recall"].append(float(any(hits)))
    return {k: sum(v) / len(v) for k, v in scores.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    args = parser.parse_args()

    settings.chunking = Chunking.UNIFORM
    gold = {g for r in load_records(GOLDEN_PATH) for g in r["gold_locators"]}
    mapped = gold_map()

    missing = sorted(g for g in gold if not (g in mapped or g.rsplit(" #", 1)[0] in mapped))
    print(f"{len(gold)} distinct gold locators; {len(gold) - len(missing)} present in the map")

    print("\noracle ceilings by coverage threshold (1.000 = the metric is not capped)")
    print(f"{'theta':>7}{'single':>10}{'multi':>10}{'exclusion':>12}")
    for theta in sorted({1.0, 0.8, 0.5, args.coverage}, reverse=True):
        c = ceilings(theta)
        print(f"{theta:>7.2f}{c.get('recall@5_single', 0):>10.3f}"
              f"{c.get('recall@5_multi', 0):>10.3f}{c.get('exclusion_recall', 0):>12.3f}")

    unreachable = sorted(g for g in gold if g not in missing and not windows_for(g, args.coverage))
    if unreachable:
        print(f"\n{len(unreachable)} provisions no window holds {args.coverage:.0%} of:")
        for g in unreachable[:8]:
            held = mapped.get(g) or mapped.get(g.rsplit(" #", 1)[0], {})
            print(f"   best window holds {max(held.values(), default=0.0):.0%}  {g}")
        print("   A provision longer than the window can never be held whole - that is a real"
              " limit of fixed-size chunking, not a flaw in the threshold.")

    if missing:
        print(f"\n{len(missing)} gold locators are ABSENT from the map entirely:")
        for g in missing[:8]:
            print(f"   {g}")
        sys.exit("the uniform arm was not built from the same units as the frozen corpus")


if __name__ == "__main__":
    main()
