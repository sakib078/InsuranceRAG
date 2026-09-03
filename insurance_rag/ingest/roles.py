"""Rule-based `chunk_role` and defined-term tagging. No LLM pass in this iteration."""

from __future__ import annotations

import re

from insurance_rag.schema import ChunkRole

__all__ = ["classify_role", "defined_terms_in", "harvest_terms"]

#: OAP 1 says "exclusion" twice in 68 pages - key on the phrasing, never the word.
_TRIGGERS: tuple[tuple[ChunkRole, tuple[str, ...]], ...] = (
    (ChunkRole.EXCLUSION, (
        "won't cover", "will not cover", "not covered", "we will not pay",
        "does not apply", "do not apply", "is not required to pay", "no benefit is payable",
        "not entitled to",
    )),
    (ChunkRole.CONDITION, (
        "statutory condition", "you must", "the insured shall", "only if", "provided that",
        "as a condition", "duties after", "notice of claim",
    )),
    (ChunkRole.SCHEDULE, ("the following table", "amount payable", "maximum amount", "rate of")),
)

_HEADING_ROLES: tuple[tuple[ChunkRole, tuple[str, ...]], ...] = (
    (ChunkRole.DEFINITION, ("definition", "interpretation", "meaning of", "words we use")),
    (ChunkRole.EXCLUSION, (
        "exclusion", "excluded", "won't cover", "wont cover", "not covered",
        "not payable", "what we do not", "we do not cover",
    )),
    (ChunkRole.CONDITION, ("statutory condition", "condition", "duties", "obligation")),
    (ChunkRole.SCHEDULE, ("schedule", "table", "amount", "rate", "indexation", "limit")),
    (ChunkRole.COVERAGE, ("coverage", "benefit", "we cover", "what we cover", "insured")),
)

_TERM_RE = re.compile(r"[“\"]([^”\"]{2,60})[”\"]\s+means")


def harvest_terms(texts: list[str]) -> set[str]:
    """Every term the corpus defines for itself, from its own definition sections."""
    return {m.group(1).strip().lower() for text in texts for m in _TERM_RE.finditer(text)}


def defined_terms_in(text: str, terms: set[str]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(sorted(t for t in terms if t in lowered))


def classify_role(
    text: str, ancestor_path: tuple[str, ...], locator: str, heading: str = ""
) -> ChunkRole:
    """Own heading first, then the heading trail, then trigger phrases."""
    if _TERM_RE.search(text) or "“" in locator:
        return ChunkRole.DEFINITION

    # The chunk's own heading is the strongest signal; a parent heading is weaker but real.
    for scope in (heading.lower(), " ".join(ancestor_path).lower()):
        if not scope:
            continue
        for role, needles in _HEADING_ROLES:
            if any(n in scope for n in needles):
                return role

    lowered = text.lower()
    for role, needles in _TRIGGERS:
        if any(n in lowered for n in needles):
            return role
    return ChunkRole.OTHER
