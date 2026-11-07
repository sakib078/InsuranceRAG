"""The agent's deterministic tools: no LLM, no network."""

from __future__ import annotations

from insurance_rag.config import settings
from insurance_rag.generation.chain import within_budget
from insurance_rag.retrieval.search import Hit, fuse, search_corpus
from insurance_rag.schema import Chunk, ChunkRole

__all__ = [
    "LIMITING_CUES", "WEIGHTS", "exclusion_probe", "cue_score", "exclusion_pass", "assemble",
]

LIMITING_CUES = ("shall not", "unless", "subject to", "does not apply", "except", "not entitled")

#: Exclusion channel weights; a weight of 0 drops the channel, so each one ablates.
WEIGHTS = {"probe": 1.0, "role": 0.4, "cues": 0.6}


def exclusion_probe(question: str) -> str:
    """The question reworded to ask for limits and exceptions; a template, no LLM."""
    return f"exceptions, limits, and when this does not apply or is not payable: {question}"


def cue_score(chunk: Chunk) -> int:
    """How many limiting phrases the chunk's text contains."""
    text = chunk.text.lower()
    return sum(cue in text for cue in LIMITING_CUES)


def exclusion_pass(question: str, *, k: int = 10, weights: dict[str, float] = WEIGHTS) -> list[Hit]:
    """The mandatory limit-finding pass: a probe query, a role-filtered query, cue promotion."""
    depth = settings.fusion_top_k
    probe = search_corpus(exclusion_probe(question), k=depth)
    role = search_corpus(question, role_filter=[ChunkRole.EXCLUSION, ChunkRole.CONDITION], k=depth)
    candidates = {c.chunk_id: c for c in probe + role}.values()
    cued = sorted((c for c in candidates if cue_score(c)), key=cue_score, reverse=True)
    channels = {"probe": probe, "role": role, "cues": cued}
    live = {name: pool for name, pool in channels.items() if weights[name]}
    return fuse(live, k=k, weights=weights)


def assemble(
    pools: dict[str, list[Chunk]], *, limit: int = settings.agent_context_tokens
) -> list[Chunk]:
    """One RRF union of every pool, the exclusion pass's top 2 first, cut to the token limit."""
    floor = pools.get("exclusion", [])[:2]
    kept = {c.chunk_id for c in floor}
    union = fuse(pools, k=sum(len(pool) for pool in pools.values()))
    rest = [hit.chunk for hit in union if hit.chunk.chunk_id not in kept]
    return within_budget(floor + rest, limit)
