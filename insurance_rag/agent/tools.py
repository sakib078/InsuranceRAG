"""The agent's deterministic tools: no LLM, no network."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from insurance_rag.config import settings
from insurance_rag.corpus.manifest import Phase, load_manifest
from insurance_rag.generation.chain import within_budget
from insurance_rag.retrieval.search import Hit, fuse, search_corpus
from insurance_rag.retrieval.store import read_chunks
from insurance_rag.schema import Chunk, ChunkRole

__all__ = [
    "ordinal_index", "fetch_neighbors", "expand", "list_documents",
    "LIMITING_CUES", "WEIGHTS", "exclusion_probe", "cue_score", "exclusion_pass", "assemble",
]

LIMITING_CUES = ("shall not", "unless", "subject to", "does not apply", "except", "not entitled")

#: Exclusion channel weights; a weight of 0 drops the channel, so each one ablates.
WEIGHTS = {"probe": 1.0, "role": 0.4, "cues": 0.6}


@lru_cache(maxsize=1)
def ordinal_index() -> dict[str, list[Chunk]]:
    """doc_id -> its chunks in ordinal order, so a list index is the ordinal."""
    index: dict[str, list[Chunk]] = {}
    for path in sorted(settings.chunks_dir.glob("*.jsonl")):
        if not path.name.startswith("_"):
            for chunk in read_chunks(path):
                index.setdefault(chunk.doc_id, []).append(chunk)
    return index


def fetch_neighbors(chunk_id: str, *, window: int = settings.agent_neighbor_window) -> list[Chunk]:
    """The chunks within `window` ordinals of this one in the same document, not itself."""
    doc_id, ordinal = chunk_id.rsplit(":", 1)
    centre = int(ordinal)
    nearby = ordinal_index()[doc_id][max(centre - window, 0) : centre + window + 1]
    return [c for c in nearby if c.ordinal != centre]


def expand(chunks: Sequence[Chunk], *, window: int = settings.agent_neighbor_window) -> list[Chunk]:
    """Neighbours of every chunk, de-duplicated, leaving out the chunks passed in."""
    own = {c.chunk_id for c in chunks}
    found: dict[str, Chunk] = {}
    for chunk in chunks:
        for neighbour in fetch_neighbors(chunk.chunk_id, window=window):
            if neighbour.chunk_id not in own:
                found.setdefault(neighbour.chunk_id, neighbour)
    return list(found.values())


def list_documents() -> list[dict]:
    """One row per v1 document: id, title, citation and status."""
    return [
        {"doc_id": r.doc_id, "title": r.title, "citation": r.citation, "status": r.status.value}
        for r in load_manifest()
        if r.phase is Phase.V1
    ]


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
