"""The one retrieval seam: dense + sparse, fused by RRF.

A cross-encoder rerank sat here and was removed - it lost on every metric and cost 124s a
query. The code and its numbers are in `artifacts/rerank.py`; the pool this returns is what it
would read if it comes back.
"""

from __future__ import annotations

from dataclasses import dataclass

from insurance_rag.config import settings
from insurance_rag.retrieval.sparse import search_sparse
from insurance_rag.retrieval.store import to_chunk, vector_store
from insurance_rag.schema import Chunk, ChunkRole

__all__ = ["search_corpus", "search_hybrid", "search_with_scores", "fuse", "Hit"]


@dataclass(frozen=True, slots=True)
class Hit:
    """A fused result. `channels` is kept because "found by both" is a signal in its own right."""

    chunk: Chunk
    channels: tuple[str, ...]
    score: float  # reciprocal rank fusion


def _filter(role_filter, doc_filter) -> dict | None:
    """PGVector JSONB filter; `None` means search the whole collection."""
    clauses = []
    if role_filter:
        clauses.append({"chunk_role": {"$in": [str(r) for r in role_filter]}})
    if doc_filter:
        clauses.append({"doc_id": {"$in": list(doc_filter)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search_with_scores(
    query: str,
    *,
    role_filter: list[ChunkRole] | None = None,
    doc_filter: list[str] | None = None,
    k: int = settings.dense_top_k,
) -> list[tuple[Chunk, float]]:
    """Cosine distance, so lower is closer. Dense only - the rank diagnosis tooling reads this."""
    hits = vector_store().similarity_search_with_score(
        query, k=k, filter=_filter(role_filter, doc_filter)
    )
    return [(to_chunk(doc), score) for doc, score in hits]


def fuse(
    channels: dict[str, list[Chunk]], *, k: int, weights: dict[str, float] | None = None
) -> list[Hit]:
    """Weighted reciprocal rank fusion: position only, so two score scales never have to meet.

    `sum(w / (rrf_k + rank))` over the channels that found a chunk. Weighting is what keeps a
    weak channel advisory - see `settings.sparse_weight`.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    found: dict[str, list[str]] = {}
    chunks: dict[str, Chunk] = {}

    for channel, hits in channels.items():
        weight = weights.get(channel, 1.0)
        for rank, chunk in enumerate(hits, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weight / (settings.rrf_k + rank)
            found.setdefault(cid, []).append(channel)
            chunks.setdefault(cid, chunk)

    ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [Hit(chunks[cid], tuple(found[cid]), scores[cid]) for cid in ranked[:k]]


def search_hybrid(
    query: str,
    *,
    role_filter: list[ChunkRole] | None = None,
    doc_filter: list[str] | None = None,
    k: int = settings.fusion_top_k,
) -> list[Hit]:
    """The shipped pipeline. Each channel searches at its own depth, then RRF orders the union."""
    dense = search_with_scores(
        query, role_filter=role_filter, doc_filter=doc_filter, k=settings.dense_top_k
    )
    # Weight 0 means the channel is off, so do not pay for it - and do not require the BM25
    # extension on a database that is only ever asked for dense results.
    sparse = search_sparse(
        query, role_filter=role_filter, doc_filter=doc_filter, k=settings.sparse_top_k
    ) if settings.sparse_weight else []
    return fuse(
        {"dense": [c for c, _ in dense], "sparse": [c for c, _ in sparse]},
        k=k,
        weights={"dense": settings.dense_weight, "sparse": settings.sparse_weight},
    )


def search_corpus(
    query: str,
    *,
    role_filter: list[ChunkRole] | None = None,
    doc_filter: list[str] | None = None,
    k: int = settings.rerank_top_k,
) -> list[Chunk]:
    """Chunks, never Documents - generation and the future agent share this one type."""
    return [hit.chunk for hit in search_hybrid(
        query, role_filter=role_filter, doc_filter=doc_filter, k=k
    )]


def main() -> None:
    """`python -m insurance_rag.retrieval.search "<query>"` - inspect retrieval without an LLM."""
    import argparse

    parser = argparse.ArgumentParser(description="Search the indexed corpus.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=settings.rerank_top_k)
    parser.add_argument("--role", action="append", type=ChunkRole, choices=list(ChunkRole))
    parser.add_argument("--doc", action="append", help="restrict to these doc_ids")
    parser.add_argument("--stage", choices=("fused", "dense", "sparse"), default="fused",
                        help="inspect one channel instead of the fused result")
    parser.add_argument("--text", action="store_true", help="print the chunk body too")
    args = parser.parse_args()

    common = {"role_filter": args.role, "doc_filter": args.doc, "k": args.k}
    if args.stage == "fused":
        rows = [(h.chunk, h.score, "+".join(h.channels)) for h in search_hybrid(args.query, **common)]
    elif args.stage == "dense":
        rows = [(c, s, "dense") for c, s in search_with_scores(args.query, **common)]
    else:
        rows = [(c, s, "sparse") for c, s in search_sparse(args.query, **common)]

    for chunk, score, channel in rows:
        print(f"{score:.4f}  {channel:<13} {chunk.chunk_role:<10} {chunk.locator}")
        if args.text:
            print(f"{chunk.text}\n")
    if not rows:
        print("no matches")


if __name__ == "__main__":
    main()
