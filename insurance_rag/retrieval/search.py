"""The one retrieval seam. Dense-only today; sparse, RRF and rerank land here later."""

from __future__ import annotations

from insurance_rag.config import settings
from insurance_rag.retrieval.store import to_chunk, vector_store
from insurance_rag.schema import Chunk, ChunkRole

__all__ = ["search_corpus", "search_with_scores"]


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
    """Cosine distance, so lower is closer. Scores are for eval, not for the answer."""
    hits = vector_store().similarity_search_with_score(
        query, k=k, filter=_filter(role_filter, doc_filter)
    )
    return [(to_chunk(doc), score) for doc, score in hits]


def search_corpus(
    query: str,
    *,
    role_filter: list[ChunkRole] | None = None,
    doc_filter: list[str] | None = None,
    k: int = settings.rerank_top_k,
) -> list[Chunk]:
    """Chunks, never Documents - generation and the future agent share this one type."""
    return [chunk for chunk, _ in search_with_scores(
        query, role_filter=role_filter, doc_filter=doc_filter, k=k
    )]


def main() -> None:
    """`python -m insurance_rag.retrieval.search "<query>"` - inspect retrieval without an LLM."""
    import argparse

    parser = argparse.ArgumentParser(description="Dense search over the indexed corpus.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=settings.rerank_top_k)
    parser.add_argument("--role", action="append", type=ChunkRole, choices=list(ChunkRole))
    parser.add_argument("--doc", action="append", help="restrict to these doc_ids")
    parser.add_argument("--text", action="store_true", help="print the chunk body too")
    args = parser.parse_args()

    hits = search_with_scores(args.query, role_filter=args.role, doc_filter=args.doc, k=args.k)
    for chunk, score in hits:
        print(f"{score:.3f}  {chunk.chunk_role:<10} {chunk.locator}")
        if args.text:
            print(f"{chunk.text}\n")
    if not hits:
        print("no matches")


if __name__ == "__main__":
    main()
