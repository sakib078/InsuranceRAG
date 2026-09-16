"""PARKED - the `ts_rank_cd` sparse channel, measured and replaced by BM25. Not on the import path.

Built for Phase 4 on stock Postgres full-text search, then replaced because `ts_rank_cd` has no
inverse document frequency. It scores term density and proximity only, so in a corpus where
nearly every chunk contains "insurance", "benefit" and "insurer", those words counted as much as
a clause number. Measured over 62 golden records, dense-only against dense + this channel at
RRF weight 0.4:

    rescued (1)   g-061  "O. Reg. 34/10 s. 55(1) - barred from the LAT"
    broke   (7)   g-006 g-008 (single), g-021 (multi), g-006 g-021 g-022 g-024 (exclusion)

Every casualty is a plain-language question carrying common statutory vocabulary. The single
rescue is a rare-token match - exactly the split IDF exists to fix, which is the case for BM25.

Kept because the negative result is worth keeping, and because it is the only sparse channel
that runs on a stock `pgvector/pgvector:pg16` image - BM25 needs ParadeDB. To revive: move back
to `insurance_rag/retrieval/`, and re-run `artifacts/migrate_fts.py` for the tsvector + GIN index.
"""

from __future__ import annotations

from insurance_rag.config import settings
from insurance_rag.retrieval.store import to_chunk
from insurance_rag.schema import Chunk, ChunkRole

__all__ = ["search_sparse", "psycopg_dsn", "FTS_EXPRESSION", "RANK_WEIGHTS"]

#: The indexed expression, shared with artifacts/migrate_fts.py so index and query cannot drift.
#: Locator unstemmed at weight A - a citation query aims at nothing else; body stemmed at B so
#: natural-language questions still match. The two-argument `to_tsvector` is IMMUTABLE and the
#: one-argument form is not, which is what a generated column requires.
FTS_EXPRESSION = (
    "setweight(to_tsvector('simple', coalesce(cmetadata ->> 'locator', '')), 'A') || "
    "setweight(to_tsvector('english', document), 'B')"
)

#: ts_rank_cd weights, ordered {D, C, B, A}: a locator hit outranks a body hit four to one.
RANK_WEIGHTS = [0.1, 0.2, 0.4, 1.0]

# plainto_tsquery conjoins its terms, so a six-word question matches nothing. Ranking wants
# "how many of these terms, how densely", so the AND is rewritten to OR before it is cast back.
_TSQUERY = (
    "replace(plainto_tsquery('simple', %(query)s)::text, '&', '|')::tsquery || "
    "replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery"
)

_SQL = """
WITH q AS (SELECT {tsquery} AS tsq)
SELECT e.document, e.cmetadata, ts_rank_cd(%(weights)s::float4[], e.fts, q.tsq) AS score
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
CROSS JOIN q
WHERE c.name = %(collection)s AND e.fts @@ q.tsq{filters}
ORDER BY score DESC, e.id
LIMIT %(k)s
"""


def psycopg_dsn() -> str:
    """The DSN without SQLAlchemy's driver suffix, which psycopg itself will not parse."""
    return settings.postgres_dsn.replace("postgresql+psycopg://", "postgresql://", 1)


def search_sparse(
    query: str,
    *,
    role_filter: list[ChunkRole] | None = None,
    doc_filter: list[str] | None = None,
    k: int = settings.sparse_top_k,
) -> list[tuple[Chunk, float]]:
    """`ts_rank_cd` over the GIN index; higher is better, unlike the dense channel's distance."""
    import psycopg

    params: dict = {
        "query": query,
        "weights": RANK_WEIGHTS,
        "collection": settings.collection_name,
        "k": k,
    }

    # Every collection lives in one heap keyed only by collection_id; a query that forgets it
    # reads another arm's rows. Filters mirror the dense side's JSONB predicates.
    filters = ""
    if role_filter:
        filters += " AND e.cmetadata ->> 'chunk_role' = ANY(%(roles)s)"
        params["roles"] = [str(r) for r in role_filter]
    if doc_filter:
        filters += " AND e.cmetadata ->> 'doc_id' = ANY(%(docs)s)"
        params["docs"] = list(doc_filter)

    sql = _SQL.format(tsquery=_TSQUERY, filters=filters)
    with psycopg.connect(psycopg_dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    from langchain_core.documents import Document

    return [
        (to_chunk(Document(page_content=text, metadata=metadata)), float(score))
        for text, metadata, score in rows
    ]
