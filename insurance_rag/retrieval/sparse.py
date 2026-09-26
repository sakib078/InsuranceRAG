"""BM25 lexical retrieval, in the table pgvector already owns - see docs/plan.md, Phase 4.

Dense search discards surface form by design, which is what lets a claimant's wording reach the
regulation's. The same property destroys `s. 31`: a citation is an identifier, not a meaning.
This is the other channel.

BM25 rather than `ts_rank_cd` because ranking here turns on *rare* terms. A clause number appears
in one chunk; "insurance" appears in nearly all of them. `ts_rank_cd` cannot tell those apart -
it has no inverse document frequency - and the measured cost was 7 broken questions against 1
rescued. See `artifacts/sparse_tsrank.py` for that arm and its numbers.

Needs ParadeDB's `pg_search`; the stock pgvector image does not carry it.
"""

from __future__ import annotations

import re
from functools import lru_cache

from insurance_rag.config import settings
from insurance_rag.retrieval.store import to_chunk
from insurance_rag.schema import Chunk, ChunkRole

__all__ = ["search_sparse", "psycopg_dsn", "dialect", "INDEX_NAME"]

INDEX_NAME = "langchain_pg_embedding_bm25_idx"

#: pg_search renamed its match operator and score function; schema name -> (operator, score fn).
_DIALECTS = {
    "pdb": ("|||", "pdb.score"),  # newer releases
    "paradedb": ("@@@", "paradedb.score"),  # earlier releases
}

#: Characters the `@@@` parser treats as syntax; lowercasing also defuses AND/OR/NOT.
_TANTIVY_SPECIAL = re.compile(r"[+\-&|!(){}\[\]^\"'~*?:\\/<>=]")

_SQL = """
SELECT e.document, e.cmetadata, {score}(e.id) AS score
FROM langchain_pg_embedding e
WHERE e.document {op} %(query)s
  AND e.collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = %(collection)s)
  {filters}
ORDER BY score DESC, e.id
LIMIT %(k)s
"""


def psycopg_dsn() -> str:
    """The DSN without SQLAlchemy's driver suffix, which psycopg itself will not parse."""
    return settings.postgres_dsn.replace("postgresql+psycopg://", "postgresql://", 1)


@lru_cache(maxsize=1)
def dialect() -> tuple[str, str]:
    """Whichever pg_search spelling this server has, resolved once per process."""
    import psycopg

    with psycopg.connect(psycopg_dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT nspname FROM pg_namespace WHERE nspname = ANY(%s)", (list(_DIALECTS),))
        found = {row[0] for row in cur.fetchall()}

    for schema, spelling in _DIALECTS.items():
        if schema in found:
            return spelling
    raise SystemExit(
        "pg_search is not installed on this database - sparse retrieval needs ParadeDB. "
        "Run scripts/migrate_bm25.py, or point IRAG_POSTGRES_DSN at the ParadeDB container."
    )


def search_sparse(
    query: str,
    *,
    role_filter: list[ChunkRole] | None = None,
    doc_filter: list[str] | None = None,
    k: int = settings.sparse_top_k,
) -> list[tuple[Chunk, float]]:
    """BM25 relevance; higher is better, unlike the dense channel's cosine distance."""
    import psycopg
    from langchain_core.documents import Document

    op, score = dialect()
    if op == "@@@":  # Tantivy query syntax; the tokenizer drops this punctuation anyway
        query = _TANTIVY_SPECIAL.sub(" ", query).lower()
    params: dict = {"query": query,"collection": settings.collection_name, "k": k}

    # Every collection lives in one heap keyed only by collection_id; a query that forgets it
    # reads another arm's rows. Filters mirror the dense side's JSONB predicates.
    filters = ""
    if role_filter:
        filters += " AND e.cmetadata ->> 'chunk_role' = ANY(%(roles)s)"
        params["roles"] = [str(r) for r in role_filter]
    if doc_filter:
        filters += " AND e.cmetadata ->> 'doc_id' = ANY(%(docs)s)"
        params["docs"] = list(doc_filter)

    sql = _SQL.format(op=op, score=score, filters=filters)
    with psycopg.connect(psycopg_dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        (to_chunk(Document(page_content=text, metadata=metadata)), float(value))
        for text, metadata, value in rows
    ]
