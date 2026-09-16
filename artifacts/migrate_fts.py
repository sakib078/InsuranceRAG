"""PARKED with `artifacts/sparse_tsrank.py` - that channel's migration. Not on the import path.

Add the generated tsvector column and its GIN index to langchain_pg_embedding. Idempotent.

Generated, not trigger-maintained: re-indexing a chunk rewrites its tsvector in the same write,
so the full-text index cannot drift from `document` and there is nothing to rebuild after an
ingest. Unlike an ANN vector index this one is exact, so it changes no measured number.
"""

from __future__ import annotations

from artifacts.sparse_tsrank import psycopg_dsn

#: Copied from `artifacts/sparse_tsrank.py` so the parked pair stays self-contained; the live
#: sparse module is BM25 now and no longer defines it. Keep the two spellings identical.
FTS_EXPRESSION = (
    "setweight(to_tsvector('simple', coalesce(cmetadata ->> 'locator', '')), 'A') || "
    "setweight(to_tsvector('english', document), 'B')"
)

COLUMN = f"""
ALTER TABLE langchain_pg_embedding
  ADD COLUMN IF NOT EXISTS fts tsvector GENERATED ALWAYS AS ({FTS_EXPRESSION}) STORED
"""

INDEX = """
CREATE INDEX IF NOT EXISTS langchain_pg_embedding_fts_idx
  ON langchain_pg_embedding USING GIN (fts)
"""

COUNT = "SELECT count(*), count(*) FILTER (WHERE fts IS NOT NULL) FROM langchain_pg_embedding"


def main() -> None:
    import psycopg

    with psycopg.connect(psycopg_dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('langchain_pg_embedding')")
        if cur.fetchone()[0] is None:
            raise SystemExit("langchain_pg_embedding does not exist - run scripts.index first")

        cur.execute(COLUMN)
        print("column fts          ok")
        cur.execute(INDEX)
        print("index  ...fts_idx   ok")
        conn.commit()

        cur.execute(COUNT)
        total, indexed = cur.fetchone()
    print(f"{indexed}/{total} rows carry a tsvector")


if __name__ == "__main__":
    main()
