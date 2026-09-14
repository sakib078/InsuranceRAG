"""Add the generated tsvector column and its GIN index to langchain_pg_embedding. Idempotent.

Generated, not trigger-maintained: re-indexing a chunk rewrites its tsvector in the same write,
so the full-text index cannot drift from `document` and there is nothing to rebuild after an
ingest. Unlike an ANN vector index this one is exact, so it changes no measured number.
"""

from __future__ import annotations

from insurance_rag.retrieval.sparse import FTS_EXPRESSION, psycopg_dsn

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
