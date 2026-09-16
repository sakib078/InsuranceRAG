"""Create the pg_search BM25 index on langchain_pg_embedding. Idempotent.

Replaces the tsvector + GIN index `artifacts/migrate_fts.py` builds. That one ranked with
`ts_rank_cd`, which has no inverse document frequency, so a clause number counted no more than
the word "insurance" - measured at 1 question rescued against 7 broken. BM25 adds IDF, term
frequency saturation and length normalisation.

Only `document` is indexed. The locator is already inside it - `context_header` prepends it to
every chunk body - and BM25 weights it heavily on its own because clause numbers are rare. The
two-field `setweight` trick existed only to work around ts_rank having no IDF.

Needs ParadeDB. The stock pgvector image does not carry pg_search.
"""

from __future__ import annotations

from insurance_rag.retrieval.sparse import INDEX_NAME, psycopg_dsn

#: pg_search changed its index syntax across releases; whichever form this server takes, wins.
#: `USING bm25` first - 0.25 accepts `USING paradedb` as an access method but the query operator
#: still refuses the index, which fails at search time rather than here.
CREATE = (
    f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON langchain_pg_embedding "
    "USING bm25 (id, document) WITH (key_field='id')",
    f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON langchain_pg_embedding "
    "USING paradedb (id, document::pdb.simple('stemmer=english')) WITH (key_field=id)",
)

KEY_TYPE = """
SELECT data_type FROM information_schema.columns
WHERE table_name = 'langchain_pg_embedding' AND column_name = 'id'
"""

#: A CREATE that does not raise is not proof of a BM25 index - check the access method it built.
VERIFY = """
SELECT am.amname FROM pg_class i
JOIN pg_am am ON am.oid = i.relam
WHERE i.relname = %s
"""


def main() -> None:
    import psycopg

    with psycopg.connect(psycopg_dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('langchain_pg_embedding')")
        if cur.fetchone()[0] is None:
            raise SystemExit("langchain_pg_embedding does not exist - run scripts.index first")

        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
        conn.commit()
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'pg_search'")
        row = cur.fetchone()
        if row is None:
            raise SystemExit("pg_search did not install - is this a ParadeDB image?")
        print(f"pg_search {row[0]}")

        # key_field has to be a unique column, and pg_search has not always accepted text keys.
        # langchain's id is a varchar, so fail here with the reason rather than on a broken index.
        cur.execute(KEY_TYPE)
        print(f"key column id is {cur.fetchone()[0]}")

        errors = []
        for statement in CREATE:
            try:
                cur.execute(statement)
                conn.commit()
            except psycopg.Error as exc:
                conn.rollback()
                errors.append(str(exc).strip().splitlines()[0])
                continue
            cur.execute(VERIFY, (INDEX_NAME,))
            built = (cur.fetchone() or [None])[0]
            if built == "bm25":
                break
            # Wrong access method: drop it, or the next form cannot claim the name.
            errors.append(f"built a {built!r} index, not bm25")
            cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
            conn.commit()
        else:
            raise SystemExit("no CREATE INDEX form produced a bm25 index:\n  " + "\n  ".join(errors))
        print(f"index {INDEX_NAME} ok (bm25)")

        cur.execute("SELECT count(*) FROM langchain_pg_embedding")
        print(f"{cur.fetchone()[0]} rows indexed")

    smoke()


def smoke() -> None:
    """Prove the index answers before any eval trusts it - a citation and a plain question."""
    from insurance_rag.retrieval.sparse import search_sparse

    for query in ("s. 31", "when can an insurer refuse to pay income replacement benefits?"):
        hits = search_sparse(query, k=3)
        print(f"\n{query!r}")
        for chunk, score in hits:
            print(f"  {score:8.4f}  {chunk.locator}")
        if not hits:
            print("  no matches")


if __name__ == "__main__":
    main()
