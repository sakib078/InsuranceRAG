"""Ingest the v1 corpus into data/chunks/{doc_id}.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict

from insurance_rag.config import DATA_DIR
from insurance_rag.corpus.manifest import Access, ManifestRow, Phase, load_manifest, select
from insurance_rag.ingest.pdf import load_pdf
from insurance_rag.ingest.roles import harvest_terms
from insurance_rag.ingest.split import to_chunks
from insurance_rag.ingest.webpages import load_web
from insurance_rag.schema import Chunk

CHUNKS_DIR = DATA_DIR / "chunks"


def load_units(row: ManifestRow, *, refresh: bool):
    return load_web(row, refresh=refresh) if row.access is Access.SCRIPT else load_pdf(row)


def write_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with (CHUNKS_DIR / f"{doc_id}.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def report(doc_id: str, chunks: list[Chunk]) -> None:
    tokens = sorted(c.token_count for c in chunks)
    roles = Counter(c.chunk_role.value for c in chunks)
    median = tokens[len(tokens) // 2] if tokens else 0
    print(f"{doc_id:34} {len(chunks):5} chunks  median {median:4} max {tokens[-1] if tokens else 0:4}")
    print(f"{'':34} {dict(roles)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", help="ingest a single document")
    parser.add_argument("--refresh", action="store_true", help="re-fetch HTML from source_url")
    args = parser.parse_args()

    rows = select(load_manifest(), phase=Phase.V1)
    if args.doc_id:
        rows = [r for r in rows if r.doc_id == args.doc_id] or parser.error(
            f"no v1 manifest row with doc_id {args.doc_id!r}"
        )

    loaded = [(row, load_units(row, refresh=args.refresh)) for row in rows]
    terms = harvest_terms([u.page_content for _, units in loaded for u in units])
    print(f"{len(terms)} defined terms harvested\n")

    for row, units in loaded:
        chunks = to_chunks(row, units, terms)
        write_chunks(row.doc_id, chunks)
        report(row.doc_id, chunks)


if __name__ == "__main__":
    main()
