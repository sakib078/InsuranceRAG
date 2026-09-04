"""Index data/chunks/*.jsonl into pgvector. Idempotent - reruns upsert by chunk_id.

Two modes. Without `--embeddings` the bi-encoder runs here, which is hours on CPU. With
`--embeddings DIR` it reads the vectors `notebooks/embed_colab.ipynb` produced on a GPU and
inserts them directly, so this becomes plain Postgres writes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from insurance_rag.config import DATA_DIR, settings
from insurance_rag.retrieval.store import (
    add_chunks,
    connection_string,
    read_chunks,
    to_document,
    vector_store,
)
from insurance_rag.schema import Chunk

CHUNKS_DIR = DATA_DIR / "chunks"


def add_precomputed(chunks: list[Chunk], path: Path) -> None:
    """Insert GPU-computed vectors, keyed to chunks by id so a stale .npz cannot slip through."""
    import numpy as np

    with np.load(path) as data:
        ids, vectors = list(data["ids"]), data["vectors"]
    if ids != [c.chunk_id for c in chunks]:
        raise SystemExit(f"{path.name}: ids do not match {path.stem}.jsonl - re-run the notebook")

    documents = [to_document(c) for c in chunks]
    vector_store().add_embeddings(
        texts=[d.page_content for d in documents],
        embeddings=vectors.tolist(),
        metadatas=[d.metadata for d in documents],
        ids=ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", help="index one document instead of the whole corpus")
    parser.add_argument("--embeddings", type=Path, help="directory of {doc_id}.npz from Colab")
    args = parser.parse_args()

    pattern = f"{args.doc_id}.jsonl" if args.doc_id else "*.jsonl"
    paths = sorted(p for p in CHUNKS_DIR.glob(pattern) if not p.name.startswith("_"))
    if not paths:
        raise SystemExit(f"no chunk files matching {pattern} in {CHUNKS_DIR}")

    print(f"collection chunks_{settings.encoder} -> {connection_string()}")
    total = 0
    for path in paths:
        chunks = read_chunks(path)
        if args.embeddings:
            npz = args.embeddings / f"{path.stem}.npz"
            if not npz.exists():
                raise SystemExit(f"missing {npz} - embed it in the notebook first")
            add_precomputed(chunks, npz)
        else:
            add_chunks(chunks)
        total += len(chunks)
        print(f"{path.stem:<40} {len(chunks):>5} chunks")
    print(f"{'TOTAL':<40} {total:>5} chunks")


if __name__ == "__main__":
    main()
