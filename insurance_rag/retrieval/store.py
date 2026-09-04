"""PGVector store: the collection is keyed by encoder so both bake-off indexes coexist."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from insurance_rag.config import settings
from insurance_rag.schema import Chunk

__all__ = ["embeddings", "vector_store", "add_chunks", "read_chunks", "to_chunk"]

#: Tuple fields survive JSONB as lists and have to be cast back on the way out.
_TUPLE_FIELDS = ("ancestor_path", "defined_terms")

def connection_string() -> str:
    """`settings.postgres_dsn` with the psycopg3 driver spelled out."""
    dsn = settings.postgres_dsn
    
    return dsn


@lru_cache(maxsize=1)
def embeddings():
    """The bi-encoder under test; queries get Qwen3's query prompt, documents do not."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.bi_encoder_model,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt_name": "query"},
    )


@lru_cache(maxsize=1)
def vector_store():
    """One collection per encoder over byte-identical chunks - see docs/plan.md, Deviation 8."""
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=embeddings(),
        collection_name=f"chunks_{settings.encoder}",
        connection=connection_string(),
        use_jsonb=True,
    )


def to_document(chunk: Chunk) -> Document:
    """Every `Chunk` field except `text` rides along as metadata so search can rebuild it."""
    metadata = asdict(chunk)
    text = metadata.pop("text")
    for field in _TUPLE_FIELDS:
        metadata[field] = list(metadata[field])
    return Document(id=chunk.chunk_id, page_content=text, metadata=metadata)


def to_chunk(doc: Document) -> Chunk:
    """Inverse of `to_document`; re-validates through `Chunk.__post_init__`."""
    metadata = {k: v for k, v in doc.metadata.items() if k in Chunk.__slots__}
    for field in _TUPLE_FIELDS:
        metadata[field] = tuple(metadata.get(field) or ())
    return Chunk(text=doc.page_content, **metadata)


def read_chunks(path: Path) -> list[Chunk]:
    """Load one `data/chunks/{doc_id}.jsonl` back into validated `Chunk` objects."""
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for field in _TUPLE_FIELDS:
            record[field] = tuple(record.get(field) or ())
        chunks.append(Chunk(**record))
    return chunks


def add_chunks(chunks: list[Chunk], *, batch_size: int = 64) -> int:
    """Upsert by `chunk_id`, so re-indexing replaces rows instead of duplicating them."""
    store = vector_store()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store.add_documents([to_document(c) for c in batch], ids=[c.chunk_id for c in batch])
    return len(chunks)
