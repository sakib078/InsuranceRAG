"""Stage B: enforce the token budget on structural units, then build `Chunk` objects."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from insurance_rag.config import settings
from insurance_rag.corpus.manifest import ManifestRow
from insurance_rag.ingest.roles import classify_role, defined_terms_in
from insurance_rag.schema import Chunk, make_chunk_id

__all__ = ["reference_tokenizer", "token_count", "to_chunks"]

#: Qwen3's tokenizer is the reference for `token_count` - see docs/plan.md, Deviation 5.
REFERENCE_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
OVERLAP_TOKENS = 120  # ~15% of the 800-token ceiling
HEADER_ALLOWANCE = 64  # room the context header takes inside the ceiling


@lru_cache(maxsize=1)
def reference_tokenizer():
    """The one tokenizer both encoders' chunks are measured against."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)


def token_count(text: str) -> int:
    return len(reference_tokenizer().encode(text, add_special_tokens=False))


@lru_cache(maxsize=1)
def _size_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        reference_tokenizer(),
        chunk_size=settings.max_chunk_tokens - HEADER_ALLOWANCE,
        chunk_overlap=OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )


def _body_budget() -> int:
    return settings.max_chunk_tokens - HEADER_ALLOWANCE


def _pieces(unit: Document) -> list[str]:
    """Split only what exceeds the ceiling; tables are never split."""
    text = unit.page_content
    if unit.metadata.get("is_table") or token_count(text) <= _body_budget():
        return [text]
    return _size_splitter().split_text(text)


def context_header(row: ManifestRow, locator: str, ancestors: tuple[str, ...]) -> str:
    """Provenance and heading trail, embedded with the chunk so short provisions have context."""
    trail = " > ".join(ancestors)
    return f"{row.title} ({row.citation}){' — ' + trail if trail else ''}\n{locator}:"


def to_chunks(row: ManifestRow, units: list[Document], terms: set[str]) -> list[Chunk]:
    """Turn structural units into validated `Chunk` objects, numbered per document."""
    chunks: list[Chunk] = []
    for unit in units:
        base = f"{row.citation} s. {unit.metadata['locator_path']}"
        ancestors = tuple(unit.metadata.get("ancestor_path") or ())
        pieces = _pieces(unit)
        for index, text in enumerate(pieces, start=1):
            text = text.strip()
            if not text:
                continue
            ordinal = len(chunks)
            locator = base if len(pieces) == 1 else f"{base} #{index}"
            text = f"{context_header(row, locator, ancestors)}\n{text}"
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(row.doc_id, ordinal),
                    doc_id=row.doc_id,
                    doc_type=row.doc_type,
                    chunk_role=classify_role(text, ancestors, locator),
                    locator=locator,
                    ancestor_path=ancestors,
                    ordinal=ordinal,
                    text=text,
                    token_count=token_count(text),
                    defined_terms=defined_terms_in(text, terms),
                    page=unit.metadata.get("page"),
                )
            )
    return chunks
