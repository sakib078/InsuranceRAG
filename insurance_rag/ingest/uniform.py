"""The ablation arm: fixed token windows that ignore every boundary the document declares.

Deliberately naive. It exists so the clause-aware chunker has something to be measured against,
and it answers the obvious objection to `recall@5_multi 0.050` - that five provision-sized
chunks rarely hold three whole provisions, while five fat windows might.

Gold locators cannot resolve against a window, so this also emits a coverage map: for each
provision, how much of it each window holds. Computed from token spans rather than substring
search, because a phrase can appear twice and a fraction cannot be guessed.
"""

from __future__ import annotations

from langchain_core.documents import Document

from insurance_rag.config import settings
from insurance_rag.corpus.manifest import ManifestRow
from insurance_rag.ingest.split import context_header, reference_tokenizer
from insurance_rag.schema import Chunk, ChunkRole, make_chunk_id

__all__ = ["to_uniform_chunks", "SEPARATOR", "ID_PREFIX"]

#: Joins one provision to the next in the flattened stream, exactly as a naive reader would see it.
SEPARATOR = "\n\n"

#: `langchain_pg_embedding` has PRIMARY KEY (id), not (id, collection_id), so an id shared with
#: the clause-aware corpus does not land in this collection - it UPDATES that one's row instead.
ID_PREFIX = "u"


def _spans(units: list[Document]) -> tuple[list[int], list[tuple[str, int, int]]]:
    """Flatten the document to token ids, remembering where each provision starts and ends."""
    tokenizer = reference_tokenizer()
    stream: list[int] = []
    spans: list[tuple[str, int, int]] = []
    separator = tokenizer.encode(SEPARATOR, add_special_tokens=False)

    for unit in units:
        ids = tokenizer.encode(unit.page_content, add_special_tokens=False)
        start = len(stream)
        stream.extend(ids)
        spans.append((unit.metadata["locator_path"], start, len(stream)))
        stream.extend(separator)
    return stream, spans


def _coverage(spans, window_start: int, window_end: int) -> dict[str, float]:
    """Share of each provision that falls inside this window; 0 is dropped, not recorded."""
    covered = {}
    for path, start, end in spans:
        overlap = min(end, window_end) - max(start, window_start)
        if overlap > 0:
            covered[path] = overlap / (end - start)
    return covered


def to_uniform_chunks(
    row: ManifestRow, units: list[Document]
) -> tuple[list[Chunk], dict[str, dict[str, float]]]:
    """Fixed windows over the flattened document, plus `locator -> {window: coverage}`."""
    tokenizer = reference_tokenizer()
    size = settings.uniform_chunk_tokens
    stride = size - settings.uniform_overlap_tokens
    stream, spans = _spans(units)

    chunks: list[Chunk] = []
    gold: dict[str, dict[str, float]] = {}

    for ordinal, start in enumerate(range(0, max(len(stream), 1), stride)):
        window = stream[start : start + size]
        text = tokenizer.decode(window).strip()
        if not text:
            continue

        # No clause path exists here - that is the point - so the window index is the locator.
        locator = f"{row.citation} s. window {ordinal + 1}"
        # Document identity comes from the manifest, not from parsing, so a naive chunker has
        # it too. Holding provenance constant keeps structure the only variable under test.
        body = f"{context_header(row, locator, ())}\n{text}"

        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(row.doc_id, len(chunks), prefix=ID_PREFIX),
                doc_id=row.doc_id,
                doc_type=row.doc_type,
                chunk_role=ChunkRole.OTHER,  # structure-blind by construction
                locator=locator,
                ancestor_path=(),
                ordinal=len(chunks),
                text=body,
                token_count=len(window) + len(tokenizer.encode(
                    context_header(row, locator, ()), add_special_tokens=False)),
                page=None,
            )
        )
        for path, share in _coverage(spans, start, start + size).items():
            gold.setdefault(f"{row.citation} s. {path}", {})[locator] = share

        if start + size >= len(stream):
            break

    return chunks, gold
