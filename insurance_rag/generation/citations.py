"""Renders a chunk's provenance block. The locator is already human-verifiable; this adds the rest."""

from __future__ import annotations

from insurance_rag.corpus.manifest import ManifestRow, Status, by_doc_id, load_manifest
from insurance_rag.schema import Chunk

__all__ = ["manifest_index", "render_citation", "licence_lines"]


def manifest_index() -> dict[str, ManifestRow]:
    """doc_id -> provenance, loaded once by the caller and passed down."""
    return by_doc_id(load_manifest())


def _short_url(url: str) -> str:
    return url.split("://", 1)[-1].rstrip("/")


def render_citation(chunk: Chunk, index: dict[str, ManifestRow]) -> str:
    """The fixed three-line block from `docs/plan.md`; revoked and unofficial sources say so."""
    row = index.get(chunk.doc_id)
    if row is None:
        return f"{chunk.locator} - unknown document {chunk.doc_id}"

    head = f"{chunk.locator} - {row.title}"
    if row.status is Status.REVOKED:
        head += "  [REVOKED]"

    date = row.consolidation_date or row.retrieval_date
    meta = " . ".join(part for part in (f"Consolidated {date}" if date else "", _short_url(row.source_url)) if part)

    lines = [head, meta, "Not an official version."]
    if not chunk.is_official:
        lines.append("Text alternative, not part of the law.")
    return "\n".join(lines)


def licence_lines(chunks: list[Chunk], index: dict[str, ManifestRow]) -> list[str]:
    """One licence note per distinct source behind an answer."""
    notes = {index[c.doc_id].licence_note for c in chunks if c.doc_id in index}
    return sorted(n for n in notes if n)
