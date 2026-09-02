"""The chunk contract.

Every ingested document, HTML or PDF, produces `Chunk` objects and nothing else.
Written before any ingestion code so both dialects target the same shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from insurance_rag.corpus.manifest import DocType

__all__ = ["ChunkRole", "DocType", "Chunk", "make_chunk_id", "LOCATOR_RE"]


class ChunkRole(StrEnum):
    """What a chunk does in the policy. `search_corpus(role_filter=...)` runs on this."""

    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    DEFINITION = "definition"
    CONDITION = "condition"
    SCHEDULE = "schedule"
    OTHER = "other"


#: A locator is "<document short cite> s. <clause path>", human-verifiable against the source.
LOCATOR_RE = re.compile(r"^.+ s\. .+$")


def make_chunk_id(doc_id: str, ordinal: int) -> str:
    return f"{doc_id}:{ordinal:04d}"


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: DocType
    chunk_role: ChunkRole
    locator: str
    ancestor_path: tuple[str, ...]
    ordinal: int
    text: str
    token_count: int
    defined_terms: tuple[str, ...] = ()
    page: int | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"{self.chunk_id}: empty text")
        if not LOCATOR_RE.match(self.locator):
            raise ValueError(f"{self.chunk_id}: locator {self.locator!r} is not '<cite> s. <path>'")
        if self.token_count <= 0:
            raise ValueError(f"{self.chunk_id}: token_count must be positive")
        if self.chunk_id != make_chunk_id(self.doc_id, self.ordinal):
            raise ValueError(f"{self.chunk_id}: does not match doc_id/ordinal")
