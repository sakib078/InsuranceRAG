"""The corpus manifest: one row per source document.

`data/manifest.csv` is hand-authored and committed; the documents themselves are not.
It records provenance, access route, licence, and a content checksum per document.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

from insurance_rag.config import DATA_DIR, PDF_DIR, RAW_HTML_DIR

MANIFEST_PATH = DATA_DIR / "manifest.csv"


class Access(StrEnum):
    SCRIPT = "script"
    MANUAL = "manual"


class Phase(StrEnum):
    V1 = "v1"
    V2 = "v2"
    EXCLUDED = "excluded"


class Status(StrEnum):
    """Whether the document is operative law. Revoked sources render a badge."""

    CURRENT = "current"
    REVOKED = "revoked"


class DocType(StrEnum):
    """Kind of source document. All but CASE_LAW are valid `Chunk.doc_type` values."""

    POLICY = "policy"
    ENDORSEMENT = "endorsement"
    BULLETIN = "bulletin"
    GUIDE = "guide"
    REGULATION = "regulation"
    STATUTE = "statute"
    CASE_LAW = "case-law"


#: The subset that may appear on an ingested chunk.
CHUNK_DOC_TYPES: frozenset[DocType] = frozenset(DocType) - {DocType.CASE_LAW}


@dataclass(frozen=True)
class ManifestRow:
    doc_id: str
    title: str
    citation: str
    source_url: str
    doc_type: DocType
    access: Access
    status: Status
    consolidation_date: str
    retrieval_date: str
    local_file: str
    sha256: str
    licence_note: str
    phase: Phase
    notes: str

    @property
    def raw_path(self) -> Path:
        """On-disk location. Manual rows keep the publisher's filename via `local_file`."""
        if self.access is Access.MANUAL:
            return PDF_DIR / (self.local_file or f"{self.doc_id}.pdf")
        return RAW_HTML_DIR / f"{self.doc_id}.html"


FIELDNAMES: tuple[str, ...] = tuple(f.name for f in fields(ManifestRow))


def load_manifest(path: Path = MANIFEST_PATH) -> list[ManifestRow]:
    """Read and validate the manifest. Raises on anything malformed."""
    if not path.exists():
        raise FileNotFoundError(f"No manifest at {path}")

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != FIELDNAMES:
            raise ValueError(
                f"{path} columns do not match ManifestRow.\n"
                f"  expected: {FIELDNAMES}\n"
                f"  found:    {tuple(reader.fieldnames or ())}"
            )
        raw_rows = list(reader)

    rows: list[ManifestRow] = []
    for lineno, raw in enumerate(raw_rows, start=2):
        try:
            rows.append(
                ManifestRow(
                    **{
                        **raw,
                        "access": Access(raw["access"]),
                        "phase": Phase(raw["phase"]),
                        "status": Status(raw["status"]),
                        "doc_type": DocType(raw["doc_type"]),
                    }
                )
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno} - {exc}") from exc

    seen: set[str] = set()
    for row in rows:
        if not row.doc_id:
            raise ValueError(f"{path} - a row has an empty doc_id")
        if row.doc_id in seen:
            raise ValueError(f"{path} - duplicate doc_id {row.doc_id!r}")
        seen.add(row.doc_id)
        if row.phase is not Phase.EXCLUDED and not row.source_url:
            raise ValueError(f"{path} - {row.doc_id} has no source_url")
        if row.phase is not Phase.EXCLUDED and row.doc_type not in CHUNK_DOC_TYPES:
            raise ValueError(
                f"{path} - {row.doc_id} is doc_type {row.doc_type!r}, which cannot be "
                f"ingested; mark it phase=excluded or give it an ingestable type"
            )

    return rows


def write_manifest(rows: list[ManifestRow], path: Path = MANIFEST_PATH) -> None:
    """Rewrite the manifest in place, preserving column order."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v) for k, v in asdict(row).items()})


def select(
    rows: list[ManifestRow],
    *,
    phase: Phase | None = None,
    access: Access | None = None,
    status: Status | None = None,
    doc_type: DocType | None = None,
) -> list[ManifestRow]:
    return [
        r
        for r in rows
        if (phase is None or r.phase is phase)
        and (access is None or r.access is access)
        and (status is None or r.status is status)
        and (doc_type is None or r.doc_type is doc_type)
    ]


def by_doc_id(rows: list[ManifestRow]) -> dict[str, ManifestRow]:
    """chunk.doc_id -> provenance, for the citation renderer."""
    return {r.doc_id: r for r in rows}
