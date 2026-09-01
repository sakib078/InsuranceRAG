"""The corpus manifest: one row per source document.

`data/manifest.csv` is the single record of what the corpus is, where each document
came from, whether a script can fetch it, and what licence it carries. It is
hand-authored and committed; the documents themselves are not (see `.gitignore`).

Three columns make it the scale-up ledger rather than a snapshot:

  ACCESS  script | manual  - FSRA returns 403 to scripts, so part of the corpus is
          reproducible by *verification* (a recorded SHA-256) rather than by script.
  PHASE   v1 | v2 | excluded  - v1 is the scriptable layer the first pipeline runs on.
          Promoting a document is a flag flip, not a rewrite. `excluded` records a
          decision not to ingest something, so the reasoning is not lost.
  SHA256  Blank until the first fetch, then fixed. e-Laws reconsolidates without
          notice; a mismatch means the corpus moved under the eval numbers.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

from insurance_rag.config import DATA_DIR, RAW_DIR

MANIFEST_PATH = DATA_DIR / "manifest.csv"


class Access(StrEnum):
    SCRIPT = "script"
    MANUAL = "manual"


class Phase(StrEnum):
    V1 = "v1"
    V2 = "v2"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ManifestRow:
    doc_id: str
    title: str
    citation: str
    source_url: str
    doc_type: str
    access: Access
    consolidation_date: str
    sha256: str
    licence_note: str
    phase: Phase
    notes: str

    @property
    def raw_path(self) -> Path:
        """Where this document lives once fetched or downloaded by hand."""
        suffix = Path(self.source_url).suffix.lower()
        if suffix not in {".html", ".htm", ".pdf", ".txt"}:
            # e-Laws URLs carry no extension and serve HTML; FSRA rows point at a
            # landing page, and the PDFs behind it are placed by hand.
            suffix = ".pdf" if self.access is Access.MANUAL else ".html"
        return RAW_DIR / f"{self.doc_id}{suffix}"


FIELDNAMES: tuple[str, ...] = tuple(f.name for f in fields(ManifestRow))


def load_manifest(path: Path = MANIFEST_PATH) -> list[ManifestRow]:
    """Read and validate the manifest. Raises on anything malformed - a corpus
    definition that is quietly wrong is worse than one that fails loudly."""
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
                    **{**raw, "access": Access(raw["access"]), "phase": Phase(raw["phase"])}
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

    return rows


def write_manifest(rows: list[ManifestRow], path: Path = MANIFEST_PATH) -> None:
    """Rewrite the manifest in place, preserving column order. Used only by
    `fetch_corpus.py --update-checksums`."""
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
) -> list[ManifestRow]:
    return [
        r
        for r in rows
        if (phase is None or r.phase is phase) and (access is None or r.access is access)
    ]
