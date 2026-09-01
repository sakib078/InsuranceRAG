#!/usr/bin/env python
"""Fetch the scriptable half of the corpus from `data/manifest.csv`.

    python scripts/fetch_corpus.py --dry-run           # show the plan, touch nothing
    python scripts/fetch_corpus.py                     # fetch phase v1, verify checksums
    python scripts/fetch_corpus.py --update-checksums  # fetch and record new checksums

`access=manual` rows are listed, never fetched. Raw bytes are written untouched.
Exit code is non-zero on any failure or checksum mismatch, so this is safe in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import replace
from datetime import date

import httpx

from insurance_rag.config import RAW_HTML_DIR
from insurance_rag.corpus.manifest import (
    Access,
    ManifestRow,
    Phase,
    load_manifest,
    select,
    write_manifest,
)

# The "curl" token is required: e-Laws serves a JavaScript shell to any other UA.
USER_AGENT = (
    "InsuranceRAG/0.1 (research corpus fetch; https://github.com/sakib078/InsuranceRAG) "
    "curl/8.4.0"
)
TIMEOUT = httpx.Timeout(30.0)
RETRIES = 3
BACKOFF_SECONDS = 2.0
POLITE_DELAY_SECONDS = 1.0

MIN_DOCUMENT_BYTES = 15_000
SHELL_MARKER = "e-Laws needs JavaScript"
STRUCTURE_MARKERS = ('class="section-e"', 'class="section"', "ConsolidationPeriod")


class NotADocument(Exception):
    """A 200 response that is not the document we asked for."""


def validate_document(row: ManifestRow, content: bytes) -> None:
    """Reject a 200 response that is a JavaScript shell rather than a document."""
    if len(content) < MIN_DOCUMENT_BYTES:
        raise NotADocument(
            f"{row.doc_id}: {len(content):,} bytes is below the {MIN_DOCUMENT_BYTES:,} floor"
        )
    text = content.decode("utf-8", errors="replace")
    if SHELL_MARKER in text:
        raise NotADocument(
            f"{row.doc_id}: got the e-Laws JavaScript shell, not the document. "
            f"The User-Agent no longer selects the pre-render - see USER_AGENT above."
        )
    if not any(marker in text for marker in STRUCTURE_MARKERS):
        raise NotADocument(
            f"{row.doc_id}: no e-Laws structural markup found "
            f"(looked for {', '.join(STRUCTURE_MARKERS)})"
        )


_SCRIPT_RE = re.compile(rb"<script(?:\s[^>]*)?>.*?</script\s*>", re.DOTALL | re.IGNORECASE)


def raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_sha256(data: bytes) -> str:
    """Stable document identity: e-Laws injects a per-request token inside <script>."""
    return hashlib.sha256(_SCRIPT_RE.sub(b"", data)).hexdigest()


def fetch(client: httpx.Client, row: ManifestRow) -> bytes:
    """GET with a small backoff. Raises the last error if every attempt fails."""
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = client.get(row.source_url)
            response.raise_for_status()
            return response.content
        except (httpx.HTTPError, httpx.StreamError) as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"{row.doc_id}: {last}") from last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=Phase,
        choices=list(Phase),
        default=Phase.V1,
        help="which corpus phase to fetch (default: v1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan without downloading"
    )
    parser.add_argument(
        "--update-checksums",
        action="store_true",
        help="write computed SHA-256 and retrieval_date back into the manifest",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the local copy verifies"
    )
    args = parser.parse_args()

    rows = load_manifest()
    scriptable = select(rows, phase=args.phase, access=Access.SCRIPT)
    manual = select(rows, phase=args.phase, access=Access.MANUAL)

    print(
        f"manifest: {len(rows)} documents, phase {args.phase} -> "
        f"{len(scriptable)} scriptable, {len(manual)} manual\n"
    )

    if args.dry_run:
        for row in scriptable:
            state = "cached" if row.raw_path.exists() else "to fetch"
            print(f"  [{state:>8}] {row.doc_id:<32} {row.source_url}")
        for row in manual:
            print(f"  [  manual] {row.doc_id:<32} place at {row.raw_path}")
        print("\ndry run - nothing downloaded")
        return 0

    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    updated: dict[str, str] = {}
    failures: list[str] = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True
    ) as client:
        for row in scriptable:
            if not args.force and row.sha256 and row.raw_path.exists():
                if content_sha256(row.raw_path.read_bytes()) == row.sha256:
                    print(f"  ok       {row.doc_id:<32} cached, checksum verified")
                    continue

            try:
                content = fetch(client, row)
                validate_document(row, content)
            except (RuntimeError, NotADocument) as exc:
                print(f"  FAILED   {exc}")
                failures.append(row.doc_id)
                continue

            digest = content_sha256(content)
            row.raw_path.write_bytes(content)
            size_kb = len(content) / 1024

            if not row.sha256:
                print(f"  new      {row.doc_id:<32} {size_kb:8.1f} KB  {digest[:16]}...")
                updated[row.doc_id] = digest
            elif digest == row.sha256:
                print(f"  ok       {row.doc_id:<32} {size_kb:8.1f} KB  checksum verified")
            else:
                print(
                    f"  CHANGED  {row.doc_id:<32} {size_kb:8.1f} KB\n"
                    f"           manifest {row.sha256[:16]}...  fetched {digest[:16]}...\n"
                    f"           source reconsolidated since the manifest was written - "
                    f"re-run the evals before trusting the old numbers"
                )
                if args.update_checksums:
                    updated[row.doc_id] = digest
                else:
                    failures.append(row.doc_id)

            time.sleep(POLITE_DELAY_SECONDS)

    if updated and args.update_checksums:
        today = date.today().isoformat()
        write_manifest(
            [
                replace(r, sha256=updated[r.doc_id], retrieval_date=today)
                if r.doc_id in updated
                else r
                for r in rows
            ]
        )
        print(f"\nmanifest updated: {len(updated)} checksum(s), retrieval_date {today}")
    elif updated:
        print(
            f"\n{len(updated)} document(s) have no recorded checksum. "
            f"Re-run with --update-checksums to record them."
        )

    if manual:
        print("\nmanual - not fetchable, download by hand:")
        for row in manual:
            print(f"  {row.doc_id:<32} {row.source_url}")
            print(f"  {'':<32} -> {row.raw_path}")

    if failures:
        print(f"\n{len(failures)} problem(s): {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
