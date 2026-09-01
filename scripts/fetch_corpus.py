#!/usr/bin/env python
"""Fetch the scriptable half of the corpus from `data/manifest.csv`.

    python scripts/fetch_corpus.py --dry-run           # show the plan, touch nothing
    python scripts/fetch_corpus.py                     # fetch phase v1, verify checksums
    python scripts/fetch_corpus.py --update-checksums  # fetch and record new checksums

Rows marked `access=manual` are never fetched - FSRA returns 403 to scripts. They are
listed at the end of the run with the exact path each document is expected at, so the
manual half of the corpus is reproducible by verification: the SHA-256 in the manifest
is what proves a hand-downloaded file is the same document the numbers were measured on.

Raw bytes are written untouched. Parsing is Step 2 - hashing a parser's output would
prove nothing about the source document.

Exit code is non-zero if any fetch failed or any checksum mismatched, so this is safe
to run in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import replace

import httpx

from insurance_rag.config import RAW_DIR
from insurance_rag.corpus.manifest import (
    Access,
    ManifestRow,
    Phase,
    load_manifest,
    select,
    write_manifest,
)

USER_AGENT = "InsuranceRAG/0.1 (research corpus fetch; https://github.com/sakib078/InsuranceRAG)"
TIMEOUT = httpx.Timeout(30.0)
RETRIES = 3
BACKOFF_SECONDS = 2.0
POLITE_DELAY_SECONDS = 1.0  # between successful fetches; e-Laws is a government host


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        help="write computed SHA-256 values back into the manifest",
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

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    updated: dict[str, str] = {}
    failures: list[str] = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True
    ) as client:
        for row in scriptable:
            # A local copy that already matches the recorded checksum is the document
            # the numbers were measured on. Don't re-fetch it.
            if not args.force and row.sha256 and row.raw_path.exists():
                if sha256(row.raw_path.read_bytes()) == row.sha256:
                    print(f"  ok       {row.doc_id:<32} cached, checksum verified")
                    continue

            try:
                content = fetch(client, row)
            except RuntimeError as exc:
                print(f"  FAILED   {exc}")
                failures.append(row.doc_id)
                continue

            digest = sha256(content)
            row.raw_path.write_bytes(content)
            size_kb = len(content) / 1024

            if not row.sha256:
                print(f"  new      {row.doc_id:<32} {size_kb:8.1f} KB  {digest[:16]}...")
                updated[row.doc_id] = digest
            elif digest == row.sha256:
                print(f"  ok       {row.doc_id:<32} {size_kb:8.1f} KB  checksum verified")
            else:
                # e-Laws reconsolidates without notice; a changed hash is a real event,
                # not noise. It means the corpus moved under the eval numbers.
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
        write_manifest([replace(r, sha256=updated.get(r.doc_id, r.sha256)) for r in rows])
        print(f"\nmanifest updated with {len(updated)} checksum(s)")
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
