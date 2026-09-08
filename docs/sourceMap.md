# Source Map

Ontario only. **20 documents in scope — 11 on disk, 9 to acquire.**
Per-document provenance, checksums, and notes live in `data/manifest.csv`.

Scripted HTML → `data/raw_html/<doc_id>.html` · hand-downloaded PDFs → `data/pdfs/`
(publisher's own filename, recorded in the manifest's `local_file` column).

| Tag | Meaning |
|---|---|
| **HAVE** | on disk, ingested, indexed |
| **FETCH** | e-Laws — `scripts/fetch_corpus.py` handles it |
| **DOWNLOAD** | FSRA — behind a Cloudflare challenge, hand-downloaded by necessity |

---

## A — Indexed now (11)

| Document | Type | Access | Tag |
|---|---|---|---|
| [O. Reg. 34/10 — SABS](https://www.ontario.ca/laws/regulation/100034) | regulation | script | **HAVE** 254 KB |
| [Insurance Act, Part VI](https://www.ontario.ca/laws/statute/90i08) | statute | script | **HAVE** 1280 KB |
| [R.R.O. 1990 Reg. 664 — Automobile Insurance](https://www.ontario.ca/laws/regulation/900664) | regulation | script | **HAVE** 80 KB |
| [R.R.O. 1990 Reg. 668 — Fault Determination](https://www.ontario.ca/laws/regulation/900668) | regulation | script | **HAVE** 47 KB |
| [O. Reg. 461/96 — Court Proceedings](https://www.ontario.ca/laws/regulation/960461) | regulation | script | **HAVE** 32 KB |
| OAP 1 — Owner's Policy (July 2026) | policy | manual | **HAVE** 68 pp |
| FSRA AU0026ORG — Minor Injury Guideline | bulletin | manual | **HAVE** |
| FSRA AU0053ORG — Attendant Care Hourly Rate (01/18) | bulletin | manual | **HAVE** |
| FSRA AU0054ORG — Revised Attendant Care Hourly Rate | bulletin | manual | **HAVE** |
| FSRA AU0125ORG — Transportation Expense Guideline | bulletin | manual | **HAVE** |
| FSRA AU0129DEC — Auto Insurance Indexation Amounts | bulletin | manual | **HAVE** |

≈ 2,900 chunks.

## B — Promote in Phase 0 (5)

Manifest rows already exist with researched notes; flip `phase` `v2` → `v1`.

| Document | Type | Status | Tag | Why |
|---|---|---|---|---|
| [O. Reg. 283/95 — Disputes Between Insurers](https://www.ontario.ca/laws/regulation/950283) | regulation | current | **FETCH** | insurer-facing register; lexical variety |
| [Compulsory Automobile Insurance Act](https://www.ontario.ca/laws/statute/90c25) | statute | current | **FETCH** | the requirement to insure, evidence of insurance |
| [Motor Vehicle Accident Claims Act](https://www.ontario.ca/laws/statute/90m41) | statute | current | **FETCH** | uninsured and unidentified motorist fund |
| [O. Reg. 403/96 — SABS](https://www.ontario.ca/laws/regulation/960403) | regulation | **revoked** | **FETCH** | negative distractor — never correct for a current-law question |
| [R.R.O. 1990 Reg. 672 — SABS](https://www.ontario.ca/laws/regulation/900672) | regulation | **revoked** | **FETCH** | negative distractor |

≈ +1,000 chunks. Both revoked rows are served at live e-Laws URLs with no structural marker;
`status` in the manifest is the only thing that distinguishes them.

## C — Add in Phase 0 (4)

New rows. No schema change; all four are either scriptable or short forms.

| Document | Type | Tag | Why |
|---|---|---|---|
| [R.R.O. 1990 Reg. 676 — Uninsured Automobile Coverage](https://www.ontario.ca/laws/regulation/900676) | regulation | **FETCH** | Reg 676 requires its terms, exclusions and limits to be in **every** motor-vehicle liability policy. Pairs with the MVAC Act in B — without it the corpus has the fund that pays out but not the coverage that triggers it. |
| [OPCF 49 — Agreement Not to Recover for Collision Damage](https://www.fsrao.ca/opcf-49-agreement-not-recover-loss-or-damage-automobile-collision-ontario-automobile-policy-oap-1) | endorsement | **DOWNLOAD** | A pure coverage-removal endorsement — the adversarial shape this project exists to handle. |
| [OPCF 20 — Coverage for Transportation Replacement](https://www.fsrao.ca/opcf-20-coverage-transportation-replacement) | endorsement | **DOWNLOAD** | Adds coverage, and interacts with AU0125ORG already in A. Clean multi-hop. |
| OPCF 44R — Family Protection Coverage | endorsement | **DOWNLOAD** | The most commonly attached OPCF in Ontario; modifies uninsured/underinsured limits, so it pairs with Reg 676. Find via the [FSRA auto forms hub](https://www.fsrao.ca/industry/auto-insurance/forms-auto-insurance). |

≈ +100–150 chunks.

**These four fill `doc_type=endorsement`, which the schema declares and zero documents
currently use** — while the Phase 7 agent is specified to check endorsements before answering.

---

## Deferred — blocked on schema, not on access

| Document | Blocker |
|---|---|
| [OAP 1 — July 2025](https://www.fsrao.ca/sites/default/files/2025-07/OAP1-2025_FINAL-EN_aoda.pdf) | `Chunk` has no `effective_from` / `effective_to` / `supersedes`. Four OAP 1 editions would all carry `status=current` with no way to rank them by date — four near-identical documents competing at nearly identical cosine distance, with no correct answer. |
| [OAP 1 — January 2024](https://www.fsrao.ca/media/14931/download) | same |
| [OAP 1 — January 2022](https://www.fsrao.ca/media/5156/download) | same |

Historical versions are the right temporal-leakage test — `status` handles current vs revoked
but cannot order live versions. Revisit when the schema carries effective dates.

## Excluded — by decision, not by access

| Document | Why |
|---|---|
| IBC consumer guides | Plain-language restatement of OAP 1. Near-duplicate chunks make recall@5 ambiguous — there is no single correct chunk when two documents say the same thing. Copyrighted besides. |
| [Aviva Ontario auto policy](https://www.aviva.ca/content/dam/aviva-public/ca/pdf/hopp-ontario-auto-policy.pdf) | Insurer packaging of OAP 1. Same near-duplicate argument; copyrighted. |
| [OAP 4 + OEF garage forms](https://www.fsrao.ca/media/7746/download) | Garage policies are a different vehicle class. No golden-set question naturally reaches them. |
| [LAT / AABS decisions](https://tribunalsontario.ca/lat-aabs/laws-rules-and-decisions/) | Read by hand as a source of realistic question phrasing. Never ingested — `case-law` is not an ingestable `doc_type`, and CanLII's terms prohibit systematic downloading. |

Out of scope entirely: the multi-province corpus in
`docs/Canadian Auto-Insurance Corpus and Contradiction-Aware Retrieval Research.md` — Québec
QPF/QEF, Alberta SPF, BC ICBC, Atlantic, territories, A2AJ bulk case law. It needs jurisdiction
routing and an authority/version data model the `Chunk` contract does not have.

---

## Access notes

**FSRA sits behind a Cloudflare challenge.** Every content page returns 403 with
`server: cloudflare` and "Just a moment... Enable JavaScript and cookies to continue" —
identical across four User-Agents including Googlebot, so not UA filtering. `robots.txt` serves
200; `sitemap.xml` does not. Unlike e-Laws this is a deliberate bot challenge, so these
documents are hand-downloaded, full stop. This applies to the OPCF forms in C.

**e-Laws is a React SPA.** HTTP 200 to everything, but a 54 KB "needs JavaScript" shell unless
the User-Agent contains `curl` or `Googlebot`. No JSON API. `validate_document()` rejects the shell.

**Checksums are content hashes.** e-Laws injects a bot-management script with a fresh session
token per request, so raw responses differ every time. The manifest hashes with `<script>`
blocks stripped.

**Revoked regulations are served at live URLs** with no structural marker. The manifest's
`status` column carries this.

**King's Printer** permits free reproduction provided the copy is accurate, acknowledges Crown
copyright, and states it is **not an official version** — a licence condition, so the disclaimer
is mandatory rather than decorative.

**OPCF forms are FSRA-approved endorsements**, same licence posture as the guidance PDFs:
`licence_note` says verify before redistribution; cite and link, never rehost.

---

## Open items

- **OAP 1 edition mismatch.** On disk: `OAP1_EN.5-07-2026_aoda (1).pdf`. The research doc cites
  the July 2026 edition as `OAP1_EN.3-2026.pdf`. Probably a later revision of the same
  instrument — confirm before a gold locator points into it.
- **`data/pdfs/Fraud Reporting Guidance-v2_EN.pdf` has no manifest row**, so ingestion cannot
  see it. Add a row or delete the file.
