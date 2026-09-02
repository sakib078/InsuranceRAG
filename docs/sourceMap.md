# Source Map

Corpus complete as of 2026-09-01 — **11 of 11 v1 documents on disk**.
Per-document provenance, checksums, and notes live in `data/manifest.csv`.

Scripted HTML → `data/raw_html/<doc_id>.html` · hand-downloaded PDFs → `data/pdfs/`
(publisher's own filename, recorded in the manifest's `local_file` column).

## v1 — the corpus the pipeline runs on

| Document | Type | Access | On disk |
|---|---|---|---|
| [O. Reg. 34/10 — SABS](https://www.ontario.ca/laws/regulation/100034) | regulation | script | 254 KB |
| [Insurance Act, Part VI](https://www.ontario.ca/laws/statute/90i08) | statute | script | 1280 KB |
| [R.R.O. 1990 Reg. 664 — Automobile Insurance](https://www.ontario.ca/laws/regulation/900664) | regulation | script | 80 KB |
| [R.R.O. 1990 Reg. 668 — Fault Determination](https://www.ontario.ca/laws/regulation/900668) | regulation | script | 47 KB |
| [O. Reg. 461/96 — Court Proceedings](https://www.ontario.ca/laws/regulation/960461) | regulation | script | 32 KB |
| OAP 1 — Owner's Policy | policy | manual | 68 pp |
| FSRA AU0026ORG — Minor Injury Guideline | bulletin | manual | ✓ |
| FSRA AU0053ORG — Attendant Care Hourly Rate (01/18) | bulletin | manual | ✓ |
| FSRA AU0054ORG — Revised Attendant Care Hourly Rate | bulletin | manual | ✓ |
| FSRA AU0125ORG — Transportation Expense Guideline | bulletin | manual | ✓ |
| FSRA AU0129DEC — Auto Insurance Indexation Amounts | bulletin | manual | ✓ |


## v2 — promote later

| Document | Note |
|---|---|
| [O. Reg. 283/95 — Disputes Between Insurers](https://www.ontario.ca/laws/regulation/950283) | insurer-facing; few natural questions |
| [Compulsory Automobile Insurance Act](https://www.ontario.ca/laws/statute/90c25) | requirement to insure |
| [Motor Vehicle Accident Claims Act](https://www.ontario.ca/laws/statute/90m41) | uninsured motorist fund |
| [O. Reg. 403/96 — SABS](https://www.ontario.ca/laws/regulation/960403) | **revoked 2020-07-03** — week 5 distractor |
| [R.R.O. 1990 Reg. 672 — SABS](https://www.ontario.ca/laws/regulation/900672) | **revoked 2020-07-03** — week 5 distractor |


## Access notes

**FSRA sits behind a Cloudflare challenge.** Every content page returns 403 with
`server: cloudflare` and "Just a moment... Enable JavaScript and cookies to continue" —
identical across four User-Agents including Googlebot, so not UA filtering. `robots.txt` serves
200; `sitemap.xml` does not. Unlike e-Laws this is a deliberate bot challenge, so these
documents are hand-downloaded, full stop.

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
