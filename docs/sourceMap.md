# Source Map

Per-document status. Full provenance, checksums, and notes live in `data/manifest.csv`.
Probed 2026-09-01 · fetched 2026-09-01.

✅ on disk in `data/raw/` · ⬜ not yet · ✖ never (excluded by decision)

## v1 — the corpus the pipeline runs on

| | Document | Type | Access | Size |
|---|---|---|---|---|
| ✅ | [O. Reg. 34/10 — SABS](https://www.ontario.ca/laws/regulation/100034) | regulation | script | 254 KB |
| ✅ | [Insurance Act, Part VI](https://www.ontario.ca/laws/statute/90i08) | statute | script | 1280 KB |
| ✅ | [R.R.O. 1990 Reg. 664 — Automobile Insurance](https://www.ontario.ca/laws/regulation/900664) | regulation | script | 80 KB |
| ✅ | [R.R.O. 1990 Reg. 668 — Fault Determination](https://www.ontario.ca/laws/regulation/900668) | regulation | script | 47 KB |
| ✅ | [O. Reg. 461/96 — Court Proceedings](https://www.ontario.ca/laws/regulation/960461) | regulation | script | 32 KB |
| ⬜ | [OAP 1 — Owner's Policy](https://www.fsrao.ca/automobile-insurance-policy-oap-1-application-and-endorsement-forms) | policy | **manual** | **required** |
| ⬜ | OPCF 44R — Family Protection Coverage | endorsement | **manual** | if substantive |
| ⬜ | OPCF 43 — Removing Depreciation Deduction | endorsement | **manual** | if substantive |
| ⬜ | FSRA **AU0026ORG** — Minor Injury Guideline | bulletin | **manual** | **required** |
| ⬜ | FSRA **AU0134INT** — Catastrophic Impairment | bulletin | **manual** | high value |
| ⬜ | FSRA **AU0053ORG** — Attendant Care Hourly Rate | bulletin | **manual** | check supersession |

**5 of 11 fetched.** Hand-download the six into `data/raw/<doc_id>.pdf`, then
`python scripts/fetch_corpus.py --update-checksums`.

**Why these three guidance documents, out of 74 active entries.** Selection was driven by
what the SABS text actually cites, not by title:

| Term in SABS | Mentions | Document that resolves it |
|---|---|---|
| "Minor Injury Guideline" | **33** — s. 40 is titled after it | AU0026ORG |
| "catastrophic impairment" | 45 | AU0134INT |
| "attendant care" | 104 | AU0053ORG |
| "Professional Services Guideline" | **0** | — dropped despite the promising name |

The MIG is the only guideline SABS names; every other "Guideline" mention is a generic
reference to one issued by the Chief Executive Officer. Skip all filing, underwriting, rate,
AMP, complaints, whistle-blower, IT-risk and innovation guidance — insurer- and
regulator-facing, so no claimant question ever reaches them.

**Supersession risk on attendant care:** AU0049, AU0053 and AU0054 all cover the hourly rate.
Take only the current one. SABS deems references to a CEO guideline to include the last
Superintendent's guideline issued before 2019-06-08, so a 2018 guideline may still govern.

**Keep-or-drop test for an endorsement:** does it carry numbered clause wording that changes
a coverage, limit, definition, or exclusion in OAP 1? If it is mostly fillable form fields,
drop it. OPCF 20 and 27 were demoted to v2 for this reason. Titles are unverified.

## v2 — promote later

| | Document | Type | Note |
|---|---|---|---|
| ⬜ | [O. Reg. 283/95 — Disputes Between Insurers](https://www.ontario.ca/laws/regulation/950283) | regulation | insurer-facing; few natural questions |
| ⬜ | [Compulsory Automobile Insurance Act](https://www.ontario.ca/laws/statute/90c25) | statute | requirement to insure |
| ⬜ | [Motor Vehicle Accident Claims Act](https://www.ontario.ca/laws/statute/90m41) | statute | uninsured motorist fund |
| ⬜ | [O. Reg. 403/96 — SABS](https://www.ontario.ca/laws/regulation/960403) | regulation | **revoked 2020-07-03** — week 5 distractor |
| ⬜ | [R.R.O. 1990 Reg. 672 — SABS](https://www.ontario.ca/laws/regulation/900672) | regulation | **revoked 2020-07-03** — week 5 distractor |


## Access notes

**FSRA sits behind a Cloudflare challenge.** Every content page returns 403 with
`server: cloudflare` and the body "Just a moment... Enable JavaScript and cookies to
continue" — identical across four User-Agents including Googlebot, so it is not UA
filtering. `robots.txt` serves 200; `sitemap.xml` does not. Unlike e-Laws, this is a
deliberate bot challenge, and scripting past it would mean evading an access control rather
than choosing a render variant. These documents are hand-downloaded, full stop. The guidance
index is 74 entries behind a JS-filtered table with a two-page drill-down to each PDF, which
is another reason to take only the 2–3 that matter.

**e-Laws is a React SPA.** It returns HTTP 200 to everything but serves a 54 KB
"needs JavaScript" shell unless the User-Agent contains `curl` or `Googlebot`. No JSON API
exists. `validate_document()` in `scripts/fetch_corpus.py` rejects the shell.

**Checksums are content hashes, not response hashes.** e-Laws injects a bot-management
script with a fresh session token per request, so the raw response differs every time.
The manifest hashes the response with `<script>` blocks stripped.

**Revoked regulations are served at live URLs** with no structural marker separating them
from current law. The manifest's `status` column carries this.

**King's Printer** permits free reproduction provided the copy is accurate, acknowledges
Crown copyright, and states it is **not an official version** — a licence condition, so the
disclaimer is mandatory rather than decorative.
