# Source Map

Where the corpus comes from, what can be scripted, and what each source costs us.
Feeds `data/manifest.csv` (Step 1).

Probed 2026-08-29. Access column is a live finding, not an assumption — re-verify before
blaming the fetch script.

## Access probe

```
403  fsrao.ca      page and PDF, plain and with a browser User-Agent
200  ontario.ca/laws   clean HTML
```

FSRA sits behind bot protection. `scripts/fetch_corpus.py` **cannot download OAP 1 or the
OPCF endorsements** — those need manual download into `data/raw/`, with SHA-256 checksums in
the manifest so the corpus stays reproducible-by-verification even though it is not
reproducible-by-script.

Worth confirming in a browser: the 403 may be IP- or region-specific rather than a universal
block.

Re-run the probe with:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -L <url>
```

## Why e-Laws is the backbone

Ontario e-Laws serves clean HTML with **explicit semantic markup**:

```
p.heading1-e / heading3-e                 hierarchy
p.section-e / subsection-e / clause-e     the legal levels, already labelled
p.definition-e / firstdef-e / defclause-e defined terms, explicitly tagged
p.ConsolidationPeriod-e / version-e       effective dates, machine-readable
p.footnote-e / Pnote-e                    footnotes, separable from body
```

For this portion of the corpus the hard chunking problems are already solved in the markup:
no PDF extraction, no heading heuristics, no hyphenation repair, no column interleaving, and
defined terms arrive pre-identified. Ancestor paths can be reconstructed exactly rather than
guessed.

So: e-Laws is the scriptable backbone. The FSRA PDFs are the harder second layer — and the
layer that creates the exclusion-vs-coverage contradictions the project is actually about.

## Sources

| Source | Contains | Access | Licence |
|---|---|---|---|
| [e-Laws — SABS, O. Reg. 34/10](https://www.ontario.ca/laws/regulation/100034) | ~70 sections: benefits, limits, **General Exclusions**, definitions; heavily cross-referenced | Scriptable, tagged HTML | King's Printer |
| [e-Laws — Fault Determination Rules, R.R.O. 1990 Reg. 668](https://www.ontario.ca/laws/regulation/900668) | 40+ numbered collision scenarios with precise conditions | Scriptable, tagged HTML | King's Printer |
| e-Laws — Insurance Act Part VI, O. Reg. 664 | Statutory conditions; the legal frame around the policy | Scriptable | King's Printer |
| [FSRA — OAP 1, application and endorsement forms](https://www.fsrao.ca/automobile-insurance-policy-oap-1-application-and-endorsement-forms) | The policy wording itself + ~30 OPCF endorsements | **Manual — 403 to scripts** | Verify per document |
| [FSRA — Auto insurance guidance](https://www.fsrao.ca/industry/auto-insurance/regulatory-framework/guidance-auto-insurance) | Bulletins and Superintendent's Guidelines, archived to 2001 | Manual | Verify per document |
| [LAT / AABS decisions](https://tribunalsontario.ca/lat-aabs/laws-rules-and-decisions/) (via [CanLII](https://www.canlii.org/en/on/laws/regu/o-reg-34-10/latest/o-reg-34-10.html)) | Real disputes interpreting these clauses | CanLII restricts bulk access | Restrictive |
| IBC consumer guides | Plain-language restatement of OAP 1 | Manual | Copyrighted |

## Licence notes

**King's Printer for Ontario** permits reproduction of statutes and regulations without
permission or charge, provided the material is reproduced accurately, Crown copyright is
acknowledged, and the copy states it is **not an official version**. Cheap to comply with —
do it in the manifest and in any surfaced citation.

**CanLII** terms restrict bulk downloading. Use LAT decisions as a source of realistic
*questions* written by hand for the golden set — do not ingest them.

**IBC and insurer material** is copyrighted. Consistent with the repo's existing stance,
source documents are not redistributed; the manifest records provenance only.

## Known gaps

- **Reg 668 collision diagrams are images** (`p.scanned-e`). The diagrams carry meaning the
  rule text does not fully capture. Disclose this in the README rather than implying the text
  is complete.
- **Older FSCO-era bulletins are scans.** Either OCR them or exclude them — decide explicitly
  and record which in the manifest.
- **Versioning.** SABS benefit levels changed materially in 2010 and 2016. Every document
  needs an effective-date field, and there must be a rule for what happens when superseded
  text is retrieved.

## Scoping note

The authoritative core is roughly 35 documents but tens of thousands of clauses. Document
*count* is the wrong target — padding toward 100 with brochures and FAQ pages adds
near-duplicates that restate the same clauses in plain language, which dilutes retrieval
signal and makes recall@k ambiguous (which chunk is the "correct" one?).

Target chunk count, not document count.