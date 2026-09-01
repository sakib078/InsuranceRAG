# Build Plan

Twelve checkpoints in three phases. Each one stops for review before the next begins.
Source spec: `insurance_rag_project.pdf` §06 (non-negotiables) and §07 (schedule).
Corpus scope and access findings: `data/sourceMap.md`.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Retrieval stack | Local open-source (BGE + cross-encoder) | Anyone can clone and reproduce the eval numbers with no API keys |
| Corpus | Ontario auto insurance only | Homogeneous corpus sharpens the exclusion-vs-coverage clause failure |
| v1 corpus | The **scriptable** layer only — five e-Laws documents | The access probe, not the wish list, defines what v1 can reproduce |
| v1 scope | Library only — no FastAPI, Docker, or deploy until Step 10 | Get measured numbers before containerising a pipeline that Step 8 restructures |
| Gold labels | Legal citations, resolved to chunk IDs at eval time | Survives re-chunking, agentic RAG, and corpus growth — one eval set spans all of them |
| Deploy | Azure Container Apps, small model profile | Only genuinely $0 option that fits a ~1GB container (scale-to-zero) |

---

## Verified sources

Probed 2026-09-01. Re-verify before blaming the fetch script. e-Laws IDs follow
`O. Reg. NN/YY -> YYNNNN`; statutes use `90i08`-style chapter IDs.

| Phase | Document | URL | Structure |
|---|---|---|---|
| **v1** | O. Reg. 34/10 — SABS (current) | `ontario.ca/laws/regulation/100034` | 81 s / 263 sub / 267 cl / 36 def; consolidated 2026-07-01 |
| **v1** | R.R.O. 1990 Reg. 668 — Fault Determination | `.../regulation/900668` | 20 s / 41 sub; **21 scanned + ~40 diagrams**; consolidated 2018-10-17 |
| **v1** | R.R.O. 1990 Reg. 664 — Automobile Insurance | `.../regulation/900664` | 30 s / 79 sub / 63 cl / 16 def; consolidated 2026-07-01 |
| **v1** | Insurance Act, Part VI | `ontario.ca/laws/statute/90i08` | 1.3 MB whole Act, 536 s — **Part VI is a slice**; carries s. 267.5 threshold |
| **v1** | O. Reg. 461/96 — Court Proceedings for Auto Accidents | `.../regulation/960461` | 13 s / 27 sub / 31 KB; deductibles and threshold mechanics |
| v2 | O. Reg. 283/95 — Disputes Between Insurers | `.../regulation/950283` | 15 s / 24 KB; insurer-facing priority disputes |
| v2 | Compulsory Automobile Insurance Act | `.../statute/90c25` | 67 KB |
| v2 | Motor Vehicle Accident Claims Act | `.../statute/90m41` | 83 KB; uninsured / unidentified motorist fund |
| v2 | O. Reg. 403/96 — SABS (accidents ≥ Nov 1996) | `.../regulation/960403` | 89 s / 431 sub — **revoked 2020-07-03** (O. Reg. 348/20); see Step 9 |
| v2 | R.R.O. 1990 Reg. 672 — SABS (accidents < Jan 1994) | `.../regulation/900672` | 29 s — **revoked 2020-07-03** (O. Reg. 346/20) |
| v2 | FSRA — OAP 1, OPCF endorsements, guidance | `fsrao.ca/...` | **403 to scripts**, plain and with a browser UA |
| excluded | IBC consumer guides | `ibc.ca` | Copyrighted *and* near-duplicate of OAP 1 |
| excluded | LAT / AABS decisions | `tribunalsontario.ca/lat-aabs/...` | CanLII restricts bulk access — read by hand for Step 3 questions |

**Why not crawl.** e-Laws does not link an Act to its regulations (the Insurance Act page
contains zero `/laws/regulation/` hrefs), so a crawl from the Act finds nothing. The real
problem is *selection* — which regulations govern, and which are revoked — and that is a
legal judgment a link-follower cannot make. e-Laws still serves revoked regulations at live
URLs with no structural marker distinguishing them from current law, so a crawl would ingest
two dead SABS versions as though they were operative: the single most damaging thing that
could happen to recall@k.

---

# Phase A — v1: a measured pipeline on the scriptable corpus

Steps 1–7. **e-Laws only, library only.** No PDF extraction, no web layer, no deploy.

### ✅ 0 — Scaffold
Directory tree, `pyproject.toml`, `config.py` with the eval/serve model profiles, `.env.example`, `.gitignore`.

---

### ⬜ 1 — Corpus assembly
Five documents, ~1,250 tagged structural units. **Target chunk count, not document count** — padding toward 100 documents with brochures adds near-duplicates that make recall@k ambiguous.

- [ ] `data/manifest.csv` — columns: `doc_id, title, citation, source_url, doc_type, access (script/manual), consolidation_date, sha256, licence_note, phase (v1/v2/excluded), notes`
- [ ] Every row from the **Verified sources** table above goes in now, including v2 and excluded — the manifest is the scale-up ledger, so Step 9 is a flag flip rather than a rewrite, and an exclusion decision is recorded rather than forgotten
- [ ] `insurance_rag/corpus/manifest.py` — typed loader (frozen dataclass + `StrEnum` for `access`/`phase`), validates columns and unique `doc_id`, raises loudly. Step 2 needs the same reader.
- [ ] `scripts/fetch_corpus.py` — fetches `access=script AND phase=v1` into `data/raw/`, writes **raw bytes untouched** (parsing is Step 2; hashing parser output would prove nothing), computes SHA-256, skips manual rows with the path they belong at. Flags: `--dry-run`, `--update-checksums`, `--force`, `--phase`.
- [ ] `sha256` ships blank and is recorded on first fetch. After that a mismatch is a hard failure — SABS and Reg 664 were both reconsolidated 2026-07-01, so this corpus moves under the eval numbers.
- [ ] King's Printer attribution + "not an official version" recorded per row
- [ ] **Review:** the manifest, before anything is downloaded

Only dependency is `httpx`; everything else is stdlib. No BeautifulSoup (Step 2), no scrapy (no discovery), no pandas (8 rows, and a dataclass validates what a DataFrame won't).

---

### ⬜ 2 — Semantic chunker
Split on clause / section / table boundaries — **not** character count. e-Laws already tags the hierarchy, so ancestor paths are reconstructed exactly rather than guessed.

**Two markup dialects.** Regulations use the `-e` suffix (`section-e`, `subsection-e`,
`clause-e`, `definition-e`); statutes drop it (`section`, `subsection`, `definition`,
`headnote`). The chunker handles both.

**Open problem:** slicing Part VI out of the 1.3 MB Insurance Act. The rest of the Act is
life, fire, and mutual insurance licensing and would badly dilute the corpus. Part markers
were not found in the body on a first pass — only in the table of contents. Solve this before
the Act row is fetched, or drop it back to v2.

Chunk schema is the contract that must survive the Step 9 PDF layer — get it right once:

```
chunk_id, doc_id, citation, ancestor_path, text,
defined_terms[], effective_date, modality (text|table|image)
```

`modality` is carried from the start even though v1 only ever emits `text`. It costs nothing
now and avoids a schema migration when the multimodal encoder lands in Step 9.

- [ ] `insurance_rag/chunking/elaws.py`
- [ ] Citation → chunk_id resolver — Step 3 and every eval downstream depend on it
- [ ] `tests/test_chunking.py`
- [ ] **Review:** sample chunks from all five documents + passing tests

---

### ⬜ 3 — Golden eval set
Written **before** any retrieval code exists, so it tests the problem rather than the solution. 30–50 pairs, all answerable from the v1 corpus alone.

Weighted toward the failure mode: SABS General Exclusions, benefit eligibility conditions, s. 267.5 threshold questions, and Reg 668 scenario pairs that differ by a single condition.

- [ ] `evals/golden_set_v1.jsonl` — `question, answer, gold_citations[], slice (exclusion|definition|fault|threshold|multi-hop)`
- [ ] Labels are **citations**, never chunk IDs
- [ ] LAT / AABS decisions read for realistic question phrasing — **not ingested**
- [ ] Exclude fault scenarios whose meaning lives in the scanned diagrams; re-admitted in Step 9. How many that is can only be counted once Step 2 runs — if it is most of the 40, Reg 668 contributes far less to v1 than assumed, and that is a real finding.
- [ ] Frozen once Step 4 runs. Harder questions go in new files, never edits to this one.
- [ ] **Review:** the Q/A pairs — read them as a domain reader, not a reviewer of code

---

### ⬜ 4 — Dense baseline + eval harness
- [ ] `insurance_rag/retrieval/dense.py`
- [ ] `evals/retrieval_eval.py` — recall@k and MRR, **with a per-slice breakdown**
- [ ] **Review:** the first real number. Baseline row of the results table.

---

### ⬜ 5 — BM25 + fusion
Sparse index to catch exact clause numbers and defined terms that embeddings blur together. Fused with reciprocal rank fusion.

- [ ] `insurance_rag/retrieval/sparse.py`, `fusion.py`
- [ ] **Review:** second number — delta vs. the dense baseline

---

### ⬜ 6 — Cross-encoder reranker
Rerank the fused top-k. This is where the headline delta should appear.

- [ ] `insurance_rag/retrieval/rerank.py`
- [ ] **Review:** third number, per slice — the interview story, measured

---

### ⬜ 7 — Generation + ragas
Library only. A callable function and a script — no web layer. This is the application-layer boundary.

- [ ] `insurance_rag/generation/answer.py` (Claude)
- [ ] `scripts/ask.py` — the only entry point in Phase A
- [ ] `evals/ragas_eval.py` — groundedness, answer relevance, context precision
- [ ] **Review:** generation eval table

---

## 🚩 v1 gate

Three retrieval numbers, a generation table, reproducible with no API key for the retrieval half.

The honest question to answer here: **did the rerank delta actually appear without OAP 1?**
If the exclusion-vs-coverage contradiction turns out to live mostly in the FSRA policy
wording, that finding gets written down — it becomes the argument for Step 9, not something
to paper over.

---

# Phase B — Agentic RAG

### ⬜ 8 — Agentic retrieval loop
Same corpus, same frozen golden set, so the delta against Step 7 is a number rather than a claim.

- [ ] Tools: `search_corpus(query, filters)`, `lookup_definition(term)`, `get_clause(citation)`, `get_neighbours(chunk_id)` — the last two are only possible because Step 2 kept citations and ancestor paths
- [ ] Query decomposition for multi-hop: *"is X covered if Y?"* → coverage clause + exclusion + definition, retrieved separately
- [ ] Self-check pass: every claim in the draft answer must carry a citation present in retrieved context
- [ ] `evals/golden_set_multihop.jsonl` — a **second** file of harder questions; the v1 set stays frozen
- [ ] Eval table gains **cost, latency, and tool-call count** columns
- [ ] **Review:** single-shot vs. agentic on both golden sets, with cost

---

# Phase C — Scale and ship

### ⬜ 9 — Corpus scale-up: the manual and multimodal layers
FSRA is 403 to scripts. Those documents are downloaded by hand into `data/raw/` with SHA-256 in the manifest — reproducible **by verification** rather than by script.

- [ ] OAP 1, ~30 OPCF endorsements, FSRA guidance — flip their manifest rows to `phase=v1`. One row per endorsement is added as each PDF is downloaded; per-endorsement URLs are not guessed, because the index page cannot be fetched.
- [ ] PDF extraction path in the chunker, emitting the **identical chunk schema**
- [ ] **Multimodal encoding** for non-text content — Reg 668 collision diagrams, scanned FSCO-era bulletins, policy tables. Either a vision encoder embedding images into the shared retrieval space, or a VLM caption/transcription pass producing text chunks. Either way they land as chunks with `modality=image|table` and a `source_page`, so a retrieved diagram is citable rather than a silent gap. Record which route each source took in the manifest.
- [ ] Re-admit the fault scenarios excluded from the golden set in Step 3
- [ ] **Temporal-disambiguation slice.** Promote O. Reg. 403/96 and R.R.O. Reg. 672 — both **revoked on 2020-07-03**, not operative law, but still served by e-Laws. Near-identical wording to O. Reg. 34/10 with *different benefit amounts*. They are the sharpest retrieval trap in the corpus, and deliberately out of v1 because they would make recall@k meaningless there. Because they are revoked rather than merely superseded, the correct behaviour is unambiguous: **never retrieve them for a current-law question**. That makes them clean negative distractors, not a versioning puzzle — an easier and more honest test than the one originally planned. Needs effective-date handling first.
- [ ] Re-run Steps 4–8 unchanged; report both corpus sizes side by side
- [ ] **Review:** the same table, two corpora

---

### ⬜ 10 — Application layer, tracing, Docker, CI
- [ ] `insurance_rag/api/` — FastAPI
- [ ] `insurance_rag/tracing/` — latency, token count, retrieved chunk IDs, cost per query
- [ ] `docker/Dockerfile` + `docker-compose.yml` (app + vector store)
- [ ] GitHub Actions: tests + eval suite on every push
- [ ] Golden-set regression gate that **fails CI when retrieval quality drops**
- [ ] **Review:** a green CI run

---

### ⬜ 11 — Deploy + README
- [ ] Live URL on Azure Container Apps (serve profile, scale-to-zero)
- [ ] Warm-up ping so a recruiter's first click isn't a cold start
- [ ] README led by the results table — both profiles, both corpora, real numbers
- [ ] 90-second demo video

---

## Known gaps

- **Reg 668 collision diagrams are images** (`p.scanned-e`) — 21 scanned paragraphs, ~40 diagrams — and carry meaning the rule text does not fully capture. v1 excludes the affected scenarios and discloses it; the Step 9 multimodal encoder is the fix, not a workaround.
- **Older FSCO-era bulletins are scans.** Same route: a VLM transcription pass, recorded per document in the manifest.
- **Tables in the FSRA PDFs** (benefit limits, premium schedules) lose structure under plain text extraction. Same path, `modality=table`.
- **Part VI cannot yet be sliced out of the Insurance Act.** Blocks that document's inclusion in v1 — see Step 2.
- **Versioning.** Every document carries an effective-date field from Step 1; the retrieval rule for superseded text is decided in Step 9, alongside the temporal slice.

---

## Ground rules

**The access probe defines the corpus, not the wish list.** v1 is what a script can fetch. Everything else is Phase C — recorded in the manifest from day one so scaling is a flag flip, not a rewrite.

**Report measured numbers, whatever they are.** The `0.61 → 0.87` in the source PDF is illustrative, not a target to reverse-engineer. This repo's entire value is that it's inspectable — a fabricated table makes it worse than no repo at all.

**The golden set comes before the retrieval code.** Step 3 precedes Step 4 for a reason. An eval set written after the fact tests what you happened to build.

**Gold labels are citations, not chunk IDs.** Chunk IDs are an implementation detail that changes every time the chunker is retuned. Citations don't. This is what lets one eval set span re-chunking, agentic RAG, and corpus growth.

**Freeze the golden set when the corpus freezes.** Harder questions go in new files, so every number stays comparable to the one before it.

**Report agentic cost next to agentic quality.** Step 8 is more expensive per query than Step 7. A quality delta without the cost column is a misleading comparison.

**Both model profiles get reported.** The deployed demo runs smaller weights than the eval. Say so in the README rather than letting someone assume otherwise.
