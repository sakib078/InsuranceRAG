# Build Plan — Agentic Insurance RAG

Five weeks. **Ship a complete v1 at week 3, harden through week 5.**
Spec: `agentic_insurance_rag_spec.pdf` (v2) — supersedes `insurance_rag_project.pdf` where they disagree.
Corpus access findings: `data/sourceMap.md`.

Every checkpoint stops for review before the next begins. Scope creep is the failure mode
this plan is built to resist: anything not on the **Definition of done** list is week 5 or later.

---

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Corpus | Ontario **auto only** — e-Laws HTML backbone + hand-downloaded FSRA policy forms | Auto has a *regulator-published* standard policy (OAP 1). Home does not — see Deviations. |
| Storage | **One Postgres 16** — pgvector + tsvector/GIN | Chunk text, embeddings, and full-text index in one store. "One Postgres over a vector DB plus a search engine" is a real architecture answer. |
| Ingestion | e-Laws pre-render (HTML) + PyMuPDF (PDF) | Two dialects, one chunk schema. |
| Agent | LangGraph, exclusion check as a **graph edge** | If the model can skip the exclusion check, eventually it will. |
| Retrieval | BGE bi-encoder + `bge-reranker` cross-encoder | No API key needed to reproduce the retrieval numbers. |
| Deploy | **AWS** — EC2 t4g.small, docker-compose, public subnet | ~$22–27/mo. AWS + Docker + CI/CD are claimed skills with no artifact behind them. |
| Eval order | Golden set **before** any retrieval code | Written after, it tests what you happened to build. |
| Gold labels | Clause **locators**, resolved to chunk IDs at eval time | Survives re-chunking, the agent, and corpus growth. |

**Superseded from the previous revision:** Azure Container Apps → AWS. Chroma + `rank-bm25`
→ Postgres. Deploy-last → ship at week 3. Multimodal diagram encoding → dropped (see
Deviation 4).

---

## Deviations from the spec — deliberate, with reasons

**1. Auto only; no home insurance.** The spec suggests "Ontario auto plus home." Ontario auto
has a regulator-published standard policy form (OAP 1) and approved endorsements (OPCF). Home
insurance has **no standard form** — every wording is insurer-copyrighted with no public
equivalent. Adding home swaps a clean licence story for a messy one and buys nothing the
agent needs.

**2. `doc_type` extended.** The spec's literals are `policy | endorsement | bulletin | guide`.
The scriptable backbone is regulations and statutes, which fit none of them. Extended to
`policy | endorsement | bulletin | guide | regulation | statute`.

**3. IBC consumer guides stay excluded.** The spec lists them as a corpus source. They are
copyrighted *and* a plain-language restatement of OAP 1 — near-duplicate chunks make recall@5
ambiguous, because there is no single correct chunk when two documents say the same thing.
The `guide` doc_type stays in the schema for FSRA consumer material if needed.

**4. No OCR, and no multimodal encoder.** The spec puts OCR out of scope and says to *flag*
scanned pages rather than index empty chunks. That kills the multimodal plan from the
previous revision. Reg 668's ~40 collision diagrams get flagged, their scenarios excluded
from the golden set, and the gap disclosed in the README.

**5. `token_count` uses the embedding model's tokenizer, not tiktoken.** The spec says
tiktoken; that is OpenAI's BPE and will mis-count for BGE. Use the actual tokenizer, or the
400–800 token targets measure the wrong thing.

**6. `bge-reranker`, not Cohere Rerank.** The spec offers either. Cohere needs an API key,
which breaks "clone the repo and reproduce the retrieval table."

**7. A web-search tier, beyond the spec — week 5, and outside the measured path.** See
*Citation and disclosure contract* below. It sits after refusal, never inside it.

---

## Citation and disclosure contract

Applies to every answer the system produces. Decided up front because it constrains the chunk
schema, the agent's output format, the UI, and the eval harness at once.

### Every corpus citation renders provenance, not just a locator

```
O. Reg. 34/10, s. 31(1)(a) — Statutory Accident Benefits Schedule
Consolidated 2026-07-01 · ontario.ca/laws/regulation/100034
Not an official version.
```

The chain is `Chunk.doc_id` → manifest row → `source_url`, `citation`, `consolidation_date`,
`status`, `licence_note`. No new storage; the manifest already carries all of it.

- **`status` is a safety field, not bookkeeping.** A chunk from a revoked document must render
  a revoked badge. e-Laws serves revoked regulations at live URLs with no structural marker —
  if the citation does not say so, nothing does.
- **`consolidation_date` tells the reader the law may have moved** since indexing.
- **FSRA PDFs are linked, never rehosted.** Serve the extracted citation plus a link to the
  original; `licence_note` on those rows says "verify per document."

### "Not advice" is a licence term, not boilerplate

King's Printer permits free reproduction **on condition** that the copy states it is not an
official version. So the disclaimer is simultaneously the spec's product decision and a
compliance requirement — which is a better answer than "we added a footer."

Rendered in three places: per citation ("not an official version"), per answer ("document
retrieval, not advice"), and in the interface chrome. Never phrase an answer as a
recommendation; always attach verifiable citations; surface uncertainty explicitly.

### Three answer states, not two

| State | Rendering | Counted in evals? |
|---|---|---|
| **Answered from corpus** | clause locator + source URL + consolidation date + licence line | yes — recall@5, exclusion recall |
| **Not in corpus** (refusal) | "this corpus does not address that" | yes — **false-answer rate** |
| **Public sources say…** (week 5) | visually distinct block, URL + retrieval date, **no clause locator**, marked unverified and not exclusion-checked | **no** |

State 3 layers *on top of* state 2 — the refusal still fires and is still tested. This is the
whole reason the web tier is safe to add: it cannot mask a refusal, because the refusal is
what it renders beneath.

---

## Verified sources

Probed 2026-09-01. e-Laws IDs: `O. Reg. NN/YY -> YYNNNN`; statutes use chapter IDs.

| Set | Document | URL | Notes |
|---|---|---|---|
| **backbone** | O. Reg. 34/10 — SABS | `ontario.ca/laws/regulation/100034` | 81 s / 263 sub / 267 cl / 36 def. **General Exclusions live here.** |
| **backbone** | R.R.O. 1990 Reg. 668 — Fault Determination | `.../regulation/900668` | 20 s / 41 sub; 21 scanned + ~40 diagrams → flag, exclude from golden set |
| **backbone** | R.R.O. 1990 Reg. 664 — Automobile Insurance | `.../regulation/900664` | 30 s / 79 sub / 63 cl / 16 def |
| **backbone** | Insurance Act, Part VI | `ontario.ca/laws/statute/90i08` | 1.3 MB whole Act — Part VI is a slice; s. 267.5 threshold. **Blocked, see Step 3.** |
| **backbone** | O. Reg. 461/96 — Court Proceedings | `.../regulation/960461` | 13 s / 31 KB; deductibles |
| **policy** | OAP 1 Owner's Policy | `fsrao.ca/...oap-1...` | **403 to scripts — download by hand.** The coverage / exclusion / endorsement backbone. |
| **policy** | OPCF endorsements (pick 4–6, incl. 44R, 43, 27, 20) | same page | Hand-download. Endorsements are what make multi-hop real. |
| **bulletin** | FSRA auto guidance | `fsrao.ca/industry/auto-insurance/...` | Hand-download 2–3. |
| distractor | O. Reg. 403/96 — SABS | `.../regulation/960403` | **Revoked 2020-07-03.** Week 5 negative-distractor slice. |
| distractor | R.R.O. 1990 Reg. 672 — SABS | `.../regulation/900672` | **Revoked 2020-07-03.** Same. |
| excluded | IBC consumer guides / LAT decisions | — | Copyright / CanLII bulk restriction. LAT read by hand for question phrasing only. |

Target: **12–15 documents.** That is the spec's range, and it is what makes the 403 survivable
— hand-downloading ten PDFs once is half an hour, not a project. At the previous revision's
50–100 target it was a wall.

### Two access findings that are already load-bearing

**e-Laws is a React SPA.** It returns HTTP 200 to everything but serves a 54 KB "needs
JavaScript" shell to unrecognised User-Agents, and the full 260 KB pre-render only when the UA
contains `curl` or `Googlebot`. A fetcher with an honest UA would have written five empty
shells, recorded their checksums, and produced a corpus with no law in it.
`validate_document()` in `scripts/fetch_corpus.py` fails loudly on this. There is no JSON API.

**FSRA 403s scripts** — plain and with a browser UA, re-verified 2026-09-01. Those documents
are reproducible *by verification*: hand-downloaded, SHA-256 recorded in the manifest.

---

# Week 1 — Schema, corpus, ingestion, golden set

### ✅ 0 — Scaffold
Directory tree, `pyproject.toml`, `config.py` profiles, `.env.example`, `.gitignore`.

**Dependency changes this plan requires:** drop `chromadb` and `rank-bm25` (Postgres does
both). Add `psycopg[binary]`, `pgvector`, `langgraph`. Keep `pymupdf`; keep `pdfplumber` only
if table extraction needs it.

---

### 🟡 1 — Corpus manifest
**Mostly done.** `data/manifest.csv` (15 rows) + `insurance_rag/corpus/manifest.py` (typed
loader, `Access` / `Phase` / `Status` enums) + `scripts/fetch_corpus.py` (UA fix,
`validate_document`).

- [ ] Add a `retrieval_date` column — the spec asks for it, and provenance is incomplete without it
- [ ] Extend `doc_type` to the six values in Deviation 2
- [ ] Promote OAP 1 + 4–6 OPCF endorsements + 2–3 FSRA bulletins from `v2` to `v1`, one row each
- [ ] Hand-download those PDFs into `data/raw/`, then record SHA-256 with `--update-checksums`
- [ ] Run the scriptable fetch: `PYTHONPATH=. python scripts/fetch_corpus.py --update-checksums`
- [ ] **Review:** the checksum diff, before committing

---

### ⬜ 2 — The chunk schema — write this first
Nothing else starts until this is fixed. It is the contract every later step depends on.

```python
class Chunk:
    chunk_id:      str
    doc_id:        str
    doc_type:      Literal["policy","endorsement","bulletin","guide","regulation","statute"]
    chunk_role:    Literal["coverage","exclusion","definition","condition","schedule","other"]
    locator:       str        # "OAP1 s.4.2(b)" / "O. Reg. 34/10 s.31(1)(a)" — human-verifiable
    ordinal:       int        # position, for neighbour fetch
    defined_terms: list[str]
    text:          str
    token_count:   int        # embedding model's tokenizer, not tiktoken
    embedding:     list[float]
```

`chunk_role` **is the field that makes the agent possible** — it is what lets a search be
scoped to exclusions only. Classify by rules first (heading text, clause numbering, trigger
phrases), fall back to an LLM pass for the ambiguous remainder, then spot-check by hand.

- [ ] `insurance_rag/schema.py`
- [ ] **Review:** the schema, before any ingestion code exists

---

### ⬜ 3 — Ingestion
Two dialects, one output schema.

**HTML (e-Laws).** Regulations tag the hierarchy with an `-e` suffix (`section-e`, `clause-e`,
`definition-e`); **statutes drop it** (`section`, `subsection`, `definition`). Handle both.

**PDF (FSRA).** Layout-aware — policy documents are two-column and a naive extract interleaves
them, silently poisoning every downstream chunk. Clause numbering parsed into `locator`;
tables extracted intact as markdown, because a deductible schedule shredded across three
chunks is unretrievable; scanned pages flagged, not indexed.

**Open problem — Insurance Act Part VI.** The fetch returns the whole 1.3 MB Act; the rest is
life / fire / mutual licensing. Part markers were not found in the body, only in the TOC.
Solve the slice or drop the document — do not let it dilute the corpus.

- [ ] `insurance_rag/ingest/elaws.py`, `insurance_rag/ingest/pdf.py`
- [ ] `insurance_rag/ingest/roles.py` — `chunk_role` classification
- [ ] `tests/test_ingest.py` — clause-boundary parsing, locator format, role classification
- [ ] **Review:** sample chunks from one regulation, one statute, OAP 1, one endorsement

---

### ⬜ 4 — Postgres in docker-compose
Up now, not at deploy time. Everything downstream writes here.

- [ ] `docker/docker-compose.yml` — Postgres 16 + pgvector
- [ ] Schema: chunks table, vector column, tsvector column + GIN index
- [ ] `scripts/build_index.py`
- [ ] **Review:** row counts per `doc_type` and per `chunk_role`

---

### ⬜ 5 — Golden set — before any retrieval code
40–60 pairs, each labelled with the **locator(s)** that should be retrieved. Three-way split:

| Split | Count | What it measures |
|---|---|---|
| single-hop | ~40% | the answer sits in one clause |
| **multi-hop** | ≥ ⅓ | coverage + an exclusion, or + a definition — **this is the story** |
| unanswerable | 15–20 | plausible, on-topic, corpus genuinely does not address |

- [ ] `evals/golden_set.jsonl`
- [ ] LAT / AABS decisions read by hand for realistic phrasing — **never ingested**
- [ ] Exclude fault scenarios whose meaning lives in the scanned diagrams
- [ ] Frozen once Step 6 runs
- [ ] **Review:** read them as a domain reader, not a code reviewer

---

# Week 2 — Measure everything

Every row of the README table is produced here, in order, each recorded as you go.

### ⬜ 6 — Uniform baseline + eval harness
The **512-token fixed-window, no structure** row. It exists to prove the clause-aware work
paid for itself — without it, "clause-aware chunking helped" is unfalsifiable.

- [ ] `insurance_rag/chunking/uniform.py` (baseline only)
- [ ] `insurance_rag/retrieval/dense.py` — pgvector cosine
- [ ] `evals/retrieval_eval.py` — recall@5, **split single-hop / multi-hop**
- [ ] **Review:** row 1. Expect multi-hop near zero — that gap is the entire argument for the agent.

### ⬜ 7 — Clause-aware chunking → row 2, the ablation delta
### ⬜ 8 — BM25 hybrid → row 3
Postgres `ts_rank`, fused with reciprocal rank fusion. Sparse is what finds `s.4.2(b)` and `OPCF 44R`.
### ⬜ 9 — Cross-encoder rerank → row 4
Rerank the fused top-20 to top-5. Bi-encoders score query and chunk independently and miss
interaction effects; a cross-encoder reads both together.
### ⬜ 10 — Definition index
Each defined term as its own retrievable unit, keyed by term — so `resolve_definition` is a
lookup rather than a similarity search.

- [ ] **Review after each:** the number, before moving on

---

# Week 3 — SHIP

**A live URL you can attach to applications this week beats a perfect system in week 5.**

### ⬜ 11 — The agent
LangGraph. Four tools, a domain control flow, one bounded retry, an explicit refusal state.

| Tool | Signature | Purpose |
|---|---|---|
| `search_corpus` | `(query, role_filter=None, doc_filter=None, k=5)` | `role_filter` is the important one |
| `resolve_definition` | `(term)` | policies redefine ordinary words |
| `fetch_neighbors` | `(chunk_id, window=1)` | adjacent clauses by `ordinal` |
| `list_documents` | `()` | lets the agent scope its own search |

```
1. DECOMPOSE           "burst pipe while away 3 weeks — am I covered?"
2. RESOLVE TERMS       resolve_definition("vacant")
3. FIND COVERAGE       search_corpus(role_filter="coverage")
4. MANDATORY EXCLUSION CHECK   <-- a graph edge, never skippable
                       search_corpus(role_filter="exclusion")
                       search_corpus(role_filter="condition")
5. SYNTHESIZE          cite every clause; state what was checked
                       low confidence ? one retry, then refuse
```

- [ ] Hard cap on total steps — six is generous. Unbounded loops read as inexperience.
- [ ] Answers state **what was checked**, not only what was found: "checked coverage s.3.1,
      exclusions s.4.2–4.7, conditions s.6" is far more trustworthy than a bare answer, and costs nothing
- [ ] Refusal is a reachable, tested state — see *Citation and disclosure contract*, state 2
- [ ] Every cited clause resolves through `doc_id` to its manifest row and renders source URL,
      consolidation date, revoked badge where applicable, and the licence line

### ⬜ 12 — API, UI, demo scenarios
- [ ] `insurance_rag/api/` — FastAPI; minimal UI (plain HTML is fine)
- [ ] `insurance_rag/citation.py` — one renderer, used by the API, the UI, and the trace log,
      so a citation cannot drift between surfaces
- [ ] Disclosure in all three places: per citation, per answer, in the chrome
- [ ] **Three pre-loaded scenarios** — a covered claim, an excluded claim, an unanswerable one
      — so a recruiter sees all three behaviours in ten seconds

### ⬜ 13 — Tests, Docker, CI
- [ ] pytest: clause-boundary parsing, locator format, `chunk_role` classification, RRF maths
- [ ] Fixtures: a few small committed PDFs so tests run with no network access
- [ ] **Agent trace assertion:** given a coverage question, assert the exclusion check actually
      ran and the step count stayed within cap. This is the test that proves the central design decision holds.
- [ ] **Refusal test** — assert refusal on the unanswerable set rather than an answer
- [ ] Dockerfile building clean from scratch; GitHub Actions running lint, tests, evals on every push

### ⬜ 14 — Deploy to AWS
- [ ] EC2 t4g.small (ARM Graviton), single AZ, **public subnet + tight security group** — no NAT
      gateway (~$32/mo to accomplish nothing on a single-box demo)
- [ ] docker-compose: app + Postgres/pgvector + Caddy for TLS
- [ ] CI → ECR via **OIDC role assumption** — no long-lived AWS keys in the repo
- [ ] **Paid plan.** The Free plan auto-closes the account at six months or when credits run
      out, whichever comes first. A demo that dies mid-job-search is worse than no demo.
- [ ] Budget alarm at $25, hard cap at $50 — set **before** deploying, not after
- [ ] Region `ca-central-1`; confirm Bedrock model availability there or call cross-region
- [ ] Embedding is a **batch job**, never on the request path. The box only embeds the incoming
      query and runs the cross-encoder, which fits comfortably in 2 GB.
- [ ] **Escape hatch:** budget one working day. If IAM or networking eats it, ship to Fly.io or
      Railway in an hour, keep applying with that URL, and migrate to AWS in week 4.
      **Missing week 3 is the real risk; deploying twice is a small cost.**

### ⬜ 15 — README
Opens with the filled ablation table, not an architecture diagram. Cost table included — a
project that documents its own running cost reads as production awareness.

**The README headline table:**

| Configuration | recall@5 single-hop | recall@5 multi-hop | exclusion recall | false-answer rate | p95 latency |
|---|---|---|---|---|---|
| Dense only, uniform 512-token chunks | | | | | |
| + clause-aware chunking | | | | | |
| + BM25 hybrid (RRF) | | | | | |
| + cross-encoder rerank | | | | | |
| + agent with coverage loop | | | | | |

---

# Week 4 — Harden

### ⬜ 16 — Generation eval + the two safety metrics
- [ ] ragas: groundedness, answer relevance, context precision
- [ ] **Exclusion recall** — of the questions where an exclusion applies, how often was it
      surfaced. A confident "yes, covered" that missed the exclusion is the worst possible failure.
- [ ] **False-answer rate** on the unanswerable set — how often it invents rather than refuses
- [ ] p95 latency and cost per query, from the trace log

### ⬜ 17 — Tracing + CI regression gate
- [ ] `insurance_rag/tracing/` — latency, tokens, retrieved chunk IDs, cost per query
- [ ] CI **fails the build** when recall or exclusion recall drops below threshold. That one
      line is worth more than the rest of the CI config: these systems regress silently.

### ⬜ 18 — SECURITY.md
- [ ] Retrieved documents are **untrusted input** — say so in writing
- [ ] **Prompt-injection test:** a document containing adversarial instructions must not change
      agent behaviour
- [ ] Domain allowlist if any fetch-by-URL feature exists — two lines, and it removes the SSRF surface

### ⬜ 19 — Retrieval failure taxonomy
Read **50 misses by hand**, categorise by root cause, fix the top two, show what moved.
This is where "tell me about a time your first approach failed" gets its answer, with numbers.

### ⬜ 20 — 90-second demo video, embedded in the README

---

# Week 5 — Differentiate

### ⬜ 21 — MCP server
Wrap `search_corpus` as an MCP tool. One artifact then evidences RAG, evaluation, testing,
deployment, agents, **and** MCP — the six things current postings ask for.

### ⬜ 22 — Web-search tier — answer state 3
Fires **only after the agent has already refused**. It cannot mask a refusal, because it
renders beneath one.

- [ ] `search_web(query)` — **regulator domain allowlist only**: `ontario.ca`, `fsrao.ca`,
      `osfi-bsif.gc.ca`, `canlii.org`. Not the open web. This is the spec's own SSRF advice, and
      it is two lines of code.
- [ ] `settings.enable_web_fallback`, **false during every eval run**. False-answer rate, the
      refusal test, and the CI regression gate must keep measuring the corpus-only behaviour.
- [ ] Results render in a visually distinct block: URL + retrieval date, **no clause locator**,
      marked unverified and explicitly *not* exclusion-checked
- [ ] Web content is untrusted input — the prompt-injection test from Step 18 extends to cover it
- [ ] New reported number: **web-fallback rate** — how often the corpus came up short
- [ ] **Review:** confirm the unanswerable set still refuses with the flag off, and still
      refuses *first* with it on

### ⬜ 23 — Confidence calibration
Reliability diagram + expected calibration error.

### ⬜ 24 — Stretch, only if time
- [ ] Long-context-versus-retrieval ablation
- [ ] **Revoked-law distractor slice** — promote O. Reg. 403/96 and Reg. 672. Both revoked
      2020-07-03, still served at live URLs with no structural marker separating them from
      current law. Near-identical wording to the current SABS with different benefit amounts.
      Correct behaviour is unambiguous: never retrieve them for a current-law question. Clean
      negative distractors, and a second angle on false-answer rate.

---

## Definition of done — v1

Not before, and **not after**. Anything beyond this list is scope creep unless it is week 5.

- [ ] A live URL a stranger can use in ten seconds without uploading anything
- [ ] README opening with the filled ablation table, not an architecture diagram
- [ ] recall@5 reported separately for single-hop and multi-hop, with the uniform baseline visible
- [ ] Exclusion recall and false-answer rate both reported
- [ ] Every answer in the UI carries a verifiable clause citation **with its source URL,
      consolidation date, and the "not an official version" line**
- [ ] Revoked source documents render a revoked badge in the citation
- [ ] "Document retrieval, not advice" stated per answer and in the interface chrome
- [ ] Passing CI badge with the eval regression gate active
- [ ] A Dockerfile that builds clean on a machine that has never seen the project
- [ ] Deployed on AWS, with the cost table and cost-control decisions in the README
- [ ] CI pushing to ECR through an OIDC role — no long-lived AWS keys in the repo
- [ ] Budget alarm set, and the account on the Paid plan so it will not auto-close
- [ ] A test asserting the exclusion check ran, and a test asserting refusal
- [ ] SECURITY.md covering prompt injection and, if applicable, the fetch allowlist
- [ ] A 90-second demo video embedded in the README

---

## Ground rules

**Ship at week 3.** The application track runs the entire time. A live URL in week 3 beats a
perfect system in week 5.

**Report measured numbers, whatever they are.** The `0.61 → 0.87` in the v1 PDF is
illustrative, not a target to reverse-engineer. This repo's value is that it is inspectable;
a fabricated table makes it worse than no repo at all.

**The golden set comes before the retrieval code**, and freezes when the corpus freezes.

**Gold labels are locators, not chunk IDs.** Chunk IDs change every time the chunker is
retuned. Locators don't.

**Keep the uniform-chunker baseline row.** Without it, every claim about clause-aware chunking
is unfalsifiable.

**The access probe defines the corpus.** HTTP 200 is not proof of a document — e-Laws proved
that. Validate content, not status codes.

**The exclusion check is a graph edge, not a prompt instruction.** If the model can skip it,
eventually it will.

**Nobody is impressed that you built a retrieval system.** They are impressed that you
*measured* one, found where it broke, fixed it, and can prove the fix. When tempted to add a
document type or polish the UI in week 4, go fill in a row of the table instead.
