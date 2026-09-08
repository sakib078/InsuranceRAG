# Agentic Insurance RAG — Decisions

The decisions this project is built on, and the reasoning behind each. Corpus and access
findings: `docs/sourceMap.md`. Spec: `docs/agentic_insurance_rag_spec.pdf` (v2).
Iteration 1's build record: `docs/RAG_pipleline.md`. Iteration 2's build steps are at the
bottom of this file.

---

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Corpus | Ontario **auto only** — e-Laws HTML backbone + hand-downloaded FSRA policy forms | Auto has a *regulator-published* standard policy (OAP 1). Home does not — see Deviation 1. |
| Storage | **One Postgres 16** — pgvector + tsvector/GIN | Chunk text, embeddings, and full-text index in one store. "One Postgres over a vector DB plus a search engine" is a real architecture answer. |
| Ingestion | e-Laws pre-render (HTML) + PyMuPDF (PDF) | Two dialects, one chunk schema. |
| Agent | LangGraph, exclusion check as a **graph edge** | If the model can skip the exclusion check, eventually it will. |
| Retrieval | **Two bi-encoders; eval picks the winner** — `Qwen3-Embedding-0.6B` vs `bge-m3`, each with its own family cross-encoder | Both run locally, so no API key is needed to reproduce the table. Published scores disagree, and neither was measured on regulation text — see Deviation 8. |
| Eval order | Golden set **before** any retrieval code | Written after, it tests what you happened to build. Iteration 1 knowingly inverted this to get one end-to-end path running; iteration 2 restores it. |
| Gold labels | Clause **locators**, resolved to chunk IDs at eval time | Survives re-chunking, the agent, and corpus growth. |

## Open decisions

**Deployment target — Azure Container Apps or AWS EC2.** Deliberately open. Both host the same
docker-compose stack; the difference is what the artifact demonstrates and what it costs to
keep alive during a job search.

| | Azure Container Apps | AWS EC2 t4g.small |
|---|---|---|
| Cost | Free grant, scale-to-zero | ~$22–27/mo, always on |
| Ops surface | Managed; no box to patch | Full control; you run the box |
| Postgres | Needs a managed DB or a sidecar with a volume | Runs in the same compose file |
| Cold start | Real, on scale-to-zero — bad for a recruiter clicking a link | None |
| Signals | Containers, managed platform | AWS + Docker + CI/CD, currently claimed with no artifact behind them |

Decided by: whether the demo must answer instantly on a cold click, and whether the AWS
signal is worth the monthly cost. Not blocking — everything upstream of deployment is
identical either way.

---

## Deviations from the spec — deliberate, with reasons



**2. `doc_type` extended.** The spec's literals are `policy | endorsement | bulletin | guide`.
The scriptable backbone is regulations and statutes, which fit none of them. Extended to
`policy | endorsement | bulletin | guide | regulation | statute`.

**5. `token_count` uses one designated reference tokenizer, not tiktoken.** The spec says
tiktoken; that is OpenAI's BPE and mis-counts for both encoders under test. It cannot be "the
embedding model's tokenizer" either, now that there are two — chunk boundaries must be
**identical** across both, or the bake-off in Deviation 8 confounds chunking with encoding.
`Qwen3-Embedding-0.6B`'s tokenizer is the reference, `token_count` targets the 400–800 band
against it, and both encoders then embed byte-identical chunks.

**6. A local cross-encoder, not Cohere Rerank.** The spec offers either. Cohere needs an API
key, which breaks "clone the repo and reproduce the retrieval table." Each bi-encoder is
paired with its own family's reranker — `Qwen3-Reranker-0.6B` or `bge-reranker-v2-m3` — so the
comparison is family against family, never a mixed pipeline.

**7. A web-search tier, beyond the spec — late, and outside the measured path.** See the
citation contract below. It sits after refusal, never inside it, and is disabled during every
eval run.

**8. Two bi-encoders carried into the ablation, not one.** The spec assumes a single embedding
model. Published scores put `Qwen3-Embedding-0.6B` at **61.83** on MTEB English v2 Retrieval.
BAAI publishes no comparable figure for `bge-m3`; its aggregate (~63.0) sits just under
`bge-large-en-v1.5` (64.23 aggregate, 54.29 Retrieval), which suggests m3 trails by roughly
7–11 points on English. **That is an inference, not a measurement**, and none of those numbers
were taken on regulation text. Both models are ~0.6B and ~560 MB at int8, both fit a 2 GB box,
and the corpus is ~2,900 chunks, heading to ~3,900 after the iteration-2 promotion — so
carrying both costs compute we already have, and settles the question with our own number
instead of someone else's citation.

`bge-large-en-v1.5` is excluded on a measurement rather than a preference. **38% of SABS
sections exceed its 512-token cap, and those sections hold 76% of the text**: s.3(1)
definitions runs 3,417 tokens, General Exclusions 1,553, and the median `"does not apply"`
section 671. A 512 ceiling makes "one clause, one chunk" unimplementable across the 400–800
band this project exists to demonstrate — and it would truncate the clause-aware rows while
leaving the 512-token baseline row intact, handicapping the arm under test.

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
| **Public sources say…** | visually distinct block, URL + retrieval date, **no clause locator**, marked unverified and not exclusion-checked | **no** |

State 3 layers *on top of* state 2 — the refusal still fires and is still tested. This is the
whole reason the web tier is safe to add: it cannot mask a refusal, because the refusal is
what it renders beneath.

---

# Iteration 2 — measure it, then improve it

## Where this starts

Iteration 1 is done: ingestion → pgvector → cited answer runs end to end, dense-only.
`scripts/ask.py` answers with real clause citations and refuses when the corpus does not
address a question. That is the whole of the pipeline that exists.

**What it cannot do is prove anything.** The README ablation table is empty, `evals/` holds
one `.gitkeep`, and `tests/` holds one `__init__.py`. Every claim in the repo is currently an
assertion. Spec v2's Definition of Done turns on numbers this project has not produced:
recall@5 split single-hop / multi-hop, exclusion recall, false-answer rate, a visible
uniform-chunker baseline, and a CI gate that fails when they regress.

Two failures are already measured and both point the same way:

- `"s. 31"` returns a top-5 cosine spread of **0.006** — no signal at all.
- Two paraphrases of one question returned **disjoint** evidence sets (s. 58 + s. 61 vs
  s. 5 + s. 6 + s. 37), and neither surfaced s. 31, the provision literally titled
  *Circumstances in which certain benefits not payable*.

Both are the case for sparse + RRF + rerank. Neither can be *claimed* as a fix without a
before-number, which is why the golden set comes first.

## Decisions taken for this iteration

| | Choice |
|---|---|
| Corpus | Promote the manifest's own v2 tier — 5 rows that already carry researched notes |
| `insurance-act-part-vi` | Keep whole; do not slice |
| Order | Golden set and harness **before** any new retrieval code |
| Ablation | One technique per row, measured against the same frozen corpus |

### Two consequences, accepted deliberately

1. **The revoked SABS will hurt the baseline, on purpose.** `sabs-o-reg-403-96` and
   `sabs-rro-672` are near-identical in wording to the current SABS with different benefit
   amounts, served by e-Laws at live URLs with no structural marker. Adding them before the
   golden set means every current-law question now competes against a near-duplicate at almost
   the same cosine distance. That is the point — it is the safety test the citation contract
   above was designed for — but the dense-only baseline row will read worse than iteration 1
   felt. Measure it, record it, then fix it in Phase 4.

2. **`insurance-act-part-vi` stays at ~1,979 chunks**, roughly two-thirds of the corpus and
   mostly life/fire/mutual licensing. No mitigation is built; `search_corpus(doc_filter=...)`
   already exists if a later measurement shows it dominating the candidate pool.

---

## Phase 0 — Freeze the corpus

Nothing downstream can start until chunk locators stop moving. Gold labels are locators, so
any ingest change after Phase 1 invalidates hand-written labels.

### Step 0.1 — Promote the v2 rows

`data/manifest.csv`, flip `phase` `v2` → `v1` on five rows:

| doc_id | Why it earns a place |
|---|---|
| `disputes-between-insurers-o-reg-283-95` | lexical variety; insurer-facing register |
| `compulsory-auto-insurance-act` | the requirement to insure, evidence of insurance |
| `motor-vehicle-accident-claims-act` | uninsured / unidentified motorist fund |
| `sabs-o-reg-403-96` | **revoked** — negative distractor |
| `sabs-rro-672` | **revoked** — negative distractor |

Their `sha256` and `retrieval_date` columns are empty and must be filled by the fetch.

### Step 0.2 — Fix the locator bugs first

These are ingest bugs that produce unciteable locators. Fixing them *after* the golden set
breaks every label that touches them.

- `insurance_rag/ingest/pdf.py` — FSRA guidelines emit title-as-locator:
  `FSRA AU0026ORG s. February 2014`, `s. Minor Injury Guideline (2)`,
  `s. Appendix B Minor Injury Guideline Fee Schedule`. A date is not a provision path.
  Suppress heading text that is a date or repeats the document title; fall back to an
  ordinal path (`s. 1`, `s. 2`) so `LOCATOR_RE` still passes and the locator is stable.
- `insurance_rag/ingest/roles.py` — `harvest_terms()` runs over whatever the invocation
  loaded, so a `--doc-id` run tags different `defined_terms` than a full run. Persist to
  `data/chunks/_terms.json` on a full run and read it back otherwise.

### Step 0.3 — Re-fetch, re-ingest, re-embed, re-index

```
python -m scripts.fetch_corpus                 # picks up the 5 promoted rows
python -m scripts.ingest                       # full run, so _terms.json is authoritative
# notebooks/embed_colab.ipynb  -> only the 5 new .npz; existing ones stay valid
python -m scripts.index --embeddings data/embeddings
```

Existing documents' chunk text does not change, so their cached `.npz` remain correct. Delete
the collection before re-indexing anyway — ordinals shift silently, and `oap-1:0042` meaning a
different provision is the failure mode that cost a day in iteration 1.

**The Colab upload trap:** `files.upload()` writes `chunks (1).zip` when `chunks.zip` already
exists, and the notebook then unzips the stale file. Add `rm -f chunks*.zip` before the upload
cell.

### Verify

- The manifest loads and reports the new row count.
- Per-document chunk counts are recorded in `docs/RAG_pipleline.md`.
- No locator in `data/chunks/*.jsonl` matches `s. (January|…|December) \d{4}`.
- `python -m insurance_rag.retrieval.search "income replacement benefit" -k 10` returns hits
  from **both** SABS versions — confirming the distractor is live and the problem is real.

### Exit criteria

Chunk ids and locators are frozen. Nothing in `insurance_rag/ingest/` changes again until
Phase 3, which writes to a separate directory and does not disturb these.

---

## Phase 1 — The golden set

The gate. `evals/golden.jsonl`, 40–60 pairs, hand-written, labelled by locator.

| Slice | Count | Definition |
|---|---|---|
| single-hop | ~18 | answer lives in one clause |
| multi-hop | ~18 (≥ one third) | coverage + an exclusion, or + a definition, or a benefit + its rate guideline |
| unanswerable | 15–20 | plausible, on-topic, corpus genuinely does not address |

```json
{
  "id": "g-014",
  "question": "is physiotherapy covered after a minor injury?",
  "hop": "multi",
  "gold_locators": ["O. Reg. 34/10 s. 18(1)", "FSRA AU0026ORG s. 1"],
  "exclusion_locators": ["O. Reg. 34/10 s. 18(1)"],
  "answerable": true,
  "notes": "the $3,500 cap is the answer; s.40 fee mechanics are adjacent, not the answer"
}
```

- `exclusion_locators` is the subset of gold that limits or excludes. Empty when none applies.
  **Exclusion recall is computed only over records where this is non-empty.**
- `answerable: false` records carry empty `gold_locators` and drive false-answer rate.
- Every locator is copied from an actual chunk in `data/chunks/*.jsonl`, never typed from
  memory — a gold label that resolves to nothing silently scores zero forever.

LAT/AABS decisions are read by hand for realistic question phrasing, per the
`lat-aabs-decisions` manifest row. Never ingested.

### Adversarial pairs to include deliberately

- A grant and the exclusion that voids it (SABS s. 14 / s. 31).
- An exclusion and the exception that restores it.
- A benefit and the guideline that sets its number (attendant care → AU0053 vs AU0054 —
  **decide which governs before writing the label**, per that row's manifest note).
- A current-law question whose highest-similarity match sits in the **revoked** SABS.
- OAP 1 phrased as a consumer would ("who and what we won't cover") vs the regulation's words.

### Verify

`evals/validate_golden.py` resolves every locator against `data/chunks/*.jsonl` and exits
non-zero on any that matches no chunk. Run before a single metric is computed.

### Exit criteria

Every locator resolves. All three slices populated. Committed **before** Phase 2 code exists.

---

## Phase 2 — The eval harness and the baseline row

```
evals/golden.jsonl           the set from Phase 1
evals/validate_golden.py     locator resolver / guard
evals/run_retrieval.py       recall@5 split by hop, exclusion recall, p95 latency
evals/run_generation.py      false-answer rate, ragas — after retrieval is settled
evals/results/{config}.json  one file per ablation row
```

**Locator → chunk id resolution.** Built once from `data/chunks/*.jsonl`, reusing
`store.read_chunks()`. A gold locator matches when `chunk.locator == gold` **or**
`chunk.locator.startswith(gold + " #")`, so oversized provisions split into ` #2`, ` #3` still
count as the same provision. This is what makes locator labels survive re-chunking, which is
the whole reason they were chosen.

| Metric | Definition |
|---|---|
| recall@5 single-hop | ≥1 gold chunk in top-5 |
| recall@5 multi-hop | **all** gold chunks in top-5 — the number that exposes the real failure |
| exclusion recall | ≥1 `exclusion_locators` chunk in top-5, over records where it is non-empty |
| revoked-leak rate | share of current-law questions with a `status=REVOKED` chunk in top-5 |
| p95 latency | from the trace log |

`revoked-leak rate` is beyond the spec. It exists because Phase 0 deliberately introduced the
distractor, and an unmeasured distractor is just noise.

Produces `evals/results/dense_clause_aware.json` — the current pipeline, unchanged. This is the
before-number every later phase is measured against. Multi-hop recall is expected to be poor.
Record it; do not fix it here.

---

## Phase 3 — The uniform-512 baseline row

Definition of Done requires the uniform-chunker baseline visible, to prove clause-aware
chunking paid for itself. `settings.uniform_chunk_tokens` (512) and `uniform_overlap_tokens`
(64) already exist and are unused.

- `scripts/ingest.py --uniform` → `data/chunks_uniform/`, fixed windows, no structure
  awareness. Locator degrades to `<citation> s. p. N #k` — still satisfies `LOCATOR_RE`, still
  human-checkable, deliberately worse.
- `store.py` — collection name gains a suffix so both indexes coexist, the same way the
  encoder bake-off already works.
- One Colab run, one index run. Produces `evals/results/dense_uniform.json`.

**Expect gold locators to resolve poorly against uniform chunks.** That is the finding, not a
bug: fall back to substring containment of the gold chunk's text for this row only, and say so
in the results file.

---

## Phase 4 — Sparse retrieval and RRF

The fix for `"s. 31"` and for the disjoint-paraphrase result.

1. **tsvector + GIN.** `langchain-postgres` stores chunk text in
   `langchain_pg_embedding.document`. Add a generated tsvector column and a GIN index over it —
   one idempotent migration, `scripts/migrate_fts.py`. This is the "one Postgres, not a vector
   DB plus a search engine" claim in the locked decisions becoming true.
2. **`insurance_rag/retrieval/sparse.py`** —
   `search_sparse(query, k=settings.sparse_top_k) -> list[tuple[str, float]]`, `ts_rank` over
   the same table via the psycopg connection already configured.
3. **Fusion inside `search_corpus`.** `settings.rrf_k` (60), `fusion_top_k` (20) and
   `sparse_top_k` (20) already exist — **no new config keys**. The signature does not change;
   this is the seam it was built for. Keep the channel label on each hit (dense / sparse /
   both) — losing it discards information the reranker and the agent both want.

### Verify

- `python -m insurance_rag.retrieval.search "s. 31"` puts s. 31 in the top 5.
- The two paraphrases from iteration 1 return **overlapping** evidence.
- `evals/results/hybrid_rrf.json` — recall up on both slices, or the technique did not work and
  that is the finding.

---

## Phase 5 — Cross-encoder rerank

- `insurance_rag/retrieval/rerank.py`. `settings.cross_encoder_model` already resolves to
  `Qwen3-Reranker-0.6B` or `bge-reranker-v2-m3` by encoder family. Never mix families
  (Deviation 6).
- Reranks `fusion_top_k` (20) down to `rerank_top_k` (5), inside `search_corpus`.
- Runs on CPU at query time — one query against 20 passages, not a batch job. Measure p95; if
  it exceeds ~2s that is a real finding for the README cost section.
- **The refusal gate belongs here, not earlier.** Iteration 1 could not set a distance
  threshold because a correct hit sat at 0.532 while noise sat at 0.67. A cross-encoder
  produces a calibrated relevance score instead, which is a threshold that can actually hold.
  `chain.py`'s `LADDER` then escalates on low confidence, not only on refusal text.
- Produces `evals/results/hybrid_rerank.json`.

---

## Phase 6 — Definition index

Spec v2 §04 lists it as a separate retrieval strategy, and `resolve_definition` is one of the
agent's four tools. Ingestion already emits `chunk_role=definition` and per-term units keyed
`s. 3(1) "accident"`, so most of the work is done.

`resolve_definition(term)` in `search.py` — exact lookup on the definition slice, not
similarity search. Feeds the agent in Phase 7. No ablation row of its own.

---

## Phase 7 — The agent

LangGraph, per spec v2 §05. Four tools: `search_corpus`, `resolve_definition`,
`fetch_neighbors`, `list_documents`.

**The mandatory exclusion check is a graph edge, not a prompt instruction** — the locked
decision above. The reason is now measured in this repo: the model answered a minor-injury
question from s. 40 fee mechanics without ever seeing the s. 18(1) cap.

- Bounded: one reformulation retry, `settings.max_agent_steps` (6) hard cap. Both exist.
- Output states what was checked, not only what was found.
- Produces `evals/results/agent.json` — the last ablation row, and the one the multi-hop column
  exists to justify.

---

## Phase 8 — Testing, CI, serving, deploy

Per spec v2 §07–08, in this order:

1. `pytest` — clause-boundary parsing, locator format, `chunk_role` classification, RRF maths.
2. Golden-set regression test + **CI gate that fails the build when recall or exclusion recall
   drops**. Spec: worth more than the rest of the CI config combined.
3. Agent trace assertion (the exclusion check actually ran) and a refusal test.
4. Prompt-injection test — retrieved documents are untrusted input. `SECURITY.md`.
5. FastAPI + minimal UI, three pre-loaded demo scenarios.
6. Dockerfile, docker-compose, GitHub Actions.
7. Deploy — **Azure vs AWS is still the open decision above; this plan does not resolve it.**

### Dependencies added, by phase

| Phase | New |
|---|---|
| 0–3 | none |
| 4 | none — psycopg is already installed |
| 5 | `sentence-transformers` (likely already present via `langchain-huggingface`) |
| 7 | `langgraph` |
| 8 | `fastapi`, `uvicorn`, `ragas`, `pytest` |

---

## End-to-end verification

The README table filled, every row from `evals/results/*.json`:

| Configuration | recall@5 single | recall@5 multi | exclusion recall | false-answer | p95 |
|---|---|---|---|---|---|
| Dense only, uniform 512 | | | | | |
| + clause-aware chunking | | | | | |
| + sparse hybrid (RRF) | | | | | |
| + cross-encoder rerank | | | | | |
| + agent with coverage loop | | | | | |

Plus, per the Definition of Done: a live URL, a passing CI badge with the regression gate
active, every answer carrying a verifiable clause citation, `SECURITY.md`, and a 90-second
demo video.

---

## What this iteration deliberately excludes

The national corpus expansion in
`docs/Canadian Auto-Insurance Corpus and Contradiction-Aware Retrieval Research.md` — Québec
QPF/QEF, Alberta SPF, BC ICBC, Atlantic, A2AJ case law. It is a good document and a plausible
next project, but it is **not an increment on this one**: its data model needs
`authority_level`, `effective_from`, `effective_to`, `supersedes` and a `MODIFIES` /
`SUPERSEDES` relationship graph, none of which `Chunk` has, plus jurisdiction routing and
authority-ranked reranking. It also makes a hand-labelled golden set impractical, which removes
the one thing that makes this repository worth showing.

Three of its ideas are worth taking without the corpus, and two are already above: per-channel
retrieval labels (Phase 4), authority as retrieval metadata (partly present as `status` /
`is_official`), and the structured `grant → exclusion → exception → regulation` answer bundle
(Phase 7's output contract).
