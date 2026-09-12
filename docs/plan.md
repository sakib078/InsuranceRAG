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
| Retrieval | **Two bi-encoders; eval picks the winner** — `Qwen3-Embedding-0.6B` vs `bge-m3`, each with its own family cross-encoder | Both run locally. Published scores disagree, and neither was measured on regulation text — see Deviation 8. **Amended in iteration 2:** the models still run locally, but the eval harness now runs on LangSmith, so reproducing the table needs a key after all. |
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
and the corpus is 3,897 chunks over 20 documents — so carrying both costs compute we already
have, and settles the question with our own number instead of someone else's citation.

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
| **Not in corpus** (refusal) | "this corpus does not address that" | yes — **Correctness**, judged against the refusal as the reference answer |
| **Public sources say…** | visually distinct block, URL + retrieval date, **no clause locator**, marked unverified and not exclusion-checked | **no** |

State 3 layers *on top of* state 2 — the refusal still fires and is still tested. This is the
whole reason the web tier is safe to add: it cannot mask a refusal, because the refusal is
what it renders beneath.

---

# Iteration 2 — measure it, then improve it

Iteration 1 shipped a working dense-only pipeline that cannot prove anything: the ablation table
is empty and every claim in the repo is an assertion. Two failures are already measured, and both
argue for sparse + RRF + rerank — but neither can be *claimed* as a fix without a before-number,
which is why the golden set comes first.

- `"s. 31"` returns a top-5 cosine spread of **0.006** — no signal at all.
- Two paraphrases of one question returned **disjoint** evidence sets (s. 58 + s. 61 vs
  s. 5 + s. 6 + s. 37), and neither surfaced s. 31, the provision literally titled
  *Circumstances in which certain benefits not payable*.

## Decisions taken for this iteration

| | Choice |
|---|---|
| Corpus | The manifest's own v2 tier, plus Reg 676 and three OPCF endorsements — 20 documents. Inventory: `docs/sourceMap.md` |
| `insurance-act-part-vi` | Keep whole; do not slice |
| Order | Golden set and harness **before** any new retrieval code |
| Ablation | One technique per row, measured against the same frozen corpus |
| Eval platform | **LangSmith owns the eval stack** — datasets, experiments, tracing, and the retrieval metrics as custom evaluators |
| Golden set | Authored by hand, including a reference `answer` per record |

**What LangSmith does and does not buy.** Its four evaluators are all LLM-as-judge, and none of
the metrics this project reports exist there — they are custom evaluators either way. What it
buys is experiment comparison across the ablation rows, tracing with p95 latency and cost, a
pytest hook for the CI gate, and one fewer library than ragas.

**One README claim dies:** *"No API key is required to reproduce the numbers in the results
table."* A LangSmith key is now required. Same reasoning that rejected Cohere Rerank in
Deviation 6, reversed deliberately for the tooling.

**`evals/golden.jsonl` stays committed in git** and is pushed to LangSmith from there. A dataset
living only in a vendor account is not diffable, not reviewable in a pull request, and gone if
the free tier lapses. LangSmith runs the experiments; git holds the labels.

### Two consequences, accepted deliberately

1. **The revoked SABS will hurt the baseline, on purpose.** `sabs-o-reg-403-96` and `sabs-rro-672`
   are near-identical to the current SABS with different benefit amounts, served at live e-Laws
   URLs with no structural marker. Every current-law question now competes against a
   near-duplicate at almost the same cosine distance. That is the safety test the citation
   contract was designed for — but the dense-only row will read worse than iteration 1 felt.
2. **`insurance-act-part-vi` stays at 1,980 chunks** — 51% of the corpus, mostly life/fire/mutual
   licensing. No mitigation built; `search_corpus(doc_filter=...)` exists if it proves to dominate
   the candidate pool.

---

## Phase 0 — Freeze the corpus — **done, except the embed/index tail**

Chunk ids and locators are frozen at **3,897 chunks over 20 documents**. Nothing in
`insurance_rag/ingest/` changes again until Phase 3, which writes to a separate directory.

| Document | Chunks | | Document | Chunks |
|---|---|---|---|---|
| `insurance-act-part-vi` | 1,980 | | `motor-vehicle-accident-claims-act` | 130 |
| `sabs-o-reg-403-96` **revoked** | 568 | | `auto-insurance-rro-664` | 129 |
| `sabs-o-reg-34-10` | 395 | | `compulsory-auto-insurance-act` | 112 |
| `oap-1` | 221 | | `sabs-rro-672` **revoked** | 92 |
| `fault-rules-rro-668` | 64 | | `disputes-between-insurers-o-reg-283-95` | 52 |
| `court-proceedings-o-reg-461-96` | 43 | | `uninsured-auto-rro-676` | 29 |
| `fsra-minor-injury-guideline` | 24 | | `fsra-indexation-amounts` | 20 |
| `opcf-44r` | 14 | | `fsra-transportation-expense` | 13 |
| `opcf-20` | 4 | | `fsra-attendant-care-rate-revised` | 3 |
| `fsra-attendant-care-rate` | 2 | | `opcf-49` | 2 |

**What changed.** Five v2 rows promoted; four rows added (Reg 676 and OPCF 49 / 20 / 44R, which
gave `doc_type=endorsement` its first documents while Phase 7 specifies an agent that checks
endorsements). `pdf.py` gained date and cover-title rejection in `_is_provision`, so a rejected
heading with nothing above it falls back to `Preamble` instead of becoming the locator. The Colab
notebook's `max_seq_length` went 1024 → 2048: two table chunks exceed 1024 and tables are never
split, so they were being **silently truncated at embedding time** — and a coverage table
truncated from the bottom keeps its headers and loses its dollar figures.

**Still to run:** the Colab embed and `scripts/index.py`. Delete the collection before
re-indexing — ordinals shift silently. Two traps: `files.upload()` writes `chunks (1).zip` when
`chunks.zip` exists and the notebook then unzips the stale one; and `opcf-49` kept 2 chunks while
its text changed, so equal chunk counts are not proof that vectors are current.

**The e-Laws path needs no work** — `s. 31(1)` classifies `exclusion` under
`PART VII GENERAL EXCLUSIONS`, `s. 14` is `coverage`, `s. 18(1)` is `schedule`. Locators, roles
and ancestor paths are correct across all 16 HTML documents.

### Known defects, accepted and recorded

| Defect | Detail |
|---|---|
| `harvest_terms()` has no `_terms.json` | A `--doc-id` run tags different `defined_terms` than a full run. Only full runs used so far, so the corpus is self-consistent |
| `FSRA AU0129DEC s. 2026` | `_CLAUSE_RE` is tested before `_DATE_RE`, so a leading year becomes a clause number |
| `FSRA approved form OAP 1 s. Ontario Automobile Policy` | The heading is a **prefix** of the title, and the rule matches exactly |
| 27 locators resolve to 2+ chunks | `webpages.py` lacks the `(2)` disambiguator `pdf.py` has. Mostly the Act; `Reg. 676 s. 1` is the one that touches a real question |
| `AU0053ORG` / `AU0054ORG` cite `s. p. 1`–`p. 3` | Docling fails on both and `COVERAGE_FLOOR` falls back to page units |
| `AU0125ORG s. Follow FSCO on social media`, and its `(2)` duplicate sections | Web chrome, and an unresolved duplicate render |
| **Roles are weak on the PDF path** | MIG 20 `other` of 24; Reg 676 26 of 29 with zero `coverage` despite *being* the uninsured coverage schedule; `OPCF 20 s. 2` "What We Will Pay" → `exclusion` and `s. 3` "Limitations" → `schedule`, backwards. **`chunk_role` is metadata — fixing it is a re-index, not a re-embed**, so it can wait |

---

## Phase 1 — The golden set — **done**

`evals/golden.jsonl`, hand-authored, labelled by locator. `evals/validate_golden.py` resolves
every locator against `data/chunks/*.jsonl` and exits non-zero on anything unresolved, ambiguous,
sourced from a revoked document, or missing an `answer`.

```json
{
  "id": "g-014",
  "question": "is physiotherapy covered after a minor injury?",
  "hop": "multi",
  "gold_locators": ["O. Reg. 34/10 s. 18(1)", "FSRA AU0026ORG s. 1"],
  "exclusion_locators": ["O. Reg. 34/10 s. 18(1)"],
  "answerable": true,
  "answer": "Yes, but capped. Treatment for a minor injury is limited to $3,500 for medical and rehabilitation goods and services, and the Minor Injury Guideline governs what is payable within it."
}
```

- `hop` is an assertion about the **question**, not a count of labels: `single` scores ≥1 gold
  chunk in top-5, `multi` scores **all** of them, `none` marks the unanswerable slice.
- `exclusion_locators` is the subset of gold that limits or excludes; exclusion recall is computed
  only over records where it is non-empty. It is a **human judgment, never derived from
  `chunk_role`** — deriving it would measure the classifier instead of retrieval.
- `answerable: false` records carry empty `gold_locators`, and their `answer` is the refusal text,
  which is what lets Correctness score a false answer as wrong.
- Every locator is copied from a real chunk. A label that resolves to nothing scores zero forever
  and reads as a retrieval failure.

### As built — 56 records, all locators verified

| Slice | Count | Target |
|---|---|---|
| single-hop answerable | 19 | ~18 |
| multi-hop answerable | 20 | ~18 |
| unanswerable | 17 | 15–20 |
| records with non-empty `exclusion_locators` | 20 | — |

79 gold locators, all resolving, none ambiguous, none from a revoked document.

Adversarial pairs included: grant + exclusion, exclusion + the exception restoring it, benefit +
its rate guideline, OAP 1 consumer phrasing against the regulation's words, and near-miss
unanswerables where the corpus holds a tempting adjacent provision (winter tires, instalment
interest).

---
## Phase 2 — The eval harness and the baseline row

```
evals/golden.jsonl           the set from Phase 1 — committed, the source of truth
evals/validate_golden.py     locator resolver / guard, offline
evals/push_dataset.py        sync golden.jsonl -> LangSmith dataset (idempotent, by id)
evals/evaluators.py          custom evaluators: citation accuracy, exclusion recall,
                             recall@5 by hop
evals/run_eval.py            client.evaluate(target, data=..., evaluators=[...])
```

**Mapping `evals/golden.jsonl` onto a LangSmith example.** `answer` is the field that makes this
trivial: `question` becomes `inputs`, `answer` becomes `outputs.answer`, and the rest —
`gold_locators`, `exclusion_locators`, `hop`, `answerable` — ride along as example metadata the
custom evaluators read back rather than re-derive. `answer` also feeds LangSmith's
**Correctness** judge, which needs a reference to compare against.

There is no `evals/results/*.json`. Experiment results live in LangSmith, one experiment per
ablation row, named for the configuration.

**Locator → chunk id resolution.** Built once from `data/chunks/*.jsonl`, reusing
`store.read_chunks()`. A gold locator matches when `chunk.locator == gold` **or**
`chunk.locator.startswith(gold + " #")`, so oversized provisions split into ` #2`, ` #3` still
count as the same provision. This is what makes locator labels survive re-chunking, which is
the whole reason they were chosen.

### The four LangSmith built-ins

| Evaluator | Goal | Mode |
|---|---|---|
| **Correctness** | how close the answer is to ground truth | needs `outputs.answer` — the reference |
| **Relevance** | whether the answer addresses the question asked | reference-free; answer vs input |
| **Groundedness** | whether the answer agrees with the retrieved chunks — hallucination check | reference-free; answer vs retrieved docs |
| **Retrieval relevance** | whether the retrieved chunks match the query | reference-free; question vs retrieved docs |

All four are LLM-as-judge. Correctness is the one the golden set's `answer` field exists for.

### Custom evaluators — only what the product is judged on

What matters for this application is an **accurate answer** carrying an **accurate citation**.
Nothing else earns a column. Each evaluator asks one question about one record and scores it;
the experiment averages. `evals/evaluators.py`.

**1. recall@5 single — was the right clause in the top 5?**
For questions answered by a single provision.

```
"is there a waiting period before IRB starts paying?"   gold: s. 6(2)
top-5:  s. 6(1)  s. 5(1)  s. 6(2)  s. 7(1)  s. 12(1)      -> 1
                          ^ found
```

At rank 6 it scores 0, even though the system would find it eventually. The model only sees 5.

**2. recall@5 multi — were *all* the right clauses in the top 5?**
Every gold locator must be present; one missing scores 0.

```
"is physiotherapy covered after a minor injury?"   gold: s. 18(1) + s. 40(1) + MIG s. 1
top-5:  s. 40(1)  s. 3(1)  s. 40(6)  MIG s. 1  s. 41(1)   -> 0   (s. 18(1) absent)
```

Harsh on purpose. An answer built from two of three controlling provisions is incomplete, and
this is the column that exposes it — the one the Phase 7 agent exists to move.

**3. Exclusion recall — did the limit surface, not just the grant?**
Runs only on records with a non-empty `exclusion_locators`; the rest are **skipped, not zeroed**.

```
"I was drunk when I crashed - can I claim benefits?"
grant s. 14 retrieved, exclusion s. 31(1) not retrieved    -> 0
```

The safety metric. Retrieving the grant and missing the exclusion produces a confident
*"yes, you're covered"* — the most expensive way to be wrong in this domain.

**4. Citation accuracy — are the citations real, and complete?**
The only one that reads what the model *wrote* rather than what retrieval *found*. Three instant
zeros: a locator that exists nowhere in the corpus (invented), a locator that was never in the
top-5 (recalled from training, not from the documents), or a chunk from the **revoked** SABS
quoted as current law. Otherwise the score is the share of gold locators actually cited —
`cited s. 18(1)` against gold `s. 18(1) + MIG s. 1` scores 0.5, correct but half-sourced. On the
17 unanswerable records the right behaviour is to cite nothing: citing nothing scores 1.

| Evaluator | Why it cannot be a built-in |
|---|---|
| Citation accuracy | No generic evaluator knows what a locator is, or which documents are revoked. This is the product's core promise measured directly |
| Exclusion recall | Domain-specific, and it must read the **hand-written** label — computing it from `chunk_role` would measure the classifier, which is wrong on exactly the provisions that matter (`s. 6(2)`) |
| recall@5 single / multi | Ground truth beats LLM judgment where ground truth exists. The stricter sibling of Retrieval relevance, and what the ablation table is built on |

Pure set arithmetic over locators. No LLM, no cost, no key — so they stay reproducible offline
even though the experiments run hosted.

**Two implementation rules that are easy to get wrong.** A gold label matches its ` #2`
sub-chunks, so labels survive re-chunking — the corpus has been re-ingested three times without
a single label breaking. And an off-slice evaluator returns `None` rather than `0`, so LangSmith
skips the record: recall@5 single is averaged over 19 records, not 56.

<details>
<summary><b>Build plan — click to expand</b></summary>

### Two experiment types, not one

The seven evaluators need different inputs and cost wildly different amounts, so they run as two
suites rather than one.

| Suite | Evaluators | LLM calls per run | Run it for |
|---|---|---|---|
| **retrieval** | recall@5 single, recall@5 multi, exclusion recall | **0** | every ablation row |
| **generation** | Correctness, Relevance, Groundedness, Retrieval relevance, citation accuracy | ~56 generations + ~224 judge calls | only rows where generation changed |

Phases 3–5 change retrieval alone. Judging those with four LLMs is ~1,100 calls to measure
something that cannot have moved. The retrieval suite is set arithmetic over locators — free,
instant, no key — so it runs on everything, and the retrieval columns of the ablation table stay
ground-truth-scored rather than LLM-opinion-scored.

### The target function's output contract

Everything else depends on this shape, because every evaluator reads from it.

```python
{
  "answer":             str,        # Correctness, Relevance, Groundedness
  "retrieved_locators": list[str],  # rank order - recall@5, exclusion recall
  "retrieved_text":     list[str],  # Groundedness, Retrieval relevance
  "cited_locators":     list[str],  # citation accuracy
}
```

The retrieval target returns only the `retrieved_*` fields and never calls an LLM.

**This requires one change to `chain.py`:** `Answer.chunks` has already been filtered by `cited()`
to what the answer referenced, so the *retrieved* set is discarded before it returns. `Answer`
gains a `retrieved` field alongside `chunks` — citation accuracy needs both (was every cited
locator actually retrieved?) and recall@5 needs the full ranked list.

### Files

```
evals/push_dataset.py   golden.jsonl -> LangSmith dataset, idempotent by record id
evals/evaluators.py     recall_single, recall_multi, exclusion_recall, citation_accuracy
evals/run_eval.py       --suite retrieval|generation  --config <ablation row name>
```

Plus `LANGSMITH_API_KEY` and `LANGSMITH_TRACING` into `.env` / `.env.example`, and the README's
no-API-key claim corrected.

**Citation accuracy** carries the only real logic. Four ways an answer cites badly:

1. cites a locator resolving to no chunk — fabricated
2. cites a locator that was never retrieved — fabricated differently
3. cites a `status=REVOKED` chunk as current law
4. fails to cite a gold locator it was given

1–3 score as hard failures, 4 as the recall half; reported as one number.

### Settled

**Judge model — a different provider and a different family from the generator.** The generator
is `openai/gpt-oss-120b` on Groq. A model judging its own family's output has an obvious
self-preference problem, and a judge on the same provider shares the same rate-limit bucket —
so a second provider fixes both at once. Groq's own `qwen/qwen3.8-27b` is the fallback if only
one key is wanted; it fixes the family problem but not the bucket.

**Rate limits.**

- `max_concurrency` held low (2–4) rather than letting LangSmith fan out.
- Exponential backoff on HTTP 429, with jitter.
- **`evaluate_existing` over `evaluate`** wherever the target has already run. Adding or swapping
  a judge then re-scores stored outputs instead of regenerating 56 answers — which is what makes
  iterating on the judges affordable.
- The retrieval suite is unaffected; it makes no calls at all.

**The retry ladder stays on for generation, and is pinned off for retrieval.** `chain.LADDER`
escalates k on a refusal, so "recall@5" measured through it would not be at k=5. Retrieval
experiments pin `k=5` to measure the stated metric honestly; generation experiments leave the
ladder on to measure the product as shipped. Two different questions, deliberately two settings.

### Order

1. `Answer` gains `retrieved`.
2. `evaluators.py` — the three custom retrieval evaluators, with unit tests over known cases.
3. `push_dataset.py`; verify 56 examples land with their metadata intact.
4. `run_eval.py --suite retrieval --config dense_clause_aware` — **the baseline row**.
5. Wire the built-ins, add citation accuracy, run the generation suite.

Steps 1–4 are built to run offline as well as through `client.evaluate()`, so the baseline exists
even if the key or the free tier becomes a problem.

</details>

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
   drops**. Spec: worth more than the rest of the CI config combined. Runs through LangSmith's
   pytest integration, which means **CI needs `LANGSMITH_API_KEY` as a repository secret** — and
   a fork's pull request cannot see it. Decide then whether the gate runs on forks at all, or
   whether `validate_golden.py` (offline) is the only check a fork gets.
3. Agent trace assertion (the exclusion check actually ran) and a refusal test.
4. Prompt-injection test — retrieved documents are untrusted input. `SECURITY.md`.
5. FastAPI + minimal UI, three pre-loaded demo scenarios.
6. Dockerfile, docker-compose, GitHub Actions.
7. Deploy — **Azure vs AWS is still the open decision above; this plan does not resolve it.**

### Dependencies added, by phase

| Phase | New |
|---|---|
| 0–1 | none |
| 2 | `langsmith` — and `ragas` is dropped from the plan, its metrics covered by LangSmith |
| 3 | none |
| 4 | none — psycopg is already installed |
| 5 | `sentence-transformers` (likely already present via `langchain-huggingface`) |
| 7 | `langgraph` |
| 8 | `fastapi`, `uvicorn`, `pytest` |

`LANGSMITH_API_KEY` and `LANGSMITH_TRACING` join `.env` and `.env.example` at Phase 2. Tracing is
env-var driven, so `chain.py` needs no code change to be traced.

---

## End-to-end verification

The README table filled, every row from `evals/results/*.json`:

| Configuration | correctness | citation accuracy | exclusion recall | recall@5 single | recall@5 multi |
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
