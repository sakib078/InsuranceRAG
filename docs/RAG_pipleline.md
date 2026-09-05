# First RAG iteration — three phases

## Context

`insurance_rag/ingest/pdf.py` and `webpages.py` exist but are empty; `retrieval/` and
`generation/` are bare `__init__.py`. Everything upstream is done: `schema.py` defines the
`Chunk` contract, `corpus/manifest.py` loads the 11-row v1 manifest, `config.py` carries the
encoder specs and chunking knobs, `scripts/fetch_corpus.py` fetches e-Laws.

Goal: a working non-agentic RAG pipeline — question in, cited answer out — built in three
phases with clean seams. No agent, no eval harness, no hybrid retrieval, no reranker in this
iteration; each of those slots into a named place below. Strictly LangChain ecosystem.

Locked from `docs/plan.md`: Postgres 16 + pgvector as the store, Qwen3-Embedding-0.6B's
tokenizer as the reference for `token_count`, citations render full provenance.

## Decisions taken in this session

| | Choice |
|---|---|
| Web | `WebBaseLoader` live against `source_url` — no per-page code changes |
| PDF | `DoclingLoader` primary; `PyMuPDFLoader` alongside for page numbers + scanned-page flag |
| Splitter | Structure split first, `RecursiveCharacterTextSplitter` as the size enforcer |
| Store | `langchain-postgres` `PGVector`, dense only; tsvector/GIN sibling table deferred |
| Phases | 1 ingestion → 2 retrieval → 3 generation |
| Short chunks | Prepend a context header to the embedded text; never merge provisions |

### Two things flagged before starting

1. **Live `WebBaseLoader` makes runs network-dependent.** Mitigated: the fetched HTML is
   written to `data/raw_html/{doc_id}.html` and read from there unless `--refresh` is passed.
   Keeps the manifest's `sha256` meaningful.
2. **`RecursiveCharacterTextSplitter` alone cannot satisfy `Chunk`.** `LOCATOR_RE` demands
   `"<cite> s. <path>"`, so a blind character split fails `__post_init__` on every chunk. It is
   used as the *second* stage only — see Phase 1, Step 3.

---

## Phase 1 — Ingestion — built

Output: `data/chunks/{doc_id}.jsonl`, one validated `Chunk` per line. Nothing touches Postgres.

### Step 1 — `insurance_rag/ingest/webpages.py`

```
load_web(row: ManifestRow, *, refresh: bool = False) -> list[Document]
```

e-Laws markup is fully structured, so the DOM is walked directly rather than regexed over
flattened text. `p.section-e` / `p.subsection-e` / `p.definition-e` (and the un-suffixed
statute variants `section`, `subsection`, plus the `S`/`Y` prefixed forms) open a new unit;
`p.clause-e`, `p.paragraph-e`, `p.subpara-e` and friends append to the open one.

- `partnum` / `heading1` / `heading2` / `headnote` occupy four ancestor slots, each one
  clearing the deeper slots — that is what makes `ancestor_path` correct rather than cumulative.
- The table of contents **is itself a `<table>`**, and its rows carry `p.table-e` like real
  data tables do. Tables containing a `p[class^="TOC"]` are dropped; the rest are emitted whole
  as pipe-delimited markdown and never split.
- `WebBaseLoader` fetches live and caches to `RAW_HTML_DIR`; the cache is read on later runs.

**Granularity: one unit per subsection, not per clause.** A clause read alone
(`"(a) the named insured;"`) is meaningless — it needs its subsection stem. Each defined term
gets its own unit, keyed `s. 3(1) “accident”`, which is the spec's separate definition index.

### Step 2 — `insurance_rag/ingest/pdf.py`

```
load_pdf(row: ManifestRow) -> list[Document]
```

- `DoclingLoader(export_type=MARKDOWN)` — layout-aware, so OAP 1's two columns do not
  interleave and its coverage tables survive whole. `manifest.csv` records the column x-starts
  that make naive extraction fail.
- `PyMuPDFLoader` alongside, for two things only: a page-number lookup for `Chunk.page`, and
  flagging pages under `SCANNED_PAGE_CHARS` (Reg 668's ~40 diagram images) so they are skipped
  rather than indexed as empty chunks.
- `MarkdownHeaderTextSplitter` over Docling's markdown splits on `#`…`####`. OAP 1's 164-entry
  embedded TOC puts the clause number in the heading (`"1.8 Who and What We Won't Cover"`), so
  the locator is read, not guessed. FSRA guidelines have no numbering and fall back to the
  heading text, which still satisfies `LOCATOR_RE`.

### Step 3 — `insurance_rag/ingest/split.py`

**Stage A** is Steps 1–2 above: they produce the locator and the ancestor path.

**Stage B — size enforcement**

```python
RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    reference_tokenizer(),                                   # Qwen3, Deviation 5
    chunk_size=settings.max_chunk_tokens - HEADER_ALLOWANCE, # 800 - 64
    chunk_overlap=OVERLAP_TOKENS,                            # 120, ~15%
    separators=["\n\n", "\n", ". ", "; ", " ", ""],
)
```

- Applied **only to units over the budget**. Sub-chunks inherit the parent locator with a
  ` #2`, ` #3` suffix, so a citation still resolves to one provision.
- Tables are never passed to Stage B — a table split across three chunks is unretrievable.
- Units under `min_chunk_tokens` are **not merged with a neighbour**. Merging two provisions
  into one chunk breaks the one-provision-one-citation contract the project rests on.

**The context header.** Measured on the 5 e-Laws documents: 2,595 of 2,610 chunks fall under
the 400-token floor — subsection-level provisions are simply short (s.31(1) is ~90 tokens), and
short text embeds poorly. Rather than merge, each chunk's embedded text is prefixed with its
own provenance and heading trail:

```
Statutory Accident Benefits Schedule (O. Reg. 34/10) — PART VII GENERAL EXCLUSIONS > Circumstances in which certain benefits not payable
O. Reg. 34/10 s. 31(1):
The insurer is not required to pay an income replacement benefit, …
```

The chunk stays one provision, the locator stays exact, and the bi-encoder gets ~40 tokens of
context it otherwise had none of. `HEADER_ALLOWANCE` reserves room for it inside the ceiling.

### Step 4 — `insurance_rag/ingest/roles.py`

`classify_role(text, ancestor_path, locator)` — rules only, no LLM pass in this iteration.
Definition first (a `“term” means` match or a quoted locator), then heading match, then trigger
phrases. Per `manifest.csv`, OAP 1 uses the word "exclusion" **twice in 68 pages**, so the
exclusion triggers are `"won't cover"`, `"not covered"`, `"does not apply"`,
`"is not required to pay"` — never the literal word alone.

`harvest_terms()` collects every term the corpus defines for itself; `defined_terms_in()` tags
each chunk with the ones it mentions.

### Step 5 — `scripts/ingest.py`

`python -m scripts.ingest [--doc-id X] [--refresh]`. Iterates the v1 manifest rows, routes on
`row.access`, builds `Chunk` objects (which self-validate), writes JSONL, and prints a
per-document count, median/max token count, and role histogram.

**Dependencies added:** `langchain-core`, `langchain-community`, `langchain-text-splitters`,
`langchain-docling` (which pulls `docling`, `pypdfium2`, `torchvision`, `rapidocr` and
downloads ~500 MB of layout models on first run).

### Phase 1 results

| Document | Chunks | Median tokens | Max |
|---|---|---|---|
| `sabs-o-reg-34-10` | 395 | 148 | 773 |
| `auto-insurance-rro-664` | 129 | — | 749 |
| `fault-rules-rro-668` | 64 | — | 305 |
| `court-proceedings-o-reg-461-96` | 43 | — | 519 |
| `insurance-act-part-vi` | 1,979 | — | 827 |
| `fsra-minor-injury-guideline` | 52 | 130 | 659 |

Verified: `s. 31(1)` lands under `PART VII GENERAL EXCLUSIONS` and classifies `exclusion`;
`s. 3(1) “accident”` classifies `definition` and carries its own defined terms.

**Known open items.**

- `insurance-act-part-vi` produces 1,979 chunks because the fetch returns the whole Act, not
  just Part VI. This is the pre-existing `BLOCKED ON STEP 3` note in `manifest.csv`.
- `harvest_terms` runs over whatever documents the invocation loaded, so a `--doc-id` run tags
  different `defined_terms` than a full run. Fix: persist the term set to
  `data/chunks/_terms.json` on a full run and reuse it.

---

## Phase 2 — Retrieval

### Step 1 — Postgres

`docker-compose.yml` with `pgvector/pgvector:pg16`, matching `settings.postgres_dsn`.

### Step 2 — `insurance_rag/retrieval/store.py`

- `embeddings()` → `HuggingFaceEmbeddings(model_name=settings.bi_encoder_model)`.
- `vector_store()` → `PGVector(collection_name=settings.encoder, connection=settings.postgres_dsn)`.
  The collection is keyed by encoder so the Deviation 8 bake-off can hold both indexes side by
  side over byte-identical chunks.
- `add_chunks(chunks)` → `Document(page_content=chunk.text, metadata={every other field})`,
  `ids=[c.chunk_id]` so re-ingest upserts instead of duplicating.

### Step 3 — `insurance_rag/retrieval/search.py`

```
search_corpus(query, *, role_filter=None, doc_filter=None, k=settings.rerank_top_k) -> list[Chunk]
```

Dense-only for this iteration: `similarity_search_with_score` plus a PGVector metadata filter on
`chunk_role` / `doc_id`. Returns `Chunk` objects, not `Document`s, so Phase 3 and the future
agent share one type.

**This is the seam for everything deferred.** Sparse `ts_rank` candidates, RRF fusion, and the
cross-encoder rerank all land inside `search_corpus` without changing its signature.

### Step 4 — `scripts/index.py`

Reads `data/chunks/*.jsonl`, calls `add_chunks`. Idempotent.

**Dependencies:** `langchain-postgres`, `langchain-huggingface`.

**Verify:** index one document, then query `"minor injury"` and `"s. 31"`. The second scoring
badly is expected, and is exactly the evidence for adding sparse later — record it, don't fix it.

---

## Phase 3 — Generation — built

### Step 1 — `insurance_rag/generation/citations.py`

`render_citation(chunk, manifest_index)` using `by_doc_id()`, emitting the block fixed in
`docs/plan.md`:

```
O. Reg. 34/10, s. 31(1)(a) — Statutory Accident Benefits Schedule
Consolidated 2026-07-01 · ontario.ca/laws/regulation/100034
Not an official version.
```

`status == REVOKED` renders a revoked badge — a safety field, not bookkeeping. FSRA rows link,
never rehost. `licence_lines()` emits one note per distinct source behind an answer.

### Step 2 — `insurance_rag/generation/chain.py`

LCEL: `search_corpus` → `format_context` (each chunk prefixed with its locator, so the model can
only cite what it was given) → `ChatPromptTemplate` → `ChatGroq(model=settings.generation_model)`
→ output parser.

The system prompt encodes the contract, not politeness: never phrase an answer as a
recommendation, cite every clause used, and refuse with "this corpus does not address that"
when context is empty or off-topic. Two of the spec's three answer states are reachable here;
the web tier stays off (`enable_web_fallback=False`).

**Open-weight model, no Anthropic credits.** `openai/gpt-oss-120b` served by Groq's free tier.
`_model()` is the only place the provider appears; `IRAG_GENERATION_MODEL` swaps the model with
no code change. `temperature=0`, so the same question gives the same answer — Phase 4 diffs
these against a golden set.

### Step 3 — `scripts/ask.py`

`python -m scripts.ask "is physiotherapy covered after a minor injury?"` → answer, citation
blocks, licence line. Appends one JSON line per query to `settings.trace_log_path`.

**Dependency:** `langchain-groq`.

**Verify:** the spec's three demo shapes — a covered claim, a question whose answer turns on an
exclusion, and an off-corpus question. The third must refuse. Open every citation by hand once
against ontario.ca to confirm the locator is real.

### Phase 3 results

`"is physiotherapy covered after a minor injury?"` refused at k=5 and answered correctly at
k=20, citing `s. 40(1)`, the MIG treatment blocks, and `s. 38(12)`. The refusal was honest: the
five nearest chunks were all *adjacent* to the answer (s.40 is fee mechanics, s.3(1) is the
definition) without containing it. Dense-only ranking, not generation, was the bottleneck —
which is the same finding as `"s. 31"` from Phase 2, arriving by a different route.

**Two mechanisms came out of that run.**

- **The retry ladder.** `LADDER = (rerank_top_k, fusion_top_k, dense_top_k)` — `(5, 20, 50)`,
  reusing knobs that already exist rather than adding a new one. `answer()` walks it and stops
  at the first non-refusal, so a refusal is only believed once all three widths have refused.
  Cost: a genuinely off-corpus question now spends three model calls instead of one.
- **Cited sources only.** `cited(text, chunks)` keeps the chunks whose locator appears verbatim
  in the answer. Rule 2 makes that a substring match, and it handles the model's combined
  `[A; B]` form for free. Before this, a k=20 run printed twenty citation blocks under a
  six-clause answer. A cited-nothing answer falls back to the full set — that is a prompt
  failure, not a reason to publish an answer with no provenance.

---

## What this iteration deliberately leaves out

Golden set and recall@5, sparse + RRF + cross-encoder rerank, the LangGraph coverage loop with
its mandatory exclusion edge, `resolve_definition` / `fetch_neighbors` / `list_documents`,
FastAPI, and deployment. Phase 2 Step 3 is the single seam all of the retrieval work returns to.

`docs/plan.md` locks "golden set **before** any retrieval code". This iteration knowingly
inverts that to get one end-to-end path running first — flagged because it is a locked decision
to reverse or keep deliberately, not by accident.

---

## How a question actually flows

`scripts/ask.py` calls `answer()`, and everything else hangs off that one function.

**1. Retrieve** — `chain.py`, inside `answer()`

```python
chunks = search_corpus(question, k=width)
```

`search_corpus` embeds the question with Qwen3's *query* prompt, runs a cosine search in
pgvector, and hands back `Chunk` objects — not `Document`s. One type flows through generation
and, later, the agent.

**2. Augment** — `format_context()`

```python
return "\n\n".join(f"[{c.locator}]\n{c.text}" for c in chunks)
```

Each excerpt is stamped with its own locator:

```
[O. Reg. 34/10 s. 18(1)]
The sum of ... shall not exceed $3,500 ...

[SABS s. 3(1) “accident”]
...
```

The locator prefix is the whole trick. The model's only vocabulary of citations is what is
physically in the prompt, so rule 2 of the system prompt — *never cite a locator that is not in
the excerpts* — is checkable rather than aspirational.

**3. Prompt** — `SYSTEM` and `USER` in `chain.py`

Two messages: the five rules, then `{context}` + `{question}`. No conversation history, no
tools. One shot.

**4. Generate** — `_chain()`

```python
return prompt | _model() | StrOutputParser()
```

An LCEL pipe. `temperature=0`, so the answer is reproducible.

**5. Retry, or refuse for real** — the `LADDER` loop

If the answer comes back as the refusal, the same question runs again at a wider `k` — 5, then
20, then 50. Only after all three does the refusal stand. Passing `-k` pins one width and skips
the ladder entirely, which is what you want when measuring.

**6. Return** — `Answer`

`Answer` carries the chunks alongside the text, filtered by `cited()` to the ones the answer
leans on. `ask.py` renders citation blocks from `result.chunks`, **never** from anything the
model wrote — so a hallucinated bracket in the prose cannot become a citation block. It can
only produce a claim with no source under it, which is visible.
