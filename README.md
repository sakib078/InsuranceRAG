# InsuranceRAG

Ask a question about Ontario auto insurance, get an answer with the clause it came from.

Every answer cites a provision you can open on ontario.ca and read yourself — or it refuses.
The interesting part isn't the pipeline; it's that the whole thing is measured against a
hand-labelled golden set, so the weak spots are numbers rather than opinions.

> **Status:** end-to-end and measured. Dense + BM25 hybrid retrieval; agent next.

## Results

62 hand-written question/answer pairs, each labelled with the clauses that should be
retrieved. 20 Ontario documents, 3,896 chunks. Retrieval scored at top-5; answers are generated
through the retry ladder, which re-asks at top-20 after a refusal.

| Configuration | recall@5 single | recall@5 multi | exclusion recall | citation accuracy | correctness | groundedness |
|---|---|---|---|---|---|---|
| Dense only | 0.667 | 0.095 | 0.462 | 0.499 | 0.710 | 0.661 |
| + BM25 sparse, fused by RRF | 0.708 | 0.095 | 0.500 | 0.520 | 0.726 | 0.677 |
| **+ retry ladder (shipped)** | **0.708** | 0.143 | **0.500** | **0.542** | **0.774** | **0.742** |
| + agent with coverage loop | — | — | — | — | — | pending |

One variable per row. BM25 adds +0.016 correctness — one answer in 62, so direction rather than
result. The ladder adds **+0.048, three answers**, by re-asking refusals against the top-20 pool;
it changes no retrieval metric because it only widens *after* a refusal.

### Ceiling — the same pipeline at top-20

What a reranker or a wider retry would have to work with:

| | top-5 | top-20 | gap |
|---|---|---|---|
| recall single | 0.708 | **0.875** | 4 questions |
| recall multi | 0.143 | **0.381** | 5 |
| exclusion recall | 0.500 | **0.731** | 6 |

Reordering the pool can reach the middle column and no further. Past it, **13 of 21 multi-hop
questions have gold clauses outside the top 20 entirely** — one query cannot surface them however
it is ranked. That is what the agent is for, and why multi-hop ignored chunk size, a lexical
channel and two cross-encoders alike.

### Built, measured, rejected

| | result | parked in |
|---|---|---|
| `ts_rank` sparse | no IDF, so "insurance" outweighed clause numbers: +1 question, −7 | `artifacts/sparse_tsrank.py` |
| Qwen3-Reranker 0.6B | lost on every metric; 124s/query | `artifacts/rerank.py` |
| gte-modernbert 150M | lost on every metric; 17s/query | `artifacts/rerank.py` |

Exclusion recall fell as the reranker grew — asked *"does this answer the question"*, a
cross-encoder prefers the clause granting a benefit to the one cancelling it. Both were measured
before BM25 replaced `ts_rank`, so they reranked a noisier pool than today's.

Two caveats on precision: groundedness is scored across the 17 refusal records where there are no
facts to be grounded in (~0.82 on the answerable slice), and BM25 ranking shifts by about one
question across index rebuilds, which is why multi-hop reads 0.095 in one table and 0.143 in the
other.

### Earlier, on 56 records

Six citation-shaped questions were added to the golden set afterwards, so these are not
comparable to the table above.

**Answer quality tracks retrieval, 3×.** Splitting a generation run by whether the right
clauses were retrieved: all found (n=15) → correctness **0.933**; some found (n=14) → 0.857;
none found (n=10) → **0.300**. Eight of the ten wrong answers had no gold clause in the top 5 —
the model wasn't inventing law, it was never shown it. Overall: citation accuracy 0.501,
correctness 0.768.

**Clause-aware chunking beats fixed windows.** A uniform 512-token chunker scored *higher* on
single-hop, 0.842 against 0.737 — by reading 3.4× more text, 2,810 prompt tokens against 825.
At a matched budget it falls to 0.421. It loses exclusion recall at every setting, and a window
cutting across provisions has no clause to cite at all. Multi-hop was identical either way, so
chunk size is not what makes multi-hop hard here.

## Why insurance is hard for RAG

A coverage clause and the exclusion that cancels it are nearly identical in wording.
*"loss or damage caused by collision"* and *"this policy does not cover loss or damage caused
by collision"* land in almost the same place in embedding space. Retrieve the wrong one and
the answer is fluent, confident, and wrong in the direction that costs a claimant money.

Two measured failures from this corpus:

- **`"s. 31"` returns a top-5 cosine spread of 0.006** — no signal at all. Clause numbers are
  lexical, and embeddings smooth them away.
- **Two paraphrases of one question returned disjoint evidence.** *"my insurance company
  stopped my wage loss payments"* and *"when can an insurer refuse to pay income replacement
  benefits?"* shared not one chunk, and neither surfaced s. 31 — the provision literally
  titled *Circumstances in which certain benefits not payable*.

A claimant asking in their own words and one asking in the regulation's words get different
law back, with no signal that the other half exists.

## Stack

| | | |
|---|---|---|
| Ingestion | Docling for PDF layout, direct DOM walk for e-Laws HTML | built |
| Chunking | One provision per chunk, never merged across clauses | built |
| Retrieval | Qwen3-Embedding-0.6B + BM25, fused by RRF → ParadeDB (Postgres + pgvector + pg_search) | built |
| Generation | `gpt-oss-120b`, open weights, served by Groq | built |
| Eval | Own locator-based evaluators + LangSmith judges | built |
| Next | LangGraph agent with a mandatory exclusion check | planned |

One Postgres holds the chunks, the embeddings and the BM25 index — rather than a vector
database beside a search engine.

The retrieval metrics need no API key: they're set arithmetic over clause locators, so
`run_eval --suite retrieval` reproduces offline. Only the LLM-judged metrics need keys, and
every provider used is on a free tier.

## Quickstart

```bash
pip install -e ".[eval,dev]"
cp .env.example .env          # set IRAG_POSTGRES_DSN — it has no default, by design

docker run -d --name paradedb -p 6025:5432 \
  -e POSTGRES_USER=... -e POSTGRES_PASSWORD=... -e POSTGRES_DB=... \
  -v paradedb_data:/var/lib/postgresql paradedb/paradedb:latest

python -m scripts.fetch_corpus   # e-Laws HTML; FSRA PDFs are hand-downloaded
python -m scripts.ingest         # → data/chunks/*.jsonl
python -m scripts.index --embeddings data/embeddings
python -m scripts.migrate_bm25   # BM25 index for the sparse channel
python -m scripts.ask "is physiotherapy covered after a minor injury?"
```

Bulk embedding runs on Colab (`notebooks/embed_colab.ipynb`); indexing inserts the
precomputed vectors, so no local GPU is needed.

Inspect retrieval without spending a model call:

```bash
python -m insurance_rag.retrieval.search "minor injury" -k 10 --text
```

Re-run the evaluation:

```bash
python -m evals.run_eval --suite retrieval  --config dense_clause_aware --misses
python -m evals.run_eval --suite generation --config dense_clause_aware --pin-k 5
```

## Corpus and licence

Twenty public Ontario documents — the SABS, OAP 1, the Insurance Act, R.R.O. 664/668/676,
OPCF endorsements, FSRA guidelines — plus two **revoked** regulations kept deliberately as
distractors, because e-Laws serves repealed law at live URLs with no structural marker.

Source documents are not committed. `data/manifest.csv` records the URL, consolidation date,
status and licence note for each; `scripts/fetch_corpus.py` rebuilds the corpus from it.

Code is MIT. Ontario legislation is reproduced under King's Printer terms, which require
stating that this is **not an official version**. Every citation the system emits says so.
