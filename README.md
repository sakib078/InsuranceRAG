# InsuranceRAG

Ask a question about Ontario auto insurance, get an answer with the clause it came from.

Every answer cites a provision you can open on ontario.ca and read yourself — or it refuses.
The interesting part isn't the pipeline; it's that the whole thing is measured against a
hand-labelled golden set, so the weak spots are numbers rather than opinions.

> **Status:** end-to-end and measured. Dense + BM25 hybrid retrieval; agent next.

## Results

62 hand-written question/answer pairs, each labelled with the clauses that should be
retrieved. 20 Ontario documents, 3,896 chunks. Every row retrieves top-5.

| Configuration | recall@5 single | recall@5 multi | exclusion recall | citation accuracy | correctness | groundedness |
|---|---|---|---|---|---|---|
| Dense only | 0.667 | 0.095 | 0.462 | 0.499 | 0.710 | 0.661 |
| **+ BM25 sparse, fused by RRF** | **0.708** | 0.095 | **0.500** | **0.520** | **0.726** | **0.677** |
| + cross-encoder rerank | — | — | — | — | — | rejected, below |
| + agent with coverage loop | — | — | — | — | — | pending |

Same 62 questions, same generator, same judge — the only variable between those two rows is the
sparse channel. **Every metric moves up and none moves down**, which is the loop closing: better
retrieval, better answers.

Read it cautiously. Correctness 0.710 → 0.726 is **one more correct answer out of 62**, well
inside the noise. What's defensible is the direction, consistent across six metrics, with the
mechanism identified before the run: BM25 rescued three retrieval records and broke one, and
every flip was predicted from the scoring function.

Groundedness understates the system — it is scored across the 17 refusal records, where there
are no facts to be grounded in. On the answerable slice it is closer to 0.82.

Two things were built, measured and thrown away. That is what the harness is for:

- **Postgres `ts_rank` for the sparse channel.** No inverse document frequency, so in a corpus
  where every chunk says "insurance" that word counted as much as a clause number. It rescued
  one question and broke seven. BM25 — same channel, same weight, IDF added — rescued three and
  broke one. Parked in `artifacts/sparse_tsrank.py`.
- **Cross-encoder reranking.** Neither `Qwen3-Reranker-0.6B` nor a 150M alternative beat dense
  on any metric, and exclusion recall fell as the model grew: a reranker asked *"does this
  answer the question"* prefers the clause granting a benefit to the one taking it away. The
  0.6B model took 124s per query to get there. Parked in `artifacts/rerank.py`.

### Measured earlier, on 56 records

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
