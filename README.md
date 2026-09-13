# InsuranceRAG

Ask a question about Ontario auto insurance, get an answer with the clause it came from.

Every answer cites a provision you can open on ontario.ca and read yourself — or it refuses.
The interesting part isn't the pipeline; it's that the whole thing is measured against a
hand-labelled golden set, so the weak spots are numbers rather than opinions.

> **Status:** end-to-end and measured. Dense retrieval only — sparse, fusion and rerank next.

## Results

Measured on 56 hand-written question/answer pairs, each labelled with the clauses that
should be retrieved. 20 Ontario documents, 3,896 chunks.

All rows retrieve top-5. Citation accuracy and correctness need a generation run.

| Configuration | recall@5 single | recall@5 multi | exclusion recall | citation accuracy | correctness |
|---|---|---|---|---|---|
| Uniform 512-token chunks | 0.842 | 0.050 | 0.350 | n/a | n/a |
| **Clause-aware (baseline)** | **0.737** | **0.050** | **0.450** | **0.501** | **0.768** |
| + hybrid retrieval & cross-encoder rerank | — | — | — | — | pending |
| + agent with coverage loop | — | — | — | — | pending |

**The uniform row wins single-hop, and that result is an artefact.** It's a fixed-window
chunker over the same corpus, encoder and questions — the control for whether structural
chunking earned its place. Five uniform windows carry **2,810 prompt tokens against 825**, so
it wins by reading 3.4× more text. Give it a matched budget — top-2, still 1,124 tokens, still
more than clause-aware gets — and it drops to **0.421**, a 0.32 loss.

It also loses exclusion recall at *every* setting, 0.350 against 0.450, even with the 3.4×
advantage. Fat windows don't help you find the clause that cancels coverage; they dilute it.

Citation accuracy is `n/a` rather than low: a window cutting across provisions has no clause to
cite, so the naive chunker forfeits the product's core promise outright.

Gold clauses resolve against uniform windows by coverage (a window must hold ≥50% of the
provision) since a window has no locator to match. Both that bound and a strict-containment
bound are recorded in `evals/results/`; the conclusions are identical under each.

### The number that matters

Splitting the same run by whether retrieval actually found the right clauses:

| gold clauses in top-5 | questions | correctness |
|---|---|---|
| all found | 15 | **0.933** |
| some found | 14 | 0.857 |
| none found | 10 | **0.300** |

**A 3× swing in answer quality, attributable entirely to retrieval.** Of the ten answers
graded wrong, eight were cases where the governing clause never reached the model. It wasn't
making things up — it was never shown the answer.

That's the case for the next phase, measured rather than assumed.

And the obvious objection — *"0.050 just means you cut the documents too small"* — was tested
and is wrong. Fixed 512-token windows score **exactly the same 0.050** on multi-hop while
reading 3.4× more text. Multi-hop retrieval is hard here for reasons that have nothing to do
with chunk size.

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
| Retrieval | Qwen3-Embedding-0.6B → Postgres 16 + pgvector | built |
| Generation | `gpt-oss-120b`, open weights, served by Groq | built |
| Eval | Own locator-based evaluators + LangSmith judges | built |
| Next | Sparse `ts_rank`, RRF, cross-encoder rerank, LangGraph agent | planned |

One Postgres holds the chunks, the embeddings and (soon) the full-text index — rather than a
vector database beside a search engine.

The retrieval metrics need no API key: they're set arithmetic over clause locators, so
`run_eval --suite retrieval` reproduces offline. Only the LLM-judged metrics need keys, and
every provider used is on a free tier.

## Quickstart

```bash
pip install -e ".[eval,dev]"
cp .env.example .env          # set IRAG_POSTGRES_DSN — it has no default, by design

docker run -d --name pgvector -p 6024:5432 \
  -e POSTGRES_USER=... -e POSTGRES_PASSWORD=... -e POSTGRES_DB=... pgvector/pgvector:pg16

python -m scripts.fetch_corpus   # e-Laws HTML; FSRA PDFs are hand-downloaded
python -m scripts.ingest         # → data/chunks/*.jsonl
python -m scripts.index --embeddings data/embeddings
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
