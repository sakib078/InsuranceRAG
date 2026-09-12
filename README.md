# InsuranceRAG

Retrieval-augmented question answering over Canadian auto insurance policy documents —
hybrid retrieval, cross-encoder reranked, and **measured at every stage**.

> **Status: end-to-end pipeline running — ingestion → pgvector → cited answer.**
> Dense retrieval only; sparse, fusion and rerank are next.

## Results

| Configuration | recall@5 | MRR | Notes |
|---|---|---|---|
| Dense only (baseline) | — | — | pending Step 4 |
| Hybrid (BM25 + dense, RRF) | — | — | pending Step 5 |
| Hybrid + cross-encoder rerank | — | — | pending Step 6 |

Measured against a 56-record golden set written *before* any retrieval code existed.
Two model profiles are reported: `eval` (full-size weights) and `serve` (the smaller
weights actually running in the deployed demo). Both run the identical pipeline.

### Findings so far — dense-only, pre-rerank

- A **definition lookup** retrieves cleanly: `"what counts as an accident?"` →
  `O. Reg. 34/10 s. 3(1) “accident”` at cosine distance 0.327.
- A **clause lookup** has no signal at all: `"s. 31"` returns a top-5 spread of 0.006.
  This is the measured case for sparse retrieval, not a hunch.
- Retrieval is **phrasing-bound, not meaning-bound.** Two paraphrases of one question —
  *"my insurance company stopped my wage loss payments"* and *"when can an insurer refuse
  to pay income replacement benefits?"* — returned **disjoint** evidence sets (s. 58 and
  s. 61 vs s. 5, s. 6 and s. 37). Both answers were defensible; neither surfaced s. 31,
  the provision actually titled *Circumstances in which certain benefits not payable*.

The third finding is the one that motivates the rest of the retrieval work: a claimant
asking in their own words and one asking in the regulation's words get different law back,
with no signal that the other half exists.

## The problem

Insurance policies are adversarially structured for naive retrieval. An exclusion
clause and the coverage clause it contradicts are near-identical in wording and
sit in the same semantic neighbourhood — "loss or damage caused by collision" and
"this policy does not cover loss or damage caused by collision" embed to almost the
same vector. Dense-only retrieval will happily hand back the wrong one, and the
generated answer will be fluent, confident, and wrong in the direction that costs
a claimant money.

Two things address it, and this repo measures both:

- **BM25 alongside dense retrieval**, because exact clause numbers and defined terms
  (`OPCF 44R`, "Named Insured") are lexical signals that embeddings smooth away.
- **A cross-encoder reranker** over the fused candidates, which scores the
  query and passage *jointly* rather than comparing two independently-produced
  vectors — the only stage that can actually see the negation.

## Corpus

Twenty public Ontario documents: the SABS (O. Reg. 34/10), OAP 1, R.R.O. 664 and 668,
O. Reg. 461/96, the Insurance Act, OPCF endorsements, and FSRA guidelines including the
Minor Injury Guideline. 3,896 chunks indexed.

Source PDFs are **not** committed. `data/manifest.csv` records the source URL,
document type, page count, and licence note for every document; `scripts/fetch_corpus.py`
reproduces the corpus from it.

## Stack

| Layer | Choice | |
|---|---|---|
| Ingestion | Docling (PDF layout) + direct e-Laws DOM walk (HTML) | built |
| Chunking | Structural — one provision per chunk, never merged across clauses | built |
| Dense retrieval | Qwen3-Embedding-0.6B → Postgres 16 + pgvector | built |
| Generation | `openai/gpt-oss-120b`, open weights, served by Groq | built |
| Sparse retrieval | Postgres `ts_rank` | planned |
| Fusion | Reciprocal rank fusion | planned |
| Reranking | Qwen3-Reranker-0.6B over fused top-k | planned |
| Retrieval eval | recall@5 by hop, exclusion recall — own evaluators, `evals/` | built |
| Generation eval | LangSmith judges + citation accuracy, `evals/` | built |
| Serving | FastAPI + Docker | planned |

The encoder is an **eval variable, not a deployment choice**: `IRAG_ENCODER=qwen3|bge-m3`
selects a bi-encoder and its own family's reranker, and both run the identical pipeline
over byte-identical chunks so the bake-off is a fair comparison.

Retrieval runs entirely on local open-source models, so **the retrieval suite
reproduces offline** — clone, fetch the corpus, `run_eval --suite retrieval`, no key.
The generation suite needs keys: one for the answer model and a LangSmith key, which
now owns the experiment stack. Both tiers are free — Groq's free tier serves the
open-weight model — so there are no paid credits anywhere in the project.

## Repository layout

```
insurance_rag/
  config.py          encoder bake-off specs, retrieval knobs, .env-only secrets
  schema.py          the Chunk contract — every ingester targets it
  providers.py       OpenAI-compatible endpoints for the generator and the judges
  ratelimit.py       retry through provider rate limits
  corpus/            manifest handling, provenance enums
  ingest/            e-Laws DOM, Docling PDF, splitting, role classification
  retrieval/         store.py (pgvector), search.py (the one retrieval seam)
  generation/        chain.py (LCEL), citations.py (provenance blocks)
  tracing/           per-query latency, tokens, chunk IDs, cost      [planned]
  api/               FastAPI app                                     [planned]
evals/               golden set, custom + LangSmith evaluators, results
tests/               chunking tests + golden-set regression gate     [planned]
scripts/             fetch_corpus.py, ingest.py, index.py, ask.py
notebooks/           embed_colab.ipynb — GPU batch embedding
```

Until `tracing/` exists, `scripts/ask.py` appends one JSON line per query to
`data/traces.jsonl`: question, answer, chunk IDs, and locators.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[eval,dev]"
cp .env.example .env

docker run -d --name pgvector -p 6024:5432 \
  -e POSTGRES_USER=... -e POSTGRES_PASSWORD=... -e POSTGRES_DB=... \
  pgvector/pgvector:pg16

python -m scripts.ingest                              # → data/chunks/*.jsonl
python -m scripts.index --embeddings data/embeddings  # vectors from the Colab notebook
python -m scripts.ask "is physiotherapy covered after a minor injury?"
```

`IRAG_POSTGRES_DSN` has no default — set it in `.env` (psycopg3 form,
`postgresql+psycopg://…`, host-side port) or config fails at import. That is
deliberate: no credential can reach a commit by sitting in `config.py`.

Bulk embedding runs on Colab (`notebooks/embed_colab.ipynb`) and `scripts/index.py`
inserts the precomputed vectors, so no local GPU is needed. To embed locally
instead, drop `--embeddings` and expect hours on CPU.

Inspect retrieval without spending a model call:

```bash
python -m insurance_rag.retrieval.search "minor injury" -k 10 --text
```

## Licence

Code is MIT. Source documents are public filings published by their respective
issuers and are not redistributed here — see `data/manifest.csv` for provenance.