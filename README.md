# InsuranceRAG

Retrieval-augmented question answering over Canadian auto insurance policy documents —
hybrid retrieval, cross-encoder reranked, and **measured at every stage**.

> **Status: in development.**

## Results

| Configuration | recall@5 | MRR | Notes |
|---|---|---|---|
| Dense only (baseline) | — | — | pending Step 4 |
| Hybrid (BM25 + dense, RRF) | — | — | pending Step 5 |
| Hybrid + cross-encoder rerank | — | — | pending Step 6 |

Measured against a 30–50 pair golden set written *before* any retrieval code existed.
Two model profiles are reported: `eval` (full-size weights) and `serve` (the smaller
weights actually running in the deployed demo). Both run the identical pipeline.

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

Public Ontario auto insurance documents: OAP-1 policy wordings, OPCF endorsements,
IBC standard forms, and FSRA bulletins.

Source PDFs are **not** committed. `data/manifest.csv` records the source URL,
document type, page count, and licence note for every document; `scripts/fetch_corpus.py`
reproduces the corpus from it.

## Stack

| Layer | Choice |
|---|---|
| Chunking | Semantic — clause / section / table boundaries, not character count |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Dense retrieval | BGE embeddings, Chroma vector store |
| Fusion | Reciprocal rank fusion |
| Reranking | Cross-encoder over fused top-k |
| Generation | Claude (`claude-opus-5`) |
| Retrieval eval | recall@k, MRR — own harness, `evals/` |
| Generation eval | ragas — groundedness, answer relevance, context precision |
| Serving | FastAPI, Docker, Azure Container Apps |

Retrieval runs entirely on local open-source models. **No API key is required to
reproduce the numbers in the results table** — clone, fetch the corpus, run the
eval suite. A key is only needed for the generation layer.

## Repository layout

```
insurance_rag/
  config.py          eval / serve model profiles, all retrieval knobs
  corpus/            manifest handling, document loading
  chunking/          semantic chunker
  retrieval/         dense, sparse, fusion, rerank
  generation/        answer synthesis
  tracing/           per-query latency, tokens, chunk IDs, cost
  api/               FastAPI app
evals/               golden set, retrieval harness, ragas suite, results
tests/               chunking tests + golden-set regression gate
scripts/             corpus fetch, index build
```

## Quickstart

> Not yet runnable — corpus and retrieval land in Steps 1–6.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[eval,dev]"
cp .env.example .env
```

## Licence

Code is MIT. Source documents are public filings published by their respective
issuers and are not redistributed here — see `data/manifest.csv` for provenance.