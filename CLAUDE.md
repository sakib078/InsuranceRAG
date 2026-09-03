# InsuranceRAG

Build plan: `docs/plan.md`. Corpus and access facts: `docs/sourceMap.md`.

## Rules for this repo
- **`docs/plan.md` and `docs/sourceMap.md` are mine.** Never edit either unless I name the
  file. Propose the wording in chat; I apply it.
- Findings are one line with a number, not a paragraph with justification.
- Build the step I name. Not the next step, not its prerequisites unless I say so.
- One-line docstrings only. Rationale lives in the docs, not the source.
- No new dependencies, config keys, or directories beyond what the named step needs.
- Never run installs, `docker compose up`, migrations, or model downloads without asking.

## Layout
- `insurance_rag/schema.py` — the `Chunk` contract; every ingester targets it
- `insurance_rag/corpus/manifest.py` — `load_manifest()`, `by_doc_id()`, provenance enums
- `insurance_rag/config.py` — `settings`
- `data/manifest.csv` — the corpus; 11 v1 documents on disk
- `scripts/fetch_corpus.py` — e-Laws fetcher with `validate_document()`