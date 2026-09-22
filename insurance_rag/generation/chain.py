"""Question -> cited answer. Retrieval is `search_corpus`; nothing else reaches the model."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from insurance_rag.config import settings
from insurance_rag.providers import chat_model
from insurance_rag.ratelimit import with_retry
from insurance_rag.retrieval.search import search_corpus
from insurance_rag.schema import Chunk

__all__ = ["Answer", "answer", "cited", "format_context", "LADDER", "REFUSAL"]

REFUSAL = "This corpus does not address that."

#: Widen the window on a refusal before believing it. Existing knobs, no new ones.
#: The `dense_top_k` rung is gone: measured at 7,719-12,534 prompt tokens, a k=50 request
#: breaches the 8,000 TPM ceiling of the free tier this was measured on and comes back HTTP 413.
#: 50 chunks also dilute the context more than they help, so the ladder tops out at 20.
LADDER: tuple[int, ...] = (settings.rerank_top_k, settings.fusion_top_k)

#: Below that 8,000 TPM ceiling with room for the system prompt, so one request can never 413.
MAX_CONTEXT_TOKENS = 6000

SYSTEM = """You answer questions about Ontario auto insurance using only the excerpts provided.

Rules, in order of priority:
1. Use only the excerpts. If they do not answer the question, reply exactly: {refusal}
2. An excerpt marked REVOKED is repealed law. Never state it as the current rule. If only
   revoked excerpts address the question, reply exactly: {refusal}
3. Cite the locator in square brackets after every statement that rests on an excerpt, e.g.
   [O. Reg. 34/10 s. 18(1)]. Never cite a locator that is not in the excerpts.
4. State what the documents say. Never advise, recommend, or predict an outcome for the reader.
5. If an excerpt limits or excludes what another grants, say so in the same answer.
6. Be brief. No preamble, no restatement of the question."""

USER = """Excerpts:

{context}

Question: {question}"""


@dataclass(frozen=True)
class Answer:
    """The model's text, the chunks it cited, and the full set it was shown."""

    question: str
    text: str
    chunks: list[Chunk]     # cited - what ask.py renders as sources
    retrieved: list[Chunk]  # everything the model saw - recall and citation metrics read this


def within_budget(chunks: list[Chunk]) -> list[Chunk]:
    """Drop the lowest-ranked chunks once the budget is spent; never truncate one mid-provision."""
    kept, spent = [], 0
    for chunk in chunks:
        if spent + chunk.token_count > MAX_CONTEXT_TOKENS:
            break
        kept.append(chunk)
        spent += chunk.token_count
    return kept or chunks[:1]  # one oversized table still beats an empty prompt


@lru_cache(maxsize=1)
def revoked_docs() -> frozenset[str]:
    """doc_ids the manifest marks repealed. e-Laws serves these at live URLs with no marker."""
    from insurance_rag.corpus.manifest import Status, load_manifest

    return frozenset(row.doc_id for row in load_manifest() if row.status is Status.REVOKED)


def format_context(chunks: list[Chunk]) -> str:
    """Each excerpt is prefixed with its locator - the model can only cite what it is given.

    Repealed excerpts are marked. `citation_accuracy` zeroes any answer resting on revoked law
    and the renderer flags it, so withholding the status here graded the model on something it
    was never shown. Parentheses, not brackets, so the marker cannot be read as part of a cite.
    """
    revoked = revoked_docs()
    return "\n\n".join(
        f"[{c.locator}]"
        f"{'  (REVOKED - repealed, not the current rule)' if c.doc_id in revoked else ''}"
        f"\n{c.text}"
        for c in within_budget(chunks)
    )


@lru_cache(maxsize=1)
def _model():
    """The open-weight generator; swapping the provider is a config key, not a code change."""
    return chat_model(settings.generation_provider, settings.generation_model, max_tokens=1024)


@lru_cache(maxsize=1)
def _chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM.replace("{refusal}", REFUSAL)), ("human", USER)]
    )
    return prompt | _model() | StrOutputParser()


def cited(text: str, chunks: list[Chunk]) -> list[Chunk]:
    """The chunks the answer actually leans on - rule 2 makes every locator appear verbatim.

    Both sides are NFKC-normalised because the generator writes U+202F, a narrow no-break space,
    where the locator has an ordinary one: measured at 21 of 62 answers whose citations a raw
    substring match could not see. NFKC leaves the curly quotes in definition locators alone.
    """
    normal = unicodedata.normalize("NFKC", text)
    return [c for c in chunks if unicodedata.normalize("NFKC", c.locator) in normal]


def answer(question: str, *, k: int | None = None) -> Answer:
    """Retry a refusal at a wider k; refuse for real only once the ladder is exhausted."""
    retrieved: list[Chunk] = []
    for width in (k,) if k else LADDER:
        retrieved = search_corpus(question, k=width)
        if not retrieved:
            break
        payload = {"context": format_context(retrieved), "question": question}
        text = with_retry(lambda: _chain().invoke(payload)).strip()
        if REFUSAL not in text:
            # A cited-nothing answer is a prompt failure, not a reason to drop the provenance.
            return Answer(question, text, cited(text, retrieved) or retrieved, retrieved)
    return Answer(question, REFUSAL, [], retrieved)
