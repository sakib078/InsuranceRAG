"""Question -> cited answer. Retrieval is `search_corpus`; nothing else reaches the model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from insurance_rag.config import settings
from insurance_rag.retrieval.search import search_corpus
from insurance_rag.schema import Chunk

__all__ = ["Answer", "answer", "cited", "format_context", "LADDER", "REFUSAL"]

REFUSAL = "This corpus does not address that."

#: Widen the window on a refusal before believing it. Existing knobs, no new ones.
LADDER: tuple[int, ...] = (settings.rerank_top_k, settings.fusion_top_k, settings.dense_top_k)

SYSTEM = """You answer questions about Ontario auto insurance using only the excerpts provided.

Rules, in order of priority:
1. Use only the excerpts. If they do not answer the question, reply exactly: {refusal}
2. Cite the locator in square brackets after every statement that rests on an excerpt, e.g.
   [O. Reg. 34/10 s. 18(1)]. Never cite a locator that is not in the excerpts.
3. State what the documents say. Never advise, recommend, or predict an outcome for the reader.
4. If an excerpt limits or excludes what another grants, say so in the same answer.
5. Be brief. No preamble, no restatement of the question."""

USER = """Excerpts:

{context}

Question: {question}"""


@dataclass(frozen=True)
class Answer:
    """The model's text plus the exact chunks it was shown, so citations cannot drift."""

    question: str
    text: str
    chunks: list[Chunk]


def format_context(chunks: list[Chunk]) -> str:
    """Each excerpt is prefixed with its locator - the model can only cite what it is given."""
    return "\n\n".join(f"[{c.locator}]\n{c.text}" for c in chunks)


@lru_cache(maxsize=1)
def _model():
    """Open-weight model on Groq; swapping the provider is this function and nothing else."""
    from langchain_groq import ChatGroq

    if not settings.groq_api_key:
        raise SystemExit("set IRAG_GROQ_API_KEY in .env - get one at console.groq.com/keys")
    return ChatGroq(
        model=settings.generation_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=1024,
    )


@lru_cache(maxsize=1)
def _chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM.replace("{refusal}", REFUSAL)), ("human", USER)]
    )
    return prompt | _model() | StrOutputParser()


def cited(text: str, chunks: list[Chunk]) -> list[Chunk]:
    """The chunks the answer actually leans on - rule 2 makes every locator appear verbatim."""
    return [c for c in chunks if c.locator in text]


def answer(question: str, *, k: int | None = None) -> Answer:
    """Retry a refusal at a wider k; refuse for real only once the ladder is exhausted."""
    for width in (k,) if k else LADDER:
        chunks = search_corpus(question, k=width)
        if not chunks:
            break
        text = _chain().invoke({"context": format_context(chunks), "question": question}).strip()
        if REFUSAL not in text:
            # A cited-nothing answer is a prompt failure, not a reason to drop the provenance.
            return Answer(question, text, cited(text, chunks) or chunks)
    return Answer(question, REFUSAL, [])
