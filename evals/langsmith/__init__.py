"""The LangSmith RAG evaluators, one file each, prompts as published."""

from evals.langsmith.correctness import correctness
from evals.langsmith.groundedness import groundedness
from evals.langsmith.relevance import relevance
from evals.langsmith.retrieval_relevance import retrieval_relevance

#: Only the two that custom evaluators do not already cover, at ground truth rather than opinion.
#: `retrieval_relevance` duplicates recall@5; `relevance` is near-constant for a system that
#: either cites clauses or emits a fixed refusal. Both stay importable for a one-off comparison.
BUILT_INS = (correctness, groundedness)

__all__ = ["correctness", "relevance", "groundedness", "retrieval_relevance", "BUILT_INS"]
