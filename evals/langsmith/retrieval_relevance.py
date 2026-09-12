"""Retrieval relevance: the retrieved chunks against the question."""

from __future__ import annotations

from typing_extensions import Annotated, TypedDict

from evals.langsmith.judge import facts, grade, score


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


#: Published prompt, unchanged.
retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def retrieval_relevance(inputs: dict, outputs: dict) -> dict:
    """An evaluator for document relevance"""
    answer = f"FACTS: {facts(outputs)}\nQUESTION: {inputs['question']}"
    verdict = grade(RetrievalRelevanceGrade, retrieval_relevance_instructions, answer)
    return score("retrieval_relevance", verdict["relevant"])
