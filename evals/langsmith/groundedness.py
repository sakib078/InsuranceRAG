"""Groundedness: the answer against the chunks it was shown. The hallucination check."""

from __future__ import annotations

from typing_extensions import Annotated, TypedDict

from evals.langsmith.judge import facts, grade, score


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


#: Published prompt, plus one line so bracketed clause locators are read as citations, not claims.
grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

A bracketed reference such as [O. Reg. 34/10 s. 18(1)] is a citation identifying which of the FACTS a statement rests on. Judge the statement it supports, not the bracket itself.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def groundedness(inputs: dict, outputs: dict) -> dict:
    """A simple evaluator for RAG answer groundedness."""
    answer = f"FACTS: {facts(outputs)}\nSTUDENT ANSWER: {outputs['answer']}"
    verdict = grade(GroundedGrade, grounded_instructions, answer)
    return score("groundedness", verdict["grounded"])
