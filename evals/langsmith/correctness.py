"""Correctness: the answer against the golden set's reference answer."""

from __future__ import annotations

from typing_extensions import Annotated, TypedDict

from evals.langsmith.judge import grade, score


class CorrectnessGrade(TypedDict):
    #: Field order is generation order - explaining first forces the reasoning before the verdict.
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


#: Published prompt, plus one paragraph for the refusal slice this corpus deliberately contains.
correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

If the GROUND TRUTH ANSWER states that the source documents do not address the question, then a STUDENT ANSWER that also declines to answer is correct, and a STUDENT ANSWER that supplies an answer anyway is not. Different wording of the same refusal is still correct.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """An evaluator for RAG answer accuracy"""
    if not outputs.get("answer"):
        return {"key": "correctness", "score": None}  # the target failed; nothing to judge
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    verdict = grade(CorrectnessGrade, correctness_instructions, answers)
    return score("correctness", verdict["correct"])
