"""The agent's per-request state and its final answer."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from insurance_rag.agent.web import WebResult
from insurance_rag.schema import Chunk

__all__ = ["AgentState", "AgentAnswer"]


class AgentState(TypedDict):
    """What every node reads and updates; `pools` merges, so each node adds its own channel."""

    question: str
    sub_questions: list[str]
    pools: Annotated[dict[str, list[Chunk]], operator.or_]
    checked: list[str]
    context: list[Chunk]
    answer: str
    retries: int
    steps: int
    tokens: int
    web: list[WebResult]
    web_summary: str
    blocked: str
    guard: list[str]
    disclaimer: str


@dataclass(frozen=True)
class AgentAnswer:
    """The text, the chunks it cited, everything it was shown, and what was checked."""

    question: str
    text: str
    chunks: list[Chunk]
    retrieved: list[Chunk]
    checked: list[str]
    steps: int
    tokens: int
    cited_nothing: bool
    web: list[WebResult] = field(default_factory=list)
    web_summary: str = ""
    blocked: str = ""
    guard: list[str] = field(default_factory=list)
    disclaimer: str = ""
