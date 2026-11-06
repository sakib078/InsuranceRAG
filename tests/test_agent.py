"""The agent graph with retrieval and synthesis faked: 0 tokens, no Postgres."""

from __future__ import annotations

import pytest

from insurance_rag.agent import graph, tools
from insurance_rag.agent.budget import BudgetExceeded
from insurance_rag.config import settings
from insurance_rag.generation.chain import REFUSAL
from insurance_rag.retrieval.search import Hit


@pytest.fixture
def chunks():
    return tools.ordinal_index()["sabs-o-reg-34-10"][:6]


@pytest.fixture(autouse=True)
def fake_retrieval(monkeypatch, chunks):
    monkeypatch.setattr(graph, "search_corpus", lambda question, k: chunks[:k])
    monkeypatch.setattr(
        tools, "exclusion_pass", lambda question: [Hit(c, ("probe",), 0.1) for c in chunks[4:]]
    )


def say(monkeypatch, text: str, tokens: int = 100) -> None:
    monkeypatch.setattr(graph, "_generate", lambda context, question: (text, tokens))


def test_cited_answer_ends_without_retry(monkeypatch, chunks):
    say(monkeypatch, f"It is limited [{chunks[0].locator}].")
    result = graph.run_agent("q")
    assert result.chunks == [chunks[0]] and result.steps == 5 and not result.cited_nothing


def test_exclusion_always_checked(monkeypatch, chunks):
    say(monkeypatch, f"[{chunks[0].locator}]")
    assert "exclusion" in graph.run_agent("q").checked


def test_refusal_retries_once_then_refuses_within_the_cap(monkeypatch):
    say(monkeypatch, REFUSAL)
    result = graph.run_agent("q")
    assert result.text == REFUSAL and result.chunks == []
    assert result.steps == 9 <= settings.max_agent_steps


def test_uncited_answer_after_the_retry_is_a_refusal(monkeypatch):
    say(monkeypatch, "An answer that cites nothing.")
    assert graph.run_agent("q").text == REFUSAL


def test_budget_stops_the_run(monkeypatch):
    say(monkeypatch, REFUSAL, tokens=settings.agent_token_budget + 1)
    with pytest.raises(BudgetExceeded):
        graph.run_agent("q")
