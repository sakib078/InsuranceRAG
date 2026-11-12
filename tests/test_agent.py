"""The agent graph with retrieval and synthesis faked: 0 tokens, no Postgres."""

from __future__ import annotations

import pytest

from insurance_rag.agent import budget, graph, tools, web
from insurance_rag.agent.budget import BudgetExceeded
from insurance_rag.agent.web import WebResult
from insurance_rag.config import settings
from insurance_rag.generation.chain import REFUSAL
from insurance_rag.retrieval.search import Hit
from insurance_rag.retrieval.store import read_chunks


@pytest.fixture
def chunks():
    return read_chunks(settings.chunks_dir / "sabs-o-reg-34-10.jsonl")[:6]


@pytest.fixture(autouse=True)
def fake_retrieval(monkeypatch, chunks):
    monkeypatch.setattr(graph, "search_corpus", lambda question, k: chunks[:k])
    monkeypatch.setattr(
        tools, "exclusion_pass", lambda question: [Hit(c, ("probe",), 0.1) for c in chunks[4:]]
    )


@pytest.fixture(autouse=True)
def fake_planner(monkeypatch):
    monkeypatch.setattr(graph, "decompose", lambda question: [])
    monkeypatch.setattr(budget, "_spent", 0)
    monkeypatch.setattr(settings, "enable_web_fallback", False)
    monkeypatch.setattr(settings, "web_summary", False)


def test_sub_questions_become_pools(monkeypatch, chunks):
    monkeypatch.setattr(graph, "decompose", lambda question: ["a", "b"])
    say(monkeypatch, f"[{chunks[0].locator}]")
    assert graph.run_agent("q").checked[:3] == ["main", "sub:0", "sub:1"]


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


def test_web_off_never_reaches_the_web_node(monkeypatch):
    say(monkeypatch, REFUSAL)
    monkeypatch.setattr(settings, "enable_web_fallback", False)
    monkeypatch.setattr(web, "web_supplement", lambda question: pytest.fail("web tier ran"))
    assert graph.run_agent("q").web == []


def test_web_results_follow_a_refusal_and_stay_out_of_the_prompt(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        graph, "_generate", lambda context, question: (prompts.append(context) or REFUSAL, 100)
    )
    monkeypatch.setattr(settings, "enable_web_fallback", True)
    hit = WebResult("title", "https://example.com", "SNIPPET", "2026-11-06")
    monkeypatch.setattr(web, "web_supplement", lambda question: [hit])
    result = graph.run_agent("q")
    assert result.text == REFUSAL and result.web == [hit]
    assert all("SNIPPET" not in context for context in prompts)


def test_the_summary_follows_the_web_results_and_stays_out_of_the_prompt(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        graph, "_generate", lambda context, question: (prompts.append(context) or REFUSAL, 100)
    )
    monkeypatch.setattr(settings, "enable_web_fallback", True)
    monkeypatch.setattr(settings, "web_summary", True)
    hit = WebResult("title", "https://example.com", "SNIPPET", "2026-11-07")
    monkeypatch.setattr(web, "web_supplement", lambda question: [hit])
    monkeypatch.setattr(graph, "summarize", lambda question, results: "Plain summary [1].")
    result = graph.run_agent("q")
    assert result.web == [hit] and result.web_summary == "Plain summary [1]."
    assert all("Plain summary" not in context for context in prompts)


def test_summary_flag_off_never_calls_the_summarizer(monkeypatch):
    say(monkeypatch, REFUSAL)
    monkeypatch.setattr(settings, "enable_web_fallback", True)
    hit = WebResult("title", "https://example.com", "snippet", "2026-11-07")
    monkeypatch.setattr(web, "web_supplement", lambda question: [hit])
    monkeypatch.setattr(graph, "summarize", lambda q, r: pytest.fail("summarizer called"))
    assert graph.run_agent("q").web_summary == ""


def test_budget_stops_the_run(monkeypatch):
    say(monkeypatch, REFUSAL, tokens=settings.agent_token_budget + 1)
    with pytest.raises(BudgetExceeded):
        graph.run_agent("q")


def _planner_models(monkeypatch, failing):
    """Replace the planner's model with one that raises for the models named in `failing`."""
    import insurance_rag.agent.decompose as planner

    tried = []

    def fake(provider, model, max_tokens=None):
        class Chat:
            def invoke(self, messages):
                tried.append(model)
                if model in failing:
                    raise RuntimeError("503 UNAVAILABLE")
                return type("Reply", (), {"content": "- first\n- second"})()

        return Chat()

    monkeypatch.setattr(planner, "chat_model", fake)
    return planner, tried


def test_the_planner_falls_back_when_its_model_is_down(monkeypatch, tmp_path):
    planner, tried = _planner_models(monkeypatch, {"models/gemini-3.1-flash-lite"})
    cache = tmp_path / "cache.jsonl"
    assert planner.decompose("q", cache=cache) == ["first", "second"]
    assert tried == ["models/gemini-3.1-flash-lite", "models/gemini-3.5-flash-lite"]
    assert cache.exists()


def test_the_planner_gives_no_sub_questions_and_caches_nothing_when_both_models_are_down(
    monkeypatch, tmp_path
):
    planner, tried = _planner_models(
        monkeypatch, {"models/gemini-3.1-flash-lite", "models/gemini-3.5-flash-lite"}
    )
    cache = tmp_path / "cache.jsonl"
    assert planner.decompose("q", cache=cache) == [] and not cache.exists()
