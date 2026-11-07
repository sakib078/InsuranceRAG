"""Web backend order and failure handling, with both backends faked: no network."""

from __future__ import annotations

from insurance_rag.agent import web
from insurance_rag.agent.web import WebResult, render_web_block

HIT = WebResult("title", "https://example.com", "snippet", "2026-11-06")


def test_ddgs_answers_first(monkeypatch):
    monkeypatch.setattr(web, "_ddgs", lambda question, k: [HIT])
    monkeypatch.setattr(web, "_searxng", lambda question, k: [])
    assert web.web_supplement("q") == [HIT]


def test_searxng_covers_a_failing_ddgs(monkeypatch):
    def broken(question, k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(web, "_ddgs", broken)
    monkeypatch.setattr(web, "_searxng", lambda question, k: [HIT])
    assert web.web_supplement("q") == [HIT]


def test_no_backend_means_no_results_and_no_error(monkeypatch):
    monkeypatch.setattr(web, "_ddgs", lambda question, k: [])
    monkeypatch.setattr(web, "_searxng", lambda question, k: [])
    assert web.web_supplement("q") == []
    assert render_web_block([]) == ""


def test_rendered_block_is_marked_unverified():
    block = render_web_block([HIT])
    assert "unverified" in block and "not exclusion-checked" in block and "2026-11-06" in block
