"""The web summary's faithfulness check and failure handling, with the model faked: no network."""

from __future__ import annotations

import pytest

from insurance_rag.agent import web_summary
from insurance_rag.agent.web import WebResult, render_web_block

IBC = WebResult(
    "Mandatory auto coverages - IBC",
    "https://www.ibc.ca/x",
    "snippet",
    "2026-11-07",
    passages=("The minimum is Third Party Liability ($200,000), Accident Benefits and more.",),
)
MTO = WebResult("Ontario.ca", "https://www.ontario.ca/y", "Every vehicle must be insured.", "d")
SOURCES = [IBC, MTO]


def test_a_summary_using_only_source_facts_passes():
    text = "Ontario needs $200,000 of liability cover [1]. Every vehicle must be insured [2]."
    assert web_summary.faithful(text, SOURCES)


def test_several_sources_in_one_bracket_are_understood():
    text = "Every vehicle must be insured [1, 2]. Cover is $200,000 [1; 2]."
    assert web_summary.faithful(text, SOURCES)
    assert not web_summary.faithful("Every vehicle must be insured [2, 5].", SOURCES)


def test_an_invented_number_fails():
    assert not web_summary.faithful("Cover of $500,000 is required [1].", SOURCES)


def test_a_sentence_without_a_source_marker_fails():
    assert not web_summary.faithful("Cover is $200,000 [1]. Everyone must have it.", SOURCES)


def test_a_marker_for_a_missing_source_fails():
    assert not web_summary.faithful("Every vehicle must be insured [3].", SOURCES)


def test_an_empty_or_oversized_summary_fails():
    assert not web_summary.faithful("   ", SOURCES)
    assert not web_summary.faithful("Every vehicle must be insured [2]. " * 60, SOURCES)


@pytest.fixture
def model_says(monkeypatch):
    def say(text):
        monkeypatch.setattr(web_summary, "_ask", lambda question, results: text)

    return say


def test_summarize_returns_a_faithful_summary(model_says):
    model_says("Every vehicle must be insured [2].")
    assert web_summary.summarize("q", SOURCES) == "Every vehicle must be insured [2]."


def test_summarize_drops_an_unfaithful_summary(model_says):
    model_says("Fines can reach $99,999 [1].")
    assert web_summary.summarize("q", SOURCES) == ""


def test_summarize_survives_a_model_failure(monkeypatch):
    def broken(question, results):
        raise RuntimeError("quota exhausted")

    monkeypatch.setattr(web_summary, "_ask", broken)
    assert web_summary.summarize("q", SOURCES) == ""


def test_no_results_means_no_model_call(monkeypatch):
    monkeypatch.setattr(web_summary, "_ask", lambda question, results: pytest.fail("model called"))
    assert web_summary.summarize("q", []) == ""


def test_the_prompt_treats_sources_as_untrusted_and_numbered():
    text = web_summary._source_text(SOURCES)
    assert '<source id="1"' in text and '<source id="2"' in text
    assert "untrusted" in web_summary.SUMMARY_SYSTEM


def test_the_block_shows_the_summary_above_numbered_sources():
    block = render_web_block(SOURCES, "Every vehicle must be insured [2].")
    assert block.index("Plain-language summary") < block.index("[1] Mandatory auto coverages")
    assert "not legal advice" in block
