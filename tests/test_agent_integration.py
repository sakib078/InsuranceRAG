"""The agent's retrieval nodes over the real corpus: needs Postgres and the encoder, no LLM."""

from __future__ import annotations

import pytest

from evals.validate_golden import GOLDEN_PATH, load_records
from insurance_rag.agent import graph
from insurance_rag.config import settings


@pytest.mark.slow
def test_exclusion_pass_runs_for_every_golden_question(monkeypatch):
    monkeypatch.setattr(settings, "agent_decompose", False)
    for record in load_records(GOLDEN_PATH):
        state = graph._run(record["question"], retrieval_only=True)
        assert "exclusion" in state["checked"], record["id"]
