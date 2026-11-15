"""Rule-based chunk roles: heading keywords must start a word; bars on recovery are exclusions."""

from insurance_rag.ingest.roles import classify_role
from insurance_rag.schema import ChunkRole


def test_a_bar_on_suing_is_an_exclusion_even_under_an_uninsured_heading():
    trail = ("PART VI AUTOMOBILE INSURANCE", "No action by uninsured owner or lessee")
    text = "Despite any other Act, a person is not entitled in an action in Ontario to recover"
    assert classify_role(text, trail, "R.S.O. 1990, c. I.8 s. 267.6(1)") == ChunkRole.EXCLUSION


def test_a_keyword_inside_a_longer_word_does_not_tag_the_heading():
    trail = ("PART X", "Fraternal Societies")
    assert classify_role("A society may apply.", trail, "R.S.O. 1990, c. I.8 s. 340(1)") == "other"


def test_a_keyword_that_starts_a_word_still_tags_the_heading():
    assert classify_role("Text.", ("Rates and risk classification",), "s. 1") == ChunkRole.SCHEDULE
    assert classify_role("Text.", ("Insured persons",), "s. 2") == ChunkRole.COVERAGE
