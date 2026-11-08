"""Web backend order and failure handling, with both backends faked: no network."""

from __future__ import annotations

import pytest

from insurance_rag.agent import web
from insurance_rag.agent.web import WebResult, render_web_block

HIT = WebResult("title", "https://example.com", "snippet", "2026-11-06")


@pytest.fixture(autouse=True)
def every_result_is_relevant(monkeypatch):
    monkeypatch.setattr(web.settings, "web_min_relevance", 0.0)
    monkeypatch.setattr(web, "_fetch", lambda url: None)  # no page reads, no network


PAGE = """<html><head><script>var tracking = 1;</script></head><body>
<nav><p>Home and about us and contact and our sites map and a long menu of text</p></nav>
<p>Short label</p>
<p>Auto insurance is mandatory for every vehicle registered in Ontario, by law.</p>
<ul><li>Mandatory auto insurance in Ontario covers liability and accident benefits.</li></ul>
<p>Les propriétaires de véhicules doivent souscrire une assurance pour conducteurs.</p>
<footer><p>Copyright notice and terms of use for this website and all of its pages</p></footer>
</body></html>"""


def test_page_text_keeps_only_readable_english_paragraphs():
    found = web._passages(PAGE)
    assert found == [
        "Auto insurance is mandatory for every vehicle registered in Ontario, by law.",
        "Mandatory auto insurance in Ontario covers liability and accident benefits.",
    ]


def test_best_passages_need_the_questions_terms_including_ontario():
    texts = [
        "Fraud affects insurance for every driver in the province.",
        "The minimum for mandatory auto insurance in Newfoundland is $200,000.",
        "Mandatory auto insurance in Ontario means every vehicle must be insured.",
        "Ontario drivers must carry mandatory auto insurance on each registered vehicle.",
    ]
    found = web._best_passages("Is auto insurance mandatory in Ontario?", texts)
    assert found == texts[2:]


def test_best_passages_keep_page_order_and_cap_the_count():
    base = "Mandatory auto insurance in Ontario rule number {}."
    texts = [base.format(i) for i in range(6)]
    assert web._best_passages("Is auto insurance mandatory in Ontario?", texts) == texts[:3]


def test_a_page_is_read_and_its_passages_attached(monkeypatch):
    monkeypatch.setattr(web, "_fetch", lambda url: PAGE)
    result = web._with_passages("Is auto insurance mandatory in Ontario?", HIT)
    assert len(result.passages) == 2 and result.snippet == HIT.snippet


def test_an_unreadable_page_leaves_the_result_as_it_was(monkeypatch):
    def broken(url):
        raise RuntimeError("timeout")

    monkeypatch.setattr(web, "_fetch", broken)
    assert web._with_passages("q", HIT) == HIT


def test_robots_txt_can_forbid_a_page(monkeypatch):
    class Robots:
        status_code = 200
        text = "User-agent: *\nDisallow: /private/"

    monkeypatch.setattr(web.httpx, "get", lambda *a, **k: Robots())
    assert web._allowed("https://example.com/public/page") is True
    assert web._allowed("https://example.com/private/page") is False


def test_passages_are_rendered_instead_of_the_snippet():
    both = ("passage one", "passage two")
    page = WebResult("T", "https://a.ca/x", "snippet text", "d", passages=both)
    block = render_web_block([page])
    assert "passage one" in block and "passage two" in block and "snippet text" not in block


def test_query_gains_the_missing_context():
    assert web._query("Do tickets raise my rates?").endswith("Ontario auto insurance")
    complete = "Is auto insurance mandatory in Ontario?"
    assert web._query(complete) == complete


def test_candidates_are_deduplicated_by_url():
    twin = WebResult("same page", "https://example.com/", "other snippet", "2026-11-06")
    assert web._dedupe([HIT, twin]) == [HIT]


def test_official_domains_outrank_sales_pages_at_equal_relevance(monkeypatch):
    sales = WebResult("sales", "https://www.tdinsurance.com/auto", "s", "d")
    blog = WebResult("blog", "https://example.org/post", "s", "d")
    official = WebResult("official", "https://www.ontario.ca/page/x", "s", "d")
    ranked = web._relevant("q", [sales, blog, official])
    assert [r.title for r in ranked] == ["official", "blog", "sales"]


def test_unlisted_insurers_and_brokers_rank_as_sales_pages():
    assert web._tier_bonus("https://www.vlginsure.com/x") < 0
    assert web._tier_bonus("https://chaseinsurancebrokers.ca/x") < 0
    assert web._tier_bonus("https://www.insurancehotline.com/x") > 0


def test_domain_match_is_exact_not_a_substring():
    assert web._tier_bonus("https://notontario.ca/x") == 0.0
    assert web._tier_bonus("https://www.fsrao.ca/x") > 0


def test_irrelevant_results_are_dropped_even_from_trusted_domains(monkeypatch):
    monkeypatch.setattr(web.settings, "web_min_relevance", 0.42)
    blurb = "Ontario's vibrant landscapes make this a great place to live and to visit."
    official = WebResult("Welcome", "https://www.ontario.ca/page/x", blurb, "d")
    monkeypatch.setattr(web, "_ddgs", lambda question, k: [official])
    monkeypatch.setattr(web, "_searxng", lambda question, k: [])
    assert web.web_supplement("Is auto insurance mandatory in Ontario?") == []


def test_a_backend_with_nothing_relevant_falls_through(monkeypatch):
    monkeypatch.setattr(web.settings, "web_min_relevance", 0.42)
    off_topic = WebResult("off topic", "https://a.ca/1", "s", "d")
    relevant = WebResult("Mandatory auto insurance in Ontario", "https://a.ca/2", "A rule.", "d")
    monkeypatch.setattr(web, "_ddgs", lambda question, k: [off_topic])
    monkeypatch.setattr(web, "_searxng", lambda question, k: [relevant])
    assert web.web_supplement("Is auto insurance mandatory in Ontario?") == [relevant]


def test_insurer_and_broker_sales_pages_are_dropped(monkeypatch):
    sales = WebResult("Mandatory auto insurance in Ontario", "https://www.intact.ca/x", "s", "d")
    info = WebResult("Mandatory auto insurance in Ontario", "https://www.fsrao.ca/x", "s", "d")
    monkeypatch.setattr(web, "_ddgs", lambda question, k: [sales, info])
    monkeypatch.setattr(web, "_searxng", lambda question, k: [])
    assert web.web_supplement("Is auto insurance mandatory in Ontario?") == [info]


def test_overlap_counts_the_questions_terms_in_the_text():
    question = "Is auto insurance mandatory in Ontario?"
    assert web._overlap(question, "Mandatory auto insurance in Ontario") == 1.0
    assert web._overlap(question, "A great place to live") == 0.0


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


def test_ad_links_are_dropped(monkeypatch):
    ad = WebResult("ad", "https://www.bing.com/aclick?ld=abc", "buy now", "2026-11-07")
    monkeypatch.setattr(web, "_ddgs", lambda question, k: [ad, HIT])
    monkeypatch.setattr(web, "_searxng", lambda question, k: [])
    assert web.web_supplement("q") == [HIT]


def test_only_ads_falls_through_to_the_next_backend(monkeypatch):
    ad = WebResult("ad", "https://duckduckgo.com/y.js?ad=1", "buy now", "2026-11-07")
    monkeypatch.setattr(web, "_ddgs", lambda question, k: [ad])
    monkeypatch.setattr(web, "_searxng", lambda question, k: [HIT])
    assert web.web_supplement("q") == [HIT]


def test_searxng_url_tolerates_a_stray_backslash(monkeypatch):
    seen = {}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"title": "t", "url": "https://example.com", "content": "c"}]}

    def fake_get(url, **kwargs):
        seen["url"] = url
        return Reply()

    monkeypatch.setattr(web.settings, "searxng_url", "http://localhost:8080\\")
    monkeypatch.setattr("httpx.get", fake_get)
    assert web._searxng("q", 3)[0].url == "https://example.com"
    assert seen["url"] == "http://localhost:8080/search"


def test_ddgs_empty_page_is_not_a_failure(monkeypatch, caplog):
    import ddgs

    class Empty:
        def text(self, question, **kwargs):
            raise RuntimeError("No results found.")

    monkeypatch.setattr(ddgs, "DDGS", Empty)
    monkeypatch.setattr(web, "_searxng", lambda question, k: [])
    assert web.web_supplement("q") == []
    assert "failed" not in caplog.text


def test_bilingual_snippet_keeps_only_its_english_half():
    blurb = (
        "Du fait de son multiculturalisme dynamique et de ses paysages diversifiés, l'Ontario "
        "est un lieu idéal de résidence et de visite.| Ontario's vibrant multiculturalism and "
        "varied landscapes make this a great place to live and to visit."
    )
    both = WebResult("Vehicle insurance and registration", "https://www.ontario.ca/x", blurb, "d")
    (kept,) = web._usable([both])
    assert kept.snippet.startswith("Ontario's vibrant") and "Du fait" not in kept.snippet


def test_french_results_are_dropped():
    french_title = WebResult("Assurance automobile obligatoire", "https://a.ca/x", "", "d")
    french_text = WebResult(
        "Insurance", "https://a.ca/y", "Votre assurance est obligatoire pour les conducteurs.", "d"
    )
    french_url = WebResult("Auto insurance", "https://www.ontario.ca/fr/x", "Auto insurance.", "d")
    assert web._usable([french_title, french_text, french_url]) == []


def test_social_and_user_generated_sites_are_blocked():
    text = "Auto insurance in Ontario."
    posts = [
        WebResult("Thread", "https://www.reddit.com/r/ontario/x", text, "d"),
        WebResult("Post", "https://m.facebook.com/p/1", text, "d"),
        WebResult("Answer", "https://www.quora.com/q", text, "d"),
    ]
    assert web._usable(posts) == []
    assert web._usable([WebResult("Auto", "https://notreddit.com/x", "Auto insurance.", "d")])


def test_plain_english_results_pass_untouched():
    english = WebResult("Auto Insurance Ontario", "https://a.ca/z", "Home | Auto | Life", "d")
    assert web._usable([english]) == [english]


def test_rendered_block_is_marked_unverified():
    block = render_web_block([HIT])
    assert "unverified" in block and "not exclusion-checked" in block and "2026-11-06" in block
