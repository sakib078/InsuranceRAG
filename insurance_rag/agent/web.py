"""The web supplement: public results shown beneath a refusal, never inside an answer."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from insurance_rag.config import settings

__all__ = ["WebResult", "web_supplement", "render_web_block"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebResult:
    """One hit; the snippet and passages are verbatim from the web, never model-written."""

    title: str
    url: str
    snippet: str
    retrieved_at: str
    passages: tuple[str, ...] = ()


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


#: Sponsored-link redirects some backends mix into organic results.
AD_MARKERS = ("bing.com/aclick", "duckduckgo.com/y.js", "googleadservices.com", "/aclk?")

#: User-generated and social sites are not sources for a legal-information product.
BLOCKED_DOMAINS = (
    "reddit.com", "facebook.com", "quora.com", "youtube.com", "youtu.be", "x.com", "twitter.com",
    "instagram.com", "tiktok.com", "pinterest.com", "linkedin.com",
)

#: URL fragments of French-language pages; the corpus and the product are English only.
FRENCH_URL_MARKERS = ("/fr/", "/fr-ca/", "//fr.")

_EN = frozenset(
    "the and of to in is for a on with are that you your this it as by or be an at from "
    "can will not".split()
)
_FR = frozenset(
    "le la les du des de est pour une un dans sur aux au qui que nous vous vos votre avec sont "
    "ses et ce cette assurance automobile obligatoire conducteur véhicule comment fait lieu".split()
)


def _is_english(text: str) -> bool:
    """English unless French words and accents outnumber English ones; no markers is English."""
    lowered = text.lower()
    words = re.findall(r"[a-zà-ÿ']+", lowered)
    english = sum(word in _EN for word in words)
    french = sum(word in _FR for word in words) + sum(c in "àâçéèêëîïôûùüÿœ" for c in lowered)
    return french <= english


def _blocked(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)


def _usable(results: list[WebResult]) -> list[WebResult]:
    """Organic, English-only results; a bilingual snippet keeps only its English parts."""
    kept = []
    for result in results:
        if _blocked(result.url) or any(m in result.url for m in AD_MARKERS + FRENCH_URL_MARKERS):
            continue
        parts = [p.strip() for p in result.snippet.split("|") if p.strip()]
        english = [p for p in parts if _is_english(p)]
        snippet = result.snippet if len(english) == len(parts) else " ".join(english)
        if not _is_english(result.title) or (parts and not english):
            continue
        kept.append(replace(result, snippet=snippet))
    return kept


def _ddgs(question: str, k: int) -> list[WebResult]:
    from ddgs import DDGS

    try:
        hits = DDGS().text(question, region="ca-en", max_results=k)
    except Exception as exc:
        if "No results found" in str(exc):  # ddgs raises on an empty page; that is not a failure
            return []
        raise
    return [WebResult(h["title"], h["href"], h["body"], _today()) for h in hits]


def _searxng(question: str, k: int) -> list[WebResult]:
    """The self-hosted fallback; needs `IRAG_SEARXNG_URL` and `json` in its `formats`."""
    if not settings.searxng_url:
        return []
    base = settings.searxng_url.strip().rstrip("/\\")  # tolerate a stray slash or backslash
    reply = httpx.get(
        f"{base}/search",
        params={"q": question, "format": "json", "language": "en"},
        timeout=10,
    )
    reply.raise_for_status()
    return [
        WebResult(r["title"], r["url"], r.get("content", ""), _today())
        for r in reply.json()["results"][:k]
    ]


#: Candidates asked of each backend before ranking keeps the best few.
CANDIDATES = 10

#: Domain priority: official sources and regulators first, auto-insurance information sites next.
TIER1 = (
    "ontario.ca", "fsrao.ca", "ibc.ca", "insurancehotline.com", "tribunalsontario.ca",
    "canlii.org", "ribo.com", "giocanada.org", "lso.ca",
)
TIER2 = ("ratehub.ca", "rates.ca", "caa.ca", "insurance-canada.ca")
#: Insurer and broker sales pages sell policies, so they are dropped (no recommending).
TIER3 = (
    "tdinsurance.com", "belairdirect.com", "intact.ca", "aviva.ca", "allstate.ca", "sonnet.ca",
    "rbcinsurance.com", "scotiainsurance.com", "pcinsurance.ca", "westernfinancialgroup.ca",
    "mitchinsurance.com", "thepersonal.com", "cooperators.ca", "wawanesa.com", "caasco.com",
    "acumeninsurance.com", "myinsurancebroker.com", "easilyinsured.ca", "surex.com",
    "mcdougallinsurance.com", "westlandinsurance.ca", "e2gold.ca",
)
SALES_WORDS = ("insur", "broker")
TIER_BONUS = {1: 0.25, 2: 0.05, 3: -0.15}


def _query(question: str) -> str:
    """The question plus the Ontario and auto-insurance context it may leave out."""
    lowered = question.lower()
    extra = [w for w in ("Ontario", "auto insurance") if w.lower() not in lowered]
    return " ".join([question.strip(), *extra])


def _dedupe(results: list[WebResult]) -> list[WebResult]:
    """One result per URL, ignoring a trailing slash or fragment."""
    seen: dict[str, WebResult] = {}
    for result in results:
        seen.setdefault(result.url.split("#")[0].rstrip("/"), result)
    return list(seen.values())


def _tier_bonus(url: str) -> float:
    """A small ranking nudge by domain tier; it never rescues an irrelevant result."""
    host = urlparse(url).netloc.lower()
    for tier, domains in ((1, TIER1), (2, TIER2), (3, TIER3)):
        if any(host == d or host.endswith("." + d) for d in domains):
            return TIER_BONUS[tier]
    if any(word in host for word in SALES_WORDS):  # an unlisted insurer or broker
        return TIER_BONUS[3]
    return 0.0


STOPWORDS = frozenset(
    "the and for are was were does did can how what when why which who you your with about from "
    "will would should could not any all but have has had".split()
)


def _terms(text: str) -> set[str]:
    """The distinctive words of a text, lowercased with a plural 's' removed."""
    words = re.findall(r"[a-z0-9$']+", text.lower())
    return {w.removesuffix("s") if len(w) > 3 else w for w in words if len(w) > 2} - STOPWORDS


def _wanted(question: str) -> set[str]:
    """What a relevant page mentions: the question's own distinctive terms, and Ontario."""
    return _terms(question) | {"ontario"}


def _overlap(question: str, text: str) -> float:
    """The share of the wanted terms that the text contains."""
    wanted = _wanted(question)
    return len(wanted & _terms(text)) / len(wanted)


def _relevant(question: str, results: list[WebResult]) -> list[WebResult]:
    """Results above the relevance floor, best first (term overlap plus the domain nudge)."""
    kept = []
    for result in results:
        score = _overlap(question, f"{result.title}. {result.snippet}")
        if score >= settings.web_min_relevance:
            kept.append((score + _tier_bonus(result.url), result))
    return [result for _, result in sorted(kept, key=lambda pair: pair[0], reverse=True)]


USER_AGENT = "insurance-rag-research/0.1"
MAX_PAGE_BYTES = 1_000_000
MAX_BLOCKS = 150  # paragraphs scored per page
PASSAGE_CHARS = (60, 600)  # shorter is a label, longer is a wall of text
PASSAGES_PER_PAGE = 3
NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]


def _allowed(url: str) -> bool:
    """Whether the site's robots.txt lets us fetch this page; an unreachable one means yes."""
    parts = urlparse(url)
    try:
        robots = httpx.get(
            f"{parts.scheme}://{parts.netloc}/robots.txt",
            headers={"User-Agent": USER_AGENT},
            timeout=5,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return True
    parser = RobotFileParser()
    parser.parse(robots.text.splitlines() if robots.status_code == 200 else [])
    return parser.can_fetch(USER_AGENT, url)


def _fetch(url: str) -> str | None:
    """The page's HTML, or None if robots.txt forbids it, it is not HTML, or it is too large."""
    if not _allowed(url):
        return None
    reply = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=10, follow_redirects=True
    )
    reply.raise_for_status()
    too_big = len(reply.content) > MAX_PAGE_BYTES
    if "html" not in reply.headers.get("content-type", "") or too_big:
        return None
    return reply.text


def _passages(html: str) -> list[str]:
    """The page's readable English paragraphs, without navigation, scripts or boilerplate."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    blocks = (" ".join(el.get_text(" ", strip=True).split()) for el in soup.find_all(["p", "li"]))
    low, high = PASSAGE_CHARS
    unique = dict.fromkeys(b for b in blocks if low <= len(b) <= high and _is_english(b))
    return list(unique)[:MAX_BLOCKS]


def _best_passages(question: str, passages: list[str]) -> list[str]:
    """Ontario passages sharing most of the question's terms, in page order."""
    wanted = _wanted(question)
    need = max(2, math.ceil(0.6 * len(wanted)))
    scored = [(len(wanted & _terms(p)), -i) for i, p in enumerate(passages)]
    ontario = [s for s in scored if "ontario" in _terms(passages[-s[1]])]
    best = sorted((s for s in ontario if s[0] >= need), reverse=True)[:PASSAGES_PER_PAGE]
    return [passages[-i] for _, i in sorted(best, key=lambda s: -s[1])]


def _with_passages(question: str, result: WebResult) -> WebResult:
    """The result with verbatim page passages; unchanged if the page cannot be read."""
    try:
        html = _fetch(result.url)
        passages = _best_passages(question, _passages(html)) if html else []
    except Exception as exc:  # one unreadable page must not lose the others
        log.warning("could not read %s: %s", result.url, exc)
        return result
    return replace(result, passages=tuple(passages))


def _candidates(query: str) -> list[WebResult]:
    """Usable results from every backend, de-duplicated; one failing backend loses nothing else."""
    found: list[WebResult] = []
    for backend in (_ddgs, _searxng):
        try:
            found += backend(query, CANDIDATES)
        except Exception as exc:  # the web tier must never break the refusal it sits under
            log.warning("web backend %s failed: %s", backend.__name__, exc)
    return _dedupe(_usable(found))


def web_supplement(question: str, *, k: int = 3) -> list[WebResult]:
    """The most relevant non-sales public pages for a refused question, with their best passages."""
    ranked = _relevant(question, _candidates(_query(question)))
    results = [r for r in ranked if _tier_bonus(r.url) != TIER_BONUS[3]]  # business rule 1
    return [_with_passages(question, result) for result in results[:k]]


def render_web_block(results: list[WebResult], summary: str = "") -> str:
    """The results as an unmistakably separate block: unverified and not exclusion-checked."""
    if not results:
        return ""
    rule = "-" * 64
    lines = [rule, "WEB RESULTS: not part of the corpus, unverified, not exclusion-checked", rule]
    if summary:
        lines += ["Plain-language summary (unverified, not legal advice):", summary, ""]
    for number, result in enumerate(results, start=1):
        lines.append(f"[{number}] {result.title}")
        lines += [f"  {text}" for text in result.passages or (result.snippet,)]
        lines.append(f"  {result.url} (retrieved {result.retrieved_at})")
    return "\n".join(lines)
