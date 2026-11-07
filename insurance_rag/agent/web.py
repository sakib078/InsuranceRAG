"""The web supplement: public results shown beneath a refusal, never inside an answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from insurance_rag.config import settings

__all__ = ["WebResult", "web_supplement", "render_web_block"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebResult:
    """One search hit; the snippet is verbatim from the provider, never model-written."""

    title: str
    url: str
    snippet: str
    retrieved_at: str


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _ddgs(question: str, k: int) -> list[WebResult]:
    from ddgs import DDGS

    hits = DDGS().text(question, max_results=k)
    return [WebResult(h["title"], h["href"], h["body"], _today()) for h in hits]


def _searxng(question: str, k: int) -> list[WebResult]:
    """The self-hosted fallback; needs `IRAG_SEARXNG_URL` and `json` in its `formats`."""
    if not settings.searxng_url:
        return []
    import httpx

    reply = httpx.get(
        f"{settings.searxng_url.rstrip('/')}/search",
        params={"q": question, "format": "json"},
        timeout=10,
    )
    reply.raise_for_status()
    return [
        WebResult(r["title"], r["url"], r.get("content", ""), _today())
        for r in reply.json()["results"][:k]
    ]


def web_supplement(question: str, *, k: int = 3) -> list[WebResult]:
    """Public results for a refused question: ddgs first, SearXNG if it fails or finds none."""
    for backend in (_ddgs, _searxng):
        try:
            results = backend(question, k)
        except Exception as exc:  # the web tier must never break the refusal it sits under
            log.warning("web backend %s failed: %s", backend.__name__, exc)
            continue
        if results:
            return results
    return []


def render_web_block(results: list[WebResult]) -> str:
    """The results as an unmistakably separate block: unverified and not exclusion-checked."""
    if not results:
        return ""
    rule = "-" * 64
    lines = [rule, "WEB RESULTS: not part of the corpus, unverified, not exclusion-checked", rule]
    for result in results:
        lines += [result.title, f"  {result.snippet}"]
        lines.append(f"  {result.url} (retrieved {result.retrieved_at})")
    return "\n".join(lines)
