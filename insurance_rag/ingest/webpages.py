"""e-Laws HTML -> structural units, one Document per citable provision."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from insurance_rag.config import RAW_HTML_DIR
from insurance_rag.corpus.manifest import ManifestRow

__all__ = ["fetch_html", "units_from_html", "load_web"]

_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*(?:\((\d+(?:\.\d+)*)\)\s)?")
_SUBSECTION_RE = re.compile(r"^\s*\((\d+(?:\.\d+)*)\)\s*")
_DEFINED_TERM_RE = re.compile(r"^\s*[\u201c\"]([^\u201d\"]+)[\u201d\"]")

#: Heading classes and the ancestor slot each one occupies.
_HEADING_DEPTH = {"partnum": 0, "heading1": 1, "heading2": 2, "headnote": 3}
_BODY = {
    "clause", "subclause", "subsubclause", "paragraph", "subpara", "subsubpara",
    "defclause", "defsubclause", "equation", "equationind1", "equationind2",
    "table", "MsoNormal",
}
_SKIP_PREFIXES = ("TOC", "footnote", "Pnote")


def _kind(tag) -> str:
    """Normalise `subsection-e`, `Ssubsection-e` and `subsection` to one name."""
    cls = (tag.get("class") or [""])[0]
    return (cls[:-2] if cls.endswith("-e") else cls).lstrip("SY")


def _text(tag) -> str:
    return re.sub(r"[\s\xa0]+", " ", tag.get_text(" ", strip=True)).strip()


def _table_markdown(table) -> str:
    """Render a data table as pipe-delimited rows; it is never split downstream."""
    rows = [[_text(c) for c in tr.find_all(["td", "th"])] for tr in table.find_all("tr")]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows if any(r))


def fetch_html(row: ManifestRow, *, refresh: bool = False) -> str:
    """Live WebBaseLoader fetch, cached to data/raw_html so eval runs stay reproducible."""
    path = RAW_HTML_DIR / f"{row.doc_id}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    from langchain_community.document_loaders import WebBaseLoader

    html = str(WebBaseLoader(row.source_url).scrape())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


def units_from_html(row: ManifestRow, html: str) -> list[Document]:
    """Walk the DOM in order, emitting one Document per section, subsection or definition."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    for table in soup.find_all("table"):
        if table.select_one('p[class^="TOC"]'):  # the table of contents is itself a table
            table.decompose()

    units: list[Document] = []
    ancestors: list[str | None] = [None] * 4
    section = provision = ""
    cur: dict | None = None

    def flush() -> None:
        nonlocal cur
        if cur and any(part.strip() for part in cur["parts"]):
            units.append(
                Document(
                    page_content="\n".join(cur["parts"]).strip(),
                    metadata={
                        "doc_id": row.doc_id,
                        "locator_path": cur["path"],
                        "heading": cur["ancestors"][-1] if cur["ancestors"] else "",
                        "ancestor_path": cur["ancestors"],
                        "is_table": cur["is_table"],
                    },
                )
            )
        cur = None

    def start(path: str, first: str, *, is_table: bool = False) -> None:
        nonlocal cur
        flush()
        cur = {
            "path": path,
            "parts": [first] if first else [],
            "ancestors": [a for a in ancestors if a],
            "is_table": is_table,
        }

    for el in soup.find_all(["p", "table"]):
        if el.name == "table":
            markdown = _table_markdown(el)
            if markdown:
                start(provision or section, markdown, is_table=True)
                flush()
            continue
        if el.find_parent("table"):
            continue  # already emitted as part of its table

        kind = _kind(el)
        if kind.startswith(_SKIP_PREFIXES):
            continue
        text = _text(el)
        if not text:
            continue

        if kind in _HEADING_DEPTH:
            flush()
            depth = _HEADING_DEPTH[kind]
            ancestors[depth] = text
            for deeper in range(depth + 1, len(ancestors)):
                ancestors[deeper] = None
        elif kind == "section":
            m = _SECTION_RE.match(text)
            if not m:
                continue
            section = m.group(1)
            provision = f"{section}({m.group(2)})" if m.group(2) else section
            start(provision, text[m.end():].strip())
        elif kind == "subsection":
            m = _SUBSECTION_RE.match(text)
            provision = f"{section}({m.group(1)})" if m else section
            start(provision, text[m.end():].strip() if m else text)
        elif kind in {"definition", "firstdef"}:
            m = _DEFINED_TERM_RE.match(text)
            start(f"{provision} \u201c{m.group(1)}\u201d" if m else provision, text)
        elif kind in _BODY and cur is not None:
            cur["parts"].append(text)

    flush()
    return [u for u in units if u.metadata["locator_path"]]


def load_web(row: ManifestRow, *, refresh: bool = False) -> list[Document]:
    """Fetch (or reuse) the e-Laws page and return its citable provisions in order."""
    return units_from_html(row, fetch_html(row, refresh=refresh))
