"""FSRA/OAP PDFs -> structural units, via layout-aware Docling with PyMuPDF for pages."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from insurance_rag.corpus.manifest import ManifestRow

__all__ = ["load_pdf", "page_index", "units_from_markdown"]

#: OAP 1 headings carry their clause number: "1.8 Who and What We Won't Cover".
_CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.*)$")
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

#: A page below this many characters is an image, not text - flag it, never index it.
SCANNED_PAGE_CHARS = 60

#: Below this share of the raw text, Docling is dropping pages - fall back to PyMuPDF.
COVERAGE_FLOOR = 0.6


def page_index(path: Path) -> tuple[list[str], set[int]]:
    """Per-page text plus the 1-based pages that are scans, from PyMuPDF."""
    import pymupdf

    pages, scanned = [], set()
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            pages.append(text)
            if len(text) < SCANNED_PAGE_CHARS:
                scanned.add(number)
    return pages, scanned


def _normalise(text: str) -> str:
    """Letters and digits only - Docling reorders columns, so spacing never lines up."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _locate_page(text: str, normalised_pages: list[str]) -> int | None:
    """First page containing this unit's opening words, compared on letters alone."""
    probe = _normalise(text)[:60]
    if len(probe) < 20:
        return None
    for number, page in enumerate(normalised_pages, start=1):
        if probe in page:
            return number
    return None


def _is_contents(heading: str, text: str) -> bool:
    """OAP 1's 164-entry embedded TOC is content to skip, not a provision to index."""
    if re.fullmatch(r"(table of )?contents", heading.strip().lower()):
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 8:
        return False
    leaders = sum(bool(re.search(r"(\.{3,}\s*\d+|\s\d+\s*\|?\s*$)", ln)) for ln in lines)
    return leaders / len(lines) > 0.7


def units_from_markdown(row: ManifestRow, markdown: str, pages: list[str]) -> list[Document]:
    """Split Docling markdown on its heading stack; the heading gives the locator."""
    splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    normalised = [_normalise(page) for page in pages]
    units: list[Document] = []

    for doc in splitter.split_text(markdown):
        text = doc.page_content.strip()
        if not text:
            continue
        stack = [doc.metadata[key] for _, key in _HEADERS if doc.metadata.get(key)]
        heading = stack[-1] if stack else ""
        if _is_contents(heading, text):
            continue
        matched = _CLAUSE_RE.match(heading)
        # Content before the first heading still belongs to the document - never drop it.
        path = matched.group(1) if matched else re.sub(r"\s+", " ", heading).strip() or "Preamble"
        units.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": row.doc_id,
                    "locator_path": path,
                    "heading": heading,
                    "ancestor_path": stack[:-1],
                    "is_table": "|" in text and text.count("\n|") >= 2,
                    "page": _locate_page(text, normalised),
                },
            )
        )
    return units


def _page_units(row: ManifestRow, pages: list[str], scanned: set[int]) -> list[Document]:
    """One unit per page, cited as `p. N`. Used when Docling drops most of the document."""
    return [
        Document(
            page_content=text.strip(),
            metadata={
                "doc_id": row.doc_id,
                "locator_path": f"p. {number}",
                "heading": "",
                "ancestor_path": [],
                "is_table": False,
                "page": number,
            },
        )
        for number, text in enumerate(pages, start=1)
        if number not in scanned and text.strip()
    ]


def load_pdf(row: ManifestRow) -> list[Document]:
    """Docling for layout and tables, PyMuPDF for page numbers, scans, and coverage rescue."""
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    pages, scanned = page_index(row.raw_path)
    loader = DoclingLoader(file_path=str(row.raw_path), export_type=ExportType.MARKDOWN)
    markdown = "\n\n".join(d.page_content for d in loader.load())

    units = [
        u for u in units_from_markdown(row, markdown, pages)
        if u.metadata.get("page") not in scanned
    ]
    # Docling's layout model discards whole pages on print-to-PDF web pages. Never lose text.
    raw_chars = sum(len(_normalise(p)) for p in pages)
    kept = sum(len(_normalise(u.page_content)) for u in units)
    if raw_chars and kept / raw_chars < COVERAGE_FLOOR:
        return _page_units(row, pages, scanned)
    return units
