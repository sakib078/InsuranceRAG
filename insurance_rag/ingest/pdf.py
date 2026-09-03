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


def _locate_page(text: str, pages: list[str]) -> int | None:
    """First page containing this unit's opening words."""
    probe = " ".join(text.split()[:8])
    if not probe:
        return None
    for number, page in enumerate(pages, start=1):
        if probe in " ".join(page.split()):
            return number
    return None


def units_from_markdown(row: ManifestRow, markdown: str, pages: list[str]) -> list[Document]:
    """Split Docling markdown on its heading stack; the heading gives the locator."""
    splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    units: list[Document] = []

    for doc in splitter.split_text(markdown):
        text = doc.page_content.strip()
        if not text:
            continue
        stack = [doc.metadata[key] for _, key in _HEADERS if doc.metadata.get(key)]
        heading = stack[-1] if stack else ""
        matched = _CLAUSE_RE.match(heading)
        path = matched.group(1) if matched else re.sub(r"\s+", " ", heading).strip()
        if not path:
            continue
        units.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": row.doc_id,
                    "locator_path": path,
                    "ancestor_path": stack[:-1],
                    "is_table": "|" in text and text.count("\n|") >= 2,
                    "page": _locate_page(text, pages),
                },
            )
        )
    return units


def load_pdf(row: ManifestRow) -> list[Document]:
    """Docling for layout and tables, PyMuPDF for page numbers and scanned-page detection."""
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    pages, scanned = page_index(row.raw_path)
    loader = DoclingLoader(file_path=str(row.raw_path), export_type=ExportType.MARKDOWN)
    markdown = "\n\n".join(d.page_content for d in loader.load())

    units = units_from_markdown(row, markdown, pages)
    return [u for u in units if u.metadata.get("page") not in scanned]
