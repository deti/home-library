"""EPUB processing utilities.

Provides structured extraction of book-level metadata, table of contents,
chapter ordering and text suitable for RAG pipelines.

The public API prefers modern Python data structures (Pydantic models)
instead of raw dicts.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from bs4 import BeautifulSoup
from ebooklib import epub
from pydantic import BaseModel, Field


class Identifier(BaseModel):
    scheme: str | None = Field(
        default=None, description="Identifier scheme, e.g., ISBN"
    )
    value: str = Field(description="Identifier value")


class Person(BaseModel):
    name: str
    role: str | None = Field(default=None, description="Role like 'author', 'editor'")


class TOCItem(BaseModel):
    title: str | None
    href: str | None
    children: list["TOCItem"] = Field(default_factory=list)


class Chapter(BaseModel):
    index: int = Field(description="Zero-based index in reading order")
    title: str | None
    href: str | None
    text: str = Field(description="Plain text content extracted from HTML")
    word_count: int = 0


class BookInfo(BaseModel):
    title: str | None
    authors: list[str] = Field(default_factory=list)
    language: str | None = None
    publisher: str | None = None
    description: str | None = None
    subjects: list[str] = Field(default_factory=list)
    identifiers: list[Identifier] = Field(default_factory=list)


class EpubDetails(BaseModel):
    path: str
    metadata: BookInfo
    toc: list[TOCItem]
    chapters: list[Chapter]


@dataclass
class _TOCEntry:
    title: str | None
    href: str | None
    children: tuple["_TOCEntry", ...] = ()


def _first_meta(book, name: str) -> str | None:
    # ebooklib metadata entries: list[tuple[value, dict_attrs]]
    vals = book.get_metadata("DC", name) or []
    return vals[0][0] if vals else None


def _multi_meta(book, name: str) -> list[str]:
    vals = book.get_metadata("DC", name) or []
    return [v for v, _attrs in vals if v]


def _collect_identifiers(book) -> list[Identifier]:
    idents: list[Identifier] = []
    for value, attrs in book.get_metadata("DC", "identifier") or []:
        scheme = None
        # Try common attributes for scheme
        for k in ("scheme", "opf:scheme", "id", "identifier-type"):
            scheme = scheme or attrs.get(k)
        if value:
            idents.append(Identifier(scheme=scheme, value=str(value)))
    return idents


def _toc_to_models(toc_list) -> list[TOCItem]:
    # toc_list is a nested list of Link/Section/tuple from ebooklib
    def convert(node) -> TOCItem:
        # node can be epub.Link, epub.Section, tuple
        title = None
        href = None
        children: list[TOCItem] = []
        if isinstance(node, epub.Link):
            title = node.title
            href = node.href
        elif isinstance(node, epub.Section):
            title = node.title
            children = [convert(n) for n in node]
        elif (
            isinstance(node, tuple)
            and len(node) == 2
            and isinstance(node[0], epub.Link)
        ):
            title = node[0].title
            href = node[0].href
            # node[1] may be list of children
            if isinstance(node[1], list | tuple):
                children = [convert(n) for n in node[1]]
        else:
            # Fallback: try attributes
            title = getattr(node, "title", None)
            href = getattr(node, "href", None)
            maybe_children = getattr(node, "children", [])
            if maybe_children:
                children = [convert(n) for n in maybe_children]
        return TOCItem(title=title, href=href, children=children)

    result: list[TOCItem] = []
    for n in toc_list or []:
        result.append(convert(n))
    return result


def _flatten_toc(toc: list[TOCItem]) -> list[TOCItem]:
    out: list[TOCItem] = []

    def walk(items: Iterable[TOCItem]):
        for it in items:
            out.append(it)
            if it.children:
                walk(it.children)

    walk(toc)
    return out


def _toc_title_by_href(toc: list[TOCItem]) -> dict[str, str]:
    flat = _flatten_toc(toc)
    mapping: dict[str, str] = {}
    for item in flat:
        if item.href and item.title:
            # Normalize href by stripping anchors
            href = item.href.split("#", 1)[0]
            mapping[href] = item.title
    return mapping


def _clean_title(title: str) -> str:
    """Clean up title formatting by fixing common issues."""
    if not title:
        return title

    # Fix missing spaces after periods (e.g., "Chapter 1.Title" -> "Chapter 1. Title")
    title = re.sub(r"(\d+)\.([A-Z])", r"\1. \2", title)

    # Fix missing spaces after "Chapter" (e.g., "Chapter1" -> "Chapter 1")
    title = re.sub(r"Chapter(\d+)", r"Chapter \1", title, flags=re.IGNORECASE)

    # Fix missing spaces after "Part" (e.g., "PartI" -> "Part I")
    title = re.sub(r"Part([IVX]+)", r"Part \1", title, flags=re.IGNORECASE)

    # Fix missing spaces after periods followed by letters (e.g., "andMaintainable" -> "and Maintainable")
    title = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)

    # Fix extra spaces around periods
    title = re.sub(r"\s*\.\s*", ". ", title)

    # Fix extra spaces around commas
    title = re.sub(r"\s*,\s*", ", ", title)

    # Fix "Part itioning" -> "Partitioning" (remove extra space)
    title = re.sub(r"Part\s+([a-z])", r"Part\1", title, flags=re.IGNORECASE)

    # Clean up excessive whitespace
    return re.sub(r"\s+", " ", title).strip()


def _extract_title_from_html(html_content: str) -> str | None:
    """Extract chapter title from HTML content using multiple strategies."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Strategy 1: Look for the first h1 tag
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
        return _clean_title(title)

    # Strategy 2: Look for the first h2 tag
    h2 = soup.find("h2")
    if h2 and h2.get_text(strip=True):
        title = h2.get_text(strip=True)
        return _clean_title(title)

    # Strategy 3: Look for title tag in head
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        title_text = title_tag.get_text(strip=True)
        # Skip generic titles like "Chapter" or "Page"
        if not re.match(r"^(Chapter|Page|Part)\s*\d*$", title_text, re.IGNORECASE):
            return _clean_title(title_text)

    # Strategy 4: Look for the first heading (h1-h6) with substantial content
    for tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        heading = soup.find(tag_name)
        if heading:
            heading_text = heading.get_text(strip=True)
            if heading_text and len(heading_text) > 3:  # Avoid very short headings
                return _clean_title(heading_text)

    # Strategy 5: Look for the first paragraph that starts with "Chapter" or "Part"
    for p in soup.find_all("p"):
        p_text = p.get_text(strip=True)
        if re.match(r"^(Chapter|Part)\s+\d+", p_text, re.IGNORECASE):
            # Extract the full title (up to first period or line break)
            title_match = re.match(r"^(Chapter|Part)\s+\d+[^.]*", p_text, re.IGNORECASE)
            if title_match:
                title = title_match.group(0).strip()
                return _clean_title(title)

    return None


def _extract_text_from_item(item) -> str:
    # item is EpubHtml
    html = item.get_content()
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    # Collapse excessive spaces
    return " ".join(text.split())


def parse_epub(path: str, include_text: bool = True) -> EpubDetails:
    """Parse an EPUB file and return structured details.

    Parameters:
        path: Path to the .epub file.
        include_text: When False, chapters.text will be empty strings but
            word_count preserved via estimation (0 without text). Useful for concise CLI output.
    """

    book = epub.read_epub(path)

    metadata = BookInfo(
        title=_first_meta(book, "title"),
        authors=_multi_meta(book, "creator"),
        language=_first_meta(book, "language"),
        publisher=_first_meta(book, "publisher"),
        description=_first_meta(book, "description"),
        subjects=_multi_meta(book, "subject"),
        identifiers=_collect_identifiers(book),
    )

    toc_models = _toc_to_models(book.toc)
    href_to_title = _toc_title_by_href(toc_models)

    chapters: list[Chapter] = []

    # book.spine is list of (idref, linear)
    for idx, (idref, _linear) in enumerate(book.spine):
        try:
            item = book.get_item_with_id(idref)
        except KeyError:
            continue
        # Only process HTML content
        media_type = getattr(item, "media_type", "")
        if not media_type or not media_type.endswith("html+xml"):
            continue
        href = getattr(item, "file_name", None)

        # Try to get title from TOC first
        title = href_to_title.get(href or "")

        # If no title from TOC, try to extract from HTML content
        if not title and include_text:
            try:
                html_content = item.get_content()
                title = _extract_title_from_html(html_content)
            except Exception:
                title = None

        text = ""
        if include_text:
            try:
                text = _extract_text_from_item(item)
            except Exception:
                text = ""
        word_count = len(text.split()) if text else 0
        chapters.append(
            Chapter(index=idx, title=title, href=href, text=text, word_count=word_count)
        )

    return EpubDetails(path=path, metadata=metadata, toc=toc_models, chapters=chapters)


__all__ = [
    "BookInfo",
    "Chapter",
    "EpubDetails",
    "Identifier",
    "Person",
    "TOCItem",
    "parse_epub",
]
