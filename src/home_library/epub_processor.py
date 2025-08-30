"""EPUB processing utilities.

Provides structured extraction of book-level metadata, table of contents,
chapter ordering and text suitable for RAG pipelines.

The public API prefers modern Python data structures (Pydantic models)
instead of raw dicts.
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup
from ebooklib import epub
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


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
    logger.debug("Extracting title from HTML content using multiple strategies")

    soup = BeautifulSoup(html_content, "html.parser")

    # Strategy 1: Look for the first h1 tag
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
        cleaned_title = _clean_title(title)
        logger.debug(f"Found title from h1 tag: '{cleaned_title}'")
        return cleaned_title

    # Strategy 2: Look for the first h2 tag
    h2 = soup.find("h2")
    if h2 and h2.get_text(strip=True):
        title = h2.get_text(strip=True)
        cleaned_title = _clean_title(title)
        logger.debug(f"Found title from h2 tag: '{cleaned_title}'")
        return cleaned_title

    # Strategy 3: Look for title tag in head
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        title_text = title_tag.get_text(strip=True)
        # Skip generic titles like "Chapter" or "Page"
        if not re.match(r"^(Chapter|Page|Part)\s*\d*$", title_text, re.IGNORECASE):
            cleaned_title = _clean_title(title_text)
            logger.debug(f"Found title from title tag: '{cleaned_title}'")
            return cleaned_title

    # Strategy 4: Look for the first heading (h1-h6) with substantial content
    for tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        heading = soup.find(tag_name)
        if heading:
            heading_text = heading.get_text(strip=True)
            if heading_text and len(heading_text) > 3:  # Avoid very short headings
                cleaned_title = _clean_title(heading_text)
                logger.debug(f"Found title from {tag_name} tag: '{cleaned_title}'")
                return cleaned_title

    # Strategy 5: Look for the first paragraph that starts with "Chapter" or "Part"
    for p in soup.find_all("p"):
        p_text = p.get_text(strip=True)
        if re.match(r"^(Chapter|Part)\s+\d+", p_text, re.IGNORECASE):
            # Extract the full title (up to first period or line break)
            title_match = re.match(r"^(Chapter|Part)\s+\d+[^.]*", p_text, re.IGNORECASE)
            if title_match:
                title = title_match.group(0).strip()
                cleaned_title = _clean_title(title)
                logger.debug(f"Found title from paragraph: '{cleaned_title}'")
                return cleaned_title

    logger.debug("No title found using any strategy")
    return None


def _extract_text_from_item(item) -> str:
    # item is EpubHtml
    try:
        html = item.get_content()
        soup = BeautifulSoup(html, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        # Collapse excessive spaces
        cleaned_text = " ".join(text.split())
        logger.debug(f"Extracted {len(cleaned_text)} characters of text from item")
    except Exception:
        logger.exception("Failed to extract text from item")
        return ""
    return cleaned_text


def _load_epub_book(path: str) -> Any:
    """Load EPUB book using ebooklib."""
    try:
        logger.debug("Reading EPUB file with ebooklib")
        book = epub.read_epub(path)
        logger.info(f"EPUB file loaded successfully: {len(book.spine)} spine items")
    except Exception:
        logger.exception(f"Failed to read EPUB file {path}")
        raise
    return book


def _extract_metadata(book: Any) -> BookInfo:
    """Extract metadata from EPUB book."""
    logger.debug("Extracting book metadata")
    try:
        metadata = BookInfo(
            title=_first_meta(book, "title"),
            authors=_multi_meta(book, "creator"),
            language=_first_meta(book, "language"),
            publisher=_first_meta(book, "publisher"),
            description=_first_meta(book, "description"),
            subjects=_multi_meta(book, "subject"),
            identifiers=_collect_identifiers(book),
        )
        logger.info(
            f"Metadata extracted - Title: '{metadata.title}', Authors: {len(metadata.authors)}, Language: {metadata.language}"
        )
    except Exception:
        logger.exception("Failed to extract metadata")
        raise
    return metadata


def _process_table_of_contents(book: Any) -> tuple[list[TOCItem], dict[str, str]]:
    """Process table of contents and return TOC models and href mappings."""
    logger.debug("Processing table of contents")
    try:
        toc_models = _toc_to_models(book.toc)
        href_to_title = _toc_title_by_href(toc_models)
        logger.info(
            f"TOC processed: {len(toc_models)} top-level items, {len(href_to_title)} href mappings"
        )
    except Exception:
        logger.exception("Failed to process table of contents")
        raise
    return toc_models, href_to_title


def _process_chapter_item(
    idx: int, idref: str, item: Any, href_to_title: dict[str, str], include_text: bool
) -> Chapter:
    """Process a single chapter item from the EPUB spine."""
    href = getattr(item, "file_name", None)
    logger.debug(f"Processing chapter {idx}: {idref} -> {href}")

    # Try to get title from TOC first
    title = href_to_title.get(href or "")
    if title:
        logger.debug(f"Found title from TOC: '{title}'")

    # If no title from TOC, try to extract from HTML content
    if not title and include_text:
        try:
            html_content = item.get_content()
            title = _extract_title_from_html(html_content)
            if title:
                logger.debug(f"Extracted title from HTML: '{title}'")
        except Exception as e:
            logger.warning(f"Failed to extract title from HTML for chapter {idx}: {e}")
            title = None

    text = ""
    if include_text:
        try:
            text = _extract_text_from_item(item)
        except Exception as e:
            logger.warning(f"Failed to extract text from chapter {idx}: {e}")
            text = ""

    word_count = len(text.split()) if text else 0
    logger.debug(f"Chapter {idx}: {word_count} words, title: '{title}'")

    return Chapter(index=idx, title=title, href=href, text=text, word_count=word_count)


def _process_chapters(
    book: Any, href_to_title: dict[str, str], include_text: bool
) -> list[Chapter]:
    """Process all chapters from the EPUB spine."""
    logger.debug("Processing chapters from spine")
    chapters: list[Chapter] = []
    processed_items = 0
    skipped_items = 0

    # book.spine is list of (idref, linear)
    for idx, (idref, _linear) in enumerate(book.spine):
        try:
            item = book.get_item_with_id(idref)
            processed_items += 1
        except KeyError:
            logger.warning(f"Could not find item with idref: {idref}")
            skipped_items += 1
            continue

        # Only process HTML content
        media_type = getattr(item, "media_type", "")
        if not media_type or not media_type.endswith("html+xml"):
            logger.debug(
                f"Skipping non-HTML item {idref} with media type: {media_type}"
            )
            skipped_items += 1
            continue

        chapter = _process_chapter_item(idx, idref, item, href_to_title, include_text)
        chapters.append(chapter)

    logger.info(
        f"Chapter processing completed: {len(chapters)} chapters, {processed_items} items processed, {skipped_items} items skipped"
    )
    return chapters


def parse_epub(path: str, include_text: bool = True) -> EpubDetails:
    """Parse an EPUB file and return structured details.

    Parameters:
        path: Path to the .epub file.
        include_text: When False, chapters.text will be empty strings but
            word_count preserved via estimation (0 without text). Useful for concise CLI output.
    """
    logger.info(f"Parsing EPUB file: {path}")
    logger.debug(f"Include text setting: {include_text}")

    book = _load_epub_book(path)
    metadata = _extract_metadata(book)
    toc_models, href_to_title = _process_table_of_contents(book)
    chapters = _process_chapters(book, href_to_title, include_text)

    total_words = sum(chapter.word_count for chapter in chapters)
    logger.info(f"Total word count across all chapters: {total_words}")

    result = EpubDetails(
        path=path, metadata=metadata, toc=toc_models, chapters=chapters
    )
    logger.info(
        f"EPUB parsing completed successfully: {len(result.chapters)} chapters, {total_words} total words"
    )

    return result


__all__ = [
    "BookInfo",
    "Chapter",
    "EpubDetails",
    "Identifier",
    "Person",
    "TOCItem",
    "parse_epub",
]
