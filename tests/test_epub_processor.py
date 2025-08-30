"""Tests for the EPUB processor module."""

import logging
from pathlib import Path

from ebooklib import epub

from home_library.epub_processor import parse_epub


# Configure logging for tests to avoid noise
logging.basicConfig(level=logging.CRITICAL)


def _make_sample_epub(tmp_path: Path) -> Path:
    book = epub.EpubBook()

    # Metadata
    book.set_identifier("bookid-123")
    book.set_title("Test Book")
    book.set_language("en")
    book.add_author("Jane Doe")
    book.add_author("John Smith")
    book.add_metadata("DC", "publisher", "Acme Publishing")
    book.add_metadata("DC", "description", "A lovely book for testing.")
    book.add_metadata("DC", "subject", "Fiction")
    book.add_metadata("DC", "subject", "Testing")
    # Extra identifier with scheme
    book.add_metadata("DC", "identifier", "ISBN-98765", {"scheme": "ISBN"})

    # Chapters
    ch1 = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
    ch1.set_content(
        b"""
        <html><head><title>Ch1</title></head>
        <body>
          <h1>Chapter 1</h1>
          <p>Hello world 123</p>
        </body></html>
        """
    )

    ch2 = epub.EpubHtml(title="Chapter 2", file_name="ch2.xhtml", lang="en")
    ch2.set_content(
        b"""
        <html><head><title>Ch2</title></head>
        <body>
          <h1>Chapter 2</h1>
          <p>Another paragraph with several words.</p>
        </body></html>
        """
    )

    # Add items
    book.add_item(ch1)
    book.add_item(ch2)

    # Navigation/NCX (required by many readers, but keep nav out of spine)
    book.add_item(epub.EpubNav())
    book.add_item(epub.EpubNcx())

    # TOC: two top-level entries, second provided as a (Link, children) tuple with anchor
    book.toc = [
        epub.Link("ch1.xhtml", "Chapter 1", "ch1"),
        (epub.Link("ch2.xhtml#frag", "Chapter 2", "ch2"), []),
    ]

    # Spine: only chapters to keep deterministic order in tests
    book.spine = [ch1, ch2]

    out = tmp_path / "sample.epub"
    epub.write_epub(str(out), book)
    return out


def test_parse_epub_basic_include_text(tmp_path: Path) -> None:
    epub_path = _make_sample_epub(tmp_path)

    details = parse_epub(str(epub_path), include_text=True)

    # Basic file/path
    assert details.path == str(epub_path)

    # Metadata assertions
    md = details.metadata
    assert md.title == "Test Book"
    assert md.language == "en"
    assert md.authors == ["Jane Doe", "John Smith"]
    assert md.publisher == "Acme Publishing"
    assert set(md.subjects) == {"Fiction", "Testing"}
    id_values = {i.value for i in md.identifiers}
    assert {"bookid-123", "ISBN-98765"}.issubset(id_values)

    # TOC structure
    assert len(details.toc) == 2
    assert details.toc[0].title == "Chapter 1"
    assert details.toc[0].href == "ch1.xhtml"
    assert details.toc[1].title == "Chapter 2"
    assert details.toc[1].href == "ch2.xhtml#frag"
    assert len(details.toc[1].children) == 0

    # Chapters: find by href to avoid depending on exact spine indices
    chapters_by_href = {c.href: c for c in details.chapters}
    # Ensure only our two chapters are present
    assert set(chapters_by_href) == {"ch1.xhtml", "ch2.xhtml"}

    ch1 = chapters_by_href["ch1.xhtml"]
    assert ch1.title == "Chapter 1"  # title mapped via TOC href
    assert ch1.word_count > 0
    assert "Hello world 123" in ch1.text

    ch2 = chapters_by_href["ch2.xhtml"]
    assert ch2.title == "Chapter 2"
    assert ch2.word_count > 0
    assert "Another paragraph" in ch2.text


def test_parse_epub_without_text(tmp_path: Path) -> None:
    epub_path = _make_sample_epub(tmp_path)

    details = parse_epub(str(epub_path), include_text=False)

    chapters_by_href = {c.href: c for c in details.chapters}
    assert set(chapters_by_href) == {"ch1.xhtml", "ch2.xhtml"}

    assert chapters_by_href["ch1.xhtml"].text == ""
    assert chapters_by_href["ch1.xhtml"].word_count == 0
    assert chapters_by_href["ch2.xhtml"].text == ""
    assert chapters_by_href["ch2.xhtml"].word_count == 0


def test_parse_epub_html_title_extraction(tmp_path: Path) -> None:
    """Test that titles are extracted from HTML when TOC titles are missing."""
    # Create a test EPUB where TOC doesn't have titles
    book = epub.EpubBook()
    book.set_identifier("bookid-456")
    book.set_title("Test Book Without TOC Titles")
    book.set_language("en")
    book.add_author("Test Author")

    # Chapters with titles in HTML but not in TOC
    ch1 = epub.EpubHtml(title=None, file_name="ch1.xhtml", lang="en")
    ch1.set_content(
        b"""
        <html><head><title>Chapter 1</title></head>
        <body>
          <h1>Chapter 1. Reliable Systems</h1>
          <p>Content about reliable systems.</p>
        </body></html>
        """
    )

    ch2 = epub.EpubHtml(title=None, file_name="ch2.xhtml", lang="en")
    ch2.set_content(
        b"""
        <html><head><title>Chapter 2</title></head>
        <body>
          <h1>Chapter 2. Scalable Architecture</h1>
          <p>Content about scalable architecture.</p>
        </body></html>
        """
    )

    book.add_item(ch1)
    book.add_item(ch2)
    book.add_item(epub.EpubNav())
    book.add_item(epub.EpubNcx())

    # TOC with minimal or no titles
    book.toc = [
        epub.Link("ch1.xhtml", None, "ch1"),
        epub.Link("ch2.xhtml", None, "ch2"),
    ]

    book.spine = [ch1, ch2]

    epub_path = tmp_path / "sample_no_toc.epub"
    epub.write_epub(str(epub_path), book)

    details = parse_epub(str(epub_path), include_text=True)

    # Verify that titles were extracted from HTML content
    chapters_by_href = {c.href: c for c in details.chapters}
    assert set(chapters_by_href) == {"ch1.xhtml", "ch2.xhtml"}

    ch1 = chapters_by_href["ch1.xhtml"]
    assert ch1.title == "Chapter 1. Reliable Systems"  # extracted from h1 tag
    assert ch1.word_count > 0

    ch2 = chapters_by_href["ch2.xhtml"]
    assert ch2.title == "Chapter 2. Scalable Architecture"  # extracted from h1 tag
    assert ch2.word_count > 0


def test_parse_epub_no_html_title_extraction_when_no_text(tmp_path: Path) -> None:
    """Test that HTML title extraction is skipped when include_text=False."""
    # Create a test EPUB where TOC doesn't have titles
    book = epub.EpubBook()
    book.set_identifier("bookid-789")
    book.set_title("Test Book Without TOC Titles")
    book.set_language("en")
    book.add_author("Test Author")

    # Chapters with titles in HTML but not in TOC
    ch1 = epub.EpubHtml(title=None, file_name="ch1.xhtml", lang="en")
    ch1.set_content(
        b"""
        <html><head><title>Chapter 1</title></head>
        <body>
          <h1>Chapter 1. Reliable Systems</h1>
          <p>Content about reliable systems.</p>
        </body></html>
        """
    )

    ch2 = epub.EpubHtml(title=None, file_name="ch2.xhtml", lang="en")
    ch2.set_content(
        b"""
        <html><head><title>Chapter 2</title></head>
        <body>
          <h1>Chapter 2. Scalable Architecture</h1>
          <p>Content about scalable architecture.</p>
        </body></html>
        """
    )

    book.add_item(ch1)
    book.add_item(ch2)
    book.add_item(epub.EpubNav())
    book.add_item(epub.EpubNcx())

    # TOC with minimal or no titles
    book.toc = [
        epub.Link("ch1.xhtml", None, "ch1"),
        epub.Link("ch2.xhtml", None, "ch2"),
    ]

    book.spine = [ch1, ch2]

    epub_path = tmp_path / "sample_no_toc_no_text.epub"
    epub.write_epub(str(epub_path), book)

    details = parse_epub(str(epub_path), include_text=False)

    # Verify that titles are None when not extracting text
    chapters_by_href = {c.href: c for c in details.chapters}
    assert set(chapters_by_href) == {"ch1.xhtml", "ch2.xhtml"}

    ch1 = chapters_by_href["ch1.xhtml"]
    assert ch1.title is None  # No HTML extraction when include_text=False
    assert ch1.text == ""
    assert ch1.word_count == 0

    ch2 = chapters_by_href["ch2.xhtml"]
    assert ch2.title is None  # No HTML extraction when include_text=False
    assert ch2.text == ""
    assert ch2.word_count == 0


def test_title_cleaning_functionality(tmp_path: Path) -> None:
    """Test that the title cleaning function works correctly."""
    # Create a test EPUB with titles that need cleaning
    book = epub.EpubBook()
    book.set_identifier("bookid-999")
    book.set_title("Test Book With Titles Needing Cleaning")
    book.set_language("en")
    book.add_author("Test Author")

    # Chapters with titles that need cleaning
    ch1 = epub.EpubHtml(title=None, file_name="ch1.xhtml", lang="en")
    ch1.set_content(
        b"""
        <html><head><title>Chapter 1</title></head>
        <body>
          <h1>Chapter1.Reliability</h1>
          <p>Content about reliability.</p>
        </body></html>
        """
    )

    ch2 = epub.EpubHtml(title=None, file_name="ch2.xhtml", lang="en")
    ch2.set_content(
        b"""
        <html><head><title>Chapter 2</title></head>
        <body>
          <h1>Chapter2.Scalability</h1>
          <p>Content about scalability.</p>
        </body></html>
        """
    )

    book.add_item(ch1)
    book.add_item(ch2)
    book.add_item(epub.EpubNav())
    book.add_item(epub.EpubNcx())

    # TOC with minimal or no titles
    book.toc = [
        epub.Link("ch1.xhtml", None, "ch1"),
        epub.Link("ch2.xhtml", None, "ch2"),
    ]

    book.spine = [ch1, ch2]

    epub_path = tmp_path / "sample_cleaning_test.epub"
    epub.write_epub(str(epub_path), book)

    details = parse_epub(str(epub_path), include_text=True)

    # Verify that titles were cleaned properly
    chapters_by_href = {c.href: c for c in details.chapters}
    assert set(chapters_by_href) == {"ch1.xhtml", "ch2.xhtml"}

    ch1 = chapters_by_href["ch1.xhtml"]
    # Should have proper spacing after "Chapter" and after period
    assert ch1.title == "Chapter 1. Reliability"

    ch2 = chapters_by_href["ch2.xhtml"]
    # Should have proper spacing after "Chapter" and after period
    assert ch2.title == "Chapter 2. Scalability"
