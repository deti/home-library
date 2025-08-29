"""Tests for the epub CLI command."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from ebooklib import epub

from home_library.cli.epub import _print_toc, main


# Ensure the package can be imported from the src/ layout during tests
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _make_sample_epub(tmp_path: Path) -> Path:
    """Create a sample EPUB file for testing."""
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

    # Navigation
    book.add_item(epub.EpubNav())
    book.add_item(epub.EpubNcx())

    # TOC: two top-level entries
    book.toc = [
        epub.Link("ch1.xhtml", "Chapter 1", "ch1"),
        (epub.Link("ch2.xhtml#frag", "Chapter 2", "ch2"), []),
    ]

    # Spine: only chapters
    book.spine = [ch1, ch2]

    out = tmp_path / "sample.epub"
    epub.write_epub(str(out), book)
    return out


class MockTocItem:
    """Mock TOC item for testing."""
    def __init__(self, title=None, href=None, children=None):
        self.title = title
        self.href = href
        self.children = children or []


def test_print_toc_basic():
    """Test basic TOC printing functionality."""
    import io
    from contextlib import redirect_stdout

    toc = [
        MockTocItem("Chapter 1", "ch1.xhtml"),
        MockTocItem("Chapter 2", "ch2.xhtml"),
    ]

    f = io.StringIO()
    with redirect_stdout(f):
        _print_toc(toc)

    output = f.getvalue()
    lines = output.strip().split("\n")

    assert len(lines) == 2
    assert "- Chapter 1 (ch1.xhtml)" in lines[0]
    assert "- Chapter 2 (ch2.xhtml)" in lines[1]


def test_print_toc_nested():
    """Test nested TOC printing with indentation."""
    import io
    from contextlib import redirect_stdout

    toc = [
        MockTocItem("Part 1", "part1.xhtml", [
            MockTocItem("Chapter 1.1", "ch1.1.xhtml"),
            MockTocItem("Chapter 1.2", "ch1.2.xhtml"),
        ]),
        MockTocItem("Part 2", "part2.xhtml"),
    ]

    f = io.StringIO()
    with redirect_stdout(f):
        _print_toc(toc)

    output = f.getvalue()
    lines = output.strip().split("\n")

    assert len(lines) == 4
    assert "- Part 1 (part1.xhtml)" in lines[0]
    assert "  - Chapter 1.1 (ch1.1.xhtml)" in lines[1]
    assert "  - Chapter 1.2 (ch1.2.xhtml)" in lines[2]
    assert "- Part 2 (part2.xhtml)" in lines[3]


def test_print_toc_untitled_items():
    """Test TOC printing with untitled items."""
    import io
    from contextlib import redirect_stdout

    toc = [
        MockTocItem(None, "untitled.xhtml"),
        MockTocItem("", "empty.xhtml"),
    ]

    f = io.StringIO()
    with redirect_stdout(f):
        _print_toc(toc)

    output = f.getvalue()
    lines = output.strip().split("\n")

    assert len(lines) == 2
    assert "- <untitled> (untitled.xhtml)" in lines[0]
    assert "- <untitled> (empty.xhtml)" in lines[1]


def test_print_toc_no_href():
    """Test TOC printing with items that have no href."""
    import io
    from contextlib import redirect_stdout

    toc = [
        MockTocItem("Chapter 1", None),
        MockTocItem("Chapter 2", ""),
    ]

    f = io.StringIO()
    with redirect_stdout(f):
        _print_toc(toc)

    output = f.getvalue()
    lines = output.strip().split("\n")

    assert len(lines) == 2
    assert "- Chapter 1" in lines[0]
    assert "- Chapter 2" in lines[1]


@patch("sys.argv", ["epub-info", "test.epub"])
def test_epub_main_basic_usage():
    """Test basic EPUB CLI usage without arguments."""
    import io
    from contextlib import redirect_stdout

    # Mock the parse_epub function to return test data
    mock_details = Mock()
    mock_details.path = "test.epub"
    mock_details.metadata = Mock()
    mock_details.metadata.title = "Test Book"
    mock_details.metadata.authors = ["Test Author"]
    mock_details.metadata.language = "en"
    mock_details.metadata.publisher = "Test Publisher"
    mock_details.metadata.subjects = ["Test Subject"]
    mock_details.metadata.identifiers = []
    mock_details.metadata.description = None
    mock_details.toc = []
    mock_details.chapters = []

    with patch("home_library.cli.epub.parse_epub", return_value=mock_details):
        f = io.StringIO()
        with redirect_stdout(f):
            main()

        output = f.getvalue()

        # Check that basic output is present
        assert "File: test.epub" in output
        assert "Title: Test Book" in output
        assert "Authors: Test Author" in output
        assert "Language: en" in output
        assert "Publisher: Test Publisher" in output
        assert "Subjects: Test Subject" in output


@patch("sys.argv", ["epub-info", "test.epub", "--json"])
def test_epub_main_json_output():
    """Test EPUB CLI with JSON output flag."""
    import io
    from contextlib import redirect_stdout

    # Mock the parse_epub function
    mock_details = Mock()
    mock_details.model_dump_json.return_value = '{"test": "data"}'

    with patch("home_library.cli.epub.parse_epub", return_value=mock_details):
        f = io.StringIO()
        with redirect_stdout(f):
            main()

        output = f.getvalue().strip()

        # Should output JSON
        assert output == '{"test": "data"}'
        mock_details.model_dump_json.assert_called_once_with(indent=2)


@patch("sys.argv", ["epub-info", "test.epub", "--include-text"])
def test_epub_main_include_text():
    """Test EPUB CLI with include-text flag."""
    import io
    from contextlib import redirect_stdout

    with patch("home_library.cli.epub.parse_epub") as mock_parse:
        mock_details = Mock()
        mock_details.path = "test.epub"
        mock_details.metadata = Mock()
        mock_details.metadata.title = "Test Book"
        mock_details.metadata.authors = []
        mock_details.metadata.language = None
        mock_details.metadata.publisher = None
        mock_details.metadata.subjects = []
        mock_details.metadata.identifiers = []
        mock_details.metadata.description = None
        mock_details.toc = []
        mock_details.chapters = []

        mock_parse.return_value = mock_details

        f = io.StringIO()
        with redirect_stdout(f):
            main()

        # Verify parse_epub was called with include_text=True
        mock_parse.assert_called_once_with("test.epub", include_text=True)


def test_epub_main_with_real_epub(tmp_path):
    """Test EPUB CLI with a real EPUB file."""
    epub_path = _make_sample_epub(tmp_path)

    # Test the CLI by patching sys.argv
    with patch("sys.argv", ["epub-info", str(epub_path)]):
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            main()

        output = f.getvalue()

        # Check that expected output is present
        assert f"File: {epub_path}" in output
        assert "Title: Test Book" in output
        assert "Authors: Jane Doe, John Smith" in output
        assert "Language: en" in output
        assert "Publisher: Acme Publishing" in output
        assert "Subjects: Fiction, Testing" in output
        assert "TOC: 2 top-level items" in output
        assert "Chapters:" in output


def test_epub_cli_module_execution(tmp_path):
    """Test running the EPUB CLI module directly."""
    epub_path = _make_sample_epub(tmp_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PATH)

    # Test with a real EPUB file
    proc = subprocess.run(
        [sys.executable, "-m", "home_library.cli.epub", str(epub_path)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert proc.returncode == 0
    # Note: stderr might contain warnings, so we don't assert it's empty

    output = proc.stdout
    assert "Test Book" in output
    assert "Jane Doe" in output
    assert "en" in output


def test_epub_cli_with_json_flag(tmp_path):
    """Test EPUB CLI with JSON output using subprocess."""
    epub_path = _make_sample_epub(tmp_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PATH)

    proc = subprocess.run(
        [sys.executable, "-m", "home_library.cli.epub", str(epub_path), "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert proc.returncode == 0
    # Note: stderr might contain warnings, so we don't assert it's empty

    # Should be valid JSON
    output = proc.stdout.strip()
    json_data = json.loads(output)

    # Check JSON structure
    assert "path" in json_data
    assert "metadata" in json_data
    assert "toc" in json_data
    assert "chapters" in json_data


def test_epub_cli_error_handling():
    """Test EPUB CLI error handling for non-existent files."""
    with patch("sys.argv", ["epub-info", "nonexistent.epub"]):

        # This should raise an exception, so we test that it fails appropriately
        with pytest.raises(Exception):  # Should fail when file doesn't exist
            main()


def test_epub_cli_script_entry_point():
    """Test that the script entry point works correctly."""
    # This test verifies the pyproject.toml script configuration
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PATH)

    # Test the module path that would be used by the script
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, r'{SRC_PATH}'); "
         "from home_library.cli.epub import main; print('CLI module imported successfully')"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert proc.returncode == 0
    # Note: stderr might contain warnings, so we don't assert it's empty
    assert "CLI module imported successfully" in proc.stdout

