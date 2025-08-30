"""Tests for the CLI search command."""

import json
from io import StringIO
from unittest.mock import patch

import pytest

from home_library.cli.search import (
    _print_detailed_results,
    _print_json_results,
    _print_search_results,
    main,
)
from home_library.search import SearchResult


class TestSearchOutputFunctions:
    """Test the search output formatting functions."""

    def test_print_search_results_with_results(self, capsys):
        """Test printing search results in human-readable format."""
        results = [
            SearchResult(
                book_title="Test Book 1",
                book_author="Author 1",
                chapter_index=1,
                chapter_title="Chapter 1",
                chunk_index=0,
                text="This is the first test text chunk with some content.",
                start_token=0,
                end_token=15,
                word_count=15,
                similarity_score=0.85,
                file_path="/path/to/book1.epub"
            ),
            SearchResult(
                book_title="Test Book 2",
                book_author="Author 2",
                chapter_index=2,
                chapter_title="Chapter 2",
                chunk_index=1,
                text="This is the second test text chunk with different content.",
                start_token=20,
                end_token=35,
                word_count=15,
                similarity_score=0.72,
                file_path="/path/to/book2.epub"
            )
        ]

        _print_search_results(results, "test query")
        captured = capsys.readouterr()

        assert "🔍 Search Results for: 'test query'" in captured.out
        assert "Found 2 relevant citations:" in captured.out
        assert "📚 Result 1 (Similarity: 0.850)" in captured.out
        assert "📚 Result 2 (Similarity: 0.720)" in captured.out
        assert "Book: Test Book 1" in captured.out
        assert "Book: Test Book 2" in captured.out
        assert "Author: Author 1" in captured.out
        assert "Author: Author 2" in captured.out
        assert "Chapter: 1" in captured.out
        assert "Chapter: 2" in captured.out
        assert "Chapter Title: Chapter 1" in captured.out
        assert "Chapter Title: Chapter 2" in captured.out
        assert "Section: Chunk 0 (tokens 0-15)" in captured.out
        assert "Section: Chunk 1 (tokens 20-35)" in captured.out
        assert "Words: 15" in captured.out
        assert "Text Preview: This is the first test text chunk with some content." in captured.out
        assert "Text Preview: This is the second test text chunk with different content." in captured.out

    def test_print_search_results_no_results(self, capsys):
        """Test printing search results when no results are found."""
        results = []
        _print_search_results(results, "test query")
        captured = capsys.readouterr()

        assert "🔍 No results found for query: 'test query'" in captured.out

    def test_print_search_results_optional_fields(self, capsys):
        """Test printing search results with missing optional fields."""
        results = [
            SearchResult(
                book_title="Test Book",
                book_author=None,  # Missing author
                chapter_index=1,
                chapter_title=None,  # Missing chapter title
                chunk_index=0,
                text="This is a test text chunk.",
                start_token=0,
                end_token=10,
                word_count=10,
                similarity_score=0.85,
                file_path="/path/to/book.epub"
            )
        ]

        _print_search_results(results, "test query")
        captured = capsys.readouterr()

        assert "Book: Test Book" in captured.out
        assert "Author:" not in captured.out  # Should not print author line
        assert "Chapter: 1" in captured.out
        assert "Chapter Title:" not in captured.out  # Should not print chapter title line

    def test_print_json_results(self, capsys):
        """Test printing search results in JSON format."""
        results = [
            SearchResult(
                book_title="Test Book",
                book_author="Test Author",
                chapter_index=1,
                chapter_title="Test Chapter",
                chunk_index=0,
                text="This is a test text chunk.",
                start_token=0,
                end_token=10,
                word_count=10,
                similarity_score=0.85,
                file_path="/path/to/book.epub"
            )
        ]

        _print_json_results(results, "test query")
        captured = capsys.readouterr()

        # Parse the JSON output
        output_data = json.loads(captured.out)

        assert output_data["query"] == "test query"
        assert output_data["total_results"] == 1
        assert len(output_data["results"]) == 1

        result = output_data["results"][0]
        assert result["book_title"] == "Test Book"
        assert result["book_author"] == "Test Author"
        assert result["chapter_index"] == 1
        assert result["chapter_title"] == "Test Chapter"
        assert result["chunk_index"] == 0
        assert result["text"] == "This is a test text chunk."
        assert result["similarity_score"] == 0.85

    def test_print_detailed_results(self, capsys):
        """Test printing detailed search results with full text."""
        results = [
            SearchResult(
                book_title="Test Book",
                book_author="Test Author",
                chapter_index=1,
                chapter_title="Test Chapter",
                chunk_index=0,
                text="This is a test text chunk with more detailed content that should be displayed in full.",
                start_token=0,
                end_token=20,
                word_count=20,
                similarity_score=0.85,
                file_path="/path/to/book.epub"
            )
        ]

        _print_detailed_results(results, "test query")
        captured = capsys.readouterr()

        assert "🔍 Detailed Search Results for: 'test query'" in captured.out
        assert "Found 1 relevant citations:" in captured.out
        assert "📚 Result 1 (Similarity: 0.850)" in captured.out
        assert "Book: Test Book" in captured.out
        assert "Author: Test Author" in captured.out
        assert "Chapter: 1" in captured.out
        assert "Chapter Title: Test Chapter" in captured.out
        assert "Section: Chunk 0 (tokens 0-20)" in captured.out
        assert "Words: 20" in captured.out
        assert "Full Text:" in captured.out
        assert "This is a test text chunk with more detailed content that should be displayed in full." in captured.out

    def test_print_detailed_results_no_results(self, capsys):
        """Test printing detailed results when no results are found."""
        results = []
        _print_detailed_results(results, "test query")
        captured = capsys.readouterr()

        assert "🔍 No results found for query: 'test query'" in captured.out


class TestSearchCLI:
    """Test the search CLI command."""

    @patch("home_library.cli.search.search_library")
    def test_main_successful_search(self, mock_search_library, capsys):
        """Test successful search execution."""
        # Mock search results
        mock_results = [
            SearchResult(
                book_title="Test Book",
                book_author="Test Author",
                chapter_index=1,
                chapter_title="Test Chapter",
                chunk_index=0,
                text="This is a test text chunk.",
                start_token=0,
                end_token=10,
                word_count=10,
                similarity_score=0.85,
                file_path="/path/to/book.epub"
            )
        ]
        mock_search_library.return_value = mock_results

        # Test with basic query
        with patch("sys.argv", ["search-library", "test query"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "🔍 Searching library for: 'test query'" in captured.out
        assert "Results limit: 5" in captured.out
        assert "Similarity threshold: 0.3" in captured.out
        assert "📚 Result 1 (Similarity: 0.850)" in captured.out

        # Verify search_library was called with correct parameters
        mock_search_library.assert_called_once_with(
            query="test query",
            limit=5,
            similarity_threshold=0.3,
            model_name=None,
            device=None
        )

    @patch("home_library.cli.search.search_library")
    def test_main_with_custom_parameters(self, mock_search_library, capsys):
        """Test search with custom parameters."""
        mock_results = []
        mock_search_library.return_value = mock_results

        # Test with custom parameters
        with patch("sys.argv", [
            "search-library",
            "test query",
            "--limit", "3",
            "--threshold", "0.7",
            "--model", "custom-model",
            "--device", "cuda"
        ]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Results limit: 3" in captured.out
        assert "Similarity threshold: 0.7" in captured.out
        assert "Model: custom-model" in captured.out
        assert "Device: cuda" in captured.out

        # Verify search_library was called with custom parameters
        mock_search_library.assert_called_once_with(
            query="test query",
            limit=3,
            similarity_threshold=0.7,
            model_name="custom-model",
            device="cuda"
        )

    @patch("home_library.cli.search.search_library")
    def test_main_with_detailed_output(self, mock_search_library, capsys):
        """Test search with detailed output."""
        mock_results = [
            SearchResult(
                book_title="Test Book",
                book_author="Test Author",
                chapter_index=1,
                chapter_title="Test Chapter",
                chunk_index=0,
                text="This is a test text chunk with detailed content.",
                start_token=0,
                end_token=15,
                word_count=15,
                similarity_score=0.85,
                file_path="/path/to/book.epub"
            )
        ]
        mock_search_library.return_value = mock_results

        # Test with detailed flag
        with patch("sys.argv", ["search-library", "test query", "--detailed"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "🔍 Detailed Search Results for: 'test query'" in captured.out
        assert "Full Text:" in captured.out
        assert "This is a test text chunk with detailed content." in captured.out

    @patch("home_library.cli.search.search_library")
    def test_main_with_json_output(self, mock_search_library, capsys):
        """Test search with JSON output."""
        mock_results = [
            SearchResult(
                book_title="Test Book",
                book_author="Test Author",
                chapter_index=1,
                chapter_title="Test Chapter",
                chunk_index=0,
                text="This is a test text chunk.",
                start_token=0,
                end_token=10,
                word_count=10,
                similarity_score=0.85,
                file_path="/path/to/book.epub"
            )
        ]
        mock_search_library.return_value = mock_results

        # Test with JSON flag
        with patch("sys.argv", ["search-library", "test query", "--json"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()

        # Verify JSON output - extract JSON from the captured output
        output_lines = captured.out.strip().split("\n")
        json_start = None
        for i, line in enumerate(output_lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        if json_start is not None:
            json_content = "\n".join(output_lines[json_start:])
            output_data = json.loads(json_content)
            assert output_data["query"] == "test query"
            assert output_data["total_results"] == 1
        else:
            pytest.fail("No JSON output found")

    def test_main_invalid_limit(self):
        """Test search with invalid limit parameter."""
        with patch("sys.argv", ["search-library", "test query", "--limit", "0"]), \
             patch("sys.stderr", StringIO()) as mock_stderr:
            exit_code = main()

        assert exit_code == 1
        assert "Error: Limit must be at least 1" in mock_stderr.getvalue()

    def test_main_invalid_threshold_low(self):
        """Test search with invalid threshold parameter (too low)."""
        with patch("sys.argv", ["search-library", "test query", "--threshold", "-0.1"]), \
             patch("sys.stderr", StringIO()) as mock_stderr:
            exit_code = main()

        assert exit_code == 1
        assert "Error: Threshold must be between 0.0 and 1.0" in mock_stderr.getvalue()

    def test_main_invalid_threshold_high(self):
        """Test search with invalid threshold parameter (too high)."""
        with patch("sys.argv", ["search-library", "test query", "--threshold", "1.1"]), \
             patch("sys.stderr", StringIO()) as mock_stderr:
            exit_code = main()

        assert exit_code == 1
        assert "Error: Threshold must be between 0.0 and 1.0" in mock_stderr.getvalue()

    @patch("home_library.cli.search.search_library")
    def test_main_search_error(self, mock_search_library):
        """Test search when an error occurs."""
        mock_search_library.side_effect = Exception("Search failed")

        with patch("sys.argv", ["search-library", "test query"]), \
             patch("sys.stderr", StringIO()) as mock_stderr:
            exit_code = main()

        assert exit_code == 1
        assert "Error performing search: Search failed" in mock_stderr.getvalue()

    @patch("home_library.cli.search.search_library")
    def test_main_keyboard_interrupt(self, mock_search_library, capsys):
        """Test search when interrupted by user."""
        mock_search_library.side_effect = KeyboardInterrupt()

        with patch("sys.argv", ["search-library", "test query"]):
            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "❌ Search interrupted by user" in captured.out

    @patch("home_library.cli.search.search_library")
    def test_main_no_results_found(self, mock_search_library, capsys):
        """Test search when no results are found."""
        mock_search_library.return_value = []

        with patch("sys.argv", ["search-library", "test query"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "🔍 No results found for query: 'test query'" in captured.out
