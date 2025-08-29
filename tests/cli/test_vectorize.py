"""Tests for the CLI vectorize command."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from home_library.cli.vectorize import _print_detailed_chunks, _print_stats, main


class TestPrintStats:
    """Test the _print_stats function."""

    def test_print_stats_basic(self, capsys):
        """Test basic statistics printing."""
        stats = {
            "file_path": "/path/to/book.epub",
            "total_chunks": 10,
            "total_words": 5000,
            "average_chunk_size": 500.0,
            "chunk_size_range": {"min": 400, "max": 600},
            "chunks_per_chapter": {0: 5, 1: 5},
            "configuration": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_dimension": 768,
                "vectorization_method": "sentence-transformers",
            },
        }

        _print_stats(stats)
        captured = capsys.readouterr()

        assert "Vectorization Statistics for: /path/to/book.epub" in captured.out
        assert "Total Chunks: 10" in captured.out
        assert "Total Words: 5,000" in captured.out
        assert "Average Chunk Size: 500.0" in captured.out
        assert "Chunk Size Range: 400 - 600 words" in captured.out
        assert "Chunk Size: 512 tokens" in captured.out
        assert "Chunk Overlap: 50 tokens" in captured.out
        assert "Embedding Dimension: 768" in captured.out
        assert "Vectorization Method: sentence-transformers" in captured.out
        assert "Chapter 0: 5 chunks" in captured.out
        assert "Chapter 1: 5 chunks" in captured.out


class TestPrintDetailedChunks:
    """Test the _print_detailed_chunks function."""

    def test_print_detailed_chunks(self, capsys):
        """Test detailed chunk information printing."""
        # Create mock chunks
        chunks = [
            Mock(
                chunk_id="chunk_0_0",
                chapter_index=0,
                chapter_title="Introduction",
                start_token=0,
                end_token=100,
                word_count=50,
                text="This is the first chunk with some sample text content.",
            ),
            Mock(
                chunk_id="chunk_1_0",
                chapter_index=1,
                chapter_title="Chapter 1",
                start_token=0,
                end_token=150,
                word_count=75,
                text="This is the second chunk from a different chapter.",
            ),
        ]

        result = Mock(chunks=chunks)

        _print_detailed_chunks(result)
        captured = capsys.readouterr()

        assert "Detailed Chunk Information:" in captured.out
        assert "Chunk ID: chunk_0_0" in captured.out
        assert "Source: Chapter 0 (Introduction)" in captured.out
        assert "Position: tokens 0-100" in captured.out
        assert "Word Count: 50" in captured.out
        assert (
            "Text Preview: This is the first chunk with some sample text content."
            in captured.out
        )
        assert "Chunk ID: chunk_1_0" in captured.out
        assert "Source: Chapter 1 (Chapter 1)" in captured.out


class TestCLIVectorize:
    """Test the main CLI function."""

    @patch("home_library.cli.vectorize.vectorize_epub")
    @patch("home_library.cli.vectorize.get_vectorization_stats")
    def test_main_success_basic(self, mock_get_stats, mock_vectorize, capsys):
        """Test successful basic vectorization."""
        # Mock the vectorization result
        mock_result = Mock()
        mock_result.chunks = []
        mock_vectorize.return_value = mock_result

        # Mock the statistics
        mock_stats = {
            "file_path": "/path/to/book.epub",
            "total_chunks": 5,
            "total_words": 2500,
            "average_chunk_size": 500.0,
            "chunk_size_range": {"min": 400, "max": 600},
            "chunks_per_chapter": {0: 3, 1: 2},
            "configuration": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_dimension": 768,
                "vectorization_method": "sentence-transformers",
            },
        }
        mock_get_stats.return_value = mock_stats

        # Test with basic arguments
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock sys.argv
            with patch("sys.argv", ["vectorize-epub", tmp_path]):
                result = main()

                assert result == 0
                captured = capsys.readouterr()
                assert (
                    "Vectorization Statistics for: /path/to/book.epub" in captured.out
                )

                # Verify functions were called
                mock_vectorize.assert_called_once_with(
                    tmp_path, chunk_size=None, chunk_overlap=None
                )
                mock_get_stats.assert_called_once_with(mock_result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("home_library.cli.vectorize.vectorize_epub")
    @patch("home_library.cli.vectorize.get_vectorization_stats")
    def test_main_success_with_custom_params(self, mock_get_stats, mock_vectorize):
        """Test successful vectorization with custom parameters."""
        # Mock the vectorization result
        mock_result = Mock()
        mock_result.chunks = []
        mock_vectorize.return_value = mock_result

        # Mock the statistics
        mock_stats = {
            "file_path": "/path/to/book.epub",
            "total_chunks": 10,
            "total_words": 5000,
            "average_chunk_size": 500.0,
            "chunk_size_range": {"min": 400, "max": 600},
            "chunks_per_chapter": {0: 5, 1: 5},
            "configuration": {
                "chunk_size": 256,
                "chunk_overlap": 25,
                "embedding_dimension": 768,
                "vectorization_method": "sentence-transformers",
            },
        }
        mock_get_stats.return_value = mock_stats

        # Test with custom parameters
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock sys.argv
            with patch(
                "sys.argv",
                [
                    "vectorize-epub",
                    tmp_path,
                    "--chunk-size",
                    "256",
                    "--chunk-overlap",
                    "25",
                ],
            ):
                result = main()

                assert result == 0

                # Verify functions were called with custom parameters
                mock_vectorize.assert_called_once_with(
                    tmp_path, chunk_size=256, chunk_overlap=25
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("home_library.cli.vectorize.vectorize_epub")
    @patch("home_library.cli.vectorize.get_vectorization_stats")
    def test_main_success_with_detailed(self, mock_get_stats, mock_vectorize, capsys):
        """Test successful vectorization with detailed output."""
        # Mock the vectorization result
        mock_chunk = Mock(
            chunk_id="chunk_0_0",
            chapter_index=0,
            chapter_title="Introduction",
            start_token=0,
            end_token=100,
            word_count=50,
            text="Sample text content for testing.",
        )
        mock_result = Mock(chunks=[mock_chunk])
        mock_vectorize.return_value = mock_result

        # Mock the statistics
        mock_stats = {
            "file_path": "/path/to/book.epub",
            "total_chunks": 1,
            "total_words": 50,
            "average_chunk_size": 50.0,
            "chunk_size_range": {"min": 50, "max": 50},
            "chunks_per_chapter": {0: 1},
            "configuration": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_dimension": 768,
                "vectorization_method": "sentence-transformers",
            },
        }
        mock_get_stats.return_value = mock_stats

        # Test with detailed flag
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock sys.argv
            with patch("sys.argv", ["vectorize-epub", tmp_path, "--detailed"]):
                result = main()

                assert result == 0
                captured = capsys.readouterr()
                assert "Detailed Chunk Information:" in captured.out
                assert "Chunk ID: chunk_0_0" in captured.out
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("home_library.cli.vectorize.vectorize_epub")
    @patch("home_library.cli.vectorize.get_vectorization_stats")
    def test_main_success_with_json(self, mock_get_stats, mock_vectorize, capsys):
        """Test successful vectorization with JSON output."""
        # Mock the vectorization result with proper mock objects that have model_dump method
        mock_chunk = Mock()
        mock_chunk.chunk_id = "chunk_0_0"
        mock_chunk.chapter_index = 0
        mock_chunk.chapter_title = "Introduction"
        mock_chunk.start_token = 0
        mock_chunk.end_token = 100
        mock_chunk.word_count = 50
        mock_chunk.text = "Sample text content for testing."
        mock_chunk.model_dump.return_value = {
            "chunk_id": "chunk_0_0",
            "chapter_index": 0,
            "chapter_title": "Introduction",
            "start_token": 0,
            "end_token": 100,
            "word_count": 50,
            "text": "Sample text content for testing.",
        }

        mock_result = Mock(chunks=[mock_chunk])
        mock_vectorize.return_value = mock_result

        # Mock the statistics
        mock_stats = {
            "file_path": "/path/to/book.epub",
            "total_chunks": 1,
            "total_words": 50,
            "average_chunk_size": 50.0,
            "chunk_size_range": {"min": 50, "max": 50},
            "chunks_per_chapter": {0: 1},
            "configuration": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_dimension": 768,
                "vectorization_method": "sentence-transformers",
            },
        }
        mock_get_stats.return_value = mock_stats

        # Test with JSON flag
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock sys.argv
            with patch("sys.argv", ["vectorize-epub", tmp_path, "--json"]):
                result = main()

                assert result == 0
                captured = capsys.readouterr()

                # Parse JSON output
                output_data = json.loads(captured.out)
                assert "stats" in output_data
                assert "chunks" in output_data
                assert output_data["stats"]["total_chunks"] == 1
                assert len(output_data["chunks"]) == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_main_file_not_found(self, capsys):
        """Test CLI with non-existent file."""
        # Mock sys.argv
        with patch("sys.argv", ["vectorize-epub", "/nonexistent/file.epub"]):
            result = main()

            assert result == 1
            captured = capsys.readouterr()
            assert "Error: File not found: /nonexistent/file.epub" in captured.err

    def test_main_invalid_extension(self, capsys):
        """Test CLI with invalid file extension."""
        # Create a temporary file with wrong extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock sys.argv
            with patch("sys.argv", ["vectorize-epub", tmp_path]):
                result = main()

                assert result == 1
                captured = capsys.readouterr()
                assert "Error: File must have .epub extension:" in captured.err
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("home_library.cli.vectorize.vectorize_epub")
    def test_main_vectorization_error(self, mock_vectorize, capsys):
        """Test CLI when vectorization fails."""
        # Mock vectorization to raise an exception
        mock_vectorize.side_effect = Exception("Vectorization failed")

        # Create a temporary EPUB file
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock sys.argv
            with patch("sys.argv", ["vectorize-epub", tmp_path]):
                result = main()

                assert result == 1
                captured = capsys.readouterr()
                assert (
                    "Error processing EPUB file: Vectorization failed" in captured.err
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
