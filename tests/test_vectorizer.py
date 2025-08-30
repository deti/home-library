"""Tests for the vectorizer module."""

import logging
from unittest.mock import Mock, patch

import pytest

from home_library.vectorizer import (
    TextChunk,
    VectorizationResult,
    _create_chunks_from_text,
    _tokenize_text,
    get_vectorization_stats,
    vectorize_epub,
)


# Configure logging for tests to avoid noise
logging.basicConfig(level=logging.CRITICAL)


class TestTextChunk:
    """Test TextChunk model."""

    def test_text_chunk_creation(self):
        """Test creating a TextChunk instance."""
        chunk = TextChunk(
            text="Sample text content",
            chunk_id="chunk_0_0",
            chapter_index=0,
            chapter_title="Introduction",
            start_token=0,
            end_token=3,
            word_count=3,
        )

        assert chunk.text == "Sample text content"
        assert chunk.chunk_id == "chunk_0_0"
        assert chunk.chapter_index == 0
        assert chunk.chapter_title == "Introduction"
        assert chunk.start_token == 0
        assert chunk.end_token == 3
        assert chunk.word_count == 3


class TestVectorizationResult:
    """Test VectorizationResult model."""

    def test_vectorization_result_creation(self):
        """Test creating a VectorizationResult instance."""
        chunks = [
            TextChunk(
                text="Sample text",
                chunk_id="chunk_0_0",
                chapter_index=0,
                chapter_title="Chapter 1",
                start_token=0,
                end_token=2,
                word_count=2,
            )
        ]

        result = VectorizationResult(
            file_path="/path/to/book.epub",
            total_chunks=1,
            total_words=2,
            chunk_size=512,
            chunk_overlap=50,
            embedding_dimension=768,
            vectorization_method="sentence-transformers",
            chunks=chunks,
        )

        assert result.file_path == "/path/to/book.epub"
        assert result.total_chunks == 1
        assert result.total_words == 2
        assert result.chunk_size == 512
        assert result.chunk_overlap == 50
        assert result.embedding_dimension == 768
        assert result.vectorization_method == "sentence-transformers"
        assert len(result.chunks) == 1


class TestTokenization:
    """Test text tokenization functions."""

    def test_tokenize_text_simple(self):
        """Test basic text tokenization."""
        text = "Hello world. This is a test!"
        tokens = _tokenize_text(text)
        expected = ["Hello", "world", "This", "is", "a", "test"]
        assert tokens == expected

    def test_tokenize_text_with_punctuation(self):
        """Test tokenization with various punctuation marks."""
        text = "Hello, world! How are you? I'm fine."
        tokens = _tokenize_text(text)
        expected = ["Hello", "world", "How", "are", "you", "I", "m", "fine"]
        assert tokens == expected

    def test_tokenize_text_empty(self):
        """Test tokenization of empty text."""
        tokens = _tokenize_text("")
        assert tokens == []

    def test_tokenize_text_whitespace_only(self):
        """Test tokenization of whitespace-only text."""
        tokens = _tokenize_text("   \n\t  ")
        assert tokens == []


class TestChunkCreation:
    """Test text chunk creation functions."""

    def test_create_chunks_short_text(self):
        """Test chunking of text shorter than chunk size."""
        text = "This is a short text that fits in one chunk."
        chunks = _create_chunks_from_text(text, 0, "Chapter 1", 100, 10)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.text == text
        assert chunk.chunk_id == "chunk_0_0"
        assert chunk.chapter_index == 0
        assert chunk.chapter_title == "Chapter 1"
        assert chunk.start_token == 0
        assert chunk.end_token > 0
        assert chunk.word_count > 0

    def test_create_chunks_long_text(self):
        """Test chunking of text longer than chunk size."""
        # Create text with more than 10 words
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = _create_chunks_from_text(text, 1, "Chapter 2", 10, 2)

        assert len(chunks) > 1
        assert all(chunk.word_count <= 10 for chunk in chunks)

        # Check chunk IDs
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"chunk_1_{i}"
            assert chunk.chapter_index == 1
            assert chunk.chapter_title == "Chapter 2"

    def test_create_chunks_empty_text(self):
        """Test chunking of empty text."""
        chunks = _create_chunks_from_text("", 0, "Chapter 1", 100, 10)
        assert chunks == []

    def test_create_chunks_with_overlap(self):
        """Test that chunks have proper overlap."""
        text = " ".join([f"word{i}" for i in range(15)])
        chunks = _create_chunks_from_text(text, 0, "Chapter 1", 10, 3)

        if len(chunks) > 1:
            # Check that consecutive chunks have overlap
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                # The end token of current should be greater than start of next
                assert current_chunk.end_token > next_chunk.start_token


class TestVectorization:
    """Test the main vectorization function."""

    @patch("home_library.vectorizer.parse_epub")
    @patch("home_library.vectorizer.get_settings")
    def test_vectorize_epub_basic(self, mock_get_settings, mock_parse_epub):
        """Test basic EPUB vectorization."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.chunk_size = 100
        mock_settings.chunk_overlap = 20
        mock_settings.embedding_dimension = 768
        mock_settings.vectorization_method = "sentence-transformers"
        mock_get_settings.return_value = mock_settings

        # Mock EPUB parsing
        mock_chapter = Mock()
        mock_chapter.index = 0
        mock_chapter.title = "Introduction"
        mock_chapter.text = " ".join([f"word{i}" for i in range(25)])

        mock_epub_details = Mock()
        mock_epub_details.chapters = [mock_chapter]
        mock_parse_epub.return_value = mock_epub_details

        # Test vectorization
        result = vectorize_epub("/path/to/book.epub")

        assert result.file_path == "/path/to/book.epub"
        assert result.total_chunks > 0
        assert result.chunk_size == 100
        assert result.chunk_overlap == 20
        assert result.embedding_dimension == 768
        assert result.vectorization_method == "sentence-transformers"

        # Verify parse_epub was called
        mock_parse_epub.assert_called_once_with("/path/to/book.epub", include_text=True)

    @patch("home_library.vectorizer.parse_epub")
    @patch("home_library.vectorizer.get_settings")
    def test_vectorize_epub_custom_params(self, mock_get_settings, mock_parse_epub):
        """Test EPUB vectorization with custom parameters."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.chunk_size = 100
        mock_settings.chunk_overlap = 20
        mock_settings.embedding_dimension = 768
        mock_settings.vectorization_method = "sentence-transformers"
        mock_get_settings.return_value = mock_settings

        # Mock EPUB parsing
        mock_chapter = Mock()
        mock_chapter.index = 0
        mock_chapter.title = "Introduction"
        mock_chapter.text = " ".join([f"word{i}" for i in range(25)])

        mock_epub_details = Mock()
        mock_epub_details.chapters = [mock_chapter]
        mock_parse_epub.return_value = mock_epub_details

        # Test with custom parameters
        result = vectorize_epub("/path/to/book.epub", chunk_size=50, chunk_overlap=10)

        assert result.chunk_size == 50
        assert result.chunk_overlap == 10


class TestStatistics:
    """Test statistics generation functions."""

    def test_get_vectorization_stats_basic(self):
        """Test basic statistics generation."""
        chunks = [
            TextChunk(
                text="First chunk text",
                chunk_id="chunk_0_0",
                chapter_index=0,
                chapter_title="Chapter 1",
                start_token=0,
                end_token=3,
                word_count=3,
            ),
            TextChunk(
                text="Second chunk text",
                chunk_id="chunk_0_1",
                chapter_index=0,
                chapter_title="Chapter 1",
                start_token=3,
                end_token=6,
                word_count=3,
            ),
            TextChunk(
                text="Third chunk from different chapter",
                chunk_id="chunk_1_0",
                chapter_index=1,
                chapter_title="Chapter 2",
                start_token=0,
                end_token=4,
                word_count=4,
            ),
        ]

        result = VectorizationResult(
            file_path="/path/to/book.epub",
            total_chunks=3,
            total_words=10,
            chunk_size=512,
            chunk_overlap=50,
            embedding_dimension=768,
            vectorization_method="sentence-transformers",
            chunks=chunks,
        )

        stats = get_vectorization_stats(result)

        assert stats["file_path"] == "/path/to/book.epub"
        assert stats["total_chunks"] == 3
        assert stats["total_words"] == 10
        assert stats["average_chunk_size"] == pytest.approx(3.33, rel=1e-2)
        assert stats["chunk_size_range"]["min"] == 3
        assert stats["chunk_size_range"]["max"] == 4
        assert stats["chunks_per_chapter"] == {0: 2, 1: 1}
        assert stats["configuration"]["chunk_size"] == 512
        assert stats["configuration"]["chunk_overlap"] == 50
        assert stats["configuration"]["embedding_dimension"] == 768
        assert stats["configuration"]["vectorization_method"] == "sentence-transformers"

    def test_get_vectorization_stats_empty(self):
        """Test statistics generation with no chunks."""
        result = VectorizationResult(
            file_path="/path/to/book.epub",
            total_chunks=0,
            total_words=0,
            chunk_size=512,
            chunk_overlap=50,
            embedding_dimension=768,
            vectorization_method="sentence-transformers",
            chunks=[],
        )

        stats = get_vectorization_stats(result)

        assert stats["total_chunks"] == 0
        assert stats["total_words"] == 0
        assert stats["average_chunk_size"] == 0
        assert stats["chunk_size_range"]["min"] == 0
        assert stats["chunk_size_range"]["max"] == 0
        assert stats["chunks_per_chapter"] == {}
