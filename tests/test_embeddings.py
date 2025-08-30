"""Tests for the embeddings module."""

import logging
from unittest.mock import Mock, patch

import numpy as np

from home_library.embeddings import (
    EmbeddingChunk,
    EmbeddingsCreator,
    EmbeddingsResult,
    create_embeddings_for_epub,
)
from home_library.vectorizer import TextChunk


# Configure logging for tests to avoid noise
logging.basicConfig(level=logging.CRITICAL)


class TestEmbeddingChunk:
    """Test the EmbeddingChunk model."""

    def test_embedding_chunk_creation(self):
        """Test creating an EmbeddingChunk instance."""
        chunk = TextChunk(
            text="Test text",
            chunk_id="test_1",
            chapter_index=0,
            chapter_title="Test Chapter",
            start_token=0,
            end_token=10,
            word_count=10,
        )

        embedding = [0.1, 0.2, 0.3]
        embedding_norm = 0.3742

        embedding_chunk = EmbeddingChunk(
            chunk=chunk,
            embedding=embedding,
            embedding_norm=embedding_norm,
        )

        assert embedding_chunk.chunk == chunk
        assert embedding_chunk.embedding == embedding
        assert embedding_chunk.embedding_norm == embedding_norm


class TestEmbeddingsResult:
    """Test the EmbeddingsResult model."""

    def test_embeddings_result_creation(self):
        """Test creating an EmbeddingsResult instance."""
        chunks = []
        result = EmbeddingsResult(
            file_path="/test/path.epub",
            total_chunks=0,
            total_words=0,
            embedding_dimension=384,
            model_name="test-model",
            device="cpu",
            batch_size=32,
            chunks=chunks,
            processing_time_seconds=1.5,
        )

        assert result.file_path == "/test/path.epub"
        assert result.total_chunks == 0
        assert result.total_words == 0
        assert result.embedding_dimension == 384
        assert result.model_name == "test-model"
        assert result.device == "cpu"
        assert result.batch_size == 32
        assert result.chunks == chunks
        assert result.processing_time_seconds == 1.5


class TestEmbeddingsCreator:
    """Test the EmbeddingsCreator class."""

    @patch("home_library.embeddings.SentenceTransformer")
    @patch("home_library.embeddings.get_settings")
    def test_init_with_defaults(self, mock_get_settings, mock_sentence_transformer):
        """Test EmbeddingsCreator initialization with default settings."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "default-model"
        mock_settings.embeddings_device = "cpu"
        mock_settings.embeddings_batch_size = 32
        mock_get_settings.return_value = mock_settings

        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        creator = EmbeddingsCreator()

        assert creator.model_name == "default-model"
        assert creator.device == "cpu"
        assert creator.batch_size == 32
        assert creator.embedding_dimension == 384
        mock_sentence_transformer.assert_called_once_with("default-model", device="cpu")

    @patch("home_library.embeddings.SentenceTransformer")
    @patch("home_library.embeddings.get_settings")
    def test_init_with_overrides(self, mock_get_settings, mock_sentence_transformer):
        """Test EmbeddingsCreator initialization with custom parameters."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "default-model"
        mock_settings.embeddings_device = "cpu"
        mock_settings.embeddings_batch_size = 32
        mock_get_settings.return_value = mock_settings

        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        creator = EmbeddingsCreator(
            model_name="custom-model", device="cuda", batch_size=64
        )

        assert creator.model_name == "custom-model"
        assert creator.device == "cuda"
        assert creator.batch_size == 64
        assert creator.embedding_dimension == 768
        mock_sentence_transformer.assert_called_once_with("custom-model", device="cuda")

    @patch("home_library.embeddings.vectorize_epub")
    @patch("home_library.embeddings.SentenceTransformer")
    @patch("home_library.embeddings.get_settings")
    def test_create_embeddings_for_epub_success(
        self, mock_get_settings, mock_sentence_transformer, mock_vectorize_epub
    ):
        """Test successful embeddings creation for an EPUB file."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "default-model"
        mock_settings.embeddings_device = "cpu"
        mock_settings.embeddings_batch_size = 32
        mock_get_settings.return_value = mock_settings

        # Mock vectorization result
        mock_chunk = TextChunk(
            text="Test text content",
            chunk_id="chunk_0_0",
            chapter_index=0,
            chapter_title="Test Chapter",
            start_token=0,
            end_token=10,
            word_count=10,
        )
        mock_vectorization_result = Mock()
        mock_vectorization_result.chunks = [mock_chunk]
        mock_vectorization_result.total_words = 10
        mock_vectorize_epub.return_value = mock_vectorization_result

        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model

        creator = EmbeddingsCreator()
        result = creator.create_embeddings_for_epub("/test/path.epub")

        assert result.total_chunks == 1
        assert result.total_words == 10
        assert result.embedding_dimension == 384
        assert len(result.chunks) == 1

        chunk = result.chunks[0]
        assert chunk.chunk == mock_chunk
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.embedding_norm > 0

    @patch("home_library.embeddings.vectorize_epub")
    @patch("home_library.embeddings.SentenceTransformer")
    @patch("home_library.embeddings.get_settings")
    def test_create_embeddings_for_epub_no_chunks(
        self, mock_get_settings, mock_sentence_transformer, mock_vectorize_epub
    ):
        """Test embeddings creation when no text chunks are found."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "default-model"
        mock_settings.embeddings_device = "cpu"
        mock_settings.embeddings_batch_size = 32
        mock_get_settings.return_value = mock_settings

        # Mock empty vectorization result
        mock_vectorization_result = Mock()
        mock_vectorization_result.chunks = []
        mock_vectorization_result.total_words = 0
        mock_vectorize_epub.return_value = mock_vectorization_result

        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        creator = EmbeddingsCreator()
        result = creator.create_embeddings_for_epub("/test/path.epub")

        assert result.total_chunks == 0
        assert result.total_words == 0
        assert result.chunks == []

    @patch("home_library.embeddings.vectorize_epub")
    @patch("home_library.embeddings.SentenceTransformer")
    @patch("home_library.embeddings.get_settings")
    def test_create_embeddings_with_custom_chunk_params(
        self, mock_get_settings, mock_sentence_transformer, mock_vectorize_epub
    ):
        """Test embeddings creation with custom chunk parameters."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "default-model"
        mock_settings.embeddings_device = "cpu"
        mock_settings.embeddings_batch_size = 32
        mock_get_settings.return_value = mock_settings

        # Mock vectorization result
        mock_vectorization_result = Mock()
        mock_vectorization_result.chunks = []
        mock_vectorization_result.total_words = 0
        mock_vectorize_epub.return_value = mock_vectorization_result

        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        creator = EmbeddingsCreator()
        creator.create_embeddings_for_epub(
            "/test/path.epub", chunk_size=256, chunk_overlap=25
        )

        mock_vectorize_epub.assert_called_once_with("/test/path.epub", 256, 25)

    def test_get_embeddings_stats_empty_result(self):
        """Test getting stats for an empty embeddings result."""
        empty_result = EmbeddingsResult(
            file_path="/test/path.epub",
            total_chunks=0,
            total_words=0,
            embedding_dimension=384,
            model_name="test-model",
            device="cpu",
            batch_size=32,
            chunks=[],
            processing_time_seconds=1.0,
        )

        creator = EmbeddingsCreator()
        stats = creator.get_embeddings_stats(empty_result)

        assert stats["total_chunks"] == 0
        assert stats["total_words"] == 0
        assert stats["message"] == "No chunks processed"

    def test_get_embeddings_stats_with_chunks(self):
        """Test getting stats for an embeddings result with chunks."""
        # Create test chunks
        chunk1 = TextChunk(
            text="First chunk text",
            chunk_id="chunk_0_0",
            chapter_index=0,
            chapter_title="Chapter 1",
            start_token=0,
            end_token=10,
            word_count=10,
        )
        chunk2 = TextChunk(
            text="Second chunk text",
            chunk_id="chunk_0_1",
            chapter_index=0,
            chapter_title="Chapter 1",
            start_token=10,
            end_token=20,
            word_count=10,
        )

        embedding_chunk1 = EmbeddingChunk(
            chunk=chunk1,
            embedding=[0.1, 0.2, 0.3],
            embedding_norm=0.3742,
        )
        embedding_chunk2 = EmbeddingChunk(
            chunk=chunk2,
            embedding=[0.4, 0.5, 0.6],
            embedding_norm=0.8771,
        )

        result = EmbeddingsResult(
            file_path="/test/path.epub",
            total_chunks=2,
            total_words=20,
            embedding_dimension=3,
            model_name="test-model",
            device="cpu",
            batch_size=32,
            chunks=[embedding_chunk1, embedding_chunk2],
            processing_time_seconds=2.0,
        )

        creator = EmbeddingsCreator()
        stats = creator.get_embeddings_stats(result)

        assert stats["total_chunks"] == 2
        assert stats["total_words"] == 20
        assert stats["average_chunk_size"] == 10.0
        assert stats["chunk_size_range"]["min"] == 10
        assert stats["chunk_size_range"]["max"] == 10
        assert stats["embedding_norms"]["average"] == 0.6257
        assert stats["embedding_norms"]["min"] == 0.3742
        assert stats["embedding_norms"]["max"] == 0.8771
        assert stats["efficiency"]["chunks_per_second"] == 1.0
        assert stats["efficiency"]["words_per_second"] == 10.0
        assert stats["chunks_per_chapter"] == {0: 2}


class TestConvenienceFunctions:
    """Test the convenience functions."""

    @patch("home_library.embeddings.EmbeddingsCreator")
    def test_create_embeddings_for_epub_function(self, mock_creator_class):
        """Test the create_embeddings_for_epub convenience function."""
        # Mock creator instance
        mock_creator = Mock()
        mock_creator.create_embeddings_for_epub.return_value = "test_result"
        mock_creator_class.return_value = mock_creator

        result = create_embeddings_for_epub(
            "/test/path.epub",
            model_name="test-model",
            device="cuda",
            batch_size=64,
            chunk_size=256,
            chunk_overlap=25,
        )

        mock_creator_class.assert_called_once_with("test-model", "cuda", 64)
        mock_creator.create_embeddings_for_epub.assert_called_once_with(
            "/test/path.epub", 256, 25
        )
        assert result == "test_result"


class TestIntegration:
    """Integration tests for the embeddings module."""

    @patch("home_library.embeddings.SentenceTransformer")
    @patch("home_library.embeddings.vectorize_epub")
    @patch("home_library.embeddings.get_settings")
    def test_full_embeddings_pipeline(
        self, mock_get_settings, mock_vectorize_epub, mock_sentence_transformer
    ):
        """Test the full embeddings pipeline from EPUB to embeddings."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "all-MiniLM-L6-v2"
        mock_settings.embeddings_device = "cpu"
        mock_settings.embeddings_batch_size = 32
        mock_get_settings.return_value = mock_settings

        # Mock vectorization result with multiple chunks
        chunks = [
            TextChunk(
                text=f"Text content for chunk {i}",
                chunk_id=f"chunk_0_{i}",
                chapter_index=0,
                chapter_title="Test Chapter",
                start_token=i * 10,
                end_token=(i + 1) * 10,
                word_count=10,
            )
            for i in range(3)
        ]

        mock_vectorization_result = Mock()
        mock_vectorization_result.chunks = chunks
        mock_vectorization_result.total_words = 30
        mock_vectorize_epub.return_value = mock_vectorization_result

        # Mock SentenceTransformer with realistic embeddings
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        # Create realistic-looking embeddings
        rng = np.random.default_rng(42)  # Use fixed seed for reproducible tests
        mock_embeddings = rng.random((3, 384)).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        # Run the pipeline
        creator = EmbeddingsCreator()
        result = creator.create_embeddings_for_epub("/test/path.epub")

        # Verify results
        assert result.total_chunks == 3
        assert result.total_words == 30
        assert result.embedding_dimension == 384
        assert result.model_name == "all-MiniLM-L6-v2"
        assert result.device == "cpu"
        assert result.batch_size == 32
        assert len(result.chunks) == 3

        # Verify each chunk has proper embeddings
        for i, embedding_chunk in enumerate(result.chunks):
            assert embedding_chunk.chunk == chunks[i]
            assert len(embedding_chunk.embedding) == 384
            assert embedding_chunk.embedding_norm > 0

        # Verify the model was called correctly
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert len(call_args[0][0]) == 3  # 3 texts
        assert call_args[1]["batch_size"] == 32
        assert call_args[1]["show_progress_bar"] is True
        assert call_args[1]["convert_to_numpy"] is True
