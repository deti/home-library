"""Tests for the search service functionality."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np

from home_library.search import SearchResult, SearchService, search_library


def _mock_db_context_manager(mock_session):
    """Helper function to create a properly mocked database context manager."""
    mock_context = MagicMock()
    mock_context.__enter__ = MagicMock(return_value=mock_session)
    mock_context.__exit__ = MagicMock(return_value=None)
    return mock_context


class TestSearchResult:
    """Test the SearchResult model."""

    def test_search_result_creation(self):
        """Test creating a SearchResult instance."""
        result = SearchResult(
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

        assert result.book_title == "Test Book"
        assert result.book_author == "Test Author"
        assert result.chapter_index == 1
        assert result.chapter_title == "Test Chapter"
        assert result.chunk_index == 0
        assert result.text == "This is a test text chunk."
        assert result.start_token == 0
        assert result.end_token == 10
        assert result.word_count == 10
        assert result.similarity_score == 0.85
        assert result.file_path == "/path/to/book.epub"

    def test_search_result_optional_fields(self):
        """Test creating a SearchResult with optional fields."""
        result = SearchResult(
            book_title="Test Book",
            chapter_index=1,
            chunk_index=0,
            text="This is a test text chunk.",
            start_token=0,
            end_token=10,
            word_count=10,
            similarity_score=0.85,
            file_path="/path/to/book.epub"
        )

        assert result.book_author is None
        assert result.chapter_title is None


class TestSearchService:
    """Test the SearchService class."""

    @patch("home_library.search.get_embeddings_model")
    @patch("home_library.search.get_settings")
    def test_search_service_initialization(self, mock_get_settings, mock_get_model):
        """Test SearchService initialization."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "test-model"
        mock_settings.embeddings_device = "cpu"
        mock_get_settings.return_value = mock_settings

        # Mock model
        mock_model = Mock()
        mock_get_model.return_value = mock_model

        service = SearchService()

        assert service.model_name == "test-model"
        assert service.device == "cpu"
        assert service.model == mock_model

    @patch("home_library.search.get_embeddings_model")
    @patch("home_library.search.get_settings")
    def test_search_service_custom_parameters(self, mock_get_settings, mock_get_model):
        """Test SearchService initialization with custom parameters."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "default-model"
        mock_settings.embeddings_device = "cpu"
        mock_get_settings.return_value = mock_settings

        # Mock model
        mock_model = Mock()
        mock_get_model.return_value = mock_model

        service = SearchService(model_name="custom-model", device="cuda")

        assert service.model_name == "custom-model"
        assert service.device == "cuda"
        assert service.model == mock_model


    @patch("home_library.search.get_db_service")
    @patch("home_library.search.get_embeddings_model")
    @patch("home_library.search.get_settings")
    def test_search_no_results(self, mock_get_settings, mock_get_model, mock_get_db):
        """Test search when no results are found."""
        # Mock settings and model
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_get_model.return_value = mock_model

        # Mock database service
        mock_db_service = Mock()
        mock_session = Mock()
        mock_db_service.get_session.return_value = _mock_db_context_manager(mock_session)
        mock_get_db.return_value = mock_db_service

        # Mock empty results
        mock_result = Mock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        service = SearchService()
        results = service.search("test query")

        assert results == []

    @patch("home_library.search.get_db_service")
    @patch("home_library.search.get_embeddings_model")
    @patch("home_library.search.get_settings")
    def test_search_with_results(self, mock_get_settings, mock_get_model, mock_get_db):
        """Test search with actual results."""
        # Mock settings and model
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_get_model.return_value = mock_model

        # Mock database service
        mock_db_service = Mock()
        mock_session = Mock()
        mock_db_service.get_session.return_value = _mock_db_context_manager(mock_session)
        mock_get_db.return_value = mock_db_service

        # Create mock database objects
        epub = Mock()
        epub.title = "Test Book"
        epub.author = "Test Author"
        epub.file_path = "/path/to/book.epub"

        chapter = Mock()
        chapter.chapter_index = 1
        chapter.title = "Test Chapter"

        chunk = Mock()
        chunk.chunk_index = 0
        chunk.text = "This is a test text chunk."
        chunk.start_token = 0
        chunk.end_token = 10
        chunk.word_count = 10

        # Mock database row with similarity
        mock_row = Mock()
        mock_row.Embedding = Mock()
        mock_row.TextChunk = chunk
        mock_row.Chapter = chapter
        mock_row.Epub = epub
        mock_row.similarity = 0.85

        # Mock results
        mock_results = [mock_row]
        mock_result = Mock()
        mock_result.all.return_value = mock_results
        mock_session.execute.return_value = mock_result

        service = SearchService()
        results = service.search("test query", limit=1)

        assert len(results) == 1
        result = results[0]
        assert result.book_title == "Test Book"
        assert result.book_author == "Test Author"
        assert result.chapter_index == 1
        assert result.chapter_title == "Test Chapter"
        assert result.chunk_index == 0
        assert result.text == "This is a test text chunk."
        assert result.similarity_score == 0.85

    @patch("home_library.search.get_db_service")
    @patch("home_library.search.get_embeddings_model")
    @patch("home_library.search.get_settings")
    def test_search_similarity_threshold(self, mock_get_settings, mock_get_model, mock_get_db):
        """Test search with similarity threshold filtering."""
        # Mock settings and model
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.9, 0.8, 0.7])]  # Different from stored embedding
        mock_get_model.return_value = mock_model

        # Mock database service
        mock_db_service = Mock()
        mock_session = Mock()
        mock_db_service.get_session.return_value = _mock_db_context_manager(mock_session)
        mock_get_db.return_value = mock_db_service

        # Create mock database objects
        epub = Mock()
        epub.title = "Test Book"
        epub.author = "Test Author"
        epub.file_path = "/path/to/book.epub"

        chapter = Mock()
        chapter.chapter_index = 1
        chapter.title = "Test Chapter"

        chunk = Mock()
        chunk.chunk_index = 0
        chunk.text = "This is a test text chunk."
        chunk.start_token = 0
        chunk.end_token = 10
        chunk.word_count = 10

        # Mock database row with similarity
        mock_row = Mock()
        mock_row.Embedding = Mock()
        mock_row.TextChunk = chunk
        mock_row.Chapter = chapter
        mock_row.Epub = epub
        mock_row.similarity = 0.7

        service = SearchService()

        # Test with very high threshold (should filter out results)
        # Mock empty results for high threshold
        mock_result_empty = Mock()
        mock_result_empty.all.return_value = []
        mock_session.execute.return_value = mock_result_empty

        results = service.search("test query", limit=1, similarity_threshold=0.99)
        assert len(results) == 0

        # Test with low threshold (should include results)
        # Mock results with data for low threshold
        mock_result_with_data = Mock()
        mock_result_with_data.all.return_value = [mock_row]
        mock_session.execute.return_value = mock_result_with_data

        results = service.search("test query", limit=1, similarity_threshold=0.1)
        assert len(results) == 1

    @patch("home_library.search.get_db_service")
    @patch("home_library.search.get_embeddings_model")
    @patch("home_library.search.get_settings")
    def test_search_limit_results(self, mock_get_settings, mock_get_model, mock_get_db):
        """Test search result limiting."""
        # Mock settings and model
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_get_model.return_value = mock_model

        # Mock database service
        mock_db_service = Mock()
        mock_session = Mock()
        mock_db_service.get_session.return_value = _mock_db_context_manager(mock_session)
        mock_get_db.return_value = mock_db_service

        # Create multiple mock results
        mock_results = []
        for i in range(10):
            epub = Mock()
            epub.title = f"Test Book {i}"
            epub.author = f"Test Author {i}"
            epub.file_path = f"/path/to/book{i}.epub"

            chapter = Mock()
            chapter.chapter_index = i
            chapter.title = f"Test Chapter {i}"

            chunk = Mock()
            chunk.chunk_index = i
            chunk.text = f"This is test text chunk {i}."
            chunk.start_token = i * 10
            chunk.end_token = (i + 1) * 10
            chunk.word_count = 10

            # Mock database row with similarity
            mock_row = Mock()
            mock_row.Embedding = Mock()
            mock_row.TextChunk = chunk
            mock_row.Chapter = chapter
            mock_row.Epub = epub
            mock_row.similarity = 0.8 - (i * 0.05)  # Decreasing similarity

            mock_results.append(mock_row)

        service = SearchService()

        # Test with limit 3
        mock_result_3 = Mock()
        mock_result_3.all.return_value = mock_results[:3]  # Simulate database LIMIT
        mock_session.execute.return_value = mock_result_3
        results = service.search("test query", limit=3)
        assert len(results) == 3

        # Test with limit 1
        mock_result_1 = Mock()
        mock_result_1.all.return_value = mock_results[:1]  # Simulate database LIMIT
        mock_session.execute.return_value = mock_result_1
        results = service.search("test query", limit=1)
        assert len(results) == 1

    @patch("home_library.search.get_db_service")
    @patch("home_library.search.get_embeddings_model")
    @patch("home_library.search.get_settings")
    def test_search_invalid_embedding_data(self, mock_get_settings, mock_get_model, mock_get_db):
        """Test search handling of invalid embedding data."""
        # Mock settings and model
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_get_model.return_value = mock_model

        # Mock database service
        mock_db_service = Mock()
        mock_session = Mock()
        mock_db_service.get_session.return_value = _mock_db_context_manager(mock_session)
        mock_get_db.return_value = mock_db_service

        # Mock empty results (database would handle invalid data gracefully)
        mock_result = Mock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        service = SearchService()
        results = service.search("test query")

        # Should handle invalid data gracefully and return empty results
        assert len(results) == 0


class TestConvenienceFunctions:
    """Test the convenience functions."""

    @patch("home_library.search.SearchService")
    def test_search_library_function(self, mock_search_service_class):
        """Test the search_library convenience function."""
        # Mock SearchService
        mock_service = Mock()
        mock_service.search.return_value = ["result1", "result2"]
        mock_search_service_class.return_value = mock_service

        # Test the function
        results = search_library("test query", limit=5, similarity_threshold=0.5)

        # Verify SearchService was created with correct parameters
        mock_search_service_class.assert_called_once_with(None, None)

        # Verify search was called with correct parameters
        mock_service.search.assert_called_once_with("test query", 5, 0.5)

        # Verify results
        assert results == ["result1", "result2"]

    @patch("home_library.search.SearchService")
    def test_search_library_with_custom_parameters(self, mock_search_service_class):
        """Test search_library with custom model and device parameters."""
        # Mock SearchService
        mock_service = Mock()
        mock_service.search.return_value = ["result1"]
        mock_search_service_class.return_value = mock_service

        # Test with custom parameters
        results = search_library(
            "test query",
            limit=3,
            similarity_threshold=0.7,
            model_name="custom-model",
            device="cuda"
        )

        # Verify SearchService was created with custom parameters
        mock_search_service_class.assert_called_once_with("custom-model", "cuda")

        # Verify search was called with correct parameters
        mock_service.search.assert_called_once_with("test query", 3, 0.7)

        # Verify results
        assert results == ["result1"]
