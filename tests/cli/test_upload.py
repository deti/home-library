"""Tests for the upload CLI command."""

import sys
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import pytest

from home_library.cli.upload import list_epubs, main, upload_epub


class TestUploadEpub:
    """Test cases for upload_epub function."""

    @patch("home_library.cli.upload.Path")
    @patch("home_library.cli.upload.parse_epub")
    @patch("home_library.cli.upload.get_db_service")
    @patch("home_library.cli.upload.vectorize_epub")
    @patch("home_library.cli.upload.get_settings")
    @patch("home_library.embeddings.get_embeddings_model")
    def test_upload_epub_success(self, mock_get_model, mock_get_settings,
                                mock_vectorize, mock_get_db_service,
                                mock_parse_epub, mock_path):
        """Test successful EPUB upload."""
        # Mock file path
        mock_file_path = Mock()
        mock_file_path.exists.return_value = True
        mock_file_path.suffix.lower.return_value = ".epub"
        mock_file_path.stat.return_value.st_size = 1024
        mock_path.return_value = mock_file_path

        # Mock EPUB parsing
        mock_epub_details = Mock()
        mock_epub_details.metadata = Mock()
        mock_epub_details.metadata.title = "Test Book"
        mock_epub_details.metadata.authors = ["Test Author"]
        mock_epub_details.metadata.language = "en"
        mock_epub_details.chapters = [
            Mock(index=0, title="Chapter 1", file_name="ch1.xhtml", text="Chapter 1 text"),
            Mock(index=1, title="Chapter 2", file_name="ch2.xhtml", text="Chapter 2 text")
        ]
        mock_parse_epub.return_value = mock_epub_details

        # Mock database service
        mock_db_service = Mock()
        mock_session = Mock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_session
        mock_context.__exit__.return_value = None
        mock_db_service.get_session.return_value = mock_context
        mock_get_db_service.return_value = mock_db_service

        # Mock vectorization
        mock_vectorization_result = Mock()
        mock_vectorization_result.chunks = [
            Mock(
                chunk_id="chunk_0_0",
                chapter_index=0,
                text="Chapter 1 chunk",
                start_token=0,
                end_token=5,
                word_count=3
            ),
            Mock(
                chunk_id="chunk_1_0",
                chapter_index=1,
                text="Chapter 2 chunk",
                start_token=0,
                end_token=5,
                word_count=3
            )
        ]
        mock_vectorize.return_value = mock_vectorization_result

        # Mock settings
        mock_settings = Mock()
        mock_settings.embeddings_model = "test-model"
        mock_get_settings.return_value = mock_settings

        # Mock embeddings model
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_get_model.return_value = mock_model

        # Mock database queries
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            upload_epub("test.epub", generate_embeddings=True)
            output = captured_output.getvalue()

            # Verify output messages
            assert "📖 Parsing EPUB:" in output
            assert "📚 Created EPUB record: Test Book" in output
            assert "📑 Created 2 chapter records" in output
            assert "🔍 Generating text chunks and embeddings..." in output
            assert "📝 Created 2 text chunk records" in output
            assert "🧠 Generating embeddings..." in output
            assert "🎯 Generated 2 embeddings" in output
            assert "✅ Successfully uploaded EPUB: Test Book" in output

            # Verify database operations
            mock_session.add.assert_called()
            mock_session.flush.assert_called()
            mock_session.commit.assert_called()

        finally:
            sys.stdout = sys.__stdout__

    @patch("home_library.cli.upload.Path")
    def test_upload_epub_file_not_found(self, mock_path):
        """Test EPUB upload with non-existent file."""
        mock_file_path = Mock()
        mock_file_path.exists.return_value = False
        mock_path.return_value = mock_file_path

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            with pytest.raises(SystemExit) as exc_info:
                upload_epub("nonexistent.epub")
            # Direct function calls return code 1 for errors
            assert exc_info.value.code == 1
            output = captured_output.getvalue()
            assert "❌ File not found:" in output
        finally:
            sys.stdout = sys.__stdout__

    @patch("home_library.cli.upload.Path")
    def test_upload_epub_not_epub_file(self, mock_path):
        """Test EPUB upload with non-EPUB file."""
        mock_file_path = Mock()
        mock_file_path.exists.return_value = True
        mock_file_path.suffix.lower.return_value = ".txt"
        mock_path.return_value = mock_file_path

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            with pytest.raises(SystemExit) as exc_info:
                upload_epub("test.txt")
            # Direct function calls return code 1 for errors
            assert exc_info.value.code == 1
            output = captured_output.getvalue()
            assert "❌ File is not an EPUB:" in output
        finally:
            sys.stdout = sys.__stdout__

    @patch("home_library.cli.upload.Path")
    @patch("home_library.cli.upload.parse_epub")
    @patch("home_library.cli.upload.get_db_service")
    def test_upload_epub_existing_epub_replace(self, mock_get_db_service,
                                             mock_parse_epub, mock_path):
        """Test EPUB upload with existing EPUB and user chooses to replace."""
        # Mock file path
        mock_file_path = Mock()
        mock_file_path.exists.return_value = True
        mock_file_path.suffix.lower.return_value = ".epub"
        mock_file_path.stat.return_value.st_size = 1024
        mock_path.return_value = mock_file_path

        # Mock EPUB parsing
        mock_epub_details = Mock()
        mock_epub_details.metadata = Mock()
        mock_epub_details.metadata.title = "Test Book"
        mock_epub_details.metadata.authors = ["Test Author"]
        mock_epub_details.metadata.language = "en"
        mock_epub_details.chapters = []
        mock_parse_epub.return_value = mock_epub_details

        # Mock database service
        mock_db_service = Mock()
        mock_session = Mock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_session
        mock_context.__exit__.return_value = None
        mock_db_service.get_session.return_value = mock_context
        mock_get_db_service.return_value = mock_db_service

        # Mock existing EPUB
        mock_existing_epub = Mock()
        mock_existing_epub.title = "Existing Book"
        mock_session.query.return_value.filter.return_value.first.return_value = mock_existing_epub

        # Mock user input to replace
        with patch("builtins.input", return_value="y"):
            # Capture stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            try:
                upload_epub("test.epub", generate_embeddings=False)
                output = captured_output.getvalue()
                assert "⚠️  EPUB already exists in database: Existing Book" in output
                assert "📚 Created EPUB record: Test Book" in output

                # Verify existing EPUB was deleted
                mock_session.delete.assert_called_with(mock_existing_epub)

            finally:
                sys.stdout = sys.__stdout__

    @patch("home_library.cli.upload.Path")
    @patch("home_library.cli.upload.parse_epub")
    @patch("home_library.cli.upload.get_db_service")
    def test_upload_epub_existing_epub_cancel(self, mock_get_db_service,
                                            mock_parse_epub, mock_path):
        """Test EPUB upload with existing EPUB and user cancels."""
        # Mock file path
        mock_file_path = Mock()
        mock_file_path.exists.return_value = True
        mock_file_path.suffix.lower.return_value = ".epub"
        mock_path.return_value = mock_file_path

        # Mock EPUB parsing
        mock_epub_details = Mock()
        mock_parse_epub.return_value = mock_epub_details

        # Mock database service
        mock_db_service = Mock()
        mock_session = Mock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_session
        mock_context.__exit__.return_value = None
        mock_db_service.get_session.return_value = mock_context
        mock_get_db_service.return_value = mock_db_service

        # Mock existing EPUB
        mock_existing_epub = Mock()
        mock_existing_epub.title = "Existing Book"
        mock_session.query.return_value.filter.return_value.first.return_value = mock_existing_epub

        # Mock user input to cancel
        with patch("builtins.input", return_value="n"):
            # Capture stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            try:
                upload_epub("test.epub", generate_embeddings=False)
                output = captured_output.getvalue()
                assert "⚠️  EPUB already exists in database: Existing Book" in output
                assert "Upload cancelled." in output

                # Verify no new EPUB was created
                mock_session.add.assert_not_called()

            finally:
                sys.stdout = sys.__stdout__


class TestListEpubs:
    """Test cases for list_epubs function."""

    @patch("home_library.cli.upload.get_db_service")
    def test_list_epubs_empty(self, mock_get_db_service):
        """Test listing EPUBs when database is empty."""
        mock_db_service = Mock()
        mock_session = Mock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_session
        mock_context.__exit__.return_value = None
        mock_db_service.get_session.return_value = mock_context
        mock_get_db_service.return_value = mock_db_service

        # Mock empty query result
        mock_session.query.return_value.all.return_value = []

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            list_epubs()
            output = captured_output.getvalue()
            assert "📚 No EPUBs found in database." in output
        finally:
            sys.stdout = sys.__stdout__

    @patch("home_library.cli.upload.get_db_service")
    def test_list_epubs_with_data(self, mock_get_db_service):
        """Test listing EPUBs when database has data."""
        mock_db_service = Mock()
        mock_session = Mock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_session
        mock_context.__exit__.return_value = None
        mock_db_service.get_session.return_value = mock_context
        mock_get_db_service.return_value = mock_db_service

        # Mock EPUB data
        mock_epub = Mock()
        mock_epub.title = "Test Book"
        mock_epub.author = "Test Author"
        mock_epub.file_path = "/path/to/book.epub"
        mock_epub.created_at = Mock()
        mock_epub.created_at.strftime.return_value = "2024-01-01 12:00:00"

        mock_session.query.return_value.all.return_value = [mock_epub]

        # Mock count queries
        mock_count_query = Mock()
        mock_count_query.count.return_value = 3  # chapters
        mock_session.query.return_value.filter.return_value = mock_count_query

        mock_chunk_count_query = Mock()
        mock_chunk_count_query.count.return_value = 15  # chunks
        mock_session.query.return_value.filter.return_value = mock_chunk_count_query

        mock_embedding_count_query = Mock()
        mock_embedding_count_query.join.return_value.filter.return_value.count.return_value = 15  # embeddings

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            list_epubs()
            output = captured_output.getvalue()
            assert "📚 Found 1 EPUB(s) in database:" in output
            assert "📖 Test Book" in output
            assert "Author: Test Author" in output
            assert "File: /path/to/book.epub" in output
            assert "Chapters: 15" in output
            assert "Chunks: 15" in output
            assert "Embeddings:" in output
            assert "Added: 2024-01-01 12:00:00" in output
        finally:
            sys.stdout = sys.__stdout__


class TestUploadCLI:
    """Test cases for upload CLI main function."""

    @patch("sys.argv", ["upload", "upload", "test.epub"])
    @patch("home_library.cli.upload.upload_epub")
    def test_main_upload(self, mock_upload_epub):
        """Test main function with upload command."""
        main()
        mock_upload_epub.assert_called_once_with("test.epub", True)

    @patch("sys.argv", ["upload", "upload", "test.epub", "--no-embeddings"])
    @patch("home_library.cli.upload.upload_epub")
    def test_main_upload_no_embeddings(self, mock_upload_epub):
        """Test main function with upload command and no-embeddings flag."""
        main()
        mock_upload_epub.assert_called_once_with("test.epub", False)

    @patch("sys.argv", ["upload", "list"])
    @patch("home_library.cli.upload.list_epubs")
    def test_main_list(self, mock_list_epubs):
        """Test main function with list command."""
        main()
        mock_list_epubs.assert_called_once()

    @patch("sys.argv", ["upload"])
    @patch("home_library.cli.upload.argparse.ArgumentParser.print_help")
    def test_main_no_command(self, mock_print_help):
        """Test main function with no command."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse returns code 1 for no command
        assert exc_info.value.code == 1
        mock_print_help.assert_called_once()

    @patch("sys.argv", ["upload", "unknown"])
    def test_main_unknown_command(self):
        """Test main function with unknown command."""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse returns code 2 for invalid arguments
            assert exc_info.value.code == 2
            # argparse prints error to stderr, not stdout
            # The error message is handled by argparse itself
        finally:
            sys.stdout = sys.__stdout__
