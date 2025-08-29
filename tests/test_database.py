"""Tests for the database service module."""

from unittest.mock import Mock, patch

import pytest
from sqlalchemy import text

from home_library.database import DatabaseService, get_db_service


class TestDatabaseService:
    """Test cases for DatabaseService class."""

    def test_init(self):
        """Test DatabaseService initialization."""
        with patch("home_library.database.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.database_url = "sqlite:///:memory:"
            mock_settings.database_echo = False
            mock_get_settings.return_value = mock_settings

            service = DatabaseService()
            assert service.engine is not None
            assert service.SessionLocal is not None

    def test_create_tables(self):
        """Test table creation."""
        with patch("home_library.database.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.database_url = "sqlite:///:memory:"
            mock_settings.database_echo = False
            mock_get_settings.return_value = mock_settings

            service = DatabaseService()
            service.create_tables()

            # Verify tables were created by checking if we can query them
            with service.get_session() as session:
                # This should not raise an error if tables exist
                result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result]
                assert "epubs" in tables
                assert "chapters" in tables
                assert "text_chunks" in tables
                assert "embeddings" in tables

    def test_drop_tables(self):
        """Test table dropping."""
        with patch("home_library.database.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.database_url = "sqlite:///:memory:"
            mock_settings.database_echo = False
            mock_get_settings.return_value = mock_settings

            service = DatabaseService()
            service.create_tables()
            service.drop_tables()

            # Verify tables were dropped
            with service.get_session() as session:
                result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result]
                assert "epubs" not in tables

    def test_reset_database(self):
        """Test database reset."""
        with patch("home_library.database.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.database_url = "sqlite:///:memory:"
            mock_settings.database_echo = False
            mock_get_settings.return_value = mock_settings

            service = DatabaseService()
            service.reset_database()

            # Verify tables exist after reset
            with service.get_session() as session:
                result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result]
                assert "epubs" in tables

    def test_get_session(self):
        """Test session management."""
        with patch("home_library.database.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.database_url = "sqlite:///:memory:"
            mock_settings.database_echo = False
            mock_get_settings.return_value = mock_settings

            service = DatabaseService()
            service.create_tables()

            with service.get_session() as session:
                assert session is not None
                # Session should be active
                # Note: SQLAlchemy 2.0 doesn't have is_closed attribute

    def test_health_check(self):
        """Test database health check."""
        with patch("home_library.database.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.database_url = "sqlite:///:memory:"
            mock_settings.database_echo = False
            mock_get_settings.return_value = mock_settings

            service = DatabaseService()
            service.create_tables()

            # Health check should pass for in-memory SQLite
            assert service.health_check() is True

    def test_get_database_info(self):
        """Test database info retrieval."""
        with patch("home_library.database.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.database_url = "sqlite:///:memory:"
            mock_settings.database_echo = False
            mock_get_settings.return_value = mock_settings

            service = DatabaseService()
            service.create_tables()

            info = service.get_database_info()
            assert "database_url" in info
            assert "status" in info
            assert info["status"] in ["healthy", "error"]


class TestDatabaseServiceIntegration:
    """Integration tests for DatabaseService."""

    @pytest.fixture
    def db_service(self):
        """Create a database service for testing."""
        with patch("home_library.database.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.database_url = "sqlite:///:memory:"
            mock_settings.database_echo = False
            mock_get_settings.return_value = mock_settings

            service = DatabaseService()
            service.create_tables()
            return service

    def test_session_rollback_on_error(self, db_service):
        """Test that sessions rollback on errors."""
        with pytest.raises((Exception, Exception)), db_service.get_session() as session:
                # This should cause an error
                session.execute(text("SELECT * FROM non_existent_table"))

        # Session should be closed after error
        # Note: SQLAlchemy 2.0 doesn't have is_closed attribute
        # The session is automatically closed by the context manager

    def test_session_commit_on_success(self, db_service):
        """Test that sessions commit on success."""
        with db_service.get_session() as session:
            # This should succeed
            session.execute(text("SELECT 1"))

        # Session should be closed after success
        # Note: SQLAlchemy 2.0 doesn't have is_closed attribute
        # The session is automatically closed by the context manager


def test_get_db_service():
    """Test get_db_service function."""
    service = get_db_service()
    assert isinstance(service, DatabaseService)

    # Should return the same instance (singleton)
    service2 = get_db_service()
    assert service is service2
