"""Tests for the migrate CLI command."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

from home_library.cli.migrate import (
    create_tables, drop_tables, reset_database, check_status, main
)


class TestMigrateCommands:
    """Test cases for migrate CLI commands."""

    @patch('home_library.cli.migrate.get_db_service')
    def test_create_tables_success(self, mock_get_db_service):
        """Test successful table creation."""
        mock_db_service = Mock()
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            create_tables()
            output = captured_output.getvalue()
            assert "✅ Database tables created successfully!" in output
            mock_db_service.create_tables.assert_called_once()
        finally:
            sys.stdout = sys.__stdout__

    @patch('home_library.cli.migrate.get_db_service')
    def test_create_tables_error(self, mock_get_db_service):
        """Test table creation with error."""
        mock_db_service = Mock()
        mock_db_service.create_tables.side_effect = Exception("Database error")
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with pytest.raises(SystemExit) as exc_info:
                create_tables()
            assert exc_info.value.code == 1
            output = captured_output.getvalue()
            assert "❌ Error creating tables: Database error" in output
        finally:
            sys.stdout = sys.__stdout__

    @patch('home_library.cli.migrate.get_db_service')
    def test_drop_tables_success(self, mock_get_db_service):
        """Test successful table dropping."""
        mock_db_service = Mock()
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            drop_tables()
            output = captured_output.getvalue()
            assert "✅ Database tables dropped successfully!" in output
            mock_db_service.drop_tables.assert_called_once()
        finally:
            sys.stdout = sys.__stdout__

    @patch('home_library.cli.migrate.get_db_service')
    def test_drop_tables_error(self, mock_get_db_service):
        """Test table dropping with error."""
        mock_db_service = Mock()
        mock_db_service.drop_tables.side_effect = Exception("Database error")
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with pytest.raises(SystemExit) as exc_info:
                drop_tables()
            assert exc_info.value.code == 1
            output = captured_output.getvalue()
            assert "❌ Error dropping tables: Database error" in output
        finally:
            sys.stdout = sys.__stdout__

    @patch('home_library.cli.migrate.get_db_service')
    def test_reset_database_success(self, mock_get_db_service):
        """Test successful database reset."""
        mock_db_service = Mock()
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            reset_database()
            output = captured_output.getvalue()
            assert "✅ Database reset successfully!" in output
            mock_db_service.reset_database.assert_called_once()
        finally:
            sys.stdout = sys.__stdout__

    @patch('home_library.cli.migrate.get_db_service')
    def test_reset_database_error(self, mock_get_db_service):
        """Test database reset with error."""
        mock_db_service = Mock()
        mock_db_service.reset_database.side_effect = Exception("Database error")
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with pytest.raises(SystemExit) as exc_info:
                reset_database()
            assert exc_info.value.code == 1
            output = captured_output.getvalue()
            assert "❌ Error resetting database: Database error" in output
        finally:
            sys.stdout = sys.__stdout__

    @patch('home_library.cli.migrate.get_db_service')
    def test_check_status_healthy(self, mock_get_db_service):
        """Test status check with healthy database."""
        mock_db_service = Mock()
        mock_db_service.health_check.return_value = True
        mock_db_service.get_database_info.return_value = {
            'database_url': 'sqlite:///test.db',
            'status': 'healthy',
            'database_size': '1.2 MB',
            'tables': [
                {'tablename': 'epubs', 'inserts': 5},
                {'tablename': 'chapters', 'inserts': 25}
            ]
        }
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            check_status()
            output = captured_output.getvalue()
            assert "✅ Database connection: HEALTHY" in output
            assert "📊 Database URL: sqlite:///test.db" in output
            assert "📊 Status: healthy" in output
            assert "📊 Database Size: 1.2 MB" in output
            assert "📊 Tables:" in output
            assert "- epubs: 5 inserts" in output
            assert "- chapters: 25 inserts" in output
        finally:
            sys.stdout = sys.__stdout__

    @patch('home_library.cli.migrate.get_db_service')
    def test_check_status_unhealthy(self, mock_get_db_service):
        """Test status check with unhealthy database."""
        mock_db_service = Mock()
        mock_db_service.health_check.return_value = False
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with pytest.raises(SystemExit) as exc_info:
                check_status()
            assert exc_info.value.code == 1
            output = captured_output.getvalue()
            assert "❌ Database connection: UNHEALTHY" in output
        finally:
            sys.stdout = sys.__stdout__

    @patch('home_library.cli.migrate.get_db_service')
    def test_check_status_error(self, mock_get_db_service):
        """Test status check with error."""
        mock_db_service = Mock()
        mock_db_service.health_check.side_effect = Exception("Connection error")
        mock_get_db_service.return_value = mock_db_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            with pytest.raises(SystemExit) as exc_info:
                check_status()
            assert exc_info.value.code == 1
            output = captured_output.getvalue()
            assert "❌ Error checking database status: Connection error" in output
        finally:
            sys.stdout = sys.__stdout__


class TestMigrateCLI:
    """Test cases for migrate CLI main function."""

    @patch('sys.argv', ['migrate', 'create'])
    @patch('home_library.cli.migrate.create_tables')
    def test_main_create(self, mock_create_tables):
        """Test main function with create command."""
        main()
        mock_create_tables.assert_called_once()

    @patch('sys.argv', ['migrate', 'drop'])
    @patch('home_library.cli.migrate.drop_tables')
    def test_main_drop(self, mock_drop_tables):
        """Test main function with drop command."""
        main()
        mock_drop_tables.assert_called_once()

    @patch('sys.argv', ['migrate', 'reset'])
    @patch('home_library.cli.migrate.reset_database')
    def test_main_reset(self, mock_reset_database):
        """Test main function with reset command."""
        main()
        mock_reset_database.assert_called_once()

    @patch('sys.argv', ['migrate', 'status'])
    @patch('home_library.cli.migrate.check_status')
    def test_main_status(self, mock_check_status):
        """Test main function with status command."""
        main()
        mock_check_status.assert_called_once()

    @patch('sys.argv', ['migrate'])
    @patch('home_library.cli.migrate.argparse.ArgumentParser.print_help')
    def test_main_no_command(self, mock_print_help):
        """Test main function with no command."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        mock_print_help.assert_called_once()

    @patch('sys.argv', ['migrate', 'unknown'])
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
            pass
        finally:
            sys.stdout = sys.__stdout__
