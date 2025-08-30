"""Database service for the home library system.

This module provides database connection management and session handling
for the SQLAlchemy ORM operations.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from home_library.models import Base
from home_library.settings import get_settings


logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for managing connections and sessions."""

    def __init__(self) -> None:
        """Initialize the database service."""
        logger.info("Initializing DatabaseService")
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the SQLAlchemy engine."""
        logger.info(
            f"Initializing database engine with URL: {self.settings.database_url}"
        )
        logger.debug(f"Database echo setting: {self.settings.database_echo}")

        # For development, use SQLite if PostgreSQL is not available
        try:
            logger.debug("Attempting to connect to PostgreSQL database")
            self.engine = create_engine(
                self.settings.database_url,
                echo=self.settings.database_echo,
                pool_pre_ping=True,
            )
            # Test connection
            with self.engine.connect() as conn:
                logger.debug("Testing PostgreSQL connection")
                conn.execute(text("SELECT 1"))
                logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            # Fallback to SQLite for development
            logger.warning(f"PostgreSQL connection failed: {e}, falling back to SQLite")
            sqlite_url = "sqlite:///./home_library.db"
            logger.info(f"Creating SQLite engine with URL: {sqlite_url}")
            self.engine = create_engine(
                sqlite_url,
                echo=self.settings.database_echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
            logger.info("SQLite engine created successfully")

        logger.debug("Creating session factory")
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.info("Database service initialization completed")

    def create_tables(self) -> None:
        """Create all database tables."""
        logger.info("Creating database tables")
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception:
            logger.exception("Failed to create database tables")
            raise

    def drop_tables(self) -> None:
        """Drop all database tables."""
        logger.warning("Dropping all database tables")
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception:
            logger.exception("Failed to drop database tables")
            raise

    def reset_database(self) -> None:
        """Reset the database by dropping and recreating all tables."""
        logger.warning("Resetting database - dropping and recreating all tables")
        try:
            self.drop_tables()
            self.create_tables()
            logger.info("Database reset completed successfully")
        except Exception:
            logger.exception("Database reset failed")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session]:
        """Get a database session with automatic cleanup."""
        logger.debug("Creating new database session")
        session = self.SessionLocal()
        try:
            logger.debug("Database session created, yielding to caller")
            yield session
            logger.debug("Committing database session")
            session.commit()
            logger.debug("Database session committed successfully")
        except Exception:
            logger.exception("Database session error, rolling back")
            session.rollback()
            raise
        finally:
            logger.debug("Closing database session")
            session.close()
            logger.debug("Database session closed")

    def health_check(self) -> bool:
        """Check if the database is accessible."""
        logger.debug("Performing database health check")
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.debug("Database health check passed")
                return True
        except Exception:
            logger.exception("Database health check failed")
            return False

    def get_database_info(self) -> dict[str, Any]:
        """Get information about the database."""
        logger.debug("Retrieving database information")
        try:
            with self.engine.connect() as conn:
                logger.debug("Connected to database for info retrieval")

                # Get table counts
                logger.debug("Querying table statistics")
                result = conn.execute(
                    text("""
                    SELECT
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """)
                )
                table_stats = [dict(row._asdict()) for row in result]
                logger.debug(f"Retrieved statistics for {len(table_stats)} tables")

                # Get database size
                logger.debug("Querying database size")
                result = conn.execute(
                    text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size
                """)
                )
                db_size = result.fetchone()[0] if result.rowcount > 0 else "Unknown"
                logger.debug(f"Database size: {db_size}")

                info = {
                    "database_url": str(self.engine.url),
                    "database_size": db_size,
                    "tables": table_stats,
                    "status": "healthy",
                }
                logger.info("Database information retrieved successfully")
                return info
        except Exception as e:
            logger.exception("Failed to retrieve database information")
            return {
                "database_url": str(self.engine.url),
                "status": "error",
                "error": str(e),
            }


# Global database service instance
logger.info("Creating global database service instance")
db_service = DatabaseService()


def get_db_service() -> DatabaseService:
    """Get the global database service instance."""
    logger.debug("Retrieving global database service instance")
    return db_service
