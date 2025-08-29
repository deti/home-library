"""Database service for the home library system.

This module provides database connection management and session handling
for the SQLAlchemy ORM operations.
"""

import json
from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from home_library.models import Base
from home_library.settings import get_settings


class DatabaseService:
    """Database service for managing connections and sessions."""

    def __init__(self) -> None:
        """Initialize the database service."""
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the SQLAlchemy engine."""
        # For development, use SQLite if PostgreSQL is not available
        try:
            self.engine = create_engine(
                self.settings.database_url,
                echo=self.settings.database_echo,
                pool_pre_ping=True,
            )
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            # Fallback to SQLite for development
            sqlite_url = "sqlite:///./home_library.db"
            self.engine = create_engine(
                sqlite_url,
                echo=self.settings.database_echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)

    def reset_database(self) -> None:
        """Reset the database by dropping and recreating all tables."""
        self.drop_tables()
        self.create_tables()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """Check if the database is accessible."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_database_info(self) -> dict[str, Any]:
        """Get information about the database."""
        try:
            with self.engine.connect() as conn:
                # Get table counts
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes
                    FROM pg_stat_user_tables 
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """))
                table_stats = [dict(row._mapping) for row in result]
                
                # Get database size
                result = conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size
                """))
                db_size = result.fetchone()[0] if result.rowcount > 0 else "Unknown"
                
                return {
                    "database_url": str(self.engine.url),
                    "database_size": db_size,
                    "tables": table_stats,
                    "status": "healthy"
                }
        except Exception as e:
            return {
                "database_url": str(self.engine.url),
                "status": "error",
                "error": str(e)
            }


# Global database service instance
db_service = DatabaseService()


def get_db_service() -> DatabaseService:
    """Get the global database service instance."""
    return db_service
