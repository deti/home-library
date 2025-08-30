"""Database models for the home library system.

This module defines SQLAlchemy models for storing EPUB metadata,
text chunks, and embeddings for RAG functionality.
"""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship
from sqlalchemy.sql import func


Base = declarative_base()


class Epub(Base):
    """EPUB file metadata."""

    __tablename__ = "epubs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    file_path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(String(500))
    author: Mapped[str | None] = mapped_column(String(500))
    language: Mapped[str | None] = mapped_column(String(10))
    file_size: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    chapters = relationship("Chapter", back_populates="epub", cascade="all, delete-orphan")
    chunks = relationship("TextChunk", back_populates="epub", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint("file_path", name="uq_epub_file_path"),
    )

    def __repr__(self) -> str:
        return f"<Epub(id={self.id}, title='{self.title}', file_path='{self.file_path}')>"


class Chapter(Base):
    """Chapter information from EPUB files."""

    __tablename__ = "chapters"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    epub_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("epubs.id"), nullable=False
    )
    chapter_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    file_name: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    epub = relationship("Epub", back_populates="chapters")
    chunks = relationship("TextChunk", back_populates="chapter", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint("epub_id", "chapter_index", name="uq_epub_chapter_index"),
    )

    def __repr__(self) -> str:
        return f"<Chapter(id={self.id}, index={self.chapter_index}, title='{self.title}')>"


class TextChunk(Base):
    """Text chunks extracted from EPUB chapters."""

    __tablename__ = "text_chunks"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    epub_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("epubs.id"), nullable=False
    )
    chapter_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chapters.id"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_token: Mapped[int] = mapped_column(Integer, nullable=False)
    end_token: Mapped[int] = mapped_column(Integer, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    epub = relationship("Epub", back_populates="chunks")
    chapter = relationship("Chapter", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False)

    # Constraints
    __table_args__ = (
        UniqueConstraint("epub_id", "chapter_id", "chunk_index", name="uq_epub_chapter_chunk"),
    )

    def __repr__(self) -> str:
        return f"<TextChunk(id={self.id}, chunk_index={self.chunk_index}, word_count={self.word_count})>"


class Embedding(Base):
    """Vector embeddings for text chunks."""

    __tablename__ = "embeddings"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    chunk_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("text_chunks.id"), nullable=False, unique=True
    )
    vector: Mapped[Vector] = mapped_column(Vector(1536), nullable=False)  # Max common embedding dimension
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    chunk = relationship("TextChunk", back_populates="embedding")

    def __repr__(self) -> str:
        return f"<Embedding(id={self.id}, model='{self.model_name}', dimension={self.embedding_dimension})>"


# Create indexes for better query performance
Index("idx_epubs_file_path", Epub.file_path)
Index("idx_chapters_epub_id", Chapter.epub_id)
Index("idx_chunks_epub_id", TextChunk.epub_id)
Index("idx_chunks_chapter_id", TextChunk.chapter_id)
Index("idx_embeddings_chunk_id", Embedding.chunk_id)
