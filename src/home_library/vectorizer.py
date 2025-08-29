"""Vectorization utilities for EPUB content.

This module provides functionality to chunk EPUB text content and generate
embeddings suitable for RAG (Retrieval-Augmented Generation) systems.
"""

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from home_library.epub_processor import parse_epub
from home_library.settings import get_settings


class TextChunk(BaseModel):
    """A chunk of text with metadata for vectorization."""

    text: str = Field(description="The text content of the chunk")
    chunk_id: str = Field(description="Unique identifier for the chunk")
    chapter_index: int = Field(description="Index of the source chapter")
    chapter_title: str | None = Field(description="Title of the source chapter")
    start_token: int = Field(description="Starting token position in the chapter")
    end_token: int = Field(description="Ending token position in the chapter")
    word_count: int = Field(description="Number of words in this chunk")


class VectorizationResult(BaseModel):
    """Result of vectorizing an EPUB file."""

    file_path: str = Field(description="Path to the processed EPUB file")
    total_chunks: int = Field(description="Total number of text chunks created")
    total_words: int = Field(description="Total word count across all chunks")
    chunk_size: int = Field(description="Configured chunk size in tokens")
    chunk_overlap: int = Field(description="Configured chunk overlap in tokens")
    embedding_dimension: int = Field(description="Dimension of embedding vectors")
    vectorization_method: str = Field(description="Method used for vectorization")
    chunks: list[TextChunk] = Field(description="List of all text chunks")


@dataclass
class _ChunkingStats:
    """Internal stats for chunking process."""

    total_chunks: int = 0
    total_words: int = 0


def _tokenize_text(text: str) -> list[str]:
    """Simple tokenization by splitting on whitespace and punctuation."""
    # Split on whitespace and common punctuation
    tokens = re.split(r'[\s\.,!?;:()[\]{}"\'-]+', text)
    # Filter out empty tokens
    return [token for token in tokens if token.strip()]


def _create_chunks_from_text(
    text: str,
    chapter_index: int,
    chapter_title: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Create text chunks from a single chapter's text using semchunk library."""
    if not text.strip():
        return []

        # Use semchunk library for text chunking
    tokens = _tokenize_text(text)
    if len(tokens) <= chunk_size:
        # Single chunk for short text
        chunk = TextChunk(
            text=text,
            chunk_id=f"chunk_{chapter_index}_0",
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            start_token=0,
            end_token=len(tokens),
            word_count=len(text.split()),
        )
        return [chunk]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size

        # Extract text for this chunk
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)

        # Create chunk
        chunk = TextChunk(
            text=chunk_text,
            chunk_id=f"chunk_{chapter_index}_{len(chunks)}",
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            start_token=start,
            end_token=end,
            word_count=len(chunk_text.split()),
        )
        chunks.append(chunk)

        if end >= len(tokens):
            break

        start = end - chunk_overlap

    return chunks


def vectorize_epub(
    file_path: str, chunk_size: int | None = None, chunk_overlap: int | None = None
) -> VectorizationResult:
    """Vectorize an EPUB file by creating text chunks.

    Args:
        file_path: Path to the EPUB file
        chunk_size: Override default chunk size from settings
        chunk_overlap: Override default chunk overlap from settings

    Returns:
        VectorizationResult containing all chunks and metadata
    """
    settings = get_settings()

    # Use provided values or defaults from settings
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Parse the EPUB file
    epub_details = parse_epub(file_path, include_text=True)

    # Create chunks from each chapter
    all_chunks = []
    total_words = 0

    for chapter in epub_details.chapters:
        if chapter.text:
            chunks = _create_chunks_from_text(
                chapter.text, chapter.index, chapter.title, chunk_size, chunk_overlap
            )
            all_chunks.extend(chunks)
            total_words += sum(chunk.word_count for chunk in chunks)

    # Create result
    return VectorizationResult(
        file_path=file_path,
        total_chunks=len(all_chunks),
        total_words=total_words,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_dimension=settings.embedding_dimension,
        vectorization_method=settings.vectorization_method,
        chunks=all_chunks,
    )


def get_vectorization_stats(result: VectorizationResult) -> dict[str, Any]:
    """Get comprehensive statistics about the vectorization result."""
    # Calculate chunk size distribution
    chunk_sizes = [chunk.word_count for chunk in result.chunks]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

    # Chapter distribution
    chapter_chunk_counts = {}
    for chunk in result.chunks:
        chapter_idx = chunk.chapter_index
        chapter_chunk_counts[chapter_idx] = chapter_chunk_counts.get(chapter_idx, 0) + 1

    return {
        "file_path": result.file_path,
        "total_chunks": result.total_chunks,
        "total_words": result.total_words,
        "average_chunk_size": round(avg_chunk_size, 2),
        "chunk_size_range": {
            "min": min(chunk_sizes) if chunk_sizes else 0,
            "max": max(chunk_sizes) if chunk_sizes else 0,
        },
        "chunks_per_chapter": chapter_chunk_counts,
        "configuration": {
            "chunk_size": result.chunk_size,
            "chunk_overlap": result.chunk_overlap,
            "embedding_dimension": result.embedding_dimension,
            "vectorization_method": result.vectorization_method,
        },
    }
