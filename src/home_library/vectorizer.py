"""Vectorization utilities for EPUB content.

This module provides functionality to chunk EPUB text content and generate
embeddings suitable for RAG (Retrieval-Augmented Generation) systems.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from home_library.epub_processor import parse_epub
from home_library.settings import get_settings


logger = logging.getLogger(__name__)


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
    logger.debug(f"Tokenizing text of length {len(text)} characters")

    # Split on whitespace and common punctuation
    tokens = re.split(r'[\s\.,!?;:()[\]{}"\'-]+', text)
    # Filter out empty tokens
    filtered_tokens = [token for token in tokens if token.strip()]

    logger.debug(
        f"Tokenization completed: {len(filtered_tokens)} tokens from {len(tokens)} raw splits"
    )
    return filtered_tokens


def _create_chunks_from_text(
    text: str,
    chapter_index: int,
    chapter_title: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Create text chunks from a single chapter's text using semchunk library."""
    if not text.strip():
        logger.debug(f"Chapter {chapter_index} has no text content, skipping")
        return []

    logger.debug(
        f"Creating chunks for chapter {chapter_index} (title: '{chapter_title}') with {len(text)} characters"
    )
    logger.debug(f"Chunk configuration: size={chunk_size}, overlap={chunk_overlap}")

    # Use semchunk library for text chunking
    tokens = _tokenize_text(text)
    if len(tokens) <= chunk_size:
        # Single chunk for short text
        logger.debug(
            f"Chapter {chapter_index} is short ({len(tokens)} tokens), creating single chunk"
        )
        chunk = TextChunk(
            text=text,
            chunk_id=f"chunk_{chapter_index}_0",
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            start_token=0,
            end_token=len(tokens),
            word_count=len(text.split()),
        )
        logger.debug(
            f"Created single chunk for chapter {chapter_index}: {chunk.word_count} words"
        )
        return [chunk]

    chunks = []
    start = 0
    chunk_count = 0

    logger.debug(
        f"Creating multiple chunks for chapter {chapter_index} with {len(tokens)} tokens"
    )

    while start < len(tokens):
        end = start + chunk_size

        # Extract text for this chunk
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)

        # Create chunk
        chunk = TextChunk(
            text=chunk_text,
            chunk_id=f"chunk_{chapter_index}_{chunk_count}",
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            start_token=start,
            end_token=end,
            word_count=len(chunk_text.split()),
        )
        chunks.append(chunk)
        chunk_count += 1

        logger.debug(
            f"Created chunk {chunk_count} for chapter {chapter_index}: tokens {start}-{end}, {chunk.word_count} words"
        )

        if end >= len(tokens):
            logger.debug(f"Reached end of chapter {chapter_index} tokens")
            break

        start = end - chunk_overlap
        logger.debug(
            f"Moving to next chunk starting at token {start} (overlap: {chunk_overlap})"
        )

    logger.info(
        f"Chapter {chapter_index} chunking completed: {len(chunks)} chunks created"
    )
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
    logger.info(f"Starting EPUB vectorization: {file_path}")

    settings = get_settings()

    # Use provided values or defaults from settings
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    logger.info(
        f"Vectorization configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )
    logger.debug(
        f"Using settings: embedding_dimension={settings.embedding_dimension}, vectorization_method={settings.vectorization_method}"
    )

    # Parse the EPUB file
    logger.debug("Parsing EPUB file to extract content and structure")
    try:
        epub_details = parse_epub(file_path, include_text=True)
        logger.info(
            f"EPUB parsed successfully: {len(epub_details.chapters)} chapters found"
        )
    except Exception:
        logger.exception(f"Failed to parse EPUB file {file_path}")
        raise

    # Create chunks from each chapter
    logger.debug("Starting chunk creation process")
    all_chunks = []
    total_words = 0
    chapters_with_content = 0
    chapters_skipped = 0

    for chapter in epub_details.chapters:
        if chapter.text:
            logger.debug(
                f"Processing chapter {chapter.index}: '{chapter.title}' with {chapter.word_count} words"
            )
            try:
                chunks = _create_chunks_from_text(
                    chapter.text,
                    chapter.index,
                    chapter.title,
                    chunk_size,
                    chunk_overlap,
                )
                all_chunks.extend(chunks)
                chapter_word_count = sum(chunk.word_count for chunk in chunks)
                total_words += chapter_word_count
                chapters_with_content += 1
                logger.debug(
                    f"Chapter {chapter.index} processed: {len(chunks)} chunks, {chapter_word_count} words"
                )
            except Exception:
                logger.exception(f"Failed to process chapter {chapter.index}")
                raise
        else:
            logger.debug(f"Skipping chapter {chapter.index} (no text content)")
            chapters_skipped += 1

    logger.info(
        f"Chunking completed: {len(all_chunks)} total chunks from {chapters_with_content} chapters"
    )
    logger.info(
        f"Total word count: {total_words}, chapters skipped: {chapters_skipped}"
    )

    # Create result
    result = VectorizationResult(
        file_path=file_path,
        total_chunks=len(all_chunks),
        total_words=total_words,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_dimension=settings.embedding_dimension,
        vectorization_method=settings.vectorization_method,
        chunks=all_chunks,
    )

    logger.info(
        f"EPUB vectorization completed successfully: {result.total_chunks} chunks, {result.total_words} words"
    )
    return result


def get_vectorization_stats(result: VectorizationResult) -> dict[str, Any]:
    """Get comprehensive statistics about the vectorization result."""
    logger.debug(f"Calculating vectorization statistics for: {result.file_path}")

    if not result.chunks:
        logger.debug("No chunks to analyze, returning basic stats")
        return {
            "file_path": result.file_path,
            "total_chunks": 0,
            "total_words": 0,
            "average_chunk_size": 0,
            "chunk_size_range": {"min": 0, "max": 0},
            "chunks_per_chapter": {},
            "configuration": {
                "chunk_size": result.chunk_size,
                "chunk_overlap": result.chunk_overlap,
                "embedding_dimension": result.embedding_dimension,
                "vectorization_method": result.vectorization_method,
            },
            "message": "No chunks processed",
        }

    # Calculate chunk size distribution
    logger.debug("Calculating chunk size distribution")
    chunk_sizes = [chunk.word_count for chunk in result.chunks]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

    # Chapter distribution
    logger.debug("Calculating chapter distribution")
    chapter_chunk_counts = {}
    for chunk in result.chunks:
        chapter_idx = chunk.chapter_index
        chapter_chunk_counts[chapter_idx] = chapter_chunk_counts.get(chapter_idx, 0) + 1

    # Calculate additional statistics
    logger.debug("Calculating additional statistics")
    min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
    max_chunk_size = max(chunk_sizes) if chunk_sizes else 0

    # Calculate chunk size variance
    if len(chunk_sizes) > 1:
        variance = sum((size - avg_chunk_size) ** 2 for size in chunk_sizes) / len(
            chunk_sizes
        )
        chunk_size_std = variance**0.5
    else:
        chunk_size_std = 0

    stats = {
        "file_path": result.file_path,
        "total_chunks": result.total_chunks,
        "total_words": result.total_words,
        "average_chunk_size": round(avg_chunk_size, 2),
        "chunk_size_range": {
            "min": min_chunk_size,
            "max": max_chunk_size,
        },
        "chunk_size_statistics": {
            "mean": round(avg_chunk_size, 2),
            "std_dev": round(chunk_size_std, 2),
            "median": sorted(chunk_sizes)[len(chunk_sizes) // 2] if chunk_sizes else 0,
        },
        "chunks_per_chapter": chapter_chunk_counts,
        "configuration": {
            "chunk_size": result.chunk_size,
            "chunk_overlap": result.chunk_overlap,
            "embedding_dimension": result.embedding_dimension,
            "vectorization_method": result.vectorization_method,
        },
    }

    logger.debug(f"Statistics calculated successfully: {len(stats)} metrics")
    logger.info(
        f"Vectorization stats: {result.total_chunks} chunks, avg size: {avg_chunk_size:.1f} words, size range: {min_chunk_size}-{max_chunk_size}"
    )

    return stats
