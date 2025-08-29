"""Embeddings creation for EPUB content using sentence-transformers.

This module provides functionality to create embeddings from EPUB text chunks
using the sentence-transformers library for RAG systems.
"""

import logging
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from home_library.settings import get_settings
from home_library.vectorizer import TextChunk, vectorize_epub


logger = logging.getLogger(__name__)


class EmbeddingChunk(BaseModel):
    """A text chunk with its corresponding embedding vector."""

    chunk: TextChunk = Field(description="The text chunk")
    embedding: list[float] = Field(description="The embedding vector")
    embedding_norm: float = Field(description="L2 norm of the embedding vector")


class EmbeddingsResult(BaseModel):
    """Result of creating embeddings for an EPUB file."""

    file_path: str = Field(description="Path to the processed EPUB file")
    total_chunks: int = Field(description="Total number of chunks processed")
    total_words: int = Field(description="Total word count across all chunks")
    embedding_dimension: int = Field(description="Dimension of embedding vectors")
    model_name: str = Field(description="Name of the sentence-transformers model used")
    device: str = Field(description="Device used for computation")
    batch_size: int = Field(description="Batch size used for processing")
    chunks: list[EmbeddingChunk] = Field(description="List of chunks with embeddings")
    processing_time_seconds: float = Field(
        description="Total processing time in seconds"
    )


class EmbeddingsCreator:
    """Creates embeddings for EPUB text chunks using sentence-transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        """Initialize the embeddings creator.

        Args:
            model_name: Override default model name from settings
            device: Override default device from settings
            batch_size: Override default batch size from settings
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embeddings_model
        self.device = device or self.settings.embeddings_device
        self.batch_size = batch_size or self.settings.embeddings_batch_size

        # Initialize the sentence transformer model
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Model loaded successfully on device: {self.device}")

        # Get actual embedding dimension from the model
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model embedding dimension: {self.embedding_dimension}")

    def create_embeddings_for_epub(
        self,
        file_path: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> EmbeddingsResult:
        """Create embeddings for an EPUB file.

            Args:
                file_path: Path to the EPUB file
                        chunk_size: Override default chunk size from settings
            chunk_overlap: Override default chunk overlap from settings

        Returns:
                EmbeddingsResult containing all chunks with their embeddings
        """
        start_time = time.time()

        # First, vectorize the EPUB to get text chunks
        logger.info(f"Vectorizing EPUB file: {file_path}")
        vectorization_result = vectorize_epub(file_path, chunk_size, chunk_overlap)

        # Extract text from chunks for embedding
        texts = [chunk.text for chunk in vectorization_result.chunks]

        if not texts:
            logger.warning("No text chunks found in EPUB file")
            return EmbeddingsResult(
                file_path=file_path,
                total_chunks=0,
                total_words=0,
                embedding_dimension=self.embedding_dimension,
                model_name=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
                chunks=[],
                processing_time_seconds=time.time() - start_time,
            )

        # Create embeddings in batches
        logger.info(
            f"Creating embeddings for {len(texts)} chunks using batch size {self.batch_size}"
        )
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Convert numpy arrays to lists and create EmbeddingChunk objects
        embedding_chunks = []
        for _i, (chunk, embedding) in enumerate(
            zip(vectorization_result.chunks, embeddings, strict=False)
        ):
            embedding_list = embedding.tolist()
            embedding_norm = float(np.linalg.norm(embedding))

            embedding_chunk = EmbeddingChunk(
                chunk=chunk, embedding=embedding_list, embedding_norm=embedding_norm
            )
            embedding_chunks.append(embedding_chunk)

        processing_time = time.time() - start_time
        logger.info(f"Embeddings created successfully in {processing_time:.2f} seconds")

        return EmbeddingsResult(
            file_path=file_path,
            total_chunks=len(embedding_chunks),
            total_words=vectorization_result.total_words,
            embedding_dimension=self.embedding_dimension,
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            chunks=embedding_chunks,
            processing_time_seconds=processing_time,
        )

    def get_embeddings_stats(self, result: EmbeddingsResult) -> dict[str, Any]:
        """Get comprehensive statistics about the embeddings result."""
        if not result.chunks:
            return {
                "file_path": result.file_path,
                "total_chunks": 0,
                "total_words": 0,
                "embedding_dimension": result.embedding_dimension,
                "model_name": result.model_name,
                "processing_time": result.processing_time_seconds,
                "message": "No chunks processed",
            }

        # Calculate embedding statistics
        embedding_norms = [chunk.embedding_norm for chunk in result.chunks]
        avg_norm = sum(embedding_norms) / len(embedding_norms)

        # Calculate chunk size distribution
        chunk_sizes = [chunk.chunk.word_count for chunk in result.chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)

        # Chapter distribution
        chapter_chunk_counts = {}
        for chunk in result.chunks:
            chapter_idx = chunk.chunk.chapter_index
            chapter_chunk_counts[chapter_idx] = (
                chapter_chunk_counts.get(chapter_idx, 0) + 1
            )

        return {
            "file_path": result.file_path,
            "total_chunks": result.total_chunks,
            "total_words": result.total_words,
            "embedding_dimension": result.embedding_dimension,
            "model_name": result.model_name,
            "device": result.device,
            "batch_size": result.batch_size,
            "processing_time_seconds": round(result.processing_time_seconds, 2),
            "average_chunk_size": round(avg_chunk_size, 2),
            "chunk_size_range": {"min": min(chunk_sizes), "max": max(chunk_sizes)},
            "embedding_norms": {
                "average": round(avg_norm, 4),
                "min": round(min(embedding_norms), 4),
                "max": round(max(embedding_norms), 4),
            },
            "chunks_per_chapter": chapter_chunk_counts,
            "efficiency": {
                "chunks_per_second": round(
                    result.total_chunks / result.processing_time_seconds, 2
                ),
                "words_per_second": round(
                    result.total_words / result.processing_time_seconds, 2
                ),
            },
        }


def create_embeddings_for_epub(
    file_path: str,
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> EmbeddingsResult:
    """Convenience function to create embeddings for an EPUB file.

    Args:
        file_path: Path to the EPUB file
        model_name: Override default model name from settings
        device: Override default device from settings
        batch_size: Override default batch size from settings
        chunk_size: Override default chunk size from settings
        chunk_overlap: Override default chunk overlap from settings

    Returns:
        EmbeddingsResult containing all chunks with their embeddings
    """
    creator = EmbeddingsCreator(model_name, device, batch_size)
    return creator.create_embeddings_for_epub(file_path, chunk_size, chunk_overlap)


def get_embeddings_model(model_name: str | None = None, device: str | None = None) -> SentenceTransformer:
    """Get a sentence transformer model instance.
    
    Args:
        model_name: Override default model name from settings
        device: Override default device from settings
        
    Returns:
        SentenceTransformer model instance
    """
    settings = get_settings()
    model_name = model_name or settings.embeddings_model
    device = device or settings.embeddings_device
    
    return SentenceTransformer(model_name, device=device)
