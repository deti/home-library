"""Search service for vector similarity search in the home library system.

This module provides functionality to search through vectorized book content
using cosine similarity between query embeddings and stored embeddings.
"""

import logging

from pydantic import BaseModel, Field
from sqlalchemy import select

from home_library.database import get_db_service
from home_library.embeddings import get_embeddings_model
from home_library.models import Chapter, Embedding, Epub, TextChunk
from home_library.settings import get_settings


logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """A search result containing book, chapter, and text chunk information."""

    book_title: str = Field(description="Title of the book")
    book_author: str | None = Field(description="Author of the book", default=None)
    chapter_index: int = Field(description="Chapter index in the book")
    chapter_title: str | None = Field(description="Title of the chapter", default=None)
    chunk_index: int = Field(description="Chunk index within the chapter")
    text: str = Field(description="Text content of the chunk")
    start_token: int = Field(description="Starting token position")
    end_token: int = Field(description="Ending token position")
    word_count: int = Field(description="Number of words in the chunk")
    similarity_score: float = Field(description="Cosine similarity score (0-1)")
    file_path: str = Field(description="Path to the EPUB file")


class SearchService:
    """Service for performing vector similarity search in the home library."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        """Initialize the search service.

        Args:
            model_name: Override default model name from settings
            device: Override default device from settings
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embeddings_model
        self.device = device or self.settings.embeddings_device

        # Initialize the sentence transformer model for query encoding
        logger.info(f"Loading search model: {self.model_name}")
        self.model = get_embeddings_model(self.model_name, self.device)
        logger.info(f"Search model loaded successfully on device: {self.device}")

    def search(
        self, query: str, limit: int = 5, similarity_threshold: float = 0.3
    ) -> list[SearchResult]:
        """Search for similar text chunks using vector similarity.

        Args:
            query: The search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1) for results

        Returns:
            List of search results ordered by similarity score
        """
        logger.info(f"Searching for query: '{query}' with limit {limit}")

        # Encode the query to get its embedding
        query_embedding = self.model.encode([query])[0]
        if hasattr(query_embedding, "tolist"):
            query_vector = query_embedding.tolist()
        else:
            query_vector = list(query_embedding)

        # Get database session
        db_service = get_db_service()

        with db_service.get_session() as session:
            # Use pgvector's cosine distance operator for efficient similarity search
            # Note: cosine distance = 1 - cosine similarity, so we order by distance ASC
            # and convert back to similarity (1 - distance) for the threshold check
            stmt = (
                select(
                    Embedding,
                    TextChunk,
                    Chapter,
                    Epub,
                    (1 - Embedding.vector.cosine_distance(query_vector)).label(
                        "similarity"
                    ),
                )
                .join(TextChunk, Embedding.chunk_id == TextChunk.id)
                .join(Chapter, TextChunk.chapter_id == Chapter.id)
                .join(Epub, TextChunk.epub_id == Epub.id)
                .filter(
                    (1 - Embedding.vector.cosine_distance(query_vector))
                    >= similarity_threshold
                )
                .order_by(Embedding.vector.cosine_distance(query_vector))
                .limit(limit)
            )

            results = session.execute(stmt).all()

            if not results:
                logger.warning("No embeddings found in database")
                return []

            # Results are already sorted and filtered by the database
            top_results = [
                {
                    "similarity": float(row.similarity),
                    "embedding": row.Embedding,
                    "chunk": row.TextChunk,
                    "chapter": row.Chapter,
                    "epub": row.Epub,
                }
                for row in results
            ]

            # Convert to SearchResult objects
            search_results = []
            for result in top_results:
                search_result = SearchResult(
                    book_title=result["epub"].title or "Unknown Title",
                    book_author=result["epub"].author,
                    chapter_index=result["chapter"].chapter_index,
                    chapter_title=result["chapter"].title,
                    chunk_index=result["chunk"].chunk_index,
                    text=result["chunk"].text,
                    start_token=result["chunk"].start_token,
                    end_token=result["chunk"].end_token,
                    word_count=result["chunk"].word_count,
                    similarity_score=round(result["similarity"], 4),
                    file_path=result["epub"].file_path,
                )
                search_results.append(search_result)

            logger.info(
                f"Found {len(search_results)} results above threshold {similarity_threshold}"
            )
            return search_results


def search_library(
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.3,
    model_name: str | None = None,
    device: str | None = None,
) -> list[SearchResult]:
    """Convenience function to search the library.

    Args:
        query: The search query text
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1) for results
        model_name: Override default model name from settings
        device: Override default device from settings

    Returns:
        List of search results ordered by similarity score
    """
    service = SearchService(model_name, device)
    return service.search(query, limit, similarity_threshold)
