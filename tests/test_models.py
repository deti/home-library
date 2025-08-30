"""Tests for the database models."""

from uuid import uuid4

from home_library.models import Chapter, Embedding, Epub, TextChunk


class TestEpub:
    """Test cases for Epub model."""

    def test_epub_creation(self):
        """Test Epub model creation."""
        epub = Epub(
            file_path="/path/to/book.epub",
            title="Test Book",
            author="Test Author",
            language="en",
            file_size=1024,
        )

        assert epub.file_path == "/path/to/book.epub"
        assert epub.title == "Test Book"
        assert epub.author == "Test Author"
        assert epub.language == "en"
        assert epub.file_size == 1024
        assert epub.id is None
        # ID is None until saved to database

    def test_epub_repr(self):
        """Test Epub model string representation."""
        epub = Epub(
            file_path="/path/to/book.epub", title="Test Book", author="Test Author"
        )

        repr_str = repr(epub)
        assert "Epub" in repr_str
        assert "Test Book" in repr_str
        assert str(epub.id) in repr_str

    def test_epub_defaults(self):
        """Test Epub model default values."""
        epub = Epub(file_path="/path/to/book.epub")

        assert epub.title is None
        assert epub.author is None
        assert epub.language is None
        assert epub.file_size is None
        assert epub.created_at is None
        assert epub.updated_at is None


class TestChapter:
    """Test cases for Chapter model."""

    def test_chapter_creation(self):
        """Test Chapter model creation."""
        epub_id = uuid4()
        chapter = Chapter(
            epub_id=epub_id,
            chapter_index=1,
            title="Chapter 1",
            file_name="chapter1.xhtml",
        )

        assert chapter.epub_id == epub_id
        assert chapter.chapter_index == 1
        assert chapter.title == "Chapter 1"
        assert chapter.file_name == "chapter1.xhtml"
        assert chapter.id is None
        # ID is None until saved to database

    def test_chapter_repr(self):
        """Test Chapter model string representation."""
        epub_id = uuid4()
        chapter = Chapter(epub_id=epub_id, chapter_index=1, title="Chapter 1")

        repr_str = repr(chapter)
        assert "Chapter" in repr_str
        assert "1" in repr_str
        assert "Chapter 1" in repr_str

    def test_chapter_defaults(self):
        """Test Chapter model default values."""
        epub_id = uuid4()
        chapter = Chapter(epub_id=epub_id, chapter_index=1)

        assert chapter.title is None
        assert chapter.file_name is None
        assert chapter.created_at is None


class TestTextChunk:
    """Test cases for TextChunk model."""

    def test_text_chunk_creation(self):
        """Test TextChunk model creation."""
        epub_id = uuid4()
        chapter_id = uuid4()

        chunk = TextChunk(
            epub_id=epub_id,
            chapter_id=chapter_id,
            chunk_index=0,
            text="This is a test chunk of text.",
            start_token=0,
            end_token=10,
            word_count=8,
        )

        assert chunk.epub_id == epub_id
        assert chunk.chapter_id == chapter_id
        assert chunk.chunk_index == 0
        assert chunk.text == "This is a test chunk of text."
        assert chunk.start_token == 0
        assert chunk.end_token == 10
        assert chunk.word_count == 8
        assert chunk.id is None
        # ID is None until saved to database

    def test_text_chunk_repr(self):
        """Test TextChunk model string representation."""
        epub_id = uuid4()
        chapter_id = uuid4()

        chunk = TextChunk(
            epub_id=epub_id,
            chapter_id=chapter_id,
            chunk_index=0,
            text="Test chunk",
            start_token=0,
            end_token=5,
            word_count=2,
        )

        repr_str = repr(chunk)
        assert "TextChunk" in repr_str
        assert "0" in repr_str
        assert "2" in repr_str

    def test_text_chunk_defaults(self):
        """Test TextChunk model default values."""
        epub_id = uuid4()
        chapter_id = uuid4()

        chunk = TextChunk(
            epub_id=epub_id,
            chapter_id=chapter_id,
            chunk_index=0,
            text="Test",
            start_token=0,
            end_token=1,
            word_count=1,
        )

        assert chunk.created_at is None


class TestEmbedding:
    """Test cases for Embedding model."""

    def test_embedding_creation(self):
        """Test Embedding model creation."""
        chunk_id = uuid4()
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        embedding = Embedding(
            chunk_id=chunk_id,
            vector=vector,  # Store as vector directly
            model_name="test-model",
            embedding_dimension=5,
        )

        assert embedding.chunk_id == chunk_id
        assert embedding.vector == vector
        assert embedding.model_name == "test-model"
        assert embedding.embedding_dimension == 5
        assert embedding.id is None
        # ID is None until saved to database

    def test_embedding_repr(self):
        """Test Embedding model string representation."""
        chunk_id = uuid4()
        vector = [0.1, 0.2, 0.3]

        embedding = Embedding(
            chunk_id=chunk_id,
            vector=vector,
            model_name="test-model",
            embedding_dimension=3,
        )

        repr_str = repr(embedding)
        assert "Embedding" in repr_str
        assert "test-model" in repr_str
        assert "3" in repr_str

    def test_embedding_defaults(self):
        """Test Embedding model default values."""
        chunk_id = uuid4()
        vector = [0.1, 0.2]

        embedding = Embedding(
            chunk_id=chunk_id,
            vector=vector,
            model_name="test-model",
            embedding_dimension=2,
        )

        assert embedding.created_at is None


class TestModelRelationships:
    """Test cases for model relationships."""

    def test_epub_chapter_relationship(self):
        """Test Epub-Chapter relationship."""
        epub = Epub(file_path="/path/to/book.epub", title="Test Book")
        chapter = Chapter(epub_id=epub.id, chapter_index=1, title="Chapter 1")

        # Test forward relationship
        epub.chapters.append(chapter)
        assert len(epub.chapters) == 1
        assert epub.chapters[0] == chapter

        # Test backward relationship
        assert chapter.epub == epub

    def test_epub_chunk_relationship(self):
        """Test Epub-TextChunk relationship."""
        epub = Epub(file_path="/path/to/book.epub", title="Test Book")
        chunk = TextChunk(
            epub_id=epub.id,
            chapter_id=uuid4(),
            chunk_index=0,
            text="Test chunk",
            start_token=0,
            end_token=5,
            word_count=2,
        )

        # Test forward relationship
        epub.chunks.append(chunk)
        assert len(epub.chunks) == 1
        assert epub.chunks[0] == chunk

        # Test backward relationship
        assert chunk.epub == epub

    def test_chapter_chunk_relationship(self):
        """Test Chapter-TextChunk relationship."""
        epub_id = uuid4()
        chapter = Chapter(epub_id=epub_id, chapter_index=1, title="Chapter 1")
        chunk = TextChunk(
            epub_id=epub_id,
            chapter_id=chapter.id,
            chunk_index=0,
            text="Test chunk",
            start_token=0,
            end_token=5,
            word_count=2,
        )

        # Test forward relationship
        chapter.chunks.append(chunk)
        assert len(chapter.chunks) == 1
        assert chapter.chunks[0] == chunk

        # Test backward relationship
        assert chunk.chapter == chapter

    def test_chunk_embedding_relationship(self):
        """Test TextChunk-Embedding relationship."""
        chunk = TextChunk(
            epub_id=uuid4(),
            chapter_id=uuid4(),
            chunk_index=0,
            text="Test chunk",
            start_token=0,
            end_token=5,
            word_count=2,
        )
        embedding = Embedding(
            chunk_id=chunk.id,
            vector=[0.1, 0.2, 0.3],
            model_name="test-model",
            embedding_dimension=3,
        )

        # Test forward relationship
        chunk.embedding = embedding
        assert chunk.embedding == embedding

        # Test backward relationship
        assert embedding.chunk == chunk


class TestModelConstraints:
    """Test cases for model constraints."""

    def test_epub_file_path_unique(self):
        """Test that EPUB file_path must be unique."""
        # This would be tested at the database level
        # For now, we just verify the constraint is defined
        epub_table = Epub.__table__
        unique_constraints = [
            c for c in epub_table.constraints if c.name == "uq_epub_file_path"
        ]
        assert len(unique_constraints) == 1

    def test_chapter_epub_chapter_index_unique(self):
        """Test that chapter index must be unique per EPUB."""
        chapter_table = Chapter.__table__
        unique_constraints = [
            c for c in chapter_table.constraints if c.name == "uq_epub_chapter_index"
        ]
        assert len(unique_constraints) == 1

    def test_chunk_epub_chapter_chunk_unique(self):
        """Test that chunk index must be unique per chapter per EPUB."""
        chunk_table = TextChunk.__table__
        unique_constraints = [
            c for c in chunk_table.constraints if c.name == "uq_epub_chapter_chunk"
        ]
        assert len(unique_constraints) == 1
