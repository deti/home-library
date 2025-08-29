"""Upload EPUB files to database CLI command.

This module provides CLI commands for uploading EPUB files to the database,
including text extraction, chunking, and embedding generation.
"""

# ruff: noqa: T201
import argparse
import json
import sys
import traceback
from pathlib import Path

from home_library.database import get_db_service
from home_library.embeddings import get_embeddings_model
from home_library.epub_processor import parse_epub
from home_library.models import Chapter, Embedding, Epub, TextChunk
from home_library.settings import get_settings
from home_library.vectorizer import vectorize_epub


def upload_epub(file_path: str, generate_embeddings: bool = True) -> None:  # noqa: PLR0912 PLR0915
    """Upload an EPUB file to the database.

    Args:
        file_path: Path to the EPUB file
        generate_embeddings: Whether to generate embeddings for text chunks
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    if file_path.suffix.lower() != ".epub":
        print(f"❌ File is not an EPUB: {file_path}")
        sys.exit(1)

    try:
        # Parse the EPUB file
        print(f"📖 Parsing EPUB: {file_path}")
        epub_details = parse_epub(str(file_path), include_text=True)

        # Get database service
        db_service = get_db_service()

        with db_service.get_session() as session:
            # Check if EPUB already exists
            existing_epub = session.query(Epub).filter(
                Epub.file_path == str(file_path)
            ).first()

            if existing_epub:
                print(f"⚠️  EPUB already exists in database: {existing_epub.title}")
                response = input("Do you want to replace it? (y/N): ")
                if response.lower() != "y":
                    print("Upload cancelled.")
                    return

                # Delete existing data
                session.delete(existing_epub)
                session.commit()

            # Create EPUB record
            epub = Epub(
                file_path=str(file_path),
                title=epub_details.metadata.title,
                author=epub_details.metadata.authors[0] if epub_details.metadata.authors else None,
                language=epub_details.metadata.language,
                file_size=file_path.stat().st_size
            )
            session.add(epub)
            session.flush()  # Get the ID

            print(f"📚 Created EPUB record: {epub.title}")

            # Create chapter records
            chapters = []
            for chapter_detail in epub_details.chapters:
                chapter = Chapter(
                    epub_id=epub.id,
                    chapter_index=chapter_detail.index,
                    title=chapter_detail.title,
                    file_name=chapter_detail.href
                )
                chapters.append(chapter)
                session.add(chapter)

            session.flush()  # Get chapter IDs

            print(f"📑 Created {len(chapters)} chapter records")

            # Generate text chunks and embeddings
            if generate_embeddings:
                print("🔍 Generating text chunks and embeddings...")
                vectorization_result = vectorize_epub(str(file_path))

                # Create text chunk records
                chunks = []
                for chunk_detail in vectorization_result.chunks:
                    # Find corresponding chapter
                    chapter = next(
                        (c for c in chapters if c.chapter_index == chunk_detail.chapter_index),
                        None
                    )

                    if chapter:
                        chunk = TextChunk(
                            epub_id=epub.id,
                            chapter_id=chapter.id,
                            chunk_index=chunk_detail.chunk_id.split("_")[-1],
                            text=chunk_detail.text,
                            start_token=chunk_detail.start_token,
                            end_token=chunk_detail.end_token,
                            word_count=chunk_detail.word_count
                        )
                        chunks.append(chunk)
                        session.add(chunk)

                session.flush()  # Get chunk IDs
                print(f"📝 Created {len(chunks)} text chunk records")

                # Generate embeddings for chunks
                if chunks:
                    print("🧠 Generating embeddings...")
                    settings = get_settings()

                    model = get_embeddings_model()

                    for chunk in chunks:
                        # Generate embedding for the chunk text
                        embedding_result = model.encode([chunk.text])[0]
                        # Handle both numpy arrays and regular lists
                        if hasattr(embedding_result, "tolist"):
                            embedding_vector = embedding_result.tolist()
                        else:
                            embedding_vector = embedding_result

                        embedding = Embedding(
                            chunk_id=chunk.id,
                            vector=json.dumps(embedding_vector),
                            model_name=settings.embeddings_model,
                            embedding_dimension=len(embedding_vector)
                        )
                        session.add(embedding)

                    print(f"🎯 Generated {len(chunks)} embeddings")

            # Commit all changes
            session.commit()

            print(f"✅ Successfully uploaded EPUB: {epub.title}")
            print("📊 Summary:")
            print(f"   - EPUB: {epub.title} by {epub.author or 'Unknown'}")
            print(f"   - Chapters: {len(chapters)}")
            if generate_embeddings:
                print(f"   - Text chunks: {len(chunks)}")
                print(f"   - Embeddings: {len(chunks)}")

    except Exception as e:
        print(f"❌ Error uploading EPUB: {e}")
        traceback.print_exc()
        sys.exit(1)


def list_epubs() -> None:
    """List all EPUBs in the database."""
    try:
        db_service = get_db_service()

        with db_service.get_session() as session:
            epubs = session.query(Epub).all()

            if not epubs:
                print("📚 No EPUBs found in database.")
                return

            print(f"📚 Found {len(epubs)} EPUB(s) in database:")
            print()

            for epub in epubs:
                # Get counts
                chapter_count = session.query(Chapter).filter(
                    Chapter.epub_id == epub.id
                ).count()

                chunk_count = session.query(TextChunk).filter(
                    TextChunk.epub_id == epub.id
                ).count()

                embedding_count = session.query(Embedding).join(TextChunk).filter(
                    TextChunk.epub_id == epub.id
                ).count()

                print(f"📖 {epub.title}")
                print(f"   Author: {epub.author or 'Unknown'}")
                print(f"   File: {epub.file_path}")
                print(f"   Chapters: {chapter_count}")
                print(f"   Chunks: {chunk_count}")
                print(f"   Embeddings: {embedding_count}")
                print(f"   Added: {epub.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print()

    except Exception as e:
        print(f"❌ Error listing EPUBs: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Upload EPUB files to database for RAG processing"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Upload command
    upload_parser = subparsers.add_parser(
        "upload", help="Upload an EPUB file to the database"
    )
    upload_parser.add_argument(
        "file_path", help="Path to the EPUB file to upload"
    )
    upload_parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation (text chunks only)"
    )

    # List command
    subparsers.add_parser(
        "list", help="List all EPUBs in the database"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the appropriate command
    if args.command == "upload":
        generate_embeddings = not args.no_embeddings
        upload_epub(args.file_path, generate_embeddings)
    elif args.command == "list":
        list_epubs()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
