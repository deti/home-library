"""CLI command to vectorize EPUB files for RAG systems."""

# ruff: noqa: T201
import argparse
import json
import sys
from pathlib import Path

from home_library.vectorizer import get_vectorization_stats, vectorize_epub


def _print_stats(stats: dict) -> None:
    """Print vectorization statistics in a human-readable format."""
    print(f"Vectorization Statistics for: {stats['file_path']}")
    print("=" * 60)

    # Basic stats
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Total Words: {stats['total_words']:,}")
    print(f"Average Chunk Size: {stats['average_chunk_size']} words")
    print(
        f"Chunk Size Range: {stats['chunk_size_range']['min']} - {stats['chunk_size_range']['max']} words"
    )

    # Configuration
    config = stats["configuration"]
    print("\nConfiguration:")
    print(f"  Chunk Size: {config['chunk_size']} tokens")
    print(f"  Chunk Overlap: {config['chunk_overlap']} tokens")
    print(f"  Embedding Dimension: {config['embedding_dimension']}")
    print(f"  Vectorization Method: {config['vectorization_method']}")

    # Chapter distribution
    print("\nChunks per Chapter:")
    chapter_counts = stats["chunks_per_chapter"]
    for chapter_idx in sorted(chapter_counts.keys()):
        count = chapter_counts[chapter_idx]
        print(f"  Chapter {chapter_idx}: {count} chunks")

    # Efficiency metrics
    if stats["total_chunks"] > 0:
        words_per_chunk = stats["total_words"] / stats["total_chunks"]
        print("\nEfficiency Metrics:")
        print(f"  Words per Chunk: {words_per_chunk:.1f}")
        print(
            f"  Chunk Utilization: {(stats['average_chunk_size'] / config['chunk_size'] * 100):.1f}%"
        )


def _print_detailed_chunks(result) -> None:
    """Print detailed information about each chunk."""
    print("\nDetailed Chunk Information:")
    print("=" * 60)

    for chunk in result.chunks:
        chapter_info = f"Chapter {chunk.chapter_index}"
        if chunk.chapter_title:
            chapter_info += f" ({chunk.chapter_title})"

        print(f"\nChunk ID: {chunk.chunk_id}")
        print(f"Source: {chapter_info}")
        print(f"Position: tokens {chunk.start_token}-{chunk.end_token}")
        print(f"Word Count: {chunk.word_count}")
        print(
            f"Text Preview: {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}"
        )


def main() -> None:
    """CLI to vectorize EPUB files and show statistics.

    Usage: vectorize-epub /path/to/book.epub [--chunk-size N] [--chunk-overlap N] [--detailed] [--json]
    """
    parser = argparse.ArgumentParser(
        prog="vectorize-epub", description="Vectorize EPUB files for RAG systems"
    )
    parser.add_argument("path", help="Path to .epub file")
    parser.add_argument(
        "--chunk-size", type=int, help="Override default chunk size (in tokens)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, help="Override default chunk overlap (in tokens)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information about each chunk",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()

    # Validate file path
    file_path = Path(args.path)
    if not file_path.exists():
        sys.stderr.write(f"Error: File not found: {file_path}")
        return 1

    if file_path.suffix.lower() != ".epub":
        sys.stderr.write(f"Error: File must have .epub extension: {file_path}")
        return 1

    try:
        # Vectorize the EPUB file
        result = vectorize_epub(
            str(file_path), chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

        # Get statistics
        stats = get_vectorization_stats(result)

        if args.json:
            # JSON output
            try:
                output_data = {
                    "stats": stats,
                    "chunks": [chunk.model_dump() for chunk in result.chunks],
                }
                print(json.dumps(output_data, indent=2))
            except (TypeError, AttributeError):
                # Handle case where chunks might be Mock objects in tests
                sys.stderr.write("Error: Cannot serialize chunks to JSON")
                return 1
        else:
            # Human-readable output
            _print_stats(stats)

            if args.detailed:
                _print_detailed_chunks(result)

    except Exception:
        sys.stderr.write("Error processing EPUB file: Vectorization failed")
        return 1
    else:
        return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
