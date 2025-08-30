"""CLI command to search the vectorized library using vector similarity."""

# ruff: noqa: T201
import argparse
import json
import sys

from home_library.search import SearchResult, search_library


def _print_search_results(results: list[SearchResult], query: str) -> None:
    """Print search results in a human-readable format."""
    if not results:
        print(f"🔍 No results found for query: '{query}'")
        return

    print(f"🔍 Search Results for: '{query}'")
    print("=" * 80)
    print(f"Found {len(results)} relevant citations:\n")

    for i, result in enumerate(results, 1):
        print(f"📚 Result {i} (Similarity: {result.similarity_score:.3f})")
        print(f"   Book: {result.book_title}")
        if result.book_author:
            print(f"   Author: {result.book_author}")
        print(f"   Chapter: {result.chapter_index}")
        if result.chapter_title:
            print(f"   Chapter Title: {result.chapter_title}")
        print(f"   Section: Chunk {result.chunk_index} (tokens {result.start_token}-{result.end_token})")
        print(f"   Words: {result.word_count}")
        print(f"   File: {result.file_path}")
        print(f"   Text Preview: {result.text[:150]}{'...' if len(result.text) > 150 else ''}")
        print()


def _print_json_results(results: list[SearchResult], query: str) -> None:
    """Print search results in JSON format."""
    output_data = {
        "query": query,
        "total_results": len(results),
        "results": [result.model_dump() for result in results]
    }
    print(json.dumps(output_data, indent=2))


def _print_detailed_results(results: list[SearchResult], query: str) -> None:
    """Print detailed search results with full text."""
    if not results:
        print(f"🔍 No results found for query: '{query}'")
        return

    print(f"🔍 Detailed Search Results for: '{query}'")
    print("=" * 80)
    print(f"Found {len(results)} relevant citations:\n")

    for i, result in enumerate(results, 1):
        print(f"📚 Result {i} (Similarity: {result.similarity_score:.3f})")
        print(f"   Book: {result.book_title}")
        if result.book_author:
            print(f"   Author: {result.book_author}")
        print(f"   Chapter: {result.chapter_index}")
        if result.chapter_title:
            print(f"   Chapter Title: {result.chapter_title}")
        print(f"   Section: Chunk {result.chunk_index} (tokens {result.start_token}-{result.end_token})")
        print(f"   Words: {result.word_count}")
        print(f"   File: {result.file_path}")
        print("   Full Text:")
        print(f"   {'-' * 40}")
        print(f"   {result.text}")
        print(f"   {'-' * 40}")
        print()


def main() -> None:
    """CLI to search the vectorized library using vector similarity.

    Usage: search-library "your search query" [--limit N] [--threshold F] [--model NAME] [--device DEVICE] [--detailed] [--json]
    """
    parser = argparse.ArgumentParser(
        prog="search-library",
        description="Search the vectorized library using vector similarity",
    )
    parser.add_argument(
        "query",
        help="Search query text"
    )

    # Search options
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum similarity score threshold 0.0-1.0 (default: 0.3)"
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        help="Override default sentence-transformers model"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Override default device for embeddings computation"
    )

    # Output options
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show full text content for each result"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.limit < 1:
        sys.stderr.write("Error: Limit must be at least 1\n")
        return 1

    if not 0.0 <= args.threshold <= 1.0:
        sys.stderr.write("Error: Threshold must be between 0.0 and 1.0\n")
        return 1

    try:
        # Perform the search
        print(f"🔍 Searching library for: '{args.query}'")
        print(f"   Results limit: {args.limit}")
        print(f"   Similarity threshold: {args.threshold}")
        if args.model:
            print(f"   Model: {args.model}")
        if args.device:
            print(f"   Device: {args.device}")
        print()

        results = search_library(
            query=args.query,
            limit=args.limit,
            similarity_threshold=args.threshold,
            model_name=args.model,
            device=args.device
        )

        # Handle output formatting
        if args.json:
            _print_json_results(results, args.query)
        elif args.detailed:
            _print_detailed_results(results, args.query)
        else:
            _print_search_results(results, args.query)
            return 0

    except KeyboardInterrupt:
        print("\n❌ Search interrupted by user")
        return 1
    except Exception as e:
        sys.stderr.write(f"Error performing search: {e!s}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
