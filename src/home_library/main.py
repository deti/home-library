"""Home Library package entry points.

Provides a simple, testable "Hello, World!" implementation and EPUB utilities CLI.
"""

import argparse
from collections.abc import Iterable

from home_library.epub_processor import parse_epub
from home_library.settings import settings


def hello() -> str:
    """Return the classic greeting.

    This pure function is easy to unit test.
    """
    return "Hello, World!"


def main() -> None:
    """Print the greeting to stdout."""
    print(hello())  # noqa: T201


def show_settings() -> None:
    """Print the app settings to stdout."""
    print(settings.model_dump_json(indent=2))  # noqa: T201


def _print_toc(toc: Iterable, indent: int = 0) -> None:
    pad = "  " * indent
    for item in toc:
        title = item.title or "<untitled>"
        href = f" ({item.href})" if item.href else ""
        print(f"{pad}- {title}{href}")  # noqa: T201
        if item.children:
            _print_toc(item.children, indent + 1)


def describe_epub_main(argv: list[str] | None = None) -> None:
    """CLI to extract and print EPUB details for a single file.

    Usage: epub-info /path/to/book.epub [--json] [--include-text]
    """
    parser = argparse.ArgumentParser(prog="epub-info", description="Show EPUB details")
    parser.add_argument("path", help="Path to .epub file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON dump (chapters text omitted unless --include-text)",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full chapter text in output (can be large)",
    )

    args = parser.parse_args(argv)

    details = parse_epub(args.path, include_text=bool(args.include_text))

    if args.json:
        # Avoid giant outputs unless explicitly requested
        print(details.model_dump_json(indent=2))  # noqa: T201
        return

    # Human-readable summary
    md = details.metadata
    print(f"File: {details.path}")  # noqa: T201
    print("Metadata:")  # noqa: T201
    print(f"  Title: {md.title or '-'}")  # noqa: T201
    print(f"  Authors: {', '.join(md.authors) if md.authors else '-'}")  # noqa: T201
    print(f"  Language: {md.language or '-'}")  # noqa: T201
    print(f"  Publisher: {md.publisher or '-'}")  # noqa: T201
    print(f"  Subjects: {', '.join(md.subjects) if md.subjects else '-'}")  # noqa: T201
    if md.identifiers:
        ids = ", ".join(
            f"{i.scheme + ':' if i.scheme else ''}{i.value}" for i in md.identifiers
        )
    else:
        ids = "-"
    print(f"  Identifiers: {ids}")  # noqa: T201
    if md.description:
        print(f"  Description: {md.description}")  # noqa: T201

    print(f"TOC: {len(details.toc)} top-level items")  # noqa: T201
    _print_toc(details.toc)

    print("Chapters:")  # noqa: T201
    for ch in details.chapters:
        t = ch.title or "<untitled>"
        print(  # noqa: T201
            f"  [{ch.index}] {t} | words={ch.word_count} | href={ch.href or '-'}"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    print("You've reached the CLI entry point.")  # noqa: T201
    main()
