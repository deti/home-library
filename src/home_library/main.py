"""Home Library package entry points.

Provides a simple, testable "Hello, World!" implementation.
"""

from __future__ import annotations


def hello() -> str:
    """Return the classic greeting.

    This pure function is easy to unit test.
    """
    return "Hello, World!"


def main() -> None:
    """Print the greeting to stdout."""
    print(hello())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
