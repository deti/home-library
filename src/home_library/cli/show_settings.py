"""CLI command to show application settings."""

from home_library.settings import settings


def main() -> None:
    """Print the app settings to stdout."""
    print(settings.model_dump_json(indent=2))  # noqa: T201


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
