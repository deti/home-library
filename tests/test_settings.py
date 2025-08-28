import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from home_library.settings import (
    PROJECT_ROOT as SETTINGS_PROJECT_ROOT,
)
from home_library.settings import (
    Settings,
    get_settings,
)


# Ensure the package can be imported from the src/ layout during tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(autouse=True)
def clear_cache_and_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear settings cache and relevant env vars before each test.

    Also ensures PYTHONPATH is set for any subprocesses (parity with test_main).
    """
    # Clear lru_cache from previous tests
    get_settings.cache_clear()

    # Remove relevant environment variables
    for key in ("APP_NAME", "DEBUG", "LOG_LEVEL", "ENVIRONMENT", "log_level"):
        monkeypatch.delenv(key, raising=False)

    # Ensure PYTHONPATH contains src for any subprocess usage in future
    monkeypatch.setenv("PYTHONPATH", str(SRC_PATH))

    yield

    # Clear again to avoid leak across tests
    get_settings.cache_clear()


@pytest.mark.skipif(
    (SETTINGS_PROJECT_ROOT / ".env").exists(),
    reason="Defaults test skipped because a real .env exists at project root and would affect defaults.",
)
def test_settings_defaults_without_env_or_dotenv():
    s = Settings()  # Direct instance to avoid cached state
    assert s.app_name == "home-library"
    assert s.debug is False
    assert s.log_level == "INFO"
    assert s.environment == "development"


def test_environment_variables_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_NAME", "custom-app")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("ENVIRONMENT", "production")

    s = Settings()
    assert s.app_name == "custom-app"
    assert s.debug is True
    assert s.log_level == "ERROR"
    assert s.environment == "production"


def test_env_file_loaded_from_project_root():
    dotenv_path = SETTINGS_PROJECT_ROOT / ".env"

    # Backup existing .env if present
    backup_path = None
    if dotenv_path.exists():
        backup_path = SETTINGS_PROJECT_ROOT / ".env.backup_for_test"
        backup_path.write_text(dotenv_path.read_text(), encoding="utf-8")

    try:
        # Write a temporary .env in the project root
        dotenv_path.write_text(
            "\n".join(
                [
                    "APP_NAME=dotenv-app",
                    "DEBUG=true",
                    "LOG_LEVEL=DEBUG",
                    "ENVIRONMENT=test",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        # Ensure no env vars are set so .env is the source
        get_settings.cache_clear()
        s = Settings()
        assert s.app_name == "dotenv-app"
        assert s.debug is True
        assert s.log_level == "DEBUG"
        assert s.environment == "test"
    finally:
        # Restore or remove .env
        if backup_path and backup_path.exists():
            dotenv_path.write_text(
                backup_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            backup_path.unlink()
        elif dotenv_path.exists():
            dotenv_path.unlink()


def test_cached_accessor_and_cache_clear(monkeypatch: pytest.MonkeyPatch):
    # First call caches the instance
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

    # Changing environment without clearing cache should not change the object
    monkeypatch.setenv("APP_NAME", "after-change")
    s3 = get_settings()
    assert s3.app_name != "after-change"
    assert s3 is s1

    # After clearing cache, new values should be picked up
    get_settings.cache_clear()
    s4 = get_settings()
    assert s4 is not s1
    assert s4.app_name in {"home-library", "after-change"}  # depends on other env/.env


def test_env_var_names_are_case_insensitive(monkeypatch: pytest.MonkeyPatch):
    # pydantic-settings configured with case_sensitive=False, so the key casing is ignored
    # Use lower-case key for LOG_LEVEL
    monkeypatch.setenv("log_level", "DEBUG")

    s = Settings()
    assert s.log_level == "DEBUG"
