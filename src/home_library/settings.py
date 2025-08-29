"""Application settings loaded via pydantic-settings.

This module defines a Settings class and a cached accessor that
loads configuration from environment variables and a .env file
located at the project root.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Compute the project root (repo root), e.g. .../home-library
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """App configuration.

    Values are sourced from (in order of precedence):
    1) Environment variables
    2) .env file(s) — see model_config.env_file
    3) Defaults defined on the fields
    """

    app_name: str = Field(
        default="home-library",
        description="Human-friendly application name.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (more verbose logs, etc.)",
    )
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = Field(
        default="INFO",
        description="Logging level.",
    )
    environment: Literal["development", "production", "test"] = Field(
        default="development",
        description="Runtime environment name.",
    )

    # Vectorization settings
    vectorization_method: Literal["sentence-transformers", "openai", "cohere"] = Field(
        default="sentence-transformers",
        description="Vectorization method to use for text embedding.",
    )
    chunk_size: int = Field(
        default=512,
        description="Number of tokens per text chunk for vectorization.",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Number of overlapping tokens between consecutive chunks.",
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension of the embedding vectors.",
    )

    # Embeddings settings
    embeddings_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformers model to use for embeddings.",
    )
    embeddings_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to use for embeddings computation.",
    )
    embeddings_batch_size: int = Field(
        default=32,
        description="Batch size for processing embeddings.",
    )

    # Database settings
    database_url: str = Field(
        default="postgresql://home_library_user:home_library_pass@localhost:5432/home_library",
        description="Database connection URL.",
    )
    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy echo mode for debugging.",
    )

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        # Read .env from the project root
        env_file=(PROJECT_ROOT / ".env",),
        env_file_encoding="utf-8",
        # No prefix; environment variables may be written as APP_NAME, DEBUG, etc.
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


# Convenient module-level instance
settings: Settings = get_settings()
