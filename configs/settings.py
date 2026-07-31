"""Environment-backed application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    random_seed: int = Field(default=42, ge=0)
    data_dir: Path = Path("data")
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "movie-lens-recommender"
    mlflow_model_name: str = "movie-lens-pytorch-recommender"
    mlflow_model_stage: str = "Production"
    mlflow_model_alias: str = "production"

    @property
    def resolved_data_dir(self) -> Path:
        """Return an absolute path to the configured data directory."""
        if self.data_dir.is_absolute():
            return self.data_dir
        return PROJECT_ROOT / self.data_dir


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load and cache project settings.

    Returns:
        Validated settings from environment variables and `.env`.
    """
    return Settings()
