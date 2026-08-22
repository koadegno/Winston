"""Application settings for Winston."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"


def get_config() -> Settings:
    """Build and return a fresh Winston settings instance."""
    return Settings()
