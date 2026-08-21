from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WINSTON_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    ffprobe_binary: str = "ffprobe"


settings = Settings()
