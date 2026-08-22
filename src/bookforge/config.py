from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BOOKFORGE_",
        extra="ignore",
    )

    environment: str = "development"
    model_backend: Literal["ollama", "openai", "fake"] = "ollama"
    model_name: str = "gemma4:e2b-it-qat"
    model_base_url: str = "http://127.0.0.1:11434"
    model_api_key: str = ""
    model_timeout_seconds: float = 180.0
    model_keep_alive: str = "10m"
    asr_backend: Literal["mlx_whisper", "whisper_trt", "disabled", "test"] = "mlx_whisper"
    asr_model: str = ".models/whisper-base.en"
    asr_engine_path: str = ""
    asr_max_audio_mb: int = 20
    data_dir: Path = Path(".bookforge/data")
    cache_dir: Path = Path(".bookforge/cache")
    allowed_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
