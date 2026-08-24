from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
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
    model_context_tokens: Annotated[int, Field(ge=2_048, le=32_768)] = 8_192
    model_max_output_tokens: Annotated[int, Field(ge=64, le=8_192)] = 4_096
    model_require_gpu: bool = False
    asset_backend: Literal["disabled", "modal", "mflux", "fake"] = "disabled"
    asset_model: str = "z-image-turbo"
    asset_command: str = "mflux-generate-z-image-turbo"
    asset_depth_command: str = "mflux-save-depth"
    asset_modal_command: str = "modal"
    asset_modal_app: Path = Path("deploy/modal_scene_foundry.py")
    asset_modal_steps: int = 4
    asset_width: int = 1024
    asset_height: int = 576
    asset_steps: int = 9
    asset_quantize: Literal[3, 4, 5, 6, 8] = 4
    asset_timeout_seconds: float = 1_800.0
    asset_low_ram: bool = True
    live_scene_backend: Literal["auto", "disabled", "fake", "modal", "modal_warm"] = "auto"
    live_scene_planner: Literal["deterministic", "model"] = "deterministic"
    live_scene_planner_timeout_seconds: Annotated[float, Field(gt=0, le=60)] = 12.0
    live_scene_planner_model_revision: Annotated[
        str,
        Field(min_length=1, max_length=200),
    ] = "configured-local-model"
    live_scene_planner_cache_entries: Annotated[int, Field(ge=0, le=256)] = 32
    live_scene_planner_auto_warmup: bool = False
    live_scene_enable_motion: bool = False
    live_scene_enable_preview: bool = True
    live_scene_output_dir: Path = Path("artifacts/live-scenes/generated")
    live_scene_master_width: int = 896
    live_scene_master_height: int = 512
    # One-step SANA-Sprint is technically supported but failed the showcase
    # duplicate-subject gate for only a 5.5% end-to-end gain. Production stays
    # on the accepted two-to-four-step range; the finite CLI remains available
    # for explicitly budgeted research.
    live_scene_master_steps: Annotated[int, Field(ge=2, le=4)] = 2
    live_scene_master_guidance_scale: Annotated[float, Field(ge=0, le=12)] = 4.5
    live_scene_auto_prewarm_on_submit: bool = False
    live_scene_modal_session_gpu_cap_usd: Annotated[float, Field(gt=0, le=10)] = 1.0
    live_scene_max_active_jobs: Annotated[int, Field(ge=1, le=16)] = 2
    live_scene_max_retained_jobs: Annotated[int, Field(ge=1, le=256)] = 64
    live_scene_event_queue_size: Annotated[int, Field(ge=1, le=128)] = 8
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

    @field_validator("live_scene_master_width", "live_scene_master_height")
    @classmethod
    def validate_live_scene_master_dimension(cls, value: int) -> int:
        if not 512 <= value <= 1_536 or value % 32:
            raise ValueError("live-scene master dimensions must be 512..1536 and divisible by 32")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
